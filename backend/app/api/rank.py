"""POST /rank: corpus-wide aggregation and ranking across all banks."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.api.chat import get_chat_engine
from app.api.compliance import ChecklistCategory, load_checklist
from app.core.auth import CurrentUser
from app.core.rate_limit import RANK_METRIC_UNITS, RANK_REFUSED_UNITS, RequestBudget, reserve
from app.observability.tracing import budget_fields, traced_request, update_span
from app.rag.chat_engine import ChatEngine
from app.rag.compliance.checklist import ChecklistItem
from app.rag.compliance.sweep import (
    BankStatus,
    RankingRouted,
    RequirementSweep,
    rank_requirement,
    ranking_refusal_for_item,
    sweep_checklist_across_banks_with_llm,
)
from app.rag.metrics.aggregation import retrieve_metric_across_banks
from app.rag.metrics.analysis import AmbiguousBank as AnalysisAmbiguousBank
from app.rag.metrics.analysis import RankingResult, ValueRow, rank_banks
from app.rag.metrics.extraction import extract_metric_across_banks
from app.rag.metrics.metric import MetricSpec, load_metric_set
from app.rag.query_classifier import QueryType, classify_query

router = APIRouter(tags=["ranking"])

# A requirement code named literally in the query, e.g. "GRI 305-4" / "GRI 2-15".
_REQUIREMENT_CODE_RE = re.compile(r"\bGRI\s*(\d+)-(\d+)\b", re.IGNORECASE)

# Wording that flips the sort direction. "Earliest"/"lowest" ascend; "latest"/"highest"
_DESCENDING_MARKERS = (
    "highest",
    "largest",
    "latest",
    "longest",
    "furthest out",
    "most",
    "tertinggi",
    "terbesar",
    "paling lama",
    "terlama",
    "paling akhir",
    "lebih lama",
)

# Wording that asks for reasoning, not just a ranking: only affects `RankResponse.note`.
_EXPLANATION_MARKERS = ("why", "explain", "jelaskan", "kenapa", "mengapa")


class RankRequest(BaseModel):
    """A corpus-wide ranking request, expressed in natural language."""

    query: str = Field(..., min_length=1, description="e.g. 'top 5 banks by net-zero year'.")
    top_n: int | None = Field(
        default=None, ge=1, description="Limit the ranked rows returned; None returns all."
    )


class RankedBank(BaseModel):
    """One ranked bank: the parsed value, the verbatim disclosure, and its citation."""

    bank: str
    value: float
    raw_value: str
    scope_label: str | None
    citation: str | None


class ExcludedBank(BaseModel):
    """A bank kept out of the ranking, with the reason it was excluded."""

    bank: str
    reason: str
    detail: str | None


class AmbiguousBank(BaseModel):
    """A bank disclosing several scoped values, held out rather than silently collapsed."""

    bank: str
    values: list[str]
    citation: str | None


class BankRequirementStatus(BaseModel):
    """One bank's outcome for a ranked requirement: a disclosure status, not a value."""

    bank: str
    status: str
    citation: str | None
    fallback_used: bool


class RequirementRanking(BaseModel):
    """A requirement compared across the corpus by disclosure status, not a strict ordering."""

    addressed: list[BankRequirementStatus]
    partial: list[BankRequirementStatus]
    not_found: list[BankRequirementStatus]


class RankResponse(BaseModel):
    """The ranking, or the refusal that replaced it."""

    query: str
    query_type: str
    target: str | None
    target_kind: Literal["metric", "requirement", "unresolved"]
    outcome: Literal["ranked", "refused"]
    route_to: str | None = None
    reason: str | None = None
    message: str | None = None
    ranked: list[RankedBank] = []
    ambiguous: list[AmbiguousBank] = []
    excluded: list[ExcludedBank] = []
    bank_count: int | None = None
    note: str | None = None
    # Populated only for a ranked (not refused) "requirement" outcome: a status comparison.
    requirement_ranking: RequirementRanking | None = None


def resolve_requirement(query: str) -> tuple[ChecklistItem, ChecklistCategory] | None:
    """Find a checklist requirement named literally in the query, across all categories."""
    match = _REQUIREMENT_CODE_RE.search(query)
    if match is None:
        return None
    wanted = f"GRI {match.group(1)}-{match.group(2)}"
    for category in ChecklistCategory:
        for item in load_checklist(category):
            if item.code.lower() == wanted.lower():
                return item, category
    return None


def resolve_metric(query: str, metrics: list[MetricSpec] | None = None) -> MetricSpec | None:
    """Find the metric a query is about, by EDGB code first, then by longest matching keyword."""
    metrics = metrics if metrics is not None else load_metric_set()
    lowered = query.lower()

    for metric in metrics:
        if re.search(rf"(?<![\w-]){re.escape(metric.code)}(?![\w-])", query, re.IGNORECASE):
            return metric

    best: tuple[int, MetricSpec] | None = None
    for metric in metrics:
        for keyword in metric.topic_keywords:
            if keyword.lower() in lowered and (best is None or len(keyword) > best[0]):
                best = (len(keyword), metric)
    return best[1] if best else None


def _ascending(query: str) -> bool:
    """Sort direction from the query's wording; ascending unless it asks for the top end."""
    lowered = query.lower()
    return not any(marker in lowered for marker in _DESCENDING_MARKERS)


def _wants_explanation(query: str) -> bool:
    """Whether the query asks for reasoning, not just a ranked list."""
    lowered = query.lower()
    return any(marker in lowered for marker in _EXPLANATION_MARKERS)


def _rank_metric(engine: ChatEngine, metric: MetricSpec, query: str, top_n: int | None):
    """Run the metric aggregation pipeline end to end (blocking; called off the loop)."""
    evidence = retrieve_metric_across_banks(
        metric,
        qdrant_client=engine.qdrant_client,
        embed_client=engine.openai_client,
        settings=engine.settings,
    )
    extractions = extract_metric_across_banks(
        metric, evidence, client=engine.openai_client, settings=engine.settings
    )
    return rank_banks(extractions, ascending=_ascending(query), top_n=top_n), len(extractions)


def _join_with_and(names: Sequence[str]) -> str:
    """'A' / 'A and B' / 'A, B, and C': no Oxford-comma special case for exactly two."""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{', '.join(names[:-1])}, and {names[-1]}"


def _leading_tie_count(ranked: Sequence[ValueRow]) -> int:
    """How many rows at the front of `ranked` share the leading value (1 if none, 0 if empty)."""
    if not ranked:
        return 0
    leading = ranked[0].value
    count = 0
    for row in ranked:
        if row.value != leading:
            break
        count += 1
    return count


def _explanation_invite_sentence(ranked: Sequence[ValueRow]) -> str:
    """Invite a bank-specific follow-up when asked "why": a ranking states value, not reasoning."""
    tie_count = _leading_tie_count(ranked)
    leading = ranked[0]
    value = leading.raw_value
    if tie_count > 1:
        return (
            f"{tie_count} banks share the leading disclosed value ({value}) - ask about a "
            f'specific bank (e.g. "why does {leading.bank}\'s report say {value}") to see '
            "what its own report says."
        )
    return (
        f"{leading.bank} has the leading disclosed value ({value}) - ask about it directly "
        f'(e.g. "why does {leading.bank}\'s report say {value}") to see what its own report '
        "says."
    )


def _format_scoped_value(value: str, label: str | None) -> str:
    """'2028 (Operational)' / bare '2028' when there is no label to attach."""
    return f"{value} ({label})" if label else value


def _ambiguous_sentence(ambiguous: Sequence[AnalysisAmbiguousBank]) -> str:
    """State each ambiguous bank's actual scoped values, verbatim, and why it's excluded."""
    per_bank = [
        f"{bank.bank} discloses "
        f"{_join_with_and([_format_scoped_value(value, label) for value, label in bank.values])}."
        for bank in ambiguous
    ]
    plural = len(ambiguous) > 1
    return (
        f"{' '.join(per_bank)} Because {'each names' if plural else 'it names'} multiple "
        f"scoped values rather than one, {'they sit' if plural else 'it sits'} outside the "
        "strict ranking."
    )


def _metric_response(query: str, metric: MetricSpec, result: RankingResult, banks: int):
    """Shape a metric ranking into the response, keeping the held-out banks visible."""
    return RankResponse(
        query=query,
        query_type=QueryType.AGGREGATION_RANKING.value,
        target=metric.code,
        target_kind="metric",
        outcome="ranked",
        ranked=[
            RankedBank(
                bank=row.bank,
                value=row.value,
                raw_value=row.raw_value,
                scope_label=row.label,
                citation=row.citation,
            )
            for row in result.ranked
        ],
        ambiguous=[
            AmbiguousBank(
                bank=bank.bank,
                values=[
                    f"{value}" + (f" ({label})" if label else "") for value, label in bank.values
                ],
                citation=bank.citation,
            )
            for bank in result.ambiguous
        ],
        excluded=[
            ExcludedBank(bank=bank.bank, reason=bank.reason.value, detail=bank.detail)
            for bank in result.excluded
        ],
        bank_count=banks,
        note=(
            "Banks disclosing several scoped values are held out of the ranking rather than "
            "collapsed to one, and banks with no disclosed value are excluded with a reason "
            "— neither is silently dropped."
            + (f" {_ambiguous_sentence(result.ambiguous)}" if result.ambiguous else "")
            + (
                f" {_explanation_invite_sentence(result.ranked)}"
                if _wants_explanation(query) and result.ranked
                else ""
            )
        ),
    )


def _refusal_response(query: str, code: str, routed: RankingRouted) -> RankResponse:
    return RankResponse(
        query=query,
        query_type=QueryType.AGGREGATION_RANKING.value,
        target=code,
        target_kind="requirement",
        outcome="refused",
        route_to=routed.route_to,
        reason=routed.reason,
        message=routed.message,
    )


def _bank_requirement_status(status: BankStatus) -> BankRequirementStatus:
    return BankRequirementStatus(
        bank=status.bank,
        status=status.status.value,
        citation=status.citation,
        fallback_used=status.fallback_used,
    )


def _requirement_response(query: str, item: ChecklistItem, view: RequirementSweep) -> RankResponse:
    """Shape a calibrated `RequirementSweep` into the response: a status comparison, not a
    numeric ranking."""
    return RankResponse(
        query=query,
        query_type=QueryType.AGGREGATION_RANKING.value,
        target=item.code,
        target_kind="requirement",
        outcome="ranked",
        bank_count=view.bank_count,
        requirement_ranking=RequirementRanking(
            addressed=[_bank_requirement_status(b) for b in view.addressed],
            partial=[_bank_requirement_status(b) for b in view.partial],
            not_found=[_bank_requirement_status(b) for b in view.not_found],
        ),
        note=(
            "Status compares how each report addresses this requirement, not how well each "
            "bank performs. This reflects the same content-reading judgement /compliance-check "
            "uses per report, not a raw retrieval-confidence proxy."
        ),
    )


def _rank_requirement(
    engine: ChatEngine, item: ChecklistItem, category: ChecklistCategory
) -> RequirementSweep | RankingRouted:
    """Run the calibrated single-requirement sweep end to end (blocking; called off the loop)."""
    sweep = sweep_checklist_across_banks_with_llm(
        [item],
        category=category.value,
        qdrant_client=engine.qdrant_client,
        embed_client=engine.openai_client,
        openai_client=engine.openai_client,
        settings=engine.settings,
    )
    return rank_requirement(sweep, item.code)


@router.post("/rank", response_model=RankResponse)
async def rank(
    payload: RankRequest,
    engine: Annotated[ChatEngine, Depends(get_chat_engine)],
    budget: Annotated[RequestBudget, Depends(reserve(RANK_METRIC_UNITS))],
    user: CurrentUser,
) -> RankResponse:
    """Answer a corpus-wide ranking question, or refuse with the reason and the right route."""
    if engine.openai_client is None and not engine.settings.openai_api_key:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not configured; cannot embed queries for retrieval.",
        )

    # `pipeline` records which of this endpoint's two very different code paths actually ran.
    with traced_request(
        name="rank",
        user_id=user.id,
        query=payload.query,
        tags=["rank"],
        metadata={"endpoint": "/rank", "pipeline": "undetermined"},
    ) as span:

        def _finish(pipeline: str, **fields: object) -> None:
            update_span(
                span,
                output=fields,
                metadata={
                    "endpoint": "/rank",
                    "pipeline": pipeline,
                    **fields,
                    **budget_fields(budget),
                },
            )

        classification = classify_query(
            payload.query, client=engine.openai_client, settings=engine.settings
        )
        if classification.query_type is not QueryType.AGGREGATION_RANKING:
            # Misrouted input, not a ranking run: refund in full.
            budget.settle(RANK_REFUSED_UNITS)
            _finish(
                "rejected_not_a_ranking_query",
                outcome="rejected",
                classified_as=classification.query_type.value,
            )
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Query classified as {classification.query_type.value}, not a corpus-wide "
                    "ranking. Use /chat for single-report questions or /compliance-check to "
                    "check one report against a checklist."
                ),
            )

        # A named requirement goes to the compliance guard, which decides without retrieving.
        requirement = resolve_requirement(payload.query)
        if requirement is not None:
            item, category = requirement
            routed = ranking_refusal_for_item(item, bank_count=21)
            if routed is not None:
                budget.settle(RANK_REFUSED_UNITS)
                _finish(
                    "compliance_ranking_refusal",
                    outcome="refused",
                    target=item.code,
                    target_kind="requirement",
                    reason=routed.reason,
                    route_to=routed.route_to,
                    retrievals=0,
                    completions=0,
                )
                return _refusal_response(payload.query, item.code, routed)

            # Not refused: run the calibrated sweep, per `ranking_refusal_for_item`'s promise.
            outcome = await run_in_threadpool(_rank_requirement, engine, item, category)
            if isinstance(outcome, RankingRouted):
                # Defensive: in practice the guard above already caught every refusal reason.
                budget.settle(RANK_REFUSED_UNITS)
                _finish(
                    "compliance_ranking_refusal_post_sweep",
                    outcome="refused",
                    target=item.code,
                    target_kind="requirement",
                    reason=outcome.reason,
                    route_to=outcome.route_to,
                )
                return _refusal_response(payload.query, item.code, outcome)
            budget.settle(RANK_METRIC_UNITS)
            response = _requirement_response(payload.query, item, outcome)
            _finish(
                "requirement_ranking",
                outcome="ranked",
                target=item.code,
                target_kind="requirement",
                bank_count=outcome.bank_count,
                addressed=len(outcome.addressed),
                partial=len(outcome.partial),
                not_found=len(outcome.not_found),
            )
            return response

        metric = resolve_metric(payload.query)
        if metric is None:
            budget.settle(RANK_REFUSED_UNITS)
            _finish(
                "unresolved_target_refusal",
                outcome="refused",
                target=None,
                reason="unresolved_target",
                retrievals=0,
                completions=0,
            )
            return RankResponse(
                query=payload.query,
                query_type=classification.query_type.value,
                target=None,
                target_kind="unresolved",
                outcome="refused",
                reason="unresolved_target",
                message=(
                    "This reads as a ranking question, but it does not name a metric from the "
                    "curated set or a GRI requirement code. Name one (e.g. 'net-zero target "
                    "year', 'E-02', or 'GRI 305-4') so the ranking has a defined target."
                ),
            )

        result, banks = await run_in_threadpool(
            _rank_metric, engine, metric, payload.query, payload.top_n
        )
        # The only path that actually fanned out across the corpus, charged in full.
        budget.settle(RANK_METRIC_UNITS)
        response = _metric_response(payload.query, metric, result, banks)
        _finish(
            "metric_aggregation",
            outcome="ranked",
            target=metric.code,
            target_kind="metric",
            bank_count=banks,
            ranked=len(response.ranked),
            ambiguous=len(response.ambiguous),
            excluded=len(response.excluded),
        )
        return response
