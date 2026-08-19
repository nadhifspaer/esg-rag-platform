"""LLM-based compliance status assessor: a calibration candidate for `StatusAssessor`."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from openai import OpenAI
from qdrant_client import QdrantClient

from app.core.config import Settings, get_settings
from app.models.evidence_text import truncate_snippet
from app.models.payload import ContentType
from app.rag.compliance.checklist import ChecklistItem
from app.rag.compliance.report import (
    ComplianceReport,
    Status,
    assess_checklist,
    default_assess_status,
)
from app.rag.compliance.retrieval import (
    DEFAULT_LIMIT,
    DEFAULT_SCORE_THRESHOLD,
    EvidenceResult,
    retrieve_evidence,
)

_TEMPERATURE = 0.0

# Generous relative to the 160-char UI excerpt cap: the judge needs enough of the chunk
# to see whether it's a disclosure-index table or real narrative.
_JUDGE_SNIPPET_MAX = 1200

_JUDGEABLE_STATUSES: dict[str, Status] = {
    Status.DISCLOSED.value: Status.DISCLOSED,
    Status.PARTIALLY_DISCLOSED.value: Status.PARTIALLY_DISCLOSED,
    Status.NOT_FOUND.value: Status.NOT_FOUND,
}


class LLMAssessorError(RuntimeError):
    """Raised when the batched judgement call cannot run or returns unusable output."""


JUDGE_SYSTEM_PROMPT = (
    "You are a strict compliance-disclosure judge for Indonesian bank sustainability "
    "reports. You are given a numbered list of REQUIREMENTs, each with its description "
    "and the single retrieved EVIDENCE passage a vector search found for it in one "
    "bank's report. For each requirement, decide whether the EVIDENCE substantively "
    "addresses what the requirement asks to be disclosed.\n\n"
    "Return exactly one status per requirement:\n"
    '- "disclosed": the EVIDENCE is real narrative or data that actually answers the '
    "requirement's ask — specific content, not just a passage that happens to mention "
    "the same words.\n"
    '- "partially disclosed": the EVIDENCE touches the requirement\'s topic but is '
    "incomplete, generic, or only partly responsive — for example it names the topic "
    "without giving the substance asked for.\n"
    '- "not found": the EVIDENCE does not address the requirement at all, or there is no '
    "usable evidence.\n\n"
    "CRITICAL DISTINCTION — read this before deciding. These reports contain "
    'disclosure-index tables: a list of metric codes ("E-01", "S-04", "G-02", ...), '
    "metric names, and page-number references, rendered here as a run of "
    'semicolon-separated cells, e.g. "E-01; Greenhouse Gas Emission Report; 75-82". That '
    "kind of table is NOT itself a disclosure of the requirement's content — it is an "
    "index pointing to where the real content lives elsewhere in the report, and the "
    "vector search frequently retrieves it because it topically overlaps every "
    "requirement's keywords at once (it literally contains a code and a page number for "
    "almost every topic). A disclosure-index table, on its own, with no narrative "
    'substance alongside it, is NEVER "disclosed" for the requirement it superficially '
    'resembles — label it "not found" if it is the only evidence and carries no '
    'narrative answering the requirement, or "partially disclosed" only if some genuine '
    "narrative accompanies it and that narrative itself falls short of fully answering "
    "the requirement.\n\n"
    "Judge ONLY the EVIDENCE given for each requirement — do not assume content exists "
    "elsewhere in the report that was not retrieved. When genuinely in doubt between two "
    'statuses, prefer the lower one ("partially disclosed" over "disclosed", "not found" '
    'over "partially disclosed").\n\n'
    "Respond ONLY with a JSON object of the form "
    '{"results": [{"code": "<requirement code>", "status": "disclosed" | '
    '"partially disclosed" | "not found"}, ...]}, with exactly one entry per requirement, '
    "in any order, using the exact code given."
)


@dataclass(frozen=True)
class PendingAssessment:
    """One requirement queued for the batched judgement, with its retrieved evidence."""

    item: ChecklistItem
    evidence: EvidenceResult


def _resolve_client(settings: Settings, injected: OpenAI | None) -> OpenAI:
    """Return the injected client, or build one from settings (needs the API key)."""
    if injected is not None:
        return injected
    if not settings.openai_api_key:
        raise LLMAssessorError("OPENAI_API_KEY is not set; cannot run the LLM assessor.")
    return OpenAI(api_key=settings.openai_api_key)


def _evidence_block(evidence: EvidenceResult) -> str:
    """The judge's view of one requirement's evidence: the top chunk's cleaned text."""
    if not evidence.results:
        return "(no evidence retrieved)"
    top = evidence.results[0]
    is_table = top.payload.content_type is ContentType.TABLE
    return truncate_snippet(top.payload.chunk_text, _JUDGE_SNIPPET_MAX, is_table=is_table)


def _build_user_message(pending: Sequence[PendingAssessment]) -> str:
    blocks = [
        f"REQUIREMENT {i}\n"
        f"code: {p.item.code}\n"
        f"description: {p.item.description}\n"
        f"EVIDENCE:\n{_evidence_block(p.evidence)}"
        for i, p in enumerate(pending, start=1)
    ]
    return "\n\n".join(blocks) + "\n\nRespond with the required JSON."


def _parse_batch(raw: str, pending: Sequence[PendingAssessment]) -> dict[str, Status]:
    """Parse and validate the judge's reply, requiring exactly one usable status per item."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMAssessorError(f"LLM assessor did not return valid JSON: {exc}") from exc

    try:
        entries = data["results"]
    except (KeyError, TypeError) as exc:
        raise LLMAssessorError(f"LLM assessor response missing 'results': {exc}") from exc

    by_code: dict[str, Status] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        code = str(entry.get("code", "")).strip()
        status = _JUDGEABLE_STATUSES.get(str(entry.get("status", "")).strip())
        if code and status is not None:
            by_code[code] = status

    expected = {p.item.code for p in pending}
    missing = expected - by_code.keys()
    if missing:
        raise LLMAssessorError(
            f"LLM assessor omitted or mislabeled requirement(s): {', '.join(sorted(missing))}"
        )
    return by_code


def llm_assess_batch(
    pending: Sequence[PendingAssessment],
    *,
    client: OpenAI | None = None,
    settings: Settings | None = None,
) -> dict[str, Status]:
    """Judge every item in `pending` with exactly one OpenAI call, keyed by requirement code."""
    if not pending:
        return {}

    settings = settings or get_settings()
    resolved_client = _resolve_client(settings, client)

    try:
        response = resolved_client.chat.completions.create(
            model=settings.openai_generation_model,
            temperature=_TEMPERATURE,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_message(pending)},
            ],
        )
    except LLMAssessorError:
        raise
    except Exception as exc:  # network/API failure -> a clear, typed error
        raise LLMAssessorError(f"LLM assessor request failed: {exc}") from exc

    return _parse_batch(response.choices[0].message.content or "", pending)


RetrieveOne = Callable[[ChecklistItem], EvidenceResult]
JudgeBatch = Callable[[Sequence[PendingAssessment]], dict[str, Status]]


def assess_checklist_with_llm(
    checklist: Sequence[ChecklistItem],
    *,
    report_name: str,
    category: str,
    retrieve: RetrieveOne,
    judge: JudgeBatch = llm_assess_batch,
) -> ComplianceReport:
    """Assess `checklist` against one report, batching the LLM judgement to one call."""
    evidence_by_code: dict[str, EvidenceResult] = {}
    status_by_code: dict[str, Status] = {}
    pending: list[PendingAssessment] = []

    for item in checklist:
        if not item.directly_disclosed:
            continue

        evidence = retrieve(item)
        evidence_by_code[item.code] = evidence

        if item.row_anchors:
            # A fact about the document, already reliable, no judgement needed.
            status_by_code[item.code] = default_assess_status(item, evidence)
        elif not evidence.results:
            # Same short-circuit `default_assess_status` applies: nothing to judge.
            status_by_code[item.code] = Status.NOT_FOUND
        else:
            pending.append(PendingAssessment(item=item, evidence=evidence))

    if pending:
        status_by_code.update(judge(pending))

    def cached_retrieve(item: ChecklistItem) -> EvidenceResult:
        return evidence_by_code[item.code]

    def cached_assess(item: ChecklistItem, _evidence: EvidenceResult) -> Status:
        return status_by_code[item.code]

    return assess_checklist(
        checklist,
        report_name=report_name,
        category=category,
        retrieve=cached_retrieve,
        assess=cached_assess,
    )


def assess_checklist_with_llm_live(
    checklist: Sequence[ChecklistItem],
    *,
    report_name: str,
    category: str,
    source_name: str,
    qdrant_client: QdrantClient,
    embed_client: OpenAI | None = None,
    openai_client: OpenAI | None = None,
    settings: Settings | None = None,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    limit: int = DEFAULT_LIMIT,
) -> ComplianceReport:
    """Production wiring: real `retrieve_evidence` + `llm_assess_batch`, scoped to one report."""
    settings = settings or get_settings()

    def retrieve(item: ChecklistItem) -> EvidenceResult:
        return retrieve_evidence(
            item,
            qdrant_client=qdrant_client,
            source_name=source_name,
            embed_client=embed_client,
            settings=settings,
            score_threshold=score_threshold,
            limit=limit,
            row_anchors=item.row_anchors,
        )

    def judge(pending: Sequence[PendingAssessment]) -> dict[str, Status]:
        return llm_assess_batch(pending, client=openai_client, settings=settings)

    return assess_checklist_with_llm(
        checklist,
        report_name=report_name,
        category=category,
        retrieve=retrieve,
        judge=judge,
    )
