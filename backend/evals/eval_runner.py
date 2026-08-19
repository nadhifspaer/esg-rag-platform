"""Score `eval_set.json` on routing/retrieval/answer. Run: python -m evals.eval_runner"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.models.conversation import ConversationTurn
from app.models.payload import Domain
from app.rag.chat_engine import build_chat_engine, scoped_source_names
from app.rag.compliance.checklist import ChecklistItem
from app.rag.compliance.sweep import ranking_refusal_for_item
from app.rag.conversation import apply_context, resolve_context
from app.rag.entity_index import default_entity_index
from app.rag.generation.loop import answer_with_retries
from app.rag.generation.prompts import CHAT_GENERATION_TYPES

EVAL_SET = Path(__file__).parent / "eval_set.json"
CHECKLIST_DIR = Path(__file__).parent.parent / "app" / "rag" / "compliance" / "checklists"

# Pipelines the runner does not drive yet; marked SKIP rather than scored.
UNWIRED_SHAPES = {"compliance_table", "ranked_with_segregation"}


class Verdict(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    # Right magnitude, re-punctuated (`3,63` -> `3.63`): narrower than a plain pass or fail.
    NUMERIC_EQUIVALENT = "NUMEQ"
    # The entry declares (via `expected_divergence`) that this answer is known to diverge.
    DIVERGENT = "DIVERGENT"
    # Same expected_divergence entry, but it actually matched ground truth this run.
    PASS_DIVERGENT = "PASS_DIVERGENT"


@dataclass
class AxisResult:
    verdict: Verdict
    detail: str = ""


@dataclass
class QuestionResult:
    qid: str
    shape: str
    routing: AxisResult
    retrieval: AxisResult
    answer: AxisResult
    elapsed: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- helpers


def _norm(text: str) -> str:
    """Collapse whitespace and lowercase, for substring matching against an answer."""
    return re.sub(r"\s+", " ", text or "").lower()


def _contains_value(answer: str, value: str) -> bool:
    """Is `value` present in `answer` as written? Deliberately literal, no decimal normalising."""
    return _norm(value) in _norm(answer)


_SIMPLE_COMMA_DECIMAL = re.compile(r"^\d+,\d+$")
_AMBIGUOUS_DOT = re.compile(r"^\d+\.\d{3}$")
_THOUSANDS_DOT_INTEGER = re.compile(r"^\d{1,3}(\.\d{3}){2,}$")


def _decimal_mark_variant(answer: str, value: str) -> bool:
    """Is `value` present as a pure decimal-mark swap (`3,63` -> `3.63`), never an ambiguous one?"""
    if not _SIMPLE_COMMA_DECIMAL.match(value.strip()):
        return False
    swapped = value.strip().replace(",", ".")
    if _AMBIGUOUS_DOT.match(swapped):
        return False
    return _norm(swapped) in _norm(answer)


def _thousands_separator_variant(answer: str, value: str) -> bool:
    """Is `value` present as a pure thousands-separator swap, mirroring `_decimal_mark_variant`?"""
    stripped = value.strip()
    if not _THOUSANDS_DOT_INTEGER.match(stripped):
        return False
    swapped = stripped.replace(".", ",")
    return _norm(swapped) in _norm(answer)


def _match_values(
    answer: str, values: Sequence[str], mode: str = "all"
) -> tuple[list[str], list[str]]:
    """Split `values` into (missing, decimal-mark-only), honouring `mode` (all/any/none)."""
    if mode == "none":
        present = [v for v in values if _contains_value(answer, v)]
        # Forbidden strings that appear are reported via `missing` for a uniform caller path.
        return present, []

    missing: list[str] = []
    reformatted: list[str] = []
    for value in values:
        if _contains_value(answer, value):
            continue
        if _decimal_mark_variant(answer, value) or _thousands_separator_variant(answer, value):
            reformatted.append(value)
        else:
            missing.append(value)

    if mode == "any":
        # Satisfied by one hit: a hit is any value that was neither missing nor reformatted.
        if len(missing) + len(reformatted) < len(values):
            return [], []
        if reformatted:
            return [], reformatted
        return missing, []
    return missing, reformatted


ABSENCE_PATTERNS = (
    "does not",
    "do not contain",
    "not disclose",
    "not separately",
    "no scope 3",
    "not reported",
    "no separately",
    "does not carry",
    "not available",
    "no ",
)

NEGATIVE_PATTERNS = ("no.", "no,", "does not have", "tidak", "no net zero", "no target")

# An answer reporting OUR retrieval failed, as opposed to reporting what the source said.
SOURCE_SILENT_PATTERNS = (
    "sources do not contain",
    "provided sources do not",
    "no information",
    "could not find",
    "not found in the",
)


def _load_checklists() -> dict[str, ChecklistItem]:
    """Load every framework's checklist into one flat, code-keyed dict; raise on a collision."""
    items: dict[str, ChecklistItem] = {}
    for framework in ("gri", "edgb"):
        for category in ("environmental", "social", "governance"):
            data = json.loads(
                (CHECKLIST_DIR / framework / f"{category}.json").read_text(encoding="utf-8")
            )
            for raw in data["items"]:
                item = ChecklistItem(**raw)
                if item.code in items:
                    raise ValueError(
                        f"duplicate checklist code {item.code!r}: "
                        f"{framework}/{category}.json collides with an earlier entry"
                    )
                items[item.code] = item
    return items


def _requirement_code(question: dict[str, Any]) -> str | None:
    """The requirement a refusal entry targets: an explicit field, or a GRI code in the question."""
    explicit = question.get("target_requirement")
    if explicit:
        return str(explicit)
    match = re.search(r"GRI\s+\d+(?:-\d+)?", question.get("question", ""))
    return match.group(0) if match else None


# --------------------------------------------------------------------------- axes


def score_routing(
    expected: dict[str, Any], query_type: str, domains: Sequence[str], scoped_to: str | None
) -> AxisResult:
    """Did the classifier pick the right query_type, domain set, and report scope?"""
    problems: list[str] = []

    want_type = expected.get("expected_query_type")
    if want_type and query_type != want_type:
        problems.append(f"query_type {query_type!r} != {want_type!r}")

    want_domains = set(expected.get("expected_domains") or [])
    got_domains = set(domains)
    if want_domains and got_domains != want_domains:
        problems.append(f"domains {sorted(got_domains)} != {sorted(want_domains)}")

    want_scope = expected.get("expected_scoped_to")
    # Compare loosely: the eval set stores the bank name, retrieval stores the citation label.
    if want_scope:
        if not scoped_to or _norm(want_scope) not in _norm(scoped_to):
            problems.append(f"scoped_to {scoped_to!r} does not match {want_scope!r}")
    elif "expected_scoped_to" in expected and expected["expected_scoped_to"] is None and scoped_to:
        problems.append(f"scoped_to {scoped_to!r} but expected domain-wide")

    if problems:
        return AxisResult(Verdict.FAIL, "; ".join(problems))
    return AxisResult(Verdict.PASS, f"{query_type} / {sorted(got_domains)} / scope={scoped_to}")


def score_retrieval(expected_sources: list[dict[str, Any]], citations: Sequence[Any]) -> AxisResult:
    """Did every expected source land in the results, at the expected page where given?"""
    if not expected_sources:
        return AxisResult(Verdict.SKIP, "no expected sources")

    got = [(str(c.source_name), str(c.page_number)) for c in citations]
    misses: list[str] = []
    page_misses: list[str] = []

    for want in expected_sources:
        needle = _norm(want.get("source_contains", ""))
        page = want.get("page_number")
        doc_hits = [(s, p) for s, p in got if needle in _norm(s)]
        if not doc_hits:
            misses.append(f"{want.get('source_contains')!r} absent")
            continue
        if page is not None and not any(p == str(page) for _, p in doc_hits):
            page_misses.append(
                f"{want.get('source_contains')!r} found but not p{page} "
                f"(got p{[p for _, p in doc_hits]})"
            )

    if misses:
        return AxisResult(Verdict.FAIL, "; ".join(misses + page_misses))
    if page_misses:
        return AxisResult(Verdict.FAIL, "doc hit, page miss: " + "; ".join(page_misses))
    return AxisResult(Verdict.PASS, f"{len(expected_sources)} source(s) matched incl. page")


def _score_answer_shape(question: dict[str, Any], answer: str) -> AxisResult:
    """Apply the entry's `expected_answer_shape` rule to the generated answer."""
    shape = question.get("expected_answer_shape", "plain_value")
    values = question.get("ground_truth_values") or []
    mode = question.get("match_mode", "all")

    if shape == "partial_evidence":
        # The failure mode is overclaiming, not omission: values are FORBIDDEN strings here.
        asserted, _ = _match_values(answer, values, "none")
        if asserted:
            return AxisResult(
                Verdict.FAIL,
                f"overclaims beyond the evidence: asserted {asserted} when the sources "
                "establish only the requirement and the filing's existence",
            )
        return AxisResult(Verdict.PASS, "no compliance/timeliness claim beyond the evidence")

    if shape in ("plain_value", "source_oddity", "disclosed_not_totalled"):
        missing, reformatted = _match_values(answer, values, mode)
        if missing:
            return AxisResult(Verdict.FAIL, f"missing value(s) {missing}")
        if reformatted:
            return AxisResult(
                Verdict.NUMERIC_EQUIVALENT,
                f"right quantity, re-punctuated: {reformatted}",
            )
        return AxisResult(Verdict.PASS, f"stated {values}")

    if shape == "verbatim":
        # No decimal-mark leniency: punctuation itself is the open question for this shape.
        missing = [v for v in values if not _contains_value(answer, v)]
        if missing:
            return AxisResult(Verdict.FAIL, f"not quoted verbatim, missing {missing}")
        return AxisResult(Verdict.PASS, f"quoted verbatim {values}")

    if shape == "multi_value":
        missing, reformatted = _match_values(answer, values, mode)
        if missing:
            return AxisResult(
                Verdict.FAIL,
                f"multi-value answer incomplete, missing {missing} — a partial answer here "
                "misstates the bank's scope",
            )
        if reformatted:
            return AxisResult(
                Verdict.NUMERIC_EQUIVALENT, f"all values present, re-punctuated: {reformatted}"
            )
        return AxisResult(Verdict.PASS, f"all {len(values)} scoped values stated")

    if shape in ("absence", "verified_negative"):
        low = _norm(answer)
        if not any(p in low for p in ABSENCE_PATTERNS):
            return AxisResult(Verdict.FAIL, "no absence/negative statement found in answer")
        return AxisResult(Verdict.PASS, "answer states the figure/requirement is absent")

    if shape == "negative_disclosure":
        low = _norm(answer)
        if any(p in low for p in SOURCE_SILENT_PATTERNS):
            return AxisResult(
                Verdict.FAIL,
                "answer reports OUR retrieval found nothing, but the source gave an "
                "explicit 'no' — these mean opposite things",
            )
        if not any(p in low for p in NEGATIVE_PATTERNS):
            return AxisResult(Verdict.FAIL, "no explicit negative disclosure found")
        return AxisResult(Verdict.PASS, "reports the source's own negative answer")

    return AxisResult(Verdict.SKIP, f"no rule for shape {shape!r}")


def score_answer(question: dict[str, Any], answer: str) -> AxisResult:
    """Score `question` against `answer`, relabelling when `expected_divergence` is set."""
    result = _score_answer_shape(question, answer)
    divergence = question.get("expected_divergence")
    if divergence and result.verdict is Verdict.PASS:
        return AxisResult(
            Verdict.PASS_DIVERGENT, f"{result.detail} (expected divergence: {divergence})"
        )
    if divergence and result.verdict is not Verdict.SKIP:
        return AxisResult(Verdict.DIVERGENT, f"{result.detail} (expected divergence: {divergence})")
    return result


# --------------------------------------------------------------------------- runners


def run_refusal(question: dict[str, Any], checklists: dict[str, ChecklistItem]) -> QuestionResult:
    """Score a refusal entry. Deterministic, decided before any retrieval, costs nothing."""
    code = _requirement_code(question)
    if not code or code not in checklists:
        return QuestionResult(
            qid=question["id"],
            shape="refusal",
            routing=AxisResult(Verdict.SKIP, "no resolvable requirement code"),
            retrieval=AxisResult(Verdict.SKIP, "refusal path performs no retrieval"),
            answer=AxisResult(
                Verdict.FAIL, f"could not resolve requirement code from entry ({code!r})"
            ),
        )

    refusal = ranking_refusal_for_item(checklists[code], bank_count=21)
    want = (question.get("ground_truth_values") or [None])[0]

    if refusal is None:
        answer_axis = AxisResult(Verdict.FAIL, f"{code} was NOT refused; expected {want!r}")
    elif want and want not in (refusal.reason, f"route_to: {refusal.route_to}"):
        answer_axis = AxisResult(
            Verdict.FAIL,
            f"refused with reason={refusal.reason!r} "
            f"route_to={refusal.route_to!r}, expected {want!r}",
        )
    else:
        answer_axis = AxisResult(
            Verdict.PASS, f"reason={refusal.reason} route_to={refusal.route_to}"
        )

    return QuestionResult(
        qid=question["id"],
        shape="refusal",
        routing=AxisResult(Verdict.SKIP, "refusal decided from the curated item, not routed"),
        retrieval=AxisResult(Verdict.SKIP, "zero embeddings, zero Qdrant round-trips"),
        answer=answer_axis,
        extra={"code": code},
    )


def _route(engine: Any, utterance: str, history: Sequence[ConversationTurn]):
    """Mirror `api/chat.py`'s routing for one turn."""
    context = resolve_context(utterance, list(history), index=default_entity_index())
    classification = engine.classify(utterance, use_llm_fallback=context.inherited_entity is None)
    classification = apply_context(classification, context)
    domains = classification.domains
    source_names = scoped_source_names(classification, domains)
    return context, classification, domains, source_names


def run_chat(question: dict[str, Any], engine: Any) -> QuestionResult:
    """Score a chat-path entry through the real retrieve -> generate -> judge loop."""
    started = time.time()
    utterance = question["question"]
    _, classification, domains, source_names = _route(engine, utterance, [])

    routing = score_routing(
        question,
        classification.query_type.value,
        [d.value for d in domains],
        source_names.get(Domain.COMPANY_DOCUMENTS),
    )

    # Mirror the endpoint's up-front rejection: a refused query type is a routing outcome,
    # so retrieval/answer are marked SKIP rather than FAIL: they never ran.
    if classification.query_type not in CHAT_GENERATION_TYPES:
        return QuestionResult(
            qid=question["id"],
            shape=question.get("expected_answer_shape", "plain_value"),
            routing=AxisResult(
                Verdict.FAIL,
                f"routed as {classification.query_type.value!r}, which /chat refuses (422)",
            ),
            retrieval=AxisResult(Verdict.SKIP, "rejected before retrieval"),
            answer=AxisResult(Verdict.SKIP, "rejected before generation"),
            elapsed=time.time() - started,
        )

    if not domains:
        return QuestionResult(
            qid=question["id"],
            shape=question.get("expected_answer_shape", "plain_value"),
            routing=routing,
            retrieval=AxisResult(Verdict.FAIL, "no domain resolved; retrieval never ran"),
            answer=AxisResult(Verdict.FAIL, "no domain resolved; generation never ran"),
            elapsed=time.time() - started,
        )

    result = answer_with_retries(
        utterance,
        retrieve=engine.make_retrieve(domains, source_names=source_names),
        generate=engine.make_generate(classification.query_type, False),
        check=engine.make_check(),
    )

    return QuestionResult(
        qid=question["id"],
        shape=question.get("expected_answer_shape", "plain_value"),
        routing=routing,
        retrieval=score_retrieval(question.get("expected_sources") or [], result.citations),
        answer=score_answer(question, result.answer),
        elapsed=time.time() - started,
        extra={
            "attempts": result.attempts,
            "stop_reason": result.stop_reason.value,
            "answer": result.answer,
            "citations": [c.label for c in result.citations],
        },
    )


def run_multi_turn(question: dict[str, Any], engine: Any) -> list[QuestionResult]:
    """Score each turn of a multi-turn entry separately, carrying history forward."""
    started = time.time()
    history: list[ConversationTurn] = []
    out: list[QuestionResult] = []

    for turn in question.get("turns", []):
        utterance = turn["utterance"]
        context, classification, domains, source_names = _route(engine, utterance, history)

        expected = {
            "expected_query_type": question.get("expected_query_type"),
            "expected_domains": question.get("expected_domains"),
            "expected_scoped_to": turn.get("expected_scoped_to"),
        }
        routing = score_routing(
            expected,
            classification.query_type.value,
            [d.value for d in domains],
            source_names.get(Domain.COMPANY_DOCUMENTS),
        )
        # Context shape is part of routing on a follow-up: inheriting the entity should
        # keep turn 2 pointed at the same report.
        want_ctx = turn.get("expected_context_source")
        got_ctx = context.context_source.value
        if want_ctx and got_ctx != want_ctx and routing.verdict is Verdict.PASS:
            routing = AxisResult(Verdict.FAIL, f"context_source {got_ctx!r} != {want_ctx!r}")
        elif want_ctx:
            routing.detail += f" / context={got_ctx}"

        if not domains:
            out.append(
                QuestionResult(
                    qid=f"{question['id']}#t{turn['turn']}",
                    shape="multi_turn",
                    routing=routing,
                    retrieval=AxisResult(Verdict.FAIL, "no domain resolved"),
                    answer=AxisResult(Verdict.FAIL, "no domain resolved"),
                )
            )
            continue

        result = answer_with_retries(
            utterance,
            retrieve=engine.make_retrieve(domains, source_names=source_names),
            generate=engine.make_generate(classification.query_type, False, history),
            check=engine.make_check(),
            retrieval_seed=context.retrieval_seed,
        )

        out.append(
            QuestionResult(
                qid=f"{question['id']}#t{turn['turn']}",
                shape="multi_turn",
                routing=routing,
                retrieval=score_retrieval(turn.get("expected_sources") or [], result.citations),
                answer=score_answer(turn, result.answer),
                elapsed=time.time() - started,
                extra={
                    "attempts": result.attempts,
                    "seed": context.retrieval_seed,
                    "answer": result.answer,
                    "citations": [c.label for c in result.citations],
                },
            )
        )
        history.append(
            ConversationTurn(
                question=utterance,
                answer=result.answer,
                retrieval_query=result.retrieval_query,
                pipeline="chat",
            )
        )

    return out


# --------------------------------------------------------------------------- report


def print_report(results: list[QuestionResult]) -> None:
    print("\n" + "=" * 100)
    print(f"{'question':<14}{'shape':<22}{'ROUTING':<10}{'RETRIEVAL':<11}{'ANSWER':<16}detail")
    print("=" * 100)

    for r in results:
        detail = r.answer.detail or r.retrieval.detail or r.routing.detail
        print(
            f"{r.qid:<14}{r.shape:<22}"
            f"{r.routing.verdict.value:<10}{r.retrieval.verdict.value:<11}"
            f"{r.answer.verdict.value:<16}{detail[:44]}"
        )
        for axis_name, axis in (("routing", r.routing), ("retrieval", r.retrieval)):
            if axis.verdict is Verdict.FAIL:
                print(f"{'':<14}  -> {axis_name}: {axis.detail}")

    print("-" * 100)
    for axis_name in ("routing", "retrieval", "answer"):
        axis_results = [getattr(r, axis_name) for r in results]
        passed = sum(1 for a in axis_results if a.verdict is Verdict.PASS)
        failed = sum(1 for a in axis_results if a.verdict is Verdict.FAIL)
        skipped = sum(1 for a in axis_results if a.verdict is Verdict.SKIP)
        numeq = sum(1 for a in axis_results if a.verdict is Verdict.NUMERIC_EQUIVALENT)
        diverged = sum(1 for a in axis_results if a.verdict is Verdict.DIVERGENT)
        pass_divergent = sum(1 for a in axis_results if a.verdict is Verdict.PASS_DIVERGENT)
        scored = passed + failed + numeq
        # NUMEQ, DIVERGENT, and PASS_DIVERGENT are all excluded from the rate: none of them
        # is a plain pass or fail of the pipeline this run.
        rate = f"{passed / scored:.0%}" if scored else "n/a"
        print(
            f"  {axis_name.upper():<10} pass {passed:>3}  fail {failed:>3}  "
            f"numeq {numeq:>3}  skip {skipped:>3}  diverged {diverged:>3}  "
            f"pass_divergent {pass_divergent:>3}   strict rate {rate}"
        )
    print("-" * 100)
    print("  Axes are independent on purpose: an entry may legitimately pass one and fail")
    print("  another (mt-001 turn 2 is expected to route correctly and retrieve wrongly).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="comma-separated question ids")
    parser.add_argument("--shape", help="only entries with this expected_answer_shape")
    parser.add_argument("--limit", type=int, help="stop after N questions")
    parser.add_argument("--set", default=str(EVAL_SET), help="path to eval_set.json")
    parser.add_argument(
        "--json",
        dest="json_out",
        help="also write per-question verdicts here as JSON, for diffing repeat runs",
    )
    args = parser.parse_args()

    data = json.loads(Path(args.set).read_text(encoding="utf-8"))
    questions = data["questions"]

    if args.only:
        wanted = {q.strip() for q in args.only.split(",")}
        questions = [q for q in questions if q["id"] in wanted]
    if args.shape:
        questions = [q for q in questions if q.get("expected_answer_shape") == args.shape]
    if args.limit:
        questions = questions[: args.limit]

    checklists = _load_checklists()
    settings = get_settings()
    engine = None
    results: list[QuestionResult] = []

    print(f"Scoring {len(questions)} question(s) from {args.set}")

    for question in questions:
        shape = question.get("expected_answer_shape", "plain_value")

        if shape in UNWIRED_SHAPES:
            results.append(
                QuestionResult(
                    qid=question["id"],
                    shape=shape,
                    routing=AxisResult(Verdict.SKIP, "pipeline not wired into the runner"),
                    retrieval=AxisResult(Verdict.SKIP, "pipeline not wired into the runner"),
                    answer=AxisResult(Verdict.SKIP, "pipeline not wired into the runner"),
                )
            )
            continue

        if shape == "refusal":
            results.append(run_refusal(question, checklists))
            continue

        if engine is None:
            print("Building chat engine (corpus scroll + BM25 index + cross-encoder) ...")
            engine = build_chat_engine(settings)

        print(f"  running {question['id']} ...", flush=True)
        # One question must never take the run down: a raised pipeline is a finding.
        try:
            if shape == "multi_turn":
                results.extend(run_multi_turn(question, engine))
            else:
                results.append(run_chat(question, engine))
        except Exception as exc:  # noqa: BLE001 (a raised pipeline is a finding, not a crash)
            print(f"    ERROR {type(exc).__name__}: {exc}", flush=True)
            results.append(
                QuestionResult(
                    qid=question["id"],
                    shape=shape,
                    routing=AxisResult(Verdict.FAIL, f"pipeline raised: {type(exc).__name__}"),
                    retrieval=AxisResult(Verdict.FAIL, "pipeline raised before results"),
                    answer=AxisResult(Verdict.FAIL, f"{type(exc).__name__}: {exc}"),
                )
            )

    print_report(results)

    if args.json_out:
        payload = [
            {
                "qid": r.qid,
                "shape": r.shape,
                "routing": r.routing.verdict.value,
                "retrieval": r.retrieval.verdict.value,
                "answer": r.answer.verdict.value,
                "routing_detail": r.routing.detail,
                "retrieval_detail": r.retrieval.detail,
                "answer_detail": r.answer.detail,
                # The generated answer and its citations, so a FAIL can be told apart from
                # a correctly-worded answer the matcher missed.
                "answer_text": r.extra.get("answer"),
                "citations": r.extra.get("citations"),
            }
            for r in results
        ]
        Path(args.json_out).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nwrote {len(payload)} per-question verdicts to {args.json_out}")


if __name__ == "__main__":
    main()
