"""RAGAS scoring: faithfulness, relevancy, context precision/recall (python -m evals.ragas_eval)."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.models.citation import Citation
from app.models.conversation import ConversationTurn
from app.rag.chat_engine import build_chat_engine
from app.rag.generation.loop import answer_with_retries
from app.rag.generation.prompts import CHAT_GENERATION_TYPES

# `_route` is imported, not reimplemented, so the two harnesses can't drift on routing.
from evals.eval_runner import UNWIRED_SHAPES, _route

EVAL_SET = Path(__file__).parent / "eval_set.json"

# Shapes the chat pipeline never answers, so there is nothing for RAGAS to score.
SKIP_SHAPES = set(UNWIRED_SHAPES) | {"refusal"}

EVALUATOR_MODEL = "gpt-4.1-mini"
EVALUATOR_EMBEDDINGS = "text-embedding-3-small"


@dataclass
class Sample:
    """One scoreable row: what RAGAS needs, plus provenance for reading the score."""

    qid: str
    question: str
    answer: str
    contexts: list[str]
    reference: str
    citations: list[str] = field(default_factory=list)
    attempts: int = 0
    capture_verified: bool = True
    elapsed: float = 0.0


@dataclass
class Skipped:
    qid: str
    reason: str


class _CapturingRetrieve:
    """Wrap the loop's `retrieve` and keep the chunks from the most recent (final) call."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.last_chunks: list[Any] = []
        self.calls = 0

    def __call__(self, query: str) -> Sequence[Any]:
        chunks = self._inner(query)
        self.last_chunks = list(chunks)
        self.calls += 1
        return chunks


def _verify_capture(captured: Sequence[Any], citations: Sequence[Citation]) -> bool:
    """Confirm the captured chunks are the ones behind the returned answer."""
    return [Citation.from_result(c) for c in captured] == list(citations)


def collect_sample(question: dict[str, Any], engine: Any) -> tuple[Sample | None, str]:
    """Run one chat question through the real pipeline and package it for RAGAS."""
    started = time.time()
    utterance = question["question"]
    _, classification, domains, source_names = _route(engine, utterance, [])

    if classification.query_type not in CHAT_GENERATION_TYPES:
        return None, f"routed {classification.query_type.value!r}, which /chat refuses (422)"
    if not domains:
        return None, "no domain resolved; retrieval never ran"

    retrieve = _CapturingRetrieve(engine.make_retrieve(domains, source_names=source_names))
    result = answer_with_retries(
        utterance,
        retrieve=retrieve,
        generate=engine.make_generate(classification.query_type, False),
        check=engine.make_check(),
    )

    return (
        Sample(
            qid=question["id"],
            question=utterance,
            answer=result.answer,
            contexts=[c.payload.chunk_text for c in retrieve.last_chunks],
            reference=question["ground_truth"],
            citations=[c.label for c in result.citations],
            attempts=result.attempts,
            capture_verified=_verify_capture(retrieve.last_chunks, result.citations),
            elapsed=time.time() - started,
        ),
        "",
    )


def collect_multi_turn(question: dict[str, Any], engine: Any) -> list[Sample]:
    """One sample per turn, carrying history forward exactly as `/chat` does."""
    history: list[ConversationTurn] = []
    out: list[Sample] = []

    for turn in question.get("turns", []):
        started = time.time()
        utterance = turn["utterance"]
        context, classification, domains, source_names = _route(engine, utterance, history)
        if classification.query_type not in CHAT_GENERATION_TYPES or not domains:
            continue

        retrieve = _CapturingRetrieve(engine.make_retrieve(domains, source_names=source_names))
        result = answer_with_retries(
            utterance,
            retrieve=retrieve,
            generate=engine.make_generate(classification.query_type, False, history),
            check=engine.make_check(),
            retrieval_seed=context.retrieval_seed,
        )

        out.append(
            Sample(
                qid=f"{question['id']}#t{turn['turn']}",
                question=utterance,
                answer=result.answer,
                contexts=[c.payload.chunk_text for c in retrieve.last_chunks],
                # A turn may carry its own ground truth; fall back to the entry's.
                reference=turn.get("ground_truth") or question.get("ground_truth", ""),
                citations=[c.label for c in result.citations],
                attempts=result.attempts,
                capture_verified=_verify_capture(retrieve.last_chunks, result.citations),
                elapsed=time.time() - started,
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


# --- RAGAS block: version-specific, imported function-local so a plan works without ragas ---


def score_with_ragas(samples: Sequence[Sample]) -> list[dict[str, Any]]:
    """Score collected samples. Targets the RAGAS 0.2+/0.3 API."""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import SingleTurnSample
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    settings = get_settings()
    llm = LangchainLLMWrapper(
        ChatOpenAI(model=EVALUATOR_MODEL, temperature=0, api_key=settings.openai_api_key)
    )
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model=EVALUATOR_EMBEDDINGS, api_key=settings.openai_api_key)
    )

    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=s.question,
                response=s.answer,
                retrieved_contexts=s.contexts,
                reference=s.reference,
            )
            for s in samples
        ]
    )

    report = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ],
        llm=llm,
        embeddings=embeddings,
    )

    # `to_pandas()` is RAGAS's documented way out; the column names are the metric names.
    frame = report.to_pandas()
    rows: list[dict[str, Any]] = []
    for sample, (_, row) in zip(samples, frame.iterrows(), strict=False):
        rows.append(
            {
                "qid": sample.qid,
                "faithfulness": _as_float(row.get("faithfulness")),
                "answer_relevancy": _as_float(
                    row.get("answer_relevancy", row.get("response_relevancy"))
                ),
                "context_precision": _as_float(
                    row.get("context_precision", row.get("llm_context_precision_with_reference"))
                ),
                "context_recall": _as_float(row.get("context_recall")),
                "attempts": sample.attempts,
                "n_contexts": len(sample.contexts),
                "capture_verified": sample.capture_verified,
                "citations": sample.citations,
                "answer_text": sample.answer,
            }
        )
    return rows


def _as_float(value: Any) -> float | None:
    """NaN and None both mean 'could not score'; kept distinguishable from a real 0.0."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if out != out else out


# --------------------------------------------------------------------------- reporting


def print_report(rows: Sequence[dict[str, Any]], skipped: Sequence[Skipped]) -> None:
    metrics = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
    print("\n" + "=" * 100)
    header = f"{'question':<14}{'faith':>8}{'relev':>8}{'ctx_prec':>10}{'ctx_rec':>9}"
    print(f"{header}   {'att':>3}  note")
    print("=" * 100)
    for r in rows:
        cells = "".join(
            f"{r[m]:>8.3f}" if isinstance(r[m], float) else f"{'n/a':>8}" for m in metrics[:2]
        )
        cells += "".join(
            f"{r[m]:>10.3f}" if isinstance(r[m], float) else f"{'n/a':>10}" for m in metrics[2:3]
        )
        cells += "".join(
            f"{r[m]:>9.3f}" if isinstance(r[m], float) else f"{'n/a':>9}" for m in metrics[3:]
        )
        note = "" if r["capture_verified"] else "CONTEXT CAPTURE UNVERIFIED"
        print(f"{r['qid']:<14}{cells}   {r['attempts']:>3}  {note}")

    print("-" * 100)
    for m in metrics:
        vals = [r[m] for r in rows if isinstance(r[m], float)]
        mean = sum(vals) / len(vals) if vals else float("nan")
        # The denominator travels with the mean: partial coverage is a different claim.
        print(f"  {m:<20} mean {mean:.3f}   over {len(vals)}/{len(rows)} scored")
    print("-" * 100)
    if skipped:
        print(f"\nskipped {len(skipped)}:")
        for s in skipped:
            print(f"  {s.qid:<10} {s.reason}")


def plan(questions: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[Skipped]]:
    """Split the eval set into what would be scored and what would not, no API calls."""
    included: list[dict[str, Any]] = []
    skipped: list[Skipped] = []
    for q in questions:
        shape = q.get("expected_answer_shape", "plain_value")
        if shape in SKIP_SHAPES:
            reason = (
                "refusal path: decided pre-retrieval, zero contexts"
                if shape == "refusal"
                else f"{shape}: pipeline not wired into this harness"
            )
            skipped.append(Skipped(q["id"], reason))
        else:
            included.append(q)
    return included, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="comma-separated question ids")
    parser.add_argument("--limit", type=int, help="stop after N questions")
    parser.add_argument("--set", default=str(EVAL_SET), help="path to eval_set.json")
    parser.add_argument("--json", dest="json_out", help="write per-sample scores here")
    parser.add_argument(
        "--plan",
        action="store_true",
        help="print what would be scored and exit, making no API calls of any kind",
    )
    args = parser.parse_args()

    questions = json.loads(Path(args.set).read_text(encoding="utf-8"))["questions"]
    if args.only:
        wanted = {q.strip() for q in args.only.split(",")}
        questions = [q for q in questions if q["id"] in wanted]

    included, skipped = plan(questions)
    if args.limit:
        included = included[: args.limit]

    if args.plan:
        n_samples = sum(
            2 if q.get("expected_answer_shape") == "multi_turn" else 1 for q in included
        )
        print(f"would score {len(included)} entries -> {n_samples} samples")
        for q in included:
            print(f"  {q['id']:<10} {q.get('expected_answer_shape', 'plain_value')}")
        print(f"\nwould skip {len(skipped)}:")
        for s in skipped:
            print(f"  {s.qid:<10} {s.reason}")
        return

    settings = get_settings()
    print("Building chat engine (corpus scroll + BM25 index + cross-encoder) ...")
    engine = build_chat_engine(settings)

    samples: list[Sample] = []
    for q in included:
        print(f"  running {q['id']} ...", flush=True)
        try:
            if q.get("expected_answer_shape") == "multi_turn":
                samples.extend(collect_multi_turn(q, engine))
            else:
                sample, reason = collect_sample(q, engine)
                if sample is None:
                    skipped.append(Skipped(q["id"], reason))
                else:
                    samples.append(sample)
        except Exception as exc:  # noqa: BLE001 (a raised pipeline is a finding, not a crash)
            print(f"    ERROR {type(exc).__name__}: {exc}")
            skipped.append(Skipped(q["id"], f"{type(exc).__name__}: {exc}"))

    if not samples:
        print("no samples collected; nothing to score")
        return

    print(f"\nscoring {len(samples)} sample(s) with RAGAS ({EVALUATOR_MODEL}) ...", flush=True)
    rows = score_with_ragas(samples)
    print_report(rows, skipped)

    if args.json_out:
        payload = {
            "evaluator_model": EVALUATOR_MODEL,
            "embedding_model": EVALUATOR_EMBEDDINGS,
            "samples": rows,
            "skipped": [{"qid": s.qid, "reason": s.reason} for s in skipped],
        }
        Path(args.json_out).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nwrote {len(rows)} sample scores to {args.json_out}")


if __name__ == "__main__":
    main()
