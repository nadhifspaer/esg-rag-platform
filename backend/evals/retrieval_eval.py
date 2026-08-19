"""Retrieval-only eval: does the correct source land in top-5? (python -m evals.retrieval_eval)"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import get_settings
from app.models.payload import Domain
from app.rag.hybrid_search import hybrid_search
from app.rag.keyword_search import KeywordIndex, load_corpus_documents
from app.rag.reranker import CrossEncoderReranker, rerank
from app.rag.vector_store import get_client

# How many candidates hybrid search returns before re-ranking, and how many we inspect.
CANDIDATE_LIMIT = 10
TOP_K = 5


@dataclass(frozen=True)
class EvalQuestion:
    """One eval item: a question, its domain, and how to recognise a correct source."""

    query: str
    domain: Domain
    expected_source_contains: str
    note: str = ""


# 9 questions, ~4-5 per domain, phrased as disclosure/standard lookups the corpus can answer.
QUESTIONS: tuple[EvalQuestion, ...] = (
    # --- company_documents ---------------------------------------------------
    EvalQuestion(
        query="What are BCA's Scope 1 and Scope 2 greenhouse gas emissions?",
        domain=Domain.COMPANY_DOCUMENTS,
        expected_source_contains="Bank Central Asia",
    ),
    EvalQuestion(
        query="How much total energy did Bank Mandiri consume during the year?",
        domain=Domain.COMPANY_DOCUMENTS,
        expected_source_contains="Bank Mandiri",
    ),
    EvalQuestion(
        query="What is Bank Raya's total number of employees in its workforce disclosure?",
        domain=Domain.COMPANY_DOCUMENTS,
        expected_source_contains="Bank Raya",
    ),
    EvalQuestion(
        query="What sustainable finance or green financing portfolio does BNI report?",
        domain=Domain.COMPANY_DOCUMENTS,
        expected_source_contains="Bank Negara Indonesia",
    ),
    EvalQuestion(
        query="What does CIMB Niaga disclose about its water or waste management?",
        domain=Domain.COMPANY_DOCUMENTS,
        expected_source_contains="CIMB Niaga",
    ),
    # --- standards -----------------------------------------------------------
    EvalQuestion(
        query="What are the reporting principles for a GRI report under GRI 1 Foundation?",
        domain=Domain.STANDARDS,
        expected_source_contains="GRI 1",
    ),
    EvalQuestion(
        query="What general disclosures about the organization does GRI 2 require?",
        domain=Domain.STANDARDS,
        expected_source_contains="GRI 2",
    ),
    EvalQuestion(
        query="How does GRI 3 guide the process of determining material topics?",
        domain=Domain.STANDARDS,
        expected_source_contains="GRI 3",
    ),
    EvalQuestion(
        query="What ESG disclosure topics does the ESG Disclosure Guide Book cover?",
        domain=Domain.STANDARDS,
        expected_source_contains="ESG Disclosure Guide Book",
    ),
)


@dataclass(frozen=True)
class QuestionResult:
    """Outcome for one question: hit/miss, the first correct result's rank, top-5 sources."""

    question: EvalQuestion
    hit: bool
    hit_rank: int | None
    top_sources: list[str]


def _first_hit_rank(source_names: list[str], needle: str) -> int | None:
    """1-based rank of the first source name containing `needle` (case-insensitive)."""
    lowered = needle.lower()
    for rank, name in enumerate(source_names, start=1):
        if lowered in name.lower():
            return rank
    return None


def run_eval() -> list[QuestionResult]:
    """Run every question through hybrid search + re-rank and score top-5 hits."""
    settings = get_settings()
    client = get_client(settings)

    # Build the BM25 index and load the cross-encoder once, up front.
    print(f"Loading corpus from Qdrant collection '{settings.qdrant_collection}' ...")
    corpus = load_corpus_documents(client, settings.qdrant_collection)
    print(f"  {len(corpus)} chunks loaded; building BM25 index.")
    keyword_index = KeywordIndex(corpus)
    reranker = CrossEncoderReranker(settings=settings)

    results: list[QuestionResult] = []
    for question in QUESTIONS:
        fused = hybrid_search(
            question.query,
            domain=question.domain,
            qdrant_client=client,
            keyword_index=keyword_index,
            settings=settings,
            limit=CANDIDATE_LIMIT,
        )
        reranked = rerank(question.query, fused, scorer=reranker, top_n=TOP_K)
        top_sources = [r.payload.source_name for r in reranked]
        hit_rank = _first_hit_rank(top_sources, question.expected_source_contains)
        results.append(
            QuestionResult(
                question=question,
                hit=hit_rank is not None,
                hit_rank=hit_rank,
                top_sources=top_sources,
            )
        )
    return results


def print_report(results: list[QuestionResult]) -> None:
    """Print a per-question line and the overall top-5 hit rate."""
    print("\n" + "=" * 78)
    print(f"RETRIEVAL-ONLY EVAL  —  top-{TOP_K} source hit rate")
    print("=" * 78)

    by_domain: dict[Domain, list[QuestionResult]] = {}
    for result in results:
        by_domain.setdefault(result.question.domain, []).append(result)

    for domain, domain_results in by_domain.items():
        print(f"\n[{domain.value}]")
        for result in domain_results:
            mark = "HIT " if result.hit else "MISS"
            rank = f"@{result.hit_rank}" if result.hit_rank else "  "
            print(f"  {mark} {rank}  {result.question.query}")
            if not result.hit:
                # Show what did come back, so a miss is diagnosable at a glance.
                shown = ", ".join(dict.fromkeys(result.top_sources)) or "(no results)"
                print(
                    f"          expected ~ '{result.question.expected_source_contains}' | "
                    f"top-5: {shown}"
                )

    hits = sum(1 for r in results if r.hit)
    total = len(results)
    rate = hits / total if total else 0.0
    print("\n" + "-" * 78)
    print(f"HIT RATE: {hits}/{total} = {rate:.0%}")
    print("-" * 78)


def main() -> None:
    try:
        results = run_eval()
    except Exception as exc:  # noqa: BLE001 - surface any config/connection issue clearly
        print(f"\nEval could not run: {type(exc).__name__}: {exc}")
        print(
            "This eval needs a configured .env (QDRANT_URL/QDRANT_API_KEY and "
            "OPENAI_API_KEY) and an ingested collection. See .env.example."
        )
        raise SystemExit(1) from exc
    print_report(results)


if __name__ == "__main__":
    main()
