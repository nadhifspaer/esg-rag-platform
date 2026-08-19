"""Per-requirement evidence retrieval: the compliance pipeline's corrective retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from openai import OpenAI
from qdrant_client import QdrantClient

from app.core.config import Settings, get_settings
from app.models.payload import Domain
from app.rag.compliance.anchors import chunk_has_anchor
from app.rag.embedder import embed_texts
from app.rag.vector_store import SearchResult, search

# Cosine-similarity floor below which the first retrieval is treated as "found nothing
# on-topic" and the broadened retry fires. A heuristic starting value, tunable per call.
DEFAULT_SCORE_THRESHOLD = 0.35

# Evidence chunks to return per requirement.
DEFAULT_LIMIT = 5


class RetrievableItem(Protocol):
    """Structural type for anything scoped retrieval runs for: code, description, topic_keywords."""

    code: str
    description: str
    topic_keywords: Sequence[str]


@dataclass(frozen=True)
class EvidenceResult:
    """The evidence retrieved for one checklist item."""

    item_code: str
    query: str
    results: list[SearchResult]
    top_score: float
    fallback_used: bool
    # True when `row_anchors` were supplied and a retrieved chunk actually contained the row.
    anchor_matched: bool = False
    anchors_requested: bool = False


def build_base_query(item: RetrievableItem) -> str:
    """The focused first-pass query for a requirement: its plain-language description."""
    return item.description


def build_broadened_query(item: RetrievableItem) -> str:
    """The broadened retry query: the description plus the item's topic keywords."""
    return item.description + " " + " ".join(item.topic_keywords)


def _top_score(results: list[SearchResult]) -> float:
    """Best cosine score among results, or 0.0 when there are none."""
    return results[0].score if results else 0.0


def retrieve_evidence(
    item: RetrievableItem,
    *,
    qdrant_client: QdrantClient,
    source_name: str | None = None,
    collection_name: str | None = None,
    embed_client: OpenAI | None = None,
    settings: Settings | None = None,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    limit: int = DEFAULT_LIMIT,
    row_anchors: Sequence[str] = (),
) -> EvidenceResult:
    """Retrieve evidence for one checklist `item`, with one rule-based retry on a weak score."""
    settings = settings or get_settings()
    collection_name = collection_name or settings.qdrant_collection

    def _search(query: str) -> list[SearchResult]:
        [vector] = embed_texts([query], client=embed_client, settings=settings)
        return search(
            qdrant_client,
            collection_name,
            query_vector=vector,
            domain=Domain.COMPANY_DOCUMENTS,
            source_name=source_name,
            limit=limit,
        )

    def _anchored(results: list[SearchResult]) -> list[SearchResult]:
        """Keep only chunks containing the anchored row, preserving score order."""
        return [r for r in results if chunk_has_anchor(r.payload.chunk_text, row_anchors)]

    def _result(query: str, results: list[SearchResult], *, fallback: bool) -> EvidenceResult:
        matched = _anchored(results) if row_anchors else []
        kept = matched if matched else results
        return EvidenceResult(
            item_code=item.code,
            query=query,
            results=kept,
            top_score=_top_score(kept),
            fallback_used=fallback,
            anchor_matched=bool(matched),
            anchors_requested=bool(row_anchors),
        )

    base_query = build_base_query(item)
    results = _search(base_query)
    first = _result(base_query, results, fallback=False)

    # Retry when the score heuristic says so, or when anchors were requested and the row
    # was not among the candidates: a broadened query may surface the chunk that has it.
    needs_retry = not results or _top_score(results) < score_threshold
    if row_anchors and not first.anchor_matched:
        needs_retry = True

    if needs_retry:
        broadened_query = build_broadened_query(item)
        return _result(broadened_query, _search(broadened_query), fallback=True)

    return first
