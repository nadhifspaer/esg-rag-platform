"""Hybrid retrieval: fuse the dense (embedding) leg with the BM25 keyword leg via Rank Fusion."""

from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI
from qdrant_client import QdrantClient

from app.core.config import Settings, get_settings
from app.models.payload import ChunkPayload, Domain
from app.rag.embedder import embed_texts
from app.rag.keyword_search import KeywordIndex
from app.rag.vector_store import SearchResult, search

# Standard RRF constant (Cormack et al., 2009). Larger k flattens the contribution
# of rank differences; 60 is the widely used default.
_RRF_K = 60


@dataclass(frozen=True)
class FusedResult:
    """One fused hit: the RRF score (not a similarity) and each leg's 1-based rank, if present."""

    id: str
    score: float
    payload: ChunkPayload
    dense_rank: int | None
    keyword_rank: int | None


def rrf_fuse(
    dense: list[SearchResult],
    keyword: list[SearchResult],
    *,
    k: int = _RRF_K,
    limit: int = 10,
) -> list[FusedResult]:
    """Merge two ranked result lists into one via Reciprocal Rank Fusion."""

    @dataclass
    class _Acc:
        payload: ChunkPayload
        dense_rank: int | None = None
        keyword_rank: int | None = None

    accumulated: dict[str, _Acc] = {}
    for rank, result in enumerate(dense, start=1):
        accumulated.setdefault(result.id, _Acc(payload=result.payload)).dense_rank = rank
    for rank, result in enumerate(keyword, start=1):
        # setdefault keeps the payload already recorded from the dense leg, if any.
        accumulated.setdefault(result.id, _Acc(payload=result.payload)).keyword_rank = rank

    fused = [
        FusedResult(
            id=point_id,
            score=(
                (1.0 / (k + acc.dense_rank) if acc.dense_rank is not None else 0.0)
                + (1.0 / (k + acc.keyword_rank) if acc.keyword_rank is not None else 0.0)
            ),
            payload=acc.payload,
            dense_rank=acc.dense_rank,
            keyword_rank=acc.keyword_rank,
        )
        for point_id, acc in accumulated.items()
    ]
    fused.sort(key=lambda f: f.score, reverse=True)
    return fused[:limit]


@dataclass(frozen=True)
class HybridSearchResult:
    """Every stage of hybrid search's output: dense/keyword's full legs, plus the fused pool."""

    dense: list[SearchResult]
    keyword: list[SearchResult]
    fused: list[FusedResult]


def hybrid_search_with_legs(
    query: str,
    *,
    domain: Domain,
    qdrant_client: QdrantClient,
    keyword_index: KeywordIndex,
    source_name: str | None = None,
    collection_name: str | None = None,
    embed_client: OpenAI | None = None,
    settings: Settings | None = None,
    dense_limit: int = 20,
    keyword_limit: int = 20,
    limit: int = 10,
) -> HybridSearchResult:
    """Run dense + BM25 search for `query` within `domain`, RRF-fuse, and return every stage."""
    settings = settings or get_settings()
    collection_name = collection_name or settings.qdrant_collection

    [query_vector] = embed_texts([query], client=embed_client, settings=settings)
    dense = search(
        qdrant_client,
        collection_name,
        query_vector=query_vector,
        domain=domain,
        source_name=source_name,
        limit=dense_limit,
    )
    keyword = keyword_index.search(query, domain, source_name=source_name, limit=keyword_limit)
    fused = rrf_fuse(dense, keyword, limit=limit)
    return HybridSearchResult(dense=dense, keyword=keyword, fused=fused)


def hybrid_search(
    query: str,
    *,
    domain: Domain,
    qdrant_client: QdrantClient,
    keyword_index: KeywordIndex,
    source_name: str | None = None,
    collection_name: str | None = None,
    embed_client: OpenAI | None = None,
    settings: Settings | None = None,
    dense_limit: int = 20,
    keyword_limit: int = 20,
    limit: int = 10,
) -> list[FusedResult]:
    """Run dense + BM25 search for `query` within `domain` and RRF-fuse (returns `.fused` only)."""
    return hybrid_search_with_legs(
        query,
        domain=domain,
        qdrant_client=qdrant_client,
        keyword_index=keyword_index,
        source_name=source_name,
        collection_name=collection_name,
        embed_client=embed_client,
        settings=settings,
        dense_limit=dense_limit,
        keyword_limit=keyword_limit,
        limit=limit,
    ).fused
