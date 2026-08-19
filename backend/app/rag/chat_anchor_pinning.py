"""Chat-path row-anchor pinning: forces a curated indicator's anchored chunk into generation."""

from __future__ import annotations

from collections.abc import Sequence

from app.rag.chat_anchor_triggers import resolve_chat_anchor_triggers
from app.rag.compliance.anchors import chunk_has_anchor
from app.rag.hybrid_search import FusedResult
from app.rag.reranker import RerankedResult
from app.rag.vector_store import SearchResult


def pin_anchored_chunks(
    query: str,
    reranked: list[RerankedResult],
    *,
    raw_dense: Sequence[SearchResult],
    raw_keyword: Sequence[SearchResult],
) -> list[RerankedResult]:
    """Force any curated environmental indicator's anchored chunk into `reranked`."""
    triggers = resolve_chat_anchor_triggers(query)
    if not triggers:
        return list(reranked)

    pinned = list(reranked)
    present_ids = {r.id for r in pinned}

    for trigger in triggers:
        if not trigger.anchors:
            continue  # recognized but unanchored (Scope 3), nothing to pin
        if any(chunk_has_anchor(r.payload.chunk_text, trigger.anchors) for r in pinned):
            continue  # the reranker already kept an anchored chunk, nothing to do

        match = _find_anchor_match(trigger.anchors, raw_dense, raw_keyword, present_ids)
        if match is None:
            continue  # not found in either raw leg either, nothing to pin
        pinned.append(
            RerankedResult(id=match.id, score=float("nan"), payload=match.payload, fused=match)
        )
        present_ids.add(match.id)

    return pinned


def _find_anchor_match(
    anchors: Sequence[str],
    raw_dense: Sequence[SearchResult],
    raw_keyword: Sequence[SearchResult],
    exclude_ids: set[str],
) -> FusedResult | None:
    """The first raw-leg candidate whose text contains one of `anchors`, keyword leg first."""
    for rank, result in enumerate(raw_keyword, start=1):
        if result.id not in exclude_ids and chunk_has_anchor(result.payload.chunk_text, anchors):
            return FusedResult(
                id=result.id,
                score=float("nan"),
                payload=result.payload,
                dense_rank=None,
                keyword_rank=rank,
            )
    for rank, result in enumerate(raw_dense, start=1):
        if result.id not in exclude_ids and chunk_has_anchor(result.payload.chunk_text, anchors):
            return FusedResult(
                id=result.id,
                score=float("nan"),
                payload=result.payload,
                dense_rank=rank,
                keyword_rank=None,
            )
    return None
