"""Tests for chat-path row-anchor pinning (Phase 2); deterministic fakes throughout, no network."""

from __future__ import annotations

import math

from app.models.payload import ChunkPayload, ContentType, Domain
from app.rag.chat_anchor_pinning import pin_anchored_chunks
from app.rag.hybrid_search import FusedResult
from app.rag.reranker import RerankedResult
from app.rag.vector_store import SearchResult

_SCOPE_1_QUERY = "BCA Scope 1 emissions"
_SCOPE_1_TEXT = "| Total Direct Emissions (Scope 1) | 6.032 |"
_E05_QUERY = "how much waste did BCA generate"
_E05_TEXT = "| E-05 | Waste Generation | Total waste generated (ton) | 0 |"
_NOISE_TEXT = "General corporate governance policy narrative, no anchored row here."


def _payload(text: str, *, page: int = 1) -> ChunkPayload:
    return ChunkPayload(
        domain=Domain.COMPANY_DOCUMENTS,
        source_name="PT Bank Central Asia Tbk — 2025 Sustainability Report",
        document_type="sustainability_report",
        page_number=page,
        content_type=ContentType.TABLE,
        chunk_text=text,
    )


def _reranked(point_id: str, text: str, *, score: float = 1.0) -> RerankedResult:
    payload = _payload(text)
    fused = FusedResult(id=point_id, score=0.5, payload=payload, dense_rank=1, keyword_rank=1)
    return RerankedResult(id=point_id, score=score, payload=payload, fused=fused)


def _search_result(point_id: str, text: str, *, score: float = 0.4) -> SearchResult:
    return SearchResult(id=point_id, score=score, payload=_payload(text))


# --- the pool-absence case (E-05 waste): reranked has nothing anchored, raw legs do ------


def test_pins_a_chunk_absent_from_reranked_but_present_in_the_dense_leg() -> None:
    reranked = [_reranked("noise-1", _NOISE_TEXT)]
    raw_dense = [_search_result("noise-1", _NOISE_TEXT), _search_result("e05-chunk", _E05_TEXT)]

    result = pin_anchored_chunks(_E05_QUERY, reranked, raw_dense=raw_dense, raw_keyword=[])

    assert [r.id for r in result] == ["noise-1", "e05-chunk"]
    pinned = result[-1]
    assert math.isnan(pinned.score)
    assert pinned.fused.dense_rank == 2
    assert pinned.fused.keyword_rank is None


def test_falls_back_to_the_keyword_leg_when_dense_has_no_match() -> None:
    reranked: list[RerankedResult] = []
    raw_keyword = [_search_result("e05-chunk", _E05_TEXT)]

    result = pin_anchored_chunks(_E05_QUERY, reranked, raw_dense=[], raw_keyword=raw_keyword)

    assert [r.id for r in result] == ["e05-chunk"]
    assert result[0].fused.dense_rank is None
    assert result[0].fused.keyword_rank == 1


# --- the presence-but-wrong-winner case (BCA emissions intensity's own shape): already kept ---


def test_does_not_duplicate_when_the_reranker_already_kept_the_anchored_chunk() -> None:
    reranked = [_reranked("scope1-chunk", _SCOPE_1_TEXT)]
    raw_dense = [_search_result("scope1-chunk", _SCOPE_1_TEXT)]

    result = pin_anchored_chunks(_SCOPE_1_QUERY, reranked, raw_dense=raw_dense, raw_keyword=[])

    assert [r.id for r in result] == ["scope1-chunk"]  # not appended a second time


# --- negative cases -----------------------------------------------------------------------


def test_no_match_is_a_no_op() -> None:
    reranked = [_reranked("noise-1", _NOISE_TEXT)]
    result = pin_anchored_chunks(
        "what is BCA's total assets", reranked, raw_dense=[], raw_keyword=[]
    )
    assert [r.id for r in result] == ["noise-1"]


def test_scope_3_never_pins_even_though_it_is_recognized() -> None:
    """GRI 305-3 is curated unanchored: recognized, but there is nothing to pin."""
    reranked: list[RerankedResult] = []
    raw_dense = [_search_result("some-chunk", "Total GHG Emissions (Scope 1, 2 and 3) | 19.170 |")]

    result = pin_anchored_chunks(
        "what does BCA report for scope 3 emissions", reranked, raw_dense=raw_dense, raw_keyword=[]
    )

    assert result == []


def test_anchor_absent_from_both_raw_legs_is_a_no_op() -> None:
    reranked = [_reranked("noise-1", _NOISE_TEXT)]
    raw_dense = [_search_result("noise-1", _NOISE_TEXT)]
    result = pin_anchored_chunks(_SCOPE_1_QUERY, reranked, raw_dense=raw_dense, raw_keyword=[])
    assert [r.id for r in result] == ["noise-1"]


# --- multiple indicators in one query -----------------------------------------------------


def test_pins_two_different_indicators_independently() -> None:
    scope_2_text = "| Total Indirect Emissions (Scope 2) | 18.862 |"
    reranked: list[RerankedResult] = []
    raw_dense = [
        _search_result("scope1-chunk", _SCOPE_1_TEXT),
        _search_result("scope2-chunk", scope_2_text),
    ]

    result = pin_anchored_chunks(
        "BCA scope 1 and scope 2 emissions", reranked, raw_dense=raw_dense, raw_keyword=[]
    )

    assert {r.id for r in result} == {"scope1-chunk", "scope2-chunk"}
