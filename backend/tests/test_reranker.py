"""Tests for the cross-encoder re-ranking stage."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pytest

from app.core.config import Settings
from app.models.payload import ChunkPayload, ContentType, Domain
from app.rag.hybrid_search import FusedResult
from app.rag.reranker import (
    CrossEncoderReranker,
    OnnxReranker,
    RerankedResult,
    build_scorer,
    rerank,
)


def _payload(source_name: str, text: str) -> ChunkPayload:
    return ChunkPayload(
        domain=Domain.STANDARDS,
        source_name=source_name,
        document_type="gri_standard",
        page_number=1,
        content_type=ContentType.TEXT,
        chunk_text=text,
    )


def _fused(point_id: str, text: str, *, source_name: str | None = None) -> FusedResult:
    return FusedResult(
        id=point_id,
        score=0.03,
        payload=_payload(source_name or point_id, text),
        dense_rank=1,
        keyword_rank=1,
    )


class _KeywordScorer:
    """Fake cross-encoder: scores a pair by how many times a marker word from the
    query appears in the document. Deterministic, no model needed."""

    def __init__(self, marker: str) -> None:
        self._marker = marker

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        return [float(doc.lower().count(self._marker)) for _query, doc in pairs]


def test_rerank_reorders_by_scorer_not_input_order() -> None:
    # Input order puts the weak candidate first; the scorer should float the
    # strong one (more markers) to the top.
    candidates = [
        _fused("weak", "mentions target once", source_name="Weak"),
        _fused("strong", "target target target", source_name="Strong"),
    ]

    result = rerank("target", candidates, scorer=_KeywordScorer("target"))

    assert [r.payload.source_name for r in result] == ["Strong", "Weak"]
    assert result[0].score == 3.0


def test_rerank_respects_top_n() -> None:
    candidates = [
        _fused("a", "target"),
        _fused("b", "target target"),
        _fused("c", "target target target"),
    ]

    result = rerank("target", candidates, scorer=_KeywordScorer("target"), top_n=2)

    assert len(result) == 2
    assert [r.id for r in result] == ["c", "b"]  # top 2 by score


def test_rerank_preserves_fused_provenance() -> None:
    candidates = [_fused("x", "target", source_name="Doc X")]

    (result,) = rerank("target", candidates, scorer=_KeywordScorer("target"))

    assert isinstance(result, RerankedResult)
    # The originating FusedResult travels with the reranked result.
    assert result.fused.dense_rank == 1
    assert result.fused.keyword_rank == 1
    assert result.payload.source_name == "Doc X"


def test_rerank_is_stable_for_equal_scores() -> None:
    # All three score 0 (no marker); a stable sort must keep input order.
    candidates = [_fused("first", "none"), _fused("second", "none"), _fused("third", "none")]

    result = rerank("target", candidates, scorer=_KeywordScorer("target"))

    assert [r.id for r in result] == ["first", "second", "third"]


def test_rerank_empty_candidates_returns_empty() -> None:
    assert rerank("target", [], scorer=_KeywordScorer("target")) == []


def test_build_scorer_defaults_to_onnx() -> None:
    # An unconfigured/default Settings must select OnnxReranker, the production default.
    assert isinstance(build_scorer(Settings()), OnnxReranker)


def test_build_scorer_selects_local_when_configured() -> None:
    # "local" remains a fully supported explicit fallback even though it is no longer
    # the default; this pins that opting back into it still works.
    assert isinstance(build_scorer(Settings(reranker_provider="local")), CrossEncoderReranker)


def test_build_scorer_selects_onnx_when_configured() -> None:
    # Pins that setting reranker_provider="onnx" explicitly still works, not just the default.
    assert isinstance(build_scorer(Settings(reranker_provider="onnx")), OnnxReranker)


# --- OnnxReranker: torch-free, the production default as of 2026-07-31 ---


@dataclass
class _FakeOnnxEncoding:
    ids: list[int]
    attention_mask: list[int]


class _FakeOnnxTokenizer:
    """Fake for tokenizers.Tokenizer: encode_batch assigns pair `i` the id `i + 1`."""

    def encode_batch(self, pairs: list[list[str]]) -> list[_FakeOnnxEncoding]:
        return [
            _FakeOnnxEncoding(ids=[i + 1, i + 1], attention_mask=[1, 1]) for i in range(len(pairs))
        ]


@dataclass
class _FakeOnnxIoSpec:
    name: str


class _FakeOnnxSession:
    """Fake for onnxruntime.InferenceSession: `run` returns each row's first `input_ids` value."""

    def __init__(self, input_names: tuple[str, ...] = ("input_ids", "attention_mask")) -> None:
        self._input_names = input_names
        self.last_feed: dict[str, np.ndarray] | None = None

    def get_inputs(self) -> list[_FakeOnnxIoSpec]:
        return [_FakeOnnxIoSpec(name=n) for n in self._input_names]

    def get_outputs(self) -> list[_FakeOnnxIoSpec]:
        return [_FakeOnnxIoSpec(name="logits")]

    def run(self, output_names: list[str], feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.last_feed = feed
        scores = feed["input_ids"][:, 0].astype(float)
        return [scores.reshape(-1, 1)]


def test_onnx_reranker_scores_via_injected_session_and_tokenizer() -> None:
    session = _FakeOnnxSession()
    scorer = OnnxReranker(session=session, tokenizer=_FakeOnnxTokenizer())

    scores = scorer.score([("q1", "d1"), ("q2", "d2"), ("q3", "d3")])

    assert scores == [1.0, 2.0, 3.0]
    assert set(session.last_feed) == {"input_ids", "attention_mask"}


def test_onnx_reranker_empty_pairs_returns_empty_without_calling_anything() -> None:
    session = _FakeOnnxSession()
    scorer = OnnxReranker(session=session, tokenizer=_FakeOnnxTokenizer())

    assert scorer.score([]) == []
    assert session.last_feed is None


def test_onnx_reranker_feeds_zero_token_type_ids_when_graph_declares_it() -> None:
    # Some exports declare token_type_ids; zeros must be fed rather than the input omitted.
    session = _FakeOnnxSession(input_names=("input_ids", "attention_mask", "token_type_ids"))
    scorer = OnnxReranker(session=session, tokenizer=_FakeOnnxTokenizer())

    scorer.score([("q", "d")])

    assert "token_type_ids" in session.last_feed
    assert (session.last_feed["token_type_ids"] == 0).all()


def test_onnx_reranker_rejects_unexpected_graph_input() -> None:
    session = _FakeOnnxSession(input_names=("input_ids", "attention_mask", "something_else"))
    scorer = OnnxReranker(session=session, tokenizer=_FakeOnnxTokenizer())

    with pytest.raises(ValueError, match="something_else"):
        scorer.score([("q", "d")])
