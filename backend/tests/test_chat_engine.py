"""Tests for `ChatEngine.make_retrieve`'s per-domain scoping and chat-path row-anchor pinning."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from qdrant_client import QdrantClient

import app.rag.chat_engine as chat_engine_module
from app.core.config import Settings
from app.models.payload import ChunkPayload, ContentType, Domain
from app.rag.chat_engine import ChatEngine
from app.rag.embedder import EMBEDDING_DIM
from app.rag.keyword_search import KeywordIndex
from app.rag.vector_store import ensure_collection, point_id, upsert_points

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect in the local Qdrant")

COLL = "test_esg_documents"
SOURCE_NAME = "PT Bank Central Asia Tbk — 2025 Sustainability Report"
OTHER_SOURCE_NAME = "PT Bank Rakyat Indonesia (Persero) Tbk — 2025 Sustainability Report"
STANDARD_SOURCE_NAME = "ESG Disclosure Guide Book"
SCOPE_1_TEXT = "| Total Direct Emissions (Scope 1) | 6.032 |"


class _FakeEmbedder:
    """Same fake as test_hybrid_search.py's: a fixed vector regardless of input text,
    so the in-memory Qdrant dense leg is deterministic without a real API call."""

    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    class _Embeddings:
        def __init__(self, vector: list[float]) -> None:
            self._vector = vector

        def create(self, model: str, input: list[str]):  # noqa: A002
            from types import SimpleNamespace

            data = [SimpleNamespace(index=i, embedding=self._vector) for i, _ in enumerate(input)]
            return SimpleNamespace(data=data)

    @property
    def embeddings(self) -> _Embeddings:
        return self._Embeddings(self._vector)


class _FixedScorer:
    """Fake cross-encoder: score looked up by exact text (Scope 1 chunk deliberately lowest)."""

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        return [self._scores.get(doc, 0.0) for _query, doc in pairs]


def _payload(
    text: str,
    *,
    page: int = 1,
    domain: Domain = Domain.COMPANY_DOCUMENTS,
    source_name: str = SOURCE_NAME,
    document_type: str = "sustainability_report",
) -> ChunkPayload:
    return ChunkPayload(
        domain=domain,
        source_name=source_name,
        document_type=document_type,
        page_number=page,
        content_type=ContentType.TABLE,
        chunk_text=text,
    )


def _dir_vector(dim_index: int) -> list[float]:
    v = [0.01] * EMBEDDING_DIM
    v[dim_index] = 1.0
    return v


def _build_corpus() -> tuple[QdrantClient, KeywordIndex]:
    """One scoped bank's report: the real Scope 1 row plus five higher-scoring noise
    chunks, all under the same source_name, all findable by the dense leg."""
    client = QdrantClient(":memory:")
    ensure_collection(client, COLL)

    noise_texts = [f"General corporate governance narrative, paragraph {i}." for i in range(5)]
    texts = [SCOPE_1_TEXT, *noise_texts]
    payloads = [_payload(text, page=i + 1) for i, text in enumerate(texts)]
    ids = [point_id(SOURCE_NAME, ContentType.TABLE.value, i + 1, 0) for i in range(len(texts))]
    upsert_points(client, COLL, ids=ids, vectors=[_dir_vector(0)] * len(texts), payloads=payloads)

    keyword_index = KeywordIndex(list(zip(ids, payloads, strict=True)))
    return client, keyword_index


def _engine(scores: dict[str, float], *, top_n: int) -> ChatEngine:
    client, keyword_index = _build_corpus()
    return ChatEngine(
        settings=Settings(reranker_top_n=top_n, qdrant_collection=COLL),
        qdrant_client=client,
        keyword_index=keyword_index,
        scorer=_FixedScorer(scores),
        openai_client=_FakeEmbedder(_dir_vector(0)),  # type: ignore[arg-type]
    )


def _build_two_domain_corpus(
    *, standards_text: str = "The ESG Disclosure Guide Book defines each metric."
) -> tuple[QdrantClient, KeywordIndex]:
    """Two domains: company_documents holds two banks' reports, standards holds one, same vector."""
    client = QdrantClient(":memory:")
    ensure_collection(client, COLL)

    bca_texts = [SCOPE_1_TEXT, "General corporate governance narrative, paragraph 0."]
    bri_texts = ["PT Bank Rakyat Indonesia governance narrative, paragraph 0."]
    standards_texts = [standards_text]

    payloads = [
        *[_payload(t, page=i + 1, source_name=SOURCE_NAME) for i, t in enumerate(bca_texts)],
        *[_payload(t, page=i + 1, source_name=OTHER_SOURCE_NAME) for i, t in enumerate(bri_texts)],
        *[
            _payload(
                t,
                page=i + 1,
                domain=Domain.STANDARDS,
                source_name=STANDARD_SOURCE_NAME,
                document_type="edgb_standard",
            )
            for i, t in enumerate(standards_texts)
        ],
    ]
    ids = [point_id(p.source_name, p.content_type.value, p.page_number, 0) for p in payloads]
    upsert_points(
        client, COLL, ids=ids, vectors=[_dir_vector(0)] * len(payloads), payloads=payloads
    )

    keyword_index = KeywordIndex(list(zip(ids, payloads, strict=True)))
    return client, keyword_index


def _two_domain_engine(scores: dict[str, float], *, top_n: int, **kwargs) -> ChatEngine:
    client, keyword_index = _build_two_domain_corpus(**kwargs)
    return ChatEngine(
        settings=Settings(reranker_top_n=top_n, qdrant_collection=COLL),
        qdrant_client=client,
        keyword_index=keyword_index,
        scorer=_FixedScorer(scores),
        openai_client=_FakeEmbedder(_dir_vector(0)),  # type: ignore[arg-type]
    )


def _scores_ranking_scope_1_chunk_last() -> dict[str, float]:
    noise_texts = [f"General corporate governance narrative, paragraph {i}." for i in range(5)]
    scores = {text: float(5 - i) for i, text in enumerate(noise_texts)}  # 5.0 .. 1.0
    scores[SCOPE_1_TEXT] = 0.0  # deliberately last
    return scores


def test_pinning_recovers_the_anchored_chunk_the_reranker_excluded() -> None:
    """top_n=1 keeps only the highest-scoring noise chunk; without pinning the Scope 1
    row (scored lowest, deliberately) would never reach generation at all."""
    engine = _engine(_scores_ranking_scope_1_chunk_last(), top_n=1)
    retrieve = engine.make_retrieve(
        (Domain.COMPANY_DOCUMENTS,), source_names={Domain.COMPANY_DOCUMENTS: SOURCE_NAME}
    )

    chunks = retrieve("BCA Scope 1 emissions")

    texts = [c.payload.chunk_text for c in chunks]
    assert SCOPE_1_TEXT in texts, "pinning should have recovered the anchored chunk"
    assert len(chunks) == 2  # top_n=1 kept normally, +1 pinned


def test_no_pinning_when_source_names_is_empty() -> None:
    """Unscoped retrieval never pins, mirrors `scoped_source_names`' own guards; a
    cross-domain or multi-bank query must not silently narrow to one bank's row."""
    engine = _engine(_scores_ranking_scope_1_chunk_last(), top_n=1)
    retrieve = engine.make_retrieve((Domain.COMPANY_DOCUMENTS,))  # no source_names at all

    chunks = retrieve("BCA Scope 1 emissions")

    texts = [c.payload.chunk_text for c in chunks]
    assert SCOPE_1_TEXT not in texts
    assert len(chunks) == 1  # top_n=1, no pinning applied


def test_no_pinning_when_the_query_matches_no_curated_indicator() -> None:
    """A scoped query that doesn't name a curated indicator is unaffected: pinning is
    additive, never a general-purpose retrieval booster."""
    engine = _engine(_scores_ranking_scope_1_chunk_last(), top_n=1)
    retrieve = engine.make_retrieve(
        (Domain.COMPANY_DOCUMENTS,), source_names={Domain.COMPANY_DOCUMENTS: SOURCE_NAME}
    )

    chunks = retrieve("what is BCA's total assets")

    assert len(chunks) == 1  # no trigger fired, nothing pinned


# --- per-domain scoping (the Case 2 cross-domain fix) -----------------------


def test_make_retrieve_scopes_each_domain_to_its_own_resolved_source_name() -> None:
    """`source_names` is per-domain: scoping company_documents must not affect standards' search."""
    engine = _two_domain_engine({}, top_n=10)  # empty scores -> every candidate scores 0.0
    retrieve = engine.make_retrieve(
        (Domain.COMPANY_DOCUMENTS, Domain.STANDARDS),
        source_names={Domain.COMPANY_DOCUMENTS: SOURCE_NAME},  # standards left unscoped
    )

    chunks = retrieve("BCA Scope 1 emissions")

    sources = {c.payload.source_name for c in chunks}
    assert OTHER_SOURCE_NAME not in sources, (
        "the company_documents leg must be confined to the named bank, not the whole domain"
    )
    assert SOURCE_NAME in sources
    assert STANDARD_SOURCE_NAME in sources, (
        "the standards leg has no source_names entry, so it must still search domain-wide"
    )


def test_pinning_only_ever_pins_from_the_domain_that_is_actually_scoped() -> None:
    """Regression test: pinning must run per scoped domain against only that domain's own legs."""
    # Structured like the real company-side table so it would get pinned if the fix regressed.
    decoy_text = "| Total Direct Emissions (Scope 1) | see GRI 305-1 definition |"
    engine = _two_domain_engine(
        _scores_ranking_scope_1_chunk_last(), top_n=1, standards_text=decoy_text
    )
    retrieve = engine.make_retrieve(
        (Domain.STANDARDS, Domain.COMPANY_DOCUMENTS),
        source_names={Domain.COMPANY_DOCUMENTS: SOURCE_NAME},  # standards stays unscoped
    )

    chunks = retrieve("BCA Scope 1 emissions")

    texts = [c.payload.chunk_text for c in chunks]
    assert SCOPE_1_TEXT in texts, "the correctly-scoped company_documents chunk should be pinned"
    assert decoy_text not in texts, (
        "the standards decoy must never be pinned — that domain was never scoped"
    )


# --- scoped-cross-domain retrieval-limit widening ----------------------------


CASE2_QUERY = "How does BCA's Scope 1 disclosure compare with what GRI 305-1 requires?"


def _build_wide_scoped_pool_corpus(*, noise_count: int) -> tuple[QdrantClient, KeywordIndex]:
    """BCA's report with `noise_count` chunks outranking the Scope 1 row, plus a standards chunk."""
    client = QdrantClient(":memory:")
    ensure_collection(client, COLL)

    noise_texts = [
        f"BCA Scope 1 disclosure comparison with GRI 305-1 requirements, paragraph {i}."
        for i in range(noise_count)
    ]
    bca_payloads = [_payload(SCOPE_1_TEXT, page=1)] + [
        _payload(t, page=i + 2) for i, t in enumerate(noise_texts)
    ]
    bca_vectors = [_dir_vector(1)] + [_dir_vector(0)] * len(noise_texts)
    bca_ids = [
        point_id(SOURCE_NAME, ContentType.TABLE.value, i + 1, 0) for i in range(len(bca_payloads))
    ]

    standards_payload = _payload(
        "The ESG Disclosure Guide Book defines each metric.",
        page=1,
        domain=Domain.STANDARDS,
        source_name=STANDARD_SOURCE_NAME,
        document_type="edgb_standard",
    )
    standards_id = point_id(STANDARD_SOURCE_NAME, ContentType.TABLE.value, 1, 0)

    payloads = bca_payloads + [standards_payload]
    vectors = bca_vectors + [_dir_vector(0)]
    ids = bca_ids + [standards_id]
    upsert_points(client, COLL, ids=ids, vectors=vectors, payloads=payloads)

    keyword_index = KeywordIndex(list(zip(ids, payloads, strict=True)))
    return client, keyword_index


def _wide_pool_engine(*, noise_count: int, top_n: int = 10) -> ChatEngine:
    client, keyword_index = _build_wide_scoped_pool_corpus(noise_count=noise_count)
    # Every candidate scores 0.0 -- this test is about what reaches the FUSED pool
    # (candidates), not about the joint cross-encoder rerank that happens after it.
    return ChatEngine(
        settings=Settings(reranker_top_n=top_n, qdrant_collection=COLL),
        qdrant_client=client,
        keyword_index=keyword_index,
        scorer=_FixedScorer({}),
        openai_client=_FakeEmbedder(_dir_vector(0)),  # type: ignore[arg-type]
    )


def test_single_domain_scoped_query_keeps_default_retrieval_limits(monkeypatch) -> None:
    """Widening in `make_retrieve` is gated on `len(domains) > 1`; single-domain must not widen."""
    engine = _engine(_scores_ranking_scope_1_chunk_last(), top_n=1)
    calls: list[dict] = []
    real = chat_engine_module.hybrid_search_with_legs

    def _spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(chat_engine_module, "hybrid_search_with_legs", _spy)

    retrieve = engine.make_retrieve(
        (Domain.COMPANY_DOCUMENTS,), source_names={Domain.COMPANY_DOCUMENTS: SOURCE_NAME}
    )
    retrieve("BCA Scope 1 emissions")

    assert len(calls) == 1, "a single-domain query should call hybrid_search_with_legs once"
    kwargs = calls[0]
    assert "dense_limit" not in kwargs
    assert "keyword_limit" not in kwargs
    assert "limit" not in kwargs


def test_cross_domain_scoped_query_widens_retrieval_limits(monkeypatch) -> None:
    """A cross-domain query with its company_documents leg scoped widens limits there only."""
    engine = _wide_pool_engine(noise_count=25)
    calls: dict[Domain, dict] = {}
    real = chat_engine_module.hybrid_search_with_legs

    def _spy(*args, **kwargs):
        calls[kwargs["domain"]] = kwargs
        return real(*args, **kwargs)

    monkeypatch.setattr(chat_engine_module, "hybrid_search_with_legs", _spy)

    retrieve = engine.make_retrieve(
        (Domain.COMPANY_DOCUMENTS, Domain.STANDARDS),
        source_names={Domain.COMPANY_DOCUMENTS: SOURCE_NAME},
    )
    retrieve(CASE2_QUERY)

    company_kwargs = calls[Domain.COMPANY_DOCUMENTS]
    assert company_kwargs["dense_limit"] == chat_engine_module._SCOPED_CROSS_DOMAIN_LIMIT
    assert company_kwargs["keyword_limit"] == chat_engine_module._SCOPED_CROSS_DOMAIN_LIMIT
    assert company_kwargs["limit"] == chat_engine_module._SCOPED_CROSS_DOMAIN_LIMIT

    standards_kwargs = calls[Domain.STANDARDS]
    assert "dense_limit" not in standards_kwargs
    assert "keyword_limit" not in standards_kwargs
    assert "limit" not in standards_kwargs


def test_scoped_low_ranked_chunk_reaches_the_fused_pool_once_widened(monkeypatch) -> None:
    """Reproduces the production gap at small scale: target must reach the pool once widened."""
    engine = _wide_pool_engine(noise_count=25)
    retrieve = engine.make_retrieve(
        (Domain.COMPANY_DOCUMENTS, Domain.STANDARDS),
        source_names={Domain.COMPANY_DOCUMENTS: SOURCE_NAME},
    )

    calls: list = []
    real_rerank = chat_engine_module.rerank

    def _spy_rerank(query, candidates, **kwargs):
        calls.append(list(candidates))
        return real_rerank(query, candidates, **kwargs)

    monkeypatch.setattr(chat_engine_module, "rerank", _spy_rerank)
    retrieve(CASE2_QUERY)

    assert len(calls) == 1
    candidate_texts = [c.payload.chunk_text for c in calls[0]]
    assert SCOPE_1_TEXT in candidate_texts, (
        "the scoped company_documents chunk should reach the fused candidate pool once "
        "dense_limit/keyword_limit/fusion limit are all widened for this cross-domain, "
        "scoped leg"
    )
