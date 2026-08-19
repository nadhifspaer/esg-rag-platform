"""Tests for per-requirement compliance retrieval, using in-memory Qdrant and a fake embedder."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from qdrant_client import QdrantClient

from app.models.payload import ChunkPayload, ContentType, Domain
from app.rag.compliance.checklist import ChecklistItem
from app.rag.compliance.retrieval import (
    build_base_query,
    build_broadened_query,
    retrieve_evidence,
)
from app.rag.embedder import EMBEDDING_DIM
from app.rag.vector_store import ensure_collection, point_id, upsert_points

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect in the local Qdrant")

COLL = "test_compliance"


def _item() -> ChecklistItem:
    return ChecklistItem(
        code="GRI 305-1",
        description="Direct (Scope 1) GHG emissions",
        topic_keywords=["scope 1 emissions", "direct emissions", "tCO2e"],
    )


def _dir_vector(index: int) -> list[float]:
    """A near-one-hot unit-ish direction vector, like the hybrid-search tests use."""
    v = [0.01] * EMBEDDING_DIM
    v[index] = 1.0
    return v


class _FakeEmbedder:
    """Maps each input query string to a preset vector (default vector otherwise),
    matching the OpenAI embeddings.create interface embed_texts expects."""

    def __init__(self, mapping: dict[str, list[float]], default: list[float]) -> None:
        self._mapping = mapping
        self._default = default

    class _Embeddings:
        def __init__(self, outer: _FakeEmbedder) -> None:
            self._outer = outer

        def create(self, model: str, input: list[str]):  # noqa: A002 - matches OpenAI API
            data = [
                SimpleNamespace(
                    index=i, embedding=self._outer._mapping.get(t, self._outer._default)
                )
                for i, t in enumerate(input)
            ]
            return SimpleNamespace(data=data)

    @property
    def embeddings(self) -> _FakeEmbedder._Embeddings:
        return self._Embeddings(self)


def _company_chunk(
    text: str, *, source: str = "PT Bank Central Asia Tbk — 2025 Sustainability Report"
) -> ChunkPayload:
    return ChunkPayload(
        domain=Domain.COMPANY_DOCUMENTS,
        source_name=source,
        document_type="sustainability_report",
        page_number=1,
        content_type=ContentType.TEXT,
        chunk_text=text,
        bank="PT Bank Central Asia Tbk",
        year=2025,
    )


def _store(client: QdrantClient, payloads_vectors: list[tuple[ChunkPayload, list[float]]]) -> None:
    ensure_collection(client, COLL)
    ids = [
        point_id(p.source_name, p.content_type.value, p.page_number, i)
        for i, (p, _) in enumerate(payloads_vectors)
    ]
    upsert_points(
        client,
        COLL,
        ids=ids,
        vectors=[v for _, v in payloads_vectors],
        payloads=[p for p, _ in payloads_vectors],
    )


# --- query building ---------------------------------------------------------


def test_query_builders() -> None:
    item = _item()
    assert build_base_query(item) == "Direct (Scope 1) GHG emissions"
    broadened = build_broadened_query(item)
    assert broadened.startswith("Direct (Scope 1) GHG emissions")
    for kw in item.topic_keywords:
        assert kw in broadened  # keywords appended as synonyms


# --- retrieval + threshold fallback -----------------------------------------


def test_strong_first_hit_no_fallback() -> None:
    item = _item()
    client = QdrantClient(":memory:")
    doc = _company_chunk("Scope 1 emissions were 24,105 tCO2e in 2024.")
    _store(client, [(doc, _dir_vector(0))])

    # Base query embeds aligned with the doc -> cosine ~1.0, above threshold.
    embedder = _FakeEmbedder({build_base_query(item): _dir_vector(0)}, default=_dir_vector(1))

    result = retrieve_evidence(
        item,
        qdrant_client=client,
        collection_name=COLL,
        embed_client=embedder,
        score_threshold=0.5,
        limit=5,
    )

    assert result.fallback_used is False
    assert result.query == build_base_query(item)
    assert result.top_score > 0.5
    assert result.results[0].payload.source_name.startswith("PT Bank Central Asia")


def test_weak_first_hit_triggers_broadened_retry() -> None:
    item = _item()
    client = QdrantClient(":memory:")
    doc = _company_chunk("Scope 1 emissions were 24,105 tCO2e in 2024.")
    _store(client, [(doc, _dir_vector(0))])

    # Base query misaligned (weak cosine ~0.15 < threshold); broadened aligns -> strong.
    embedder = _FakeEmbedder(
        {build_base_query(item): _dir_vector(1), build_broadened_query(item): _dir_vector(0)},
        default=_dir_vector(1),
    )

    result = retrieve_evidence(
        item,
        qdrant_client=client,
        collection_name=COLL,
        embed_client=embedder,
        score_threshold=0.5,
        limit=5,
    )

    assert result.fallback_used is True
    assert result.query == build_broadened_query(item)
    assert result.top_score > 0.5  # the retry found a strong match


def test_empty_first_result_triggers_fallback() -> None:
    item = _item()
    client = QdrantClient(":memory:")
    # Only a STANDARDS doc exists -> the company_documents filter returns nothing.
    standards = ChunkPayload(
        domain=Domain.STANDARDS,
        source_name="GRI 305",
        document_type="gri_standard",
        page_number=1,
        content_type=ContentType.TEXT,
        chunk_text="Scope 1 emissions definition",
    )
    _store(client, [(standards, _dir_vector(0))])

    embedder = _FakeEmbedder({}, default=_dir_vector(0))

    result = retrieve_evidence(
        item,
        qdrant_client=client,
        collection_name=COLL,
        embed_client=embedder,
        score_threshold=0.5,
        limit=5,
    )

    # Fallback fired on empty results; still nothing in company_documents.
    assert result.fallback_used is True
    assert result.results == []
    assert result.top_score == 0.0


def test_threshold_controls_the_trigger() -> None:
    item = _item()
    client = QdrantClient(":memory:")
    doc = _company_chunk("Scope 1 emissions were 24,105 tCO2e in 2024.")
    _store(client, [(doc, _dir_vector(0))])
    # Base query gives a moderate cosine (~0.15) via a misaligned direction.
    embedder = _FakeEmbedder({build_base_query(item): _dir_vector(1)}, default=_dir_vector(0))

    # A low threshold accepts the moderate base hit (no fallback)...
    low = retrieve_evidence(
        item,
        qdrant_client=client,
        collection_name=COLL,
        embed_client=embedder,
        score_threshold=0.1,
        limit=5,
    )
    assert low.fallback_used is False
    # ...a high threshold rejects it and triggers the retry.
    high = retrieve_evidence(
        item,
        qdrant_client=client,
        collection_name=COLL,
        embed_client=embedder,
        score_threshold=0.9,
        limit=5,
    )
    assert high.fallback_used is True


def test_evidence_is_scoped_to_company_documents() -> None:
    item = _item()
    client = QdrantClient(":memory:")
    company = _company_chunk("Scope 1 emissions were 24,105 tCO2e.")
    standards = ChunkPayload(
        domain=Domain.STANDARDS,
        source_name="GRI 305",
        document_type="gri_standard",
        page_number=1,
        content_type=ContentType.TEXT,
        chunk_text="Scope 1 definition",
    )
    _store(client, [(company, _dir_vector(0)), (standards, _dir_vector(0))])
    embedder = _FakeEmbedder({}, default=_dir_vector(0))

    result = retrieve_evidence(
        item,
        qdrant_client=client,
        collection_name=COLL,
        embed_client=embedder,
        score_threshold=0.5,
        limit=5,
    )

    assert result.results, "expected a company_documents hit"
    assert all(r.payload.domain is Domain.COMPANY_DOCUMENTS for r in result.results)


def test_source_name_scopes_to_one_report_even_when_another_scores_higher() -> None:
    item = _item()
    client = QdrantClient(":memory:")
    target = "PT Bank BCA Tbk — 2025 Sustainability Report"
    other = "PT Bank Mandiri (Persero) Tbk — 2025 Sustainability Report"
    # The OTHER bank's chunk is a perfect match (dir 0); the TARGET bank's chunk is a
    # weaker match (dir 1). Without scoping, the other bank would top the results.
    _store(
        client,
        [
            (_company_chunk("strong match", source=other), _dir_vector(0)),
            (_company_chunk("weaker match", source=target), _dir_vector(1)),
        ],
    )
    embedder = _FakeEmbedder({}, default=_dir_vector(0))  # query aligns with the OTHER bank

    result = retrieve_evidence(
        item,
        qdrant_client=client,
        source_name=target,
        collection_name=COLL,
        embed_client=embedder,
        score_threshold=0.0,  # accept whatever the scoped search returns
        limit=5,
    )

    assert result.results, "expected the target report's chunk"
    # Only the target report's chunks come back, despite the other bank scoring higher.
    assert all(r.payload.source_name == target for r in result.results)
    assert all(r.payload.source_name != other for r in result.results)
