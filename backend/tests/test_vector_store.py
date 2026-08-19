"""Tests for the Qdrant vector store."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue

import app.rag.vector_store as vector_store
from app.models.payload import ChunkPayload, ContentType, Domain
from app.rag.embedder import EMBEDDING_DIM
from app.rag.vector_store import (
    _UPSERT_BATCH,
    _UPSERT_MAX_ATTEMPTS,
    SearchResult,
    delete_by_source,
    ensure_collection,
    point_id,
    search,
    upsert_points,
)

# In-memory Qdrant warns that payload indexes are no-ops locally; that is expected
# and irrelevant to these tests.
pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect in the local Qdrant")

COLL = "test_esg_documents"


# --- helpers ---------------------------------------------------------------


def _payload(
    source_name: str,
    *,
    domain: Domain = Domain.COMPANY_DOCUMENTS,
    page: int = 1,
) -> ChunkPayload:
    return ChunkPayload(
        domain=domain,
        source_name=source_name,
        document_type="sustainability_report",
        page_number=page,
        content_type=ContentType.TEXT,
        chunk_text=f"{source_name} chunk on page {page}",
    )


def _upsert_doc(
    client: QdrantClient,
    collection: str,
    source_name: str,
    count: int,
    *,
    domain: Domain = Domain.COMPANY_DOCUMENTS,
) -> int:
    """Upsert `count` distinct text chunks for one document."""
    payloads = [_payload(source_name, domain=domain, page=i + 1) for i in range(count)]
    ids = [
        point_id(source_name, ContentType.TEXT.value, p.page_number, i)
        for i, p in enumerate(payloads)
    ]
    vectors = [[0.1 + 0.01 * i] * EMBEDDING_DIM for i in range(count)]
    return upsert_points(client, collection, ids=ids, vectors=vectors, payloads=payloads)


def _count_source(client: QdrantClient, collection: str, source_name: str) -> int:
    return client.count(
        collection,
        count_filter=Filter(
            must=[FieldCondition(key="source_name", match=MatchValue(value=source_name))]
        ),
    ).count


# --- fake-client unit tests: index creation & backfill ----------------------


class _FakeQdrant:
    """Records collection/index calls so ensure_collection's index logic can be
    asserted without a server (in-memory Qdrant can't, as it ignores indexes)."""

    def __init__(self, *, exists: bool = False, payload_schema: dict | None = None) -> None:
        self._exists = exists
        self.payload_schema = dict(payload_schema or {})
        self.created_collection = False
        self.vectors_config = None
        self.created_indexes: list[str] = []

    def collection_exists(self, collection_name: str) -> bool:
        return self._exists

    def create_collection(self, collection_name: str, vectors_config) -> None:
        self.created_collection = True
        self._exists = True
        self.vectors_config = vectors_config

    def get_collection(self, collection_name: str):
        return SimpleNamespace(payload_schema=self.payload_schema)

    def create_payload_index(self, collection_name: str, field_name: str, field_schema) -> None:
        self.created_indexes.append(field_name)
        self.payload_schema[field_name] = field_schema


def test_fresh_collection_created_with_both_indexes_and_vector_params() -> None:
    fake = _FakeQdrant(exists=False)
    created = ensure_collection(fake, "c")
    assert created is True
    assert fake.created_collection is True
    assert fake.created_indexes == ["domain", "source_name"]
    assert fake.vectors_config.size == 1536
    assert fake.vectors_config.distance == Distance.COSINE


def test_existing_collection_backfills_only_missing_source_name_index() -> None:
    fake = _FakeQdrant(exists=True, payload_schema={"domain": "keyword"})
    created = ensure_collection(fake, "c")
    assert created is False
    assert fake.created_collection is False
    assert fake.created_indexes == ["source_name"]  # domain already indexed, skipped


def test_existing_collection_with_both_indexes_creates_nothing() -> None:
    fake = _FakeQdrant(exists=True, payload_schema={"domain": "keyword", "source_name": "keyword"})
    created = ensure_collection(fake, "c")
    assert created is False
    assert fake.created_indexes == []


# --- pure logic -------------------------------------------------------------


def test_point_id_is_deterministic_and_distinct() -> None:
    base = point_id("Doc A", "text", 1, 0)
    assert base == point_id("Doc A", "text", 1, 0)  # deterministic
    assert base != point_id("Doc A", "text", 1, 1)  # different chunk
    assert base != point_id("Doc A", "table", 1, 0)  # different content type
    assert base != point_id("Doc B", "text", 1, 0)  # different document


# --- in-memory integration --------------------------------------------------


def test_ensure_collection_creates_then_reports_existing() -> None:
    client = QdrantClient(":memory:")
    assert ensure_collection(client, COLL) is True
    assert ensure_collection(client, COLL) is False


def test_created_collection_has_1536_dim_cosine() -> None:
    client = QdrantClient(":memory:")
    ensure_collection(client, COLL)
    vectors = client.get_collection(COLL).config.params.vectors
    assert vectors.size == 1536
    assert vectors.distance == Distance.COSINE


def test_upsert_points_persists_and_returns_count() -> None:
    client = QdrantClient(":memory:")
    ensure_collection(client, COLL)
    assert _upsert_doc(client, COLL, "Doc A", 3) == 3
    assert client.count(COLL).count == 3


def test_upsert_empty_returns_zero_without_error() -> None:
    client = QdrantClient(":memory:")
    ensure_collection(client, COLL)
    assert upsert_points(client, COLL, ids=[], vectors=[], payloads=[]) == 0


def test_delete_by_source_removes_only_that_document() -> None:
    client = QdrantClient(":memory:")
    ensure_collection(client, COLL)
    _upsert_doc(client, COLL, "Doc A", 3)
    _upsert_doc(client, COLL, "Doc B", 1)

    delete_by_source(client, COLL, "Doc A")

    assert _count_source(client, COLL, "Doc A") == 0
    assert _count_source(client, COLL, "Doc B") == 1
    assert client.count(COLL).count == 1


# --- in-memory integration: domain-filtered search --------------------------
# In-memory Qdrant applies payload filters (by scan) even though it ignores indexes.


def _dir_vector(dim_index: int, dim: int = EMBEDDING_DIM) -> list[float]:
    """A vector pointing mostly along `dim_index` (1.0 there, 0.01 elsewhere, never all-zero)."""
    v = [0.01] * dim
    v[dim_index] = 1.0
    return v


def _upsert_one(
    client: QdrantClient,
    collection: str,
    source_name: str,
    *,
    domain: Domain,
    vector: list[float],
    page: int = 1,
) -> None:
    """Upsert a single chunk for one document with an explicit vector and domain."""
    payload = _payload(source_name, domain=domain, page=page)
    pid = point_id(source_name, ContentType.TEXT.value, page, 0)
    upsert_points(client, collection, ids=[pid], vectors=[vector], payloads=[payload])


def test_search_returns_only_the_requested_domain() -> None:
    client = QdrantClient(":memory:")
    ensure_collection(client, COLL)
    _upsert_one(client, COLL, "Company A", domain=Domain.COMPANY_DOCUMENTS, vector=_dir_vector(0))
    _upsert_one(client, COLL, "Standard X", domain=Domain.STANDARDS, vector=_dir_vector(0))

    results = search(client, COLL, query_vector=_dir_vector(0), domain=Domain.COMPANY_DOCUMENTS)

    assert results, "expected at least one company hit"
    assert all(r.payload.domain is Domain.COMPANY_DOCUMENTS for r in results)
    assert all(r.payload.source_name != "Standard X" for r in results)


def test_search_orders_by_similarity_most_similar_first() -> None:
    client = QdrantClient(":memory:")
    ensure_collection(client, COLL)
    _upsert_one(client, COLL, "Doc dim0", domain=Domain.COMPANY_DOCUMENTS, vector=_dir_vector(0))
    _upsert_one(client, COLL, "Doc dim1", domain=Domain.COMPANY_DOCUMENTS, vector=_dir_vector(1))
    _upsert_one(client, COLL, "Doc dim2", domain=Domain.COMPANY_DOCUMENTS, vector=_dir_vector(2))

    # Query aligned with dim1 -> "Doc dim1" must rank first.
    results = search(client, COLL, query_vector=_dir_vector(1), domain=Domain.COMPANY_DOCUMENTS)

    assert [r.payload.source_name for r in results][0] == "Doc dim1"
    # Scores are sorted descending.
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_respects_limit() -> None:
    client = QdrantClient(":memory:")
    ensure_collection(client, COLL)
    for i in range(5):
        _upsert_one(
            client, COLL, f"Doc {i}", domain=Domain.COMPANY_DOCUMENTS, vector=_dir_vector(i)
        )

    results = search(
        client, COLL, query_vector=_dir_vector(0), domain=Domain.COMPANY_DOCUMENTS, limit=2
    )

    assert len(results) == 2


def test_search_returns_typed_scored_results() -> None:
    client = QdrantClient(":memory:")
    ensure_collection(client, COLL)
    _upsert_one(client, COLL, "Company A", domain=Domain.COMPANY_DOCUMENTS, vector=_dir_vector(0))

    (result,) = search(client, COLL, query_vector=_dir_vector(0), domain=Domain.COMPANY_DOCUMENTS)

    assert isinstance(result, SearchResult)
    assert isinstance(result.score, float)
    assert isinstance(result.payload, ChunkPayload)
    assert result.payload.source_name == "Company A"
    assert isinstance(result.id, str)


def test_search_score_threshold_drops_low_scoring_hits() -> None:
    client = QdrantClient(":memory:")
    ensure_collection(client, COLL)
    # Aligned doc scores ~1.0; the orthogonal-ish doc scores much lower.
    _upsert_one(client, COLL, "Aligned", domain=Domain.COMPANY_DOCUMENTS, vector=_dir_vector(0))
    _upsert_one(client, COLL, "Different", domain=Domain.COMPANY_DOCUMENTS, vector=_dir_vector(1))

    results = search(
        client,
        COLL,
        query_vector=_dir_vector(0),
        domain=Domain.COMPANY_DOCUMENTS,
        score_threshold=0.9,
    )

    assert [r.payload.source_name for r in results] == ["Aligned"]


def test_search_empty_collection_returns_empty_list() -> None:
    client = QdrantClient(":memory:")
    ensure_collection(client, COLL)

    results = search(client, COLL, query_vector=_dir_vector(0), domain=Domain.COMPANY_DOCUMENTS)

    assert results == []


# --- unit: upsert retry & batching (the pilot-run WriteTimeout fix) ----------


class _FlakyUpsertClient:
    """Fake Qdrant client for upsert only: raises a transient error on the first
    `fail_times` calls, then records the points of each successful call."""

    def __init__(self, fail_times: int = 0) -> None:
        self.fail_times = fail_times
        self.attempts = 0
        self.batches: list[list] = []

    def upsert(self, collection_name: str, points: list) -> None:
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise ResponseHandlingException("The write operation timed out")
        self.batches.append(list(points))


def _one_batch_points(n: int = 3):
    ids = [point_id("Doc", "text", 1, i) for i in range(n)]
    vectors = [[0.1] * EMBEDDING_DIM for _ in range(n)]
    payloads = [_payload("Doc", page=1) for _ in range(n)]
    return ids, vectors, payloads


@pytest.fixture
def recorded_sleeps(monkeypatch):
    """Replace vector_store's time.sleep with a recorder (no real backoff wait)."""
    sleeps: list[float] = []
    monkeypatch.setattr(vector_store.time, "sleep", sleeps.append)
    return sleeps


def test_upsert_retries_then_succeeds_on_transient_error(recorded_sleeps) -> None:
    client = _FlakyUpsertClient(fail_times=2)  # fail attempts 1 and 2, succeed on 3
    ids, vectors, payloads = _one_batch_points(3)
    assert upsert_points(client, "c", ids=ids, vectors=vectors, payloads=payloads) == 3
    assert client.attempts == 3  # two failures + one success
    assert recorded_sleeps == [1, 2]  # linear backoff before attempts 2 and 3


def test_upsert_gives_up_after_exhausting_attempts(recorded_sleeps) -> None:
    client = _FlakyUpsertClient(fail_times=99)  # always fails
    ids, vectors, payloads = _one_batch_points(3)
    with pytest.raises(ResponseHandlingException):
        upsert_points(client, "c", ids=ids, vectors=vectors, payloads=payloads)
    assert client.attempts == _UPSERT_MAX_ATTEMPTS  # capped, does not retry forever
    assert recorded_sleeps == [1, 2]  # backoff before each retry, none after the last


def test_upsert_splits_large_input_into_batches() -> None:
    client = _FlakyUpsertClient(fail_times=0)
    n = 2 * _UPSERT_BATCH + 5  # 133 -> three batches (64, 64, 5)
    ids, vectors, payloads = _one_batch_points(n)
    assert upsert_points(client, "c", ids=ids, vectors=vectors, payloads=payloads) == n
    assert client.attempts == 3
    assert [len(b) for b in client.batches] == [_UPSERT_BATCH, _UPSERT_BATCH, 5]
    assert sum(len(b) for b in client.batches) == n


# --- live test (opt-in: explicit flag, real server) -------------------------

_RUN_LIVE = os.environ.get("RUN_LIVE_QDRANT_TESTS") == "1"
live = pytest.mark.skipif(
    not _RUN_LIVE,
    reason="RUN_LIVE_QDRANT_TESTS!=1; skipping live Qdrant tests (opt-in only)",
)

_LIVE_SOURCE = "PYTEST — vector_store live roundtrip"


@live
def test_live_indexes_present_and_delete_then_upsert_drops_orphans() -> None:
    from app.core.config import get_settings
    from app.rag.vector_store import get_client

    settings = get_settings()
    collection = settings.qdrant_collection
    client = get_client(settings)

    try:
        ensure_collection(client, collection)  # create or reuse

        # Both indexes must exist (fresh) or have been back-filled (existing).
        schema = client.get_collection(collection).payload_schema or {}
        assert "domain" in schema, "domain index missing on real server"
        assert "source_name" in schema, "source_name index missing on real server"

        # First ingestion: 3 points for the test document.
        _upsert_doc(client, collection, _LIVE_SOURCE, 3)
        assert _count_source(client, collection, _LIVE_SOURCE) == 3

        # Re-ingestion with one fewer chunk: delete-then-upsert drops the orphan.
        delete_by_source(client, collection, _LIVE_SOURCE)
        _upsert_doc(client, collection, _LIVE_SOURCE, 2)
        assert _count_source(client, collection, _LIVE_SOURCE) == 2
    finally:
        # Never leave test points behind in the real collection.
        delete_by_source(client, collection, _LIVE_SOURCE)
