"""Qdrant access: one client, one collection, domain as an indexed filter field."""

from __future__ import annotations

import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from app.core.config import Settings, get_settings
from app.models.payload import ChunkPayload, Domain
from app.rag.embedder import EMBEDDING_DIM

# Fixed namespace so deterministic point IDs are stable across processes/runs.
_POINT_NAMESPACE = uuid.UUID("6f0d7c2e-1a4b-5c8d-9e0f-1a2b3c4d5e6f")

# The field retrieval filters on for domain-scoped search.
_DOMAIN_FIELD = "domain"

# The field that identifies a document; point IDs derive from it.
_SOURCE_FIELD = "source_name"

# Payload fields we filter on and therefore must index; a real Qdrant server (unlike
# in-memory Qdrant) rejects a filter on an unindexed field.
_INDEXED_FIELDS = (_DOMAIN_FIELD, _SOURCE_FIELD)

# Generous timeout/batch/retry settings for writing to Qdrant Cloud over the public internet.
_CLIENT_TIMEOUT_SECONDS = 60
_UPSERT_BATCH = 64
_UPSERT_MAX_ATTEMPTS = 3


def get_client(settings: Settings | None = None) -> QdrantClient:
    """Build a Qdrant client from settings (api_key blank -> local, unauthenticated)."""
    settings = settings or get_settings()
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        timeout=_CLIENT_TIMEOUT_SECONDS,
    )


def _ensure_payload_indexes(client: QdrantClient, collection_name: str) -> None:
    """Create any missing keyword indexes on the fields we filter by (idempotent, self-healing)."""
    indexed = set((client.get_collection(collection_name).payload_schema or {}).keys())
    for field in _INDEXED_FIELDS:
        if field not in indexed:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    *,
    vector_size: int = EMBEDDING_DIM,
) -> bool:
    """Ensure the collection and its filter indexes exist. Returns True if created."""
    created = not client.collection_exists(collection_name)
    if created:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
    _ensure_payload_indexes(client, collection_name)
    return created


def point_id(source_name: str, content_type: str, page_number: int, chunk_index: int) -> str:
    """Deterministic point ID for one chunk, so re-ingesting overwrites, not duplicates."""
    key = f"{source_name}|{content_type}|{page_number}|{chunk_index}"
    return str(uuid.uuid5(_POINT_NAMESPACE, key))


def delete_by_source(client: QdrantClient, collection_name: str, source_name: str) -> None:
    """Delete every point belonging to one document, matched by `source_name`."""
    client.delete(
        collection_name=collection_name,
        points_selector=FilterSelector(
            filter=Filter(
                must=[FieldCondition(key=_SOURCE_FIELD, match=MatchValue(value=source_name))]
            )
        ),
    )


def _upsert_batch_with_retry(
    client: QdrantClient, collection_name: str, points: list[PointStruct]
) -> None:
    """Upsert one batch, retrying on transient network errors (upsert is idempotent, so safe)."""
    for attempt in range(1, _UPSERT_MAX_ATTEMPTS + 1):
        try:
            client.upsert(collection_name=collection_name, points=points)
            return
        except ResponseHandlingException:
            if attempt == _UPSERT_MAX_ATTEMPTS:
                raise
            time.sleep(attempt)  # linear backoff: 1s, then 2s


def upsert_points(
    client: QdrantClient,
    collection_name: str,
    *,
    ids: Sequence[str],
    vectors: Sequence[Sequence[float]],
    payloads: Sequence[ChunkPayload],
) -> int:
    """Upsert chunk vectors + payloads, batched and retried. Returns the count upserted."""
    points = [
        PointStruct(id=pid, vector=list(vector), payload=payload.to_payload())
        for pid, vector, payload in zip(ids, vectors, payloads, strict=True)
    ]
    if not points:
        return 0
    for start in range(0, len(points), _UPSERT_BATCH):
        _upsert_batch_with_retry(client, collection_name, points[start : start + _UPSERT_BATCH])
    return len(points)


# --- retrieval: domain-filtered vector search -------------------------------


@dataclass(frozen=True)
class SearchResult:
    """One scored hit from a retrieval search: Qdrant's cosine score plus the parsed payload."""

    id: str
    score: float
    payload: ChunkPayload


def _query_filter(domain: Domain, source_name: str | None) -> Filter:
    """A Qdrant filter restricting a search to one domain, optionally one document."""
    must = [FieldCondition(key=_DOMAIN_FIELD, match=MatchValue(value=domain.value))]
    if source_name is not None:
        must.append(FieldCondition(key=_SOURCE_FIELD, match=MatchValue(value=source_name)))
    return Filter(must=must)


def search(
    client: QdrantClient,
    collection_name: str,
    *,
    query_vector: Sequence[float],
    domain: Domain,
    source_name: str | None = None,
    limit: int = 10,
    score_threshold: float | None = None,
) -> list[SearchResult]:
    """Return the chunks in one domain closest to `query_vector`, most similar first."""
    response = client.query_points(
        collection_name=collection_name,
        query=list(query_vector),
        query_filter=_query_filter(domain, source_name),
        limit=limit,
        score_threshold=score_threshold,
        with_payload=True,
    )
    return [
        SearchResult(
            id=str(point.id),
            score=point.score,
            payload=ChunkPayload.from_payload(point.payload or {}),
        )
        for point in response.points
    ]
