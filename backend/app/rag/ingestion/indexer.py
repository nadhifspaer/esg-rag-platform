"""Index a document's chunks into Qdrant: build payloads -> embed -> upsert."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from openai import OpenAI
from qdrant_client import QdrantClient

from app.core.config import Settings, get_settings
from app.models.payload import Domain
from app.rag.embedder import embed_texts
from app.rag.ingestion.chunker import Chunk
from app.rag.ingestion.payload_builder import build_chunk_payload
from app.rag.vector_store import delete_by_source, ensure_collection, point_id, upsert_points


def index_chunks(
    chunks: Sequence[Chunk],
    *,
    domain: Domain,
    metadata: Mapping[str, Any],
    qdrant_client: QdrantClient,
    collection: str | None = None,
    embed_client: OpenAI | None = None,
    settings: Settings | None = None,
) -> int:
    """Embed and upsert one document's chunks. Returns the number of points upserted."""
    if not chunks:
        return 0

    settings = settings or get_settings()
    collection = collection or settings.qdrant_collection

    payloads = [build_chunk_payload(chunk, domain=domain, metadata=metadata) for chunk in chunks]
    vectors = embed_texts(
        [chunk.chunk_text for chunk in chunks], client=embed_client, settings=settings
    )
    ids = [
        point_id(
            payload.source_name,
            chunk.content_type.value,
            chunk.page_number,
            chunk.chunk_index,
        )
        for payload, chunk in zip(payloads, chunks, strict=True)
    ]

    ensure_collection(qdrant_client, collection)
    # Clear this document's existing points first, so a re-ingestion that yields
    # fewer chunks leaves no orphaned points behind.
    delete_by_source(qdrant_client, collection, payloads[0].source_name)
    return upsert_points(qdrant_client, collection, ids=ids, vectors=vectors, payloads=payloads)
