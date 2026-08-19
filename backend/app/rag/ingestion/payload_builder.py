"""Turn a chunk + its document's manifest metadata into a `ChunkPayload`."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.models.payload import ChunkPayload, Domain
from app.rag.ingestion.chunker import Chunk

# Em dash used consistently across citation labels.
_DASH = "—"


def build_source_name(metadata: Mapping[str, Any]) -> str:
    """Build the human-readable citation label for a document from its metadata."""
    document_type = metadata.get("document_type")

    if document_type == "sustainability_report":
        return f"{metadata['bank']} {_DASH} {metadata['year']} Sustainability Report"

    if document_type in {"gri_standard", "edgb_standard", "roadmap"}:
        return metadata["official_title"]

    raise ValueError(
        f"cannot build a source_name for document_type {document_type!r}; "
        "add a rule in build_source_name for this document type"
    )


def build_chunk_payload(
    chunk: Chunk,
    *,
    domain: Domain,
    metadata: Mapping[str, Any],
) -> ChunkPayload:
    """Assemble the `ChunkPayload` for one chunk, filtering manifest metadata to declared fields."""
    fields: dict[str, Any] = {
        key: value for key, value in metadata.items() if key in ChunkPayload.model_fields
    }
    fields.update(
        domain=domain,
        source_name=build_source_name(metadata),
        page_number=chunk.page_number,
        content_type=chunk.content_type,
        chunk_text=chunk.chunk_text,
    )
    if chunk.image_url is not None:
        fields["image_url"] = chunk.image_url
    return ChunkPayload(**fields)
