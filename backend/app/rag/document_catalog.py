"""Derive the ingested-document catalog from Qdrant payloads: there is no separate registry."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from app.models.payload import ChunkPayload, Domain


@dataclass(frozen=True)
class DocumentSummary:
    """One ingested document, aggregated from all its chunks, keyed by `source_name`."""

    source_name: str
    domain: Domain
    document_type: str
    bank: str | None
    year: int | None
    chunk_count: int
    page_count: int


@dataclass
class _Acc:
    """Mutable accumulator while folding a document's chunks together."""

    source_name: str
    domain: Domain
    document_type: str
    bank: str | None
    year: int | None
    chunk_count: int = 0
    pages: set[int] = field(default_factory=set)


def build_document_catalog(
    documents: Iterable[tuple[str, ChunkPayload]],
) -> list[DocumentSummary]:
    """Fold `(point_id, ChunkPayload)` pairs into one `DocumentSummary` per document."""
    accumulators: dict[str, _Acc] = {}
    for _point_id, payload in documents:
        acc = accumulators.get(payload.source_name)
        if acc is None:
            acc = _Acc(
                source_name=payload.source_name,
                domain=payload.domain,
                document_type=payload.document_type,
                bank=payload.bank,
                year=payload.year,
            )
            accumulators[payload.source_name] = acc
        acc.chunk_count += 1
        acc.pages.add(payload.page_number)

    summaries = [
        DocumentSummary(
            source_name=acc.source_name,
            domain=acc.domain,
            document_type=acc.document_type,
            bank=acc.bank,
            year=acc.year,
            chunk_count=acc.chunk_count,
            page_count=len(acc.pages),
        )
        for acc in accumulators.values()
    ]
    summaries.sort(key=lambda d: (d.domain.value, d.source_name))
    return summaries
