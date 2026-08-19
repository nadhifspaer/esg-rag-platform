"""Tests for building the ingested-document catalog from chunk payloads."""

from __future__ import annotations

from app.models.payload import ChunkPayload, ContentType, Domain
from app.rag.document_catalog import build_document_catalog


def _chunk(
    source: str,
    domain: Domain,
    *,
    page: int,
    doc_type: str,
    bank=None,
    year=None,
):
    payload = ChunkPayload(
        domain=domain,
        source_name=source,
        document_type=doc_type,
        page_number=page,
        content_type=ContentType.TEXT,
        chunk_text="x",
        bank=bank,
        year=year,
    )
    return ("id", payload)


def test_aggregates_chunks_and_distinct_pages_per_document() -> None:
    docs = [
        _chunk(
            "BCA — 2025",
            Domain.COMPANY_DOCUMENTS,
            page=1,
            doc_type="sustainability_report",
            bank="BCA",
            year=2025,
        ),
        _chunk(
            "BCA — 2025",
            Domain.COMPANY_DOCUMENTS,
            page=1,
            doc_type="sustainability_report",
            bank="BCA",
            year=2025,
        ),
        _chunk(
            "BCA — 2025",
            Domain.COMPANY_DOCUMENTS,
            page=2,
            doc_type="sustainability_report",
            bank="BCA",
            year=2025,
        ),
    ]
    catalog = build_document_catalog(docs)
    assert len(catalog) == 1
    doc = catalog[0]
    assert doc.source_name == "BCA — 2025"
    assert doc.chunk_count == 3
    assert doc.page_count == 2  # pages {1, 2} deduped
    assert doc.bank == "BCA"
    assert doc.year == 2025
    assert doc.domain is Domain.COMPANY_DOCUMENTS


def test_groups_by_source_name_and_sorts_by_domain_then_name() -> None:
    docs = [
        _chunk("ESG Disclosure Guide Book", Domain.STANDARDS, page=1, doc_type="edgb_standard"),
        _chunk("GRI 1: Foundation", Domain.STANDARDS, page=1, doc_type="gri_standard"),
        _chunk("BCA — 2025", Domain.COMPANY_DOCUMENTS, page=1, doc_type="sustainability_report"),
        _chunk("GRI 2: General Disclosures", Domain.STANDARDS, page=1, doc_type="gri_standard"),
    ]
    catalog = build_document_catalog(docs)
    ordered = [(d.domain.value, d.source_name) for d in catalog]
    assert ordered == [
        ("company_documents", "BCA — 2025"),
        ("standards", "ESG Disclosure Guide Book"),
        ("standards", "GRI 1: Foundation"),
        ("standards", "GRI 2: General Disclosures"),
    ]


def test_standards_and_regulations_carry_no_bank_or_year() -> None:
    catalog = build_document_catalog(
        [_chunk("GRI 1: Foundation", Domain.STANDARDS, page=1, doc_type="gri_standard")]
    )
    assert catalog[0].bank is None
    assert catalog[0].year is None


def test_empty_corpus_yields_empty_catalog() -> None:
    assert build_document_catalog([]) == []
