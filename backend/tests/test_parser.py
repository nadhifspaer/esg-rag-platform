"""Tests for PDF text extraction and manifest metadata passthrough."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pypdf import PdfWriter

from app.models.payload import Domain
from app.rag.ingestion.parser import (
    ManifestError,
    PageText,
    extract_document,
    extract_pages,
    infer_domain,
    load_manifest,
    looks_scanned,
    lookup_metadata,
)

MANIFEST = {
    "domains": {
        "company_documents": [
            {
                "file": "BANK-BCA.pdf",
                "bank": "Bank Central Asia",
                "year": 2025,
                "document_type": "sustainability_report",
            }
        ],
        "standards": [
            {
                "file": "gri-1.pdf",  # note: differs in case from disk
                "document_type": "gri_standard",
                "official_title": "GRI 1: Foundation",
                "aliases": ["GRI 1", "GRI1"],
            }
        ],
    }
}


@pytest.fixture
def raw_root(tmp_path: Path) -> Path:
    """A miniature data/raw/ tree with a manifest and blank PDFs."""
    (tmp_path / "manifest.json").write_text(json.dumps(MANIFEST), encoding="utf-8")
    for domain in Domain:
        (tmp_path / domain.value).mkdir()
    _write_blank_pdf(tmp_path / "company_documents" / "BANK-BCA.pdf", pages=3)
    _write_blank_pdf(tmp_path / "standards" / "GRI-1.pdf", pages=2)
    return tmp_path


def _write_blank_pdf(path: Path, pages: int) -> None:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    with path.open("wb") as fh:
        writer.write(fh)


def test_infer_domain_from_folder(raw_root: Path) -> None:
    pdf = raw_root / "standards" / "GRI-1.pdf"
    assert infer_domain(pdf) is Domain.STANDARDS


def test_infer_domain_rejects_unknown_folder(tmp_path: Path) -> None:
    stray = tmp_path / "not_a_domain" / "x.pdf"
    stray.parent.mkdir()
    stray.touch()
    with pytest.raises(ManifestError):
        infer_domain(stray)


def test_metadata_passthrough_keeps_every_field(raw_root: Path) -> None:
    """Heterogeneous fields must survive verbatim, not be reduced to a fixed set."""
    manifest = load_manifest(raw_root)
    pdf = raw_root / "standards" / "GRI-1.pdf"
    metadata = lookup_metadata(pdf, Domain.STANDARDS, manifest)

    assert metadata["document_type"] == "gri_standard"
    assert metadata["official_title"] == "GRI 1: Foundation"
    assert metadata["aliases"] == ["GRI 1", "GRI1"]
    assert "file" not in metadata  # identity, exposed as ExtractedDocument.file_name


def test_metadata_lookup_is_case_insensitive(raw_root: Path) -> None:
    """Manifest says 'gri-1.pdf'; disk says 'GRI-1.pdf'."""
    manifest = load_manifest(raw_root)
    pdf = raw_root / "standards" / "GRI-1.pdf"
    assert lookup_metadata(pdf, Domain.STANDARDS, manifest)["document_type"] == "gri_standard"


def test_unregistered_document_raises(raw_root: Path) -> None:
    manifest = load_manifest(raw_root)
    orphan = raw_root / "standards" / "Unlisted.pdf"
    _write_blank_pdf(orphan, pages=1)
    with pytest.raises(ManifestError, match="No manifest.json entry"):
        lookup_metadata(orphan, Domain.STANDARDS, manifest)


def test_extract_pages_numbers_pages_from_one(raw_root: Path) -> None:
    pages = extract_pages(raw_root / "company_documents" / "BANK-BCA.pdf")
    assert [p.page_number for p in pages] == [1, 2, 3]


def test_extract_document_combines_text_and_metadata(raw_root: Path) -> None:
    doc = extract_document(raw_root / "company_documents" / "BANK-BCA.pdf", raw_root=raw_root)
    assert doc.domain is Domain.COMPANY_DOCUMENTS
    assert doc.file_name == "BANK-BCA.pdf"
    assert doc.metadata["bank"] == "Bank Central Asia"
    assert doc.page_count == 3


def test_blank_pdf_is_flagged_as_needing_ocr(raw_root: Path) -> None:
    """Blank pages stand in for an image-only scan: no extractable text."""
    doc = extract_document(raw_root / "company_documents" / "BANK-BCA.pdf", raw_root=raw_root)
    assert doc.needs_ocr is True


def test_native_text_pages_are_not_flagged() -> None:
    pages = [PageText(page_number=i, text="x" * 500) for i in range(1, 6)]
    assert looks_scanned(pages) is False
