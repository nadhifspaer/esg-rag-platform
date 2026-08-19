"""PDF text extraction: a PDF file -> raw text per page, plus its metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pdfplumber

from app.models.payload import Domain
from app.rag.ingestion.table_extractor import BBox, find_meaningful_tables

if TYPE_CHECKING:
    from pdfplumber.page import Page

# --- Scanned-PDF detection heuristics: conservative, flags for review rather than acting ---
MIN_CHARS_PER_PAGE = 50
MAX_EMPTY_PAGE_RATIO = 0.8

MANIFEST_FILENAME = "manifest.json"


class ManifestError(RuntimeError):
    """Raised when a PDF has no usable manifest entry or sits in an unknown domain."""


@dataclass(frozen=True)
class PageText:
    """Raw text of a single PDF page."""

    page_number: int  # 1-based, matches the page_number in ChunkPayload
    text: str

    @property
    def char_count(self) -> int:
        return len(self.text.strip())

    @property
    def is_empty(self) -> bool:
        return self.char_count < MIN_CHARS_PER_PAGE


@dataclass(frozen=True)
class ExtractedDocument:
    """A parsed PDF: its pages of raw text plus all metadata known about it."""

    file_path: Path
    file_name: str
    domain: Domain
    metadata: dict[str, Any]  # every field from the manifest entry, verbatim
    pages: list[PageText]
    needs_ocr: bool

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def total_chars(self) -> int:
        return sum(p.char_count for p in self.pages)

    @property
    def empty_page_count(self) -> int:
        return sum(1 for p in self.pages if p.is_empty)


# --- Manifest -------------------------------------------------------------


def load_manifest(raw_root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Load manifest.json into {domain: {lowercased filename: entry}}."""
    manifest_path = raw_root / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise ManifestError(f"Manifest not found at {manifest_path}")

    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    index: dict[str, dict[str, dict[str, Any]]] = {}
    for domain_name, entries in raw.get("domains", {}).items():
        index[domain_name] = {entry["file"].lower(): entry for entry in entries}
    return index


def infer_domain(pdf_path: Path) -> Domain:
    """Derive the knowledge domain from the PDF's parent folder name."""
    folder = pdf_path.parent.name
    try:
        return Domain(folder)
    except ValueError as exc:
        valid = ", ".join(d.value for d in Domain)
        raise ManifestError(
            f"{pdf_path.name} is in folder '{folder}', which is not a known domain ({valid})"
        ) from exc


def lookup_metadata(
    pdf_path: Path,
    domain: Domain,
    manifest: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Return every metadata field the manifest records for this document (minus `file`)."""
    entries = manifest.get(domain.value, {})
    entry = entries.get(pdf_path.name.lower())
    if entry is None:
        raise ManifestError(
            f"No manifest.json entry for '{pdf_path.name}' under domain '{domain.value}'. "
            "Every document must be registered before ingestion."
        )
    return {key: value for key, value in entry.items() if key != "file"}


# --- Extraction -----------------------------------------------------------


def _not_within_bboxes(obj: dict[str, Any], bboxes: list[BBox]) -> bool:
    """True if a pdfplumber object's midpoint lies outside every table bounding box."""
    h_mid = (obj["x0"] + obj["x1"]) / 2
    v_mid = (obj["top"] + obj["bottom"]) / 2
    return not any(x0 <= h_mid < x1 and top <= v_mid < bottom for x0, top, x1, bottom in bboxes)


def _page_text_excluding_tables(page: Page) -> str:
    """Extract a page's plain text with its meaningful table regions removed."""
    bboxes = [bbox for _rows, bbox in find_meaningful_tables(page)]
    source = page.filter(lambda obj: _not_within_bboxes(obj, bboxes)) if bboxes else page
    return source.extract_text() or ""


def extract_pages(pdf_path: Path) -> list[PageText]:
    """Extract raw text from every page of a PDF, in order (1-based numbering), tables excluded."""
    with pdfplumber.open(str(pdf_path)) as pdf:
        return [
            PageText(page_number=i, text=_page_text_excluding_tables(page))
            for i, page in enumerate(pdf.pages, start=1)
        ]


def looks_scanned(pages: list[PageText]) -> bool:
    """True if the document appears to be image-only and would need OCR."""
    if not pages:
        return True
    empty_ratio = sum(1 for p in pages if p.is_empty) / len(pages)
    return empty_ratio >= MAX_EMPTY_PAGE_RATIO


def extract_document(
    pdf_path: Path,
    manifest: dict[str, dict[str, dict[str, Any]]] | None = None,
    raw_root: Path | None = None,
) -> ExtractedDocument:
    """Parse one PDF into per-page text plus its full manifest metadata."""
    pdf_path = Path(pdf_path)
    if manifest is None:
        manifest = load_manifest(raw_root or pdf_path.parent.parent)

    domain = infer_domain(pdf_path)
    metadata = lookup_metadata(pdf_path, domain, manifest)
    pages = extract_pages(pdf_path)

    return ExtractedDocument(
        file_path=pdf_path,
        file_name=pdf_path.name,
        domain=domain,
        metadata=metadata,
        pages=pages,
        needs_ocr=looks_scanned(pages),
    )


def iter_documents(raw_root: Path) -> list[ExtractedDocument]:
    """Parse every PDF under `raw_root/<domain>/`, in sorted order."""
    manifest = load_manifest(raw_root)
    documents: list[ExtractedDocument] = []
    for domain in Domain:
        domain_dir = raw_root / domain.value
        if not domain_dir.is_dir():
            continue
        for pdf_path in sorted(domain_dir.glob("*.pdf")):
            documents.append(extract_document(pdf_path, manifest=manifest))
    return documents
