"""Tests for document-type-driven chunking."""

from __future__ import annotations

from pathlib import Path

from app.models.payload import Domain
from app.rag.ingestion.chunker import (
    NARRATIVE,
    chunk_document,
    chunk_text,
    strategy_for,
)
from app.rag.ingestion.parser import ExtractedDocument, PageText

# Default multi-paragraph text for _document() below, used across the strategy
# and chunk-mechanics tests that don't care about specific content.
SAMPLE_TEXT = "\n\n".join(
    [
        "BAB I KETENTUAN UMUM Dalam Peraturan Otoritas Jasa Keuangan ini yang "
        "dimaksud dengan Keuangan Berkelanjutan adalah dukungan menyeluruh dari "
        "sektor jasa keuangan untuk menciptakan pertumbuhan ekonomi yang "
        "berkelanjutan dengan menyelaraskan kepentingan ekonomi, sosial, dan "
        "lingkungan hidup.",
        "Pasal 1 Lembaga Jasa Keuangan, Emiten, dan Perusahaan Publik wajib "
        "menerapkan Keuangan Berkelanjutan dalam kegiatan usahanya secara "
        "bertahap sesuai dengan ketentuan yang berlaku dan rencana aksi yang "
        "telah disusun oleh masing-masing lembaga.",
        "Pasal 2 Penerapan Keuangan Berkelanjutan bertujuan menyediakan sumber "
        "pendanaan yang dibutuhkan untuk mencapai target pembangunan "
        "berkelanjutan dan meningkatkan daya tahan sektor jasa keuangan "
        "terhadap risiko perubahan iklim.",
        "Pasal 3 Lembaga Jasa Keuangan wajib menyusun Rencana Aksi Keuangan "
        "Berkelanjutan yang memuat prioritas dan target jangka pendek maupun "
        "jangka panjang, serta disampaikan kepada Otoritas Jasa Keuangan sesuai "
        "jadwal yang ditetapkan.",
        "Roadmap ini menjabarkan arah pengembangan keuangan berkelanjutan di "
        "Indonesia untuk periode berikutnya, mencakup penguatan kerangka "
        "kebijakan, pengembangan produk keuangan berkelanjutan, dan peningkatan "
        "kapasitas sumber daya manusia di sektor jasa keuangan nasional.",
    ]
)


def _document(document_type: str, text: str = SAMPLE_TEXT, pages: int = 2):
    """Build parser output with the given document_type; text held constant."""
    return ExtractedDocument(
        file_path=Path(f"/fake/{document_type}.pdf"),
        file_name=f"{document_type}.pdf",
        domain=Domain.STANDARDS,  # both live here, proving domain isn't the signal
        metadata={"document_type": document_type, "year": 2021},
        pages=[PageText(page_number=i, text=text) for i in range(1, pages + 1)],
        needs_ocr=False,
    )


# --- Strategy selection ---------------------------------------------------


def test_narrative_types_map_to_narrative() -> None:
    for document_type in (
        "roadmap",
        "sustainability_report",
        "gri_standard",
        "edgb_standard",
    ):
        assert strategy_for(document_type) is NARRATIVE, document_type


def test_unknown_and_missing_types_fall_back_to_narrative() -> None:
    assert strategy_for("something_new") is NARRATIVE
    assert strategy_for(None) is NARRATIVE


# --- The core comparison --------------------------------------------------


# --- Chunk mechanics ------------------------------------------------------


def test_chunks_carry_page_numbers_and_never_span_pages() -> None:
    chunks = chunk_document(_document("gri_standard", pages=3))
    assert {c.page_number for c in chunks} == {1, 2, 3}


def test_chunk_indexes_are_sequential() -> None:
    chunks = chunk_document(_document("roadmap", pages=2))
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_empty_pages_produce_no_chunks() -> None:
    doc = _document("gri_standard", text="   \n\n  ")
    assert chunk_document(doc) == []


def test_oversized_unit_is_split_rather_than_emitted_whole() -> None:
    """A single paragraph longer than max_chars must not become one giant chunk."""
    giant = " ".join(["kata"] * 2000)
    chunks = chunk_text(giant, NARRATIVE)
    assert len(chunks) > 1
    assert all(c.count(" ") > 0 for c in chunks)
