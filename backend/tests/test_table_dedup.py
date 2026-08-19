"""Tests that table content is not duplicated between plain-text and table chunks."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.models.payload import ContentType
from app.rag.ingestion.parser import extract_pages
from app.rag.ingestion.table_extractor import extract_tables

# reportlab draws real vector grid lines, which is what pdfplumber's default
# line-based table detection keys on. Skip cleanly if it isn't installed.
pytest.importorskip("reportlab")

# Text that appears ONLY inside the table, used to prove it doesn't leak into
# the plain-text extraction. Distinctive numbers, unlikely to collide with prose.
TABLE_ONLY_VALUES = ("1,234", "1,180", "45,600", "43,900")

# Prose that sits above and below the table and MUST survive in the page text.
PROSE_ABOVE = "Our total greenhouse gas emissions are summarized in the table below."
PROSE_BELOW = "These figures were independently assured by a third party."


def _build_pdf_with_table_and_prose(path: Path) -> None:
    """Write a one-page PDF: a prose line, a lined 3x3 table, another prose line."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    styles = getSampleStyleSheet()
    table = Table(
        [
            ["Scope", "2023", "2024"],
            ["Scope 1", "1,234", "1,180"],
            ["Scope 2", "45,600", "43,900"],
        ],
        style=TableStyle([("GRID", (0, 0), (-1, -1), 1, colors.black)]),
    )
    SimpleDocTemplate(str(path), pagesize=letter).build(
        [
            Paragraph(PROSE_ABOVE, styles["Normal"]),
            Spacer(1, 24),
            table,
            Spacer(1, 24),
            Paragraph(PROSE_BELOW, styles["Normal"]),
        ]
    )


@pytest.fixture
def pdf_with_table(tmp_path: Path) -> Path:
    path = tmp_path / "table_and_prose.pdf"
    _build_pdf_with_table_and_prose(path)
    return path


def test_table_becomes_exactly_one_table_chunk(pdf_with_table: Path) -> None:
    tables = extract_tables(pdf_with_table, document_type="sustainability_report")
    assert len(tables) == 1
    chunk = tables[0]
    assert chunk.content_type is ContentType.TABLE
    for value in TABLE_ONLY_VALUES:
        assert value in chunk.chunk_text, f"{value!r} missing from the table chunk"


def test_table_content_is_excluded_from_page_text(pdf_with_table: Path) -> None:
    """The whole point: the table's numbers must not also appear in the prose."""
    pages = extract_pages(pdf_with_table)
    assert len(pages) == 1
    text = pages[0].text

    # Surrounding prose survives.
    assert "greenhouse gas emissions are summarized" in text
    assert "independently assured" in text

    # The table's distinctive values do NOT leak into the plain text.
    for value in TABLE_ONLY_VALUES:
        assert value not in text, f"{value!r} leaked into page text (duplicated as prose)"
