"""Table extraction: PDF tables -> markdown chunks tagged `content_type=table`, via pdfplumber."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pdfplumber

from app.models.payload import ContentType
from app.rag.ingestion.chunker import Chunk

if TYPE_CHECKING:
    from pdfplumber.page import Page

# pdfplumber over-detects tables; requiring a header row, a data row, and two columns filters most.
MIN_ROWS = 2
MIN_COLUMNS = 2

STRATEGY_NAME = "table_extraction"

# A raw pdfplumber table: rows of cells, where a cell may be None (empty).
RawTable = list[list[str | None]]

# A pdfplumber bounding box: (x0, top, x1, bottom) in PDF coordinates.
BBox = tuple[float, float, float, float]


def _clean_cell(cell: str | None) -> str:
    """Normalise one cell for markdown: no None, no newlines, no raw pipes."""
    if cell is None:
        return ""
    return " ".join(cell.split()).replace("|", "\\|")


def is_meaningful_table(table: RawTable) -> bool:
    """Reject tables too small or too empty to be real."""
    if not table or len(table) < MIN_ROWS:
        return False
    if max((len(row) for row in table), default=0) < MIN_COLUMNS:
        return False
    # A table where nearly every cell is blank is almost always a layout artefact.
    cells = [_clean_cell(c) for row in table for c in row]
    return sum(1 for c in cells if c) >= MIN_ROWS


def table_to_markdown(table: RawTable) -> str:
    """Render a pdfplumber table as a markdown table, padded to a uniform width."""
    rows = [[_clean_cell(cell) for cell in row] for row in table]
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]

    header, *body = rows
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def find_meaningful_tables(page: Page) -> list[tuple[RawTable, BBox]]:
    """Detect a page's meaningful tables; shared by `extract_tables` and `parser.py`'s exclusion."""
    found: list[tuple[RawTable, BBox]] = []
    for table in page.find_tables():
        rows = table.extract()
        if is_meaningful_table(rows):
            found.append((rows, table.bbox))
    return found


def extract_tables(pdf_path: Path, document_type: str | None = None) -> list[Chunk]:
    """Extract every table in a PDF as a markdown chunk (never split, to keep the header row)."""
    chunks: list[Chunk] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            for rows, _bbox in find_meaningful_tables(page):
                chunks.append(
                    Chunk(
                        chunk_text=table_to_markdown(rows),
                        page_number=page_number,
                        chunk_index=len(chunks),
                        document_type=document_type,
                        strategy_name=STRATEGY_NAME,
                        content_type=ContentType.TABLE,
                    )
                )
    return chunks
