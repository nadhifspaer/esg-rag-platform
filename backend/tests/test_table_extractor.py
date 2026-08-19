"""Tests for pdfplumber table extraction and markdown rendering."""

from __future__ import annotations

from app.models.payload import ContentType
from app.rag.ingestion.table_extractor import (
    is_meaningful_table,
    table_to_markdown,
)

EMISSIONS_TABLE = [
    ["Scope", "2023 (tCO2e)", "2024 (tCO2e)"],
    ["Scope 1", "1,234", "1,180"],
    ["Scope 2", "45,600", "43,900"],
]


def test_first_row_becomes_markdown_header() -> None:
    md = table_to_markdown(EMISSIONS_TABLE)
    lines = md.splitlines()
    assert lines[0] == "| Scope | 2023 (tCO2e) | 2024 (tCO2e) |"
    assert lines[1] == "| --- | --- | --- |"
    assert lines[2] == "| Scope 1 | 1,234 | 1,180 |"


def test_numbers_are_preserved_verbatim() -> None:
    """The whole point of structural extraction: digits come out exactly."""
    md = table_to_markdown(EMISSIONS_TABLE)
    for value in ("1,234", "1,180", "45,600", "43,900"):
        assert value in md


def test_none_cells_render_as_empty() -> None:
    md = table_to_markdown([["A", "B"], ["1", None]])
    assert md.splitlines()[-1] == "| 1 |  |"


def test_cell_newlines_are_collapsed() -> None:
    """Wrapped cell text must not break the markdown row structure."""
    md = table_to_markdown([["Header", "Value"], ["Total\nemissions", "10"]])
    assert md.splitlines()[-1] == "| Total emissions | 10 |"
    assert len(md.splitlines()) == 3


def test_pipes_in_cells_are_escaped() -> None:
    md = table_to_markdown([["A", "B"], ["x | y", "2"]])
    assert "x \\| y" in md


def test_ragged_rows_are_padded_to_uniform_width() -> None:
    md = table_to_markdown([["A", "B", "C"], ["1"]])
    assert md.splitlines()[-1] == "| 1 |  |  |"


def test_meaningful_table_accepted() -> None:
    assert is_meaningful_table(EMISSIONS_TABLE) is True


def test_single_row_table_rejected() -> None:
    assert is_meaningful_table([["only", "header"]]) is False


def test_single_column_table_rejected() -> None:
    assert is_meaningful_table([["a"], ["b"], ["c"]]) is False


def test_mostly_empty_table_rejected() -> None:
    assert is_meaningful_table([[None, None], [None, None]]) is False


def test_empty_input_rejected() -> None:
    assert is_meaningful_table([]) is False


def test_chunk_content_type_constant_is_table() -> None:
    """Guards the tag these chunks must carry into Qdrant."""
    assert ContentType.TABLE.value == "table"
