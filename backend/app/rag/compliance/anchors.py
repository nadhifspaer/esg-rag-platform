"""Locating a checklist requirement's answering row inside a disclosure-index table."""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

# Cell contents that mean "no value stated". A literal "0" is deliberately NOT here: a
# reported zero is a disclosure (BCA's E-05 waste is 0, and that is a real reported figure).
_BLANK_VALUES = frozenset({"", "-", "--", "—", "null", "n/a", "na", "nil", "tidak ada"})

# How many rows past the anchor to scan for a value when the anchor row's own cell is empty.
_LOOKAHEAD_ROWS = 6


@dataclass(frozen=True)
class AnchorMatch:
    """The row an anchor resolved to, and the value read from it."""

    anchor: str
    label: str
    value: str
    line: str
    value_from_following_row: bool

    @property
    def has_value(self) -> bool:
        """True when the located row (or its block) actually states something."""
        return self.value.strip().lower() not in _BLANK_VALUES


def _rows(text: str) -> Iterator[list[str]]:
    """Yield markdown table rows as cell lists, skipping separator rows."""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if all(set(cell) <= {"-", ":"} and cell for cell in cells):
            continue  # a | --- | --- | separator
        yield cells


def _leads_row(anchor: str, first_cell: str) -> bool:
    """True when `anchor` identifies this row, i.e. appears in its first cell (whole-token)."""
    pattern = rf"(?<![\w-]){re.escape(anchor)}(?![\w-])"
    return re.search(pattern, first_cell, re.IGNORECASE) is not None


def find_anchor_row(text: str, anchors: Sequence[str]) -> AnchorMatch | None:
    """Locate the first row in `text` identified by any of `anchors`, most specific first."""
    rows = list(_rows(text))
    for anchor in anchors:
        for index, cells in enumerate(rows):
            if not cells or not _leads_row(anchor, cells[0]):
                continue
            value = cells[-1] if len(cells) > 1 else ""
            if value.strip().lower() not in _BLANK_VALUES:
                return AnchorMatch(anchor, cells[0], value, " | ".join(cells), False)
            # Block-header shape: the answer is in a row below the anchor row.
            for following in rows[index + 1 : index + 1 + _LOOKAHEAD_ROWS]:
                if len(following) > 1 and following[-1].strip().lower() not in _BLANK_VALUES:
                    return AnchorMatch(anchor, cells[0], following[-1], " | ".join(following), True)
            return AnchorMatch(anchor, cells[0], "", " | ".join(cells), False)
    return None


def chunk_has_anchor(text: str, anchors: Sequence[str]) -> bool:
    """True when this chunk contains a row identified by one of `anchors`."""
    return find_anchor_row(text, anchors) is not None
