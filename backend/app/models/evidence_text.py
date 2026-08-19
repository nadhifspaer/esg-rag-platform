"""Shared evidence-text cleanup: markdown-table stripping and word-boundary truncation."""

from __future__ import annotations

import re

# Default excerpt length, the same cap `/compliance-check` evidence snippets already use.
SNIPPET_MAX = 160

# A markdown table's header-separator row wherever it appears in flowing text, e.g. `| --- | --- |`.
_TABLE_SEPARATOR_RE = re.compile(r"\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)+\|?")


def clean_table_markup(text: str) -> str:
    """Strip markdown table syntax so an excerpt reads as prose, not raw cells."""
    without_separators = _TABLE_SEPARATOR_RE.sub(" ", text)
    cell_joined = re.sub(r"\s*\|\s*", "; ", without_separators)
    # Collapse a run of "; " left by an empty cell, so it doesn't read as a missing value.
    return re.sub(r"(?:;\s*){2,}", "; ", cell_joined).strip(" ;")


def truncate_snippet(text: str, max_len: int = SNIPPET_MAX, *, is_table: bool = False) -> str:
    """Clean markdown table syntax, collapse whitespace, and truncate at a word boundary."""
    cleaned = re.sub(r"\s+", " ", clean_table_markup(text)).strip()
    if is_table or len(cleaned) <= max_len:
        return cleaned
    truncated = cleaned[:max_len]
    if not cleaned[max_len].isspace():
        boundary = truncated.rfind(" ")
        if boundary > 0:
            truncated = truncated[:boundary]
    return truncated.rstrip(" ;") + "…"
