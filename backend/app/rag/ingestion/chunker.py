"""Chunking: parser output -> chunks sized by document type."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.models.payload import ContentType

if TYPE_CHECKING:
    # Guarded to break a runtime cycle: parser imports table_extractor, which imports
    # this module for `Chunk`. Only ever used as a type hint here.
    from app.rag.ingestion.parser import ExtractedDocument

# --- Strategies -----------------------------------------------------------


@dataclass(frozen=True)
class ChunkingStrategy:
    """How to split one document type's text."""

    name: str
    max_chars: int  # target size for a chunk's own content, before overlap
    overlap_chars: int  # trailing context carried into the next chunk
    unit_pattern: str  # regex; splits text into atomic units (never split across)

    def split_units(self, text: str) -> list[str]:
        """Split text into the smallest pieces this strategy keeps intact."""
        parts = re.split(self.unit_pattern, text, flags=re.MULTILINE)
        return [p.strip() for p in parts if p and p.strip()]


# Narrative prose: blank line = paragraph break.
_PARAGRAPH_BOUNDARY = r"\n\s*\n"

NARRATIVE = ChunkingStrategy(
    name="narrative",
    max_chars=1800,
    overlap_chars=200,
    unit_pattern=_PARAGRAPH_BOUNDARY,
)

# document_type -> strategy.
STRATEGY_BY_DOCUMENT_TYPE: dict[str, ChunkingStrategy] = {
    "sustainability_report": NARRATIVE,
    "gri_standard": NARRATIVE,
    "edgb_standard": NARRATIVE,
}

DEFAULT_STRATEGY = NARRATIVE


def strategy_for(document_type: str | None) -> ChunkingStrategy:
    """Look up the chunking strategy for a document type (unknown/missing -> NARRATIVE)."""
    if document_type is None:
        return DEFAULT_STRATEGY
    return STRATEGY_BY_DOCUMENT_TYPE.get(document_type, DEFAULT_STRATEGY)


# --- Chunks ---------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    """One chunk of text, ready to be turned into a ChunkPayload."""

    chunk_text: str
    page_number: int  # 1-based, inherited from the source page
    chunk_index: int  # position within this producer's output, 0-based
    document_type: str | None
    strategy_name: str  # how this chunk was produced
    content_type: ContentType = ContentType.TEXT
    # Set only for chart_caption chunks: the stored page image the caption came from.
    image_url: str | None = None

    @property
    def char_count(self) -> int:
        return len(self.chunk_text)


# --- Splitting ------------------------------------------------------------


def _hard_split(unit: str, max_chars: int) -> list[str]:
    """Break an oversized unit on whitespace so no chunk grows unbounded."""
    words = unit.split()
    pieces: list[str] = []
    current: list[str] = []
    length = 0
    for word in words:
        addition = len(word) + (1 if current else 0)
        if current and length + addition > max_chars:
            pieces.append(" ".join(current))
            current, length = [word], len(word)
        else:
            current.append(word)
            length += addition
    if current:
        pieces.append(" ".join(current))
    return pieces


def _overlap_tail(text: str, overlap_chars: int) -> str:
    """Take the last ~overlap_chars of text, trimmed to a word boundary."""
    if overlap_chars <= 0 or len(text) <= overlap_chars:
        return text if overlap_chars > 0 else ""
    tail = text[-overlap_chars:]
    space = tail.find(" ")
    return tail[space + 1 :] if space != -1 else tail


def chunk_text(text: str, strategy: ChunkingStrategy) -> list[str]:
    """Split one page of text into chunks per `strategy`, each carrying the prior chunk's tail."""
    units: list[str] = []
    for unit in strategy.split_units(text):
        units.extend(
            _hard_split(unit, strategy.max_chars) if len(unit) > strategy.max_chars else [unit]
        )

    chunks: list[str] = []
    current: list[str] = []
    length = 0

    def flush() -> None:
        nonlocal current, length
        if not current:
            return
        body = "\n\n".join(current)
        if chunks and strategy.overlap_chars:
            body = f"{_overlap_tail(chunks[-1], strategy.overlap_chars)}\n\n{body}"
        chunks.append(body)
        current, length = [], 0

    for unit in units:
        addition = len(unit) + (2 if current else 0)
        if current and length + addition > strategy.max_chars:
            flush()
        current.append(unit)
        length += len(unit) + (2 if len(current) > 1 else 0)
    flush()

    return chunks


def chunk_document(doc: ExtractedDocument) -> list[Chunk]:
    """Split a parsed document into chunks, page by page, using its manifest `document_type`."""
    document_type = doc.metadata.get("document_type")
    strategy = strategy_for(document_type)

    chunks: list[Chunk] = []
    for page in doc.pages:
        if not page.text.strip():
            continue
        for body in chunk_text(page.text, strategy):
            chunks.append(
                Chunk(
                    chunk_text=body,
                    page_number=page.page_number,
                    chunk_index=len(chunks),
                    document_type=document_type,
                    strategy_name=strategy.name,
                )
            )
    return chunks
