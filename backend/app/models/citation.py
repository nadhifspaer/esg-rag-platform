"""Citation model: the app's one consistent way to attribute a retrieved chunk."""

from __future__ import annotations

from typing import Protocol, Self

from pydantic import BaseModel, ConfigDict

from app.models.evidence_text import truncate_snippet
from app.models.payload import ChunkPayload, ContentType

# How each content type is annotated in the human-readable `label` (prose gets no suffix).
_CONTENT_TYPE_SUFFIX: dict[ContentType, str] = {
    ContentType.TEXT: "",
    ContentType.TABLE: " (table)",
    ContentType.CHART_CAPTION: " (figure)",
}


class RetrievedChunk(Protocol):
    """Structural type for anything a citation can be built from: exposes `ChunkPayload`."""

    @property
    def payload(self) -> ChunkPayload: ...


class Citation(BaseModel):
    """Attribution for one retrieved chunk: which document, which page, what content."""

    model_config = ConfigDict(frozen=True)

    source_name: str
    page_number: int
    content_type: ContentType
    excerpt: str

    @classmethod
    def from_payload(cls, payload: ChunkPayload) -> Self:
        """Build a citation from a chunk's payload, including a cleaned, truncated excerpt."""
        return cls(
            source_name=payload.source_name,
            page_number=payload.page_number,
            content_type=payload.content_type,
            excerpt=truncate_snippet(
                payload.chunk_text, is_table=payload.content_type is ContentType.TABLE
            ),
        )

    @classmethod
    def from_result(cls, result: RetrievedChunk) -> Self:
        """Build a citation from any retrieval result carrying a payload."""
        return cls.from_payload(result.payload)

    @property
    def label(self) -> str:
        """The canonical one-line citation string, e.g. "GRI 305: Emissions 2016, p. 4 (table)"."""
        return f"{self.source_name}, p. {self.page_number}{_CONTENT_TYPE_SUFFIX[self.content_type]}"
