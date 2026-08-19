"""Qdrant chunk payload schema: the single source of truth for stored-chunk metadata."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Domain(StrEnum):
    """The two knowledge domains. Each value is the exact Qdrant metadata tag."""

    COMPANY_DOCUMENTS = "company_documents"
    STANDARDS = "standards"


class ContentType(StrEnum):
    """How a chunk's text was produced."""

    TEXT = "text"
    TABLE = "table"
    CHART_CAPTION = "chart_caption"


class ChunkPayload(BaseModel):
    """Metadata + text stored alongside each vector as a Qdrant point payload."""

    model_config = ConfigDict(extra="forbid")

    domain: Domain
    source_name: str = Field(
        ...,
        description="Human-readable document name, e.g. 'BCA Sustainability Report 2023'.",
    )
    document_type: str = Field(
        ...,
        description=(
            "Document class. Known vocabulary: 'sustainability_report', 'gri_standard', "
            "'edgb_standard', 'roadmap'. Free-form str, not an enum — new "
            "document classes can appear without a schema change."
        ),
    )
    page_number: int = Field(..., ge=1, description="1-based source page the chunk came from.")
    content_type: ContentType
    chunk_text: str = Field(
        ...,
        description="The chunk's text (caption text for chart_caption; markdown/CSV for tables).",
    )
    official_title: str | None = Field(
        None,
        description=(
            "Full formal title of the document where it differs from the short "
            "source_name, e.g. 'Roadmap Keuangan Berkelanjutan Tahap II (2021-2025)'."
        ),
    )
    bank: str | None = Field(
        None, description="Bank name, for company_documents; None where not applicable."
    )
    year: int | None = Field(None, description="Reporting/publication year where applicable.")
    image_url: str | None = Field(
        None,
        description="URL of the stored original page image. Required for chart_caption chunks.",
    )

    @model_validator(mode="after")
    def _chart_caption_requires_image(self) -> Self:
        if self.content_type is ContentType.CHART_CAPTION and not self.image_url:
            raise ValueError("image_url is required when content_type is 'chart_caption'")
        return self

    def to_payload(self) -> dict[str, Any]:
        """Serialize to the plain, JSON-native dict stored as a Qdrant point payload."""
        return self.model_dump(mode="json")

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Self:
        """Reconstruct from a Qdrant point payload, validating on the way in."""
        return cls.model_validate(payload)
