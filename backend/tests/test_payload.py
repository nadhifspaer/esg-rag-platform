"""Tests for the Qdrant chunk payload schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models.payload import ChunkPayload, ContentType, Domain


def _text_chunk(**overrides: object) -> ChunkPayload:
    data: dict[str, object] = {
        "domain": Domain.COMPANY_DOCUMENTS,
        "source_name": "BCA Sustainability Report 2023",
        "document_type": "sustainability_report",
        "page_number": 12,
        "content_type": ContentType.TEXT,
        "chunk_text": "Scope 1 emissions were 1,234 tCO2e in 2023.",
        "bank": "BCA",
        "year": 2023,
    }
    data.update(overrides)
    return ChunkPayload(**data)  # type: ignore[arg-type]


def test_valid_text_chunk_builds() -> None:
    chunk = _text_chunk()
    assert chunk.domain is Domain.COMPANY_DOCUMENTS
    assert chunk.image_url is None


def test_to_payload_serializes_enums_as_strings() -> None:
    payload = _text_chunk().to_payload()
    assert payload["domain"] == "company_documents"
    assert payload["content_type"] == "text"


def test_round_trip_through_payload() -> None:
    original = _text_chunk()
    restored = ChunkPayload.from_payload(original.to_payload())
    assert restored == original


def test_chart_caption_requires_image_url() -> None:
    with pytest.raises(ValidationError):
        _text_chunk(content_type=ContentType.CHART_CAPTION, image_url=None)


def test_chart_caption_with_image_url_is_valid() -> None:
    chunk = _text_chunk(
        content_type=ContentType.CHART_CAPTION,
        image_url="https://storage.example/bca-2023-p12.png",
    )
    assert chunk.content_type is ContentType.CHART_CAPTION


def test_invalid_domain_rejected() -> None:
    with pytest.raises(ValidationError):
        _text_chunk(domain="regulation")  # not one of the three enum values


def test_page_number_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        _text_chunk(page_number=0)


def test_official_title_defaults_to_none_and_is_settable() -> None:
    assert _text_chunk().official_title is None
    titled = _text_chunk(
        official_title="Roadmap Keuangan Berkelanjutan Tahap II (2021-2025)",
    )
    assert titled.to_payload()["official_title"].startswith("Roadmap Keuangan")


def test_unmodelled_extra_field_is_rejected() -> None:
    """extra='forbid': unknown keys must raise, not vanish silently."""
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _text_chunk(some_unmodelled_field="whatever")


def test_aliases_is_rejected_on_chunk_payload() -> None:
    """aliases stays in manifest.json, read at classify time: duplicating it per chunk is waste."""
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _text_chunk(aliases=["BCA", "Bank Central Asia"])


@pytest.mark.parametrize(
    "document_type",
    [
        "sustainability_report",
        "gri_standard",
        "edgb_standard",
        "roadmap",
    ],
)
def test_known_document_types_are_accepted(document_type: str) -> None:
    """Documents the corpus's document_type vocabulary (see data/raw/manifest.json)."""
    assert _text_chunk(document_type=document_type).document_type == document_type
