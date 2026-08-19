"""Tests for the Citation model: building citations from chunks and formatting them."""

from __future__ import annotations

from app.models.citation import Citation
from app.models.payload import ChunkPayload, ContentType, Domain
from app.rag.vector_store import SearchResult


def _payload(
    *,
    source_name: str = "PT Bank Central Asia Tbk — 2025 Sustainability Report",
    page_number: int = 8,
    content_type: ContentType = ContentType.TEXT,
    image_url: str | None = None,
    chunk_text: str = "some retrieved text",
) -> ChunkPayload:
    return ChunkPayload(
        domain=Domain.COMPANY_DOCUMENTS,
        source_name=source_name,
        document_type="sustainability_report",
        page_number=page_number,
        content_type=content_type,
        chunk_text=chunk_text,
        image_url=image_url,
    )


def test_from_payload_maps_the_three_identity_fields() -> None:
    citation = Citation.from_payload(_payload(page_number=12, content_type=ContentType.TABLE))

    assert citation.source_name == "PT Bank Central Asia Tbk — 2025 Sustainability Report"
    assert citation.page_number == 12
    assert citation.content_type is ContentType.TABLE


# --- excerpt: asserted on the Citation object itself, not on truncate_snippet in isolation --


def test_from_payload_puts_the_cleaned_markdown_table_on_the_excerpt_field() -> None:
    """A raw markdown-table `chunk_text` must come out cleaned on `Citation.excerpt` itself,
    exercising the wiring in `from_payload`, not just `truncate_snippet` as a bare function."""
    raw = (
        "| Total Direct Emissions (Scope 1) | 6.032 |\n"
        "| --- | --- |\n"
        "| Total Indirect Emissions (Scope 2) | 147.851 |"
    )
    citation = Citation.from_payload(_payload(content_type=ContentType.TABLE, chunk_text=raw))

    assert citation.excerpt == (
        "Total Direct Emissions (Scope 1); 6.032; Total Indirect Emissions (Scope 2); 147.851"
    )
    # The artifacts the cleanup exists to remove must not survive onto the object's own
    # field: this is what "on the Citation object itself" actually rules out.
    assert "|" not in citation.excerpt
    assert "---" not in citation.excerpt


def test_from_payload_truncates_a_long_chunk_text_at_a_word_boundary() -> None:
    """The excerpt on the Citation object is capped and word-boundary-truncated, the same as
    a compliance evidence snippet, checked on `.excerpt` itself, not a bare function call."""
    long_text = (
        "This report describes the Board of Commissioners policy on anti-corruption and "
        "ethics, covering whistleblowing channels, gift policy, conflict of interest "
        "disclosure, and annual training for all employees and management across every "
        "business unit and subsidiary."
    )
    citation = Citation.from_payload(_payload(content_type=ContentType.TEXT, chunk_text=long_text))

    assert len(long_text) > 160
    assert len(citation.excerpt) <= 160
    assert citation.excerpt.endswith("…")  # the real ellipsis character, not "..."
    # Word-boundary, not mid-word: the excerpt ends cleanly at a whole word ("interest"),
    # not a fragment of the next one ("interes" or similar).
    assert citation.excerpt[:-1].endswith("interest")


def test_from_payload_does_not_truncate_a_long_table_chunk_text() -> None:
    """The 160-char cap is prose-only; a `content_type=table` excerpt is returned in full."""
    raw = (
        "| Direct emissions from stationary combustion | 1.813 |\n"
        "| --- | --- |\n"
        "| Direct emissions from mobile combustion | 11 |\n"
        "| Total Direct Emissions (Scope 1) | 6.032 |\n"
        "| Category 2: Indirect GHG emissions from imported energy | |"
    )
    citation = Citation.from_payload(_payload(content_type=ContentType.TABLE, chunk_text=raw))

    assert "…" not in citation.excerpt
    assert "Direct emissions from mobile combustion; 11" in citation.excerpt
    assert len(citation.excerpt) > 160


def test_from_payload_collapses_whitespace_in_a_short_chunk_text() -> None:
    citation = Citation.from_payload(
        _payload(content_type=ContentType.TEXT, chunk_text="Board  gender   diversity is 30%.")
    )

    assert citation.excerpt == "Board gender diversity is 30%."


def test_from_result_uses_the_result_payload() -> None:
    # A real retrieval result type satisfies the RetrievedChunk protocol.
    result = SearchResult(id="p1", score=0.9, payload=_payload(page_number=3))

    citation = Citation.from_result(result)

    assert citation.page_number == 3
    assert citation.source_name.startswith("PT Bank Central Asia Tbk")


def test_label_has_no_annotation_for_text() -> None:
    citation = Citation.from_payload(_payload(page_number=8, content_type=ContentType.TEXT))
    assert citation.label == "PT Bank Central Asia Tbk — 2025 Sustainability Report, p. 8"


def test_label_annotates_tables() -> None:
    citation = Citation.from_payload(
        _payload(
            source_name="POJK No.51/POJK.03/2017 — Attachment 1",
            page_number=3,
            content_type=ContentType.TABLE,
        )
    )
    assert citation.label == "POJK No.51/POJK.03/2017 — Attachment 1, p. 3 (table)"


def test_label_annotates_chart_captions() -> None:
    # chart_caption payloads require an image_url (payload validator), but the citation
    # itself stays text-only, no image field, just the "(figure)" annotation.
    citation = Citation.from_payload(
        _payload(
            source_name="ESG Disclosure Guide Book",
            page_number=15,
            content_type=ContentType.CHART_CAPTION,
            image_url="https://example/img.png",
        )
    )
    assert citation.label == "ESG Disclosure Guide Book, p. 15 (figure)"


def test_citations_are_frozen_and_dedupe_in_a_set() -> None:
    a = Citation.from_payload(_payload(page_number=8))
    b = Citation.from_payload(_payload(page_number=8))  # identical -> same citation
    c = Citation.from_payload(_payload(page_number=9))  # different page -> distinct

    assert a == b
    assert {a, b, c} == {a, c}  # duplicates collapse; the distinct page survives
