"""Tests for compliance report assembly, status heuristic, and citation formatting."""

from __future__ import annotations

from app.models.evidence_text import SNIPPET_MAX
from app.models.payload import ChunkPayload, ContentType, Domain
from app.rag.compliance.checklist import ChecklistItem
from app.rag.compliance.report import (
    DISCLAIMER,
    STATUS_NOTE,
    ComplianceReport,
    Status,
    assess_checklist,
    default_assess_status,
    format_evidence_citation,
)
from app.rag.compliance.retrieval import EvidenceResult
from app.rag.vector_store import SearchResult


def _item(code: str = "GRI 305-1", desc: str = "Direct (Scope 1) GHG emissions") -> ChecklistItem:
    return ChecklistItem(code=code, description=desc, topic_keywords=["scope 1", "tCO2e"])


def _company_result(
    text: str, score: float, *, page: int = 13, content_type: ContentType = ContentType.TABLE
) -> SearchResult:
    payload = ChunkPayload(
        domain=Domain.COMPANY_DOCUMENTS,
        source_name="PT Bank Central Asia Tbk — 2025 Sustainability Report",
        document_type="sustainability_report",
        page_number=page,
        content_type=content_type,
        chunk_text=text,
        bank="PT Bank Central Asia Tbk",
        year=2025,
    )
    return SearchResult(id=f"c-{page}", score=score, payload=payload)


def _evidence(results: list[SearchResult], *, fallback: bool = False) -> EvidenceResult:
    top = results[0].score if results else 0.0
    return EvidenceResult(
        item_code="X", query="q", results=results, top_score=top, fallback_used=fallback
    )


# --- status heuristic -------------------------------------------------------


def test_status_bands() -> None:
    item = _item()
    assert default_assess_status(item, _evidence([_company_result("t", 0.7)])) is Status.DISCLOSED
    assert (
        default_assess_status(item, _evidence([_company_result("t", 0.42)]))
        is Status.PARTIALLY_DISCLOSED
    )
    assert default_assess_status(item, _evidence([_company_result("t", 0.20)])) is Status.NOT_FOUND
    assert default_assess_status(item, _evidence([])) is Status.NOT_FOUND


# --- citation formatting ----------------------------------------------------


def test_company_citation_uses_source_name() -> None:
    payload = _company_result("t", 0.7, page=13).payload
    assert format_evidence_citation(payload) == (
        "PT Bank Central Asia Tbk — 2025 Sustainability Report, p. 13 (table)"
    )


# --- assembly ---------------------------------------------------------------


def test_assess_checklist_builds_rows_summary_and_disclaimer() -> None:
    checklist = [
        _item("GRI 305-1", "Direct (Scope 1) GHG emissions"),
        _item("GRI 305-4", "GHG emissions intensity"),
        _item("GRI 306-3", "Waste generated"),
    ]
    # Crafted evidence: disclosed, partial, not-found (empty).
    canned = {
        "GRI 305-1": _evidence([_company_result("Total Direct Emissions (Scope 1) 1,345.9", 0.65)]),
        "GRI 305-4": _evidence([_company_result("Emissions intensity per revenue", 0.40)]),
        "GRI 306-3": _evidence([]),
    }
    report = assess_checklist(
        checklist,
        report_name="PT Bank Central Asia Tbk — 2025 Sustainability Report",
        category="environmental",
        retrieve=lambda item: canned[item.code],
    )

    assert isinstance(report, ComplianceReport)
    by_code = {r.code: r for r in report.results}
    assert by_code["GRI 305-1"].status is Status.DISCLOSED
    assert by_code["GRI 305-1"].evidence_snippet is not None
    assert by_code["GRI 305-1"].citation is not None
    assert by_code["GRI 305-1"].rejected_as_insubstantial is False
    assert by_code["GRI 305-4"].status is Status.PARTIALLY_DISCLOSED
    assert by_code["GRI 305-4"].rejected_as_insubstantial is True
    # NOT_FOUND with a genuinely empty retrieval: nothing to show, nothing rejected.
    assert by_code["GRI 306-3"].status is Status.NOT_FOUND
    assert by_code["GRI 306-3"].evidence_snippet is None
    assert by_code["GRI 306-3"].citation is None
    assert by_code["GRI 306-3"].rejected_as_insubstantial is False

    assert "1 addressed, 1 partially addressed, and 1 not found" in report.summary
    assert report.disclaimer == DISCLAIMER


# --- rejected_as_insubstantial: retrieved but judged insufficient (vs. an empty retrieval) ---


def test_not_found_with_retrieved_evidence_still_shows_it_and_flags_rejection() -> None:
    item = _item("GRI 412-1", "Human rights review")
    # Below PARTIAL_SCORE (0.35) -> NOT_FOUND, but a chunk *was* retrieved.
    low_score_evidence = _evidence([_company_result("unrelated boilerplate text", 0.10)])

    report = assess_checklist(
        [item],
        report_name="PT Bank Central Asia Tbk — 2025 Sustainability Report",
        category="social",
        retrieve=lambda _item: low_score_evidence,
    )

    row = report.results[0]
    assert row.status is Status.NOT_FOUND
    assert row.evidence_snippet is not None
    assert row.citation is not None
    assert row.rejected_as_insubstantial is True


def test_genuinely_empty_retrieval_never_flags_rejection() -> None:
    item = _item("GRI 412-1", "Human rights review")

    report = assess_checklist(
        [item],
        report_name="PT Bank Central Asia Tbk — 2025 Sustainability Report",
        category="social",
        retrieve=lambda _item: _evidence([]),
    )

    row = report.results[0]
    assert row.status is Status.NOT_FOUND
    assert row.evidence_snippet is None
    assert row.citation is None
    assert row.rejected_as_insubstantial is False


def test_not_directly_disclosed_never_flags_rejection() -> None:
    item = ChecklistItem(
        code="GRI 305-3", description="Scope 3", topic_keywords=["k"], directly_disclosed=False
    )

    report = assess_checklist(
        [item],
        report_name="PT Bank Central Asia Tbk — 2025 Sustainability Report",
        category="environmental",
        retrieve=lambda _item: (_ for _ in ()).throw(AssertionError("should never retrieve")),
    )

    row = report.results[0]
    assert row.status is Status.NOT_DIRECTLY_DISCLOSED
    assert row.rejected_as_insubstantial is False


def test_disclosed_never_flags_rejection() -> None:
    item = _item("GRI 305-1", "Direct (Scope 1) GHG emissions")
    evidence = _evidence([_company_result("Total Direct Emissions (Scope 1) 1,345.9", 0.65)])

    report = assess_checklist(
        [item],
        report_name="PT Bank Central Asia Tbk — 2025 Sustainability Report",
        category="environmental",
        retrieve=lambda _item: evidence,
    )

    row = report.results[0]
    assert row.status is Status.DISCLOSED
    assert row.rejected_as_insubstantial is False


def test_to_markdown_renders_table_summary_disclaimer_and_cleans_pipes() -> None:
    checklist = [_item("GRI 305-1", "Direct (Scope 1) GHG emissions")]
    # Evidence text contains pipes (real table markdown), must not break the row or leak.
    evidence = _evidence([_company_result("| Scope 1 | 1,345.9 tCO2e |", 0.7)])
    report = assess_checklist(
        checklist,
        report_name="PT Bank Central Asia Tbk — 2025 Sustainability Report",
        category="environmental",
        retrieve=lambda item: evidence,
    )
    md = report.to_markdown()

    assert "| Requirement | Status | Evidence | Citation |" in md
    assert "✅ Addressed" in md
    assert "✅ Disclosed" not in md  # the misreadable label is gone
    assert "**Summary:**" in md
    assert STATUS_NOTE in md  # the clarifier line is rendered
    assert DISCLAIMER in md
    # The snippet's own pipes convert to "; " before the row is built, so none survive to escape.
    data_rows = [ln for ln in md.splitlines() if ln.startswith("| GRI 305-1")]
    assert len(data_rows) == 1
    assert data_rows[0].count("|") == 5
    assert "\\|" not in data_rows[0]
    assert "Scope 1; 1,345.9 tCO2e" in data_rows[0]


def test_evidence_snippet_drops_table_separator_row_and_reads_as_a_list() -> None:
    """Real case (BCA/governance GRI 2-9): a separator row must not render as literal dashes."""
    raw = (
        "| Governance | G-01 | Management Diversity and Independence | 304-305 | "
        "| --- | --- | --- | --- | "
        "| | G-02 | Total Attendance of Directors and Commissioners to General Meetings | 306 |"
    )
    checklist = [_item("GRI 2-9", "Governance body composition")]
    evidence = _evidence([_company_result(raw, 0.7)])
    report = assess_checklist(
        checklist, report_name="Report", category="governance", retrieve=lambda item: evidence
    )
    snippet = report.results[0].evidence_snippet
    assert snippet is not None
    assert "---" not in snippet
    assert "\\|" not in snippet
    assert "|" not in snippet
    assert "Governance; G-01; Management Diversity and Independence; 304-305" in snippet


def test_evidence_snippet_truncates_at_a_word_boundary_not_mid_word() -> None:
    """Real case (BCA/governance GRI 2-9): the 160-char cut landed inside "total", a broken word."""
    raw = ("x" * 155) + " Commissioners General Meetings"
    checklist = [_item("GRI 2-9", "Governance body composition")]
    evidence = _evidence([_company_result(raw, 0.7, content_type=ContentType.TEXT)])
    report = assess_checklist(
        checklist, report_name="Report", category="governance", retrieve=lambda item: evidence
    )
    snippet = report.results[0].evidence_snippet
    assert snippet is not None
    assert snippet == ("x" * 155) + "…"
    assert "Commi" not in snippet  # no partial word survives before the ellipsis


def test_evidence_snippet_does_not_drop_a_whole_word_that_already_fit() -> None:
    """Regression: backing up to the previous space must not discard a word that already fit."""
    raw = ("x" * 155) + " SASB more-text-that-will-not-fit"
    checklist = [_item("GRI 205-1", "Anti-corruption policies and procedures")]
    evidence = _evidence([_company_result(raw, 0.7, content_type=ContentType.TEXT)])
    report = assess_checklist(
        checklist, report_name="Report", category="governance", retrieve=lambda item: evidence
    )
    snippet = report.results[0].evidence_snippet
    assert snippet == ("x" * 155) + " SASB…"


def test_evidence_snippet_preserves_a_full_table_row_past_the_prose_cap() -> None:
    """Real case (BCA GRI 305-1): TABLE chunks skip the 160-char cap, so the full row survives."""
    raw = (
        "| Name | Total emission (tCO2e) |\n| --- | --- |\n"
        "| Direct emissions from stationary combustion | 1.813 |\n"
        "| Direct emissions from mobile combustion | 11 |\n"
        "| Total Direct Emissions (Scope 1) | 6.032 |"
    )
    checklist = [_item("GRI 305-1", "Direct (Scope 1) GHG emissions")]
    evidence = _evidence([_company_result(raw, 0.7)])  # content_type defaults to TABLE
    report = assess_checklist(
        checklist, report_name="Report", category="environmental", retrieve=lambda item: evidence
    )
    snippet = report.results[0].evidence_snippet
    assert snippet is not None
    assert "…" not in snippet  # nothing was cut, so no ellipsis is appended
    assert "Direct emissions from mobile combustion; 11" in snippet
    assert "Total Direct Emissions (Scope 1); 6.032" in snippet
    assert len(snippet) > SNIPPET_MAX


def test_evidence_snippet_collapses_an_empty_table_cell_to_one_separator() -> None:
    """A blank leading column must not render as "; ; ", which reads as a typo, not empty."""
    raw = "Value one | | Value two"
    checklist = [_item("GRI 2-9", "Governance body composition")]
    evidence = _evidence([_company_result(raw, 0.7)])
    report = assess_checklist(
        checklist, report_name="Report", category="governance", retrieve=lambda item: evidence
    )
    snippet = report.results[0].evidence_snippet
    assert snippet == "Value one; Value two"


def test_not_found_rows_show_dashes_in_markdown() -> None:
    checklist = [_item("GRI 306-3", "Waste generated")]
    report = assess_checklist(
        checklist,
        report_name="Report",
        category="environmental",
        retrieve=lambda item: _evidence([]),
    )
    md = report.to_markdown()
    row = next(ln for ln in md.splitlines() if ln.startswith("| GRI 306-3"))
    assert "❌ Not found" in row
    assert "| — | — |" in row  # evidence and citation dashes


def test_to_markdown_flags_rejected_but_retrieved_evidence_with_qualifier() -> None:
    """Rejected-but-retrieved evidence must render visually distinct from an empty retrieval."""
    checklist = [_item("GRI 412-1", "Human rights review")]
    # Below PARTIAL_SCORE (0.35) -> NOT_FOUND, but a chunk *was* retrieved -> rejected.
    evidence = _evidence([_company_result("Unrelated boilerplate text", 0.10)])
    report = assess_checklist(
        checklist, report_name="Report", category="social", retrieve=lambda item: evidence
    )
    row = report.results[0]
    assert row.rejected_as_insubstantial is True  # precondition for this test to mean anything

    md = report.to_markdown()
    line = next(ln for ln in md.splitlines() if ln.startswith("| GRI 412-1"))
    _, _, evidence_cell, _ = (c.strip() for c in line.strip("|").split("|"))
    assert "Unrelated boilerplate text" in evidence_cell
    assert "_(evidence found, judged insufficient)_" in evidence_cell
    # The qualifier is appended after the snippet, not prepended before it.
    assert evidence_cell.index("Unrelated boilerplate text") < evidence_cell.index(
        "_(evidence found, judged insufficient)_"
    )


def test_to_markdown_omits_qualifier_when_evidence_not_rejected() -> None:
    """Guard against the qualifier firing wrongly: an accepted snippet renders unqualified."""
    checklist = [_item("GRI 305-1", "Direct (Scope 1) GHG emissions")]
    evidence = _evidence([_company_result("Total Direct Emissions (Scope 1) 1,345.9", 0.65)])
    report = assess_checklist(
        checklist, report_name="Report", category="environmental", retrieve=lambda item: evidence
    )
    row = report.results[0]
    assert row.rejected_as_insubstantial is False  # precondition for this test to mean anything

    md = report.to_markdown()
    line = next(ln for ln in md.splitlines() if ln.startswith("| GRI 305-1"))
    _, _, evidence_cell, _ = (c.strip() for c in line.strip("|").split("|"))
    assert "Total Direct Emissions (Scope 1) 1,345.9" in evidence_cell
    assert "_(evidence found, judged insufficient)_" not in evidence_cell
