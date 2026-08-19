"""Tests for row-anchored compliance retrieval and status assessment."""

from __future__ import annotations

from app.models.payload import ChunkPayload, ContentType, Domain
from app.rag.compliance.anchors import chunk_has_anchor, find_anchor_row
from app.rag.compliance.checklist import ChecklistItem
from app.rag.compliance.report import Status, assess_checklist, default_assess_status
from app.rag.compliance.retrieval import EvidenceResult
from app.rag.vector_store import SearchResult

# Real p.8 index table: the code sits in a LATER cell and the row ends in a page number.
INDEX_TABLE = (
    "| Kinerja | Kode | Nama Metrik | Halaman di Laporan Keberlanjutan/Tahunan |\n"
    "| --- | --- | --- | --- |\n"
    "| Lingkungan | E-01 | Laporan Emisi Gas Rumah Kaca | 89 |\n"
    "|  | E-02 | Intensitas Emisi Gas Rumah Kaca | 90 |\n"
    "|  | E-03 | Konsumsi Energi Listrik | 92 |"
)

# Real p.13 value table: the code LEADS the row and the row ends in the disclosed figure.
VALUE_TABLE = (
    "| E-02 | GHG Emissions Intensity | Total scope 1 and 2 emissions produced per "
    "revenue of listed compant (tCO2e/Rp) | 1,37 |\n"
    "| --- | --- | --- | --- |\n"
    "| E-03 | Electricity Consumption | Total amount of energy directly consumed "
    "(kWh or J) | 193.043.553 |\n"
    "| E-05 | Waste Generation | Total waste generated (ton) | 0 |"
)

# Real E-07 block: the anchor row's own value cell is empty; the answer is below it.
BLOCK_TABLE = (
    "| E-07 Komitmen Perusahaan untuk mengurangi Emisi Gas Rumah Kaca (Emission Reduction) |  |\n"
    "| --- | --- |\n"
    "| Apakah Perusahaan memiliki komitmen mengurangi emisi? | Tidak |\n"
    "| Target pengurangan emisi GRK | 0 % |"
)


def _chunk(text: str, score: float = 0.6) -> SearchResult:
    return SearchResult(
        id="1",
        score=score,
        payload=ChunkPayload(
            domain=Domain.COMPANY_DOCUMENTS,
            source_name="PT Bank Central Asia Tbk — 2025 Sustainability Report",
            document_type="sustainability_report",
            page_number=13,
            content_type=ContentType.TABLE,
            chunk_text=text,
        ),
    )


def _evidence(*chunks: SearchResult, anchor_matched: bool = True) -> EvidenceResult:
    return EvidenceResult(
        item_code="GRI 305-4",
        query="q",
        results=list(chunks),
        top_score=chunks[0].score if chunks else 0.0,
        fallback_used=False,
        anchor_matched=anchor_matched,
        anchors_requested=True,
    )


def _item(
    code: str = "GRI 305-4", anchors: tuple[str, ...] = ("E-02",), **kw: object
) -> ChecklistItem:
    return ChecklistItem(
        code=code, description="d", topic_keywords=["k"], row_anchors=anchors, **kw
    )


# --- the first-cell rule: an index row must never be read as a value row ----------------


def test_anchor_ignores_the_index_row_where_the_value_is_a_page_number() -> None:
    """The whole reason the anchor must lead the row: p.8 would yield '90' as E-02's value."""
    assert find_anchor_row(INDEX_TABLE, ["E-02"]) is None
    assert chunk_has_anchor(INDEX_TABLE, ["E-02"]) is False


def test_anchor_reads_the_value_row() -> None:
    match = find_anchor_row(VALUE_TABLE, ["E-02"])

    assert match is not None
    assert match.value == "1,37"
    assert match.has_value is True
    assert match.value_from_following_row is False


def test_anchor_matches_whole_tokens_only() -> None:
    assert find_anchor_row(VALUE_TABLE, ["E-0"]) is None  # not a prefix match
    assert find_anchor_row(VALUE_TABLE, ["E-05"]) is not None


def test_reported_zero_counts_as_a_disclosed_value() -> None:
    """BCA really reports 0 tonnes of waste: a figure, not a missing value."""
    match = find_anchor_row(VALUE_TABLE, ["E-05"])

    assert match is not None and match.value == "0"
    assert match.has_value is True


# --- block-header shape: the answer lives below the anchor row --------------------------


def test_anchor_reads_a_value_from_the_row_below_a_block_header() -> None:
    match = find_anchor_row(BLOCK_TABLE, ["E-07"])

    assert match is not None
    assert match.value == "Tidak"
    assert match.value_from_following_row is True
    # "Tidak" means the report addresses the topic and answers no, still a disclosure.
    assert match.has_value is True


def test_anchors_are_tried_in_order_most_specific_first() -> None:
    """GRI 305-5 lists E-07 then E-06; with only E-06 present the second anchor fires."""
    e06_only = "| E-06 Company commitment to Net Zero Emission Target |  |\n| Does it? | No |"

    assert find_anchor_row(BLOCK_TABLE, ["E-07", "E-06"]).anchor == "E-07"
    assert find_anchor_row(e06_only, ["E-07", "E-06"]).anchor == "E-06"


# --- the assessor now reads the row, not the score --------------------------------------


def test_anchored_status_comes_from_the_row_not_the_score() -> None:
    """A low-scoring chunk containing the row outranks the score band's verdict."""
    low_scoring = _evidence(_chunk(VALUE_TABLE, score=0.10))

    assert default_assess_status(_item(), low_scoring) is Status.DISCLOSED


def test_anchored_status_is_not_found_when_the_row_is_absent() -> None:
    """The discrimination the score band could not do: a high score with no anchored row."""
    high_scoring_wrong_chunk = _evidence(_chunk(INDEX_TABLE, score=0.95), anchor_matched=False)

    assert default_assess_status(_item(), high_scoring_wrong_chunk) is Status.NOT_FOUND


def test_anchored_status_is_partial_when_the_row_states_nothing() -> None:
    empty_row = "| E-02 | GHG Emissions Intensity | tCO2e/Rp |  |"

    assert (
        default_assess_status(_item(), _evidence(_chunk(empty_row))) is Status.PARTIALLY_DISCLOSED
    )


def test_unanchored_items_keep_the_score_band_path() -> None:
    """Narrative requirements still use the heuristic, where prose genuinely differs."""
    unanchored = _item(anchors=())

    assert (
        default_assess_status(unanchored, _evidence(_chunk("prose", score=0.80)))
        is Status.DISCLOSED
    )
    assert default_assess_status(unanchored, _evidence(_chunk("prose", score=0.40))) is (
        Status.PARTIALLY_DISCLOSED
    )
    assert (
        default_assess_status(unanchored, _evidence(_chunk("prose", score=0.10)))
        is Status.NOT_FOUND
    )


# --- directly_disclosed=False short-circuits before any retrieval -----------------------


def test_not_directly_disclosed_skips_retrieval_entirely() -> None:
    """GRI 305-3's real shape: searching could only return a misleading near-miss."""
    scope_3 = _item(code="GRI 305-3", anchors=(), directly_disclosed=False)
    calls: list[str] = []

    def retrieve(item: ChecklistItem) -> EvidenceResult:
        calls.append(item.code)
        raise AssertionError("retrieval ran for a not-directly-disclosed requirement")

    report = assess_checklist(
        [scope_3], report_name="BCA", category="environmental", retrieve=retrieve
    )

    assert calls == []
    row = report.results[0]
    assert row.status is Status.NOT_DIRECTLY_DISCLOSED
    assert row.evidence_snippet is None and row.citation is None


def test_summary_excludes_uncheckable_requirements_from_the_denominator() -> None:
    """A requirement with no row in the corpus is not a failure to disclose."""
    checked = _item()
    skipped = _item(code="GRI 305-3", anchors=(), directly_disclosed=False)

    report = assess_checklist(
        [checked, skipped],
        report_name="BCA",
        category="environmental",
        retrieve=lambda item: _evidence(_chunk(VALUE_TABLE)),
    )

    assert "Of 1 environmental disclosure requirements" in report.summary
    assert "A further 1 could not be checked" in report.summary
