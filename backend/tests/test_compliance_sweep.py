"""Tests for the cross-bank checklist sweep, using the real environmental checklist."""

from __future__ import annotations

import pytest

from app.api.compliance import ChecklistCategory, load_checklist
from app.models.payload import ChunkPayload, ContentType, Domain
from app.rag.compliance.checklist import ChecklistItem
from app.rag.compliance.report import Status
from app.rag.compliance.retrieval import EvidenceResult
from app.rag.compliance.sweep import (
    RankingRouted,
    banks_with_status,
    rank_banks_by_coverage,
    rank_requirement,
    ranking_refusal_for_item,
    requirement_view,
    sweep_checklist,
    sweep_checklist_with_llm,
)
from app.rag.vector_store import SearchResult

# Real E-02 value row, the shape the anchors resolve against.
E02_ROW = (
    "| E-02 | GHG Emissions Intensity | Total scope 1 and 2 emissions produced per "
    "revenue of listed compant (tCO2e/Rp) | 1,37 |"
)

BANKS = [
    ("PT Bank Central Asia Tbk", "PT Bank Central Asia Tbk — 2025 Sustainability Report"),
    ("PT Bank Mandiri (Persero) Tbk", "PT Bank Mandiri (Persero) Tbk — 2025 Sustainability Report"),
    ("PT Bank Jago Tbk", "PT Bank Jago Tbk — 2025 Sustainability Report"),
]


@pytest.fixture
def checklist() -> list[ChecklistItem]:
    return load_checklist(ChecklistCategory.ENVIRONMENTAL)


def _evidence(item: ChecklistItem, text: str, score: float = 0.6) -> EvidenceResult:
    results = []
    if text:
        results = [
            SearchResult(
                id="1",
                score=score,
                payload=ChunkPayload(
                    domain=Domain.COMPANY_DOCUMENTS,
                    source_name="s",
                    document_type="sustainability_report",
                    page_number=13,
                    content_type=ContentType.TABLE,
                    chunk_text=text,
                ),
            )
        ]
    return EvidenceResult(
        item_code=item.code,
        query="q",
        results=results,
        top_score=score if results else 0.0,
        fallback_used=False,
        anchor_matched=bool(results),
        anchors_requested=bool(item.row_anchors),
    )


def test_sweep_returns_one_report_per_bank(checklist: list[ChecklistItem]) -> None:
    sweep = sweep_checklist(
        checklist,
        BANKS,
        lambda source_name, item: _evidence(item, E02_ROW),
        category="environmental",
    )

    assert sweep.bank_count == len(BANKS)
    assert [b.bank for b in sweep.banks] == [b for b, _ in BANKS]
    assert sweep.requirement_count == len(checklist)
    # The caveats travel with the result: a cross-bank table must not shed them.
    assert sweep.disclaimer and sweep.status_note and sweep.sweep_note


def test_requirement_view_partitions_every_bank(checklist: list[ChecklistItem]) -> None:
    sweep = sweep_checklist(
        checklist,
        BANKS,
        lambda source_name, item: _evidence(item, E02_ROW),
        category="environmental",
    )
    view = requirement_view(sweep, "GRI 305-4")

    total = len(view.addressed) + len(view.partial) + len(view.not_found) + len(view.not_checkable)
    assert total == view.bank_count  # no bank silently missing from the comparison


def test_scope_3_is_not_checkable_for_every_bank(checklist: list[ChecklistItem]) -> None:
    """The curated directly_disclosed=False propagates through the whole fan-out."""
    sweep = sweep_checklist(
        checklist,
        BANKS,
        lambda source_name, item: _evidence(item, E02_ROW),
        category="environmental",
    )
    view = requirement_view(sweep, "GRI 305-3")

    assert len(view.not_checkable) == len(BANKS)
    assert not view.addressed and not view.partial and not view.not_found
    # Crucially it is NOT reported as a failure to disclose.
    assert not banks_with_status(sweep, "GRI 305-3", Status.NOT_FOUND)


def test_banks_without_the_anchored_row_are_reported_as_failing(
    checklist: list[ChecklistItem],
) -> None:
    """'Which banks fail GRI 305-4': the query the sweep exists to answer."""

    def retrieve(source_name: str, item: ChecklistItem) -> EvidenceResult:
        # Only BCA's report carries the anchored row; the others return an unrelated chunk.
        has_row = "Central Asia" in source_name
        return _evidence(item, E02_ROW if has_row else "| Something else | 1 |")

    sweep = sweep_checklist(checklist, BANKS, retrieve, category="environmental")

    failing = banks_with_status(sweep, "GRI 305-4", Status.NOT_FOUND)
    assert {b.bank for b in failing} == {
        "PT Bank Mandiri (Persero) Tbk",
        "PT Bank Jago Tbk",
    }
    addressed = banks_with_status(sweep, "GRI 305-4", Status.DISCLOSED)
    assert [b.bank for b in addressed] == ["PT Bank Central Asia Tbk"]


def test_requirement_view_rejects_a_code_outside_the_checklist(
    checklist: list[ChecklistItem],
) -> None:
    """An empty result would read as 'no bank addresses this', a different claim."""
    sweep = sweep_checklist(
        checklist,
        BANKS,
        lambda source_name, item: _evidence(item, E02_ROW),
        category="environmental",
    )

    with pytest.raises(KeyError):
        requirement_view(sweep, "GRI 999-9")


def test_coverage_ranking_excludes_uncheckable_from_addressed(
    checklist: list[ChecklistItem],
) -> None:
    sweep = sweep_checklist(
        checklist,
        BANKS,
        lambda source_name, item: _evidence(item, E02_ROW),
        category="environmental",
    )
    ranking = rank_banks_by_coverage(sweep)

    assert len(ranking) == len(BANKS)
    for coverage in ranking:
        assert coverage.not_checkable == 1  # GRI 305-3, for every bank
        assert (
            coverage.addressed + coverage.partial + coverage.not_found + coverage.not_checkable
            == coverage.requirement_count
        )


# --- ranking mode: refuse where a ranking would misrepresent a non-result ---------------


def test_ranking_routes_a_coded_requirement_to_the_metrics_pipeline(
    checklist: list[ChecklistItem],
) -> None:
    """GRI 305-4 is answered by E-02, which every bank files: uniform, not a comparison."""
    sweep = sweep_checklist(
        checklist,
        BANKS,
        lambda source_name, item: _evidence(item, E02_ROW),
        category="environmental",
    )

    routed = rank_requirement(sweep, "GRI 305-4")

    assert isinstance(routed, RankingRouted)
    assert routed.reason == "coded_numeric_requirement"
    assert routed.route_to == "metrics"
    assert "metric aggregation pipeline" in routed.message


def test_ranking_refuses_a_not_directly_disclosed_requirement(
    checklist: list[ChecklistItem],
) -> None:
    sweep = sweep_checklist(
        checklist,
        BANKS,
        lambda source_name, item: _evidence(item, E02_ROW),
        category="environmental",
    )

    routed = rank_requirement(sweep, "GRI 305-3")

    assert isinstance(routed, RankingRouted)
    assert routed.reason == "not_directly_disclosed"
    assert routed.route_to == "none"
    # Must not read as an accusation against the banks.
    assert "failing to disclose" in routed.message


def test_ranking_refuses_an_unanchored_requirement_as_unverified() -> None:
    """Unanchored requirements are NOT rankable: the raw score band is uncalibrated, unverified."""
    social = load_checklist(ChecklistCategory.SOCIAL)
    sweep = sweep_checklist(
        social,
        BANKS,
        lambda source_name, item: _evidence(item, "some prose", score=0.55),
        category="social",
    )

    routed = rank_requirement(sweep, "GRI 401-1")

    assert isinstance(routed, RankingRouted)
    assert routed.reason == "unverified_ranking_output"
    assert "unverified output, not a finding" in routed.message
    # The underlying data is still reachable for inspection: refusal is about presenting
    # it as a ranking, not about hiding it.
    assert requirement_view(sweep, "GRI 401-1").bank_count == len(BANKS)


def test_refusal_is_decidable_from_the_item_without_running_a_sweep() -> None:
    """Lets the endpoint decline a query at zero retrieval cost; `calibrated` defaults True."""
    environmental = {i.code: i for i in load_checklist(ChecklistCategory.ENVIRONMENTAL)}
    social = {i.code: i for i in load_checklist(ChecklistCategory.SOCIAL)}

    anchored = ranking_refusal_for_item(environmental["GRI 305-4"], bank_count=21)
    scope_3 = ranking_refusal_for_item(environmental["GRI 305-3"], bank_count=21)
    narrative = ranking_refusal_for_item(social["GRI 401-1"], bank_count=21)
    narrative_uncalibrated = ranking_refusal_for_item(
        social["GRI 401-1"], bank_count=21, calibrated=False
    )

    assert anchored.reason == "coded_numeric_requirement" and anchored.route_to == "metrics"
    assert scope_3.reason == "not_directly_disclosed"
    assert narrative is None  # calibrated: rankable now
    assert narrative_uncalibrated.reason == "unverified_ranking_output"


def test_calibrated_sweep_ranks_an_unanchored_requirement_instead_of_refusing() -> None:
    """The exact case `test_ranking_refuses_an_unanchored_requirement_as_unverified` refuses
    for the raw engine must be a real ranking for the calibrated one."""
    social = load_checklist(ChecklistCategory.SOCIAL)

    def fake_judge(pending):
        # Trivial stand-in: every item with evidence is "disclosed" (judge quality is out of scope).
        return {p.item.code: Status.DISCLOSED for p in pending}

    sweep = sweep_checklist_with_llm(
        social,
        BANKS,
        lambda source_name, item: _evidence(item, "some prose", score=0.55),
        category="social",
        judge=fake_judge,
    )

    assert sweep.calibrated is True
    routed = rank_requirement(sweep, "GRI 401-1")

    assert not isinstance(routed, RankingRouted)
    assert len(routed.addressed) == len(BANKS)
