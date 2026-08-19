from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.rag.compliance.retrieval import EvidenceResult
from app.rag.metrics.aggregation import BankMetricEvidence
from app.rag.metrics.analysis import (
    ExclusionReason,
    filter_banks,
    mean,
    parse_metric_number,
    rank_banks,
    total,
)
from app.rag.metrics.extraction import (
    ExtractedValue,
    ExtractionStatus,
    MetricExtraction,
    extract_metric_value,
)
from app.rag.metrics.metric import MetricSpec

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> list[MetricExtraction]:
    """Rebuild MetricExtraction objects from a committed live-run fixture."""
    data = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return [
        MetricExtraction(
            bank=row["bank"],
            metric_code=row["metric_code"],
            value_type=row["value_type"],
            status=ExtractionStatus(row["status"]),
            values=tuple(ExtractedValue(value=v["value"], label=v["label"]) for v in row["values"]),
            citation=row["citation"],
            note=row["note"],
        )
        for row in data["results"]
    ]


@pytest.fixture
def e06() -> list[MetricExtraction]:
    return load_fixture("e06_live.json")


@pytest.fixture
def s05() -> list[MetricExtraction]:
    return load_fixture("s05_live.json")


# --- fixture integrity: guard the real data the rest of the file reasons about ----------


def test_e06_fixture_is_the_real_21_bank_run(e06: list[MetricExtraction]) -> None:
    assert len(e06) == 21
    counts = {status: 0 for status in ExtractionStatus}
    for extraction in e06:
        counts[extraction.status] += 1
    assert counts[ExtractionStatus.DISCLOSED] == 10
    assert counts[ExtractionStatus.NOT_DISCLOSED] == 7
    assert counts[ExtractionStatus.MULTIPLE_DISCLOSED] == 4


def test_s05_fixture_is_fully_disclosed_and_dot_free(s05: list[MetricExtraction]) -> None:
    assert len(s05) == 21
    assert all(e.status is ExtractionStatus.DISCLOSED for e in s05)
    # No dots anywhere is why this fixture can carry the arithmetic tests unambiguously.
    assert all("." not in v.value for e in s05 for v in e.values)


# --- the load-bearing default: multiple_disclosed is segregated, not ranked -------------


def test_multi_value_banks_are_held_out_of_ranking_by_default(e06: list[MetricExtraction]) -> None:
    result = rank_banks(e06)

    assert len(result.ranked) == 10  # only the single-value banks
    assert len(result.ambiguous) == 4
    ambiguous_banks = {bank.bank for bank in result.ambiguous}
    assert ambiguous_banks == {
        "PT Bank Negara Indonesia (Persero) Tbk",
        "PT Bank Mandiri (Persero) Tbk",
        "PT Bank CIMB Niaga Tbk",
        "PT Bank SMBC Indonesia Tbk",
    }
    # Held out, but fully preserved: both scoped years travel with each bank.
    bni = next(b for b in result.ambiguous if "Negara Indonesia" in b.bank)
    assert bni.values == (
        ("2028", "Operational Net Zero Emission"),
        ("2060", "Financed Emissions Net Zero"),
    )
    # And none of them leaked into the ranking.
    assert ambiguous_banks.isdisjoint({row.bank for row in result.ranked})


def test_default_ranking_leads_with_a_whole_footprint_year_not_an_operations_year(
    e06: list[MetricExtraction],
) -> None:
    """The failure the default prevents: under `explode`, ops-only years lead, not the footprint."""
    default_top = rank_banks(e06, ascending=True, top_n=1).ranked[0]
    assert default_top.bank == "PT Bank Aladin Syariah Tbk"
    assert default_top.value == 2040.0

    exploded_top4 = rank_banks(e06, ascending=True, top_n=4, explode=True).ranked
    assert [row.value for row in exploded_top4] == [2028.0, 2030.0, 2030.0, 2030.0]
    assert all(row.label is not None for row in exploded_top4)  # every row names its scope
    assert exploded_top4[0].bank == "PT Bank Negara Indonesia (Persero) Tbk"
    assert exploded_top4[0].label == "Operational Net Zero Emission"


def test_explode_expands_a_multi_bank_into_one_row_per_scope(e06: list[MetricExtraction]) -> None:
    result = rank_banks(e06, explode=True)

    assert result.exploded is True
    assert not result.ambiguous  # nothing held out any more
    assert len(result.ranked) == 10 + (4 * 2)  # 10 single-value banks + 4 banks x 2 scopes
    bni_rows = [r for r in result.ranked if "Negara Indonesia" in r.bank]
    assert {(r.value, r.label) for r in bni_rows} == {
        (2028.0, "Operational Net Zero Emission"),
        (2060.0, "Financed Emissions Net Zero"),
    }


def test_single_value_banks_rank_with_an_unspecified_scope(e06: list[MetricExtraction]) -> None:
    """A bare year has no stated scope: label is None, never assumed to be whole-entity."""
    aladin = next(r for r in rank_banks(e06).ranked if "Aladin" in r.bank)
    assert aladin.label is None
    assert aladin.raw_value == "2040"


# --- not_disclosed: excluded with a reason, never a sentinel that sorts -----------------


def test_not_disclosed_banks_are_excluded_with_reasons_not_sorted(
    e06: list[MetricExtraction],
) -> None:
    result = rank_banks(e06)

    assert len(result.excluded) == 7
    excluded_banks = {bank.bank for bank in result.excluded}
    assert "PT Bank Jago Tbk" in excluded_banks
    assert excluded_banks.isdisjoint({row.bank for row in result.ranked})
    # Every one was actually read by the model, so none is a retrieval gap.
    assert all(bank.reason is ExclusionReason.NOT_DISCLOSED_IN_REPORT for bank in result.excluded)
    # No sentinel leaked in: nothing ranks at 0 or a far-future year.
    assert all(2040.0 <= row.value <= 2060.0 for row in result.ranked)
    assert result.bank_count == 21  # the denominator survives


def test_retrieval_gap_is_distinguished_from_a_real_absence() -> None:
    """Runs the real 6.5.4 extractor so the note-text coupling cannot drift silently."""
    metric = MetricSpec(
        code="E-06",
        description="Net-zero target year.",
        value_type="year",
        topic_keywords=["net zero"],
    )
    empty = BankMetricEvidence(
        bank="PT Bank X",
        source_name="PT Bank X — 2025 Sustainability Report",
        found=False,
        top_score=0.0,
        fallback_used=False,
        evidence=EvidenceResult(
            item_code="E-06", query="q", results=[], top_score=0.0, fallback_used=False
        ),
    )

    def must_not_call(system: str, user: str) -> str:
        raise AssertionError("model called for a found=False bank")

    skipped = extract_metric_value(metric, empty, model=must_not_call)
    result = rank_banks([skipped])

    assert not result.ranked
    assert result.excluded[0].reason is ExclusionReason.NOT_DISCLOSED_RETRIEVAL_GAP


# --- filtering -------------------------------------------------------------------------


def test_filter_before_2050_reports_matched_and_unmatched(e06: list[MetricExtraction]) -> None:
    result = filter_banks(e06, lambda year: year < 2050)

    assert [row.bank for row in result.matched] == ["PT Bank Aladin Syariah Tbk"]
    assert len(result.unmatched) == 9  # the other readable banks, not discarded
    assert len(result.ambiguous) == 4
    assert len(result.excluded) == 7
    assert (
        len(result.matched) + len(result.unmatched) + len(result.ambiguous) + len(result.excluded)
        == result.bank_count
    )


def test_filter_under_explode_picks_up_the_operations_years(e06: list[MetricExtraction]) -> None:
    result = filter_banks(e06, lambda year: year < 2050, explode=True)

    assert {row.bank for row in result.matched} == {
        "PT Bank Aladin Syariah Tbk",
        "PT Bank Negara Indonesia (Persero) Tbk",
        "PT Bank Mandiri (Persero) Tbk",
        "PT Bank CIMB Niaga Tbk",
        "PT Bank SMBC Indonesia Tbk",
    }
    assert not result.ambiguous


# --- arithmetic ------------------------------------------------------------------------


def test_mean_training_hours_across_all_21_banks(s05: list[MetricExtraction]) -> None:
    result = mean(s05)

    assert result.n_included == 21
    assert result.bank_count == 21
    assert result.value == pytest.approx(51.94)
    assert not result.excluded and not result.ambiguous


def test_total_training_hours_across_all_21_banks(s05: list[MetricExtraction]) -> None:
    result = total(s05)

    assert result.value == pytest.approx(1090.74)
    assert result.n_included == 21


def test_inline_units_are_stripped_from_real_values(s05: list[MetricExtraction]) -> None:
    """Four banks report '<n> hours/employee' inline; they must still be arithmetic-eligible."""
    rows = {row.bank: row for row in rank_banks(s05).ranked}
    jago = rows["PT Bank Jago Tbk"]
    assert jago.raw_value == "47 hours/employee"  # preserved verbatim
    assert jago.value == 47.0  # and parsed
    assert rows["PT Bank Mestika Dharma Tbk"].value == 0.0  # a real zero, not a missing value


def test_mean_carries_its_denominator_when_banks_are_excluded(e06: list[MetricExtraction]) -> None:
    """A mean over part of the corpus must not look like a corpus average."""
    result = mean(e06)

    assert result.value == pytest.approx(2055.0)
    assert result.n_included == 10  # not 21
    assert result.bank_count == 21
    assert len(result.ambiguous) == 4
    assert len(result.excluded) == 7


def test_sum_of_years_is_rejected(e06: list[MetricExtraction]) -> None:
    with pytest.raises(ValueError, match="not meaningful for value_type 'year'"):
        total(e06)


def test_multi_value_banks_are_excluded_from_arithmetic_by_default(
    e06: list[MetricExtraction],
) -> None:
    """Including both of a bank's years would count that bank twice."""
    default_n = mean(e06).n_included
    exploded_n = mean(e06, explode=True).n_included

    assert default_n == 10
    assert exploded_n == 18  # 10 + 4 banks x 2 scopes


# --- number parsing: the corrected separator rule, on real E-02 strings -----------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0,0018", 0.0018),  # BRI / MNC emissions intensity
        ("1,37", 1.37),  # BCA / Aladin emissions intensity
        ("8.216,29", 8216.29),  # MNC water, dot thousands AND comma decimal in one value
        ("2.500.000", 2_500_000.0),  # Panin electricity
        ("193.043.553", 193_043_553.0),  # BCA electricity
        ("1.44", 1.44),  # lone dot, 2 trailing digits -> cannot be a thousands separator
        ("2060", 2060.0),
        ("0", 0.0),
        ("47 hours/employee", 47.0),
        ("98,15 %", 98.15),
    ],
)
def test_parses_real_corpus_values(raw: str, expected: float) -> None:
    value, reason = parse_metric_number(raw)
    assert reason is None
    assert value == pytest.approx(expected)


def test_lone_dot_before_three_digits_is_refused_not_guessed() -> None:
    """Panin's real E-02 value: 5813 under the thousands convention, or 5.813 as a decimal."""
    value, reason = parse_metric_number("5.813")

    assert value is None
    assert reason is not None and "ambiguous" in reason


def test_unrecognised_trailing_text_is_refused() -> None:
    value, reason = parse_metric_number("1.44 tCO2e/IDR billion")

    assert value is None
    assert reason is not None and "unrecognised trailing text" in reason


def test_unparseable_value_becomes_an_explicit_exclusion(e06: list[MetricExtraction]) -> None:
    """A refused value is surfaced as `unparseable`, never silently dropped from the corpus."""
    ambiguous_bank = MetricExtraction(
        bank="PT Bank Panin Dubai Syariah Tbk",
        metric_code="E-02",
        value_type="numeric",
        status=ExtractionStatus.DISCLOSED,
        values=(ExtractedValue(value="5.813", label=None),),
        citation="PT Bank Panin Dubai Syariah Tbk — 2025 Sustainability Report, p. 13",
        note=None,
    )

    result = rank_banks([ambiguous_bank])

    assert not result.ranked
    assert result.excluded[0].reason is ExclusionReason.UNPARSEABLE
    assert result.bank_count == 1  # still counted in the denominator


def test_ties_break_deterministically_on_bank_name(e06: list[MetricExtraction]) -> None:
    ranked = rank_banks(e06).ranked
    tied = [row.bank for row in ranked if row.value == 2050.0]
    assert tied == sorted(tied)
