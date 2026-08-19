from __future__ import annotations

from app.models.payload import ChunkPayload, ContentType, Domain
from app.rag.compliance.retrieval import EvidenceResult
from app.rag.metrics.aggregation import BankMetricEvidence
from app.rag.metrics.extraction import (
    ExtractionStatus,
    extract_metric_across_banks,
    extract_metric_value,
)
from app.rag.metrics.metric import MetricSpec
from app.rag.vector_store import SearchResult

METRIC = MetricSpec(
    code="E-06",
    description="Net-zero target year: the year the bank commits to reach net-zero.",
    value_type="year",
    topic_keywords=["net zero", "net-zero target year"],
)

# --- real evidence text from the live 6.5.3 run -----------------------------

JAGO_EVIDENCE = (
    "| E-06 Company commitment to Net Zero Emission Target |  |\n"
    "| --- | --- |\n"
    "| Does the Company have a net zero emission target commitment? | No |\n"
    "| What year is the Company’s net zero emission published target? | null |\n"
    "| Please provide a brief description and a link to documentation explaining the "
    "Company’s commitment in achieving net zero emission target. |  |\n"
    "| Bank Jago will continuously enhance its approach to climate-related considerations "
    "and strengthen climate risk management. ... Moving forward, the Bank will continue to "
    "refine its methodology in line with industry developments. |  |"
)

BNI_EVIDENCE = (
    "| E-06 Komitmen Perusahaan untuk mencapai target Net Zero Emission |  |\n"
    "| --- | --- |\n"
    "| Apakah Perusahaan memiliki komitmen pencapaian target net zero? | Ya |\n"
    "| Tahun berapa Perusahaan menargetkan pencapaian Net Zero emission yang dipublikasi? "
    "| 2060 |\n"
    "| ... | BNI berkomitmen mendukung pencapaian target Net Zero Emission (NZE) nasional "
    "... Sejalan dengan target NZE Indonesia pada tahun 2060 atau lebih cepat, BNI telah "
    "menetapkan target Net Zero Emission Operasional pada tahun 2028 dan Net Zero Emission "
    "Pembiayaan pada tahun 2060. ... |  |"
)


def _evidence(
    bank: str, text: str, *, found: bool = True, score: float = 0.6
) -> BankMetricEvidence:
    source_name = f"{bank} — 2025 Sustainability Report"
    results: list[SearchResult] = []
    if text:
        payload = ChunkPayload(
            domain=Domain.COMPANY_DOCUMENTS,
            source_name=source_name,
            document_type="sustainability_report",
            page_number=14,
            content_type=ContentType.TABLE,
            chunk_text=text,
        )
        results = [SearchResult(id="1", score=score, payload=payload)]
    ev = EvidenceResult(
        item_code="E-06",
        query="q",
        results=results,
        top_score=score if results else 0.0,
        fallback_used=False,
    )
    return BankMetricEvidence(
        bank=bank,
        source_name=source_name,
        found=found,
        top_score=score if results else 0.0,
        fallback_used=False,
        evidence=ev,
    )


# --- Jago: found, but the block says "No" / year null -> not disclosed -------


def test_jago_real_snippet_extracts_not_disclosed() -> None:
    evidence = _evidence("PT Bank Jago Tbk", JAGO_EVIDENCE)

    # The real model, faithfully reading Jago's "No"/null, returns no value.
    def stub(system: str, user: str) -> str:
        return '{"values": [], "note": "Report answers \\"No\\"; target year is null."}'

    result = extract_metric_value(METRIC, evidence, model=stub)

    assert result.status is ExtractionStatus.NOT_DISCLOSED
    assert result.values == ()
    assert result.note is not None  # carries why it is not disclosed
    assert result.value_type == "year"
    assert result.citation == "PT Bank Jago Tbk — 2025 Sustainability Report, p. 14 (table)"


# --- BNI: found, states two scoped years -> multiple disclosed ---------------


def test_bni_real_snippet_surfaces_both_years_as_multiple_disclosed() -> None:
    evidence = _evidence("PT Bank Negara Indonesia (Persero) Tbk", BNI_EVIDENCE)

    def stub(system: str, user: str) -> str:
        return (
            '{"values": [{"value": "2028", "label": "Operational NZE"}, '
            '{"value": "2060", "label": "Financing NZE"}], "note": null}'
        )

    result = extract_metric_value(METRIC, evidence, model=stub)

    # The ambiguity is surfaced, not silently collapsed to one year.
    assert result.status is ExtractionStatus.MULTIPLE_DISCLOSED
    assert len(result.values) == 2
    assert {v.value for v in result.values} == {"2028", "2060"}
    labels = {v.value: v.label for v in result.values}
    assert labels["2028"] == "Operational NZE"
    assert labels["2060"] == "Financing NZE"


# --- found=False: skip the LLM entirely --------------------------------------


def test_found_false_skips_llm_and_is_not_disclosed() -> None:
    evidence = _evidence("PT Bank X", "", found=False)

    def must_not_call(system: str, user: str) -> str:
        raise AssertionError("model was called for a found=False bank")

    result = extract_metric_value(METRIC, evidence, model=must_not_call)

    assert result.status is ExtractionStatus.NOT_DISCLOSED
    assert result.values == ()
    assert result.citation is None
    assert "found=False" in (result.note or "")


# --- single value -> disclosed ----------------------------------------------


def test_single_value_is_disclosed() -> None:
    evidence = _evidence("PT Bank Y", "| E-06 ... | Yes | 2050 |")

    def stub(system: str, user: str) -> str:
        return '{"values": [{"value": "2050", "label": null}], "note": null}'

    result = extract_metric_value(METRIC, evidence, model=stub)

    assert result.status is ExtractionStatus.DISCLOSED
    assert len(result.values) == 1
    assert result.values[0].value == "2050"
    assert result.values[0].label is None


def test_empty_values_with_a_note_is_not_disclosed() -> None:
    # Even if the model volunteers a note, no value means not_disclosed.
    evidence = _evidence("PT Bank Z", "some evidence")

    def stub(system: str, user: str) -> str:
        return '{"values": [], "note": "no target stated"}'

    result = extract_metric_value(METRIC, evidence, model=stub)
    assert result.status is ExtractionStatus.NOT_DISCLOSED


# --- across banks: one result per bank, LLM only for found banks -------------


def test_extract_across_banks_skips_not_found_and_counts_calls() -> None:
    evidences = [
        _evidence("Bank A", "| ... | Yes | 2050 |"),  # found -> 1 call
        _evidence("Bank B", "", found=False),  # not found -> 0 calls
        _evidence("Bank C", "| ... | Yes | 2045 |"),  # found -> 1 call
    ]
    calls = {"n": 0}

    def counting_stub(system: str, user: str) -> str:
        calls["n"] += 1
        return '{"values": [{"value": "2050", "label": null}], "note": null}'

    results = extract_metric_across_banks(METRIC, evidences, model=counting_stub)

    assert [r.bank for r in results] == ["Bank A", "Bank B", "Bank C"]  # one per bank
    assert calls["n"] == 2  # only the two found banks cost a call
    assert results[0].status is ExtractionStatus.DISCLOSED
    assert results[1].status is ExtractionStatus.NOT_DISCLOSED  # skipped
    assert results[2].status is ExtractionStatus.DISCLOSED
