from __future__ import annotations

import json
from pathlib import Path

from app.models.payload import ChunkPayload, ContentType, Domain
from app.rag.compliance.resolver import all_company_reports
from app.rag.compliance.retrieval import EvidenceResult
from app.rag.metrics.aggregation import enumerate_bank_evidence
from app.rag.metrics.metric import MetricSpec
from app.rag.vector_store import SearchResult

_METRICS_FILE = Path("app/rag/metrics/company_metrics.json")


def _load_real_metric(code: str) -> MetricSpec:
    """Load one real MetricSpec from the committed metric set (filtering `source`)."""
    data = json.loads(_METRICS_FILE.read_text(encoding="utf-8"))
    entry = next(m for m in data["metrics"] if m["code"] == code)
    fields = set(MetricSpec.model_fields)
    return MetricSpec(**{k: v for k, v in entry.items() if k in fields})


def _evidence(score: float, *, has_result: bool = True) -> EvidenceResult:
    results: list[SearchResult] = []
    if has_result:
        payload = ChunkPayload(
            domain=Domain.COMPANY_DOCUMENTS,
            source_name="s",
            document_type="sustainability_report",
            page_number=13,
            content_type=ContentType.TABLE,
            chunk_text="net zero target year 2050",
        )
        results = [SearchResult(id="1", score=score, payload=payload)]
    return EvidenceResult(
        item_code="E-06",
        query="q",
        results=results,
        top_score=score if has_result else 0.0,
        fallback_used=False,
    )


def test_runs_across_all_21_banks_and_returns_exactly_21_results() -> None:
    metric = _load_real_metric("E-06")
    banks = all_company_reports()  # the real manifest enumeration
    assert len(banks) == 21  # the corpus has 21 company reports (BTPN/SMBC deduped)

    # Fake retrieval: first bank strong (found), second below threshold (not found),
    # every other bank empty (not found), a realistic mix (net zero is often absent).
    scores = {banks[0][1]: 0.62, banks[1][1]: 0.20}

    def fake_retrieve(m: MetricSpec, source_name: str) -> EvidenceResult:
        if source_name in scores:
            return _evidence(scores[source_name])
        return _evidence(0.0, has_result=False)

    results = enumerate_bank_evidence(metric, banks, fake_retrieve, found_threshold=0.35)

    # Exactly one result per bank, in manifest order, no bank skipped or duplicated.
    assert len(results) == 21
    assert [r.bank for r in results] == [b[0] for b in banks]
    assert len({r.bank for r in results}) == 21

    # Every result, found or not, carries raw evidence; none is silently dropped.
    assert all(r.evidence is not None for r in results)

    # Classification: strong -> found; below-threshold -> not found; empty -> not found.
    assert results[0].found is True
    assert results[0].top_score == 0.62
    assert results[1].found is False  # 0.20 is below the 0.35 threshold
    assert all(r.found is False for r in results[2:])  # empty retrieval
    assert all(len(r.evidence.results) == 0 for r in results[2:])

    # Exactly one bank resolved as "found" here.
    assert sum(1 for r in results if r.found) == 1


def test_empty_and_below_threshold_are_both_explicit_not_found() -> None:
    metric = _load_real_metric("E-06")
    banks = [
        ("Bank A", "A — 2025 Sustainability Report"),
        ("Bank B", "B — 2025 Sustainability Report"),
    ]

    def retrieve(m: MetricSpec, source_name: str) -> EvidenceResult:
        return _evidence(0.10) if source_name.startswith("A") else _evidence(0.0, has_result=False)

    results = enumerate_bank_evidence(metric, banks, retrieve, found_threshold=0.35)
    assert len(results) == 2
    # Bank A: a result came back but scored too low. Bank B: nothing came back. Both not found.
    assert results[0].found is False and len(results[0].evidence.results) == 1
    assert results[1].found is False and len(results[1].evidence.results) == 0


def test_score_at_threshold_counts_as_found() -> None:
    metric = _load_real_metric("E-06")
    banks = [("Bank A", "A — 2025 Sustainability Report")]
    results = enumerate_bank_evidence(
        metric, banks, lambda m, s: _evidence(0.35), found_threshold=0.35
    )
    assert results[0].found is True  # >= threshold, inclusive


def test_all_company_reports_reconstructs_real_source_names() -> None:
    reports = dict(all_company_reports())
    assert len(reports) == 21
    # A resolved source_name matches the byte-for-byte value used everywhere else.
    assert (
        reports["PT Bank Central Asia Tbk"]
        == "PT Bank Central Asia Tbk — 2025 Sustainability Report"
    )
