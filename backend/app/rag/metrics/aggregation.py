"""Enumerate-all-banks retriever: step one of the metric aggregation pipeline (6.5.3)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI
from qdrant_client import QdrantClient

from app.core.config import Settings
from app.rag.compliance.resolver import all_company_reports
from app.rag.compliance.retrieval import (
    DEFAULT_LIMIT,
    DEFAULT_SCORE_THRESHOLD,
    EvidenceResult,
    retrieve_evidence,
)
from app.rag.metrics.metric import MetricSpec

# A per-bank retrieval: given the metric and one bank's source_name, return its evidence.
PerBankRetrieve = Callable[[MetricSpec, str], EvidenceResult]


@dataclass(frozen=True)
class BankMetricEvidence:
    """Raw retrieval evidence for one metric against one bank's report (no value parsed yet)."""

    bank: str
    source_name: str
    found: bool
    top_score: float
    fallback_used: bool
    evidence: EvidenceResult


def enumerate_bank_evidence(
    metric: MetricSpec,
    banks: Sequence[tuple[str, str]],
    retrieve: PerBankRetrieve,
    *,
    found_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> list[BankMetricEvidence]:
    """Run `retrieve` for `metric` against every `(bank, source_name)` and classify each result."""
    results: list[BankMetricEvidence] = []
    for bank, source_name in banks:
        evidence = retrieve(metric, source_name)
        found = bool(evidence.results) and evidence.top_score >= found_threshold
        results.append(
            BankMetricEvidence(
                bank=bank,
                source_name=source_name,
                found=found,
                top_score=evidence.top_score,
                fallback_used=evidence.fallback_used,
                evidence=evidence,
            )
        )
    return results


def retrieve_metric_across_banks(
    metric: MetricSpec,
    *,
    qdrant_client: QdrantClient,
    embed_client: OpenAI | None = None,
    settings: Settings | None = None,
    collection_name: str | None = None,
    found_threshold: float = DEFAULT_SCORE_THRESHOLD,
    limit: int = DEFAULT_LIMIT,
    manifest_path: Path | None = None,
) -> list[BankMetricEvidence]:
    """Retrieve `metric` evidence for every bank in the manifest (the production wiring)."""
    banks = all_company_reports(manifest_path=manifest_path)

    def retrieve(item: MetricSpec, source_name: str) -> EvidenceResult:
        return retrieve_evidence(
            item,
            qdrant_client=qdrant_client,
            source_name=source_name,
            embed_client=embed_client,
            settings=settings,
            collection_name=collection_name,
            score_threshold=found_threshold,
            limit=limit,
        )

    return enumerate_bank_evidence(metric, banks, retrieve, found_threshold=found_threshold)
