"""Cross-bank checklist sweep: run one checklist against every bank in the corpus."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI
from qdrant_client import QdrantClient

from app.core.config import Settings
from app.rag.compliance.checklist import ChecklistItem
from app.rag.compliance.llm_assessor import (
    JudgeBatch,
    PendingAssessment,
    assess_checklist_with_llm,
    llm_assess_batch,
)
from app.rag.compliance.report import (
    DISCLAIMER,
    STATUS_NOTE,
    ComplianceReport,
    RequirementResult,
    Status,
    StatusAssessor,
    assess_checklist,
    default_assess_status,
)
from app.rag.compliance.resolver import all_company_reports
from app.rag.compliance.retrieval import (
    DEFAULT_LIMIT,
    DEFAULT_SCORE_THRESHOLD,
    EvidenceResult,
    retrieve_evidence,
)

# An extra caveat the single-report path does not need: a cross-bank table invites reading
# the ordering as a ranking of ESG quality, which it is not.
SWEEP_NOTE = (
    "Statuses compare how each report addresses a requirement, not how well each bank "
    "performs. A bank may address a topic and report an unfavourable answer, and reports "
    "differ in structure and length in ways that affect retrieval."
)

# One bank's evidence retrieval for one requirement: (source_name, item) -> evidence.
PerBankRetrieve = Callable[[str, ChecklistItem], EvidenceResult]


@dataclass(frozen=True)
class BankSweepResult:
    bank: str
    source_name: str
    report: ComplianceReport


@dataclass(frozen=True)
class BankStatus:
    """One bank's outcome for one requirement, with its retrieval provenance."""

    bank: str
    status: Status
    top_score: float
    citation: str | None
    fallback_used: bool


@dataclass(frozen=True)
class RequirementSweep:
    """One requirement, pivoted across banks into its status groups (partitions every bank)."""

    code: str
    description: str
    addressed: tuple[BankStatus, ...]
    partial: tuple[BankStatus, ...]
    not_found: tuple[BankStatus, ...]
    not_checkable: tuple[BankStatus, ...]
    bank_count: int


@dataclass(frozen=True)
class BankCoverage:
    """How many of the checklist's requirements one bank addressed, for ranking."""

    bank: str
    addressed: int
    partial: int
    not_found: int
    not_checkable: int
    requirement_count: int


@dataclass(frozen=True)
class ChecklistSweep:
    """A whole checklist run against every bank, plus the caveats that must travel with it."""

    category: str
    codes: tuple[str, ...]
    banks: tuple[BankSweepResult, ...]
    disclaimer: str
    status_note: str
    sweep_note: str
    # Codes answered by a coded/numeric row; ranking is refused for these, see `rank_requirement`.
    anchored_codes: frozenset[str] = frozenset()
    # True when every bank's status came from the LLM content-reading assessor rather than
    # the raw retrieval-score band; `rank_requirement` reads this to decide refusal.
    calibrated: bool = False

    @property
    def bank_count(self) -> int:
        return len(self.banks)

    @property
    def requirement_count(self) -> int:
        return len(self.codes)


def sweep_checklist(
    checklist: Sequence[ChecklistItem],
    banks: Sequence[tuple[str, str]],
    retrieve: PerBankRetrieve,
    *,
    category: str,
    assess: StatusAssessor = default_assess_status,
) -> ChecklistSweep:
    results: list[BankSweepResult] = []
    for bank, source_name in banks:

        def scoped(item: ChecklistItem, _source: str = source_name) -> EvidenceResult:
            return retrieve(_source, item)

        results.append(
            BankSweepResult(
                bank=bank,
                source_name=source_name,
                report=assess_checklist(
                    checklist,
                    report_name=bank,
                    category=category,
                    retrieve=scoped,
                    assess=assess,
                ),
            )
        )

    return ChecklistSweep(
        category=category,
        codes=tuple(item.code for item in checklist),
        banks=tuple(results),
        disclaimer=DISCLAIMER,
        status_note=STATUS_NOTE,
        sweep_note=SWEEP_NOTE,
        anchored_codes=frozenset(item.code for item in checklist if item.row_anchors),
    )


def sweep_checklist_across_banks(
    checklist: Sequence[ChecklistItem],
    *,
    category: str,
    qdrant_client: QdrantClient,
    embed_client: OpenAI | None = None,
    settings: Settings | None = None,
    collection_name: str | None = None,
    assess: StatusAssessor = default_assess_status,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    limit: int = DEFAULT_LIMIT,
    manifest_path: Path | None = None,
) -> ChecklistSweep:
    """Sweep `checklist` across every bank in the manifest (the production wiring)."""
    banks = all_company_reports(manifest_path=manifest_path)

    def retrieve(source_name: str, item: ChecklistItem) -> EvidenceResult:
        return retrieve_evidence(
            item,
            qdrant_client=qdrant_client,
            source_name=source_name,
            embed_client=embed_client,
            settings=settings,
            collection_name=collection_name,
            score_threshold=score_threshold,
            limit=limit,
            row_anchors=item.row_anchors,
        )

    return sweep_checklist(checklist, banks, retrieve, category=category, assess=assess)


def sweep_checklist_with_llm(
    checklist: Sequence[ChecklistItem],
    banks: Sequence[tuple[str, str]],
    retrieve: PerBankRetrieve,
    *,
    category: str,
    judge: JudgeBatch = llm_assess_batch,
) -> ChecklistSweep:
    """Run `checklist` against every bank via the LLM assessor, not the raw score band."""
    results: list[BankSweepResult] = []
    for bank, source_name in banks:

        def scoped(item: ChecklistItem, _source: str = source_name) -> EvidenceResult:
            return retrieve(_source, item)

        results.append(
            BankSweepResult(
                bank=bank,
                source_name=source_name,
                report=assess_checklist_with_llm(
                    checklist,
                    report_name=bank,
                    category=category,
                    retrieve=scoped,
                    judge=judge,
                ),
            )
        )

    return ChecklistSweep(
        category=category,
        codes=tuple(item.code for item in checklist),
        banks=tuple(results),
        disclaimer=DISCLAIMER,
        status_note=STATUS_NOTE,
        sweep_note=SWEEP_NOTE,
        anchored_codes=frozenset(item.code for item in checklist if item.row_anchors),
        calibrated=True,
    )


def sweep_checklist_across_banks_with_llm(
    checklist: Sequence[ChecklistItem],
    *,
    category: str,
    qdrant_client: QdrantClient,
    embed_client: OpenAI | None = None,
    openai_client: OpenAI | None = None,
    settings: Settings | None = None,
    collection_name: str | None = None,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    limit: int = DEFAULT_LIMIT,
    manifest_path: Path | None = None,
) -> ChecklistSweep:
    """Sweep `checklist` across every bank using the LLM assessor (the production wiring)."""
    banks = all_company_reports(manifest_path=manifest_path)

    def retrieve(source_name: str, item: ChecklistItem) -> EvidenceResult:
        return retrieve_evidence(
            item,
            qdrant_client=qdrant_client,
            source_name=source_name,
            embed_client=embed_client,
            settings=settings,
            collection_name=collection_name,
            score_threshold=score_threshold,
            limit=limit,
            row_anchors=item.row_anchors,
        )

    def judge(pending: Sequence[PendingAssessment]) -> dict[str, Status]:
        return llm_assess_batch(pending, client=openai_client, settings=settings)

    return sweep_checklist_with_llm(checklist, banks, retrieve, category=category, judge=judge)


@dataclass(frozen=True)
class RankingRouted:
    """Returned instead of a ranking when the requirement is not rankable by this pipeline."""

    code: str
    reason: str
    message: str
    route_to: str


def _refusal(
    code: str,
    *,
    anchored: bool,
    all_not_checkable: bool,
    bank_count: int,
    calibrated: bool = False,
) -> RankingRouted | None:
    """The single ranking-refusal decision, shared by both entry points so they can't diverge."""
    if anchored:
        return RankingRouted(
            code=code,
            reason="coded_numeric_requirement",
            route_to="metrics",
            message=(
                f"{code} is answered by a coded disclosure-index row that every bank in this "
                f"corpus files, so a compliance ranking would report all {bank_count} banks "
                "identically — that is a real fact about the filings, not a comparison. The "
                "figures that separate the banks are in the row's value; use the metric "
                "aggregation pipeline to extract and rank those."
            ),
        )

    if all_not_checkable:
        return RankingRouted(
            code=code,
            reason="not_directly_disclosed",
            route_to="none",
            message=(
                f"{code} is not disclosed as a separate line item anywhere in this corpus, "
                "so there is nothing to rank. It was not searched, and no bank should be "
                "read as failing to disclose it."
            ),
        )

    if calibrated:
        # Status came from the LLM content-reading assessor, not the uncalibrated score
        # band the refusal below exists for. Nothing to refuse.
        return None

    # Unanchored, uncalibrated: the status comes from a score band shown to track proximity
    # to threshold rather than disclosure, refused so a ranked table doesn't read as a finding.
    return RankingRouted(
        code=code,
        reason="unverified_ranking_output",
        route_to="none",
        message=(
            f"{code} has no coded row to anchor on, so its status comes from a retrieval "
            "score band that is not calibrated for this corpus. Measured across all 21 "
            "banks, whether a requirement appears to discriminate tracks only whether its "
            "score cluster straddles the 0.50 threshold, not how the reports differ — so a "
            "ranking here would be unverified output, not a finding. Deferred to Stage 8: "
            "calibrate or replace the assessor before trusting ranking output on unanchored "
            "requirements. The per-bank statuses remain available via requirement_view for "
            "inspection, clearly marked as unverified."
        ),
    )


def ranking_refusal_for_item(
    item: ChecklistItem, *, bank_count: int, calibrated: bool = True
) -> RankingRouted | None:
    """Decide refusal from the requirement alone, before spending a sweep."""
    return _refusal(
        item.code,
        anchored=bool(item.row_anchors),
        all_not_checkable=not item.directly_disclosed,
        bank_count=bank_count,
        calibrated=calibrated,
    )


def rank_requirement(sweep: ChecklistSweep, code: str) -> RequirementSweep | RankingRouted:
    """Rank banks for `code`, or refuse and route when a ranking would mislead."""
    view = requirement_view(sweep, code)
    refusal = _refusal(
        code,
        anchored=code in sweep.anchored_codes,
        all_not_checkable=bool(view.bank_count) and len(view.not_checkable) == view.bank_count,
        bank_count=sweep.bank_count,
        calibrated=sweep.calibrated,
    )
    return refusal if refusal is not None else view


def _requirement_row(result: BankSweepResult, code: str) -> tuple[RequirementResult, str]:
    """Find one requirement's row in a bank's report, with the bank name."""
    for row in result.report.results:
        if row.code == code:
            return row, result.bank
    raise KeyError(f"requirement {code!r} is not in this sweep's checklist")


def requirement_view(sweep: ChecklistSweep, code: str) -> RequirementSweep:
    """Pivot one requirement across every bank into status groups, ordered by score then name."""
    if code not in sweep.codes:
        raise KeyError(f"requirement {code!r} is not in this sweep's checklist")

    grouped: dict[Status, list[BankStatus]] = {status: [] for status in Status}
    description = ""
    for result in sweep.banks:
        row, bank = _requirement_row(result, code)
        description = description or row.description
        grouped[row.status].append(
            BankStatus(
                bank=bank,
                status=row.status,
                top_score=row.top_score,
                citation=row.citation,
                fallback_used=row.fallback_used,
            )
        )

    def ordered(status: Status) -> tuple[BankStatus, ...]:
        return tuple(sorted(grouped[status], key=lambda b: (-b.top_score, b.bank)))

    return RequirementSweep(
        code=code,
        description=description,
        addressed=ordered(Status.DISCLOSED),
        partial=ordered(Status.PARTIALLY_DISCLOSED),
        not_found=ordered(Status.NOT_FOUND),
        not_checkable=ordered(Status.NOT_DIRECTLY_DISCLOSED),
        bank_count=sweep.bank_count,
    )


def banks_with_status(sweep: ChecklistSweep, code: str, status: Status) -> tuple[BankStatus, ...]:
    """The banks whose outcome for `code` is `status`, e.g. which banks fail GRI 305-1."""
    view = requirement_view(sweep, code)
    return {
        Status.DISCLOSED: view.addressed,
        Status.PARTIALLY_DISCLOSED: view.partial,
        Status.NOT_FOUND: view.not_found,
        Status.NOT_DIRECTLY_DISCLOSED: view.not_checkable,
    }[status]


def rank_banks_by_coverage(
    sweep: ChecklistSweep, *, descending: bool = True
) -> tuple[BankCoverage, ...]:
    """Rank banks by how many of the checklist's requirements they address (not ESG quality)."""
    coverage = [
        BankCoverage(
            bank=result.bank,
            addressed=sum(1 for r in result.report.results if r.status is Status.DISCLOSED),
            partial=sum(1 for r in result.report.results if r.status is Status.PARTIALLY_DISCLOSED),
            not_found=sum(1 for r in result.report.results if r.status is Status.NOT_FOUND),
            not_checkable=sum(
                1 for r in result.report.results if r.status is Status.NOT_DIRECTLY_DISCLOSED
            ),
            requirement_count=len(result.report.results),
        )
        for result in sweep.banks
    ]
    return tuple(
        sorted(
            coverage,
            key=lambda c: (
                (-c.addressed, -c.partial, c.bank)
                if descending
                else (c.addressed, c.partial, c.bank)
            ),
        )
    )
