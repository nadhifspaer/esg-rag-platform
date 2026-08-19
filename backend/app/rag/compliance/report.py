"""Compliance report assembly: run a checklist and format the structured result."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from app.models.evidence_text import truncate_snippet
from app.models.payload import ChunkPayload, ContentType
from app.rag.compliance.anchors import find_anchor_row
from app.rag.compliance.checklist import ChecklistItem
from app.rag.compliance.retrieval import EvidenceResult

DISCLAIMER = (
    "This is an automated document-comparison aid, not a certified compliance audit. "
    "Results come from automated retrieval over the report's text and may miss "
    "disclosures or misjudge completeness — verify against the source document before "
    "relying on them."
)

# One-line clarifier: status says whether the report addresses a topic, not whether the
# answer is favorable.
STATUS_NOTE = (
    "Status reflects whether the report addresses each topic, not whether the reported "
    "answer is favorable."
)

# Status score bands for the default heuristic assessor (top cosine similarity). Coarse
# retrieval-confidence proxies, to be calibrated against the eval set, not exact.
DISCLOSED_SCORE = 0.50
PARTIAL_SCORE = 0.35

# Content-type annotations for citations (tables/figures read differently from prose).
_CONTENT_TYPE_SUFFIX: dict[ContentType, str] = {
    ContentType.TEXT: "",
    ContentType.TABLE: " (table)",
    ContentType.CHART_CAPTION: " (figure)",
}


class Status(StrEnum):
    """A requirement's disclosure status in the checked report."""

    DISCLOSED = "disclosed"
    PARTIALLY_DISCLOSED = "partially disclosed"
    NOT_FOUND = "not found"
    # No row in this corpus answers the requirement at all; set from the checklist's
    # curated `directly_disclosed=False`, never inferred.
    NOT_DIRECTLY_DISCLOSED = "not directly disclosed"


# Reader-facing labels. DISCLOSED shows as "Addressed", not "Disclosed", to avoid a
# reader misreading it as "the company has/does the thing".
_STATUS_DISPLAY: dict[Status, str] = {
    Status.DISCLOSED: "✅ Addressed",
    Status.PARTIALLY_DISCLOSED: "🟡 Partial",
    Status.NOT_FOUND: "❌ Not found",
    # Neutral on purpose: this says something about our corpus, not about the bank.
    Status.NOT_DIRECTLY_DISCLOSED: "➖ Not separately disclosed in this corpus",
}

# Statuses where evidence was retrieved but judged not substantively responsive, as
# opposed to DISCLOSED (accepted) or NOT_DIRECTLY_DISCLOSED (never retrieved at all).
_INSUBSTANTIAL_STATUSES = frozenset({Status.NOT_FOUND, Status.PARTIALLY_DISCLOSED})


@dataclass(frozen=True)
class RequirementResult:
    """One row of the compliance table: a requirement and how the report addressed it."""

    code: str
    description: str
    status: Status
    evidence_snippet: str | None
    citation: str | None
    top_score: float
    fallback_used: bool
    rejected_as_insubstantial: bool = False


StatusAssessor = Callable[[ChecklistItem, EvidenceResult], Status]


def default_assess_status(item: ChecklistItem, evidence: EvidenceResult) -> Status:
    """Status from the requirement's anchored row when it has one, else from the score bands."""
    anchors = getattr(item, "row_anchors", ())
    if anchors:
        if not evidence.results:
            return Status.NOT_FOUND
        match = find_anchor_row(evidence.results[0].payload.chunk_text, anchors)
        if match is None:
            return Status.NOT_FOUND
        return Status.DISCLOSED if match.has_value else Status.PARTIALLY_DISCLOSED

    if not evidence.results or evidence.top_score < PARTIAL_SCORE:
        return Status.NOT_FOUND
    if evidence.top_score >= DISCLOSED_SCORE:
        return Status.DISCLOSED
    return Status.PARTIALLY_DISCLOSED


def format_evidence_citation(payload: ChunkPayload) -> str:
    """Build a citation string for an evidence chunk: source_name, page, content-type annotation."""
    suffix = _CONTENT_TYPE_SUFFIX.get(payload.content_type, "")
    page = payload.page_number
    return f"{payload.source_name}, p. {page}{suffix}"


def _cell(text: str) -> str:
    """Escape a value for a markdown table cell (pipes break the table; tables have them)."""
    return re.sub(r"\s+", " ", text).replace("|", "\\|").strip()


def _summarize(results: Sequence[RequirementResult], report_name: str, category: str) -> str:
    """A short, deterministic natural-language summary from the status counts."""
    counts = Counter(r.status for r in results)
    checked = len(results) - counts[Status.NOT_DIRECTLY_DISCLOSED]
    summary = (
        f"Of {checked} {category} disclosure requirements checked against "
        f"{report_name}, {counts[Status.DISCLOSED]} addressed, "
        f"{counts[Status.PARTIALLY_DISCLOSED]} partially addressed, and "
        f"{counts[Status.NOT_FOUND]} not found."
    )
    # Excluded from the denominator rather than counted as a failure: never checkable.
    if counts[Status.NOT_DIRECTLY_DISCLOSED]:
        summary += (
            f" A further {counts[Status.NOT_DIRECTLY_DISCLOSED]} could not be checked, "
            "because this corpus does not disclose them as separate line items."
        )
    return summary


@dataclass(frozen=True)
class ComplianceReport:
    """The assembled compliance result, renderable into a chat response."""

    report_name: str
    checklist_category: str
    results: list[RequirementResult]
    summary: str
    disclaimer: str

    def to_markdown(self) -> str:
        """Render as a markdown table + summary + disclaimer for the chat response."""
        lines = [
            f"**Compliance check — {self.checklist_category.capitalize()} disclosures**",
            f"_Report: {self.report_name}_",
            "",
            "| Requirement | Status | Evidence | Citation |",
            "| --- | --- | --- | --- |",
        ]
        for r in self.results:
            requirement = _cell(f"{r.code} — {r.description}")
            if r.evidence_snippet:
                evidence = _cell(r.evidence_snippet)
                if r.rejected_as_insubstantial:
                    evidence += " _(evidence found, judged insufficient)_"
            else:
                evidence = "—"
            citation = _cell(r.citation) if r.citation else "—"
            lines.append(
                f"| {requirement} | {_STATUS_DISPLAY[r.status]} | {evidence} | {citation} |"
            )
        lines += [
            "",
            f"**Summary:** {self.summary}",
            "",
            f"_{STATUS_NOTE}_",
            "",
            f"_{self.disclaimer}_",
        ]
        return "\n".join(lines)


def assess_checklist(
    checklist: Sequence[ChecklistItem],
    *,
    report_name: str,
    category: str,
    retrieve: Callable[[ChecklistItem], EvidenceResult],
    assess: StatusAssessor = default_assess_status,
) -> ComplianceReport:
    """Run `retrieve` for every item in `checklist` and assemble a `ComplianceReport`."""
    results: list[RequirementResult] = []
    for item in checklist:
        # Curated as unanswerable by this corpus, short-circuit before spending the retrieval.
        if not getattr(item, "directly_disclosed", True):
            results.append(
                RequirementResult(
                    code=item.code,
                    description=item.description,
                    status=Status.NOT_DIRECTLY_DISCLOSED,
                    evidence_snippet=None,
                    citation=None,
                    top_score=0.0,
                    fallback_used=False,
                    rejected_as_insubstantial=False,
                )
            )
            continue

        evidence = retrieve(item)
        status = assess(item, evidence)

        snippet: str | None = None
        citation: str | None = None
        if evidence.results:
            top = evidence.results[0]
            snippet = truncate_snippet(
                top.payload.chunk_text, is_table=top.payload.content_type is ContentType.TABLE
            )
            citation = format_evidence_citation(top.payload)

        results.append(
            RequirementResult(
                code=item.code,
                description=item.description,
                status=status,
                evidence_snippet=snippet,
                citation=citation,
                top_score=evidence.top_score,
                fallback_used=evidence.fallback_used,
                rejected_as_insubstantial=bool(evidence.results)
                and status in _INSUBSTANTIAL_STATUSES,
            )
        )

    return ComplianceReport(
        report_name=report_name,
        checklist_category=category,
        results=results,
        summary=_summarize(results, report_name, category),
        disclaimer=DISCLAIMER,
    )
