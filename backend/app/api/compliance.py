"""POST /compliance-check: run the compliance pipeline for one report."""

from __future__ import annotations

import json
from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from qdrant_client.models import FieldCondition, Filter, MatchValue
from starlette.concurrency import run_in_threadpool

from app.api.chat import get_chat_engine
from app.core.auth import CurrentUser
from app.core.rate_limit import COMPLIANCE_SWEEP_UNITS, RequestBudget, reserve
from app.observability.tracing import budget_fields, traced_request, update_span
from app.rag.chat_engine import ChatEngine
from app.rag.compliance.checklist import ChecklistItem
from app.rag.compliance.llm_assessor import assess_checklist_with_llm_live
from app.rag.compliance.report import STATUS_NOTE, ComplianceReport
from app.rag.compliance.resolver import (
    AmbiguousCompanyError,
    CompanyNotFoundError,
    NotACompanyError,
    resolve_company_source_name,
)

router = APIRouter(tags=["compliance"])

# backend/app/api/compliance.py -> parents[1] is backend/app.
_CHECKLISTS_DIR = Path(__file__).resolve().parents[1] / "rag" / "compliance" / "checklists"


class ChecklistCategory(StrEnum):
    """The E/S/G checklist to run, one curated file per category."""

    ENVIRONMENTAL = "environmental"
    SOCIAL = "social"
    GOVERNANCE = "governance"


class FrameworkName(StrEnum):
    """Which disclosure framework's checklist to run: GRI default, EDGB curated incrementally."""

    GRI = "gri"
    EDGB = "edgb"


class ChecklistNotFoundError(RuntimeError):
    """No checklist file is curated for a given (framework, category) pair, a clean 404."""


class ComplianceRequest(BaseModel):
    """A compliance-check request: which company report, checked against which framework's
    checklist for which E/S/G category."""

    company: str = Field(
        ...,
        min_length=1,
        description="Company reference: alias, ticker, or legal name (e.g. 'BCA').",
    )
    framework: FrameworkName = Field(
        default=FrameworkName.GRI,
        description=(
            "Which disclosure framework's checklist to check the report against. Defaults "
            "to GRI — the only framework curated across all three categories today, and "
            "the framework every existing caller already expects without sending this field."
        ),
    )
    category: ChecklistCategory = Field(
        ..., description="The E/S/G disclosure checklist to check the report against."
    )


class RequirementRow(BaseModel):
    """One requirement's result: its status in the report, with evidence and citation."""

    code: str
    description: str
    status: str
    evidence_snippet: str | None
    citation: str | None
    top_score: float
    fallback_used: bool
    rejected_as_insubstantial: bool


class ComplianceResponse(BaseModel):
    """The structured compliance result for one report against one framework's checklist."""

    report_name: str
    framework: str
    category: str
    results: list[RequirementRow]
    summary: str
    status_note: str
    disclaimer: str
    markdown: str


def load_checklist(
    category: ChecklistCategory, framework: FrameworkName = FrameworkName.GRI
) -> list[ChecklistItem]:
    """Load and parse a (framework, category) checklist JSON into `ChecklistItem`s."""
    path = _CHECKLISTS_DIR / framework.value / f"{category.value}.json"
    if not path.exists():
        raise ChecklistNotFoundError(
            f"No checklist curated for framework={framework.value!r}, category={category.value!r}."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return [ChecklistItem(**item) for item in data["items"]]


def _report_exists(engine: ChatEngine, source_name: str) -> bool:
    """True if any chunk with this `source_name` is in the corpus (indexed-field count)."""
    result = engine.qdrant_client.count(
        collection_name=engine.settings.qdrant_collection,
        count_filter=Filter(
            must=[FieldCondition(key="source_name", match=MatchValue(value=source_name))]
        ),
        exact=True,
    )
    return result.count > 0


def _run_compliance(
    engine: ChatEngine,
    items: list[ChecklistItem],
    source_name: str,
    category: ChecklistCategory,
) -> ComplianceReport:
    """Run the checklist against the report, retrieval scoped to that report's source_name."""
    return assess_checklist_with_llm_live(
        items,
        report_name=source_name,
        category=category.value,
        source_name=source_name,
        qdrant_client=engine.qdrant_client,
        embed_client=engine.openai_client,
        openai_client=engine.openai_client,
        settings=engine.settings,
    )


def _serialize(report: ComplianceReport, framework: FrameworkName) -> ComplianceResponse:
    """Turn the pipeline's `ComplianceReport` into the API response model."""
    return ComplianceResponse(
        report_name=report.report_name,
        framework=framework.value,
        category=report.checklist_category,
        results=[
            RequirementRow(
                code=r.code,
                description=r.description,
                status=r.status.value,
                evidence_snippet=r.evidence_snippet,
                citation=r.citation,
                top_score=r.top_score,
                fallback_used=r.fallback_used,
                rejected_as_insubstantial=r.rejected_as_insubstantial,
            )
            for r in report.results
        ],
        summary=report.summary,
        status_note=STATUS_NOTE,
        disclaimer=report.disclaimer,
        markdown=report.to_markdown(),
    )


@router.post("/compliance-check", response_model=ComplianceResponse)
async def compliance_check(
    payload: ComplianceRequest,
    engine: Annotated[ChatEngine, Depends(get_chat_engine)],
    budget: Annotated[RequestBudget, Depends(reserve(COMPLIANCE_SWEEP_UNITS))],
    user: CurrentUser,
) -> ComplianceResponse:
    """Run the compliance pipeline for one company report against one E/S/G checklist."""
    if engine.openai_client is None and not engine.settings.openai_api_key:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not configured; cannot embed queries for retrieval.",
        )

    # For resolving the informal company reference to the exact stored source_name.
    # All three resolution failures are unprocessable input -> 422.
    try:
        source_name = resolve_company_source_name(payload.company)
    except (CompanyNotFoundError, AmbiguousCompanyError, NotACompanyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # A resolved-but-not-ingested report is a missing resource, not all-"not found" rows.
    if not await run_in_threadpool(_report_exists, engine, source_name):
        raise HTTPException(
            status_code=404,
            detail=f"No ingested report found for {source_name!r}.",
        )

    # Load the checklist (404 if uncurated) and run the pipeline off the event loop.
    try:
        items = load_checklist(payload.category, payload.framework)
    except ChecklistNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    with traced_request(
        name="compliance-check",
        user_id=user.id,
        query=f"{payload.company} vs {payload.framework.value} {payload.category.value}",
        tags=["compliance-check", payload.framework.value, payload.category.value],
        metadata={
            "endpoint": "/compliance-check",
            "pipeline": "compliance_checklist_sweep",
            "company": payload.company,
            "resolved_source_name": source_name,
            "framework": payload.framework.value,
            "category": payload.category.value,
            "requirement_count": len(items),
        },
    ) as span:
        report = await run_in_threadpool(
            _run_compliance, engine, items, source_name, payload.category
        )
        # A fixed-shape sweep: the reservation is exact, but settled explicitly anyway.
        budget.settle(COMPLIANCE_SWEEP_UNITS)

        statuses: Counter[str] = Counter(r.status.value for r in report.results)
        update_span(
            span,
            output={
                "report_name": report.report_name,
                "requirements": len(report.results),
                "status_counts": dict(statuses),
                # Per-requirement evidence provenance, the compliance equivalent of /chat's hits.
                "evidence": [
                    {
                        "code": r.code,
                        "status": r.status.value,
                        "citation": r.citation,
                        "top_score": round(r.top_score, 4),
                        "fallback_used": r.fallback_used,
                    }
                    for r in report.results
                ],
            },
            metadata={
                "endpoint": "/compliance-check",
                "pipeline": "compliance_checklist_sweep",
                "framework": payload.framework.value,
                "category": payload.category.value,
                "resolved_source_name": source_name,
                "status_counts": dict(statuses),
                "fallbacks_used": sum(1 for r in report.results if r.fallback_used),
                **budget_fields(budget),
            },
        )
    return _serialize(report, payload.framework)
