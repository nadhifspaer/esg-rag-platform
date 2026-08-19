"""Tests for the POST /compliance-check endpoint."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app.api.compliance as compliance_api
from app.api.chat import get_chat_engine
from app.api.compliance import ComplianceResponse
from app.main import app
from app.rag.compliance.report import (
    DISCLAIMER,
    STATUS_NOTE,
    ComplianceReport,
    RequirementResult,
    Status,
)

client = TestClient(app)


# --- fakes ------------------------------------------------------------------


def _fake_engine(*, count: int = 5, has_key: bool = True) -> SimpleNamespace:
    """A stand-in engine: only the fields /compliance-check touches."""
    qdrant = SimpleNamespace(
        count=lambda **kwargs: SimpleNamespace(count=count),
    )
    return SimpleNamespace(
        qdrant_client=qdrant,
        openai_client=object() if has_key else None,
        settings=SimpleNamespace(
            qdrant_collection="esg_documents", openai_api_key="k" if has_key else ""
        ),
    )


def _use_engine(engine: SimpleNamespace) -> None:
    app.dependency_overrides[get_chat_engine] = lambda: engine


@pytest.fixture(autouse=True)
def _clear_overrides() -> Iterator[None]:
    yield
    app.dependency_overrides.clear()


def _canned_report() -> ComplianceReport:
    results = [
        RequirementResult(
            code="GRI 401-1",
            description="Turnover",
            status=Status.DISCLOSED,
            evidence_snippet="860 employees resigned",
            citation="PT Bank Central Asia Tbk — 2025 Sustainability Report, p. 14 (table)",
            top_score=0.6,
            fallback_used=False,
            rejected_as_insubstantial=False,
        ),
        RequirementResult(
            code="GRI 412-1",
            description="Human rights",
            status=Status.NOT_FOUND,
            evidence_snippet=None,
            citation=None,
            top_score=0.1,
            fallback_used=True,
            rejected_as_insubstantial=False,
        ),
        RequirementResult(
            code="GRI 406-1",
            description="Non-discrimination incidents",
            status=Status.PARTIALLY_DISCLOSED,
            evidence_snippet="disclosure-index table only, no narrative",
            citation="PT Bank Central Asia Tbk — 2025 Sustainability Report, p. 9 (table)",
            top_score=0.42,
            fallback_used=False,
            rejected_as_insubstantial=True,
        ),
    ]
    return ComplianceReport(
        report_name="PT Bank Central Asia Tbk — 2025 Sustainability Report",
        checklist_category="social",
        results=results,
        summary="Of 3 social requirements checked, 1 addressed, 1 not found, 1 rejected.",
        disclaimer=DISCLAIMER,
    )


# --- 422: resolution failures (real resolver) -------------------------------


def test_unknown_company_returns_422() -> None:
    _use_engine(_fake_engine())
    response = client.post(
        "/compliance-check", json={"company": "Foobar Bank", "category": "social"}
    )
    assert response.status_code == 422
    assert "match any known bank" in response.json()["detail"]


def test_standard_reference_as_company_returns_422() -> None:
    _use_engine(_fake_engine())
    response = client.post("/compliance-check", json={"company": "GRI 1", "category": "governance"})
    assert response.status_code == 422
    assert "not a company report" in response.json()["detail"]


def test_ambiguous_company_returns_422() -> None:
    _use_engine(_fake_engine())
    response = client.post(
        "/compliance-check", json={"company": "compare BCA and Mandiri", "category": "social"}
    )
    assert response.status_code == 422
    assert "more than one bank" in response.json()["detail"]


def test_invalid_category_returns_422() -> None:
    # Not one of the three enum values -> pydantic rejects before the route body.
    _use_engine(_fake_engine())
    response = client.post("/compliance-check", json={"company": "BCA", "category": "financial"})
    assert response.status_code == 422


# --- 404 / 503 --------------------------------------------------------------


def test_resolved_but_not_ingested_returns_404() -> None:
    # BCA resolves fine, but the corpus count comes back zero.
    _use_engine(_fake_engine(count=0))
    response = client.post("/compliance-check", json={"company": "BCA", "category": "social"})
    assert response.status_code == 404
    assert "No ingested report found" in response.json()["detail"]


def test_missing_openai_key_returns_503() -> None:
    _use_engine(_fake_engine(has_key=False))
    response = client.post("/compliance-check", json={"company": "BCA", "category": "social"})
    assert response.status_code == 503


def test_missing_engine_returns_503() -> None:
    # No override and no engine on app.state.
    response = client.post("/compliance-check", json={"company": "BCA", "category": "social"})
    assert response.status_code == 503


# --- 200: happy path (pipeline monkeypatched) -------------------------------


def test_successful_check_returns_structured_result(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_engine(_fake_engine(count=5))
    monkeypatch.setattr(
        compliance_api, "assess_checklist_with_llm_live", lambda *a, **k: _canned_report()
    )

    response = client.post("/compliance-check", json={"company": "BCA", "category": "social"})

    assert response.status_code == 200
    body = ComplianceResponse.model_validate(response.json())
    assert body.report_name == "PT Bank Central Asia Tbk — 2025 Sustainability Report"
    assert body.category == "social"
    assert [r.code for r in body.results] == ["GRI 401-1", "GRI 412-1", "GRI 406-1"]
    assert body.results[0].status == "disclosed"
    assert body.results[0].rejected_as_insubstantial is False
    assert body.results[1].status == "not found"
    assert body.results[1].citation is None
    assert body.results[1].rejected_as_insubstantial is False
    # Evidence was retrieved but judged insubstantial: snippet/citation still travel
    # through the API, and the flag is what tells this apart from a genuine empty retrieval.
    assert body.results[2].status == "partially disclosed"
    assert body.results[2].evidence_snippet == "disclosure-index table only, no narrative"
    assert body.results[2].citation is not None
    assert body.results[2].rejected_as_insubstantial is True
    assert body.status_note == STATUS_NOTE
    assert body.disclaimer == DISCLAIMER
    # The rendered markdown table travels alongside the structured rows.
    assert "| Requirement | Status | Evidence | Citation |" in body.markdown


# --- framework: default GRI, explicit EDGB, 404 on an uncurated combination -------------


def test_omitted_framework_defaults_to_gri_byte_identical_to_explicit_gri(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No existing caller (the frontend included) sends `framework`; this pins that
    behaviour stays byte-identical to explicitly asking for GRI, not just 'still 200'."""
    _use_engine(_fake_engine(count=5))
    monkeypatch.setattr(
        compliance_api, "assess_checklist_with_llm_live", lambda *a, **k: _canned_report()
    )

    omitted = client.post("/compliance-check", json={"company": "BCA", "category": "social"})
    explicit = client.post(
        "/compliance-check", json={"company": "BCA", "framework": "gri", "category": "social"}
    )

    assert omitted.status_code == explicit.status_code == 200
    assert omitted.json() == explicit.json()
    assert omitted.json()["framework"] == "gri"


def test_edgb_environmental_returns_the_real_edgb_checklist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Not mocked: exercises the real load_checklist -> checklists/edgb/environmental.json.
    _use_engine(_fake_engine(count=5))
    monkeypatch.setattr(
        compliance_api, "assess_checklist_with_llm_live", lambda *a, **k: _canned_report()
    )

    response = client.post(
        "/compliance-check",
        json={"company": "BCA", "framework": "edgb", "category": "environmental"},
    )

    assert response.status_code == 200
    body = ComplianceResponse.model_validate(response.json())
    assert body.framework == "edgb"


def test_uncurated_framework_category_combination_returns_404(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Redirect the checklists dir at an empty directory to exercise the 404 path directly.
    monkeypatch.setattr(compliance_api, "_CHECKLISTS_DIR", tmp_path)
    _use_engine(_fake_engine(count=5))
    response = client.post(
        "/compliance-check",
        json={"company": "BCA", "framework": "edgb", "category": "governance"},
    )
    assert response.status_code == 404
    assert "No checklist curated" in response.json()["detail"]


def test_invalid_framework_returns_422() -> None:
    # Not one of the three enum values -> pydantic rejects before the route body, same as
    # an invalid category.
    _use_engine(_fake_engine())
    response = client.post(
        "/compliance-check", json={"company": "BCA", "framework": "sasb", "category": "social"}
    )
    assert response.status_code == 422
