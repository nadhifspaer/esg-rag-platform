"""Tests for GET /documents (served from the engine's document catalog)."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api.chat import get_chat_engine
from app.main import app
from app.models.payload import Domain
from app.rag.document_catalog import DocumentSummary

client = TestClient(app)


def _catalog() -> list[DocumentSummary]:
    return [
        DocumentSummary(
            "BCA — 2025", Domain.COMPANY_DOCUMENTS, "sustainability_report", "BCA", 2025, 61, 20
        ),
        DocumentSummary("GRI 1: Foundation", Domain.STANDARDS, "gri_standard", None, None, 40, 15),
        DocumentSummary(
            "ESG Disclosure Guide Book",
            Domain.STANDARDS,
            "edgb_standard",
            None,
            None,
            30,
            12,
        ),
    ]


def _use_engine_with(catalog: list[DocumentSummary]) -> None:
    app.dependency_overrides[get_chat_engine] = lambda: SimpleNamespace(document_catalog=catalog)


@pytest.fixture(autouse=True)
def _clear_overrides() -> Iterator[None]:
    yield
    app.dependency_overrides.clear()


def test_lists_all_documents() -> None:
    _use_engine_with(_catalog())
    response = client.get("/documents")
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 3
    first = body["documents"][0]
    assert first["source_name"] == "BCA — 2025"
    assert first["domain"] == "company_documents"
    assert first["bank"] == "BCA"
    assert first["year"] == 2025
    assert first["chunk_count"] == 61
    assert first["page_count"] == 20


def test_filters_by_domain() -> None:
    _use_engine_with(_catalog())
    response = client.get("/documents", params={"domain": "company_documents"})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["documents"][0]["source_name"] == "BCA — 2025"


def test_invalid_domain_rejected() -> None:
    _use_engine_with(_catalog())
    response = client.get("/documents", params={"domain": "financial"})
    assert response.status_code == 422


def test_missing_engine_returns_503() -> None:
    # No override and no engine on app.state.
    response = client.get("/documents")
    assert response.status_code == 503
