"""Tests for the /health endpoint's live Qdrant dependency check."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api.chat import get_chat_engine_optional
from app.main import app

client = TestClient(app)


def _engine(*, count: int = 61, raises: bool = False) -> SimpleNamespace:
    def _count(**kwargs: object) -> SimpleNamespace:
        if raises:
            raise RuntimeError("connection refused")
        return SimpleNamespace(count=count)

    return SimpleNamespace(
        qdrant_client=SimpleNamespace(count=_count),
        settings=SimpleNamespace(qdrant_collection="esg_documents"),
    )


@pytest.fixture(autouse=True)
def _clear_overrides() -> Iterator[None]:
    yield
    app.dependency_overrides.clear()


def test_health_ok_when_qdrant_answers() -> None:
    app.dependency_overrides[get_chat_engine_optional] = lambda: _engine(count=61)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "environment" in body
    assert body["checks"]["qdrant"]["status"] == "ok"
    assert "61 points" in body["checks"]["qdrant"]["detail"]


def test_health_degraded_when_qdrant_query_fails() -> None:
    app.dependency_overrides[get_chat_engine_optional] = lambda: _engine(raises=True)
    response = client.get("/health")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["checks"]["qdrant"]["status"] == "error"
    assert "connection refused" in body["checks"]["qdrant"]["detail"]


def test_health_degraded_when_engine_not_initialized() -> None:
    app.dependency_overrides[get_chat_engine_optional] = lambda: None
    response = client.get("/health")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["checks"]["qdrant"]["status"] == "error"
    assert "not initialized" in body["checks"]["qdrant"]["detail"]
