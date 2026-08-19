"""Health check endpoint: reports liveness plus a real Qdrant dependency check."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app.api.chat import get_chat_engine_optional
from app.core.config import get_settings
from app.rag.chat_engine import ChatEngine

router = APIRouter(tags=["health"])


class ComponentHealth(BaseModel):
    """One dependency's health: `status` is "ok" or "error"; `detail` explains either."""

    status: str
    detail: str | None = None


class HealthChecks(BaseModel):
    """The dependency checks this service runs. Qdrant only, see the module docstring."""

    qdrant: ComponentHealth


class HealthResponse(BaseModel):
    """Overall health: `status` is "ok" only if every check passed, else "degraded"."""

    status: str
    environment: str
    checks: HealthChecks


def _check_qdrant(engine: ChatEngine | None) -> ComponentHealth:
    """Run a real query against Qdrant and report whether it answered."""
    if engine is None:
        return ComponentHealth(status="error", detail="chat engine not initialized")
    try:
        collection = engine.settings.qdrant_collection
        result = engine.qdrant_client.count(collection_name=collection, exact=True)
        return ComponentHealth(status="ok", detail=f"{result.count} points in '{collection}'")
    except Exception as exc:  # noqa: BLE001 (any failure means the dependency is unhealthy)
        return ComponentHealth(status="error", detail=f"qdrant query failed: {exc}")


@router.get("/health", response_model=HealthResponse)
async def health(
    response: Response,
    engine: Annotated[ChatEngine | None, Depends(get_chat_engine_optional)],
) -> HealthResponse:
    """Report liveness plus a live Qdrant connectivity check; 503 when degraded."""
    qdrant = await run_in_threadpool(_check_qdrant, engine)
    status = "ok" if qdrant.status == "ok" else "degraded"
    if status != "ok":
        response.status_code = 503
    return HealthResponse(
        status=status,
        environment=get_settings().environment,
        checks=HealthChecks(qdrant=qdrant),
    )
