"""FastAPI application entrypoint: builds the app, wires CORS, and mounts the routers."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.api.compliance import router as compliance_router
from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.api.rank import router as rank_router
from app.core.auth import require_user
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.rag.chat_engine import build_chat_engine

settings = get_settings()

# Before any logger is used, since uvicorn leaves the root logger unconfigured otherwise.
configure_logging(settings.log_level)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the chat engine once at startup and stash it on `app.state`."""
    try:
        app.state.chat_engine = build_chat_engine(settings)
        logger.info("chat engine ready")
    except Exception as exc:  # noqa: BLE001 (startup must not crash on a dependency)
        app.state.chat_engine = None
        logger.warning("chat engine unavailable at startup: %s", exc)
    yield


app = FastAPI(
    title="ESG RAG Platform API",
    version="0.1.0",
    lifespan=lifespan,
)

# Allow the Next.js frontend (configured origins) to call this API from a browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Route policy: /health and /documents are public; /chat, /compliance-check, /rank require auth.
_authenticated = [Depends(require_user)]

app.include_router(health_router)
app.include_router(documents_router)
app.include_router(chat_router, dependencies=_authenticated)
app.include_router(compliance_router, dependencies=_authenticated)
app.include_router(rank_router, dependencies=_authenticated)
