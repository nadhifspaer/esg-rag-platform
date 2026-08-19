from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


@lru_cache
def get_langfuse():  # noqa: ANN201 - langfuse.Langfuse, imported lazily
    """The process-wide Langfuse client, or None when tracing is not configured."""
    settings = get_settings()
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return None
    try:
        from langfuse import Langfuse

        client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
            environment=settings.environment,
        )
        logger.info("langfuse tracing enabled (host=%s)", settings.langfuse_host)
        return client
    except Exception as exc:  # noqa: BLE001 - never let observability break startup
        logger.warning("langfuse unavailable, tracing disabled: %s", exc)
        return None


def tracing_enabled() -> bool:
    return get_langfuse() is not None


def openai_class(settings: Settings | None = None):  # noqa: ANN201 - an OpenAI class
    """The OpenAI client class to build request clients from, instrumented when tracing is on."""
    settings = settings or get_settings()
    if settings.langfuse_public_key and settings.langfuse_secret_key:
        try:
            from langfuse.openai import OpenAI as InstrumentedOpenAI

            return InstrumentedOpenAI
        except Exception as exc:  # noqa: BLE001
            logger.warning("langfuse OpenAI instrumentation unavailable: %s", exc)
    from openai import OpenAI

    return OpenAI


def _propagated_metadata(metadata: dict[str, Any]) -> dict[str, str]:
    """Pre-serialize trace metadata so non-string values survive Langfuse's `str()` coercion."""
    prepared: dict[str, str] = {}
    for key, value in metadata.items():
        prepared[key] = value if isinstance(value, str) else json.dumps(value, default=str)
    return prepared


@contextmanager
def traced_request(
    *,
    name: str,
    user_id: str | None,
    query: str,
    tags: Sequence[str] = (),
    metadata: dict[str, Any] | None = None,
    session_id: str | None = None,
) -> Iterator[Any]:
    """One trace per API request, yielding the root span (or None when tracing is off)."""
    client = get_langfuse()
    if client is None:
        yield None
        return

    # Only the *setup* is guarded: the endpoint's own exceptions must pass through unwrapped.
    try:
        from langfuse import propagate_attributes

        span_cm = client.start_as_current_observation(
            name=name, as_type="span", input={"query": query}
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("langfuse trace setup failed for %s: %s", name, exc)
        yield None
        return

    with span_cm as span:
        with propagate_attributes(
            user_id=user_id,
            session_id=session_id,
            trace_name=name,
            tags=list(tags),
            metadata=_propagated_metadata(metadata or {}),
        ):
            yield span


def update_span(span: Any, **fields: Any) -> None:
    """Set output/metadata on a span, tolerating a disabled or broken tracer."""
    if span is None:
        return
    try:
        span.update(**fields)
    except Exception as exc:  # noqa: BLE001
        logger.warning("langfuse span update failed: %s", exc)


def record_retrieval(span: Any, chunks: Sequence[Any], *, name: str = "retrieval") -> None:
    """Attach the retrieval hits (which chunks, from where, at what score) to the span."""
    if span is None:
        return
    try:
        hits = []
        for chunk in chunks:
            # A scored result exposes `.payload`/`.score`; a `Citation` carries the fields directly.
            payload = getattr(chunk, "payload", chunk)
            source_name = getattr(payload, "source_name", None)
            if source_name is None:
                continue
            hit = {
                "source_name": source_name,
                "page_number": payload.page_number,
                "content_type": payload.content_type.value,
            }
            score = getattr(chunk, "score", None)
            if score is not None:
                hit["score"] = round(float(score), 4)
            hits.append(hit)
        with span.start_as_current_observation(name=name, as_type="retriever") as retrieval:
            retrieval.update(output={"hit_count": len(hits), "hits": hits})
    except Exception as exc:  # noqa: BLE001
        logger.warning("langfuse retrieval span failed: %s", exc)


def budget_fields(budget: Any) -> dict[str, Any]:
    """The rate-limit outcome for this request, read from the settled `RequestBudget`."""
    if budget is None:
        return {}
    return {
        "rate_limit_reserved_units": budget.reserved,
        "rate_limit_charged_units": budget.charged,
        "rate_limit_balance_after": budget.balance_after,
    }


def flush() -> None:
    """Force-send buffered spans. Called at shutdown so nothing is lost on a redeploy."""
    client = get_langfuse()
    if client is None:
        return
    try:
        client.flush()
    except Exception as exc:  # noqa: BLE001
        logger.warning("langfuse flush failed: %s", exc)
