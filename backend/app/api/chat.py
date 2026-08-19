"""POST /chat: classify, retrieve + generate through the bounded agentic loop, stream."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.core.auth import CurrentUser
from app.core.rate_limit import CHAT_ATTEMPT_UNITS, CHAT_MAX_UNITS, RequestBudget, reserve
from app.models.conversation import MAX_HISTORY_TURNS, ConversationTurn
from app.models.payload import Domain
from app.observability.tracing import budget_fields, record_retrieval, traced_request, update_span
from app.rag.chat_engine import ChatEngine, scoped_source_names
from app.rag.compliance.resolver import resolve_category
from app.rag.conversation import apply_context, resolve_context, sanitize_history
from app.rag.entity_index import EntityKind, default_entity_index
from app.rag.generation.generator import GenerationError
from app.rag.generation.loop import ChatResult, answer_with_retries
from app.rag.generation.prompts import CHAT_GENERATION_TYPES
from app.rag.generation.self_check import SelfCheckError
from app.rag.query_classifier import QueryType

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

# Answer text is streamed in fixed-size slices; the answer is already complete when this runs.
_ANSWER_CHUNK_CHARS = 60


class ChatRequest(BaseModel):
    """A chat request: the question, an optional domain filter, and the model toggle."""

    query: str = Field(..., min_length=1, description="The user's question.")
    domain: Domain | None = Field(
        default=None,
        description=(
            "Optional domain filter. When set, retrieval is confined to this domain, "
            "overriding the classifier's inferred domain. When omitted, the classifier's "
            "domain routing is used."
        ),
    )
    high_accuracy: bool = Field(
        default=False,
        description="Use the high-accuracy generation model (gpt-4.1) instead of the default.",
    )
    history: list[ConversationTurn] = Field(
        default_factory=list,
        max_length=MAX_HISTORY_TURNS,
        description=(
            "The conversation so far, oldest first — sent by the client, never stored by "
            "the server. Used to interpret an elliptical follow-up ('what about Mandiri?') "
            "and to decide what its first retrieval searches for. Capped at "
            f"{MAX_HISTORY_TURNS} turns."
        ),
    )
    conversation_id: str | None = Field(
        default=None,
        max_length=100,
        description=(
            "Opaque client-generated conversation id. Used only to group this request's "
            "trace into a Langfuse session; it is never stored, never trusted for access "
            "control, and never used to meter the rate limit."
        ),
    )


def get_chat_engine_optional(request: Request) -> ChatEngine | None:
    """Return the process-wide chat engine, or None if it was not built."""
    return getattr(request.app.state, "chat_engine", None)


def get_chat_engine(request: Request) -> ChatEngine:
    """Dependency: return the process-wide chat engine, or 503 if it was not built."""
    engine = get_chat_engine_optional(request)
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Chat engine is not available (check Qdrant/OpenAI configuration).",
        )
    return engine


def _sse(event: str, data: dict) -> str:
    """Format one Server-Sent Event. `ensure_ascii=False` keeps Bahasa Indonesia intact."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _answer_slices(text: str) -> Iterator[str]:
    """Yield the answer in fixed-size character slices for incremental delivery."""
    for start in range(0, len(text), _ANSWER_CHUNK_CHARS):
        yield text[start : start + _ANSWER_CHUNK_CHARS]


@router.post("/chat")
async def chat(
    payload: ChatRequest,
    engine: Annotated[ChatEngine, Depends(get_chat_engine)],
    budget: Annotated[RequestBudget, Depends(reserve(CHAT_MAX_UNITS))],
    user: CurrentUser,
) -> StreamingResponse:
    """Answer a chat query through the retrieval/generation loop and stream the result."""
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="query must not be empty.")

    # Client-owned history: sanitize before anything reads it (citation markers stripped).
    history = sanitize_history(payload.history)

    # Resolve an elliptical follow-up against the conversation, deterministic, no model call.
    context = resolve_context(query, history, index=default_entity_index())

    # Classify off the event loop; LLM fallback is skipped once rules confirm the remaining
    # uncertainty is just "which report", not "what kind of question".
    classification = await run_in_threadpool(engine.classify, query, use_llm_fallback=False)
    if not classification.confident and (
        context.inherited_entity is None
        or classification.query_type is not QueryType.COMPLIANCE_CHECK
    ):
        classification = await run_in_threadpool(engine.classify, query, use_llm_fallback=True)
    # Fold in an inherited entity when the utterance named none, so scoping can confine retrieval.
    classification = apply_context(classification, context)

    # An explicit request domain overrides the classifier's inferred domains; otherwise
    # use what the classifier resolved.
    domains = (payload.domain,) if payload.domain is not None else classification.domains

    # --- Up-front routing validation: reject as clean HTTP errors before streaming. ---
    if classification.query_type is QueryType.COMPLIANCE_CHECK:
        # Structured detail (not just a message) so a frontend can auto-route to /compliance-check.
        company_entities = [m for m in classification.entities if m.entity.kind is EntityKind.BANK]
        company = company_entities[0].entity.canonical_name if len(company_entities) == 1 else None
        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    "This looks like a compliance-check query. Those are handled by the "
                    "dedicated compliance pipeline, not the chat endpoint."
                ),
                "query_type": QueryType.COMPLIANCE_CHECK.value,
                "company": company,
                # None means genuinely ambiguous: the caller must ask, never guess a category.
                "category": resolve_category(query),
            },
        )
    if classification.query_type not in CHAT_GENERATION_TYPES:
        # Derived from the set the prompts actually support, so a new query type can't reopen
        # the gap that once let AGGREGATION_RANKING fall through to an unhandled 500.
        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    f"This query was routed as '{classification.query_type.value}', which "
                    "the chat endpoint does not answer. Corpus-wide questions about the set "
                    "of banks are handled by the ranking pipeline; rephrase to ask about one "
                    "bank's own disclosure if you meant a single-report lookup."
                ),
                "query_type": classification.query_type.value,
            },
        )
    if context.compliance_follow_up:
        # A company swap right after a compliance answer is a repeat compliance check, refused.
        company = context.entity.entity.canonical_name if context.entity else None
        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    "This looks like a follow-up compliance check on a different company. "
                    "Send it to /compliance-check rather than /chat — answering it here "
                    "would produce an ordinary retrieval answer that reads like a compliance "
                    "verdict without being one."
                ),
                "query_type": QueryType.COMPLIANCE_CHECK.value,
                "company": company,
                # Left None rather than guessed: the follow-up carries no topic keywords to match.
                "category": None,
            },
        )
    if not domains:
        # No entity named and no explicit filter: retrieval would have to search
        # unfiltered, which this project forbids. Ask the caller to specify a domain.
        raise HTTPException(
            status_code=422,
            detail=(
                "Could not determine which knowledge domain to search. Rephrase to name a "
                "bank or standard, or set `domain` explicitly "
                "(company_documents or standards). On a follow-up question, "
                "naming the bank again is usually enough."
            ),
        )
    if engine.openai_client is None and not engine.settings.openai_api_key:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not configured; cannot generate answers.",
        )

    model = (
        engine.settings.openai_generation_model_high
        if payload.high_accuracy
        else engine.settings.openai_generation_model
    )

    # Confine an unambiguous domain's leg to the one bank/standard it names, per-domain
    # (see `scoped_source_names`); an ambiguous or low-confidence domain searches unscoped.
    source_names = scoped_source_names(classification, domains)
    scoped_to = {d.value: source_names.get(d) for d in domains}

    trace_metadata = {
        "endpoint": "/chat",
        "pipeline": "chat_agentic_loop",
        "query_type": classification.query_type.value,
        "domains": [d.value for d in domains],
        "scoped_to": scoped_to,
        "model": model,
        "high_accuracy": payload.high_accuracy,
        "classifier_method": classification.method.value,
        "classifier_confident": classification.confident,
        # `context_source` names which follow-up shape this turn actually was.
        "history_turns": len(history),
        "context_source": context.context_source.value,
        "inherited_entity": (
            context.inherited_entity.entity.canonical_name if context.inherited_entity else None
        ),
        "retrieval_seed": context.retrieval_seed,
    }

    def _events() -> Iterator[str]:
        # Routing is known before the loop runs, so send it first.
        yield _sse(
            "meta",
            {
                "query_type": classification.query_type.value,
                "domains": [d.value for d in domains],
                # The single report retrieval was confined to, or null for a domain-wide search.
                "scoped_to": scoped_to,
                "model": model,
                "high_accuracy": payload.high_accuracy,
                "classifier": {
                    "method": classification.method.value,
                    "confident": classification.confident,
                    "reason": classification.reason,
                },
                # How an elliptical follow-up was interpreted, for the client to show.
                "conversation_id": payload.conversation_id,
                "history_turns": len(history),
                "context_source": context.context_source.value,
                "resolved_question": context.resolved_question,
                "retrieval_seed": context.retrieval_seed,
                "inherited_entity": (
                    context.inherited_entity.entity.canonical_name
                    if context.inherited_entity
                    else None
                ),
            },
        )

        with traced_request(
            name="chat",
            user_id=user.id,
            query=query,
            tags=["chat", classification.query_type.value],
            metadata=trace_metadata,
            session_id=payload.conversation_id,
        ) as span:
            try:
                result: ChatResult = answer_with_retries(
                    context.resolved_question,
                    retrieve=engine.make_retrieve(domains, source_names=source_names),
                    generate=engine.make_generate(
                        classification.query_type, payload.high_accuracy, history
                    ),
                    check=engine.make_check(),
                    retrieval_seed=context.retrieval_seed,
                )
            except (GenerationError, SelfCheckError) as exc:
                # A failure after streaming started: an SSE error, not a broken connection.
                logger.warning("chat generation failed: %s", exc)
                update_span(span, level="ERROR", status_message=str(exc))
                yield _sse("error", {"message": str(exc)})
                return
            except Exception:
                # Any other unanticipated failure, caught so it doesn't silently truncate the
                # stream; logged with full traceback, client only sees a generic message.
                logger.exception("chat generation failed with an unexpected error")
                update_span(span, level="ERROR", status_message="unexpected error")
                yield _sse(
                    "error",
                    {
                        "message": (
                            "Something went wrong while generating this answer. Please try again."
                        )
                    },
                )
                return

            # Settle against what the loop actually ran (worst case was reserved up front).
            budget.settle(result.attempts * CHAT_ATTEMPT_UNITS)

            # Reuses this same ChatResult so the trace tag cannot drift from the response.
            record_retrieval(span, result.citations, name="retrieval (final attempt)")
            update_span(
                span,
                output={"answer": result.answer, "citations": len(result.citations)},
                metadata={
                    **trace_metadata,
                    "attempts": result.attempts,
                    "retries": result.retries,
                    "stop_reason": result.stop_reason.value,
                    "retrieval_query": result.retrieval_query,
                    **budget_fields(budget),
                },
            )

        yield _sse(
            "citations",
            {
                "citations": [
                    {
                        "index": i,
                        "label": citation.label,
                        "source_name": citation.source_name,
                        "page_number": citation.page_number,
                        "content_type": citation.content_type.value,
                        "excerpt": citation.excerpt,
                    }
                    for i, citation in enumerate(result.citations, start=1)
                ]
            },
        )

        for slice_ in _answer_slices(result.answer):
            yield _sse("token", {"text": slice_})

        yield _sse(
            "done",
            {
                "attempts": result.attempts,
                "retries": result.retries,
                "stop_reason": result.stop_reason.value,
                "retrieval_query": result.retrieval_query,
            },
        )

    # Runs this sync generator in a threadpool; headers keep proxies from buffering the stream.
    return StreamingResponse(
        _events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
