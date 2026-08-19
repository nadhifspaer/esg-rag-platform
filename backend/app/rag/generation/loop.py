"""Bounded agentic retry loop: the chat pipeline's control flow."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from app.models.citation import Citation, RetrievedChunk
from app.rag.generation.self_check import (
    GroundednessDecision,
    Verdict,
    check_groundedness,
)

logger = logging.getLogger(__name__)

# Hard cap: 2 retries => at most 3 total attempts, code-enforced regardless of the model's verdict.
DEFAULT_MAX_RETRIES = 2

# Injected stage signatures: `retrieve(query)` -> ranked chunks, `generate(question, chunks)`
# -> full answer text, `check(question, answer, chunks)` -> groundedness verdict.
Retrieve = Callable[[str], Sequence[RetrievedChunk]]
Generate = Callable[[str, Sequence[RetrievedChunk]], str]
Check = Callable[[str, str, Sequence[RetrievedChunk]], GroundednessDecision]


class StopReason(StrEnum):
    """Why the loop stopped: the judge passed the answer, or the hard cap was hit."""

    SUFFICIENT = "sufficient"
    RETRY_CAP = "retry_cap"


@dataclass(frozen=True)
class ChatResult:
    """The loop's outcome: final answer, citations, iteration count, the query that fetched it."""

    answer: str
    citations: list[Citation]
    attempts: int
    retries: int
    stop_reason: StopReason
    retrieval_query: str


def _finish(
    answer: str,
    chunks: Sequence[RetrievedChunk],
    attempts: int,
    stop_reason: StopReason,
    retrieval_query: str,
) -> ChatResult:
    """Build the result and log the iteration count for this request."""
    retries = attempts - 1
    logger.info(
        "chat answer loop: attempts=%d retries=%d stop_reason=%s",
        attempts,
        retries,
        stop_reason.value,
    )
    return ChatResult(
        answer=answer,
        citations=[Citation.from_result(chunk) for chunk in chunks],
        attempts=attempts,
        retries=retries,
        stop_reason=stop_reason,
        retrieval_query=retrieval_query,
    )


def answer_with_retries(
    query: str,
    *,
    retrieve: Retrieve,
    generate: Generate,
    check: Check = check_groundedness,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retrieval_seed: str | None = None,
) -> ChatResult:
    """Answer `query`, retrying retrieval+generation while the self-check says insufficient."""
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")

    search_query = retrieval_seed or query
    answer = ""
    chunks: Sequence[RetrievedChunk] = ()
    retrieval_query = query

    for attempt in range(1, max_retries + 2):  # attempts 1 .. max_retries+1
        retrieval_query = search_query
        chunks = retrieve(retrieval_query)
        # Generate and judge against the ORIGINAL question; the rewritten query only ever
        # steers retrieval, never what we are actually answering.
        answer = generate(query, chunks)
        decision = check(query, answer, chunks)
        if decision.verdict is Verdict.SUFFICIENT:
            return _finish(answer, chunks, attempt, StopReason.SUFFICIENT, retrieval_query)
        # Insufficient: line up the rewritten query for the next attempt (if any remain).
        search_query = decision.rewritten_query or query

    # Cap reached with every attempt judged insufficient: return the best available
    # answer (the last one: each retry refined the query, so it is the most-informed).
    return _finish(answer, chunks, max_retries + 1, StopReason.RETRY_CAP, retrieval_query)
