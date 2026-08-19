"""Conversation turn schema: the shape of client-supplied chat history."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

# Caps on client-supplied history: bounds prompt-token growth from one request.
MAX_HISTORY_TURNS = 8
MAX_TURN_CHARS = 4000


class Pipeline(StrEnum):
    """Which pipeline produced a turn (chat, compliance-check, or rank all render in the thread)."""

    CHAT = "chat"
    COMPLIANCE = "compliance"
    RANK = "rank"


class ConversationTurn(BaseModel):
    """One completed exchange: what the user asked and what the assistant answered."""

    model_config = ConfigDict(frozen=True)

    question: str = Field(..., min_length=1, max_length=MAX_TURN_CHARS)
    answer: str = Field(..., min_length=1, max_length=MAX_TURN_CHARS)
    retrieval_query: str | None = Field(default=None, max_length=MAX_TURN_CHARS)
    pipeline: Pipeline = Pipeline.CHAT
