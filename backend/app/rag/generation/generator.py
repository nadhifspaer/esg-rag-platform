"""Generation service: retrieved chunks + a query -> a streamed, cited answer."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from openai import OpenAI

from app.core.config import Settings, get_settings
from app.models.citation import Citation, RetrievedChunk
from app.models.conversation import ConversationTurn
from app.rag.generation.prompts import build_chat_prompt
from app.rag.query_classifier import QueryType

# Grounded, quote-exactly answers want determinism over variety.
_TEMPERATURE = 0.0


class GenerationError(RuntimeError):
    """Raised when generation cannot start (unconfigured key) or the API call fails."""


@dataclass(frozen=True)
class GeneratedAnswer:
    """A streamable answer plus the metadata known before any token is produced."""

    citations: list[Citation]
    model: str
    tokens: Iterator[str]


def _resolve_client(settings: Settings, injected: OpenAI | None) -> OpenAI:
    """Return the injected client, or build one from settings (needs the API key)."""
    if injected is not None:
        return injected
    if not settings.openai_api_key:
        raise GenerationError(
            "OPENAI_API_KEY is not set; cannot generate. Add it to your .env (see .env.example)."
        )
    return OpenAI(api_key=settings.openai_api_key)


def generate_answer(
    query: str,
    chunks: Sequence[RetrievedChunk],
    *,
    query_type: QueryType,
    high_accuracy: bool = False,
    history: Sequence[ConversationTurn] = (),
    client: OpenAI | None = None,
    settings: Settings | None = None,
) -> GeneratedAnswer:
    """Generate a streamed, citation-grounded answer for `query` from `chunks`."""
    settings = settings or get_settings()
    resolved_client = _resolve_client(settings, client)
    model = (
        settings.openai_generation_model_high if high_accuracy else settings.openai_generation_model
    )
    # Built eagerly so an invalid query type (e.g. compliance-check) fails now, not mid-stream.
    prompt = build_chat_prompt(query_type, query, chunks, history)

    def _stream() -> Iterator[str]:
        try:
            response = resolved_client.chat.completions.create(
                model=model,
                messages=prompt.messages,
                temperature=_TEMPERATURE,
                stream=True,
            )
            for event in response:
                if not event.choices:
                    continue  # e.g. a trailing usage-only event carries no choice
                delta = event.choices[0].delta.content
                if delta:
                    yield delta
        except GenerationError:
            raise
        except Exception as exc:  # network/API failure -> a clear, typed error
            raise GenerationError(f"generation request failed: {exc}") from exc

    return GeneratedAnswer(citations=prompt.citations, model=model, tokens=_stream())
