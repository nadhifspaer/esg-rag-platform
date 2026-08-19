"""Embed text with OpenAI `text-embedding-3-small`: used by every chunk/query embed call."""

from __future__ import annotations

from openai import OpenAI

from app.core.config import Settings, get_settings

# text-embedding-3-small returns 1536-dim vectors. This drives the Qdrant vector
# size; the two must match or upserts are rejected.
EMBEDDING_DIM = 1536

# OpenAI accepts many inputs per request; batching keeps requests well under the
# per-request token ceiling while still amortizing round-trips over many chunks.
_BATCH_SIZE = 128


class EmbeddingError(RuntimeError):
    """Raised when embeddings cannot be produced (unconfigured key, API error)."""


def _resolve_client(settings: Settings, injected: OpenAI | None) -> OpenAI:
    """Return the injected client, or build one from settings (needs the API key)."""
    if injected is not None:
        return injected
    if not settings.openai_api_key:
        raise EmbeddingError(
            "OPENAI_API_KEY is not set; cannot embed. Add it to your .env (see .env.example)."
        )
    return OpenAI(api_key=settings.openai_api_key)


def embed_texts(
    texts: list[str],
    *,
    client: OpenAI | None = None,
    settings: Settings | None = None,
) -> list[list[float]]:
    """Embed a list of texts, returning one vector per text in order (batched internally)."""
    if not texts:
        return []
    settings = settings or get_settings()
    client = _resolve_client(settings, client)
    model = settings.openai_embedding_model

    vectors: list[list[float]] = []
    for start in range(0, len(texts), _BATCH_SIZE):
        batch = texts[start : start + _BATCH_SIZE]
        try:
            response = client.embeddings.create(model=model, input=batch)
        except Exception as exc:  # network/API failure -> a clear, typed error
            raise EmbeddingError(f"embedding request failed: {exc}") from exc
        # Sort by index defensively so order matches the input regardless of API response order.
        vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
    return vectors
