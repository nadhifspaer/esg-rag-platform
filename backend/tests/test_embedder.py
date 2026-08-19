"""Tests for `embed_texts`."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.rag.embedder import _BATCH_SIZE, EMBEDDING_DIM, EmbeddingError, embed_texts

# --- fakes & helpers -------------------------------------------------------


class _FakeEmbeddings:
    """Records each create() call. Returns one vector per input whose element 0
    encodes that input's index within the batch, so order can be verified."""

    def __init__(self, *, fail: bool = False, shuffle: bool = False) -> None:
        self.fail = fail
        self.shuffle = shuffle
        self.calls: list[tuple[str, list[str]]] = []

    def create(self, model, input):  # noqa: A002 - 'input' matches the OpenAI API
        self.calls.append((model, list(input)))
        if self.fail:
            raise RuntimeError("simulated API failure")
        data = [
            SimpleNamespace(index=i, embedding=[float(i)] * EMBEDDING_DIM)
            for i in range(len(input))
        ]
        if self.shuffle:
            data.reverse()  # return out of order; embed_texts must re-sort by index
        return SimpleNamespace(data=data)


class FakeOpenAI:
    def __init__(self, **kwargs) -> None:
        self.embeddings = _FakeEmbeddings(**kwargs)

    @property
    def calls(self) -> list[tuple[str, list[str]]]:
        return self.embeddings.calls


def _settings(api_key: str = "test-key") -> Settings:
    return Settings(openai_api_key=api_key, openai_embedding_model="text-embedding-3-small")


# --- unit tests ------------------------------------------------------------


def test_embed_texts_batches_large_input() -> None:
    client = FakeOpenAI()
    n = 2 * _BATCH_SIZE + 44  # spans three requests
    texts = [f"chunk {i}" for i in range(n)]

    vectors = embed_texts(texts, client=client, settings=_settings())

    assert len(vectors) == n
    batch_sizes = [len(batch) for _model, batch in client.calls]
    assert batch_sizes == [_BATCH_SIZE, _BATCH_SIZE, 44]
    assert all(model == "text-embedding-3-small" for model, _batch in client.calls)


def test_embed_texts_empty_returns_empty_without_calling_api() -> None:
    client = FakeOpenAI()
    assert embed_texts([], client=client, settings=_settings()) == []
    assert client.calls == []  # no request made for empty input


def test_embed_texts_preserves_input_order() -> None:
    client = FakeOpenAI(shuffle=True)  # API hands data back out of order
    vectors = embed_texts(["a", "b", "c", "d"], client=client, settings=_settings())
    # Each vector's element 0 is its index; sorted back into input order.
    assert [v[0] for v in vectors] == [0.0, 1.0, 2.0, 3.0]


def test_embed_texts_wraps_api_failure_in_embedding_error() -> None:
    client = FakeOpenAI(fail=True)
    with pytest.raises(EmbeddingError, match="embedding request failed"):
        embed_texts(["x"], client=client, settings=_settings())


def test_embed_texts_missing_api_key_raises_embedding_error() -> None:
    with pytest.raises(EmbeddingError):
        embed_texts(["x"], settings=_settings(api_key=""))  # no client, no key


# --- live test (opt-in: explicit flag) --------------------------------------

_RUN_LIVE = os.environ.get("RUN_LIVE_EMBEDDER_TESTS") == "1"
live = pytest.mark.skipif(
    not _RUN_LIVE,
    reason="RUN_LIVE_EMBEDDER_TESTS!=1; skipping live embedder tests (opt-in only)",
)


@live
def test_live_embeds_real_texts_with_expected_dim() -> None:
    vectors = embed_texts(["hello world", "sustainable finance disclosure"])
    assert len(vectors) == 2
    assert all(len(v) == EMBEDDING_DIM for v in vectors)
    assert vectors[0] != vectors[1]  # different texts -> different vectors
