"""Tests for the generation service."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.models.payload import ChunkPayload, ContentType, Domain
from app.rag.generation.generator import GenerationError, generate_answer
from app.rag.generation.prompts import SINGLE_DOMAIN_SYSTEM_PROMPT
from app.rag.query_classifier import QueryType
from app.rag.vector_store import SearchResult


class _FakeStreamClient:
    """Fake OpenAI client: `create` records its kwargs and returns a generator of stream events."""

    def __init__(self, deltas: list[str | None]) -> None:
        self._deltas = deltas
        self.create_calls: list[dict] = []
        outer = self

        class _Completions:
            def create(self, **kwargs):  # noqa: ANN003 - mirrors the OpenAI signature
                outer.create_calls.append(kwargs)

                def events():
                    for delta in outer._deltas:
                        yield SimpleNamespace(
                            choices=[SimpleNamespace(delta=SimpleNamespace(content=delta))]
                        )

                return events()

        self.chat = SimpleNamespace(completions=_Completions())


def _settings() -> Settings:
    return Settings(
        openai_api_key="test-key",
        openai_generation_model="gpt-4.1-mini",
        openai_generation_model_high="gpt-4.1",
    )


def _chunk(text: str = "BCA reported Scope 1 emissions of 24,105 tCO2e.") -> SearchResult:
    payload = ChunkPayload(
        domain=Domain.COMPANY_DOCUMENTS,
        source_name="PT Bank Central Asia Tbk — 2025 Sustainability Report",
        document_type="sustainability_report",
        page_number=8,
        content_type=ContentType.TEXT,
        chunk_text=text,
    )
    return SearchResult(id="c1", score=1.0, payload=payload)


def test_streams_deltas_in_order() -> None:
    client = _FakeStreamClient(["Emissions ", "were ", "24,105 tCO2e [1]."])
    answer = generate_answer(
        "What are BCA's Scope 1 emissions?",
        [_chunk()],
        query_type=QueryType.SINGLE_DOMAIN,
        client=client,
        settings=_settings(),
    )
    assert list(answer.tokens) == ["Emissions ", "were ", "24,105 tCO2e [1]."]


def test_default_model_is_the_mini() -> None:
    client = _FakeStreamClient(["x"])
    answer = generate_answer(
        "q", [_chunk()], query_type=QueryType.SINGLE_DOMAIN, client=client, settings=_settings()
    )
    list(answer.tokens)  # drive the stream so the API call happens
    assert answer.model == "gpt-4.1-mini"
    assert client.create_calls[0]["model"] == "gpt-4.1-mini"


def test_high_accuracy_flag_selects_the_high_model() -> None:
    client = _FakeStreamClient(["x"])
    answer = generate_answer(
        "q",
        [_chunk()],
        query_type=QueryType.SINGLE_DOMAIN,
        high_accuracy=True,
        client=client,
        settings=_settings(),
    )
    list(answer.tokens)
    assert answer.model == "gpt-4.1"
    assert client.create_calls[0]["model"] == "gpt-4.1"


def test_sends_the_right_prompt_template_and_streams_on() -> None:
    client = _FakeStreamClient(["x"])
    answer = generate_answer(
        "q", [_chunk()], query_type=QueryType.SINGLE_DOMAIN, client=client, settings=_settings()
    )
    list(answer.tokens)
    call = client.create_calls[0]
    assert call["messages"][0]["content"] == SINGLE_DOMAIN_SYSTEM_PROMPT
    assert call["stream"] is True


def test_citations_available_before_streaming() -> None:
    client = _FakeStreamClient(["x"])
    answer = generate_answer(
        "q", [_chunk()], query_type=QueryType.SINGLE_DOMAIN, client=client, settings=_settings()
    )
    # Known up front, without consuming any tokens.
    assert len(answer.citations) == 1
    assert answer.citations[0].label == (
        "PT Bank Central Asia Tbk — 2025 Sustainability Report, p. 8"
    )
    assert client.create_calls == []  # nothing sent yet


def test_stream_start_is_lazy() -> None:
    client = _FakeStreamClient(["a", "b"])
    answer = generate_answer(
        "q", [_chunk()], query_type=QueryType.SINGLE_DOMAIN, client=client, settings=_settings()
    )
    assert client.create_calls == []  # returning does not call the API
    next(iter(answer.tokens))
    assert len(client.create_calls) == 1  # first token drives the call


def test_empty_and_none_deltas_are_skipped() -> None:
    client = _FakeStreamClient(["A", None, "", "B"])
    answer = generate_answer(
        "q", [_chunk()], query_type=QueryType.SINGLE_DOMAIN, client=client, settings=_settings()
    )
    assert list(answer.tokens) == ["A", "B"]


def test_compliance_check_is_rejected() -> None:
    client = _FakeStreamClient(["x"])
    with pytest.raises(ValueError, match="compliance"):
        generate_answer(
            "q",
            [_chunk()],
            query_type=QueryType.COMPLIANCE_CHECK,
            client=client,
            settings=_settings(),
        )


def test_missing_api_key_raises_generation_error() -> None:
    with pytest.raises(GenerationError, match="OPENAI_API_KEY"):
        generate_answer(
            "q",
            [_chunk()],
            query_type=QueryType.SINGLE_DOMAIN,
            settings=Settings(openai_api_key=""),
        )
