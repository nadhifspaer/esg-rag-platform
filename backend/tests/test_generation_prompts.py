"""Tests for the chat generation prompt templates."""

from __future__ import annotations

import pytest

from app.models.conversation import ConversationTurn
from app.models.payload import ChunkPayload, ContentType, Domain
from app.rag.generation.prompts import (
    CROSS_DOMAIN_SYSTEM_PROMPT,
    SINGLE_DOMAIN_SYSTEM_PROMPT,
    build_chat_prompt,
    build_cross_domain_prompt,
    build_single_domain_prompt,
    render_sources,
)
from app.rag.query_classifier import QueryType
from app.rag.vector_store import SearchResult


def _chunk(
    source_name: str,
    text: str,
    *,
    page: int = 1,
    domain: Domain = Domain.COMPANY_DOCUMENTS,
    document_type: str = "sustainability_report",
) -> SearchResult:
    payload = ChunkPayload(
        domain=domain,
        source_name=source_name,
        document_type=document_type,
        page_number=page,
        content_type=ContentType.TEXT,
        chunk_text=text,
    )
    return SearchResult(id=f"{source_name}-{page}", score=1.0, payload=payload)


# --- render_sources ---------------------------------------------------------


def test_render_sources_numbers_chunks_and_aligns_citations() -> None:
    chunks = [
        _chunk(
            "PT Bank Central Asia Tbk — 2025 Sustainability Report",
            "Scope 1 emissions were X.",
            page=8,
        ),
        _chunk(
            "PT Bank Mandiri (Persero) Tbk — 2025 Sustainability Report",
            "Energy use was Y.",
            page=4,
        ),
    ]
    text, citations = render_sources(chunks)

    assert "[1]" in text and "[2]" in text
    # Each source's label and text appear in the rendered block.
    assert citations[0].label in text
    assert "Scope 1 emissions were X." in text
    assert citations[1].label in text
    # Ordering: citations[i] corresponds to marker [i+1].
    assert len(citations) == 2
    assert citations[0].page_number == 8
    assert citations[1].page_number == 4


def test_render_sources_empty_is_a_placeholder() -> None:
    text, citations = render_sources([])
    assert citations == []
    assert text == "(no sources retrieved)"


# --- template structure -----------------------------------------------------


def test_single_domain_prompt_structure() -> None:
    chunks = [
        _chunk(
            "GRI 1: Foundation",
            "Reporting principles include accuracy.",
            domain=Domain.STANDARDS,
            document_type="gri_standard",
        )
    ]
    prompt = build_single_domain_prompt("What are the reporting principles?", chunks)

    assert [m["role"] for m in prompt.messages] == ["system", "user"]
    assert prompt.messages[0]["content"] == SINGLE_DOMAIN_SYSTEM_PROMPT
    user = prompt.messages[1]["content"]
    assert "What are the reporting principles?" in user
    assert "Reporting principles include accuracy." in user
    assert len(prompt.citations) == 1


def test_cross_domain_prompt_structure() -> None:
    chunks = [
        _chunk(
            "PT Bank Central Asia Tbk — 2025 Sustainability Report",
            "BCA reports Scope 1 emissions of X.",
            page=8,
        ),
        _chunk(
            "GRI 305: Emissions",
            "Organizations must disclose gross direct GHG emissions.",
            page=2,
            domain=Domain.STANDARDS,
            document_type="gri_standard",
        ),
    ]
    prompt = build_cross_domain_prompt("Does BCA's report align with GRI 305?", chunks)

    assert prompt.messages[0]["content"] == CROSS_DOMAIN_SYSTEM_PROMPT
    user = prompt.messages[1]["content"]
    assert "Does BCA's report align with GRI 305?" in user
    # Both domains' chunks are rendered as numbered sources.
    assert "[1]" in user and "[2]" in user
    assert len(prompt.citations) == 2


def test_both_templates_require_inline_bracket_citations() -> None:
    # The exact marker example the caller parses must be taught in both system prompts.
    assert "[1]" in SINGLE_DOMAIN_SYSTEM_PROMPT
    assert "[1]" in CROSS_DOMAIN_SYSTEM_PROMPT
    assert "square bracket" in SINGLE_DOMAIN_SYSTEM_PROMPT
    assert "square bracket" in CROSS_DOMAIN_SYSTEM_PROMPT


def test_cross_domain_prompt_is_distinct_and_warns_against_conflation() -> None:
    assert CROSS_DOMAIN_SYSTEM_PROMPT != SINGLE_DOMAIN_SYSTEM_PROMPT
    # The core project concern: don't merge a standard/regulation requirement with a
    # company's reported figure.
    assert "distinct" in CROSS_DOMAIN_SYSTEM_PROMPT
    assert "not a certified compliance audit" in CROSS_DOMAIN_SYSTEM_PROMPT


# --- dispatcher -------------------------------------------------------------


def test_build_chat_prompt_dispatches_by_query_type() -> None:
    chunks = [_chunk("PT Bank Central Asia Tbk — 2025 Sustainability Report", "text")]

    single = build_chat_prompt(QueryType.SINGLE_DOMAIN, "q", chunks)
    cross = build_chat_prompt(QueryType.CROSS_DOMAIN, "q", chunks)

    assert single.messages[0]["content"] == SINGLE_DOMAIN_SYSTEM_PROMPT
    assert cross.messages[0]["content"] == CROSS_DOMAIN_SYSTEM_PROMPT


def test_history_is_rendered_as_real_message_turns() -> None:
    """Prior turns become user/assistant messages ahead of the current question."""
    history = [
        ConversationTurn(question="what does BCA report for scope 1", answer="6.032 tCO2e."),
        ConversationTurn(question="and scope 2?", answer="147.851 tCO2e."),
    ]
    chunks = [_chunk("PT Bank Mandiri (Persero) Tbk — 2025 Sustainability Report", "1,37")]

    prompt = build_single_domain_prompt("what about Mandiri?", chunks, history)

    assert [m["role"] for m in prompt.messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert prompt.messages[1]["content"] == "what does BCA report for scope 1"
    assert prompt.messages[2]["content"] == "6.032 tCO2e."
    # The current turn is still the last message, and still carries the SOURCES block.
    assert "SOURCES:" in prompt.messages[-1]["content"]
    assert "what about Mandiri?" in prompt.messages[-1]["content"]


def test_history_adds_rules_about_not_citing_it() -> None:
    """Source numbers restart every turn, so the model must not cite a replayed answer."""
    history = [ConversationTurn(question="q", answer="a")]

    prompt = build_single_domain_prompt("follow-up", [], history)

    system = prompt.messages[0]["content"]
    assert system.startswith(SINGLE_DOMAIN_SYSTEM_PROMPT)
    assert "restarts" in system
    assert "never as evidence" in system


def test_the_current_turn_message_is_unchanged_by_history() -> None:
    """The citation contract lives in this message; history must not perturb it."""
    chunks = [_chunk("GRI 1: Foundation", "Reporting principles include accuracy.")]
    history = [ConversationTurn(question="earlier", answer="earlier answer")]

    without = build_single_domain_prompt("What are the principles?", chunks)
    with_history = build_single_domain_prompt("What are the principles?", chunks, history)

    assert with_history.messages[-1] == without.messages[-1]
    assert with_history.citations == without.citations


def test_no_history_leaves_the_prompt_byte_identical() -> None:
    """A first turn must produce exactly the prompt it produced before multi-turn existed."""
    chunks = [_chunk("GRI 1: Foundation", "Reporting principles include accuracy.")]

    assert (
        build_single_domain_prompt("q", chunks, []).messages
        == build_single_domain_prompt("q", chunks).messages
    )
    assert build_single_domain_prompt("q", chunks, []).messages[0]["content"] == (
        SINGLE_DOMAIN_SYSTEM_PROMPT
    )


def test_build_chat_prompt_threads_history_for_both_query_types() -> None:
    history = [ConversationTurn(question="q", answer="a")]

    for query_type in (QueryType.SINGLE_DOMAIN, QueryType.CROSS_DOMAIN):
        prompt = build_chat_prompt(query_type, "follow-up", [], history)
        assert [m["role"] for m in prompt.messages] == ["system", "user", "assistant", "user"]


def test_build_chat_prompt_rejects_compliance_check() -> None:
    chunks = [_chunk("PT Bank Central Asia Tbk — 2025 Sustainability Report", "text")]
    with pytest.raises(ValueError, match="compliance"):
        build_chat_prompt(QueryType.COMPLIANCE_CHECK, "q", chunks)
