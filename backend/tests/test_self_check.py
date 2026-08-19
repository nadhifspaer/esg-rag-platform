"""Tests for the groundedness self-check judge."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.models.payload import ChunkPayload, ContentType, Domain
from app.rag.generation.self_check import (
    JUDGE_SYSTEM_PROMPT,
    GroundednessDecision,
    SelfCheckError,
    Verdict,
    check_groundedness,
    strip_search_operators,
)
from app.rag.vector_store import SearchResult


class _FakeJudgeClient:
    """Fake OpenAI client returning a preset JSON body and recording the call kwargs."""

    def __init__(self, reply: dict | str) -> None:
        self._reply = reply if isinstance(reply, str) else json.dumps(reply)
        self.create_calls: list[dict] = []
        outer = self

        class _Completions:
            def create(self, **kwargs):  # noqa: ANN003 - mirrors the OpenAI signature
                outer.create_calls.append(kwargs)
                message = SimpleNamespace(content=outer._reply)
                return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        self.chat = SimpleNamespace(completions=_Completions())


def _settings() -> Settings:
    return Settings(openai_api_key="test-key", openai_generation_model="gpt-4.1-mini")


def _chunk(text: str, source: str = "GRI 1: Foundation") -> SearchResult:
    payload = ChunkPayload(
        domain=Domain.STANDARDS,
        source_name=source,
        document_type="gri_standard",
        page_number=4,
        content_type=ContentType.TEXT,
        chunk_text=text,
    )
    return SearchResult(id="c1", score=1.0, payload=payload)


def test_sufficient_drops_the_rewritten_query() -> None:
    client = _FakeJudgeClient({"verdict": "sufficient", "rewritten_query": "ignored"})
    decision = check_groundedness(
        "What are the GRI reporting principles?",
        "They include accuracy and clarity [1].",
        [_chunk("Reporting principles: accuracy, clarity.")],
        client=client,
        settings=_settings(),
    )
    assert decision == GroundednessDecision(verdict=Verdict.SUFFICIENT, rewritten_query=None)


def test_insufficient_keeps_the_rewritten_query() -> None:
    client = _FakeJudgeClient(
        {
            "verdict": "insufficient",
            "rewritten_query": "PT Bank Central Asia Tbk Scope 1 Scope 2 emissions tCO2e 2024",
        }
    )
    decision = check_groundedness(
        "What are BCA's Scope 1 emissions?",
        "The context does not contain BCA's emissions.",
        [_chunk("Reporting principles: accuracy, clarity.")],
        client=client,
        settings=_settings(),
    )
    assert decision.verdict is Verdict.INSUFFICIENT
    assert decision.rewritten_query == (
        "PT Bank Central Asia Tbk Scope 1 Scope 2 emissions tCO2e 2024"
    )


def test_insufficient_with_empty_rewrite_falls_back_to_original_query() -> None:
    client = _FakeJudgeClient({"verdict": "insufficient", "rewritten_query": ""})
    decision = check_groundedness(
        "What are BCA's Scope 1 emissions?",
        "Not available.",
        [_chunk("unrelated text")],
        client=client,
        settings=_settings(),
    )
    assert decision.verdict is Verdict.INSUFFICIENT
    assert decision.rewritten_query == "What are BCA's Scope 1 emissions?"


def test_prompt_carries_question_answer_and_context() -> None:
    client = _FakeJudgeClient({"verdict": "sufficient", "rewritten_query": ""})
    check_groundedness(
        "What are the GRI reporting principles?",
        "Accuracy and clarity [1].",
        [_chunk("Reporting principles: accuracy, clarity.")],
        client=client,
        settings=_settings(),
    )
    user = client.create_calls[0]["messages"][1]["content"]
    assert "What are the GRI reporting principles?" in user
    assert "Accuracy and clarity [1]." in user
    assert "Reporting principles: accuracy, clarity." in user  # the rendered context


def test_bad_json_raises() -> None:
    client = _FakeJudgeClient("not json")
    with pytest.raises(SelfCheckError, match="valid JSON"):
        check_groundedness("q", "a", [_chunk("t")], client=client, settings=_settings())


def test_unknown_verdict_raises() -> None:
    client = _FakeJudgeClient({"verdict": "maybe", "rewritten_query": ""})
    with pytest.raises(SelfCheckError, match="unknown verdict"):
        check_groundedness("q", "a", [_chunk("t")], client=client, settings=_settings())


def test_missing_api_key_raises() -> None:
    with pytest.raises(SelfCheckError, match="OPENAI_API_KEY"):
        check_groundedness("q", "a", [_chunk("t")], settings=Settings(openai_api_key=""))


# --- rewritten-query sanitisation (neither index implements web-search operators) -------


def test_strips_the_operators_seen_in_the_real_failure() -> None:
    assert (
        strip_search_operators("BCA scope 1 emissions data site:bca.co.id filetype:pdf")
        == "BCA scope 1 emissions data"
    )


@pytest.mark.parametrize(
    "operator",
    ["site:example.com", "filetype:pdf", "inurl:esg", "intitle:report", "ext:xlsx", "-site:x.com"],
)
def test_strips_each_known_operator(operator: str) -> None:
    assert strip_search_operators(f"scope 1 emissions {operator}") == "scope 1 emissions"


def test_strips_bare_urls() -> None:
    assert strip_search_operators("emissions https://bca.co.id/report www.bca.co.id") == "emissions"


def test_preserves_standard_codes() -> None:
    """The identifiers that matter most must survive untouched (colons and hyphens included)."""
    query = "GRI 305-1 laporan keberlanjutan EDGB E-04: Emissions 2016"
    assert strip_search_operators(query) == query


def test_an_all_operator_query_falls_back_rather_than_emptying() -> None:
    assert strip_search_operators("site:bca.co.id filetype:pdf") == "site:bca.co.id filetype:pdf"


def test_sanitisation_is_applied_to_the_judges_output() -> None:
    """End to end: a junk rewrite never reaches the retry."""
    client = _FakeJudgeClient(
        {"verdict": "insufficient", "rewritten_query": "BCA scope 1 site:bca.co.id filetype:pdf"}
    )
    decision = check_groundedness("q", "a", [_chunk("t")], client=client, settings=_settings())
    assert decision.rewritten_query == "BCA scope 1"


def test_the_judge_prompt_forbids_operator_syntax() -> None:
    """The prompt rule and the code guard are both required; this pins the prompt half."""
    assert "site:" in JUDGE_SYSTEM_PROMPT and "filetype:" in JUDGE_SYSTEM_PROMPT
    assert "NOT a web search" in JUDGE_SYSTEM_PROMPT


# --- absence-of-data must trigger a retry (prompt-only fix; pins the rule's presence) ----


def test_the_prompt_states_the_absence_rule_as_overriding() -> None:
    assert "OVERRIDING RULE" in JUDGE_SYSTEM_PROMPT
    assert "the verdict is ALWAYS" in JUDGE_SYSTEM_PROMPT


def test_the_prompt_distinguishes_an_absent_disclosure_from_a_negative_one() -> None:
    """A report answering "Tidak" HAS disclosed its position: that must stay sufficient,
    or every correctly-answered "does this bank have a target?" would retry pointlessly."""
    assert "Tidak" in JUDGE_SYSTEM_PROMPT
    assert "absent disclosure with a negative one" in JUDGE_SYSTEM_PROMPT


def test_sufficient_requires_stating_the_asked_for_facts() -> None:
    assert "ALL THREE" in JUDGE_SYSTEM_PROMPT
    assert "is not, in whole or in part, a report that the information is missing" in (
        JUDGE_SYSTEM_PROMPT
    )
