"""Tests for the query classifier: rule-based routing and the LLM fallback."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.models.payload import Domain
from app.rag.query_classifier import (
    ClassifierMethod,
    QueryClassifierError,
    QueryType,
    classify_query,
)


class _FakeChatClient:
    """Minimal stand-in for `OpenAI`: returns a preset JSON body from chat.completions
    and records whether it was called, so tests can assert the fallback fired (or not)."""

    def __init__(self, reply: dict) -> None:
        self._reply = json.dumps(reply)
        self.called = False

        outer = self

        class _Completions:
            def create(self, **kwargs):  # noqa: ANN003 - mirrors the OpenAI signature
                outer.called = True
                message = SimpleNamespace(content=outer._reply)
                return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        self.chat = SimpleNamespace(completions=_Completions())


def _settings() -> Settings:
    # A key value so the fallback would build a client if not injected; the fake is
    # injected in every test, so no real call is ever made.
    return Settings(openai_api_key="test-key")


# --- rule-based pass (no LLM needed) ---------------------------------------


def test_single_domain_when_one_bank_named() -> None:
    client = _FakeChatClient({})
    result = classify_query("what are BCA's scope 1 emissions", client=client, settings=_settings())

    assert result.query_type is QueryType.SINGLE_DOMAIN
    assert result.domains == (Domain.COMPANY_DOCUMENTS,)
    assert result.method is ClassifierMethod.RULE_BASED
    assert result.confident is True
    assert client.called is False  # rules were confident; no fallback


def test_cross_domain_when_entities_span_domains() -> None:
    client = _FakeChatClient({})
    result = classify_query(
        "does BCA's report mention GRI 305", client=client, settings=_settings()
    )

    assert result.query_type is QueryType.CROSS_DOMAIN
    assert set(result.domains) == {Domain.COMPANY_DOCUMENTS, Domain.STANDARDS}
    assert client.called is False


def test_compliance_check_when_company_and_standard_named_with_intent() -> None:
    client = _FakeChatClient({})
    result = classify_query(
        "does BCA's sustainability report comply with GRI 3",
        client=client,
        settings=_settings(),
    )

    assert result.query_type is QueryType.COMPLIANCE_CHECK
    assert Domain.COMPANY_DOCUMENTS in result.domains
    assert Domain.STANDARDS in result.domains
    assert result.confident is True
    assert client.called is False


# --- LLM fallback -----------------------------------------------------------


def test_topic_query_with_no_entity_escalates_to_llm() -> None:
    client = _FakeChatClient(
        {
            "query_type": "single_domain",
            "domains": ["standards"],
            "entities": [],
            "reasoning": "general question about GRI disclosure requirements",
        }
    )
    result = classify_query(
        "what does a foundation-level disclosure require", client=client, settings=_settings()
    )

    assert client.called is True  # rules found no entity -> fallback fired
    assert result.method is ClassifierMethod.LLM_FALLBACK
    assert result.query_type is QueryType.SINGLE_DOMAIN
    assert result.domains == (Domain.STANDARDS,)


def test_ambiguous_compliance_query_escalates_and_resolves_entities() -> None:
    # "comply" + a bank, but no standard/regulation named -> rules unconfident -> LLM.
    client = _FakeChatClient(
        {
            "query_type": "compliance_check",
            "domains": ["company_documents", "standards"],
            "entities": ["GRI 3"],  # returned as an alias; must resolve to the entity
            "reasoning": "checking BCA against GRI 3",
        }
    )
    result = classify_query(
        "is BCA compliant with the standard", client=client, settings=_settings()
    )

    assert client.called is True
    assert result.query_type is QueryType.COMPLIANCE_CHECK
    assert set(result.domains) == {Domain.COMPANY_DOCUMENTS, Domain.STANDARDS}
    # The alias "GRI 3" was grounded back to the real standard entity.
    assert any(e.entity.canonical_name == "GRI 3: Material Topics" for e in result.entities)


def test_fallback_disabled_returns_rule_best_guess_without_calling_llm() -> None:
    client = _FakeChatClient({})
    result = classify_query(
        "what does a foundation-level disclosure require",
        client=client,
        settings=_settings(),
        use_llm_fallback=False,
    )

    assert client.called is False
    assert result.method is ClassifierMethod.RULE_BASED
    assert result.confident is False
    assert result.domains == ()  # domain undetermined, not "all domains"


def test_missing_api_key_degrades_gracefully_to_rule_guess() -> None:
    # No injected client and no key: the fallback can't run, so we keep the rule guess.
    result = classify_query(
        "what does a foundation-level disclosure require",
        settings=Settings(openai_api_key=""),
    )

    assert result.method is ClassifierMethod.RULE_BASED
    assert result.confident is False


def test_bad_llm_json_raises_inside_helper() -> None:
    # Guards the parser directly: a non-JSON body is a typed error, not a crash.
    from app.rag.entity_index import default_entity_index
    from app.rag.query_classifier import _parse_llm_response

    with pytest.raises(QueryClassifierError):
        _parse_llm_response("not json", default_entity_index())
