"""Tests for the POST /chat endpoint's routing, validation, and SSE event stream."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api.chat import get_chat_engine
from app.main import app
from app.models.citation import RetrievedChunk
from app.models.payload import ChunkPayload, ContentType, Domain
from app.rag.entity_index import Entity, EntityKind, EntityMatch
from app.rag.generation.generator import GenerationError
from app.rag.generation.loop import StopReason
from app.rag.generation.self_check import GroundednessDecision, Verdict
from app.rag.query_classifier import (
    ClassifierMethod,
    QueryClassification,
    QueryType,
)

client = TestClient(app)


# --- fakes ------------------------------------------------------------------


def _chunk(source: str = "GRI 305", page: int = 7) -> RetrievedChunk:
    """A minimal retrieval result: anything exposing a `.payload` satisfies Citation."""
    payload = ChunkPayload(
        domain=Domain.STANDARDS,
        source_name=source,
        document_type="standard",
        page_number=page,
        content_type=ContentType.TABLE,
        chunk_text="Scope 1 emissions are direct GHG emissions.",
    )
    return SimpleNamespace(payload=payload)


@dataclass
class FakeEngine:
    """Stand-in for ChatEngine: canned classification + trivial loop callables."""

    classification: QueryClassification
    answer_text: str = "Scope 1 covers direct emissions [1]."
    generate_error: bool = False
    generate_unexpected_error: bool = False
    retrieve_source_names: Mapping[Domain, str | None] | None = None
    # Recorded so multi-turn tests can assert decisions otherwise unobservable from the endpoint.
    retrieve_queries: list[str] = field(default_factory=list)
    generate_history: Sequence[object] = ()
    use_llm_fallback: bool | None = None
    # Lets a test exercise chat.py's two-pass classify: successive calls return these in
    # order (last entry repeats once exhausted); None means every call returns `classification`.
    classification_sequence: list[QueryClassification] | None = None
    classify_call_count: int = 0
    settings: SimpleNamespace = field(
        default_factory=lambda: SimpleNamespace(
            openai_generation_model="gpt-4.1-mini",
            openai_generation_model_high="gpt-4.1",
            openai_api_key="test-key",
        )
    )
    openai_client: object | None = object()

    def classify(self, query: str, *, use_llm_fallback: bool = True) -> QueryClassification:
        self.use_llm_fallback = use_llm_fallback
        if self.classification_sequence is not None:
            index = min(self.classify_call_count, len(self.classification_sequence) - 1)
            result = self.classification_sequence[index]
        else:
            result = self.classification
        self.classify_call_count += 1
        return result

    def make_retrieve(
        self, domains: Sequence[Domain], *, source_names: Mapping[Domain, str | None] | None = None
    ):
        # Recorded so tests can assert *what retrieval was scoped to*, which is the
        # endpoint's decision and not otherwise observable from the streamed events.
        self.retrieve_source_names = source_names

        def retrieve(query: str) -> Sequence[RetrievedChunk]:
            self.retrieve_queries.append(query)
            return [_chunk()]

        return retrieve

    def make_generate(
        self, query_type: QueryType, high_accuracy: bool, history: Sequence[object] = ()
    ):
        self.generate_history = history

        def generate(question: str, chunks: Sequence[RetrievedChunk]) -> str:
            if self.generate_error:
                raise GenerationError("boom: generation failed")
            if self.generate_unexpected_error:
                # A failure type the endpoint's except clauses don't specifically name.
                raise RuntimeError("internal: qdrant collection esg_documents_v3 timed out")
            return self.answer_text

        return generate

    def make_check(self):
        def check(question: str, answer: str, chunks: Sequence[RetrievedChunk]):
            return GroundednessDecision(verdict=Verdict.SUFFICIENT, rewritten_query=None)

        return check


def _classification(
    query_type: QueryType = QueryType.SINGLE_DOMAIN,
    domains: tuple[Domain, ...] = (Domain.STANDARDS,),
    entities: tuple[EntityMatch, ...] = (),
) -> QueryClassification:
    return QueryClassification(
        query_type=query_type,
        domains=domains,
        entities=entities,
        method=ClassifierMethod.RULE_BASED,
        confident=True,
        reason="test",
    )


def _use_engine(engine: FakeEngine) -> None:
    app.dependency_overrides[get_chat_engine] = lambda: engine


@pytest.fixture(autouse=True)
def _clear_overrides() -> Iterator[None]:
    yield
    app.dependency_overrides.clear()


# --- SSE parsing helper -----------------------------------------------------


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    """Parse an SSE body into an ordered list of (event, data-dict) pairs."""
    events: list[tuple[str, dict]] = []
    for block in body.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event = ""
        data = ""
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data = line[len("data:") :].strip()
        events.append((event, json.loads(data) if data else {}))
    return events


# --- happy path -------------------------------------------------------------


def test_chat_streams_meta_citations_tokens_and_done() -> None:
    _use_engine(FakeEngine(_classification()))

    response = client.post("/chat", json={"query": "What is Scope 1?"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(response.text)
    names = [name for name, _ in events]

    # meta first, then citations, then one or more token events, then done last.
    assert names[0] == "meta"
    assert "citations" in names
    assert "done" in names
    assert names.index("citations") < names.index("token") < names.index("done")
    assert names[-1] == "done"

    meta = dict(events)["meta"]
    assert meta["query_type"] == "single_domain"
    assert meta["domains"] == ["standards"]
    assert meta["model"] == "gpt-4.1-mini"

    # The streamed token slices reconstruct the full answer exactly.
    answer = "".join(data["text"] for name, data in events if name == "token")
    assert answer == "Scope 1 covers direct emissions [1]."

    citations = dict(events)["citations"]["citations"]
    assert citations[0]["index"] == 1
    assert citations[0]["label"] == "GRI 305, p. 7 (table)"

    done = dict(events)["done"]
    assert done["attempts"] == 1
    assert done["retries"] == 0
    assert done["stop_reason"] == StopReason.SUFFICIENT.value


def test_high_accuracy_toggle_selects_high_model() -> None:
    _use_engine(FakeEngine(_classification()))

    response = client.post("/chat", json={"query": "What is Scope 1?", "high_accuracy": True})

    assert response.status_code == 200
    meta = dict(_parse_sse(response.text))["meta"]
    assert meta["model"] == "gpt-4.1"
    assert meta["high_accuracy"] is True


def test_explicit_domain_overrides_classifier() -> None:
    # Classifier would pick standards; the request forces company_documents.
    _use_engine(FakeEngine(_classification(domains=(Domain.STANDARDS,))))

    response = client.post("/chat", json={"query": "BCA emissions", "domain": "company_documents"})

    assert response.status_code == 200
    meta = dict(_parse_sse(response.text))["meta"]
    assert meta["domains"] == ["company_documents"]


# --- retrieval scoping ------------------------------------------------------


def _bank_entity(canonical: str = "PT Bank Central Asia Tbk") -> EntityMatch:
    return EntityMatch(
        entity=Entity(
            kind=EntityKind.BANK,
            domain=Domain.COMPANY_DOCUMENTS,
            canonical_name=canonical,
            surface_forms=("bca",),
            bank=canonical,
        ),
        matched_text="bca",
    )


def test_a_single_named_bank_scopes_retrieval_to_that_report() -> None:
    """The endpoint must both *use* the scope and *report* it."""
    engine = FakeEngine(
        _classification(domains=(Domain.COMPANY_DOCUMENTS,), entities=(_bank_entity(),))
    )
    _use_engine(engine)

    response = client.post("/chat", json={"query": "what does BCA report for scope 1 emissions"})

    assert response.status_code == 200
    expected = "PT Bank Central Asia Tbk — 2025 Sustainability Report"
    assert engine.retrieve_source_names == {Domain.COMPANY_DOCUMENTS: expected}
    assert dict(_parse_sse(response.text))["meta"]["scoped_to"] == {"company_documents": expected}


def test_a_query_naming_no_bank_searches_the_domain_unscoped() -> None:
    engine = FakeEngine(_classification(domains=(Domain.COMPANY_DOCUMENTS,)))
    _use_engine(engine)

    response = client.post("/chat", json={"query": "what are typical scope 1 emissions"})

    assert response.status_code == 200
    assert engine.retrieve_source_names == {Domain.COMPANY_DOCUMENTS: None}
    assert dict(_parse_sse(response.text))["meta"]["scoped_to"] == {"company_documents": None}


def test_a_cross_domain_query_scopes_company_documents_but_not_standards() -> None:
    """Naming one bank scopes the `company_documents` leg to it; `standards` stays domain-wide."""
    engine = FakeEngine(
        _classification(
            query_type=QueryType.CROSS_DOMAIN,
            domains=(Domain.COMPANY_DOCUMENTS, Domain.STANDARDS),
            entities=(_bank_entity(),),
        )
    )
    _use_engine(engine)

    response = client.post("/chat", json={"query": "does BCA's report align with GRI 305"})

    assert response.status_code == 200
    expected = "PT Bank Central Asia Tbk — 2025 Sustainability Report"
    assert engine.retrieve_source_names == {
        Domain.COMPANY_DOCUMENTS: expected,
        Domain.STANDARDS: None,
    }
    assert dict(_parse_sse(response.text))["meta"]["scoped_to"] == {
        "company_documents": expected,
        "standards": None,
    }


# --- routing validation -----------------------------------------------------


def test_compliance_check_query_is_rejected() -> None:
    _use_engine(FakeEngine(_classification(query_type=QueryType.COMPLIANCE_CHECK)))

    response = client.post("/chat", json={"query": "Does BCA comply with GRI 305?"})

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "compliance" in detail["message"].lower()
    assert detail["query_type"] == "compliance_check"


def test_compliance_check_rejection_names_the_resolved_company() -> None:
    """A frontend auto-routing this to /compliance-check needs the resolved company, not a
    string to re-parse: this is the caller-facing contract that closes that gap."""
    _use_engine(
        FakeEngine(
            _classification(query_type=QueryType.COMPLIANCE_CHECK, entities=(_bank_entity(),))
        )
    )

    response = client.post(
        "/chat", json={"query": "Does BCA's report cover Scope 1 emissions per GRI 305-1?"}
    )

    detail = response.json()["detail"]
    assert detail["company"] == "PT Bank Central Asia Tbk"
    # The real environmental checklist's topic_keywords include "scope 1"/"emissions", so this
    # resolves a category too, pinned so a future keyword-list edit that breaks it is caught.
    assert detail["category"] == "environmental"


def test_compliance_check_rejection_reports_no_company_when_ambiguous() -> None:
    """Two banks named (or none) is exactly the case the compliance endpoint itself would
    422 on as ambiguous/not-found: the frontend must not be handed a guess here."""
    _use_engine(FakeEngine(_classification(query_type=QueryType.COMPLIANCE_CHECK, entities=())))

    response = client.post("/chat", json={"query": "Does this report comply with GRI 305?"})

    assert response.json()["detail"]["company"] is None


def test_aggregation_ranking_rejection_is_structured() -> None:
    _use_engine(FakeEngine(_classification(query_type=QueryType.AGGREGATION_RANKING)))

    response = client.post("/chat", json={"query": "which bank has the earliest net-zero target"})

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["query_type"] == "aggregation_ranking"
    assert "ranking" in detail["message"].lower()


def test_undetermined_domain_is_rejected() -> None:
    # Classifier resolved no domain and the request gave no filter.
    _use_engine(FakeEngine(_classification(domains=())))

    response = client.post("/chat", json={"query": "tell me about sustainability"})

    assert response.status_code == 422
    assert "domain" in response.json()["detail"].lower()


def test_undetermined_domain_is_answerable_with_explicit_filter() -> None:
    # Same unresolved classification, but the request supplies the domain.
    _use_engine(FakeEngine(_classification(domains=())))

    response = client.post(
        "/chat", json={"query": "tell me about sustainability", "domain": "company_documents"}
    )

    assert response.status_code == 200
    meta = dict(_parse_sse(response.text))["meta"]
    assert meta["domains"] == ["company_documents"]


def test_missing_engine_returns_503() -> None:
    # No dependency override and no engine on app.state -> service unavailable.
    response = client.post("/chat", json={"query": "What is Scope 1?"})
    assert response.status_code == 503


def test_empty_query_is_rejected() -> None:
    _use_engine(FakeEngine(_classification()))
    response = client.post("/chat", json={"query": "   "})
    assert response.status_code == 422


# --- multi-turn conversation (real manifest-backed entity index, as in production) ------

_BCA_TURN = {
    "question": "what does BCA report for scope 1 emissions",
    "answer": "BCA reports total Scope 1 emissions of 6.032 tCO2e [1].",
    "retrieval_query": "BCA Scope 1 emissions",
}


def test_an_entity_swap_follow_up_searches_the_previous_topic() -> None:
    """ "What about Mandiri?" must not be searched literally: it has no topical terms."""
    engine = FakeEngine(_classification(domains=(Domain.COMPANY_DOCUMENTS,)))
    _use_engine(engine)

    response = client.post("/chat", json={"query": "what about Mandiri?", "history": [_BCA_TURN]})

    assert response.status_code == 200
    # The seed reached retrieval, not the raw utterance.
    assert engine.retrieve_queries == ["mandiri Scope 1 emissions"]
    meta = dict(_parse_sse(response.text))["meta"]
    assert meta["context_source"] == "entity_swap"
    assert meta["retrieval_seed"] == "mandiri Scope 1 emissions"
    # The question the user actually typed is what gets answered.
    assert meta["resolved_question"] == "what about Mandiri?"


def test_a_topic_shift_follow_up_inherits_the_bank_and_skips_the_llm_fallback() -> None:
    """The unresolved classification is what the rule pass returns for "and waste?"; the
    inherited entity is what lets the endpoint scope retrieval instead of sweeping 21 banks."""
    engine = FakeEngine(_classification(domains=(), entities=()))
    _use_engine(engine)

    response = client.post("/chat", json={"query": "and waste?", "history": [_BCA_TURN]})

    assert response.status_code == 200
    meta = dict(_parse_sse(response.text))["meta"]
    assert meta["context_source"] == "inherited_entity"
    assert meta["inherited_entity"] == "PT Bank Central Asia Tbk"
    # apply_context supplied the domain the classifier could not resolve...
    assert meta["domains"] == ["company_documents"]
    # ...and retrieval was confined to that bank's report rather than searching all 21.
    assert engine.retrieve_source_names == {
        Domain.COMPANY_DOCUMENTS: "PT Bank Central Asia Tbk — 2025 Sustainability Report"
    }
    # The classifier's paid LLM fallback is skipped when the conversation already answers it.
    assert engine.use_llm_fallback is False
    # The topic itself is not overwritten by the inherited bank name.
    assert engine.retrieve_queries == ["and waste?"]


def test_a_first_turn_still_uses_the_llm_fallback() -> None:
    """Skipping the fallback is specific to inheritance; a first turn still escalates if unsure."""
    rule_guess = QueryClassification(
        query_type=QueryType.SINGLE_DOMAIN,
        domains=(),
        entities=(),
        method=ClassifierMethod.RULE_BASED,
        confident=False,
        reason="no known entity named; domain undetermined by rules",
    )
    llm_result = _classification(domains=(Domain.STANDARDS,))
    engine = FakeEngine(classification=llm_result, classification_sequence=[rule_guess, llm_result])
    _use_engine(engine)

    response = client.post("/chat", json={"query": "what is Scope 1?"})

    assert response.status_code == 200
    assert engine.use_llm_fallback is True
    assert engine.classify_call_count == 2
    assert dict(_parse_sse(response.text))["meta"]["context_source"] == "utterance"


def test_history_is_sanitized_before_reaching_generation() -> None:
    """Stale `[n]` markers must not survive into the prompt: numbering restarts each turn."""
    engine = FakeEngine(_classification(domains=(Domain.COMPANY_DOCUMENTS,)))
    _use_engine(engine)

    response = client.post("/chat", json={"query": "what about Mandiri?", "history": [_BCA_TURN]})

    assert response.status_code == 200
    assert len(engine.generate_history) == 1
    assert "[1]" not in engine.generate_history[0].answer
    assert engine.generate_history[0].answer.endswith("6.032 tCO2e.")


def test_conversation_id_is_echoed_but_never_required() -> None:
    engine = FakeEngine(_classification(domains=(Domain.STANDARDS,)))
    _use_engine(engine)

    with_id = client.post(
        "/chat", json={"query": "what is Scope 1?", "conversation_id": "conv-abc"}
    )
    assert dict(_parse_sse(with_id.text))["meta"]["conversation_id"] == "conv-abc"

    without = client.post("/chat", json={"query": "what is Scope 1?"})
    assert dict(_parse_sse(without.text))["meta"]["conversation_id"] is None


def test_two_conversations_under_the_same_user_do_not_influence_each_other() -> None:
    """The leak guarantee: both requests share one Supabase subject, so nothing may leak between."""
    engine = FakeEngine(_classification(domains=(Domain.COMPANY_DOCUMENTS,)))
    _use_engine(engine)

    first = client.post(
        "/chat",
        json={
            "query": "what about Mandiri?",
            "history": [_BCA_TURN],
            "conversation_id": "visitor-one",
        },
    )
    assert dict(_parse_sse(first.text))["meta"]["retrieval_seed"] == ("mandiri Scope 1 emissions")

    # A different visitor on the same shared account, with no history of their own.
    second = client.post(
        "/chat", json={"query": "what about Mandiri?", "conversation_id": "visitor-two"}
    )
    second_meta = dict(_parse_sse(second.text))["meta"]
    assert second_meta["history_turns"] == 0
    assert second_meta["context_source"] == "utterance"
    # No trace of visitor one's topic.
    assert second_meta["retrieval_seed"] == "what about Mandiri?"


def test_a_compliance_follow_up_is_refused_with_the_company_named() -> None:
    """Answering here would return a cited lookup that reads like a compliance verdict."""
    _use_engine(FakeEngine(_classification(domains=(Domain.COMPANY_DOCUMENTS,))))

    response = client.post(
        "/chat",
        json={
            "query": "and Mandiri?",
            "history": [
                {
                    "question": "does BCA comply with GRI 305?",
                    "answer": "| GRI 305-1 | Addressed | ... |",
                    "pipeline": "compliance",
                }
            ],
        },
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["query_type"] == "compliance_check"
    assert detail["company"] == "PT Bank Mandiri (Persero) Tbk"
    assert detail["category"] is None
    assert "/compliance-check" in detail["message"]


def test_aggregation_ranking_follow_up_with_no_keyword_still_escalates_to_llm() -> None:
    """A follow-up naming no entity of its own must still get a shot at the LLM fallback."""
    rule_guess = QueryClassification(
        query_type=QueryType.SINGLE_DOMAIN,
        domains=(),
        entities=(),
        method=ClassifierMethod.RULE_BASED,
        confident=False,
        reason="no known entity named; domain undetermined by rules",
    )
    llm_result = _classification(
        query_type=QueryType.AGGREGATION_RANKING, domains=(Domain.COMPANY_DOCUMENTS,)
    )
    engine = FakeEngine(classification=llm_result, classification_sequence=[rule_guess, llm_result])
    _use_engine(engine)

    response = client.post(
        "/chat",
        json={"query": "how does this look across the industry?", "history": [_BCA_TURN]},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["query_type"] == "aggregation_ranking"
    # Proves the escalation actually happened: first call rule-only, second with the
    # fallback enabled, rather than the assertion above passing by coincidence.
    assert engine.classify_call_count == 2


# --- history validation -----------------------------------------------------


def test_history_over_the_turn_cap_is_rejected() -> None:
    _use_engine(FakeEngine(_classification()))

    response = client.post("/chat", json={"query": "and waste?", "history": [_BCA_TURN] * 9})

    assert response.status_code == 422


def test_an_over_long_history_turn_is_rejected() -> None:
    """A loud 422 rather than a silent trim: the client's prompt budget is bounded here."""
    _use_engine(FakeEngine(_classification()))

    response = client.post(
        "/chat",
        json={
            "query": "and waste?",
            "history": [{"question": "q", "answer": "x" * 4001}],
        },
    )

    assert response.status_code == 422


def test_an_empty_history_turn_is_rejected() -> None:
    _use_engine(FakeEngine(_classification()))

    response = client.post(
        "/chat", json={"query": "and waste?", "history": [{"question": "", "answer": "a"}]}
    )

    assert response.status_code == 422


# --- mid-stream error -------------------------------------------------------


def test_generation_failure_emits_error_event() -> None:
    _use_engine(FakeEngine(_classification(), generate_error=True))

    response = client.post("/chat", json={"query": "What is Scope 1?"})

    # Streaming already started (200 + meta sent), so the failure surfaces as an SSE
    # error event, not an HTTP error code.
    assert response.status_code == 200
    events = _parse_sse(response.text)
    names = [name for name, _ in events]
    assert names[0] == "meta"
    assert "error" in names
    assert "token" not in names and "done" not in names
    assert "boom" in dict(events)["error"]["message"]


def test_unexpected_generation_error_still_emits_error_event_not_silent_truncation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unanticipated failure must still emit a visible `error` event, not truncate silently."""
    _use_engine(FakeEngine(_classification(), generate_unexpected_error=True))

    with caplog.at_level("ERROR"):
        response = client.post("/chat", json={"query": "What is Scope 1?"})

    assert response.status_code == 200
    events = _parse_sse(response.text)
    names = [name for name, _ in events]
    assert names[0] == "meta"
    assert "error" in names
    assert "token" not in names and "done" not in names

    # The client gets a generic message, never the raw exception string, which could
    # leak internals (here, a fake collection name standing in for something real).
    message = dict(events)["error"]["message"]
    assert "qdrant" not in message.lower()
    assert "esg_documents_v3" not in message
    assert message == "Something went wrong while generating this answer. Please try again."

    # But the real error, with a traceback, is captured server-side for Railway logs.
    error_records = [r for r in caplog.records if r.levelname == "ERROR"]
    assert any("unexpected error" in r.message for r in error_records)
    assert any(r.exc_info is not None for r in error_records)
    assert "RuntimeError" in caplog.text
    assert "qdrant collection esg_documents_v3 timed out" in caplog.text
