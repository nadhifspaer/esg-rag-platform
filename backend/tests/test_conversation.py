"""Tests for multi-turn context resolution (`app/rag/conversation.py`)."""

from __future__ import annotations

import pytest

from app.models.conversation import ConversationTurn, Pipeline
from app.models.payload import Domain
from app.rag.conversation import (
    ContextSource,
    apply_context,
    resolve_context,
    sanitize_history,
    strip_citation_markers,
)
from app.rag.entity_index import Entity, EntityIndex, EntityKind
from app.rag.query_classifier import ClassifierMethod, QueryClassification, QueryType

# --- fixtures ---------------------------------------------------------------

BCA = Entity(
    kind=EntityKind.BANK,
    domain=Domain.COMPANY_DOCUMENTS,
    canonical_name="PT Bank Central Asia Tbk",
    surface_forms=("pt bank central asia tbk", "bca"),
    bank="PT Bank Central Asia Tbk",
)
MANDIRI = Entity(
    kind=EntityKind.BANK,
    domain=Domain.COMPANY_DOCUMENTS,
    canonical_name="PT Bank Mandiri (Persero) Tbk",
    surface_forms=("pt bank mandiri (persero) tbk", "mandiri", "bmri"),
    bank="PT Bank Mandiri (Persero) Tbk",
)
GRI_305 = Entity(
    kind=EntityKind.STANDARD,
    domain=Domain.STANDARDS,
    canonical_name="GRI 305: Emissions 2016",
    surface_forms=("gri 305: emissions 2016", "gri 305"),
    official_title="GRI 305: Emissions 2016",
)


@pytest.fixture
def index() -> EntityIndex:
    return EntityIndex([BCA, MANDIRI, GRI_305])


def _turn(
    question: str,
    answer: str = "BCA reports total Scope 1 emissions of 6.032 tCO2e [1].",
    retrieval_query: str | None = "BCA Scope 1 emissions",
    pipeline: Pipeline = Pipeline.CHAT,
) -> ConversationTurn:
    return ConversationTurn(
        question=question, answer=answer, retrieval_query=retrieval_query, pipeline=pipeline
    )


def _classification(
    domains: tuple[Domain, ...] = (),
    entities: tuple = (),
    confident: bool = False,
) -> QueryClassification:
    return QueryClassification(
        query_type=QueryType.SINGLE_DOMAIN,
        domains=domains,
        entities=entities,
        method=ClassifierMethod.RULE_BASED,
        confident=confident,
        reason="no known entity named; domain undetermined by rules",
    )


# --- shape 1: no history (the regression guard) -----------------------------


def test_no_history_returns_the_utterance_untouched(index: EntityIndex) -> None:
    """A first turn must behave exactly as it did before multi-turn existed."""
    utterance = "what does BCA report for scope 1 emissions"

    context = resolve_context(utterance, [], index=index)

    assert context.resolved_question == utterance
    assert context.retrieval_seed == utterance
    assert context.context_source is ContextSource.UTTERANCE
    assert context.inherited_entity is None
    assert context.compliance_follow_up is False


def test_a_self_contained_follow_up_is_also_used_as_is(index: EntityIndex) -> None:
    """History present, but the utterance names both its entity and its topic."""
    history = [_turn("what does BCA report for scope 1 emissions")]

    context = resolve_context("what is Mandiri's waste generation", history, index=index)

    assert context.retrieval_seed == "what is Mandiri's waste generation"
    assert context.context_source is ContextSource.UTTERANCE
    # The utterance named its own entity, so nothing is inherited.
    assert context.inherited_entity is None


# --- shape 2: entity-swap ---------------------------------------------------


def test_entity_swap_substitutes_the_new_bank_into_the_previous_search(
    index: EntityIndex,
) -> None:
    """ "What about Mandiri?": the topic comes from the prior turn's search terms."""
    history = [_turn("what does BCA report for scope 1 emissions")]

    context = resolve_context("what about Mandiri?", history, index=index)

    assert context.context_source is ContextSource.ENTITY_SWAP
    # BEFORE (what a no-history resolution still produces):
    assert resolve_context("what about Mandiri?", [], index=index).retrieval_seed == (
        "what about Mandiri?"
    )
    # AFTER:
    assert context.retrieval_seed == "mandiri Scope 1 emissions"
    # The question being answered is still what the user actually typed.
    assert context.resolved_question == "what about Mandiri?"


def test_entity_swap_falls_back_to_the_previous_question_when_no_search_query_is_echoed(
    index: EntityIndex,
) -> None:
    """A client that does not echo `retrieval_query` still gets a usable seed."""
    history = [_turn("BCA scope 1 emissions", retrieval_query=None)]

    context = resolve_context("and Mandiri?", history, index=index)

    assert context.retrieval_seed == "mandiri scope 1 emissions"


def test_entity_swap_leaves_the_seed_alone_when_it_names_no_previous_entity(
    index: EntityIndex,
) -> None:
    """No name to replace: the metadata filter already confines the search to one report,
    so injecting the bank name would only add the boilerplate-promoting term back."""
    history = [_turn("total direct emissions", retrieval_query="total direct emissions scope 1")]

    context = resolve_context("what about Mandiri?", history, index=index)

    assert context.retrieval_seed == "total direct emissions scope 1"


# --- shape 3: topic-shift ---------------------------------------------------


def test_topic_shift_inherits_the_entity_but_searches_the_utterance(
    index: EntityIndex,
) -> None:
    """ "And waste?": the entity is inherited, the topic stays (bank name not in the seed)."""
    history = [_turn("what does BCA report for scope 1 emissions")]

    context = resolve_context("and waste?", history, index=index)

    assert context.context_source is ContextSource.INHERITED_ENTITY
    assert context.retrieval_seed == "and waste?"
    assert context.inherited_entity is not None
    assert context.inherited_entity.entity is BCA


# --- shape 4: pure reference ------------------------------------------------


@pytest.mark.parametrize(
    "utterance",
    ["explain that more simply", "why?", "jelaskan lagi", "what does that mean"],
)
def test_pure_reference_reuses_the_previous_search(index: EntityIndex, utterance: str) -> None:
    """Neither entity nor topic. Reusing the previous search is arguably correct here rather
    than a fallback: the topic has not changed, only the requested treatment of it."""
    history = [_turn("what does BCA report for scope 1 emissions")]

    context = resolve_context(utterance, history, index=index)

    assert context.context_source is ContextSource.PREVIOUS_QUERY
    assert context.retrieval_seed == "BCA Scope 1 emissions"
    assert context.inherited_entity is not None
    assert context.inherited_entity.entity is BCA
    # BEFORE: the bare utterance, which retrieves on discourse words alone.
    assert resolve_context(utterance, [], index=index).retrieval_seed == utterance


# --- inheritance rules ------------------------------------------------------


def test_a_named_entity_always_beats_an_inherited_one(index: EntityIndex) -> None:
    """ "What about Mandiri?" must route to Mandiri, never back to BCA."""
    history = [_turn("what does BCA report for scope 1 emissions")]

    context = resolve_context("what about Mandiri?", history, index=index)
    classification = apply_context(
        _classification(domains=(Domain.COMPANY_DOCUMENTS,), confident=True), context
    )

    # Nothing was inherited, so the classification (which found Mandiri itself) is untouched.
    assert context.inherited_entity is None
    assert classification.entities == ()


def test_nothing_is_inherited_when_the_previous_turn_named_two_entities(
    index: EntityIndex,
) -> None:
    """ "Compare BCA and BRI" then "and waste?": which bank is a guess, so it stays domain-wide."""
    history = [_turn("compare BCA and Mandiri scope 1 emissions")]

    context = resolve_context("and waste?", history, index=index)

    assert context.inherited_entity is None
    assert context.context_source is ContextSource.UTTERANCE


def test_inheritance_walks_back_past_turns_that_name_nothing(index: EntityIndex) -> None:
    history = [
        _turn("what does BCA report for scope 1 emissions"),
        _turn("and scope 2?", retrieval_query="BCA scope 2 emissions"),
    ]

    context = resolve_context("and waste?", history, index=index)

    assert context.inherited_entity is not None
    assert context.inherited_entity.entity is BCA


def test_a_fabricated_bank_in_client_history_is_never_inherited(index: EntityIndex) -> None:
    """History is client-owned and therefore untrusted. Inheritance is re-grounded through
    the manifest, so a name the client invented cannot become a routing decision."""
    history = [_turn("what does Bank Fictitious report for scope 1 emissions")]

    context = resolve_context("and waste?", history, index=index)

    assert context.inherited_entity is None
    assert context.context_source is ContextSource.UTTERANCE


def test_apply_context_folds_an_inherited_entity_into_the_classification(
    index: EntityIndex,
) -> None:
    """The unresolved classification is what the rule pass really returns for "and waste?":
    no entity, no domain, not confident, which leaves retrieval sweeping all 21 filings."""
    history = [_turn("what does BCA report for scope 1 emissions")]
    context = resolve_context("and waste?", history, index=index)

    resolved = apply_context(_classification(), context)

    assert resolved.entities == (context.inherited_entity,)
    assert resolved.domains == (Domain.COMPANY_DOCUMENTS,)
    # Confidence is what lets `scoped_source_names` confine the search to one report; an
    # unconfident inheritance would leave the exact 21-bank sweep this module prevents.
    assert resolved.confident is True
    assert resolved.method is ClassifierMethod.RULE_BASED
    assert "inherited" in resolved.reason


def test_apply_context_is_a_no_op_without_an_inherited_entity(index: EntityIndex) -> None:
    original = _classification(domains=(Domain.STANDARDS,), confident=True)
    context = resolve_context("what is GRI 305", [], index=index)

    assert apply_context(original, context) is original


# --- compliance follow-up ---------------------------------------------------


def test_a_company_swap_after_a_compliance_turn_is_flagged(index: EntityIndex) -> None:
    """Compliance answers render in the chat thread, so the conversation mixes pipelines."""
    history = [
        _turn(
            "does BCA comply with GRI 305?",
            answer="| GRI 305-1 | Addressed | ... |",
            retrieval_query=None,
            pipeline=Pipeline.COMPLIANCE,
        )
    ]

    context = resolve_context("and Mandiri?", history, index=index)

    assert context.compliance_follow_up is True


def test_a_topic_shift_after_a_compliance_turn_is_not_flagged(index: EntityIndex) -> None:
    """Deliberately narrower than "any follow-up to a compliance turn": "and waste?" could
    equally want an ordinary cited lookup, and refusing it would block a fair question."""
    history = [
        _turn(
            "does BCA comply with GRI 305?",
            answer="| GRI 305-1 | Addressed | ... |",
            retrieval_query=None,
            pipeline=Pipeline.COMPLIANCE,
        )
    ]

    context = resolve_context("and waste?", history, index=index)

    assert context.compliance_follow_up is False


def test_a_company_swap_after_a_chat_turn_is_not_flagged(index: EntityIndex) -> None:
    history = [_turn("what does BCA report for scope 1 emissions")]

    assert resolve_context("and Mandiri?", history, index=index).compliance_follow_up is False


# --- sanitisation -----------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("BCA reports 6.032 tCO2e [1].", "BCA reports 6.032 tCO2e."),
        ("Supported by two sources [2][3] here.", "Supported by two sources here."),
        ("No markers at all.", "No markers at all."),
        ("Spacing   is   collapsed [1]  .", "Spacing is collapsed."),
    ],
)
def test_strip_citation_markers(raw: str, expected: str) -> None:
    """Source numbering restarts every turn, so a replayed `[2]` points at a different
    document than this turn's [2]. The markers must not survive into the prompt."""
    assert strip_citation_markers(raw) == expected


def test_sanitize_history_strips_markers_and_drops_empty_turns() -> None:
    turns = [
        _turn("q1", answer="answer one [1]."),
        _turn("q2", answer="[1]"),  # nothing but a marker -> empty after stripping
        _turn("  q3  ", answer="answer three [2][3]"),
    ]

    cleaned = sanitize_history(turns)

    assert [t.question for t in cleaned] == ["q1", "q3"]
    assert [t.answer for t in cleaned] == ["answer one.", "answer three"]


def test_sanitize_history_preserves_pipeline_and_retrieval_query() -> None:
    turns = [_turn("q", retrieval_query="seed terms", pipeline=Pipeline.COMPLIANCE)]

    cleaned = sanitize_history(turns)

    assert cleaned[0].pipeline is Pipeline.COMPLIANCE
    assert cleaned[0].retrieval_query == "seed terms"
