"""Tests for confining chat retrieval to one document per domain (`scoped_source_names`)."""

from __future__ import annotations

import pytest

from app.models.conversation import ConversationTurn
from app.models.payload import Domain
from app.rag.chat_engine import scoped_source_names
from app.rag.compliance.resolver import source_name_for_entity
from app.rag.conversation import apply_context, resolve_context
from app.rag.entity_index import Entity, EntityKind, EntityMatch, default_entity_index
from app.rag.query_classifier import ClassifierMethod, QueryClassification, QueryType

BCA_SOURCE = "PT Bank Central Asia Tbk — 2025 Sustainability Report"


def _bank(canonical: str = "PT Bank Central Asia Tbk") -> Entity:
    return Entity(
        kind=EntityKind.BANK,
        domain=Domain.COMPANY_DOCUMENTS,
        canonical_name=canonical,
        surface_forms=("bca", canonical.lower()),
        bank=canonical,
    )


def _standard(canonical: str = "GRI 305: Emissions 2016") -> Entity:
    return Entity(
        kind=EntityKind.STANDARD,
        domain=Domain.STANDARDS,
        canonical_name=canonical,
        surface_forms=("gri 305",),
        official_title=canonical,
    )


def _classification(
    *entities: Entity,
    domains: tuple[Domain, ...] = (Domain.COMPANY_DOCUMENTS,),
    confident: bool = True,
    query_type: QueryType = QueryType.SINGLE_DOMAIN,
) -> QueryClassification:
    return QueryClassification(
        query_type=query_type,
        domains=domains,
        entities=tuple(EntityMatch(entity=e, matched_text="x") for e in entities),
        method=ClassifierMethod.RULE_BASED,
        confident=confident,
        reason="test",
    )


# --- entity -> source_name --------------------------------------------------


def test_source_name_for_entity_matches_the_stored_label() -> None:
    """Reconstructed by the same function that built the label at ingestion time."""
    assert source_name_for_entity(_bank()) == BCA_SOURCE


def test_source_name_for_entity_returns_none_for_a_non_bank() -> None:
    assert source_name_for_entity(_standard()) is None


def test_source_name_for_entity_returns_none_for_a_bank_absent_from_the_manifest() -> None:
    assert source_name_for_entity(_bank("PT Bank Not In Manifest Tbk")) is None


# --- the scoping decision ---------------------------------------------------


def test_one_confident_bank_in_company_documents_scopes_to_that_report() -> None:
    assert scoped_source_names(_classification(_bank()), (Domain.COMPANY_DOCUMENTS,)) == {
        Domain.COMPANY_DOCUMENTS: BCA_SOURCE
    }


def test_an_inherited_entity_scopes_the_search_on_a_follow_up() -> None:
    """The point of multi-turn carry-forward: `apply_context` folding in the prior entity."""
    history = [
        ConversationTurn(
            question="what does BCA report for scope 1 emissions",
            answer="6.032 tCO2e.",
            retrieval_query="BCA Scope 1 emissions",
        )
    ]
    context = resolve_context("and waste?", history, index=default_entity_index())

    unresolved = QueryClassification(
        query_type=QueryType.SINGLE_DOMAIN,
        domains=(),
        entities=(),
        method=ClassifierMethod.RULE_BASED,
        confident=False,
        reason="no known entity named; domain undetermined by rules",
    )
    # BEFORE: nothing to scope to (no domains requested at all).
    assert scoped_source_names(unresolved, (Domain.COMPANY_DOCUMENTS,)) == {
        Domain.COMPANY_DOCUMENTS: None
    }

    # AFTER: the inherited entity is enough, through the unchanged guards.
    resolved = apply_context(unresolved, context)
    assert scoped_source_names(resolved, resolved.domains) == {Domain.COMPANY_DOCUMENTS: BCA_SOURCE}


def test_two_banks_are_not_scoped() -> None:
    """ "Compare BCA and BRI": picking one would answer a different question."""
    classification = _classification(_bank(), _bank("PT Bank Rakyat Indonesia (Persero) Tbk"))
    assert scoped_source_names(classification, (Domain.COMPANY_DOCUMENTS,)) == {
        Domain.COMPANY_DOCUMENTS: None
    }


def test_no_entity_is_not_scoped() -> None:
    assert scoped_source_names(_classification(), (Domain.COMPANY_DOCUMENTS,)) == {
        Domain.COMPANY_DOCUMENTS: None
    }


def test_cross_domain_scopes_company_documents_to_the_named_bank() -> None:
    """Case 2 fix: a bare code resolves no standard entity; only company_documents scopes."""
    classification = _classification(
        _bank(),
        domains=(Domain.COMPANY_DOCUMENTS, Domain.STANDARDS),
        query_type=QueryType.CROSS_DOMAIN,
    )
    assert scoped_source_names(classification, (Domain.COMPANY_DOCUMENTS, Domain.STANDARDS)) == {
        Domain.COMPANY_DOCUMENTS: BCA_SOURCE,
        Domain.STANDARDS: None,
    }


def test_cross_domain_scopes_both_legs_when_a_bank_and_a_standard_are_both_named() -> None:
    """ "Does BCA align with GRI 3?" names a bank and a standard; both legs scope independently."""
    classification = _classification(
        _bank(),
        _standard(),
        domains=(Domain.COMPANY_DOCUMENTS, Domain.STANDARDS),
        query_type=QueryType.CROSS_DOMAIN,
    )
    assert scoped_source_names(classification, (Domain.COMPANY_DOCUMENTS, Domain.STANDARDS)) == {
        Domain.COMPANY_DOCUMENTS: BCA_SOURCE,
        Domain.STANDARDS: "GRI 305: Emissions 2016",
    }


def test_two_standard_entities_are_not_scoped() -> None:
    """Mirrors the two-bank guard: naming two standards makes "which one" a guess."""
    classification = _classification(
        _standard("GRI 305: Emissions 2016"),
        _standard("GRI 306: Waste 2020"),
        domains=(Domain.STANDARDS,),
    )
    assert scoped_source_names(classification, (Domain.STANDARDS,)) == {Domain.STANDARDS: None}


def test_one_confident_standard_in_standards_domain_scopes_to_that_document() -> None:
    """The standards-side counterpart to the company_documents case: a single-domain
    standards query naming one standard now scopes too, symmetric with the bank side."""
    classification = _classification(_standard(), domains=(Domain.STANDARDS,))
    assert scoped_source_names(classification, (Domain.STANDARDS,)) == {
        Domain.STANDARDS: "GRI 305: Emissions 2016"
    }


def test_low_confidence_is_not_scoped() -> None:
    """An LLM-fallback guess is too weak a basis for excluding candidates from any domain,
    checked once, up front, rather than per domain."""
    classification = _classification(
        _bank(),
        _standard(),
        domains=(Domain.COMPANY_DOCUMENTS, Domain.STANDARDS),
        confident=False,
    )
    assert scoped_source_names(classification, (Domain.COMPANY_DOCUMENTS, Domain.STANDARDS)) == {
        Domain.COMPANY_DOCUMENTS: None,
        Domain.STANDARDS: None,
    }


@pytest.mark.parametrize(
    "domains",
    [(Domain.STANDARDS,)],
)
def test_a_bank_entity_outside_company_documents_is_not_scoped(domains) -> None:
    """Effective, not classifier, domains decide scoping: `standards`-only scopes no bank."""
    result = scoped_source_names(_classification(_bank()), domains)
    assert result == {Domain.STANDARDS: None}
    assert Domain.COMPANY_DOCUMENTS not in result
