"""Tests for the company-reference -> source_name resolver, against the real manifest (no mocks)."""

from __future__ import annotations

import pytest

from app.rag.compliance.resolver import (
    AmbiguousCompanyError,
    CompanyNotFoundError,
    NotACompanyError,
    resolve_category,
    resolve_company_source_name,
)

# The exact string ingestion stored on every BANK-BCA chunk, verified against the live corpus.
BCA_SOURCE_NAME = "PT Bank Central Asia Tbk — 2025 Sustainability Report"


def test_resolves_bca_alias_to_real_source_name_byte_for_byte() -> None:
    # The headline case: the informal "BCA" alone reconstructs the exact stored label.
    result = resolve_company_source_name("BCA")
    assert result == BCA_SOURCE_NAME
    # Byte-for-byte, not just "contains": the em dash and the " 2025 Sustainability
    # Report" suffix must all be exact, or retrieve_evidence's source_name filter misses.
    assert result.encode("utf-8") == BCA_SOURCE_NAME.encode("utf-8")


@pytest.mark.parametrize(
    "reference",
    ["BCA", "bca", "Bank Central Asia", "BBCA", "PT Bank Central Asia Tbk", "  BCA  "],
)
def test_all_bca_surface_forms_resolve_to_the_same_source_name(reference: str) -> None:
    assert resolve_company_source_name(reference) == BCA_SOURCE_NAME


def test_unknown_reference_raises_not_found() -> None:
    with pytest.raises(CompanyNotFoundError):
        resolve_company_source_name("Foobar Bank")


def test_standard_reference_raises_not_a_company() -> None:
    # "GRI 1" resolves to a standard entity, not a bank.
    with pytest.raises(NotACompanyError):
        resolve_company_source_name("GRI 1")


def test_multiple_banks_raise_ambiguous() -> None:
    with pytest.raises(AmbiguousCompanyError):
        resolve_company_source_name("compare BCA and Mandiri")


def test_another_bank_resolves_independently() -> None:
    # A second bank, to prove the reconstruction is per-entry, not hardcoded to BCA.
    assert (
        resolve_company_source_name("Mandiri")
        == "PT Bank Mandiri (Persero) Tbk — 2025 Sustainability Report"
    )


# --- category resolution (run against the real checklist JSON, no mocks) ----------------


def test_resolves_environmental_from_emissions_wording() -> None:
    assert resolve_category("Does BCA report its Scope 1 and Scope 2 emissions?") == "environmental"


def test_resolves_social_from_workforce_wording() -> None:
    assert resolve_category("Does BCA report on employee training hours?") == "social"


def test_resolves_governance_from_anti_corruption_wording() -> None:
    assert resolve_category("Does BCA have a policy on anti-corruption and ethics?") == "governance"


def test_returns_none_for_a_requirement_no_checklist_curates() -> None:
    """A materiality (GRI 3) question, uncovered by any checklist, comes back ambiguous."""
    query = "Bagaimana Bank BCA menentukan topik material ESG sesuai dengan persyaratan GRI 3?"
    assert resolve_category(query) is None


def test_longest_keyword_wins_over_a_generic_substring() -> None:
    """Mirrors rank.resolve_metric's tie-break: a multi-word phrase beats a shorter one."""
    # "board of directors" (governance) and "training" (social) both present; not an arbitrary pick.
    assert resolve_category("board of directors composition and diversity") == "governance"
