"""Tests for the entity index: the rule-based entity matcher behind classification."""

from __future__ import annotations

from app.rag.entity_index import (
    EntityIndex,
    _build_entities,
    default_entity_index,
)

# A small manifest-shaped fixture covering both domains, including the "BANK"
# denylist case.
_MANIFEST_DOMAINS = {
    "company_documents": [
        {"file": "BANK-BCA.pdf", "bank": "PT Bank Central Asia Tbk", "aliases": ["BCA", "BBCA"]},
        {
            "file": "BANK-AGI.pdf",
            "bank": "PT Bank Artha Graha Internasional Tbk",
            "aliases": ["AGI", "Artha Graha", "Bank Artha Graha", "INPC"],
        },
        {
            "file": "BANK-ALADIN.pdf",
            "bank": "PT Bank Aladin Syariah Tbk",
            "aliases": ["Bank Aladin", "ALADIN", "BANK"],
        },
        {"file": "BANK-MEGA.pdf", "bank": "PT Bank Mega Tbk", "aliases": ["Bank Mega", "MEGA"]},
    ],
    "standards": [
        {
            "file": "GRI-3.pdf",
            "official_title": "GRI 3: Material Topics",
            "aliases": ["GRI 3", "GRI3"],
        },
    ],
}


def _index() -> EntityIndex:
    return EntityIndex(_build_entities(_MANIFEST_DOMAINS))


# --- building ---------------------------------------------------------------


def test_ambiguous_alias_bank_is_excluded() -> None:
    aladin = next(
        e for e in _build_entities(_MANIFEST_DOMAINS) if "aladin" in e.canonical_name.lower()
    )
    assert "bank" not in aladin.surface_forms  # the "BANK" ticker is denylisted
    assert "aladin" in aladin.surface_forms  # its real aliases survive


# --- matching ---------------------------------------------------------------


def test_aliases_resolve_to_the_same_entity() -> None:
    index = _index()
    for surface in ("AGI", "Artha Graha", "Bank Artha Graha Internasional"):
        matches = index.match(f"tell me about {surface}'s emissions")
        assert len(matches) == 1
        assert matches[0].entity.canonical_name == "PT Bank Artha Graha Internasional Tbk"


def test_whole_token_boundary_prevents_substring_false_match() -> None:
    index = _index()
    # "MEGA" must not fire inside "megawatt"; "GRI 3" must not fire inside "GRI 305".
    assert index.match("how many megawatt hours did the bank use") == []
    assert index.match("does it cover GRI 305 emissions") == []


def test_bank_alone_does_not_match_any_entity() -> None:
    index = _index()
    # "bank" is denylisted, so a generic sentence naming no specific bank matches nothing.
    assert index.match("what does the bank report on climate risk") == []


def test_real_repo_manifest_loads_and_resolves_task_examples() -> None:
    index = default_entity_index()
    # The aliases called out in the task description resolve to the AGI entity.
    for surface in ("AGI", "Artha Graha", "Bank Artha Graha Internasional"):
        matches = index.match(surface)
        assert any(
            m.entity.canonical_name == "PT Bank Artha Graha Internasional Tbk" for m in matches
        ), surface
