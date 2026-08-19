"""Tests for the chat-path row-anchor trigger map (Phase 1)."""

from __future__ import annotations

from app.rag.chat_anchor_triggers import resolve_chat_anchor_triggers

_SCOPE_1_ANCHORS = (
    "Total Direct Emissions (Scope 1)",
    "Category 1: Direct GHG emissions and removals",
)
_SCOPE_2_ANCHORS = (
    "Total Indirect Emissions (Scope 2)",
    "Category 2: Indirect GHG emissions from imported energy",
)


def _indicators(query: str) -> set[str]:
    return {t.indicator for t in resolve_chat_anchor_triggers(query)}


# --- the five-bank Scope 1 test set (this session), all sourced from real scripts ------


def test_bca_terse_smoke_test_query_triggers_scope_1() -> None:
    triggers = resolve_chat_anchor_triggers("BCA Scope 1 emissions")
    assert [t.indicator for t in triggers] == ["scope_1"]
    assert triggers[0].anchors == _SCOPE_1_ANCHORS


def test_mandiri_original_and_retry_rewrite_both_trigger_scope_1() -> None:
    assert _indicators("what does Mandiri report for scope 1 emissions") == {"scope_1"}
    assert _indicators(
        "Bank Mandiri Scope 1 emissions 2025 Sustainability Report GHG emissions"
    ) == {"scope_1"}


def test_bni_original_and_retry_rewrite_both_trigger_scope_1() -> None:
    assert _indicators("what does BNI report for scope 1 emissions") == {"scope_1"}
    assert _indicators("BNI Scope 1 emissions data 2025 Sustainability Report") == {"scope_1"}


def test_cimb_niaga_original_and_retry_rewrite_both_trigger_scope_1() -> None:
    assert _indicators("what does CIMB Niaga report for scope 1 emissions") == {"scope_1"}
    assert _indicators("CIMB Niaga Scope 1 emissions data reported actual figures or amounts") == {
        "scope_1"
    }


def test_ocbc_nisp_query_triggers_scope_1() -> None:
    # Matches identically to the terse form via \bscope[\s-]?1\b.
    assert _indicators("what does OCBC NISP report for scope 1 emissions") == {"scope_1"}


# --- the waste query that broke topic_keywords ------------------------------------------


def test_bca_waste_query_triggers_e05_despite_word_order() -> None:
    """ "how much waste did BCA generate": not a substring of "waste generated", the
    exact case docs/architecture.md's 2026-07-25 entry measured topic_keywords missing."""
    triggers = resolve_chat_anchor_triggers("how much waste did BCA generate")
    assert [t.indicator for t in triggers] == ["E-05"]
    assert triggers[0].anchors == ("E-05",)


# --- supplementary real queries (docs/architecture.md, 2026-07-25 entry) ----------------


def test_bca_emissions_intensity_triggers_e02() -> None:
    triggers = resolve_chat_anchor_triggers("BCA emissions intensity")
    assert [t.indicator for t in triggers] == ["E-02"]
    assert triggers[0].anchors == ("E-02",)


def test_bca_net_zero_target_year_triggers_e06() -> None:
    triggers = resolve_chat_anchor_triggers("BCA net zero target year")
    assert [t.indicator for t in triggers] == ["E-06"]
    assert triggers[0].anchors == ("E-06",)


# --- negative controls -------------------------------------------------------------------


def test_unrelated_query_triggers_nothing() -> None:
    assert resolve_chat_anchor_triggers("what is BCA's total assets") == []


def test_scope_3_is_recognized_but_unanchored() -> None:
    """GRI 305-3 is curated with row_anchors=() (directly_disclosed=False), recognized,
    never pinned, distinct from no match at all."""
    triggers = resolve_chat_anchor_triggers("what does BCA report for scope 3 emissions")
    assert [t.indicator for t in triggers] == ["scope_3"]
    assert triggers[0].anchors == ()


# --- indicator coverage sanity (not part of the reviewed test set, but each rule should
# fire on at least one plausible phrasing so a broken regex doesn't go unnoticed) --------


def test_electricity_consumption_triggers_e03() -> None:
    assert _indicators("how much electricity did BCA consume") == {"E-03"}


def test_water_consumption_triggers_e04() -> None:
    assert _indicators("BCA water consumption") == {"E-04"}


def test_reduction_commitment_triggers_e07() -> None:
    assert _indicators("does BCA commit to reducing emissions") == {"E-07"}


def test_a_query_can_match_more_than_one_indicator() -> None:
    assert _indicators("BCA scope 1 and scope 2 emissions") == {"scope_1", "scope_2"}
