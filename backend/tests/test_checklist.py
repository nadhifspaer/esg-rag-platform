"""Tests for the ChecklistItem schema (the compliance requirement data shape)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

import app.api.compliance as compliance_api
from app.api.compliance import (
    ChecklistCategory,
    ChecklistNotFoundError,
    FrameworkName,
    load_checklist,
)
from app.rag.compliance.checklist import ChecklistItem


def test_valid_item_holds_the_three_fields() -> None:
    item = ChecklistItem(
        code="GRI 305-1",
        description="Direct (Scope 1) GHG emissions",
        topic_keywords=["scope 1", "direct emissions", "GHG"],
    )
    assert item.code == "GRI 305-1"
    assert item.description == "Direct (Scope 1) GHG emissions"
    assert item.topic_keywords == ("scope 1", "direct emissions", "GHG")


def test_loads_from_dict_and_coerces_list_to_tuple() -> None:
    # Simulates loading one entry from a JSON/YAML checklist file.
    item = ChecklistItem.model_validate(
        {"code": "X-1", "description": "desc", "topic_keywords": ["a", "b"]}
    )
    assert item.topic_keywords == ("a", "b")


def test_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ChecklistItem(code="X-1", description="d", topic_keywords=["a"], category="environmental")


def test_rejects_blank_code_and_description() -> None:
    with pytest.raises(ValidationError):
        ChecklistItem(code="   ", description="d", topic_keywords=["a"])
    with pytest.raises(ValidationError):
        ChecklistItem(code="X-1", description="", topic_keywords=["a"])


def test_rejects_empty_or_all_blank_keywords() -> None:
    with pytest.raises(ValidationError):
        ChecklistItem(code="X-1", description="d", topic_keywords=[])
    with pytest.raises(ValidationError):
        ChecklistItem(code="X-1", description="d", topic_keywords=["  ", ""])


def test_trims_and_dedupes() -> None:
    item = ChecklistItem(
        code="  X-1  ",
        description="  desc  ",
        topic_keywords=[" a ", "a", "b", ""],
    )
    assert item.code == "X-1"
    assert item.description == "desc"
    assert item.topic_keywords == ("a", "b")  # trimmed, blanks dropped, duplicate removed


def test_is_frozen() -> None:
    item = ChecklistItem(code="X-1", description="d", topic_keywords=["a"])
    with pytest.raises(ValidationError):
        item.code = "Y-1"  # type: ignore[misc]


# --- row anchors: locating the requirement's row in an EDGB-indexed report --------------


def test_row_anchors_default_to_empty_and_directly_disclosed_defaults_true() -> None:
    """Both anchor fields are optional, so existing unanchored checklists keep loading."""
    item = ChecklistItem(code="X-1", description="d", topic_keywords=["a"])
    assert item.row_anchors == ()
    assert item.directly_disclosed is True


def test_row_anchors_trim_dedupe_and_keep_order() -> None:
    """Order is meaningful, most specific anchor first, so it must survive cleaning."""
    item = ChecklistItem(
        code="GRI 305-5",
        description="d",
        topic_keywords=["a"],
        row_anchors=[" E-07 ", "E-06", "E-07", "  "],
    )
    assert item.row_anchors == ("E-07", "E-06")


def test_row_anchors_may_be_empty_unlike_topic_keywords() -> None:
    """An unanchored requirement is a valid state; topic_keywords still cannot be empty."""
    item = ChecklistItem(
        code="X-1", description="d", topic_keywords=["a"], row_anchors=[], directly_disclosed=False
    )
    assert item.row_anchors == ()
    assert item.directly_disclosed is False


# --- the real curated environmental checklist ------------------------------------------


def test_real_environmental_checklist_carries_its_curated_anchors() -> None:
    """Guards the 2026-07-25 curation against the real corpus's two anchoring idioms."""
    items = {item.code: item for item in load_checklist(ChecklistCategory.ENVIRONMENTAL)}

    # EDGB coded rows, where the code sits on the value row.
    assert items["GRI 305-4"].row_anchors == ("E-02",)
    assert items["GRI 302-1"].row_anchors == ("E-03",)
    assert items["GRI 306-3"].row_anchors == ("E-05",)
    # An uncoded labelled row: the emissions table carries no EDGB code of its own.
    assert items["GRI 305-1"].row_anchors[0] == "Total Direct Emissions (Scope 1)"
    assert items["GRI 305-2"].row_anchors[0] == "Total Indirect Emissions (Scope 2)"
    # Two anchors, because GRI 305-5's definition spans two EDGB indicators.
    assert items["GRI 305-5"].row_anchors == ("E-07", "E-06")
    # E-01 is never an anchor: its block holds report metadata, not the scope figures.
    assert all("E-01" not in item.row_anchors for item in items.values())


def test_gri_305_3_is_marked_not_directly_disclosed() -> None:
    """Scope 3 has no row here; the flag distinguishes "unanchorable" from "not curated"."""
    items = {item.code: item for item in load_checklist(ChecklistCategory.ENVIRONMENTAL)}
    scope_3 = items["GRI 305-3"]

    assert scope_3.row_anchors == ()
    assert scope_3.directly_disclosed is False
    # Every other environmental requirement is anchored and directly disclosed.
    others = [i for code, i in items.items() if code != "GRI 305-3"]
    assert all(i.directly_disclosed and i.row_anchors for i in others)


def test_unanchored_categories_still_load() -> None:
    """Social and governance are unanchored except a handful of curated items."""
    social = load_checklist(ChecklistCategory.SOCIAL)
    governance = load_checklist(ChecklistCategory.GOVERNANCE)
    assert social
    assert governance

    anchored_social = {item.code: item for item in social if item.row_anchors}
    assert set(anchored_social) == {"GRI 412-1", "GRI 406-1"}
    assert anchored_social["GRI 412-1"].row_anchors == ("S-09",)
    assert anchored_social["GRI 406-1"].row_anchors == ("S-08",)
    assert all(
        item.row_anchors == () and item.directly_disclosed
        for item in social
        if item.code not in {"GRI 412-1", "GRI 406-1"}
    )

    anchored = {item.code: item for item in governance if item.row_anchors}
    assert set(anchored) == {"GRI 205-1", "GRI 2-10", "GRI 2-11", "GRI 2-15"}
    assert anchored["GRI 205-1"].row_anchors == ("G-07",)
    assert anchored["GRI 2-10"].row_anchors == ("G-06",)
    assert anchored["GRI 2-11"].row_anchors == ("G-03",)
    assert anchored["GRI 2-15"].row_anchors == ("G-09", "G-08")
    assert all(
        item.row_anchors == () and item.directly_disclosed
        for item in governance
        if item.code not in {"GRI 205-1", "GRI 2-10", "GRI 2-11", "GRI 2-15"}
    )


# --- framework: checklists/{framework}/{category}.json ---------------------------------


def test_load_checklist_defaults_to_gri() -> None:
    """No `framework` arg -> the same GRI checklist every pre-framework call site expects."""
    default = load_checklist(ChecklistCategory.ENVIRONMENTAL)
    explicit = load_checklist(ChecklistCategory.ENVIRONMENTAL, FrameworkName.GRI)
    assert default == explicit
    assert default[0].code.startswith("GRI ")


def test_load_checklist_loads_the_real_edgb_environmental_checklist() -> None:
    """The 7 EDGB codes (E-01..E-07), each anchored on itself, confirmed live against
    BANK-BCA's own index table, not invented."""
    items = {
        item.code: item
        for item in load_checklist(ChecklistCategory.ENVIRONMENTAL, FrameworkName.EDGB)
    }
    assert list(items) == ["E-01", "E-02", "E-03", "E-04", "E-05", "E-06", "E-07"]
    for code, item in items.items():
        assert item.row_anchors == (code,)
        assert item.directly_disclosed is True


def test_load_checklist_raises_for_an_uncurated_combination(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A missing (framework, category) file must raise, never substitute another checklist."""
    monkeypatch.setattr(compliance_api, "_CHECKLISTS_DIR", tmp_path)
    with pytest.raises(ChecklistNotFoundError):
        load_checklist(ChecklistCategory.ENVIRONMENTAL, FrameworkName.GRI)


def test_load_checklist_loads_the_real_edgb_social_checklist() -> None:
    """The 12 EDGB social codes (S-01..S-12), each anchored on itself."""
    items = {
        item.code: item for item in load_checklist(ChecklistCategory.SOCIAL, FrameworkName.EDGB)
    }
    assert list(items) == [f"S-{i:02d}" for i in range(1, 13)]
    for code, item in items.items():
        assert item.row_anchors == (code,)
        assert item.directly_disclosed is True
    # S-12's real value row is CSR, not conflict-of-interest; pin the corrected reading.
    assert items["S-12"].description.lower().startswith("corporate social responsibility")


def test_load_checklist_loads_the_real_edgb_governance_checklist() -> None:
    """The 9 EDGB governance codes (G-01..G-09), each anchored on itself."""
    items = {
        item.code: item for item in load_checklist(ChecklistCategory.GOVERNANCE, FrameworkName.EDGB)
    }
    assert list(items) == [f"G-{i:02d}" for i in range(1, 10)]
    for code, item in items.items():
        assert item.row_anchors == (code,)
        assert item.directly_disclosed is True


def test_every_checklist_file_on_disk_loads_cleanly() -> None:
    """Blanket safety net: every checklists/{framework}/{category}.json parses into valid items."""
    checked = 0
    for framework in FrameworkName:
        for category in ChecklistCategory:
            path = compliance_api._CHECKLISTS_DIR / framework.value / f"{category.value}.json"
            if not path.exists():
                continue
            items = load_checklist(category, framework)
            assert items, f"{framework.value}/{category.value}.json loaded with no items"
            checked += 1
    assert checked == 6  # gri + edgb, all three E/S/G categories each
