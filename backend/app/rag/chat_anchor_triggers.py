r"""Chat-path row-anchor trigger map: detects a chat query asking about a curated indicator."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from app.rag.compliance.checklist import ChecklistItem

_CHECKLISTS_DIR = Path(__file__).resolve().parent / "compliance" / "checklists"


@dataclass(frozen=True)
class ChatAnchorTrigger:
    """One recognized environmental indicator and the row anchors that answer it (may be empty)."""

    indicator: str
    anchors: tuple[str, ...]


def _row_anchors(framework: str, category: str, code: str) -> tuple[str, ...]:
    """Read one checklist item's `row_anchors` directly off its JSON file, by code."""
    path = _CHECKLISTS_DIR / framework / f"{category}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for item in data["items"]:
        if item["code"] == code:
            return ChecklistItem(**item).row_anchors
    raise KeyError(f"{code!r} not found in {path}")


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


# One rule per curated indicator; all `groups` must match (AND), OR within each group.
@dataclass(frozen=True)
class _Rule:
    indicator: str
    groups: tuple[re.Pattern[str], ...]
    anchors: tuple[str, ...]


_RULES: tuple[_Rule, ...] = (
    _Rule(
        "scope_1", (_rx(r"\bscope[\s-]?1\b"),), _row_anchors("gri", "environmental", "GRI 305-1")
    ),
    _Rule(
        "scope_2", (_rx(r"\bscope[\s-]?2\b"),), _row_anchors("gri", "environmental", "GRI 305-2")
    ),
    # Recognized on purpose, anchors=() by curation: GRI 305-3 has no retrievable row.
    _Rule(
        "scope_3", (_rx(r"\bscope[\s-]?3\b"),), _row_anchors("gri", "environmental", "GRI 305-3")
    ),
    _Rule(
        "E-02",
        (_rx(r"\bintensity\b"), _rx(r"\b(emissions?|ghg|carbon|co2e?)\b")),
        _row_anchors("edgb", "environmental", "E-02"),
    ),
    _Rule(
        "E-03",
        (_rx(r"\b(electricity|energy|power)\b"), _rx(r"\bconsum\w*\b|\busage\b|\bused?\b")),
        _row_anchors("edgb", "environmental", "E-03"),
    ),
    _Rule(
        "E-04",
        (_rx(r"\bwater\b"), _rx(r"\bconsum\w*\b|\busage\b|\bused?\b")),
        _row_anchors("edgb", "environmental", "E-04"),
    ),
    _Rule(
        "E-05",
        (_rx(r"\bwaste\b"), _rx(r"\bgenerat\w*\b|\btotal\b|\bproduce[ds]?\b")),
        _row_anchors("edgb", "environmental", "E-05"),
    ),
    _Rule("E-06", (_rx(r"\bnet[\s-]?zero\b"),), _row_anchors("edgb", "environmental", "E-06")),
    _Rule(
        "E-07",
        (_rx(r"\breduc\w*\b"), _rx(r"\b(emissions?|ghg|carbon|co2e?|target)\b")),
        _row_anchors("edgb", "environmental", "E-07"),
    ),
)


def resolve_chat_anchor_triggers(query: str) -> list[ChatAnchorTrigger]:
    """Which curated environmental indicators (if any) `query` is asking about."""
    return [
        ChatAnchorTrigger(indicator=rule.indicator, anchors=rule.anchors)
        for rule in _RULES
        if all(group.search(query) for group in rule.groups)
    ]
