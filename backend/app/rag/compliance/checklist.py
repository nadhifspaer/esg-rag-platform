"""Compliance checklist item schema: the shape of one disclosure requirement."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChecklistItem(BaseModel):
    """One disclosure requirement in a compliance checklist: code, description, keywords."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(
        ...,
        min_length=1,
        description=(
            "Stable identifier for the requirement, e.g. a GRI disclosure code like "
            "'GRI 305-1' or an EDGB metric code like 'E-02'. Used verbatim as the row key "
            "in the compliance output table."
        ),
    )
    description: str = Field(
        ...,
        min_length=1,
        description="Short human-readable statement of what the requirement asks to be disclosed.",
    )
    topic_keywords: tuple[str, ...] = Field(
        ...,
        min_length=1,
        description=(
            "Keywords/synonyms describing the requirement's topic. Used to build the "
            "per-requirement retrieval query and to broaden it (add a synonym) in the "
            "retrieval-robustness fallback when the first retrieval scores below threshold."
        ),
    )
    row_anchors: tuple[str, ...] = Field(
        default=(),
        description=(
            "Literal row identifiers that locate this requirement's answering row in the "
            "company reports' disclosure-index tables. Deliberately NOT EDGB-specific: the "
            "corpus uses two idioms, and an anchor may be either an EDGB metric code that "
            "sits on the value row ('E-02', 'E-03') or a literal row label from an uncoded "
            "table ('Total Direct Emissions (Scope 1)'). Several anchors may apply when one "
            "GRI requirement spans more than one indicator, most-specific first. Empty means "
            "the requirement is not anchored — either not yet curated, or, when "
            "`directly_disclosed` is False, deliberately unanchorable."
        ),
    )
    directly_disclosed: bool = Field(
        default=True,
        description=(
            "False when this corpus contains no retrievable row that answers the requirement "
            "— the figure exists only as arithmetic across other rows, so no retrieval or "
            "status assessment can find it. Distinguishes 'deliberately unanchored because "
            "nothing to anchor to' from 'anchors not curated yet', which an empty "
            "`row_anchors` alone cannot express. Set for GRI 305-3: the reports carry no "
            "Scope 3 row, only a Scope 1+2 total and a Scope 1,2,3 total."
        ),
    )

    @field_validator("code", "description")
    @classmethod
    def _require_non_blank(cls, value: str) -> str:
        """Trim surrounding whitespace and reject a blank/whitespace-only value."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank or whitespace-only")
        return stripped

    @field_validator("topic_keywords")
    @classmethod
    def _clean_keywords(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Trim each keyword, drop blanks and duplicates, require at least one to remain."""
        cleaned = tuple(dict.fromkeys(keyword.strip() for keyword in value if keyword.strip()))
        if not cleaned:
            raise ValueError("topic_keywords must contain at least one non-blank keyword")
        return cleaned

    @field_validator("row_anchors")
    @classmethod
    def _clean_anchors(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Trim each anchor and drop blanks/duplicates; may legitimately end up empty."""
        return tuple(dict.fromkeys(anchor.strip() for anchor in value if anchor.strip()))
