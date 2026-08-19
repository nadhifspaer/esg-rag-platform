"""Extractable-metric schema: the shape of one metric to pull from a document."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MetricValueType(StrEnum):
    """The expected form of an extracted metric value: a closed set an extractor can parse."""

    YEAR = "year"  # a calendar year, e.g. a net-zero target year (2050)
    PERCENTAGE = "percentage"  # a value expressed as a percent, e.g. 34.5
    NUMERIC = "numeric"  # a bare quantity with a unit reported separately, e.g. tCO2e


class MetricSpec(BaseModel):
    """One extractable metric: code, description, expected value type, and keywords."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(
        ...,
        min_length=1,
        description=(
            "Stable identifier for the metric, e.g. a GRI disclosure code like 'GRI 305-1' "
            "or an EDGB metric code like 'E-01'. Used verbatim as the key for the extracted "
            "value."
        ),
    )
    description: str = Field(
        ...,
        min_length=1,
        description="Short human-readable statement of what the metric is.",
    )
    value_type: MetricValueType = Field(
        ...,
        description=(
            "The expected form of the value (year / percentage / numeric), so extraction "
            "knows what shape to parse and validate."
        ),
    )
    topic_keywords: tuple[str, ...] = Field(
        ...,
        min_length=1,
        description=(
            "Keywords/synonyms describing the metric's topic. Used to build the retrieval "
            "query and to broaden it (add a synonym) in the retrieval-robustness fallback."
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


# The curated metric set lives beside this module, same pattern as the compliance
# checklists living beside `checklist.py`.
_METRIC_SET_PATH = Path(__file__).resolve().parent / "company_metrics.json"


def load_metric_set(path: Path | None = None) -> list[MetricSpec]:
    """Load the curated metric set into `MetricSpec`s, dropping the provenance-only `source`."""
    data = json.loads((path or _METRIC_SET_PATH).read_text(encoding="utf-8"))
    return [
        MetricSpec(**{key: value for key, value in entry.items() if key != "source"})
        for entry in data["metrics"]
    ]
