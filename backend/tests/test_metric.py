"""Tests for the MetricSpec schema: shape discipline only (no metric content yet)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.rag.metrics.metric import MetricSpec, MetricValueType


def test_valid_spec_holds_the_four_fields() -> None:
    spec = MetricSpec(
        code="GRI 305-1",
        description="Direct (Scope 1) GHG emissions in tCO2e",
        value_type=MetricValueType.NUMERIC,
        topic_keywords=("scope 1", "tCO2e"),
    )
    assert spec.code == "GRI 305-1"
    assert spec.description == "Direct (Scope 1) GHG emissions in tCO2e"
    assert spec.value_type is MetricValueType.NUMERIC
    assert spec.topic_keywords == ("scope 1", "tCO2e")


def test_loads_from_dict_and_coerces_list_and_string_enum() -> None:
    # Simulates loading one entry from a JSON/YAML metric file: value_type as a string,
    # topic_keywords as a list.
    spec = MetricSpec(
        code="GRI 305-5",
        description="Net-zero target year",
        value_type="year",
        topic_keywords=["net zero", "target year"],
    )
    assert spec.value_type is MetricValueType.YEAR
    assert spec.topic_keywords == ("net zero", "target year")


@pytest.mark.parametrize("value_type", ["year", "percentage", "numeric"])
def test_accepts_each_value_type(value_type: str) -> None:
    spec = MetricSpec(code="X-1", description="d", value_type=value_type, topic_keywords=["a"])
    assert spec.value_type.value == value_type


def test_rejects_unknown_value_type() -> None:
    with pytest.raises(ValidationError):
        MetricSpec(code="X-1", description="d", value_type="currency", topic_keywords=["a"])


def test_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        MetricSpec(
            code="X-1",
            description="d",
            value_type="numeric",
            topic_keywords=["a"],
            unit="tCO2e",  # not a declared field -> extra="forbid" rejects it
        )


def test_rejects_blank_code_and_description() -> None:
    with pytest.raises(ValidationError):
        MetricSpec(code="   ", description="d", value_type="numeric", topic_keywords=["a"])
    with pytest.raises(ValidationError):
        MetricSpec(code="X-1", description="  ", value_type="numeric", topic_keywords=["a"])


def test_rejects_empty_or_all_blank_keywords() -> None:
    with pytest.raises(ValidationError):
        MetricSpec(code="X-1", description="d", value_type="numeric", topic_keywords=[])
    with pytest.raises(ValidationError):
        MetricSpec(code="X-1", description="d", value_type="numeric", topic_keywords=["  ", ""])


def test_trims_and_dedupes_keywords() -> None:
    spec = MetricSpec(
        code="X-1",
        description="d",
        value_type="numeric",
        topic_keywords=[" scope 1 ", "scope 1", "tCO2e"],
    )
    assert spec.topic_keywords == ("scope 1", "tCO2e")


def test_is_frozen() -> None:
    spec = MetricSpec(code="X-1", description="d", value_type="numeric", topic_keywords=["a"])
    with pytest.raises(ValidationError):
        spec.code = "Y-2"  # type: ignore[misc]  # frozen model rejects mutation
