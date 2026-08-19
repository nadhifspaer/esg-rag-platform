from __future__ import annotations

import json
from enum import Enum
from unittest import mock

import pytest

from app.models.citation import Citation
from app.models.payload import ContentType
from app.observability import tracing


class _Colour(str, Enum):  # noqa: UP042 - `(str, Enum)` is the subject, not incidental
    """Mirrors how every enum in this codebase is declared: `(str, Enum)`, not `StrEnum`."""

    RED = "red"


def _round_trip(value: object) -> object:
    """What the Langfuse backend gets back after the value crosses the wire."""
    serialized = tracing._propagated_metadata({"k": value})["k"]
    return json.loads(serialized)


@pytest.mark.parametrize(
    "value",
    [
        ["company_documents"],
        ["company_documents", "regulations"],
        [],
        True,
        False,
        None,
        7,
        0,
        1.5,
        {"disclosed": 6, "not directly disclosed": 1},
    ],
)
def test_metadata_values_survive_the_round_trip(value: object) -> None:
    """Every non-string value comes back as itself, not as its Python repr."""
    assert _round_trip(value) == value


def test_strings_are_passed_through_untouched() -> None:
    """`str()` is already the identity for strings; quoting them would add literal quotes."""
    prepared = tracing._propagated_metadata({"pipeline": "chat_agentic_loop"})
    assert prepared["pipeline"] == "chat_agentic_loop"


def test_the_real_chat_metadata_shape_is_not_stringified() -> None:
    """The exact fields that came back wrong from the live JP-region trace."""
    prepared = tracing._propagated_metadata(
        {
            "endpoint": "/chat",
            "domains": ["company_documents"],
            "scoped_to": None,
            "high_accuracy": False,
            "classifier_confident": True,
        }
    )
    # The regression: Python repr reached the backend as an unparseable string.
    assert prepared["domains"] != "['company_documents']"
    assert prepared["high_accuracy"] != "False"
    assert prepared["scoped_to"] != "None"

    assert json.loads(prepared["domains"]) == ["company_documents"]
    assert json.loads(prepared["high_accuracy"]) is False
    assert json.loads(prepared["classifier_confident"]) is True
    assert json.loads(prepared["scoped_to"]) is None
    assert prepared["endpoint"] == "/chat"


def test_unserializable_values_degrade_instead_of_raising() -> None:
    """An exotic value must not raise out of a tracing call: `default=str` catches it."""
    prepared = tracing._propagated_metadata({"obj": object()})
    assert isinstance(prepared["obj"], str)


def test_str_enums_carry_their_value_not_their_repr() -> None:
    """`(str, Enum)` members take the passthrough branch, keeping their string value, not repr."""
    prepared = tracing._propagated_metadata({"colour": _Colour.RED})
    assert prepared["colour"] == "red"


def test_every_value_is_a_string_langfuse_can_accept() -> None:
    """Langfuse validates propagated metadata as strings; nothing may leak through as-is."""
    prepared = tracing._propagated_metadata({"a": ["x"], "b": 1, "c": None, "d": "s"})
    assert all(isinstance(v, str) for v in prepared.values())


# --- Tracing disabled or broken must never affect the request ---------------------


def test_helpers_are_no_ops_without_a_span() -> None:
    tracing.update_span(None, output={"answer": "x"})
    tracing.record_retrieval(None, [])
    assert tracing.budget_fields(None) == {}


def test_traced_request_yields_none_when_tracing_is_off() -> None:
    with mock.patch.object(tracing, "get_langfuse", lambda: None):
        with tracing.traced_request(name="chat", user_id="u", query="q") as span:
            assert span is None


def test_traced_request_does_not_swallow_endpoint_exceptions() -> None:
    """An HTTPException raised inside the body is control flow, not something to eat."""
    with mock.patch.object(tracing, "get_langfuse", lambda: None):
        with pytest.raises(ValueError, match="boom"):  # noqa: PT012 - the raise is the subject
            with tracing.traced_request(name="rank", user_id="u", query="q"):
                raise ValueError("boom")


def test_conversation_id_is_propagated_as_the_langfuse_session() -> None:
    """Session grouping is the only thing that can separate conversations sharing one demo login."""
    captured: dict[str, object] = {}

    class _Ctx:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_exc: object) -> None:
            return None

    def _fake_propagate(**kwargs: object) -> _Ctx:
        captured.update(kwargs)
        return _Ctx()

    class _Span:
        def __enter__(self) -> _Span:
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

    class _Client:
        def start_as_current_observation(self, **_kwargs: object) -> _Span:
            return _Span()

    with mock.patch.object(tracing, "get_langfuse", lambda: _Client()):
        with mock.patch("langfuse.propagate_attributes", _fake_propagate):
            with tracing.traced_request(
                name="chat", user_id="shared-demo-user", query="and waste?", session_id="conv-1"
            ):
                pass

    assert captured["session_id"] == "conv-1"
    assert captured["user_id"] == "shared-demo-user"


def test_session_id_defaults_to_none_for_a_single_turn_request() -> None:
    with mock.patch.object(tracing, "get_langfuse", lambda: None):
        with tracing.traced_request(name="chat", user_id="u", query="q") as span:
            assert span is None


# --- Retrieval hits: both chunk shapes must record ---------------------------------


class _FakeSpan:
    """Captures what `record_retrieval` writes, standing in for a Langfuse span."""

    def __init__(self) -> None:
        self.recorded: dict | None = None

    def start_as_current_observation(self, **_kwargs: object) -> _FakeSpan:
        return self

    def __enter__(self) -> _FakeSpan:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def update(self, **fields: object) -> None:
        self.recorded = fields.get("output")  # type: ignore[assignment]


def test_record_retrieval_reads_citations() -> None:
    """`ChatResult` exposes `Citation`s directly; reading `.payload` regressed to `hit_count: 0`."""
    span = _FakeSpan()
    tracing.record_retrieval(
        span,
        [
            Citation(
                source_name="BCA 2025",
                page_number=13,
                content_type=ContentType.TABLE,
                excerpt="Total Direct Emissions (Scope 1): 6.032 tCO2e",
            )
        ],
    )
    assert span.recorded == {
        "hit_count": 1,
        "hits": [{"source_name": "BCA 2025", "page_number": 13, "content_type": "table"}],
    }
