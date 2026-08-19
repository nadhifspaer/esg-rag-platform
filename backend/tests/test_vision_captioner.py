"""Tests for the vision captioner."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app.core.config import Settings
from app.rag.ingestion.vision_captioner import (
    PageCaption,
    VisionCaptionError,
    caption_page_image,
)

# --- fakes & helpers -------------------------------------------------------


class _FakeCompletions:
    """Records each create() call and returns one canned JSON body."""

    def __init__(self, body: str) -> None:
        self._body = body
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self._body)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    """Stands in for openai.OpenAI: mirrors `.chat.completions.create(...)`."""

    def __init__(self, body: str) -> None:
        self.completions = _FakeCompletions(body)
        self.chat = SimpleNamespace(completions=self.completions)

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.completions.calls


def _settings(api_key: str = "test-key") -> Settings:
    """Explicit settings so tests never read the real .env or build a real client."""
    return Settings(
        openai_api_key=api_key,
        openai_vision_model="gpt-4.1-mini",
        openai_vision_model_high="gpt-4.1",
    )


def _body(has_chart: bool, caption: str) -> str:
    return json.dumps({"has_chart": has_chart, "caption": caption})


# Representative canned replies (one per scenario the live model produces).
CHART_CAPTION = (
    "A grouped bar chart titled 'Scope 1 Emissions' with years on the x-axis and "
    "tCO2e on the y-axis; stationary combustion 1,813 tCO2e in 2023; total 6,032 "
    "tCO2e. Emissions rose over the period."
)
# A purely conceptual diagram: categories + a relationship word, and NO digits.
DIAGRAM_CAPTION = (
    "A segmented category wheel diagram titled 'Ecosystem of Sustainable Finance'. "
    "Components: Policy, Products, Market Infrastructure, Coordination, "
    "Non-governmental Support, Human Resources, Awareness; grouped as interconnected "
    "parts of one ecosystem."
)
# The tightened behavior: notes a table exists but restates none of its Rp values.
EMBEDDED_TABLE_CAPTION = (
    "A timeline diagram of sustainable-finance milestones from 2012 to 2019, in four "
    "stages from Starting the Journey to Achievements. The diagram includes a data "
    "table of financial figures, whose values are not restated here."
)
RP_VALUES = ("913.15", "809.75", "59.9", "35.6")  # figures that must never appear


# --- unit: detection & mapping ---------------------------------------------


def test_chart_detected_returns_caption_and_default_model() -> None:
    client = FakeClient(_body(True, CHART_CAPTION))
    result = caption_page_image(b"img-bytes", client=client, settings=_settings())
    assert result.has_chart is True
    assert result.caption == CHART_CAPTION
    assert result.model == "gpt-4.1-mini"


def test_diagram_detected_keeps_categories_and_relationships_without_numbers() -> None:
    client = FakeClient(_body(True, DIAGRAM_CAPTION))
    result = caption_page_image(b"img", client=client, settings=_settings())
    assert result.has_chart is True
    assert result.caption is not None
    # Categories named individually and a relationship word survive verbatim.
    for category in ("Policy", "Products", "Human Resources", "Awareness"):
        assert category in result.caption
    assert "interconnected" in result.caption
    # A purely conceptual diagram carries no numbers, none fabricated.
    assert not any(ch.isdigit() for ch in result.caption)


@pytest.mark.parametrize("scenario", ["pure_table", "pure_text"])
def test_no_graphic_declined(scenario: str) -> None:
    """Pure table and pure text both return has_chart:false -> declined."""
    client = FakeClient(_body(False, ""))
    result = caption_page_image(b"img", client=client, settings=_settings())
    assert result.has_chart is False
    assert result.caption is None


def test_table_embedded_in_figure_notes_table_but_restates_no_values() -> None:
    client = FakeClient(_body(True, EMBEDDED_TABLE_CAPTION))
    result = caption_page_image(b"img", client=client, settings=_settings())
    assert result.has_chart is True  # still a diagram, so still captioned
    assert result.caption is not None
    assert "data table" in result.caption  # may note the table's presence...
    for value in RP_VALUES:  # ...but never restate its values
        assert value not in result.caption


def test_has_chart_true_but_blank_caption_treated_as_no_graphic() -> None:
    """An inconsistent 'chart but empty caption' reply must not emit a blank chunk."""
    client = FakeClient(_body(True, "   "))
    result = caption_page_image(b"img", client=client, settings=_settings())
    assert result.has_chart is False
    assert result.caption is None


# --- unit: model selection --------------------------------------------------


def test_escalation_selects_high_model() -> None:
    client = FakeClient(_body(True, CHART_CAPTION))
    result = caption_page_image(b"img", escalate=True, client=client, settings=_settings())
    assert result.model == "gpt-4.1"
    assert client.calls[0]["model"] == "gpt-4.1"


def test_default_selects_mini_model() -> None:
    client = FakeClient(_body(True, CHART_CAPTION))
    caption_page_image(b"img", client=client, settings=_settings())
    assert client.calls[0]["model"] == "gpt-4.1-mini"


# --- unit: request construction & errors ------------------------------------


def test_request_sends_image_as_data_url_and_both_prompts() -> None:
    client = FakeClient(_body(True, CHART_CAPTION))
    caption_page_image(b"PNGDATA", image_format="png", client=client, settings=_settings())
    call = client.calls[0]
    assert call["response_format"] == {"type": "json_object"}
    messages = call["messages"]
    assert [m["role"] for m in messages] == ["system", "user"]
    user_content = messages[1]["content"]
    assert {part["type"] for part in user_content} == {"text", "image_url"}
    image_part = next(p for p in user_content if p["type"] == "image_url")
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")


def test_missing_api_key_raises_vision_caption_error() -> None:
    with pytest.raises(VisionCaptionError):
        caption_page_image(b"img", settings=_settings(api_key=""))


def test_invalid_json_reply_raises_vision_caption_error() -> None:
    client = FakeClient("this is not json")
    with pytest.raises(VisionCaptionError):
        caption_page_image(b"img", client=client, settings=_settings())


# --- live tests (opt-in: real key + source PDFs) ----------------------------

_RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
# Explicit opt-in only: a plain `pytest` run must never hit the real API, even
# when a real key is present in .env. Set RUN_LIVE_VISION_TESTS=1 to enable.
_RUN_LIVE = os.environ.get("RUN_LIVE_VISION_TESTS") == "1"
live = pytest.mark.skipif(
    not _RUN_LIVE,
    reason="RUN_LIVE_VISION_TESTS!=1; skipping live vision tests (opt-in only)",
)


def _live_caption(rel_pdf: str, page: int) -> PageCaption:
    """Render a real page and caption it with the real model, or skip if absent."""
    from app.rag.ingestion.figure_extractor import render_page

    pdf = _RAW / rel_pdf
    if not pdf.exists():
        pytest.skip(f"source PDF not present: {pdf}")
    rendered = render_page(pdf, page)
    return caption_page_image(rendered.image_bytes, image_format=rendered.image_format)


@live
def test_live_double_materiality_captioned() -> None:
    result = _live_caption("standards/EDGB.pdf", 15)
    assert result.has_chart is True
    assert result.caption


@live
def test_live_pure_table_page_declined() -> None:
    result = _live_caption("company_documents/BANK-BCA.pdf", 2)
    assert result.has_chart is False
    assert result.caption is None


@live
def test_live_bordered_table_page_declined_no_restated_values() -> None:
    """Regression guard: a styled/bordered GHG table must decline as a table, not a chart."""
    result = _live_caption("standards/EDGB.pdf", 49)
    assert result.has_chart is False
    assert result.caption is None
