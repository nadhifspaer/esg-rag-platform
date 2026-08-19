"""Figure captioning: a rendered page image -> a plain-language caption."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass

from openai import OpenAI

from app.core.config import Settings, get_settings

# The model is shown a full page and must describe charts and diagrams precisely
# while refusing to transcribe standalone tables or invent data that isn't there.
SYSTEM_PROMPT = (
    "You are a meticulous visual-content extraction assistant for ESG and "
    "sustainability reports published by Indonesian banks and regulators. You are "
    "shown one full page image and must caption its GRAPHICAL content. Two kinds "
    "of graphical content are in scope:\n\n"
    "1. DATA CHARTS / GRAPHS — bar, line, pie/donut, area, scatter, and similar "
    "visualizations where data is shown pictorially (bar heights, line positions, "
    "pie slices).\n"
    "2. CONCEPTUAL DIAGRAMS / INFOGRAPHICS — timelines, process or cycle wheels, "
    "relationship or flow maps, hierarchy/organization diagrams, and segmented "
    "category wheels, where the meaning is in the structure and the relationships "
    "between parts rather than in plotted numbers.\n\n"
    "CRITICAL — A TABLE IS NOT A GRAPHIC. A table is any content laid out as rows "
    "and columns of cells: cell borders or gridlines, a styled or colored header "
    "row, and values or labels aligned in columns. Cell borders, header rows, and "
    "column grids are TABLE signals, NOT chart signals — no matter how visually "
    "structured, styled, or colorful the table is, and whether its cells hold "
    "numbers or words. If the page's prominent structured visual content is a "
    "table, it is a TABLE PAGE: report NO graphic. Never reclassify a styled table "
    "as a 'wheel', 'diagram', 'infographic', or 'chart'. This rule has no "
    "exceptions.\n\n"
    "The only subtlety: a genuine chart or diagram (a bar/line/pie chart, or a "
    "timeline, category wheel, relationship map, or flow diagram) may have a small "
    "table sitting beside or beneath it. Caption the page only when such a genuine "
    "chart or diagram is the PRIMARY graphical content; then describe that diagram "
    "and merely note any secondary table exists. If the table is the main event, "
    "there is no graphic to caption.\n\n"
    "Ignore plain body text. Numbers inside any bordered box, grid, or table are "
    "off-limits everywhere — never restate a table's values, even a table embedded "
    "in a diagram. Another tool extracts tables structurally and is the sole source "
    "of truth for any tabular number; you may note a table exists but must not "
    "reproduce its contents. If the page has neither a chart nor a meaningful "
    "diagram — only body text and/or tables — report that there is no graphic.\n\n"
    "Never invent or estimate anything. Values shown pictorially as part of a "
    "chart — a bar's height, a plotted point, a labelled pie slice — are in scope "
    "and should be read out; tabular numbers (per the rule above) are not. For a "
    "purely conceptual diagram that shows no plotted data, describe its structure "
    "and relationships and do NOT fabricate axis labels, quantities, units, or "
    "data points. If something is present but not legible, say it is unclear "
    "rather than guessing — an acknowledged gap is far better than a confidently "
    "wrong number in a citation."
)

USER_PROMPT = (
    "Examine this page and first identify its prominent structured visual content. "
    "If that content is a TABLE — rows and columns of cells, with borders/gridlines, "
    "a header row, or column-aligned values, however styled or colorful — then the "
    "page has NO graphic to caption; return has_chart=false. Only when a genuine "
    "data chart or conceptual diagram is the PRIMARY content do you caption it, "
    "writing a single plain-language caption that a search system can index, "
    "following the matching instructions below.\n\n"
    "For each DATA CHART, the caption must state:\n"
    "- the chart type (e.g. grouped bar chart, line chart, donut chart);\n"
    "- the chart title or heading, if one is shown;\n"
    "- the axis labels and their units (both the x-axis and the y-axis);\n"
    "- every specific data point, each with its unit and its year (or category), "
    "listed individually rather than summarized away;\n"
    "- the overall trend or takeaway in plain language (e.g. 'total emissions fell "
    "steadily from 2020 to 2023').\n\n"
    "For each CONCEPTUAL DIAGRAM / INFOGRAPHIC, the caption must state:\n"
    "- the diagram type (e.g. timeline, process/cycle wheel, relationship map, "
    "hierarchy diagram, segmented category wheel);\n"
    "- the key categories, stages, or entities it shows, named individually;\n"
    "- how they relate to each other (a sequence, a cycle, a hierarchy, "
    "cause-and-effect, or a grouping);\n"
    "- do NOT restate numeric figures from any data table embedded in the diagram "
    "(for example an Rp figures box on a timeline); you may note that the diagram "
    "includes a data table, but table_extractor — not this caption — records its "
    "values. Never invent axes, quantities, units, or data points that the diagram "
    "does not itself show pictorially.\n\n"
    "If the page has several graphics, describe each of them in turn within the "
    "same caption.\n\n"
    'Respond ONLY with a JSON object of the form {"has_chart": true, "caption": '
    '"<caption text>"}. Set "has_chart" to false with an empty caption whenever the '
    "page has no genuine chart or diagram as its primary graphical content — "
    "including any page whose main structured content is a table (even a styled or "
    "bordered one), or that is only body text. A table is never a chart."
)


class VisionCaptionError(RuntimeError):
    """Raised when the vision model cannot be called or returns unusable output."""


@dataclass(frozen=True)
class PageCaption:
    """The result of captioning one page image; `caption` is None whenever `has_chart` is False."""

    has_chart: bool  # a chart OR a meaningful diagram is present (see module docstring)
    caption: str | None
    model: str


def _to_data_url(image_bytes: bytes, image_format: str) -> str:
    """Encode raw image bytes as a base64 `data:` URL for the vision API."""
    fmt = image_format.lower()
    mime = "image/jpeg" if fmt in {"jpg", "jpeg"} else f"image/{fmt}"
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _resolve_client(settings: Settings, injected: OpenAI | None) -> OpenAI:
    """Return the injected client, or build one from settings (needs the API key)."""
    if injected is not None:
        return injected
    if not settings.openai_api_key:
        raise VisionCaptionError(
            "OPENAI_API_KEY is not set; cannot call the vision model. "
            "Add it to your .env (see .env.example)."
        )
    return OpenAI(api_key=settings.openai_api_key)


def _parse_response(raw: str, model: str) -> PageCaption:
    """Turn the model's JSON string into a `PageCaption` (chart only when flag AND text present)."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise VisionCaptionError(f"vision model did not return valid JSON: {exc}") from exc

    text = str(data.get("caption") or "").strip()
    has_chart = bool(data.get("has_chart")) and bool(text)
    return PageCaption(has_chart=has_chart, caption=text if has_chart else None, model=model)


def caption_page_image(
    image_bytes: bytes,
    *,
    image_format: str = "png",
    escalate: bool = False,
    client: OpenAI | None = None,
    settings: Settings | None = None,
) -> PageCaption:
    """Caption the chart on one rendered page image (gpt-4.1-mini default, gpt-4.1 escalated)."""
    settings = settings or get_settings()
    client = _resolve_client(settings, client)
    model = settings.openai_vision_model_high if escalate else settings.openai_vision_model

    response = client.chat.completions.create(
        model=model,
        temperature=0,  # deterministic captions: same page -> same reading, run to run
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    # "high" detail so small axis labels and legend text are legible.
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _to_data_url(image_bytes, image_format),
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
    )

    content = response.choices[0].message.content or ""
    return _parse_response(content, model)
