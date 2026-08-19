"""Value extraction: step two of the metric aggregation pipeline (6.5.4)."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from openai import OpenAI

from app.core.config import Settings, get_settings
from app.models.citation import Citation
from app.rag.metrics.aggregation import BankMetricEvidence
from app.rag.metrics.metric import MetricSpec

_TEMPERATURE = 0.0

# How many top evidence chunks to feed the model.
_MAX_EVIDENCE_CHUNKS = 3

# Injected model call: (system_prompt, user_prompt) -> raw JSON string.
ExtractionModel = Callable[[str, str], str]


class ExtractionError(RuntimeError):
    """Raised when extraction cannot run (no key) or the model returns unusable output."""


class ExtractionStatus(StrEnum):
    """How many values the evidence disclosed for the metric."""

    DISCLOSED = "disclosed"  # exactly one value
    MULTIPLE_DISCLOSED = "multiple_disclosed"  # >= 2 distinct, scope-labelled values
    NOT_DISCLOSED = "not_disclosed"  # none stated (or found=False)


@dataclass(frozen=True)
class ExtractedValue:
    """One disclosed value, kept verbatim, with the scope label the evidence gave it."""

    value: str
    label: str | None = None


@dataclass(frozen=True)
class MetricExtraction:
    """The structured extraction result for one metric against one bank."""

    bank: str
    metric_code: str
    value_type: str
    status: ExtractionStatus
    values: tuple[ExtractedValue, ...]
    citation: str | None
    note: str | None


_SYSTEM_PROMPT = (
    "You extract the value(s) a bank's sustainability report discloses for ONE ESG metric. "
    "You are given the METRIC (its description and expected value_type) and EVIDENCE "
    "(verbatim excerpts from the report). Report ONLY what the EVIDENCE literally states — "
    "never guess, infer, estimate, convert, or round.\n\n"
    "value_type is one of:\n"
    "- year: a four-digit calendar year (e.g. 2050).\n"
    "- percentage: a percent value (e.g. 34.5).\n"
    "- numeric: a bare quantity (e.g. 1.37); its unit may be reported separately — do not "
    "invent one.\n\n"
    'Respond ONLY with a JSON object: {"values": [{"value": "<verbatim value>", "label": '
    '"<scope or null>"}], "note": "<short clarifier or null>"}.\n\n'
    "Rules:\n"
    '- If the EVIDENCE states exactly one value, return one entry with "label": null.\n'
    "- If it states MORE THAN ONE distinct value for this metric under different scopes or "
    "qualifiers (e.g. an operational target year AND a separate financing target year), "
    'return one entry per value, each "label" naming its scope exactly as the evidence '
    "labels it. Do NOT pick one and drop the others.\n"
    '- If the EVIDENCE states NO value — it is absent, null, "No", "Tidak", "N/A", or the '
    'report says it has no such target/figure — return "values": [] and say why in "note".\n'
    "- Copy figures, years, and percentages EXACTLY as written (keep the original decimal "
    'mark, e.g. "1,37").\n'
    "- Only report values actually present in the EVIDENCE. When in doubt, return an empty "
    "list. Never fabricate a value the evidence does not contain."
)


def _render_evidence(
    bank_evidence: BankMetricEvidence, max_chunks: int = _MAX_EVIDENCE_CHUNKS
) -> str:
    """Render the top evidence chunks (with page markers) into the prompt's EVIDENCE block."""
    blocks = [
        f"[p.{result.payload.page_number}] {result.payload.chunk_text.strip()}"
        for result in bank_evidence.evidence.results[:max_chunks]
    ]
    return "\n\n".join(blocks)


def _user_prompt(metric: MetricSpec, evidence_text: str) -> str:
    """The model's user turn: the metric, its value_type, and the evidence to read."""
    return (
        f"METRIC: {metric.description}\n"
        f"value_type: {metric.value_type.value}\n\n"
        f"EVIDENCE:\n{evidence_text}\n\n"
        "Extract the disclosed value(s) as the required JSON."
    )


def _default_model(client: OpenAI | None, settings: Settings) -> ExtractionModel:
    """Build the real OpenAI extraction call: one non-streamed JSON completion, cheap model."""
    resolved = client or (
        OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
    )
    if resolved is None:
        raise ExtractionError("OPENAI_API_KEY is not set; cannot extract metric values.")

    def _call(system_prompt: str, user_prompt: str) -> str:
        response = resolved.chat.completions.create(
            model=settings.openai_generation_model,
            temperature=_TEMPERATURE,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""

    return _call


def _clean(text: object) -> str | None:
    """Trim a model-returned string; treat empty/'null' as None."""
    if text is None:
        return None
    stripped = str(text).strip()
    return stripped if stripped and stripped.lower() != "null" else None


def _parse(raw: str, metric: MetricSpec, bank_evidence: BankMetricEvidence) -> MetricExtraction:
    """Turn the model's JSON into a `MetricExtraction`, deriving status from the value count."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ExtractionError(f"extraction did not return valid JSON: {exc}") from exc

    values: list[ExtractedValue] = []
    for entry in data.get("values") or []:
        value = _clean(entry.get("value") if isinstance(entry, dict) else entry)
        if value is None:
            continue  # an entry with no actual value is not a disclosure
        label = _clean(entry.get("label")) if isinstance(entry, dict) else None
        values.append(ExtractedValue(value=value, label=label))

    # Status is derived from how many distinct values the evidence actually stated.
    if not values:
        status = ExtractionStatus.NOT_DISCLOSED
    elif len(values) == 1:
        status = ExtractionStatus.DISCLOSED
    else:
        status = ExtractionStatus.MULTIPLE_DISCLOSED

    citation = (
        Citation.from_result(bank_evidence.evidence.results[0]).label
        if bank_evidence.evidence.results
        else None
    )
    return MetricExtraction(
        bank=bank_evidence.bank,
        metric_code=metric.code,
        value_type=metric.value_type.value,
        status=status,
        values=tuple(values),
        citation=citation,
        note=_clean(data.get("note")),
    )


def extract_metric_value(
    metric: MetricSpec,
    bank_evidence: BankMetricEvidence,
    *,
    model: ExtractionModel | None = None,
    client: OpenAI | None = None,
    settings: Settings | None = None,
) -> MetricExtraction:
    """Extract `metric`'s disclosed value(s) from one bank's evidence (skipped if found=False)."""
    if not bank_evidence.found:
        return MetricExtraction(
            bank=bank_evidence.bank,
            metric_code=metric.code,
            value_type=metric.value_type.value,
            status=ExtractionStatus.NOT_DISCLOSED,
            values=(),
            citation=None,
            note="no evidence retrieved (found=False); extraction skipped",
        )

    settings = settings or get_settings()
    model = model or _default_model(client, settings)
    raw = model(_SYSTEM_PROMPT, _user_prompt(metric, _render_evidence(bank_evidence)))
    return _parse(raw, metric, bank_evidence)


def extract_metric_across_banks(
    metric: MetricSpec,
    bank_evidences: Sequence[BankMetricEvidence],
    *,
    model: ExtractionModel | None = None,
    client: OpenAI | None = None,
    settings: Settings | None = None,
) -> list[MetricExtraction]:
    """Extract `metric` for every bank's evidence: one LLM call per found bank, order preserved."""
    settings = settings or get_settings()
    # Build the model once and reuse across banks; found=False banks still skip it.
    resolved_model = model or _default_model(client, settings)
    return [
        extract_metric_value(metric, evidence, model=resolved_model, settings=settings)
        for evidence in bank_evidences
    ]
