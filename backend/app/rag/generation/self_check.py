"""Groundedness self-check: the judge for the chat pipeline's retry decision."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from openai import OpenAI

from app.core.config import Settings, get_settings
from app.models.citation import RetrievedChunk
from app.rag.generation.prompts import render_sources

_TEMPERATURE = 0.0


class Verdict(StrEnum):
    """The judge's decision. `SUFFICIENT` stops the loop; `INSUFFICIENT` triggers a retry."""

    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"


@dataclass(frozen=True)
class GroundednessDecision:
    """The self-check result: verdict plus a rewritten query when INSUFFICIENT."""

    verdict: Verdict
    rewritten_query: str | None


class SelfCheckError(RuntimeError):
    """Raised when the self-check cannot run (no key) or returns unusable output."""


JUDGE_SYSTEM_PROMPT = (
    "You are a strict groundedness judge for a retrieval-augmented ESG assistant covering "
    "Indonesian banking sustainability reports and ESG standards (GRI). You are given a "
    "QUESTION, an ANSWER the "
    "assistant produced, and the CONTEXT (retrieved source excerpts) the answer was based "
    "on. Your only job is to decide whether the CONTEXT actually supports the ANSWER to "
    "the QUESTION.\n\n"
    "Return exactly one verdict:\n"
    '- "sufficient": ALL THREE of the following hold. (1) The ANSWER states the actual '
    "facts, figures, or explanation the QUESTION asked for. (2) Every claim it makes is "
    "supported by the CONTEXT. (3) The ANSWER is not, in whole or in part, a report that "
    "the information is missing.\n"
    '- "insufficient": anything else.\n\n'
    "OVERRIDING RULE — READ THIS BEFORE DECIDING. If the ANSWER reports that the "
    'information is not present, the verdict is ALWAYS "insufficient", with no '
    "exceptions. This holds even when the ANSWER is completely accurate about the CONTEXT "
    "— that accuracy is precisely why it is easy to mislabel. An answer saying the data is "
    "absent is a correct description of a FAILED RETRIEVAL, not an answer to the user's "
    "question, and the verdict exists to decide whether to retrieve again, not to grade "
    "the answer's honesty. Verbatim examples that are all INSUFFICIENT:\n"
    "  * \"The provided sources do not contain any information regarding BCA's Scope 1 "
    'emissions."\n'
    '  * "The sources do not contain this information. The emissions data available is '
    'for a different bank."\n'
    '  * "The provided sources do not contain X. The sources only cover Y and Z."\n'
    "That last shape is the most easily mistaken one: naming what the CONTEXT *does* "
    "contain is still not answering the QUESTION, and the accompanying claims being "
    "supported does not rescue it. Partial counts too — if the QUESTION asks for several "
    "figures and the ANSWER supplies some while reporting the rest as missing, that is "
    "INSUFFICIENT.\n\n"
    "Do NOT confuse an absent disclosure with a negative one. If the CONTEXT shows the "
    'report itself answering "no" or "Tidak" — for example a net-zero commitment row '
    'answered "Tidak" with no target year — then the report HAS disclosed its position, '
    "and an ANSWER reporting that position is a real answer and can be "
    '"sufficient". The distinction is whether the SOURCE gave an answer (sufficient) or '
    "whether our retrieval failed to find one (insufficient).\n\n"
    "Judge ONLY against the CONTEXT provided. Do not use outside knowledge, and do not "
    "reward a fluent or confident answer that the CONTEXT does not back. When genuinely "
    'in doubt, prefer "insufficient".\n\n'
    'When the verdict is "insufficient", also write a rewritten search query more likely '
    "to retrieve the missing information on the next attempt. Keep the user's original "
    "intent — never switch to a different question — but make it specific and include the "
    "key identifiers that matter for retrieval where relevant: the bank name, or a "
    'standard code such as "GRI 305". '
    "Stay within these two knowledge domains; never point outside them. When the verdict "
    'is "sufficient", the rewritten query must be an empty string.\n\n'
    "The rewritten query is NOT a web search. It is matched against an internal index of "
    "already-retrieved document text by semantic similarity and exact keyword overlap, so "
    "write it as plain natural-language words and identifiers that would literally appear "
    "in the source documents. Never use web-search operator syntax — no site:, filetype:, "
    "inurl:, intitle:, ext:, no leading +/- operators, and no URLs or domain names. Those "
    "are not understood as operators here; they are matched as ordinary text, so they pull "
    "in whichever passage happens to contain that literal string (a query containing "
    '"site:bca.co.id" retrieves the page that prints the bank\'s website address) and they '
    "dilute the real search terms. Terms describing the wanted row or figure work best: "
    'for example "total direct emissions Scope 1 tCO2e stationary combustion", never '
    '"scope 1 emissions site:bca.co.id filetype:pdf".\n\n'
    "Respond ONLY with a JSON object of the form "
    '{"verdict": "sufficient" | "insufficient", "rewritten_query": "<query or '
    'empty string>"}.'
)


# Web-search operator names the rewritten query must never carry: an explicit allow-list.
_SEARCH_OPERATORS = (
    "site",
    "filetype",
    "ext",
    "inurl",
    "intitle",
    "intext",
    "allintitle",
    "allinurl",
    "cache",
    "related",
    "link",
    "define",
)
_OPERATOR_RE = re.compile(
    rf"(?<![\w-])[+-]?(?:{'|'.join(_SEARCH_OPERATORS)}):\S*",
    re.IGNORECASE,
)
# Explicit URLs only (scheme or www-prefixed), for the same reason.
_URL_RE = re.compile(r"(?<![\w-])(?:https?://|www\.)\S+", re.IGNORECASE)


def strip_search_operators(query: str) -> str:
    """Remove web-search operator syntax from a query, enforced in code, not by prompt."""
    cleaned = _URL_RE.sub(" ", _OPERATOR_RE.sub(" ", query))
    cleaned = " ".join(cleaned.split())
    return cleaned or query


def _resolve_client(settings: Settings, injected: OpenAI | None) -> OpenAI:
    """Return the injected client, or build one from settings (needs the API key)."""
    if injected is not None:
        return injected
    if not settings.openai_api_key:
        raise SelfCheckError("OPENAI_API_KEY is not set; cannot run the groundedness self-check.")
    return OpenAI(api_key=settings.openai_api_key)


def _user_message(query: str, answer: str, context: str) -> str:
    """The judge's user turn: the question, the answer, and the context it was based on."""
    return (
        f"QUESTION:\n{query}\n\n"
        f"ANSWER:\n{answer}\n\n"
        f"CONTEXT:\n{context}\n\n"
        "Judge whether the CONTEXT supports the ANSWER, and respond in the required JSON."
    )


def _parse(raw: str, query: str) -> GroundednessDecision:
    """Turn the judge's JSON reply into a validated `GroundednessDecision`."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SelfCheckError(f"self-check did not return valid JSON: {exc}") from exc

    try:
        verdict = Verdict(data["verdict"])
    except (KeyError, ValueError) as exc:
        raise SelfCheckError(f"self-check returned an unknown verdict: {exc}") from exc

    if verdict is Verdict.SUFFICIENT:
        return GroundednessDecision(verdict=verdict, rewritten_query=None)

    rewritten = strip_search_operators(str(data.get("rewritten_query") or "").strip())
    return GroundednessDecision(verdict=verdict, rewritten_query=rewritten or query)


def check_groundedness(
    query: str,
    answer: str,
    chunks: Sequence[RetrievedChunk],
    *,
    client: OpenAI | None = None,
    settings: Settings | None = None,
) -> GroundednessDecision:
    """Judge whether `answer` is supported by `chunks` as a response to `query`."""
    settings = settings or get_settings()
    resolved_client = _resolve_client(settings, client)
    context, _ = render_sources(chunks)

    try:
        response = resolved_client.chat.completions.create(
            model=settings.openai_generation_model,
            temperature=_TEMPERATURE,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": _user_message(query, answer, context)},
            ],
        )
    except SelfCheckError:
        raise
    except Exception as exc:  # network/API failure -> a clear, typed error
        raise SelfCheckError(f"self-check request failed: {exc}") from exc

    return _parse(response.choices[0].message.content or "", query)
