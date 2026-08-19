"""Prompt templates for citation-grounded chat generation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.models.citation import Citation, RetrievedChunk
from app.models.conversation import ConversationTurn
from app.rag.query_classifier import QueryType

# The query types these chat prompts can actually build. Exported so callers gate on the
# same set the dispatcher uses, rather than a parallel list that can drift.
CHAT_GENERATION_TYPES = frozenset({QueryType.SINGLE_DOMAIN, QueryType.CROSS_DOMAIN})

# Shared closing rules on how to cite. The `[1]` / `[2][3]` examples are load-bearing: they
# show the model the exact marker syntax the caller parses back to Citations.
_CITATION_RULES = (
    "Cite inline. Immediately after each claim, add the supporting source number(s) in "
    "square brackets, e.g. [1] or [2][3] when several sources support it. Cite only "
    "sources you actually used, and never a number that is not in the SOURCES list. "
    "Every factual sentence must carry at least one citation.\n"
    "Ground every statement in the SOURCES — never use outside knowledge or assumptions. "
    "If the SOURCES do not contain the answer, say so plainly rather than guessing. "
    "Quote figures, dates, and identifiers (emission amounts, years, and "
    "standard codes) exactly as they appear; do not round, convert, or estimate.\n"
    "Answer in the same language as the question. Be concise and specific."
)

SINGLE_DOMAIN_SYSTEM_PROMPT = (
    "You are an ESG research assistant for the Indonesian banking sector. You answer a "
    "question using ONLY the numbered SOURCES provided — excerpts retrieved from a "
    "single knowledge domain (bank sustainability reports, or ESG reporting standards "
    "such as a GRI/EDGB standard). Answer the question directly from those excerpts.\n\n"
    + _CITATION_RULES
)

CROSS_DOMAIN_SYSTEM_PROMPT = (
    "You are an ESG research assistant for the Indonesian banking sector. You answer a "
    "comparison or cross-reference question using ONLY the numbered SOURCES provided. "
    "The SOURCES come from MORE THAN ONE knowledge domain — for example a bank's "
    "sustainability report alongside a GRI/EDGB standard — "
    "and your job is to relate them accurately.\n"
    "Keep the sources distinct. A standard's requirement or definition "
    "is NOT the same as a company's reported figure or practice; never merge them into "
    "one claim or let one stand in for the other. Attribute each fact to the specific "
    "source it came from. Make the comparison legible: state what each source says, then "
    "how they relate — align, differ, or leave a gap. If one side of the comparison is "
    "absent from the SOURCES (the requirement is present but the company's disclosure is "
    "not, or vice versa), say so explicitly instead of inferring it. This is a "
    "descriptive document comparison, not a certified compliance audit.\n\n" + _CITATION_RULES
)


# Appended only when history is present, so a first turn's prompt stays byte-identical.
_HISTORY_RULES = (
    "\n\nThe messages before the final one are earlier turns of this same conversation. "
    'They are provided so you can resolve what the question refers to — "that", "its", '
    "or a bank or standard named earlier — and for nothing else. Treat them as "
    "context, never as evidence: a claim made in an earlier answer is not a source, and must "
    "not be repeated as fact unless the current SOURCES support it. Source numbering restarts "
    "at [1] on every turn, so cite only numbers from the SOURCES list in the final message."
)


@dataclass(frozen=True)
class GenerationPrompt:
    """A ready-to-send prompt plus the citation map for its numbered sources."""

    messages: list[dict[str, str]]
    citations: list[Citation]


def render_sources(chunks: Sequence[RetrievedChunk]) -> tuple[str, list[Citation]]:
    """Render retrieved chunks into a numbered SOURCES block and the matching citations."""
    citations = [Citation.from_result(chunk) for chunk in chunks]
    blocks = [
        f"[{i}] {citation.label}\n{chunk.payload.chunk_text.strip()}"
        for i, (chunk, citation) in enumerate(zip(chunks, citations, strict=True), start=1)
    ]
    text = "\n\n".join(blocks) if blocks else "(no sources retrieved)"
    return text, citations


def _user_message(sources_text: str, question: str) -> str:
    """The user turn: the numbered sources, the question, and a citation reminder."""
    return (
        f"SOURCES:\n{sources_text}\n\n"
        f"QUESTION: {question}\n\n"
        "Answer using only the SOURCES above, with an inline [n] citation for every claim."
    )


def render_history_messages(
    history: Sequence[ConversationTurn],
) -> list[dict[str, str]]:
    """Render prior turns as real `user`/`assistant` messages, assumed already sanitised."""
    messages: list[dict[str, str]] = []
    for turn in history:
        messages.append({"role": "user", "content": turn.question})
        messages.append({"role": "assistant", "content": turn.answer})
    return messages


def _build(
    system_prompt: str,
    question: str,
    chunks: Sequence[RetrievedChunk],
    history: Sequence[ConversationTurn] = (),
) -> GenerationPrompt:
    """Assemble a GenerationPrompt from a system prompt, the question, and the chunks."""
    sources_text, citations = render_sources(chunks)
    system_content = system_prompt + (_HISTORY_RULES if history else "")
    messages = [{"role": "system", "content": system_content}]
    messages.extend(render_history_messages(history))
    messages.append({"role": "user", "content": _user_message(sources_text, question)})
    return GenerationPrompt(messages=messages, citations=citations)


def build_single_domain_prompt(
    question: str,
    chunks: Sequence[RetrievedChunk],
    history: Sequence[ConversationTurn] = (),
) -> GenerationPrompt:
    """Build the prompt for a single-domain lookup query."""
    return _build(SINGLE_DOMAIN_SYSTEM_PROMPT, question, chunks, history)


def build_cross_domain_prompt(
    question: str,
    chunks: Sequence[RetrievedChunk],
    history: Sequence[ConversationTurn] = (),
) -> GenerationPrompt:
    """Build the prompt for a cross-domain comparison query."""
    return _build(CROSS_DOMAIN_SYSTEM_PROMPT, question, chunks, history)


def build_chat_prompt(
    query_type: QueryType,
    question: str,
    chunks: Sequence[RetrievedChunk],
    history: Sequence[ConversationTurn] = (),
) -> GenerationPrompt:
    """Dispatch to the right template for a chat query type (rejects COMPLIANCE_CHECK)."""
    if query_type not in CHAT_GENERATION_TYPES:
        raise ValueError(
            f"query_type {query_type.value!r} is not a chat generation type; it is not in "
            f"CHAT_GENERATION_TYPES ({sorted(t.value for t in CHAT_GENERATION_TYPES)}). "
            "Callers should reject it before reaching generation."
        )
    if query_type is QueryType.SINGLE_DOMAIN:
        return build_single_domain_prompt(question, chunks, history)
    if query_type is QueryType.CROSS_DOMAIN:
        return build_cross_domain_prompt(question, chunks, history)
    raise ValueError(  # pragma: no cover - unreachable while the set and branches agree
        f"query_type {query_type.value!r} is not a chat generation type; compliance-check "
        "queries are handled by app/rag/compliance/, not these prompts."
    )
