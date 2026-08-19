"""Multi-turn context resolution: turning a follow-up utterance into a searchable question."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

from app.models.conversation import ConversationTurn, Pipeline
from app.models.payload import Domain
from app.rag.entity_index import EntityIndex, EntityMatch
from app.rag.query_classifier import QueryClassification

# Inline citation markers the generator emits; must not survive into a replayed assistant
# turn, since `render_sources` renumbers sources from 1 on every turn.
_CITATION_MARKER_RE = re.compile(r"\[\d+\]")

# Discourse/meta words with no ESG topic, used to test whether an utterance is substantive.
_FILLER_WORDS = frozenset(
    {
        # English framing / anaphora
        "a",
        "about",
        "again",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "clarify",
        "could",
        "did",
        "do",
        "does",
        "elaborate",
        "explain",
        "for",
        "from",
        "further",
        "give",
        "how",
        "i",
        "in",
        "is",
        "it",
        "its",
        "just",
        "like",
        "me",
        "mean",
        "means",
        "more",
        "much",
        "of",
        "ok",
        "okay",
        "on",
        "or",
        "please",
        "put",
        "same",
        "say",
        "show",
        "similar",
        "simpler",
        "simply",
        "so",
        "some",
        "still",
        "such",
        "tell",
        "terms",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "too",
        "up",
        "us",
        "was",
        "were",
        "what",
        "when",
        "which",
        "who",
        "why",
        "with",
        "would",
        "you",
        "your",
        # Indonesian framing / anaphora
        "adalah",
        "apa",
        "apakah",
        "atau",
        "bagaimana",
        "dan",
        "dari",
        "dengan",
        "di",
        "itu",
        "jelaskan",
        "juga",
        "kalau",
        "kenapa",
        "lagi",
        "lebih",
        "mengapa",
        "nya",
        "pada",
        "sama",
        "saya",
        "sederhana",
        "tentang",
        "tersebut",
        "untuk",
        "yang",
    }
)


class ContextSource(StrEnum):
    """How the retrieval seed for this turn was decided, reported in `meta` and traced."""

    # Self-contained: the utterance is used as-is. Every first turn, and any follow-up that
    # names both its entity and its topic. Byte-identical to the pre-multi-turn behaviour.
    UTTERANCE = "utterance"
    # "What about Mandiri?": names an entity, no topic. The new entity is substituted into
    # the previous turn's retrieval query.
    ENTITY_SWAP = "entity_swap"
    # "And waste?": names a topic, no entity. The entity is inherited for routing only;
    # the utterance itself is the seed.
    INHERITED_ENTITY = "inherited_entity"
    # "Explain that more simply.": neither. The previous turn's retrieval query is reused.
    PREVIOUS_QUERY = "previous_query"


@dataclass(frozen=True)
class ResolvedContext:
    """What this turn should answer, and what it should search for."""

    resolved_question: str
    retrieval_seed: str
    context_source: ContextSource
    named_entity: EntityMatch | None = None
    inherited_entity: EntityMatch | None = None
    compliance_follow_up: bool = False

    @property
    def entity(self) -> EntityMatch | None:
        """The entity this turn is about, however it was determined: named wins over inherited."""
        return self.named_entity or self.inherited_entity


def strip_citation_markers(text: str) -> str:
    """Remove inline `[n]` citation markers and tidy the resulting whitespace."""
    cleaned = _CITATION_MARKER_RE.sub("", text)
    # Markers usually sit before punctuation, so removing them leaves a stray space.
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
    return " ".join(cleaned.split())


def sanitize_history(turns: Iterable[ConversationTurn]) -> tuple[ConversationTurn, ...]:
    """Normalise client-supplied history into what is safe to render into a prompt."""
    cleaned: list[ConversationTurn] = []
    for turn in turns:
        question = " ".join(turn.question.split())
        answer = strip_citation_markers(turn.answer)
        if not question or not answer:
            continue
        cleaned.append(turn.model_copy(update={"question": question, "answer": answer}))
    return tuple(cleaned)


def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace, the same shape `EntityIndex` matches in."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _token_pattern(surface_form: str) -> re.Pattern[str]:
    """Whole-token matcher for a normalized form, mirroring `entity_index`'s boundary rule."""
    escaped = re.escape(surface_form).replace(r"\ ", r"\s+").replace(" ", r"\s+")
    return re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])")


def _strip_entity_forms(text: str, matches: Sequence[EntityMatch]) -> str:
    """Remove every surface form of the matched entities from normalized `text`."""
    stripped = text
    for match in matches:
        for form in match.entity.surface_forms:
            stripped = _token_pattern(form).sub(" ", stripped)
    return stripped


def _is_topical(utterance: str, matches: Sequence[EntityMatch]) -> bool:
    """True if the utterance says anything of its own beyond entity names and filler."""
    residue = _strip_entity_forms(_normalize(utterance), matches)
    for token in re.findall(r"[a-z0-9][a-z0-9'-]*", residue):
        if token not in _FILLER_WORDS:
            return True
    return False


def _previous_seed(history: Sequence[ConversationTurn]) -> str:
    """The most recent turn's search terms, for a follow-up that supplies none."""
    last = history[-1]
    if last.pipeline is Pipeline.CHAT and last.retrieval_query:
        return " ".join(last.retrieval_query.split())
    return last.question


def _inheritable_entity(
    history: Sequence[ConversationTurn], index: EntityIndex
) -> EntityMatch | None:
    """The entity to carry forward, or None if there is no unambiguous one."""
    for turn in reversed(history):
        matches = index.match(turn.question)
        if not matches:
            continue
        return matches[0] if len(matches) == 1 else None
    return None


def _substitute_entity(seed: str, previous: EntityMatch, new_form: str) -> str:
    """Swap the previous turn's entity for the new one inside the previous seed."""
    for form in sorted(previous.entity.surface_forms, key=len, reverse=True):
        pattern = _token_pattern(form)
        if pattern.search(_normalize(seed)):
            return " ".join(re.sub(pattern.pattern, new_form, seed, flags=re.IGNORECASE).split())
    return seed


def resolve_context(
    utterance: str,
    history: Sequence[ConversationTurn],
    *,
    index: EntityIndex,
) -> ResolvedContext:
    """Resolve a possibly-elliptical utterance against the conversation so far."""
    if not history:
        matches = index.match(utterance)
        return ResolvedContext(
            utterance,
            utterance,
            ContextSource.UTTERANCE,
            named_entity=matches[0] if len(matches) == 1 else None,
        )

    matches = index.match(utterance)
    topical = _is_topical(utterance, matches)

    if matches:
        if topical:
            return ResolvedContext(
                utterance,
                utterance,
                ContextSource.UTTERANCE,
                named_entity=matches[0] if len(matches) == 1 else None,
            )
        # Entity-swap; if the previous turn named none, `_substitute_entity` leaves the seed alone.
        previous = _inheritable_entity(history, index)
        seed = _previous_seed(history)
        if previous is not None and previous.entity != matches[0].entity:
            seed = _substitute_entity(seed, previous, matches[0].matched_text)
        return ResolvedContext(
            resolved_question=utterance,
            retrieval_seed=seed,
            context_source=ContextSource.ENTITY_SWAP,
            named_entity=matches[0] if len(matches) == 1 else None,
            # Only an entity swap is treated as a repeat compliance check: unambiguous,
            # unlike a topic-shift or a plain "why?" after a compliance answer.
            compliance_follow_up=history[-1].pipeline is Pipeline.COMPLIANCE,
        )

    inherited = _inheritable_entity(history, index)
    if inherited is None:
        # Nothing to carry forward: behave exactly as a first turn would.
        return ResolvedContext(utterance, utterance, ContextSource.UTTERANCE)

    if topical:
        return ResolvedContext(
            resolved_question=utterance,
            retrieval_seed=utterance,
            context_source=ContextSource.INHERITED_ENTITY,
            inherited_entity=inherited,
        )

    return ResolvedContext(
        resolved_question=utterance,
        retrieval_seed=_previous_seed(history),
        context_source=ContextSource.PREVIOUS_QUERY,
        inherited_entity=inherited,
    )


def apply_context(
    classification: QueryClassification, context: ResolvedContext
) -> QueryClassification:
    """Fold an inherited entity into a classification the utterance alone could not resolve."""
    inherited = context.inherited_entity
    if inherited is None or classification.entities:
        return classification

    domain: Domain = inherited.entity.domain
    return replace(
        classification,
        domains=(domain,),
        entities=(inherited,),
        confident=True,
        reason=(
            f"{classification.reason}; entity inherited from an earlier turn "
            f"({inherited.entity.canonical_name})"
        ),
    )
