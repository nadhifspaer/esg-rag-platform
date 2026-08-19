from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum

from openai import OpenAI

from app.core.config import Settings, get_settings
from app.models.payload import Domain
from app.rag.entity_index import Entity, EntityIndex, EntityKind, EntityMatch, default_entity_index


class QueryType(StrEnum):
    """The four query shapes, each with a distinct retrieval route."""

    SINGLE_DOMAIN = "single_domain"
    CROSS_DOMAIN = "cross_domain"
    COMPLIANCE_CHECK = "compliance_check"
    # Spans the whole corpus rather than one report; routed to the enumerate-all-banks
    # pipelines (metric aggregation or the compliance sweep), see `api/rank.py`.
    AGGREGATION_RANKING = "aggregation_ranking"


class ClassifierMethod(StrEnum):
    """Which pass produced the classification, recorded for Langfuse tracing."""

    RULE_BASED = "rule_based"
    LLM_FALLBACK = "llm_fallback"


@dataclass(frozen=True)
class QueryClassification:
    """The classifier's output: query type, domain filter, matched entities, and provenance."""

    query_type: QueryType
    domains: tuple[Domain, ...]
    entities: tuple[EntityMatch, ...]
    method: ClassifierMethod
    confident: bool
    reason: str


# Compliance / comparison intent markers, bilingual (EN/ID). Broad by design: on their
# own they never force a compliance classification (see `_rule_based`), only escalate.
_COMPLIANCE_MARKERS: tuple[str, ...] = (
    r"compl(?:y|ies|iance|iant)",
    r"conform(?:s|ing|ance)?",
    r"align(?:s|ed|ment|ing)?",
    r"adher(?:e|es|ing|ence)",
    r"in accordance with",
    r"in line with",
    r"meet(?:s|ing)?",
    r"fulfil(?:s|ls|ling|led|lment)?",
    r"gaps?",
    r"gap analysis",
    # Indonesian
    r"sesuai",  # in accordance with / conform
    r"kepatuhan",  # compliance
    r"mematuhi",  # to comply with
    r"memenuhi",  # to meet / fulfil
    r"selaras",  # aligned / in harmony
)

_COMPLIANCE_RE = re.compile(r"(?<![a-z])(?:" + "|".join(_COMPLIANCE_MARKERS) + r")(?![a-z])")

# Aggregation/ranking intent markers: the query is about the corpus as a set, not one report.
_AGGREGATION_MARKERS: tuple[str, ...] = (
    r"top \d+",
    r"which banks?",
    r"what banks?",
    r"how many banks?",
    r"all banks?",
    r"across (?:all )?banks?",
    r"every bank",
    r"rank(?:s|ed|ing)?",
    r"compare (?:all|the) banks?",
    r"highest",
    r"lowest",
    r"earliest",
    r"latest",
    r"longest",
    r"shortest",
    r"soonest",
    r"biggest",
    r"largest",
    r"smallest",
    r"greatest",
    r"newest",
    r"oldest",
    r"average",
    r"most",
    r"least",
    r"leaderboard",
    # Indonesian; "paling \w+" covers the periphrastic superlative family in one marker.
    r"paling \w+",
    r"bank mana",
    r"bank apa",  # "which/what bank", an equally common interrogative to "bank mana"
    r"berapa banyak bank",
    r"semua bank",
    r"seluruh bank",
    r"peringkat",
    r"tertinggi",
    r"terendah",
    r"rata-rata",
)

_AGGREGATION_RE = re.compile(r"(?<![a-z])(?:" + "|".join(_AGGREGATION_MARKERS) + r")(?![a-z])")

# A bare GRI disclosure code ("GRI 305-1"), distinct from a full standard title ("GRI 3: ...").
_GRI_CODE_RE = re.compile(r"(?<![a-z0-9])gri\s?(\d{3}(?:-\d+)?)(?![a-z0-9])")


def _gri_codes_named(query: str) -> tuple[str, ...]:
    """Bare GRI disclosure codes named in the query, deduped, in first-seen order."""
    seen: dict[str, None] = {}
    for match in _GRI_CODE_RE.finditer(_normalize(query)):
        seen.setdefault(f"GRI {match.group(1)}", None)
    return tuple(seen)


class QueryClassifierError(RuntimeError):
    """Raised when the LLM fallback is required but cannot be completed."""


def _normalize(text: str) -> str:
    """Same normalization the entity index uses, for marker scanning."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _has_compliance_intent(query: str) -> bool:
    """True if the query contains compliance/comparison phrasing."""
    return bool(_COMPLIANCE_RE.search(_normalize(query)))


def _has_aggregation_intent(query: str) -> bool:
    """True if the query asks about the corpus as a set rather than one document."""
    return bool(_AGGREGATION_RE.search(_normalize(query)))


def _domains_of(matches: list[EntityMatch]) -> list[Domain]:
    """Distinct domains among matched entities, in first-seen order."""
    seen: dict[Domain, None] = {}
    for match in matches:
        seen.setdefault(match.entity.domain, None)
    return list(seen)


# Standards-domain entities a curated checklist actually exists for: not every
# standards-domain entity has one, so `has_standard_or_reg` checks identity, not domain.
_CHECKLIST_STANDARD_TITLES = frozenset({"ESG Disclosure Guide Book"})


def _is_checklist_framework_entity(entity: Entity) -> bool:
    """True if `entity` is one of the checklist frameworks (GRI, EDGB)."""
    if entity.kind == EntityKind.STANDARD:
        return entity.canonical_name.startswith("GRI") or (
            entity.canonical_name in _CHECKLIST_STANDARD_TITLES
        )
    return False


def _rule_based(query: str, index: EntityIndex) -> tuple[QueryClassification, bool]:
    """Classify by rules alone. Returns `(classification, confident)`."""
    matches = index.match(query)
    domains = _domains_of(matches)
    gri_codes = _gri_codes_named(query)
    if gri_codes and Domain.STANDARDS not in domains:
        domains.append(Domain.STANDARDS)
    domain_set = set(domains)
    has_company = Domain.COMPANY_DOCUMENTS in domain_set
    has_standard_or_reg = bool(gri_codes) or any(
        _is_checklist_framework_entity(m.entity) for m in matches
    )
    compliance_intent = _has_compliance_intent(query)
    entities = tuple(matches)
    # Entity names plus any bare GRI codes, for reason strings: codes never appear in
    # `entities` itself, so this is what keeps them visible in the trace.
    named = tuple(m.entity.canonical_name for m in matches) + gri_codes

    # Aggregation/ranking outranks compliance intent: the compliance pipeline resolves
    # exactly one company report, so it can't answer "which banks comply with GRI 305-4".
    if _has_aggregation_intent(query):
        return (
            QueryClassification(
                query_type=QueryType.AGGREGATION_RANKING,
                domains=(Domain.COMPANY_DOCUMENTS,),
                entities=entities,
                method=ClassifierMethod.RULE_BASED,
                confident=True,
                reason=(
                    "aggregation/ranking phrasing: spans the corpus rather than one report"
                    + (" (compliance phrasing also present)" if compliance_intent else "")
                ),
            ),
            True,
        )

    # Compliance-check needs both sides named: a company and a standard.
    if compliance_intent and has_company and has_standard_or_reg:
        return (
            QueryClassification(
                query_type=QueryType.COMPLIANCE_CHECK,
                domains=tuple(domains),
                entities=entities,
                method=ClassifierMethod.RULE_BASED,
                confident=True,
                reason=(
                    f"compliance phrasing plus a company and a standard named ({', '.join(named)})"
                ),
            ),
            True,
        )

    # Compliance phrasing but only one side named (or neither): let the LLM settle it.
    if compliance_intent and not (has_company and has_standard_or_reg):
        return (
            QueryClassification(
                query_type=QueryType.COMPLIANCE_CHECK,
                domains=tuple(domains),
                entities=entities,
                method=ClassifierMethod.RULE_BASED,
                confident=False,
                reason="compliance phrasing but company and standard not both named",
            ),
            False,
        )

    # No compliance intent: route by how many domains the named entities span.
    if len(domains) >= 2:
        return (
            QueryClassification(
                query_type=QueryType.CROSS_DOMAIN,
                domains=tuple(domains),
                entities=entities,
                method=ClassifierMethod.RULE_BASED,
                confident=True,
                reason=f"entities span multiple domains: {[d.value for d in domains]}",
            ),
            True,
        )
    if len(domains) == 1:
        return (
            QueryClassification(
                query_type=QueryType.SINGLE_DOMAIN,
                domains=tuple(domains),
                entities=entities,
                method=ClassifierMethod.RULE_BASED,
                confident=True,
                reason=f"all named entities are in {domains[0].value}",
            ),
            True,
        )

    # No entity named at all: rules cannot pick a domain, that is the LLM's job.
    return (
        QueryClassification(
            query_type=QueryType.SINGLE_DOMAIN,
            domains=(),
            entities=entities,
            method=ClassifierMethod.RULE_BASED,
            confident=False,
            reason="no known entity named; domain undetermined by rules",
        ),
        False,
    )


# --- LLM fallback -----------------------------------------------------------

_LLM_SYSTEM_PROMPT = (
    "You classify a user's question about Indonesian banking ESG documents into a "
    "retrieval route. There are exactly two knowledge domains:\n"
    "- company_documents: sustainability / ESG disclosure reports published by "
    "individual banks.\n"
    "- standards: ESG reporting standards (the GRI standards, the ESG Disclosure "
    "Guide Book).\n\n"
    "And exactly four query types:\n"
    "- single_domain: answerable from one domain only.\n"
    "- cross_domain: needs two or more domains compared or combined, but is NOT a "
    "formal compliance check.\n"
    "- compliance_check: asks whether a specific bank's report meets / complies with / "
    "aligns with a standard (or the gaps between them). Requires a "
    "company on one side and a standard on the other.\n"
    "- aggregation_ranking: asks about the set of banks as a whole rather than one "
    "report — a ranking, a comparison across all/many banks, or a superlative or "
    'average over the corpus (e.g. "which bank has the earliest net-zero target", '
    '"top 5 banks by emissions intensity", "what is the average training hours '
    'across banks"). This does NOT require a specific bank to be named — the question '
    "spans the corpus, so an empty entities list is normal and correct here, unlike "
    'compliance_check. Domains is always ["company_documents"] for this type.\n\n'
    "Only use domains and entity names from the provided catalogue. If the question "
    "names something not in the catalogue, still pick the most appropriate domain(s) "
    "but return an empty entities list. Respond ONLY with a JSON object: "
    '{"query_type": "...", "domains": ["..."], "entities": ["<canonical name>"], '
    '"reasoning": "<one short sentence>"}.'
)


def _entity_catalogue(index: EntityIndex) -> str:
    """A compact per-domain listing of canonical entity names for the LLM prompt."""
    by_domain: dict[Domain, list[str]] = {d: [] for d in Domain}
    for entity in index.entities:
        by_domain[entity.domain].append(entity.canonical_name)
    lines = []
    for domain, names in by_domain.items():
        if names:
            lines.append(f"{domain.value}: " + "; ".join(names))
    return "\n".join(lines)


def _resolve_client(settings: Settings, injected: OpenAI | None) -> OpenAI:
    """Return the injected client, or build one from settings (needs the API key)."""
    if injected is not None:
        return injected
    if not settings.openai_api_key:
        raise QueryClassifierError(
            "OPENAI_API_KEY is not set; cannot run the LLM classification fallback."
        )
    return OpenAI(api_key=settings.openai_api_key)


def _resolve_entities(names: list[str], index: EntityIndex) -> tuple[EntityMatch, ...]:
    """Map LLM-returned entity names back to indexed entities, dropping unknown names."""
    resolved: dict[str, EntityMatch] = {}
    for name in names:
        for match in index.match(name):
            resolved.setdefault(match.entity.canonical_name, match)
    return tuple(resolved.values())


def _parse_llm_response(raw: str, index: EntityIndex) -> QueryClassification:
    """Turn the LLM's JSON reply into a validated `QueryClassification`."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise QueryClassifierError(f"LLM classifier did not return valid JSON: {exc}") from exc

    try:
        query_type = QueryType(data["query_type"])
    except (KeyError, ValueError) as exc:
        raise QueryClassifierError(f"LLM returned an unknown query_type: {exc}") from exc

    domains: list[Domain] = []
    for value in data.get("domains", []):
        try:
            domain = Domain(value)
        except ValueError:
            continue  # ignore any domain outside the fixed three
        if domain not in domains:
            domains.append(domain)

    entities = _resolve_entities(data.get("entities", []), index)
    reasoning = str(data.get("reasoning") or "").strip() or "classified by LLM fallback"

    if query_type is QueryType.AGGREGATION_RANKING:
        # Same invariant `_rule_based` enforces: always company_documents, never a
        # domain the model might otherwise have guessed.
        domains = [Domain.COMPANY_DOCUMENTS]

    return QueryClassification(
        query_type=query_type,
        domains=tuple(domains),
        entities=entities,
        method=ClassifierMethod.LLM_FALLBACK,
        confident=True,
        reason=reasoning,
    )


def _classify_with_llm(
    query: str,
    index: EntityIndex,
    *,
    client: OpenAI | None,
    settings: Settings,
) -> QueryClassification:
    """Run the single-call LLM fallback classification."""
    client = _resolve_client(settings, client)
    user_prompt = (
        f"Entity catalogue:\n{_entity_catalogue(index)}\n\nClassify this question:\n{query}"
    )
    response = client.chat.completions.create(
        model=settings.openai_generation_model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _LLM_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return _parse_llm_response(response.choices[0].message.content or "", index)


def classify_query(
    query: str,
    *,
    index: EntityIndex | None = None,
    client: OpenAI | None = None,
    settings: Settings | None = None,
    use_llm_fallback: bool = True,
) -> QueryClassification:
    """Classify `query` into a type and domain filter, escalating to the LLM fallback if needed."""
    index = index or default_entity_index()
    settings = settings or get_settings()

    classification, confident = _rule_based(query, index)
    if confident or not use_llm_fallback:
        return classification

    try:
        return _classify_with_llm(query, index, client=client, settings=settings)
    except QueryClassifierError:
        # No key or a bad LLM reply: keep the rule-based best guess, still unconfident.
        return classification
