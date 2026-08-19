from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.models.payload import Domain

# Repo-root manifest, resolved independently of the working directory.
_MANIFEST_PATH = Path(__file__).resolve().parents[3] / "data" / "raw" / "manifest.json"

# Surface forms too generic to be reliable entity signals (e.g. "bank" is Bank Aladin's ticker).
_AMBIGUOUS_SURFACE_FORMS = frozenset({"bank"})


class EntityKind(StrEnum):
    """What sort of thing an entity is, maps 1:1 to a domain, named from the query's POV."""

    BANK = "bank"
    STANDARD = "standard"


@dataclass(frozen=True)
class Entity:
    """One logical entity from the manifest: a bank or a standard, with all its surface forms."""

    kind: EntityKind
    domain: Domain
    canonical_name: str
    surface_forms: tuple[str, ...]
    bank: str | None = None
    official_title: str | None = None


@dataclass(frozen=True)
class EntityMatch:
    """An entity the query named, plus the surface form that triggered the match."""

    entity: Entity
    matched_text: str


def _normalize(text: str) -> str:
    """Lowercase and collapse internal whitespace to single spaces."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _boundary_pattern(surface_form: str) -> re.Pattern[str]:
    """Compile a whole-token matcher for a normalized surface form."""
    escaped = re.escape(surface_form).replace(r"\ ", r"\s+").replace(" ", r"\s+")
    return re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])")


@dataclass(frozen=True)
class _CompiledEntity:
    """An Entity paired with its precompiled surface-form patterns (built once)."""

    entity: Entity
    patterns: tuple[tuple[str, re.Pattern[str]], ...]  # (normalized surface form, matcher)


def _dedupe_surface_forms(raw_forms: list[str]) -> tuple[str, ...]:
    """Normalize, drop the ambiguous denylist and empties, and dedupe preserving order."""
    seen: dict[str, None] = {}
    for form in raw_forms:
        norm = _normalize(form)
        if not norm or norm in _AMBIGUOUS_SURFACE_FORMS:
            continue
        seen.setdefault(norm, None)
    return tuple(seen)


def _build_entities(domains: dict[str, list[dict[str, Any]]]) -> list[Entity]:
    """Turn the manifest's `domains` mapping into a flat list of `Entity`."""
    entities: list[Entity] = []

    for entry in domains.get(Domain.COMPANY_DOCUMENTS.value, []):
        bank = entry["bank"]
        forms = _dedupe_surface_forms([bank, *entry.get("aliases", [])])
        entities.append(
            Entity(
                kind=EntityKind.BANK,
                domain=Domain.COMPANY_DOCUMENTS,
                canonical_name=bank,
                surface_forms=forms,
                bank=bank,
            )
        )

    for entry in domains.get(Domain.STANDARDS.value, []):
        title = entry["official_title"]
        forms = _dedupe_surface_forms([title, *entry.get("aliases", [])])
        entities.append(
            Entity(
                kind=EntityKind.STANDARD,
                domain=Domain.STANDARDS,
                canonical_name=title,
                surface_forms=forms,
                official_title=title,
            )
        )

    return entities


class EntityIndex:
    """Matches a query against every known entity's canonical name and aliases."""

    def __init__(self, entities: list[Entity]) -> None:
        self._compiled: list[_CompiledEntity] = [
            _CompiledEntity(
                entity=entity,
                patterns=tuple((form, _boundary_pattern(form)) for form in entity.surface_forms),
            )
            for entity in entities
        ]

    @property
    def entities(self) -> tuple[Entity, ...]:
        """All indexed entities, used to build the LLM fallback's catalog."""
        return tuple(c.entity for c in self._compiled)

    @classmethod
    def from_manifest(cls, manifest_path: Path | None = None) -> EntityIndex:
        """Load and index the manifest at `manifest_path` (default: the repo manifest)."""
        path = manifest_path or _MANIFEST_PATH
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(_build_entities(raw.get("domains", {})))

    def match(self, query: str) -> list[EntityMatch]:
        """Return one `EntityMatch` per distinct entity named (longest form wins per entity)."""
        normalized = _normalize(query)
        matches: list[EntityMatch] = []
        for compiled in self._compiled:
            best: str | None = None
            for form, pattern in compiled.patterns:
                if pattern.search(normalized) and (best is None or len(form) > len(best)):
                    best = form
            if best is not None:
                matches.append(EntityMatch(entity=compiled.entity, matched_text=best))
        return matches


@lru_cache
def default_entity_index() -> EntityIndex:
    """Return a cached `EntityIndex` built from the repo's `data/raw/manifest.json`."""
    return EntityIndex.from_manifest()
