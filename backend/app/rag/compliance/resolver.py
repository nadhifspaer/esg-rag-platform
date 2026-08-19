"""Resolve an informal company reference to the exact stored `source_name`."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

from app.rag.entity_index import Entity, EntityIndex, EntityKind, default_entity_index
from app.rag.ingestion.payload_builder import build_source_name

# backend/app/rag/compliance/resolver.py -> parents[4] is the repo root, same manifest the
# entity index and ingestion read.
_MANIFEST_PATH = Path(__file__).resolve().parents[4] / "data" / "raw" / "manifest.json"


class CompanyResolutionError(ValueError):
    """Base class for a company reference that cannot be resolved to one report."""


class CompanyNotFoundError(CompanyResolutionError):
    """The reference names no known bank (nor anything else in the corpus)."""


class AmbiguousCompanyError(CompanyResolutionError):
    """The reference names more than one bank: which report to check is ambiguous."""


class NotACompanyError(CompanyResolutionError):
    """The reference resolves to a standard, not a company report."""


@cache
def _company_entries_by_bank(manifest_path: Path) -> dict[str, dict[str, Any]]:
    """Map each company report's canonical bank name to its full manifest entry."""
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = raw.get("domains", {}).get("company_documents", [])
    return {entry["bank"]: entry for entry in entries}


def all_company_reports(*, manifest_path: Path | None = None) -> list[tuple[str, str]]:
    """Every company report as `(canonical_bank_name, source_name)`, in manifest order."""
    manifest_path = manifest_path or _MANIFEST_PATH
    return [
        (bank, build_source_name(entry))
        for bank, entry in _company_entries_by_bank(manifest_path).items()
    ]


def source_name_for_entity(entity: Entity, *, manifest_path: Path | None = None) -> str | None:
    """The stored `source_name` for an already-matched bank `Entity`, or None if unscoped-able."""
    if entity.kind is not EntityKind.BANK:
        return None
    entries = _company_entries_by_bank(manifest_path or _MANIFEST_PATH)
    entry = entries.get(entity.bank or entity.canonical_name)
    return build_source_name(entry) if entry is not None else None


# GRI only, deliberately: powers /chat's compliance-check category guess, which has no
# framework concept of its own yet, see `api.compliance.FrameworkName`.
_CHECKLISTS_DIR = Path(__file__).resolve().parent / "checklists" / "gri"
_CHECKLIST_CATEGORIES: tuple[str, ...] = ("environmental", "social", "governance")


@cache
def _category_keywords(checklists_dir: Path) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Each checklist category's `topic_keywords`, lowercased and de-duplicated."""
    result: list[tuple[str, tuple[str, ...]]] = []
    for category in _CHECKLIST_CATEGORIES:
        path = checklists_dir / f"{category}.json"
        if not path.exists():
            result.append((category, ()))
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        keywords: list[str] = []
        for item in data.get("items", []):
            keywords.extend(str(keyword).lower() for keyword in item.get("topic_keywords", []))
        result.append((category, tuple(dict.fromkeys(keywords))))
    return tuple(result)


def resolve_category(query: str, *, checklists_dir: Path | None = None) -> str | None:
    """Infer which E/S/G checklist category a natural-language question is about (or None)."""
    checklists_dir = checklists_dir or _CHECKLISTS_DIR
    lowered = query.lower()
    best: tuple[int, str] | None = None
    for category, keywords in _category_keywords(checklists_dir):
        for keyword in keywords:
            if keyword in lowered and (best is None or len(keyword) > best[0]):
                best = (len(keyword), category)
    return best[1] if best else None


def resolve_company_source_name(
    reference: str,
    *,
    index: EntityIndex | None = None,
    manifest_path: Path | None = None,
) -> str:
    """Resolve an informal company `reference` to the exact stored `source_name`."""
    index = index or default_entity_index()
    manifest_path = manifest_path or _MANIFEST_PATH

    matches = index.match(reference)
    banks = [m for m in matches if m.entity.kind is EntityKind.BANK]

    if len(banks) > 1:
        named = ", ".join(m.entity.canonical_name for m in banks)
        raise AmbiguousCompanyError(
            f"{reference!r} names more than one bank ({named}); name a single company."
        )
    if not banks:
        if matches:
            named = ", ".join(m.entity.canonical_name for m in matches)
            raise NotACompanyError(
                f"{reference!r} resolves to {named}, which is not a company report."
            )
        raise CompanyNotFoundError(f"{reference!r} does not match any known bank.")

    entity = banks[0].entity
    source_name = source_name_for_entity(entity, manifest_path=manifest_path)
    if source_name is None:  # entity index and manifest disagree, a data/build inconsistency
        raise CompanyNotFoundError(
            f"{entity.canonical_name!r} has no manifest entry to build a source_name from."
        )
    return source_name
