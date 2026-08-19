"""Cross-bank analysis over extracted metric values: step three of the pipeline (6.5.5)."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

from app.rag.metrics.extraction import ExtractionStatus, MetricExtraction

# 6.5.4 marks a bank it skipped for lack of retrieval with this substring in `note`.
_RETRIEVAL_GAP_MARKER = "found=False"

# Leading numeric token: digits with dots/commas as separators.
_NUMBER_TOKEN = re.compile(r"^[+-]?[\d][\d.,]*")

# Trailing units seen in the real corpus. An unrecognised trailing token is refused, not
# stripped: silently discarding text after a number is how a wrong unit basis slips in.
_KNOWN_UNITS = frozenset(
    {
        "%",
        "hour/employee",
        "hours/employee",
        "hour",
        "hours",
        "jam",
        "jam/karyawan",
        "tco2e",
        "tco2e/rp",
        "ton",
        "kwh",
        "m3",
    }
)


class ExclusionReason(StrEnum):
    """Why a bank is not in the computed set. Every excluded bank carries one."""

    NOT_DISCLOSED_IN_REPORT = "not_disclosed_in_report"  # read, states no value
    NOT_DISCLOSED_RETRIEVAL_GAP = "not_disclosed_retrieval_gap"  # we found no evidence
    UNPARSEABLE = "unparseable"  # a value is stated but cannot be read as a number


@dataclass(frozen=True)
class ValueRow:
    """One (bank, value) pair eligible for ranking or arithmetic."""

    bank: str
    value: float
    raw_value: str
    label: str | None
    citation: str | None


@dataclass(frozen=True)
class AmbiguousBank:
    """A `multiple_disclosed` bank held out of the computation, with all its values kept."""

    bank: str
    values: tuple[tuple[str, str | None], ...]  # (raw value, scope label)
    citation: str | None


@dataclass(frozen=True)
class ExcludedBank:
    """A bank excluded from the computation, with the reason and any detail."""

    bank: str
    reason: ExclusionReason
    detail: str | None


@dataclass(frozen=True)
class RankingResult:
    """A ranking plus everything it could not rank: the excluded parts are the point."""

    ranked: tuple[ValueRow, ...]
    ambiguous: tuple[AmbiguousBank, ...]
    excluded: tuple[ExcludedBank, ...]
    bank_count: int
    exploded: bool


@dataclass(frozen=True)
class FilterResult:
    """Banks matching a predicate, plus the non-matching, ambiguous, and excluded ones."""

    matched: tuple[ValueRow, ...]
    unmatched: tuple[ValueRow, ...]
    ambiguous: tuple[AmbiguousBank, ...]
    excluded: tuple[ExcludedBank, ...]
    bank_count: int
    exploded: bool


@dataclass(frozen=True)
class Aggregate:
    """A corpus-wide sum or mean, carried together with its denominator."""

    op: str
    value: float | None
    n_included: int
    bank_count: int
    ambiguous: tuple[AmbiguousBank, ...]
    excluded: tuple[ExcludedBank, ...]
    exploded: bool


def parse_metric_number(raw: str) -> tuple[float | None, str | None]:
    """Parse a verbatim disclosed value into a float, or refuse with a reason to guess."""
    text = raw.strip()
    if not text:
        return None, "empty value"

    match = _NUMBER_TOKEN.match(text)
    if match is None:
        return None, f"no leading numeric token in {raw!r}"

    token = match.group(0)
    remainder = text[match.end() :].strip()
    if remainder and remainder.lower() not in _KNOWN_UNITS:
        return None, f"unrecognised trailing text {remainder!r} in {raw!r}"

    if token.count(",") > 1:
        return None, f"multiple commas in {raw!r}"

    if "," in token:
        # Comma is the decimal mark; any dots in the integer part are thousands separators.
        integer_part, _, decimal_part = token.partition(",")
        integer_digits = integer_part.replace(".", "")
        if not decimal_part.isdigit() or not integer_digits.lstrip("+-").isdigit():
            return None, f"malformed number {raw!r}"
        return float(f"{integer_digits}.{decimal_part}"), None

    dot_count = token.count(".")
    if dot_count == 0:
        return float(token), None
    if dot_count > 1:
        # Only a thousands-separated integer has several dots (2.500.000).
        return float(token.replace(".", "")), None

    _, _, after_dot = token.partition(".")
    if len(after_dot) == 3:
        # 5.813 -> 5813 (thousands) or 5.813 (decimal)? Both readings are live. Refuse.
        return None, (
            f"ambiguous separator in {raw!r}: a lone dot before exactly three digits is "
            "either a thousands separator or a decimal point in this corpus"
        )
    return float(token), None


def _exclusion_reason(extraction: MetricExtraction) -> ExclusionReason:
    """Distinguish a retrieval gap from a real absence, using 6.5.4's skip note."""
    if _RETRIEVAL_GAP_MARKER in (extraction.note or ""):
        return ExclusionReason.NOT_DISCLOSED_RETRIEVAL_GAP
    return ExclusionReason.NOT_DISCLOSED_IN_REPORT


def partition(
    extractions: Sequence[MetricExtraction], *, explode: bool = False
) -> tuple[list[ValueRow], list[AmbiguousBank], list[ExcludedBank]]:
    """Split extractions into usable rows, ambiguous banks, and excluded banks."""
    rows: list[ValueRow] = []
    ambiguous: list[AmbiguousBank] = []
    excluded: list[ExcludedBank] = []

    for extraction in extractions:
        if extraction.status is ExtractionStatus.NOT_DISCLOSED:
            excluded.append(
                ExcludedBank(
                    bank=extraction.bank,
                    reason=_exclusion_reason(extraction),
                    detail=extraction.note,
                )
            )
            continue

        if extraction.status is ExtractionStatus.MULTIPLE_DISCLOSED and not explode:
            ambiguous.append(
                AmbiguousBank(
                    bank=extraction.bank,
                    values=tuple((v.value, v.label) for v in extraction.values),
                    citation=extraction.citation,
                )
            )
            continue

        for value in extraction.values:
            number, reason = parse_metric_number(value.value)
            if number is None:
                excluded.append(
                    ExcludedBank(
                        bank=extraction.bank,
                        reason=ExclusionReason.UNPARSEABLE,
                        detail=reason,
                    )
                )
                continue
            rows.append(
                ValueRow(
                    bank=extraction.bank,
                    value=number,
                    raw_value=value.value,
                    label=value.label,
                    citation=extraction.citation,
                )
            )

    return rows, ambiguous, excluded


def rank_banks(
    extractions: Sequence[MetricExtraction],
    *,
    ascending: bool = True,
    top_n: int | None = None,
    explode: bool = False,
) -> RankingResult:
    """Rank banks by their disclosed value (ascending = smallest/earliest first)."""
    rows, ambiguous, excluded = partition(extractions, explode=explode)
    ordered = sorted(rows, key=lambda row: (row.value, row.bank), reverse=not ascending)
    return RankingResult(
        ranked=tuple(ordered if top_n is None else ordered[:top_n]),
        ambiguous=tuple(ambiguous),
        excluded=tuple(excluded),
        bank_count=len(extractions),
        exploded=explode,
    )


def filter_banks(
    extractions: Sequence[MetricExtraction],
    predicate: Callable[[float], bool],
    *,
    explode: bool = False,
) -> FilterResult:
    """Split banks by `predicate` over the parsed value (e.g. `lambda year: year < 2050`)."""
    rows, ambiguous, excluded = partition(extractions, explode=explode)
    matched = tuple(row for row in rows if predicate(row.value))
    unmatched = tuple(row for row in rows if not predicate(row.value))
    return FilterResult(
        matched=matched,
        unmatched=unmatched,
        ambiguous=tuple(ambiguous),
        excluded=tuple(excluded),
        bank_count=len(extractions),
        exploded=explode,
    )


def _aggregate(
    extractions: Sequence[MetricExtraction], op: str, *, explode: bool
) -> tuple[list[ValueRow], Aggregate]:
    """Shared partition + envelope for the arithmetic ops; `value` filled in by the caller."""
    rows, ambiguous, excluded = partition(extractions, explode=explode)
    return rows, Aggregate(
        op=op,
        value=None,
        n_included=len(rows),
        bank_count=len(extractions),
        ambiguous=tuple(ambiguous),
        excluded=tuple(excluded),
        exploded=explode,
    )


def total(extractions: Sequence[MetricExtraction], *, explode: bool = False) -> Aggregate:
    """Sum the disclosed values across banks (rejects value_type='year')."""
    if extractions and extractions[0].value_type == "year":
        raise ValueError("sum is not meaningful for value_type 'year'; use mean() or rank_banks()")
    rows, envelope = _aggregate(extractions, "sum", explode=explode)
    return replace(envelope, value=sum(row.value for row in rows) if rows else None)


def mean(extractions: Sequence[MetricExtraction], *, explode: bool = False) -> Aggregate:
    """Average the disclosed values across banks (unweighted)."""
    rows, envelope = _aggregate(extractions, "mean", explode=explode)
    return replace(envelope, value=(sum(r.value for r in rows) / len(rows)) if rows else None)
