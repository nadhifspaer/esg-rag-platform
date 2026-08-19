"""Tests for the BM25 keyword-search leg."""

from __future__ import annotations

from app.models.payload import ChunkPayload, ContentType, Domain
from app.rag.keyword_search import KeywordIndex, tokenize


def _payload(
    source_name: str,
    text: str,
    *,
    domain: Domain = Domain.STANDARDS,
    page: int = 1,
) -> ChunkPayload:
    return ChunkPayload(
        domain=domain,
        source_name=source_name,
        document_type="gri_standard",
        page_number=page,
        content_type=ContentType.TEXT,
        chunk_text=text,
    )


def _index(*docs: tuple[str, ChunkPayload]) -> KeywordIndex:
    return KeywordIndex(docs)


# --- tokenizer --------------------------------------------------------------


def test_tokenize_splits_standard_code_into_matchable_tokens() -> None:
    # Punctuation is a separator, so the rare identifiers become their own tokens.
    assert tokenize("GRI 305-1/EDGB E-04") == ["gri", "305", "1", "edgb", "e", "04"]


def test_tokenize_lowercases_and_drops_punctuation() -> None:
    assert tokenize("GRI 305-1, Scope 1!") == ["gri", "305", "1", "scope", "1"]


def test_tokenize_empty_string_is_empty() -> None:
    assert tokenize("   ") == []


# --- the exact-citation scenario (the reason this leg exists) ----------------


def test_exact_standard_number_outranks_a_different_standard() -> None:
    idx = _index(
        (
            "g305",
            _payload(
                "GRI 305",
                "Global Reporting Initiative Standard GRI 305 on Emissions covers "
                "disclosure requirements for greenhouse gas reporting by financial "
                "institutions.",
            ),
        ),
        (
            "g306",
            _payload(
                "GRI 306",
                "Global Reporting Initiative Standard GRI 306 on Waste covers "
                "disclosure requirements for waste management reporting by financial "
                "institutions.",
            ),
        ),
    )

    results = idx.search("GRI 305", Domain.STANDARDS, limit=5)

    assert results, "expected a keyword hit"
    # The chunk containing the exact number '305' wins; '305' is rare (high IDF) and
    # absent from the GRI 306 chunk, which only shares the common 'gri'/'standard' tokens.
    assert results[0].payload.source_name == "GRI 305"


def test_search_returns_only_matching_docs() -> None:
    idx = _index(
        ("a", _payload("Has term", "emissions scope 1 and scope 2 disclosure")),
        ("b", _payload("No term", "board governance and anti-corruption policy")),
    )

    results = idx.search("emissions", Domain.STANDARDS, limit=5)

    assert [r.payload.source_name for r in results] == ["Has term"]


def test_rarer_query_term_drives_ranking() -> None:
    # 'sustainability' appears in every doc (low IDF); only one mentions 'taksonomi'
    # (rare, high IDF). A query for both must surface the taksonomi doc first.
    idx = _index(
        ("a", _payload("Common only", "sustainability sustainability report")),
        ("b", _payload("Rare term", "sustainability taksonomi hijau Indonesia")),
        ("c", _payload("Common two", "sustainability finance roadmap")),
    )

    results = idx.search("sustainability taksonomi", Domain.STANDARDS, limit=5)

    assert results[0].payload.source_name == "Rare term"


# --- domain scoping ---------------------------------------------------------


def test_search_is_domain_scoped() -> None:
    idx = _index(
        ("std", _payload("Std doc", "emissions disclosure", domain=Domain.STANDARDS)),
        (
            "co",
            _payload("Company doc", "emissions disclosure", domain=Domain.COMPANY_DOCUMENTS),
        ),
    )

    std_hits = idx.search("emissions", Domain.STANDARDS, limit=5)
    company_hits = idx.search("emissions", Domain.COMPANY_DOCUMENTS, limit=5)

    assert [r.payload.source_name for r in std_hits] == ["Std doc"]
    assert [r.payload.source_name for r in company_hits] == ["Company doc"]


def test_search_unknown_domain_returns_empty() -> None:
    idx = _index(("co", _payload("Company doc", "emissions", domain=Domain.COMPANY_DOCUMENTS)))
    # No standards docs were indexed.
    assert idx.search("emissions", Domain.STANDARDS, limit=5) == []


# --- edges ------------------------------------------------------------------


def test_search_respects_limit() -> None:
    docs = [(f"d{i}", _payload(f"Doc {i}", "emissions carbon disclosure")) for i in range(5)]
    idx = KeywordIndex(docs)
    assert len(idx.search("emissions", Domain.STANDARDS, limit=2)) == 2


def test_query_with_no_corpus_overlap_returns_empty() -> None:
    idx = _index(("a", _payload("Doc", "emissions disclosure")))
    assert idx.search("xyzzy", Domain.STANDARDS, limit=5) == []


def test_empty_index_returns_empty() -> None:
    assert KeywordIndex([]).search("anything", Domain.STANDARDS, limit=5) == []


# --- source_name scoping ----------------------------------------------------


def test_search_scoped_to_one_source_excludes_every_other_document() -> None:
    idx = _index(
        ("a", _payload("Report A", "total direct emissions scope 1")),
        ("b", _payload("Report B", "total direct emissions scope 1")),
        ("c", _payload("Report C", "total direct emissions scope 1")),
    )
    hits = idx.search("scope 1 emissions", Domain.STANDARDS, source_name="Report B", limit=5)
    assert [h.payload.source_name for h in hits] == ["Report B"]


def test_scoping_filters_before_the_limit_is_applied() -> None:
    """The scoped chunk survives scoring below `limit` others (filtered before truncation)."""
    docs = [(f"d{i}", _payload(f"Other {i}", "emissions emissions emissions")) for i in range(10)]
    docs.append(("wanted", _payload("Wanted report", "emissions mentioned once")))
    idx = KeywordIndex(docs)

    unscoped = idx.search("emissions", Domain.STANDARDS, limit=3)
    assert "Wanted report" not in [h.payload.source_name for h in unscoped]

    scoped = idx.search("emissions", Domain.STANDARDS, source_name="Wanted report", limit=3)
    assert [h.payload.source_name for h in scoped] == ["Wanted report"]


def test_unscoped_search_is_unchanged() -> None:
    idx = _index(
        ("a", _payload("Report A", "emissions disclosure")),
        ("b", _payload("Report B", "emissions disclosure")),
    )
    assert len(idx.search("emissions", Domain.STANDARDS, limit=5)) == 2


def test_scoping_to_an_unknown_source_returns_empty() -> None:
    idx = _index(("a", _payload("Report A", "emissions disclosure")))
    assert idx.search("emissions", Domain.STANDARDS, source_name="Nope", limit=5) == []
