"""Tests for RRF fusion and the hybrid_search orchestrator."""

from __future__ import annotations

import pytest
from qdrant_client import QdrantClient

from app.models.payload import ChunkPayload, ContentType, Domain
from app.rag.embedder import EMBEDDING_DIM
from app.rag.hybrid_search import FusedResult, hybrid_search, hybrid_search_with_legs, rrf_fuse
from app.rag.keyword_search import KeywordIndex
from app.rag.vector_store import SearchResult, ensure_collection, point_id, upsert_points

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect in the local Qdrant")

COLL = "test_esg_documents"


def _payload(source_name: str, text: str, *, domain: Domain = Domain.STANDARDS) -> ChunkPayload:
    return ChunkPayload(
        domain=domain,
        source_name=source_name,
        document_type="gri_standard",
        page_number=1,
        content_type=ContentType.TEXT,
        chunk_text=text,
    )


def _result(point_id_: str, score: float, source_name: str) -> SearchResult:
    return SearchResult(id=point_id_, score=score, payload=_payload(source_name, source_name))


# --- rrf_fuse ---------------------------------------------------------------


def test_rrf_rewards_documents_found_by_both_legs() -> None:
    # "both" is #2 in each leg; "dense_only" is #1 in dense; "kw_only" is #1 in kw.
    dense = [_result("dense_only", 0.9, "A"), _result("both", 0.8, "B")]
    keyword = [_result("kw_only", 12.0, "C"), _result("both", 5.0, "B")]

    fused = rrf_fuse(dense, keyword, limit=10)

    # "both" appears in two lists so its summed RRF score beats either single-leg #1.
    assert fused[0].id == "both"
    top = fused[0]
    assert top.dense_rank == 2
    assert top.keyword_rank == 2


def test_rrf_records_component_ranks_and_none_when_absent() -> None:
    dense = [_result("x", 0.5, "X")]
    keyword = [_result("y", 3.0, "Y")]

    fused = {f.id: f for f in rrf_fuse(dense, keyword, limit=10)}

    assert fused["x"].dense_rank == 1 and fused["x"].keyword_rank is None
    assert fused["y"].keyword_rank == 1 and fused["y"].dense_rank is None


def test_rrf_orders_by_score_and_respects_limit() -> None:
    dense = [_result(f"d{i}", 1.0 - i * 0.1, f"D{i}") for i in range(5)]
    keyword: list[SearchResult] = []

    fused = rrf_fuse(dense, keyword, limit=3)

    assert len(fused) == 3
    scores = [f.score for f in fused]
    assert scores == sorted(scores, reverse=True)
    assert [f.id for f in fused] == ["d0", "d1", "d2"]  # dense order preserved


def test_rrf_empty_inputs_return_empty() -> None:
    assert rrf_fuse([], [], limit=10) == []


# --- hybrid_search end to end ----------------------------------------------


class _FakeEmbedder:
    """Stands in for OpenAI: maps a query to a fixed vector, so the dense leg is deterministic."""

    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    class _Embeddings:
        def __init__(self, vector: list[float]) -> None:
            self._vector = vector

        def create(self, model: str, input: list[str]):  # noqa: A002 - matches OpenAI API
            from types import SimpleNamespace

            data = [SimpleNamespace(index=i, embedding=self._vector) for i, _ in enumerate(input)]
            return SimpleNamespace(data=data)

    @property
    def embeddings(self) -> _Embeddings:
        return self._Embeddings(self._vector)


def _dir_vector(dim_index: int) -> list[float]:
    v = [0.01] * EMBEDDING_DIM
    v[dim_index] = 1.0
    return v


def test_hybrid_promotes_exact_citation_over_a_semantic_distractor() -> None:
    client = QdrantClient(":memory:")
    ensure_collection(client, COLL)

    # The chunk that literally contains the standard number.
    exact = _payload(
        "GRI 305",
        "GRI 305: Emissions covers disclosure requirements for greenhouse gas reporting.",
    )
    # A distractor on the same topic but without the literal code, so keyword search misses it.
    distractor = _payload(
        "Topic match",
        "Guidance on greenhouse gas emissions disclosure and sustainability reporting practices.",
    )

    # Dense vectors align the distractor with the query, so semantic search ranks it #1.
    exact_id = point_id("GRI 305", ContentType.TEXT.value, 1, 0)
    distractor_id = point_id("Topic match", ContentType.TEXT.value, 1, 0)
    upsert_points(
        client,
        COLL,
        ids=[exact_id, distractor_id],
        vectors=[_dir_vector(1), _dir_vector(0)],
        payloads=[exact, distractor],
    )

    keyword_index = KeywordIndex([(exact_id, exact), (distractor_id, distractor)])
    embedder = _FakeEmbedder(_dir_vector(0))  # aligns with the distractor

    fused = hybrid_search(
        "GRI 305",
        domain=Domain.STANDARDS,
        qdrant_client=client,
        keyword_index=keyword_index,
        collection_name=COLL,
        embed_client=embedder,  # type: ignore[arg-type]
        limit=5,
    )

    assert isinstance(fused[0], FusedResult)
    # The exact chunk gets a strong keyword-leg signal the distractor cannot, lifting fusion.
    assert fused[0].payload.source_name == "GRI 305"
    assert fused[0].keyword_rank == 1
    # The distractor was found only by the dense leg, no keyword signal at all.
    by_id = {f.id: f for f in fused}
    assert by_id[distractor_id].keyword_rank is None


def test_hybrid_is_domain_scoped_across_both_legs() -> None:
    client = QdrantClient(":memory:")
    ensure_collection(client, COLL)

    std = _payload("Std doc", "emissions disclosure requirement", domain=Domain.STANDARDS)
    company = _payload(
        "Company doc", "emissions disclosure requirement", domain=Domain.COMPANY_DOCUMENTS
    )
    std_id = point_id("Std doc", ContentType.TEXT.value, 1, 0)
    company_id = point_id("Company doc", ContentType.TEXT.value, 1, 0)
    upsert_points(
        client,
        COLL,
        ids=[std_id, company_id],
        vectors=[_dir_vector(0), _dir_vector(0)],
        payloads=[std, company],
    )

    keyword_index = KeywordIndex([(std_id, std), (company_id, company)])
    embedder = _FakeEmbedder(_dir_vector(0))

    fused = hybrid_search(
        "emissions disclosure",
        domain=Domain.STANDARDS,
        qdrant_client=client,
        keyword_index=keyword_index,
        collection_name=COLL,
        embed_client=embedder,  # type: ignore[arg-type]
        limit=5,
    )

    # Only the standards doc may appear: neither leg may leak the company doc.
    assert fused, "expected at least one hit"
    assert all(f.payload.domain is Domain.STANDARDS for f in fused)
    assert all(f.payload.source_name != "Company doc" for f in fused)


# --- source_name scoping ----------------------------------------------------


def _three_report_corpus() -> tuple[QdrantClient, KeywordIndex]:
    """Three same-domain reports carrying near-identical text: this corpus in miniature."""
    client = QdrantClient(":memory:")
    ensure_collection(client, COLL)

    payloads = [
        _payload(
            name, "total direct emissions scope 1 greenhouse gas", domain=Domain.COMPANY_DOCUMENTS
        )
        for name in ("Report A", "Report B", "Report C")
    ]
    ids = [point_id(p.source_name, ContentType.TEXT.value, 1, 0) for p in payloads]
    upsert_points(client, COLL, ids=ids, vectors=[_dir_vector(0)] * 3, payloads=payloads)
    return client, KeywordIndex(list(zip(ids, payloads, strict=True)))


def test_hybrid_scoped_to_one_report_returns_only_that_report() -> None:
    """Both legs must honour the filter: if either ignored it, fusion would re-admit the
    other reports, which is exactly the bug this fixes."""
    client, keyword_index = _three_report_corpus()

    fused = hybrid_search(
        "scope 1 emissions",
        domain=Domain.COMPANY_DOCUMENTS,
        qdrant_client=client,
        keyword_index=keyword_index,
        source_name="Report B",
        collection_name=COLL,
        embed_client=_FakeEmbedder(_dir_vector(0)),  # type: ignore[arg-type]
        limit=10,
    )

    assert fused, "expected the scoped report's chunk"
    assert {f.payload.source_name for f in fused} == {"Report B"}


def test_hybrid_without_a_source_name_still_spans_the_domain() -> None:
    """The regression guard for every unscoped path: behaviour must be unchanged."""
    client, keyword_index = _three_report_corpus()

    fused = hybrid_search(
        "scope 1 emissions",
        domain=Domain.COMPANY_DOCUMENTS,
        qdrant_client=client,
        keyword_index=keyword_index,
        collection_name=COLL,
        embed_client=_FakeEmbedder(_dir_vector(0)),  # type: ignore[arg-type]
        limit=10,
    )

    assert {f.payload.source_name for f in fused} == {"Report A", "Report B", "Report C"}


# --- hybrid_search_with_legs -------------------------------------------------


def test_with_legs_fused_field_matches_hybrid_search_exactly() -> None:
    """Same work, same result: hybrid_search() is this function minus the raw legs."""
    client, keyword_index = _three_report_corpus()
    kwargs = dict(
        domain=Domain.COMPANY_DOCUMENTS,
        qdrant_client=client,
        keyword_index=keyword_index,
        collection_name=COLL,
        embed_client=_FakeEmbedder(_dir_vector(0)),  # type: ignore[arg-type]
        limit=10,
    )

    plain = hybrid_search("scope 1 emissions", **kwargs)  # type: ignore[arg-type]
    with_legs = hybrid_search_with_legs("scope 1 emissions", **kwargs)  # type: ignore[arg-type]

    assert [f.id for f in with_legs.fused] == [f.id for f in plain]


def test_with_legs_exposes_a_candidate_the_fused_limit_truncated_away() -> None:
    """The whole point: a chunk RRF-truncation drops can still be read off the raw legs."""
    client = QdrantClient(":memory:")
    ensure_collection(client, COLL)
    payloads = [
        _payload(
            name, "total direct emissions scope 1 greenhouse gas", domain=Domain.COMPANY_DOCUMENTS
        )
        for name in ("R1", "R2", "R3", "R4", "R5")
    ]
    ids = [point_id(p.source_name, ContentType.TEXT.value, 1, 0) for p in payloads]
    upsert_points(client, COLL, ids=ids, vectors=[_dir_vector(0)] * 5, payloads=payloads)
    keyword_index = KeywordIndex(list(zip(ids, payloads, strict=True)))

    result = hybrid_search_with_legs(
        "scope 1 emissions",
        domain=Domain.COMPANY_DOCUMENTS,
        qdrant_client=client,
        keyword_index=keyword_index,
        collection_name=COLL,
        embed_client=_FakeEmbedder(_dir_vector(0)),  # type: ignore[arg-type]
        dense_limit=20,
        keyword_limit=20,
        limit=1,
    )

    assert len(result.fused) == 1
    assert len(result.dense) == 5
    assert len(result.keyword) == 5
    # every id truncated out of the fused pool is still readable off the raw legs
    fused_ids = {f.id for f in result.fused}
    assert {r.id for r in result.dense} - fused_ids  # at least one survivor left over
