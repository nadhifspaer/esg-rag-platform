"""Tests for the bounded agentic retry loop."""

from __future__ import annotations

import logging

import pytest

from app.models.payload import ChunkPayload, ContentType, Domain
from app.rag.generation.loop import StopReason, answer_with_retries
from app.rag.generation.self_check import GroundednessDecision, Verdict
from app.rag.vector_store import SearchResult


def _chunk(source: str) -> SearchResult:
    payload = ChunkPayload(
        domain=Domain.COMPANY_DOCUMENTS,
        source_name=source,
        document_type="sustainability_report",
        page_number=1,
        content_type=ContentType.TEXT,
        chunk_text=f"text from {source}",
    )
    return SearchResult(id=source, score=1.0, payload=payload)


class _Retriever:
    """Records the query of each call; returns a preset chunk list per call index."""

    def __init__(self, per_call: list[list[SearchResult]]) -> None:
        self.queries: list[str] = []
        self._per_call = per_call

    def __call__(self, query: str) -> list[SearchResult]:
        self.queries.append(query)
        i = len(self.queries) - 1
        return self._per_call[i] if i < len(self._per_call) else self._per_call[-1]


class _Generator:
    """Records (question, chunks) per call; returns a preset answer per call index."""

    def __init__(self, answers: list[str]) -> None:
        self.calls: list[tuple[str, list]] = []
        self._answers = answers

    def __call__(self, question: str, chunks) -> str:
        self.calls.append((question, list(chunks)))
        i = len(self.calls) - 1
        return self._answers[i] if i < len(self._answers) else self._answers[-1]


class _Checker:
    """Records calls; returns preset decisions in order (last one repeats)."""

    def __init__(self, decisions: list[GroundednessDecision]) -> None:
        self.calls: list[tuple[str, str]] = []
        self._decisions = decisions

    def __call__(self, question: str, answer: str, chunks) -> GroundednessDecision:
        self.calls.append((question, answer))
        i = len(self.calls) - 1
        return self._decisions[i] if i < len(self._decisions) else self._decisions[-1]


_SUFFICIENT = GroundednessDecision(verdict=Verdict.SUFFICIENT, rewritten_query=None)


def _insufficient(rewrite: str | None) -> GroundednessDecision:
    return GroundednessDecision(verdict=Verdict.INSUFFICIENT, rewritten_query=rewrite)


def test_stops_on_first_sufficient_no_retry() -> None:
    retrieve = _Retriever([[_chunk("BCA")]])
    generate = _Generator(["answer 1"])
    check = _Checker([_SUFFICIENT])

    result = answer_with_retries("q", retrieve=retrieve, generate=generate, check=check)

    assert result.stop_reason is StopReason.SUFFICIENT
    assert result.attempts == 1 and result.retries == 0
    assert result.answer == "answer 1"
    assert retrieve.queries == ["q"]
    assert len(generate.calls) == 1 and len(check.calls) == 1


def test_retries_once_then_succeeds() -> None:
    retrieve = _Retriever([[_chunk("A")], [_chunk("B")]])
    generate = _Generator(["weak answer", "good answer"])
    check = _Checker([_insufficient("better query"), _SUFFICIENT])

    result = answer_with_retries("original q", retrieve=retrieve, generate=generate, check=check)

    assert result.stop_reason is StopReason.SUFFICIENT
    assert result.attempts == 2 and result.retries == 1
    assert result.answer == "good answer"
    # First retrieval used the original query; the retry used the rewritten one.
    assert retrieve.queries == ["original q", "better query"]
    assert result.retrieval_query == "better query"


def test_hard_cap_stops_at_two_retries() -> None:
    retrieve = _Retriever([[_chunk("A")], [_chunk("B")], [_chunk("C")]])
    generate = _Generator(["a1", "a2", "a3"])
    # Always insufficient: the cap, not the judge, must stop the loop.
    check = _Checker([_insufficient("q1"), _insufficient("q2"), _insufficient("q3")])

    result = answer_with_retries("q", retrieve=retrieve, generate=generate, check=check)

    assert result.stop_reason is StopReason.RETRY_CAP
    assert result.attempts == 3 and result.retries == 2
    assert result.answer == "a3"  # best available = last attempt
    # Exactly 3 attempts; the 3rd rewrite ("q3") is never used for retrieval.
    assert retrieve.queries == ["q", "q1", "q2"]
    assert len(generate.calls) == 3 and len(check.calls) == 3


def test_rewritten_query_steers_retrieval_not_generation() -> None:
    retrieve = _Retriever([[_chunk("A")], [_chunk("B")]])
    generate = _Generator(["a1", "a2"])
    check = _Checker([_insufficient("rewritten"), _SUFFICIENT])

    answer_with_retries("the user question", retrieve=retrieve, generate=generate, check=check)

    # Generation always answers the ORIGINAL question, both attempts.
    assert [q for q, _ in generate.calls] == ["the user question", "the user question"]
    # Retrieval switched to the rewritten query on the retry.
    assert retrieve.queries == ["the user question", "rewritten"]


def test_citations_come_from_the_final_chunks() -> None:
    retrieve = _Retriever([[_chunk("first doc")], [_chunk("final doc")]])
    generate = _Generator(["a1", "a2"])
    check = _Checker([_insufficient("q2"), _SUFFICIENT])

    result = answer_with_retries("q", retrieve=retrieve, generate=generate, check=check)

    assert len(result.citations) == 1
    assert result.citations[0].source_name == "final doc"


def test_empty_rewrite_falls_back_to_original_query() -> None:
    retrieve = _Retriever([[_chunk("A")], [_chunk("B")]])
    generate = _Generator(["a1", "a2"])
    check = _Checker([_insufficient(None), _SUFFICIENT])

    answer_with_retries("original", retrieve=retrieve, generate=generate, check=check)

    assert retrieve.queries == ["original", "original"]  # None rewrite -> reuse original


def test_max_retries_zero_means_single_attempt() -> None:
    retrieve = _Retriever([[_chunk("A")]])
    generate = _Generator(["only answer"])
    check = _Checker([_insufficient("would-be retry")])

    result = answer_with_retries(
        "q", retrieve=retrieve, generate=generate, check=check, max_retries=0
    )

    assert result.stop_reason is StopReason.RETRY_CAP
    assert result.attempts == 1 and result.retries == 0
    assert retrieve.queries == ["q"]  # no retry despite insufficient


def test_iteration_count_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    retrieve = _Retriever([[_chunk("A")], [_chunk("B")]])
    generate = _Generator(["a1", "a2"])
    check = _Checker([_insufficient("q2"), _SUFFICIENT])

    with caplog.at_level(logging.INFO, logger="app.rag.generation.loop"):
        answer_with_retries("q", retrieve=retrieve, generate=generate, check=check)

    assert "attempts=2 retries=1 stop_reason=sufficient" in caplog.text


def test_retrieval_seed_steers_only_the_first_search() -> None:
    """The multi-turn seed and the judge's rewrite share a slot: the seed governs attempt 1 only."""
    retrieve = _Retriever([[_chunk("BCA")], [_chunk("Mandiri")]])
    generate = _Generator(["answer 1", "answer 2"])
    check = _Checker([_insufficient("mandiri emissions intensity"), _SUFFICIENT])

    result = answer_with_retries(
        "what about Mandiri?",
        retrieve=retrieve,
        generate=generate,
        check=check,
        retrieval_seed="mandiri Scope 1 emissions",
    )

    assert retrieve.queries == ["mandiri Scope 1 emissions", "mandiri emissions intensity"]
    # Both the answer and the judge always target the user's actual question.
    assert [q for q, _ in generate.calls] == ["what about Mandiri?", "what about Mandiri?"]
    assert [q for q, _ in check.calls] == ["what about Mandiri?", "what about Mandiri?"]
    assert result.retrieval_query == "mandiri emissions intensity"


def test_no_retrieval_seed_reproduces_single_turn_behaviour() -> None:
    """The default must be indistinguishable from before the parameter existed."""
    retrieve = _Retriever([[_chunk("BCA")]])

    answer_with_retries(
        "q", retrieve=retrieve, generate=_Generator(["a"]), check=_Checker([_SUFFICIENT])
    )
    assert retrieve.queries == ["q"]

    explicit_none = _Retriever([[_chunk("BCA")]])
    answer_with_retries(
        "q",
        retrieve=explicit_none,
        generate=_Generator(["a"]),
        check=_Checker([_SUFFICIENT]),
        retrieval_seed=None,
    )
    assert explicit_none.queries == ["q"]


def test_negative_max_retries_rejected() -> None:
    with pytest.raises(ValueError, match="max_retries"):
        answer_with_retries(
            "q",
            retrieve=_Retriever([[_chunk("A")]]),
            generate=_Generator(["a"]),
            check=_Checker([_SUFFICIENT]),
            max_retries=-1,
        )
