"""Tests for the batched LLM compliance-status assessor."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.models.payload import ChunkPayload, ContentType, Domain
from app.rag.compliance.checklist import ChecklistItem
from app.rag.compliance.llm_assessor import (
    JUDGE_SYSTEM_PROMPT,
    LLMAssessorError,
    PendingAssessment,
    assess_checklist_with_llm,
    llm_assess_batch,
)
from app.rag.compliance.report import Status
from app.rag.compliance.retrieval import EvidenceResult
from app.rag.vector_store import SearchResult


class _FakeJudgeClient:
    """Fake OpenAI client returning a preset JSON body and recording the call kwargs."""

    def __init__(self, reply: dict | str) -> None:
        self._reply = reply if isinstance(reply, str) else json.dumps(reply)
        self.create_calls: list[dict] = []
        outer = self

        class _Completions:
            def create(self, **kwargs):  # noqa: ANN003 - mirrors the OpenAI signature
                outer.create_calls.append(kwargs)
                message = SimpleNamespace(content=outer._reply)
                return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        self.chat = SimpleNamespace(completions=_Completions())


def _settings() -> Settings:
    return Settings(openai_api_key="test-key", openai_generation_model="gpt-4.1-mini")


def _item(
    code: str, desc: str = "some requirement", anchors: tuple[str, ...] = ()
) -> ChecklistItem:
    return ChecklistItem(code=code, description=desc, topic_keywords=["k"], row_anchors=anchors)


def _result(
    text: str, score: float = 0.6, content_type: ContentType = ContentType.TEXT
) -> SearchResult:
    payload = ChunkPayload(
        domain=Domain.COMPANY_DOCUMENTS,
        source_name="s",
        document_type="sustainability_report",
        page_number=1,
        content_type=content_type,
        chunk_text=text,
    )
    return SearchResult(id="c1", score=score, payload=payload)


def _evidence(results: list[SearchResult]) -> EvidenceResult:
    top = results[0].score if results else 0.0
    return EvidenceResult(
        item_code="x", query="q", results=results, top_score=top, fallback_used=False
    )


# --- llm_assess_batch: the judge itself --------------------------------------


def test_empty_pending_makes_no_call() -> None:
    client = _FakeJudgeClient({"results": []})
    assert llm_assess_batch([], client=client, settings=_settings()) == {}
    assert client.create_calls == []


def test_one_call_covers_every_pending_item() -> None:
    pending = [
        PendingAssessment(_item("A"), _evidence([_result("narrative A")])),
        PendingAssessment(_item("B"), _evidence([_result("narrative B")])),
        PendingAssessment(_item("C"), _evidence([_result("narrative C")])),
    ]
    client = _FakeJudgeClient(
        {
            "results": [
                {"code": "A", "status": "disclosed"},
                {"code": "B", "status": "partially disclosed"},
                {"code": "C", "status": "not found"},
            ]
        }
    )
    result = llm_assess_batch(pending, client=client, settings=_settings())
    assert result == {
        "A": Status.DISCLOSED,
        "B": Status.PARTIALLY_DISCLOSED,
        "C": Status.NOT_FOUND,
    }
    assert len(client.create_calls) == 1  # exactly one call, regardless of item count


def test_prompt_carries_every_requirement_and_its_evidence() -> None:
    pending = [
        PendingAssessment(_item("A", "desc A"), _evidence([_result("evidence text A")])),
        PendingAssessment(_item("B", "desc B"), _evidence([_result("evidence text B")])),
    ]
    client = _FakeJudgeClient(
        {"results": [{"code": "A", "status": "disclosed"}, {"code": "B", "status": "not found"}]}
    )
    llm_assess_batch(pending, client=client, settings=_settings())
    user = client.create_calls[0]["messages"][1]["content"]
    for text in ("A", "desc A", "evidence text A", "B", "desc B", "evidence text B"):
        assert text in user


def test_no_evidence_renders_as_no_evidence_retrieved() -> None:
    pending = [PendingAssessment(_item("A"), _evidence([]))]
    client = _FakeJudgeClient({"results": [{"code": "A", "status": "not found"}]})
    llm_assess_batch(pending, client=client, settings=_settings())
    user = client.create_calls[0]["messages"][1]["content"]
    assert "(no evidence retrieved)" in user


def test_bad_json_raises() -> None:
    client = _FakeJudgeClient("not json")
    pending = [PendingAssessment(_item("A"), _evidence([_result("t")]))]
    with pytest.raises(LLMAssessorError, match="valid JSON"):
        llm_assess_batch(pending, client=client, settings=_settings())


def test_missing_code_raises() -> None:
    client = _FakeJudgeClient({"results": [{"code": "A", "status": "disclosed"}]})
    pending = [
        PendingAssessment(_item("A"), _evidence([_result("t")])),
        PendingAssessment(_item("B"), _evidence([_result("t")])),
    ]
    with pytest.raises(LLMAssessorError, match="B"):
        llm_assess_batch(pending, client=client, settings=_settings())


def test_unknown_status_value_counts_as_missing() -> None:
    client = _FakeJudgeClient({"results": [{"code": "A", "status": "compliant"}]})
    pending = [PendingAssessment(_item("A"), _evidence([_result("t")]))]
    with pytest.raises(LLMAssessorError, match="omitted or mislabeled"):
        llm_assess_batch(pending, client=client, settings=_settings())


def test_missing_api_key_raises() -> None:
    pending = [PendingAssessment(_item("A"), _evidence([_result("t")]))]
    with pytest.raises(LLMAssessorError, match="OPENAI_API_KEY"):
        llm_assess_batch(pending, settings=Settings(openai_api_key=""))


def test_prompt_names_the_disclosure_index_table_distinction() -> None:
    """Pins the prompt rule this module exists for: the disclosure-index-table finding."""
    assert "disclosure-index table" in JUDGE_SYSTEM_PROMPT
    assert "is NEVER" in JUDGE_SYSTEM_PROMPT


# --- assess_checklist_with_llm: the batching orchestration -------------------


def test_exactly_one_llm_call_regardless_of_checklist_size() -> None:
    checklist = [_item("A"), _item("B"), _item("C"), _item("D")]
    evidence_by_code = {code: _evidence([_result(f"narrative {code}")]) for code in "ABCD"}
    calls: list[list[PendingAssessment]] = []

    def judge(pending):
        calls.append(list(pending))
        return {p.item.code: Status.DISCLOSED for p in pending}

    report = assess_checklist_with_llm(
        checklist,
        report_name="r",
        category="environmental",
        retrieve=lambda item: evidence_by_code[item.code],
        judge=judge,
    )

    assert len(calls) == 1
    assert len(calls[0]) == 4
    assert all(r.status is Status.DISCLOSED for r in report.results)


def test_anchored_items_never_reach_the_llm() -> None:
    """An anchored item is decided by `default_assess_status`'s anchor path, not judged."""
    anchored = _item("A", anchors=("E-01",))
    unanchored = _item("B")
    table_row = _result("| E-01 | GHG Emission Report | 1,234 |", content_type=ContentType.TABLE)
    evidence_by_code = {
        "A": _evidence([table_row]),
        "B": _evidence([_result("real narrative for B")]),
    }
    judge_calls = []

    def judge(pending):
        judge_calls.append(list(pending))
        return {p.item.code: Status.DISCLOSED for p in pending}

    assess_checklist_with_llm(
        [anchored, unanchored],
        report_name="r",
        category="environmental",
        retrieve=lambda item: evidence_by_code[item.code],
        judge=judge,
    )

    assert len(judge_calls) == 1
    assert [p.item.code for p in judge_calls[0]] == ["B"]  # only the unanchored item


def test_no_evidence_items_short_circuit_without_a_judgement() -> None:
    item = _item("A")
    judge_calls = []

    def judge(pending):
        judge_calls.append(list(pending))
        return {}

    report = assess_checklist_with_llm(
        [item],
        report_name="r",
        category="environmental",
        retrieve=lambda _item: _evidence([]),
        judge=judge,
    )

    assert judge_calls == []  # never called, nothing to judge
    assert report.results[0].status is Status.NOT_FOUND


def test_not_directly_disclosed_items_are_never_retrieved_or_judged() -> None:
    checklist = [
        ChecklistItem(code="A", description="d", topic_keywords=["k"], directly_disclosed=False)
    ]
    retrieve_calls = []
    judge_calls = []

    def retrieve(item):
        retrieve_calls.append(item.code)
        raise AssertionError("should never be called")

    def judge(pending):
        judge_calls.append(pending)
        return {}

    report = assess_checklist_with_llm(
        checklist, report_name="r", category="environmental", retrieve=retrieve, judge=judge
    )

    assert retrieve_calls == []
    assert judge_calls == []
    assert report.results[0].status is Status.NOT_DIRECTLY_DISCLOSED


def test_disclosure_index_table_only_evidence_labels_partial_not_disclosed() -> None:
    """End to end with a scripted judge reply: table-only evidence for an unanchored item."""
    item = _item("GRI-305-3", "sustainable financial products disclosure")
    index_row = _result(
        "E-01; GHG Emission Report; 75-82; S-01; Gender Equality; 65",
        content_type=ContentType.TABLE,
    )
    table_only = _evidence([index_row])

    def judge(pending):
        return {"GRI-305-3": Status.PARTIALLY_DISCLOSED}

    report = assess_checklist_with_llm(
        [item],
        report_name="r",
        category="social",
        retrieve=lambda _item: table_only,
        judge=judge,
    )
    assert report.results[0].status is Status.PARTIALLY_DISCLOSED
