"""Tests for the POST /rank endpoint (corpus-wide aggregation and ranking)."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app.api.rank as rank_api
from app.api.chat import get_chat_engine
from app.api.rank import _ascending, _wants_explanation, resolve_metric, resolve_requirement
from app.main import app
from app.rag.compliance.report import Status
from app.rag.compliance.sweep import BankStatus, RequirementSweep
from app.rag.metrics.analysis import AmbiguousBank as AnalysisAmbiguous
from app.rag.metrics.analysis import ExcludedBank as AnalysisExcluded
from app.rag.metrics.analysis import ExclusionReason, RankingResult, ValueRow
from app.rag.metrics.metric import load_metric_set

client = TestClient(app)


def _fake_engine(*, has_key: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        qdrant_client=object(),
        openai_client=object() if has_key else None,
        settings=SimpleNamespace(
            qdrant_collection="esg_documents", openai_api_key="k" if has_key else ""
        ),
    )


def _use_engine(engine: SimpleNamespace) -> None:
    app.dependency_overrides[get_chat_engine] = lambda: engine


@pytest.fixture(autouse=True)
def _clear_overrides() -> Iterator[None]:
    yield
    app.dependency_overrides.clear()


# --- target resolution ------------------------------------------------------


def test_resolves_a_metric_by_topic_keyword_and_by_code() -> None:
    assert resolve_metric("top 5 banks by earliest net-zero target year").code == "E-06"
    assert resolve_metric("rank banks by E-02").code == "E-02"
    assert resolve_metric("which banks have the most branches") is None


def test_resolves_a_requirement_code_across_all_categories() -> None:
    item, category = resolve_requirement("which banks are compliant with GRI 305-4")
    assert item.code == "GRI 305-4"
    assert category.value == "environmental"
    item, category = resolve_requirement("which banks meet GRI 412-1")
    assert item.code == "GRI 412-1"
    assert category.value == "social"
    assert resolve_requirement("which banks rank highest overall") is None


def test_sort_direction_covers_indonesian_and_english_superlatives() -> None:
    """`paling lama` used to match no marker at all and silently fall through to ascending."""
    assert _ascending("bank dengan target net-zero paling lama") is False
    assert _ascending("bank mana yang paling lama mencapai net zero") is False
    assert _ascending("bank dengan target terlama") is False
    assert _ascending("target paling akhir") is False
    assert _ascending("bank dengan target lebih lama") is False
    assert _ascending("which bank has the longest net-zero timeline") is False
    assert _ascending("which bank's target year is furthest out") is False
    # Unchanged existing behaviour: ascending default, and the earlier markers still descend.
    assert _ascending("top 5 banks by earliest net-zero target year") is True
    assert _ascending("which bank has the highest emissions intensity") is False
    assert _ascending("bank dengan emisi tertinggi") is False


# --- refusal paths: real classifier, real checklist, no retrieval -----------


def test_coded_requirement_is_refused_and_routed_to_metrics() -> None:
    _use_engine(_fake_engine())

    response = client.post("/rank", json={"query": "which banks are compliant with GRI 305-4"})

    assert response.status_code == 200
    body = response.json()
    assert body["query_type"] == "aggregation_ranking"
    assert body["target"] == "GRI 305-4"
    assert body["outcome"] == "refused"
    assert body["route_to"] == "metrics"
    assert body["reason"] == "coded_numeric_requirement"
    assert body["ranked"] == []


def test_not_directly_disclosed_requirement_is_refused_with_nothing_to_rank() -> None:
    """The one refusal reason that is never about calibration: GRI 305-3 has no row
    anywhere in this corpus to rank, calibrated engine or not."""
    _use_engine(_fake_engine())

    response = client.post("/rank", json={"query": "which banks are compliant with GRI 305-3"})

    body = response.json()
    assert body["outcome"] == "refused"
    assert body["reason"] == "not_directly_disclosed"
    assert body["route_to"] == "none"
    assert "nothing to rank" in body["message"]


# --- requirement path: calibrated sweep, ranked not refused -----------------


def test_unanchored_requirement_returns_a_real_ranking(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calibration fix: an unanchored requirement is no longer refused, a real comparison now."""
    view = RequirementSweep(
        code="GRI 401-1",
        description="New employee hires and employee turnover.",
        addressed=(
            BankStatus(
                bank="Bank A",
                status=Status.DISCLOSED,
                top_score=0.7,
                citation="c",
                fallback_used=False,
            ),
        ),
        partial=(
            BankStatus(
                bank="Bank B",
                status=Status.PARTIALLY_DISCLOSED,
                top_score=0.6,
                citation="c",
                fallback_used=False,
            ),
        ),
        not_found=(
            BankStatus(
                bank="Bank C",
                status=Status.NOT_FOUND,
                top_score=0.0,
                citation=None,
                fallback_used=False,
            ),
        ),
        not_checkable=(),
        bank_count=3,
    )
    monkeypatch.setattr(rank_api, "_rank_requirement", lambda *a, **k: view)
    _use_engine(_fake_engine())

    response = client.post("/rank", json={"query": "which banks are compliant with GRI 401-1"})

    body = response.json()
    assert body["outcome"] == "ranked"
    assert body["target"] == "GRI 401-1"
    assert body["target_kind"] == "requirement"
    assert body["bank_count"] == 3
    assert [b["bank"] for b in body["requirement_ranking"]["addressed"]] == ["Bank A"]
    assert [b["bank"] for b in body["requirement_ranking"]["partial"]] == ["Bank B"]
    assert [b["bank"] for b in body["requirement_ranking"]["not_found"]] == ["Bank C"]
    assert "content-reading judgement" in body["note"]


def test_non_ranking_query_is_rejected_with_a_pointer_to_the_right_endpoint() -> None:
    """Names a bank and no ranking wording, so rules classify it single_domain, not the LLM."""
    _use_engine(_fake_engine())

    response = client.post(
        "/rank", json={"query": "what does the BCA report say about scope 1 emissions"}
    )

    assert response.status_code == 422
    assert "/compliance-check" in response.json()["detail"]


def test_ranking_query_with_no_resolvable_target_is_refused() -> None:
    _use_engine(_fake_engine())

    response = client.post("/rank", json={"query": "which banks are the greenest overall"})

    body = response.json()
    assert body["outcome"] == "refused"
    assert body["reason"] == "unresolved_target"
    assert body["target_kind"] == "unresolved"


def test_missing_api_key_is_a_503() -> None:
    _use_engine(_fake_engine(has_key=False))

    response = client.post("/rank", json={"query": "top 5 banks by net-zero target year"})

    assert response.status_code == 503


# --- metric path ------------------------------------------------------------


def test_metric_ranking_keeps_ambiguous_and_excluded_banks_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ranking must never quietly shrink the corpus: held-out banks are in the response."""
    ranked = RankingResult(
        ranked=(
            ValueRow(bank="Bank A", value=2040.0, raw_value="2040", label=None, citation="c"),
            ValueRow(bank="Bank B", value=2050.0, raw_value="2050", label=None, citation="c"),
        ),
        ambiguous=(
            AnalysisAmbiguous(
                bank="BNI", values=(("2028", "Operational"), ("2060", "Financed")), citation="c"
            ),
        ),
        excluded=(
            AnalysisExcluded(
                bank="Bank C", reason=ExclusionReason.NOT_DISCLOSED_IN_REPORT, detail="no target"
            ),
        ),
        bank_count=4,
        exploded=False,
    )
    monkeypatch.setattr(rank_api, "_rank_metric", lambda *a, **k: (ranked, 4))
    _use_engine(_fake_engine())

    response = client.post(
        "/rank", json={"query": "top 5 banks by earliest net-zero target year", "top_n": 5}
    )

    body = response.json()
    assert body["outcome"] == "ranked"
    assert body["target"] == "E-06"
    assert body["target_kind"] == "metric"
    assert [row["bank"] for row in body["ranked"]] == ["Bank A", "Bank B"]
    assert body["ambiguous"][0]["bank"] == "BNI"
    assert body["ambiguous"][0]["values"] == ["2028 (Operational)", "2060 (Financed)"]
    assert body["excluded"][0]["reason"] == "not_disclosed_in_report"
    assert body["bank_count"] == 4
    # The ambiguous-bank sentence is appended to `note`, built from BNI's real scoped values
    # -- not a generic "also disclosed <metric>" clause, and not hardcoded to this one query.
    assert (
        "BNI discloses 2028 (Operational) and 2060 (Financed). Because it names multiple "
        "scoped values rather than one, it sits outside the strict ranking." in body["note"]
    )


def test_no_ambiguous_sentence_when_ambiguous_list_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sentence is only appended when there is something to name."""
    ranked = RankingResult(
        ranked=(ValueRow(bank="Bank A", value=2040.0, raw_value="2040", label=None, citation="c"),),
        ambiguous=(),
        excluded=(),
        bank_count=1,
        exploded=False,
    )
    monkeypatch.setattr(rank_api, "_rank_metric", lambda *a, **k: (ranked, 1))
    _use_engine(_fake_engine())

    response = client.post("/rank", json={"query": "top 5 banks by earliest net-zero target year"})

    assert "discloses" not in response.json()["note"]


# --- _ambiguous_sentence: per-bank scoped values, verbatim, no cap at two -------------------


def test_ambiguous_sentence_singular_grammar_for_one_bank() -> None:
    ambiguous = (
        AnalysisAmbiguous(
            bank="BNI", values=(("2028", "Operational"), ("2060", "Financed")), citation="c"
        ),
    )

    sentence = rank_api._ambiguous_sentence(ambiguous)

    assert sentence == (
        "BNI discloses 2028 (Operational) and 2060 (Financed). Because it names multiple "
        "scoped values rather than one, it sits outside the strict ranking."
    )


def test_ambiguous_sentence_lists_every_bank_and_every_scoped_value() -> None:
    """Real shape: several ambiguous banks, each listing its values; unlabelled is a bare value."""
    ambiguous = (
        AnalysisAmbiguous(
            bank="BNI", values=(("2028", "Operational"), ("2060", "Financed")), citation="c"
        ),
        AnalysisAmbiguous(
            bank="Mandiri",
            values=(("2030", "Operations"), ("2050", "Scope 1+2"), ("2060", "Financing")),
            citation="c",
        ),
        AnalysisAmbiguous(
            bank="Bank X", values=(("2035", None), ("2055", "Financed")), citation="c"
        ),
    )

    sentence = rank_api._ambiguous_sentence(ambiguous)

    assert sentence == (
        "BNI discloses 2028 (Operational) and 2060 (Financed). "
        "Mandiri discloses 2030 (Operations), 2050 (Scope 1+2), and 2060 (Financing). "
        "Bank X discloses 2035 and 2055 (Financed). "
        "Because each names multiple scoped values rather than one, they sit outside the "
        "strict ranking."
    )


def test_ambiguous_sentence_lands_on_note_field_specifically() -> None:
    """Pins the *field*, not just the text: a future refactor that moves this sentence into
    `message`/`reason`/`route_to` (or drops it from `note`) fails here, not just visually."""
    metric = next(m for m in load_metric_set() if m.code == "E-06")
    result = RankingResult(
        ranked=(),
        ambiguous=(
            AnalysisAmbiguous(
                bank="BNI", values=(("2028", "Operational"), ("2060", "Financed")), citation="c"
            ),
        ),
        excluded=(),
        bank_count=1,
        exploded=False,
    )

    response = rank_api._metric_response("q", metric, result, 1)

    assert "BNI discloses 2028 (Operational) and 2060 (Financed)." in response.note
    assert response.message is None
    assert response.reason is None
    assert response.route_to is None


# --- explanation-intent detection and the bank-follow-up invite sentence -------------------


def test_wants_explanation_covers_indonesian_and_english_markers() -> None:
    assert _wants_explanation("Bank apa yang memiliki target net zero paling lama, jelaskan kenapa")
    assert _wants_explanation("explain why banks differ")
    assert _wants_explanation("why is BCA the highest")
    assert _wants_explanation("mengapa BCA berbeda")
    assert not _wants_explanation("top 5 banks by earliest net-zero target year")
    assert not _wants_explanation("which bank has the most branches")


def test_explanation_invite_does_not_change_ranking_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole point of gating this on `_wants_explanation` in `note` construction only:
    the exact same ranked rows, in the exact same order, whether or not the query asks why."""
    ranked = RankingResult(
        ranked=(
            ValueRow(bank="Bank A", value=2040.0, raw_value="2040", label=None, citation="c"),
            ValueRow(bank="Bank B", value=2050.0, raw_value="2050", label=None, citation="c"),
        ),
        ambiguous=(),
        excluded=(),
        bank_count=2,
        exploded=False,
    )
    monkeypatch.setattr(rank_api, "_rank_metric", lambda *a, **k: (ranked, 2))
    _use_engine(_fake_engine())

    plain = client.post(
        "/rank", json={"query": "top 5 banks by earliest net-zero target year"}
    ).json()
    asking_why = client.post(
        "/rank", json={"query": "top 5 banks by earliest net-zero target year, explain why"}
    ).json()

    assert [row["bank"] for row in plain["ranked"]] == [row["bank"] for row in asking_why["ranked"]]
    assert plain["ranked"] == asking_why["ranked"]


def test_no_explanation_sentence_when_query_does_not_ask_why(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ranked = RankingResult(
        ranked=(ValueRow(bank="Bank A", value=2040.0, raw_value="2040", label=None, citation="c"),),
        ambiguous=(),
        excluded=(),
        bank_count=1,
        exploded=False,
    )
    monkeypatch.setattr(rank_api, "_rank_metric", lambda *a, **k: (ranked, 1))
    _use_engine(_fake_engine())

    response = client.post("/rank", json={"query": "top 5 banks by earliest net-zero target year"})

    assert "ask about" not in response.json()["note"]


def test_explanation_sentence_names_the_single_leading_bank_when_there_is_no_tie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ranked = RankingResult(
        ranked=(
            ValueRow(bank="Bank X", value=2040.0, raw_value="2040", label=None, citation="c"),
            ValueRow(bank="Bank Y", value=2050.0, raw_value="2050", label=None, citation="c"),
        ),
        ambiguous=(),
        excluded=(),
        bank_count=2,
        exploded=False,
    )
    monkeypatch.setattr(rank_api, "_rank_metric", lambda *a, **k: (ranked, 2))
    _use_engine(_fake_engine())

    response = client.post(
        "/rank", json={"query": "which bank has the earliest net-zero target, explain why"}
    )

    assert response.json()["note"].endswith(
        'Bank X has the leading disclosed value (2040) - ask about it directly (e.g. "why '
        "does Bank X's report say 2040\") to see what its own report says."
    )


def test_explanation_sentence_states_the_tie_count_for_two_tied_banks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ranked = RankingResult(
        ranked=(
            ValueRow(bank="Bank A", value=2060.0, raw_value="2060", label=None, citation="c"),
            ValueRow(bank="Bank B", value=2060.0, raw_value="2060", label=None, citation="c"),
            ValueRow(bank="Bank C", value=2050.0, raw_value="2050", label=None, citation="c"),
        ),
        ambiguous=(),
        excluded=(),
        bank_count=3,
        exploded=False,
    )
    monkeypatch.setattr(rank_api, "_rank_metric", lambda *a, **k: (ranked, 3))
    _use_engine(_fake_engine())

    response = client.post(
        "/rank", json={"query": "which bank has the longest net-zero target, explain why"}
    )

    assert response.json()["note"].endswith(
        "2 banks share the leading disclosed value (2060) - ask about a specific bank (e.g. "
        '"why does Bank A\'s report say 2060") to see what its own report says.'
    )


def test_explanation_sentence_tie_count_is_not_hardcoded_to_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real corpus case: 6 banks actually do share E-06's leading value (2060). The count
    must come from the data, not a hardcoded pair."""
    ranked = RankingResult(
        ranked=(
            ValueRow(bank="Bank A", value=2060.0, raw_value="2060", label=None, citation="c"),
            ValueRow(bank="Bank B", value=2060.0, raw_value="2060", label=None, citation="c"),
            ValueRow(bank="Bank C", value=2060.0, raw_value="2060", label=None, citation="c"),
            ValueRow(bank="Bank D", value=2060.0, raw_value="2060", label=None, citation="c"),
            ValueRow(bank="Bank E", value=2060.0, raw_value="2060", label=None, citation="c"),
            ValueRow(bank="Bank F", value=2060.0, raw_value="2060", label=None, citation="c"),
            ValueRow(bank="Bank G", value=2050.0, raw_value="2050", label=None, citation="c"),
        ),
        ambiguous=(),
        excluded=(),
        bank_count=7,
        exploded=False,
    )
    monkeypatch.setattr(rank_api, "_rank_metric", lambda *a, **k: (ranked, 7))
    _use_engine(_fake_engine())

    response = client.post(
        "/rank", json={"query": "which bank has the longest net-zero target, explain why"}
    )

    assert "6 banks share the leading disclosed value (2060)" in response.json()["note"]


def test_explanation_invite_sentence_is_absent_when_ranked_list_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No leading value exists to describe when the whole corpus is ambiguous/excluded."""
    ranked = RankingResult(
        ranked=(),
        ambiguous=(
            AnalysisAmbiguous(
                bank="BNI", values=(("2028", "Operational"), ("2060", "Financed")), citation="c"
            ),
        ),
        excluded=(),
        bank_count=1,
        exploded=False,
    )
    monkeypatch.setattr(rank_api, "_rank_metric", lambda *a, **k: (ranked, 1))
    _use_engine(_fake_engine())

    response = client.post(
        "/rank", json={"query": "top 5 banks by earliest net-zero target year, explain why"}
    )

    assert "leading disclosed value" not in response.json()["note"]
