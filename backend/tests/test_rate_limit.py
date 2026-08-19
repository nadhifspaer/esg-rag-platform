from __future__ import annotations

import time
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api.chat import get_chat_engine
from app.core.rate_limit import (
    CHAT_MAX_UNITS,
    RANK_METRIC_UNITS,
    Clock,
    InMemoryBudgetStore,
    RequestBudget,
    set_budget_store,
)
from app.main import app
from tests.conftest import STUB_USER

client = TestClient(app)

USER = "user-a"
OTHER = "user-b"


class FrozenClock:
    """A `Clock` that only moves when told to, so refill-timing tests stop racing wall time."""

    def __init__(self, start: float = 0.0) -> None:
        self._t = start

    def __call__(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:
        self._t += seconds


def _store(
    capacity: float = 10.0, per_hour: float = 3600.0, *, now: Clock = time.monotonic
) -> InMemoryBudgetStore:
    """A store with an explicit budget. Default refill is 1 unit/second for fast assertions."""
    return InMemoryBudgetStore(capacity=capacity, refill_per_second=per_hour / 3600.0, now=now)


# --- the bucket -------------------------------------------------------------


def test_spending_within_budget_is_allowed_and_reduces_the_balance() -> None:
    store = _store(capacity=10)
    decision = store.spend(USER, 3)
    assert decision.allowed and decision.balance == 7


def test_spending_beyond_the_budget_is_refused_with_a_wait() -> None:
    store = _store(capacity=10, per_hour=3600)  # 1 unit/second
    store.spend(USER, 10)
    decision = store.spend(USER, 5)
    assert not decision.allowed
    assert decision.retry_after_seconds == pytest.approx(5, abs=0.1)


def test_budgets_are_per_user() -> None:
    """One user exhausting their budget must not affect anyone else."""
    store = _store(capacity=10)
    store.spend(USER, 10)
    assert not store.spend(USER, 1).allowed
    assert store.spend(OTHER, 10).allowed


def test_a_refused_request_does_not_reset_the_refill_clock() -> None:
    """Otherwise a client could hold itself at zero forever by retrying in a tight loop."""
    store = _store(capacity=10)
    store.spend(USER, 10)
    first = store.spend(USER, 4)
    second = store.spend(USER, 4)
    assert second.retry_after_seconds <= first.retry_after_seconds


def test_refund_returns_units_and_never_exceeds_capacity() -> None:
    store = _store(capacity=10)
    store.spend(USER, 8)
    store.refund(USER, 8)
    assert store.balance(USER) == pytest.approx(10)
    store.refund(USER, 100)
    assert store.balance(USER) == pytest.approx(10)


def test_budget_refills_over_time() -> None:
    store = _store(capacity=10, per_hour=3600)
    store.spend(USER, 10)
    store._buckets[USER] = (0.0, store._buckets[USER][1] - 4)  # simulate 4 seconds passing
    assert store.balance(USER) == pytest.approx(4, abs=0.1)


# --- reserve / settle -------------------------------------------------------


def test_settling_below_the_reservation_refunds_the_difference() -> None:
    """A 1-attempt turn must not be charged the 3-attempt worst case (clock frozen, no race)."""
    clock = FrozenClock()
    store = _store(capacity=10, now=clock)
    store.spend(USER, CHAT_MAX_UNITS)
    RequestBudget(store, USER, CHAT_MAX_UNITS).settle(1)
    assert store.balance(USER) == pytest.approx(10 - 1)


def test_settling_at_zero_refunds_the_whole_reservation() -> None:
    """The /rank refusal case: reserve the worst case, charge nothing."""
    store = _store(capacity=30)
    store.spend(USER, RANK_METRIC_UNITS)
    RequestBudget(store, USER, RANK_METRIC_UNITS).settle(0)
    assert store.balance(USER) == pytest.approx(30)


def test_settling_twice_is_a_no_op() -> None:
    store = _store(capacity=30)
    store.spend(USER, 20)
    budget = RequestBudget(store, USER, 20)
    budget.settle(0)
    budget.settle(0)
    assert store.balance(USER) == pytest.approx(30)


def test_an_unsettled_reservation_stands() -> None:
    """Failing closed: a handler that crashes mid-way does not get a free request."""
    store = _store(capacity=30)
    store.spend(USER, 20)
    assert store.balance(USER) == pytest.approx(10)


# --- through the API --------------------------------------------------------


@pytest.fixture
def small_budget() -> Iterator[InMemoryBudgetStore]:
    """A budget just large enough for one /rank ranking, with no refill during the test."""
    store = _store(capacity=RANK_METRIC_UNITS, per_hour=0.0001)
    set_budget_store(store)
    yield store
    set_budget_store(None)


def _fake_rank_engine() -> SimpleNamespace:
    return SimpleNamespace(
        qdrant_client=object(),
        openai_client=object(),
        settings=SimpleNamespace(qdrant_collection="esg_documents", openai_api_key="k"),
    )


def test_a_rank_refusal_costs_nothing_and_can_be_repeated(
    small_budget: InMemoryBudgetStore,
) -> None:
    """The headline behaviour: refusals are free, so they never exhaust the budget."""
    app.dependency_overrides[get_chat_engine] = lambda: _fake_rank_engine()
    try:
        for _ in range(5):
            response = client.post("/rank", json={"query": "which banks comply with GRI 305-4"})
            assert response.status_code == 200
            assert response.json()["outcome"] == "refused"
        assert small_budget.balance(STUB_USER.id) == pytest.approx(RANK_METRIC_UNITS)
    finally:
        app.dependency_overrides.pop(get_chat_engine, None)


def test_an_exhausted_budget_returns_429_with_retry_after() -> None:
    """A clear, actionable error, never a silent failure; an engine lets the request reach it."""
    store = _store(capacity=RANK_METRIC_UNITS, per_hour=3600)
    store.spend(STUB_USER.id, RANK_METRIC_UNITS)  # drain it
    set_budget_store(store)
    app.dependency_overrides[get_chat_engine] = lambda: _fake_rank_engine()
    try:
        response = client.post("/chat", json={"query": "scope 1 emissions"})
        assert response.status_code == 429
        assert "Retry-After" in response.headers
        assert "budget" in response.json()["detail"].lower()
    finally:
        app.dependency_overrides.pop(get_chat_engine, None)
        set_budget_store(None)


def test_a_downed_engine_does_not_consume_budget() -> None:
    """Pins that ordering: a 503 leaves the caller's balance untouched."""
    store = _store(capacity=RANK_METRIC_UNITS)
    set_budget_store(store)
    try:
        assert client.post("/chat", json={"query": "scope 1"}).status_code == 503
        assert store.balance(STUB_USER.id) == pytest.approx(RANK_METRIC_UNITS)
    finally:
        set_budget_store(None)


def test_the_cost_table_prices_rank_an_order_of_magnitude_above_chat() -> None:
    """Pins the ratio the design rests on: a flat per-request cap would erase it."""
    assert RANK_METRIC_UNITS >= 10 * CHAT_MAX_UNITS / 3
    assert RANK_METRIC_UNITS > CHAT_MAX_UNITS
