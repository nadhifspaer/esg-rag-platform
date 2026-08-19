from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from fastapi import Depends, HTTPException, status

from app.core.auth import AuthenticatedUser, require_user
from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Defaulted factory so callers never need to touch it; tests inject a deterministic one.
Clock = Callable[[], float]


# --- what each pipeline costs -- one unit ~= one chat attempt, everything else relative ---

CHAT_ATTEMPT_UNITS = 1
CHAT_MAX_UNITS = 3  # the loop's hard cap is 3 attempts, so this is the true worst case
COMPLIANCE_SWEEP_UNITS = 3
RANK_METRIC_UNITS = 20
RANK_REFUSED_UNITS = 0


@dataclass(frozen=True)
class Decision:
    """The outcome of asking to spend units: allowed, plus the wait if not."""

    allowed: bool
    balance: float
    retry_after_seconds: float


class BudgetStore(Protocol):
    """Per-user unit budget. Implementations must be safe for concurrent requests."""

    def spend(self, user_id: str, units: float) -> Decision:
        """Deduct `units` if the balance allows, otherwise refuse and say how long to wait."""

    def refund(self, user_id: str, units: float) -> None:
        """Return unspent units (a reservation settled below its worst case)."""

    def balance(self, user_id: str) -> float:
        """Current balance, for diagnostics and response headers."""


class InMemoryBudgetStore:
    """Token-bucket budgets held in this process; not durable, not shared across workers."""

    def __init__(
        self,
        capacity: float,
        refill_per_second: float,
        *,
        now: Clock = time.monotonic,
    ) -> None:
        self._capacity = capacity
        self._refill_per_second = refill_per_second
        self._now = now
        self._buckets: dict[str, tuple[float, float]] = {}  # user -> (tokens, last_touched)
        self._lock = threading.Lock()

    def _current(self, user_id: str, now: float) -> float:
        tokens, last = self._buckets.get(user_id, (self._capacity, now))
        return min(self._capacity, tokens + (now - last) * self._refill_per_second)

    def spend(self, user_id: str, units: float) -> Decision:
        now = self._now()
        with self._lock:
            tokens = self._current(user_id, now)
            if tokens < units:
                shortfall = units - tokens
                wait = shortfall / self._refill_per_second if self._refill_per_second else math.inf
                # Persist the refill even on refusal, so the clock is not reset by retrying.
                self._buckets[user_id] = (tokens, now)
                return Decision(False, tokens, wait)
            self._buckets[user_id] = (tokens - units, now)
            return Decision(True, tokens - units, 0.0)

    def refund(self, user_id: str, units: float) -> None:
        if units <= 0:
            return
        now = self._now()
        with self._lock:
            tokens = self._current(user_id, now)
            self._buckets[user_id] = (min(self._capacity, tokens + units), now)

    def balance(self, user_id: str) -> float:
        return self._current(user_id, self._now())

    def reset(self) -> None:
        """Drop all budgets. Test-support only; never called by request handling."""
        with self._lock:
            self._buckets.clear()


_store: BudgetStore | None = None


def get_budget_store(settings: Settings | None = None) -> BudgetStore:
    """The process-wide store, built once from settings."""
    global _store
    if _store is None:
        settings = settings or get_settings()
        _store = InMemoryBudgetStore(
            capacity=float(settings.rate_limit_burst_units),
            refill_per_second=settings.rate_limit_units_per_hour / 3600.0,
        )
    return _store


def set_budget_store(store: BudgetStore | None) -> None:
    """Replace (or clear) the process-wide store. Test-support only."""
    global _store
    _store = store


def _too_many_requests(decision: Decision, reserved: float) -> HTTPException:
    """429 that says what ran out, how long to wait, and why the cost differs by endpoint."""
    wait = max(1, math.ceil(decision.retry_after_seconds))
    return HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail=(
            f"Query budget exhausted. This request needs {reserved:g} cost units and "
            f"{decision.balance:.1f} are available. Budget refills continuously; retry in "
            f"about {wait}s. Endpoints cost different amounts — a /rank ranking is roughly "
            f"an order of magnitude more expensive than a /chat turn, and a /rank refusal "
            f"costs nothing."
        ),
        headers={"Retry-After": str(wait)},
    )


class RequestBudget:
    """One request's reservation, settled by the handler once it knows what it ran."""

    def __init__(self, store: BudgetStore, user_id: str, reserved: float) -> None:
        self._store = store
        self._user_id = user_id
        self._reserved = reserved
        self._settled = False
        # Recorded on settlement so the Langfuse trace reports the real applied charge.
        self.charged: float | None = None
        self.balance_after: float | None = None

    @property
    def reserved(self) -> float:
        return self._reserved

    def settle(self, units: float) -> None:
        if self._settled:
            return
        self._settled = True
        refund = self._reserved - units
        if refund > 0:
            self._store.refund(self._user_id, refund)
        self.charged = units
        self.balance_after = round(self._store.balance(self._user_id), 2)
        logger.info(
            "rate limit: user=%s reserved=%g charged=%g balance=%.1f",
            self._user_id,
            self._reserved,
            units,
            self.balance_after,
        )


def reserve(units: float):
    """Build a FastAPI dependency reserving `units` for one request."""

    def dependency(
        user: AuthenticatedUser = Depends(require_user),  # noqa: B008 - FastAPI DI
    ) -> RequestBudget:
        settings = get_settings()
        store = get_budget_store(settings)
        decision = store.spend(user.id, units)
        if not decision.allowed:
            logger.info("rate limit: user=%s refused, needed=%g", user.id, units)
            raise _too_many_requests(decision, units)
        return RequestBudget(store, user.id, units)

    return dependency
