"""Shared pytest fixtures: mainly the auth stub (the gate itself is tested in `test_auth.py`)."""

from __future__ import annotations

from collections.abc import Iterator
from unittest import mock

import pytest

from app.core.auth import AuthenticatedUser, require_user
from app.core.rate_limit import set_budget_store
from app.main import app
from app.observability import tracing

# Shaped like a real Supabase access token's claims, minus the parts nothing reads.
STUB_USER = AuthenticatedUser(
    id="00000000-0000-4000-8000-000000000001",
    email="test@example.com",
    role="authenticated",
    claims={"sub": "00000000-0000-4000-8000-000000000001", "role": "authenticated"},
)


@pytest.fixture(autouse=True)
def _stub_authenticated_user() -> Iterator[None]:
    """Treat every request as coming from a logged-in user, unless a test says otherwise."""
    app.dependency_overrides[require_user] = lambda: STUB_USER
    yield
    app.dependency_overrides.pop(require_user, None)


@pytest.fixture(autouse=True)
def _tracing_disabled() -> Iterator[None]:
    """Never emit traces from the test suite, even with real Langfuse keys in the local .env."""
    with mock.patch.object(tracing, "get_langfuse", lambda: None):
        yield


@pytest.fixture(autouse=True)
def _fresh_rate_limit_budget() -> Iterator[None]:
    """Give every test a full budget, so one test's spending can't leak into the next."""
    set_budget_store(None)
    yield
    set_budget_store(None)
