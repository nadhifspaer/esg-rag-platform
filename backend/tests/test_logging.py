"""Tests for application log configuration."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest

from app.core.logging import APP_LOGGER_NAME, configure_logging, reset_logging


@pytest.fixture(autouse=True)
def _clean_logging() -> Iterator[None]:
    """Each test starts from an unconfigured `app` logger and leaves it configured as the
    app itself would, so test ordering cannot change what later tests observe."""
    reset_logging()
    yield
    reset_logging()
    configure_logging("INFO")


def test_an_info_record_from_app_code_reaches_a_handler(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The end-to-end property. Before this fix the record was dropped at level checking."""
    configure_logging("INFO")
    # caplog attaches to root, so allow propagation for the duration of this assertion.
    logging.getLogger(APP_LOGGER_NAME).propagate = True
    with caplog.at_level(logging.INFO, logger=APP_LOGGER_NAME):
        logging.getLogger("app.core.rate_limit").info("reserved=20 charged=0")
    assert "reserved=20 charged=0" in caplog.text


def test_unconfigured_app_loggers_would_have_dropped_info() -> None:
    """Pins the original defect, so a future change that removes the call is caught."""
    assert logging.getLogger("app.core.rate_limit").getEffectiveLevel() == logging.WARNING


def test_configure_sets_the_level_on_every_app_child() -> None:
    configure_logging("INFO")
    for name in ("app.core.rate_limit", "app.api.chat", "app.rag.generation.loop"):
        assert logging.getLogger(name).getEffectiveLevel() == logging.INFO


def test_repeated_configuration_does_not_duplicate_handlers() -> None:
    """`--reload` re-imports main.py on every code change; without the guard each reload
    would add a handler and every line would print N times."""
    for _ in range(5):
        configure_logging("INFO")
    assert len(logging.getLogger(APP_LOGGER_NAME).handlers) == 1


def test_records_do_not_propagate_to_root() -> None:
    """Stops double emission if anything else ever configures root."""
    configure_logging("INFO")
    assert logging.getLogger(APP_LOGGER_NAME).propagate is False


def test_level_is_configurable() -> None:
    configure_logging("WARNING")
    assert logging.getLogger("app.core.rate_limit").getEffectiveLevel() == logging.WARNING


def test_uvicorns_handler_is_reused_when_present() -> None:
    """Keeps application records in the same stream and format as the request log."""
    uvicorn_logger = logging.getLogger("uvicorn")
    sentinel = logging.StreamHandler()
    uvicorn_logger.addHandler(sentinel)
    try:
        configure_logging("INFO")
        assert logging.getLogger(APP_LOGGER_NAME).handlers == [sentinel]
    finally:
        uvicorn_logger.removeHandler(sentinel)


def test_a_standalone_handler_is_used_without_uvicorn() -> None:
    """Works under pytest or a plain script, where uvicorn was never involved."""
    assert not logging.getLogger("uvicorn").handlers
    configure_logging("INFO")
    handlers = logging.getLogger(APP_LOGGER_NAME).handlers
    assert len(handlers) == 1 and handlers[0].formatter is not None
