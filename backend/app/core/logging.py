"""Make the application's own log records actually appear (uvicorn leaves root unconfigured)."""

from __future__ import annotations

import logging
import sys

# The parent of every `app.*` logger in this codebase.
APP_LOGGER_NAME = "app"

_LOG_FORMAT = "%(levelname)-8s %(name)s: %(message)s"


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Attach a handler to the `app` logger and set its level. Safe to call repeatedly."""
    logger = logging.getLogger(APP_LOGGER_NAME)
    logger.setLevel(level.upper())

    # Records stop here rather than bubbling to root, so they are never emitted twice.
    logger.propagate = False

    if not logger.handlers:
        logger.addHandler(_build_handler())
    return logger


def _build_handler() -> logging.Handler:
    """Uvicorn's handler if it is running, otherwise a plain stderr handler."""
    uvicorn_handlers = logging.getLogger("uvicorn").handlers
    if uvicorn_handlers:
        return uvicorn_handlers[0]
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    return handler


def reset_logging() -> None:
    """Detach handlers and restore propagation. Test-support only."""
    logger = logging.getLogger(APP_LOGGER_NAME)
    logger.handlers.clear()
    logger.propagate = True
    logger.setLevel(logging.NOTSET)
