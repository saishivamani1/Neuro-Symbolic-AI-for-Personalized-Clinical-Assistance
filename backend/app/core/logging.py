"""
app/core/logging.py

Structured logging configuration for the platform.

Uses Python's standard `logging` module with a JSON-capable formatter
(via python-json-logger) so that logs can be ingested by log aggregators
(Loki, Datadog, CloudWatch, etc.) without extra parsing.

Usage
-----
    from app.core.logging import get_logger

    logger = get_logger(__name__)
    logger.info("Document ingested", extra={"document_id": doc_id})

IMPORTANT: Do NOT log patient-identifiable information (PII) in
production.  Use anonymised identifiers (patient_id) instead of
names, dates of birth, or contact details.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from app.core.config import get_settings

# --------------------------------------------------------------------------- #
# Optional structured JSON formatter
# python-json-logger v3.x: from pythonjsonlogger.jsonlogger import JsonFormatter
# python-json-logger v2.x: from pythonjsonlogger import jsonlogger
# --------------------------------------------------------------------------- #
_JsonFormatter = None
try:
    # v3.x path (preferred)
    from pythonjsonlogger.jsonlogger import JsonFormatter as _JsonFormatter  # type: ignore[import,no-redef]
    _JSON_LOGGER_AVAILABLE = True
except ImportError:
    try:
        # v2.x fallback
        from pythonjsonlogger import jsonlogger as _jl  # type: ignore[import]
        _JsonFormatter = _jl.JsonFormatter  # type: ignore[assignment]
        _JSON_LOGGER_AVAILABLE = True
    except ImportError:  # pragma: no cover
        _JSON_LOGGER_AVAILABLE = False

class _PlainTextFormatter(logging.Formatter):
    """Human-readable fallback formatter used when python-json-logger is not
    installed or when log_format == 'text'."""

    FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    def __init__(self) -> None:
        super().__init__(fmt=self.FMT, datefmt="%Y-%m-%dT%H:%M:%S")


def _build_handler(log_format: str) -> logging.StreamHandler:  # type: ignore[type-arg]
    handler = logging.StreamHandler(sys.stdout)
    if log_format == "json" and _JSON_LOGGER_AVAILABLE and _JsonFormatter is not None:
        formatter = _JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
        )
    else:
        formatter = _PlainTextFormatter()
    handler.setFormatter(formatter)
    return handler


def configure_logging() -> None:
    """Set up the root logger.  Call once at application startup."""
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    # Remove any existing handlers to avoid duplicate output when
    # configure_logging() is called more than once (e.g. during tests).
    root.handlers.clear()
    root.addHandler(_build_handler(settings.log_format))

    # Suppress noisy third-party loggers.
    for noisy in ("httpx", "httpcore", "urllib3", "chromadb.telemetry"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
    """
    return logging.getLogger(name)


# --------------------------------------------------------------------------- #
# Request-scoped log helpers
# --------------------------------------------------------------------------- #

def log_request_start(
    logger: logging.Logger,
    request_id: str,
    endpoint: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """Emit a structured INFO record at the start of a request."""
    payload: dict[str, Any] = {"request_id": request_id, "endpoint": endpoint}
    if extra:
        payload.update(extra)
    logger.info("Request started", extra=payload)


def log_request_end(
    logger: logging.Logger,
    request_id: str,
    endpoint: str,
    latency_ms: float,
    extra: dict[str, Any] | None = None,
) -> None:
    """Emit a structured INFO record at the end of a request."""
    payload: dict[str, Any] = {
        "request_id": request_id,
        "endpoint": endpoint,
        "latency_ms": round(latency_ms, 2),
    }
    if extra:
        payload.update(extra)
    logger.info("Request completed", extra=payload)
