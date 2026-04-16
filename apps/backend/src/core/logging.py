"""
Structured logging configuration — structlog.

Call configure_logging() once in the application lifespan (before any I/O).
After that, every module can do:

    import structlog
    log = structlog.get_logger(__name__)
    log.info("prediction_complete", class_=1, probability=0.92)

Output shapes
-------------
  debug=True  → human-readable, colour-coded console output (development).
  debug=False → newline-delimited JSON (ndjson), suitable for log aggregators
                such as Loki, ELK or AWS CloudWatch (production).

JSON log line example
---------------------
  {"event": "prediction_complete", "class_": 1, "probability": 0.92,
   "level": "info", "logger": "src.services.model_service",
   "timestamp": "2024-06-01T12:00:00.123456Z"}
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(*, debug: bool = False) -> None:
    """
    Configure structlog and the stdlib logging bridge.

    Parameters
    ----------
    debug:
        True  → ConsoleRenderer with colours (development).
        False → JSONRenderer — machine-readable ndjson (production).
    """
    level = logging.DEBUG if debug else logging.INFO

    # ── Processors shared by both modes ──────────────────────────────────
    shared_processors: list[structlog.types.Processor] = [
        # Merge any bound context variables (e.g. request_id set by middleware)
        structlog.contextvars.merge_contextvars,
        # Add log level and ISO-8601 UTC timestamp to every event.
        # NOTE: add_logger_name is intentionally omitted — it accesses
        # logger.name which is only available on stdlib Logger objects,
        # not on structlog's PrintLogger.  Module context is instead
        # carried by the positional argument passed to get_logger(__name__).
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        # Render Python stack info if present
        structlog.processors.StackInfoRenderer(),
    ]

    if debug:
        # ── Development: pretty, colour-coded output ──────────────────────
        processors: list[structlog.types.Processor] = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    else:
        # ── Production: structured JSON ───────────────────────────────────
        processors = shared_processors + [
            # Format exc_info as a structured "exception" dict
            structlog.processors.ExceptionRenderer(),
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        # Cache the bound logger per module for zero-overhead repeated calls
        cache_logger_on_first_use=True,
    )

    # ── Bridge stdlib logging → structlog ─────────────────────────────────
    # Third-party libraries (uvicorn, SQLAlchemy, alembic) emit via stdlib.
    # force=True resets any previous root-logger configuration.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
        force=True,
    )
