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
from typing import TextIO

import structlog

from src.core.log_sanitizer import redact_sensitive


def configure_logging(*, debug: bool = False, stream: TextIO | None = None) -> None:
    """
    Configure structlog and the stdlib logging bridge.

    Parameters
    ----------
    debug:
        True  → ConsoleRenderer with colours (development).
        False → JSONRenderer — machine-readable ndjson (production).
    stream:
        Destino da saída de log. ``None`` (padrão de produção) → ``sys.stdout``.
        Os testes de privacidade (RNF-63, ``tests/test_log_privacy.py``)
        passam um buffer aqui para inspecionar a linha REAL emitida pelo
        logger — provando que o sanitizer está no caminho, não só testando
        a função isolada.
    """
    out_stream: TextIO = stream if stream is not None else sys.stdout
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
        # RNF-63 — o sanitizer roda logo antes do ConsoleRenderer: redige
        # todos os kwargs do evento. O traceback (exc_info) continua sendo
        # formatado bonito pelo ConsoleRenderer (dev), fora do alcance do
        # sanitizer — aceitável só em modo debug local.
        processors: list[structlog.types.Processor] = shared_processors + [
            redact_sensitive,
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    else:
        # ── Production: structured JSON ───────────────────────────────────
        processors = shared_processors + [
            # Format exc_info as a structured "exception" dict FIRST, so o
            # sanitizer da RNF-63 logo abaixo também varre o texto do
            # traceback renderizado (um secret que tenha ido parar numa
            # mensagem de exceção é redigido antes de virar JSON).
            structlog.processors.ExceptionRenderer(),
            redact_sensitive,
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=out_stream),
        # Cache the bound logger per module for zero-overhead repeated calls.
        # Desligado quando um stream de teste é injetado — senão o logger
        # cacheado apontaria para o buffer do teste anterior.
        cache_logger_on_first_use=stream is None,
    )

    # ── Bridge stdlib logging → structlog ─────────────────────────────────
    # Third-party libraries (uvicorn, SQLAlchemy, alembic) emit via stdlib.
    # force=True resets any previous root-logger configuration.
    logging.basicConfig(
        format="%(message)s",
        stream=out_stream,
        level=level,
        force=True,
    )
