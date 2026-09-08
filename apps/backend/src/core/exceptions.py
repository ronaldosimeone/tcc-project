"""
Custom domain exceptions and global FastAPI exception handlers.

Handler registration order in main.py
--------------------------------------
1. AppError          → app_error_handler        (domain errors, 4xx/5xx)
2. RateLimitExceeded → rate_limit_exceeded_handler (429, in rate_limit.py)
3. Exception         → unhandled_exception_handler (catch-all, RNF-18)

FastAPI dispatches to the most specific handler by walking the exception
MRO, so AppError always takes precedence over the generic Exception catch-all.
"""

from __future__ import annotations

import structlog
from fastapi import Request, status
from fastapi.responses import JSONResponse

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Domain exceptions
# ---------------------------------------------------------------------------


class AppError(Exception):
    """Base class for all application-level errors."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    detail: str = "An unexpected error occurred."

    def __init__(self, detail: str | None = None) -> None:
        self.detail = detail or self.__class__.detail
        super().__init__(self.detail)


class NotFoundError(AppError):
    status_code = status.HTTP_404_NOT_FOUND
    detail = "Resource not found."


class ConflictError(AppError):
    status_code = status.HTTP_409_CONFLICT
    detail = "Resource already exists."


class UnauthorizedError(AppError):
    status_code = status.HTTP_401_UNAUTHORIZED
    detail = "Authentication required."


class ForbiddenError(AppError):
    status_code = status.HTTP_403_FORBIDDEN
    detail = "Insufficient permissions."


class ModelNotAvailableError(AppError):
    """Raised when the ML model singleton was not loaded during startup."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    detail = "Prediction model is not available. Check application startup logs."


class MCPUnavailableError(AppError):
    """RF-22 — o servidor MCP (apps/mcp-server) não respondeu ou retornou erro
    ao chamar `search_maintenance_manual`. Nunca expõe a URL interna do MCP
    nem o traceback ao cliente (RF-22 §13)."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    detail = "Serviço de busca em manuais técnicos (MCP) indisponível no momento."


class OllamaUnavailableError(AppError):
    """RF-22 — o Ollama (host.docker.internal:11434) não respondeu, recusou a
    conexão, deu timeout, ou o modelo configurado não está disponível."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    detail = "Serviço de geração de sugestões (Ollama) indisponível no momento."


class OllamaResponseError(AppError):
    """RF-22 — o Ollama respondeu, mas o conteúdo não é um Markdown válido
    (vazio, JSON, HTML) — nunca repassado ao cliente como se fosse o plano."""

    status_code = status.HTTP_502_BAD_GATEWAY
    detail = "Resposta inválida do serviço de geração de sugestões."


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """Convert any AppError subclass into a standardised JSON error response."""
    log.warning(
        "app_error",
        error=exc.__class__.__name__,
        detail=exc.detail,
        status_code=exc.status_code,
        path=str(request.url.path),
        method=request.method,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.__class__.__name__, "detail": exc.detail},
    )


async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """
    Catch-all handler for any exception not covered by a specific handler.

    RNF-18 — guarantees:
      • HTTP 500 is always returned (no 200 with a confusing body).
      • The response body never contains stack traces, exception class
        names, or internal details that could aid an attacker.
      • The full exception — including traceback — is logged server-side
        so engineers can diagnose the problem without exposing it to clients.
    """
    log.exception(
        "unhandled_exception",
        exc_type=type(exc).__name__,
        path=str(request.url.path),
        method=request.method,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "detail": (
                "An unexpected error occurred on the server. "
                "Our team has been notified. Please try again later."
            ),
        },
    )
