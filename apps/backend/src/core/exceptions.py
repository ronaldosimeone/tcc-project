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


class NotificationTestRateLimitedError(AppError):
    """RF-25 — o botão "Testar Notificação" foi clicado de novo antes do
    fim de uma janela curta anti-spam (10s, independente do rate limit de
    15 min do RF-24 — ver `services/telegram_alert_rate_limiter.py`)."""

    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    detail = "Aguarde alguns segundos antes de testar novamente."


class TelegramNotificationError(AppError):
    """RF-24 — falha ao enviar notificação crítica via Telegram (config
    ausente, timeout, HTTP 4xx/5xx, JSON inválido, ou `ok: false` na
    resposta). SEMPRE capturada internamente por
    `CriticalFailureNotificationService` — nunca deve derrubar a
    predição/inferência que já foi calculada (RF-24 §16). `detail` nunca
    contém o token nem a URL completa do Bot API."""

    status_code = status.HTTP_502_BAD_GATEWAY
    detail = "Falha ao enviar notificação crítica via Telegram."


class EmailNotificationError(AppError):
    """RF-25 — falha ao enviar e-mail via Resend (config ausente, timeout,
    HTTP não-2xx, JSON inválido). Mesma política do TelegramNotificationError:
    SEMPRE capturada internamente, nunca derruba a predição/inferência.
    `detail` nunca contém a API key do Resend."""

    status_code = status.HTTP_502_BAD_GATEWAY
    detail = "Falha ao enviar notificação crítica por e-mail."


class NoNotificationChannelEnabledError(AppError):
    """RF-25 §10 — POST /v1/settings/alerts/test chamado sem nenhum canal
    (Telegram/e-mail) habilitado na configuração — nada para testar."""

    status_code = status.HTTP_400_BAD_REQUEST
    detail = "Nenhum canal de notificação está habilitado."


class NotificationTestFailedError(AppError):
    """RF-25 — todos os canais habilitados falharam durante o teste manual
    (ao menos um habilitado, per NoNotificationChannelEnabledError acima,
    mas nenhum enviou com sucesso). `detail` é a mensagem combinada por
    canal (ex.: "Telegram: falhou · E-mail: falhou"), nunca um segredo."""

    status_code = status.HTTP_502_BAD_GATEWAY
    detail = "Não foi possível enviar a notificação em nenhum canal habilitado."


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
