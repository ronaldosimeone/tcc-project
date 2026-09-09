"""
Settings router — RF-25 / RNF-49.

GET  /v1/settings/alerts       — configuração global atual (ou o default).
PUT  /v1/settings/alerts       — salva a configuração global (limiar + canais).
POST /v1/settings/alerts/test  — envia uma notificação de teste aos canais habilitados.

Protegido pelo mesmo `X-Admin-Token` já usado por `/models` (RF-11) —
nenhum mecanismo de autenticação novo (o projeto não possui sistema de
usuários; a configuração é global para o sistema, não por usuário — ver
`models/alert_settings.py`).

Zero regra de negócio aqui — apenas I/O e injeção, conforme Clean Arch
(CLAUDE.md §3): toda a lógica vive em `AlertSettingsService` e
`NotificationTestService`.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.core.auth import require_admin_token
from src.schemas.alert_settings import (
    AlertSettingsResponse,
    AlertSettingsUpdateRequest,
    NotificationTestResponse,
)
from src.services.alert_service import (
    get_alert_settings_service,
    get_notification_test_service,
)
from src.services.alert_settings_service import AlertSettingsService
from src.services.notification_test_service import NotificationTestService

router: APIRouter = APIRouter(
    prefix="/v1/settings",
    tags=["Alert Settings"],
    dependencies=[Depends(require_admin_token)],
)


@router.get(
    "/alerts",
    response_model=AlertSettingsResponse,
    summary="Configuração global de alerta atual",
    description=(
        "[RF-25] Devolve o limiar de alerta crítico configurado. Se nenhuma "
        "configuração foi salva ainda, devolve o default do RF-24 (0.85) "
        "SEM criar um registro no banco — leitura sem efeito colateral. "
        "Requires `X-Admin-Token` header."
    ),
)
async def get_alert_settings(
    service: AlertSettingsService = Depends(get_alert_settings_service),
) -> AlertSettingsResponse:
    config = await service.get_settings()
    return AlertSettingsResponse(
        alert_threshold=config.alert_threshold,
        telegram_enabled=config.telegram_enabled,
        email_enabled=config.email_enabled,
        alert_email=config.alert_email,
    )


@router.put(
    "/alerts",
    response_model=AlertSettingsResponse,
    summary="Salva a configuração global de alerta",
    description=(
        "[RF-25] Valida `0.5 <= alert_threshold <= 0.95` (422 caso contrário) "
        "e persiste no PostgreSQL — atômico, no máximo uma configuração "
        "global (ver `models/alert_settings.py`). O valor passa a ser usado "
        "imediatamente por `CriticalFailureNotificationService` (RF-24) na "
        "próxima predição. Requires `X-Admin-Token` header."
    ),
)
async def update_alert_settings(
    payload: AlertSettingsUpdateRequest,
    service: AlertSettingsService = Depends(get_alert_settings_service),
) -> AlertSettingsResponse:
    saved = await service.upsert_settings(
        alert_threshold=payload.alert_threshold,
        telegram_enabled=payload.telegram_enabled,
        email_enabled=payload.email_enabled,
        alert_email=payload.alert_email,
    )
    return AlertSettingsResponse(
        alert_threshold=saved.alert_threshold,
        telegram_enabled=saved.telegram_enabled,
        email_enabled=saved.email_enabled,
        alert_email=saved.alert_email,
    )


@router.post(
    "/alerts/test",
    response_model=NotificationTestResponse,
    summary="Envia uma notificação de teste aos canais habilitados",
    description=(
        "[RF-25] Valida a configuração de notificação sem depender de uma "
        "predição real — testa exatamente os canais habilitados "
        "(Telegram e/ou e-mail via Resend), usando os adapters reais do "
        "RF-24/RF-25 com mensagens claramente marcadas como teste. "
        "Protegido por um rate limit próprio de 10s (anti-duplo-clique), "
        "independente do rate limit de 15 min de falha crítica. Nunca "
        "aceita token/chat ID/API key do cliente — o destino é sempre a "
        "configuração do backend. Requires `X-Admin-Token` header."
    ),
    responses={
        400: {"description": "Nenhum canal de notificação está habilitado."},
        429: {"description": "Teste chamado de novo antes do TTL anti-spam expirar."},
        502: {"description": "Todos os canais habilitados falharam ao enviar."},
    },
)
async def test_notification(
    service: NotificationTestService = Depends(get_notification_test_service),
) -> NotificationTestResponse:
    message = await service.send_test_notification()
    return NotificationTestResponse(message=message)
