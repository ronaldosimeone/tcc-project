"""
CriticalFailureNotificationService — RF-24 / RNF-48 (+ RF-25 / RNF-49 + RF-?? / RNF-50/51).

Orquestra: threshold (> 0.85 por padrão, estrito) -> rate limit por
equipamento (15 min, TelegramAlertRateLimiter) -> canais de notificação
habilitados (Telegram e/ou e-mail, RF-25).

RF-25: o limiar efetivo e os canais habilitados agora vêm de
`AlertSettingsService` (configuração global persistida em
`/v1/settings/alerts`) quando uma configuração foi salva;
`CRITICAL_FAILURE_THRESHOLD` (0.85) + Telegram sempre ligado + e-mail sempre
desligado continuam sendo os valores usados enquanto ninguém configurou nada
(comportamento original do RF-24 preservado — ver
`AlertSettingsService.get_settings`) e o fallback de segurança se a leitura
da configuração falhar por qualquer motivo (DB indisponível etc.).
`settings_service=None` / `email_adapter=None` (defaults) preservam o
comportamento 100% original do RF-24 — usado pelos testes existentes de
`test_notifications.py` sem nenhuma alteração.

RNF-50/51 — envio assíncrono via Celery: o ENVIO real (chamadas de rede ao
Telegram/Resend, o que pode bloquear por segundos) foi extraído para
`dispatch_channels()`, chamado de dois jeitos:
  - `enqueue_notification=None` (default, usado por TODOS os testes
    existentes de RF-24/RF-25 sem nenhuma alteração): `notify_if_critical`
    chama `dispatch_channels` diretamente, in-process — comportamento
    síncrono ORIGINAL, byte a byte.
  - `enqueue_notification=<callable>` (usado SÓ pela wiring real de
    produção, `services/alert_service.py`): `notify_if_critical` decide
    enviar ou não (threshold) e adquire o rate limit (ainda síncrono e
    atômico, Postgres, RF-24 inalterado) e então ENFILEIRA um payload
    serializável via Celery em vez de chamar os adapters — retorna
    IMEDIATAMENTE, sem aguardar nenhuma chamada de rede. O envio real
    acontece depois, no processo do `celery-worker`
    (`src/tasks/notification_tasks.py`), que reconstrói os adapters e chama
    `dispatch_channels()` — a MESMA lógica de fan-out/falha-parcial/release,
    sem nenhuma duplicação de regra de negócio.

    prediction
        v
    probability > threshold efetivo ?  (RF-25: configurável; RF-24: 0.85 fixo
        |                                enquanto não configurado)
        v
    try_acquire(equipment_id)  -- atômico, ver telegram_alert_rate_limiter.py
    (protege AMBOS os canais — 1 episódio de falha crítica por equipamento
     a cada 15 min, independente de quantos canais estão habilitados — e
     continua ANTES do enqueue, não dentro da task, para preservar a mesma
     garantia atômica de "no máximo um envio por janela" sem depender de
     idempotência na fila)
        v
    rate limited? -> loga e retorna (nenhum canal é chamado, nada enfileirado)
        v
    enqueue_notification is None?
        SIM -> dispatch_channels(...) direto, in-process (mesmo processo)
        NÃO -> enqueue_notification(payload) -- Celery .delay(), retorna já
               (dispatch_channels roda depois, no celery-worker)

Nunca levanta exceção — uma falha de notificação externa não pode derrubar
o pipeline de inferência que já calculou a predição (RF-24 §16). Chamado a
partir de `AlertService.process_prediction` (RF-14), não do pipeline de ML
diretamente — reaproveita o mesmo ponto de integração já usado pelo alerta
WebSocket, sem uma segunda inferência nem lógica duplicada.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import structlog

from src.core.exceptions import EmailNotificationError, TelegramNotificationError
from src.services.alert_settings_service import (
    AlertSettingsService,
    AlertSettingsSnapshot,
)
from src.services.email_notification_adapter import EmailNotificationAdapter
from src.services.telegram_alert_rate_limiter import TelegramAlertRateLimiter
from src.services.telegram_notification_adapter import (
    CriticalFailureNotification,
    NotificationAdapter,
)

log = structlog.get_logger(__name__)

# RF-24 — regra de negócio fixa: estritamente > 0.85 (nunca >=). Constante
# própria — ver comparação com RF-14 (0.70) e RF-22 (0.70) no docstring do
# módulo: mesma faixa de valores por coincidência de domínio, três decisões
# de negócio independentes.
CRITICAL_FAILURE_THRESHOLD: float = 0.85

# Snapshot usado quando `settings_service` é `None` ou a leitura falha —
# comportamento ORIGINAL do RF-24, byte a byte: Telegram sempre tentado,
# e-mail nunca (canal não existia antes do RF-25).
_RF24_DEFAULT_SNAPSHOT = AlertSettingsSnapshot(
    alert_threshold=CRITICAL_FAILURE_THRESHOLD,
    telegram_enabled=True,
    email_enabled=False,
    alert_email=None,
)

# Assinatura do enfileiramento — só recebe um dict JSON-serializável (RNF-50
# §6). Injetado pelo chamador (produção: `enqueue_critical_failure_notification`
# de `src/tasks/notification_tasks.py`, que faz `.delay(payload)`) para este
# módulo NUNCA precisar importar Celery/tasks diretamente (evita import
# circular — `notification_tasks.py` importa ESTA classe para reconstruir o
# service no worker).
EnqueueNotificationFn = Callable[[dict[str, Any]], None]


class CriticalFailureNotificationService:
    def __init__(
        self,
        adapter: NotificationAdapter,
        rate_limiter: TelegramAlertRateLimiter,
        dashboard_url: str,
        settings_service: AlertSettingsService | None = None,
        email_adapter: EmailNotificationAdapter | None = None,
        enqueue_notification: EnqueueNotificationFn | None = None,
    ) -> None:
        self._adapter = adapter
        self._rate_limiter = rate_limiter
        # Sem barra final — a rota é sempre concatenada com "/" explícito
        # (ver _build_dashboard_url), nunca duplicando ou omitindo a barra.
        self._dashboard_url = dashboard_url.rstrip("/")
        # RF-25 — `None` preserva o comportamento original do RF-24, usado
        # pelos testes existentes de test_notifications.py sem alteração.
        self._settings_service = settings_service
        self._email_adapter = email_adapter
        # RNF-50 — `None` preserva o comportamento síncrono original (usado
        # por TODOS os testes existentes de RF-24/RF-25); só a wiring real
        # de produção (`services/alert_service.py`) passa um enqueue real.
        self._enqueue_notification = enqueue_notification

    async def _resolve_settings(self) -> AlertSettingsSnapshot:
        """RF-25 — configuração efetiva: salva no banco, ou o snapshot
        default do RF-24 se nunca configurada ou se a leitura falhar (DB
        indisponível nunca pode bloquear uma notificação crítica real)."""
        if self._settings_service is None:
            return _RF24_DEFAULT_SNAPSHOT
        try:
            return await self._settings_service.get_settings()
        except Exception:
            log.warning(
                "alert_settings_read_failed_using_default",
                default=CRITICAL_FAILURE_THRESHOLD,
            )
            return _RF24_DEFAULT_SNAPSHOT

    async def notify_if_critical(
        self,
        *,
        equipment_id: str,
        equipment_name: str,
        probability: float,
        timestamp: str,
    ) -> None:
        """Nunca lança — qualquer falha (rate limit, canal de notificação,
        ou erro inesperado) é logada e engolida aqui. Chamador nunca
        precisa de try/except.

        RNF-50: quando `enqueue_notification` foi injetado (produção), este
        método NUNCA aguarda uma chamada de rede — decide enviar (threshold)
        e adquire o rate limit (Postgres, atômico, síncrono — RF-24
        inalterado), depois enfileira e retorna imediatamente."""
        config = await self._resolve_settings()
        if probability <= config.alert_threshold:
            return

        try:
            acquired = await self._rate_limiter.try_acquire(equipment_id)
        except Exception:
            # Falha no próprio rate limiter (ex.: DB indisponível) não pode
            # derrubar a inferência — loga e desiste desta notificação.
            log.exception(
                "critical_failure_rate_limiter_error", equipment_id=equipment_id
            )
            return

        if not acquired:
            log.info(
                "critical_failure_notification_rate_limited",
                equipment_id=equipment_id,
                probability=probability,
            )
            return

        dashboard_url = self._build_dashboard_url(equipment_id)
        dispatch_kwargs: dict[str, Any] = {
            "equipment_id": equipment_id,
            "equipment_name": equipment_name,
            "probability": probability,
            "timestamp": timestamp,
            "dashboard_url": dashboard_url,
            "telegram_enabled": config.telegram_enabled,
            "email_enabled": config.email_enabled,
            "alert_email": config.alert_email,
        }

        if self._enqueue_notification is not None:
            # RNF-50 — desacopla o envio real (rede) do caminho de
            # inferência. `dispatch_kwargs` é um dict só com tipos
            # JSON-serializáveis (str/float/bool/None) — nenhum objeto
            # SQLAlchemy/FastAPI/adapter atravessa o broker.
            self._enqueue_notification(dispatch_kwargs)
            log.info(
                "critical_failure_notification_enqueued",
                equipment_id=equipment_id,
                probability=probability,
            )
            return

        # Sem Celery configurado (testes, ou uso direto) — dispatch síncrono,
        # comportamento ORIGINAL do RF-24/RF-25 preservado 1:1.
        await self.dispatch_channels(**dispatch_kwargs)

    async def dispatch_channels(
        self,
        *,
        equipment_id: str,
        equipment_name: str,
        probability: float,
        timestamp: str,
        dashboard_url: str,
        telegram_enabled: bool,
        email_enabled: bool,
        alert_email: str | None,
    ) -> None:
        """
        Envio real aos canais habilitados — a MESMA lógica de
        fan-out/falha-parcial/release do RF-24/RF-25 (RNF-50 §7: "não
        duplique o rate limiter", "reaproveitar os adapters existentes").
        Chamado tanto pelo fallback síncrono de `notify_if_critical` (sem
        Celery) quanto pela Celery task no processo do worker (com Celery)
        — único ponto de verdade da regra de fan-out, sem duplicação.

        Um canal falhar NUNCA impede a tentativa do outro (RF-25 §17).
        Nenhum canal enviado com sucesso -> libera o rate limit (permite
        retry na próxima predição crítica); ao menos um enviado -> mantém o
        lock (evita reenviar o mesmo episódio pelo canal que já funcionou).
        """
        telegram_ok = False
        email_ok = False

        if telegram_enabled:
            notification = CriticalFailureNotification(
                equipment_id=equipment_id,
                equipment_name=equipment_name,
                probability=probability,
                timestamp=timestamp,
                dashboard_url=dashboard_url,
            )
            try:
                await self._adapter.send_critical_failure(notification)
                telegram_ok = True
            except TelegramNotificationError as exc:
                log.warning(
                    "critical_failure_telegram_failed",
                    equipment_id=equipment_id,
                    error=exc.detail,
                )
            except Exception:
                log.exception(
                    "critical_failure_telegram_unexpected_error",
                    equipment_id=equipment_id,
                )
        else:
            log.debug("critical_failure_telegram_disabled", equipment_id=equipment_id)

        if email_enabled and alert_email and self._email_adapter:
            try:
                await self._email_adapter.send_critical_failure_email(
                    to=alert_email,
                    equipment_name=equipment_name,
                    probability=probability,
                    timestamp=timestamp,
                    dashboard_url=dashboard_url,
                )
                email_ok = True
            except EmailNotificationError as exc:
                log.warning(
                    "critical_failure_email_failed",
                    equipment_id=equipment_id,
                    error=exc.detail,
                )
            except Exception:
                log.exception(
                    "critical_failure_email_unexpected_error",
                    equipment_id=equipment_id,
                )
        elif email_enabled:
            log.debug(
                "critical_failure_email_skipped_no_address_or_adapter",
                equipment_id=equipment_id,
            )

        if not telegram_ok and not email_ok:
            # Nenhum canal enviou com sucesso (ou nenhum estava habilitado)
            # — não pode consumir a janela; permite retry na próxima
            # predição crítica (RF-24 §5) em vez de bloquear por 15 min por
            # causa de uma falha transitória.
            await self._rate_limiter.release(equipment_id)
            log.warning(
                "critical_failure_notification_all_channels_failed",
                equipment_id=equipment_id,
                telegram_enabled=telegram_enabled,
                email_enabled=email_enabled,
            )
            return

        log.info(
            "critical_failure_notification_sent",
            equipment_id=equipment_id,
            probability=probability,
            telegram_ok=telegram_ok,
            email_ok=email_ok,
        )

    def _build_dashboard_url(self, equipment_id: str) -> str:
        """
        URL absoluta construída SÓ a partir de `DASHBOARD_URL` (configuração
        do operador) + rota já existente no frontend (`/sensors/[id]`, ver
        `apps/frontend/app/sensors/[id]`) — nunca a partir de dado vindo da
        predição/LLM (RF-24 §4, evita open redirect). `equipment_id` só pode
        ser um dos valores internos que este serviço já recebeu do pipeline,
        nunca input de rede externo. Construída no processo WEB (antes do
        enqueue) — o payload que atravessa o broker já carrega a URL final,
        pronta; o worker não precisa (nem deve) reconstruí-la.
        """
        return f"{self._dashboard_url}/sensors/{equipment_id}"
