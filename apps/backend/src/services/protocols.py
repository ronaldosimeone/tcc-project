"""
Protocols de fronteira — RNF-56.

Interfaces estruturais (`typing.Protocol`, duck-typing estático — nenhuma
herança exigida) para as dependências que os ROUTERS recebem via
`Depends()`. O objetivo: um router declara `service: XProtocol =
Depends(get_x_service)` em vez de `service: ConcreteXClass = ...` — o
router passa a depender só do CONTRATO (os métodos que ele realmente
chama), nunca da classe concreta/infraestrutura por trás dela.

Auditoria (RNF-56 §1): nem toda classe usada por um router virou Protocol
aqui — só as que routers dependem DIRETAMENTE via `Depends` e que
encapsulam uma fronteira real (rede externa, estado compartilhado,
infraestrutura). Funções de módulo simples já injetadas via `Depends(get_db)`
+ uma função (`save_prediction`, `list_predictions`, `list_drift_reports`,
`check_health`) NÃO ganharam Protocol — já são testáveis via
`dependency_overrides[get_db]` (padrão consolidado em todo o projeto, ver
`tests/test_predictions_endpoint.py` etc.) e um Protocol de um método só
não adicionaria isolamento real, só cerimônia (RNF-56 §3: "não crie
Protocol para absolutamente todas as classes").

Cada classe concreta já satisfaz seu Protocol estruturalmente (mesmos
nomes de método, mesma assinatura) — nenhuma delas precisou ser alterada
para "implementar" o Protocol explicitamente.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from starlette.websockets import WebSocket

from src.schemas.maintenance import (
    MaintenanceSuggestionRequest,
    MaintenanceSuggestionResponse,
)
from src.schemas.models import ModelSummary
from src.schemas.predict import PredictRequest, PredictResponse
from src.schemas.stream import SensorReading

# Tipos de dado (Enum / dataclass), não infraestrutura — mesmos imports já
# feitos por `schemas/simulator.py` e pelos próprios services que os
# definem; reaproveitados aqui só para anotar os Protocols corretamente.
from src.services.alert_settings_service import AlertSettingsSnapshot
from src.services.maintenance_suggestion_service import SuggestionStreamEvent
from src.services.simulator import SimulatorMode


@runtime_checkable
class ModelServiceProtocol(Protocol):
    """Fronteira usada por `routers/predict.py` — inferência ML (RF-05)."""

    def predict(self, request: PredictRequest) -> PredictResponse: ...


@runtime_checkable
class InferenceCacheProtocol(Protocol):
    """Fronteira usada por `routers/predict.py` — cache de predição
    (RNF-70/RNF-71). O router nunca vê o cliente Redis, só get/set."""

    async def get(self, key: str) -> PredictResponse | None: ...

    async def set(self, key: str, value: PredictResponse) -> None: ...


@runtime_checkable
class AlertServiceProtocol(Protocol):
    """Fronteira usada por `routers/predict.py` — decisão de alerta crítico
    (RF-14/RF-24), que por sua vez compõe WS + Postgres + Celery
    internamente (detalhes que o router nunca vê)."""

    async def process_prediction(
        self, prediction: dict[str, Any]
    ) -> dict[str, Any]: ...


@runtime_checkable
class MaintenanceSuggestionServiceProtocol(Protocol):
    """Fronteira usada por `routers/maintenance.py` — RAG via MCP + Ollama
    (RF-22/23). O router não deve saber que existe um `MCPSearchClient`/
    `OllamaClient` por trás — só chama `suggest`/`suggest_stream`."""

    async def suggest(
        self, request: MaintenanceSuggestionRequest
    ) -> MaintenanceSuggestionResponse: ...

    def suggest_stream(
        self, request: MaintenanceSuggestionRequest
    ) -> AsyncIterator[SuggestionStreamEvent]: ...


@runtime_checkable
class AlertSettingsServiceProtocol(Protocol):
    """Fronteira usada por `routers/settings.py` (RF-25)."""

    async def get_settings(self) -> AlertSettingsSnapshot: ...

    async def upsert_settings(
        self,
        *,
        alert_threshold: float,
        telegram_enabled: bool,
        email_enabled: bool,
        alert_email: str | None,
    ) -> AlertSettingsSnapshot: ...


@runtime_checkable
class NotificationTestServiceProtocol(Protocol):
    """Fronteira usada por `routers/settings.py` — botão "Testar
    Notificação" (RF-25)."""

    async def send_test_notification(self) -> str: ...


class ModelRegistryProtocol(Protocol):
    """Fronteira usada por `routers/models.py` (RF-11/RNF-25) — hot-swap
    de modelo + leitura de status. `list_summaries`/`ensure_artefact_ready`
    encapsulam a lógica de "artefato pronto no disco" que antes vivia no
    router (RNF-56 §5 — lógica pesada fora do router)."""

    @property
    def active_name(self) -> str: ...

    async def swap(self, model_name: str) -> str: ...

    def list_summaries(self) -> list[ModelSummary]: ...

    def ensure_artefact_ready(self, model_name: str) -> None: ...


@runtime_checkable
class AlertBroadcasterProtocol(Protocol):
    """Fronteira usada por `routers/alerts_ws.py` — o router não deve
    conhecer `ConnectionManager` (infraestrutura WS em `core/ws_manager.py`)
    diretamente, só o contrato de broadcast/ack."""

    async def connect(self, websocket: WebSocket) -> None: ...

    async def send_personal(
        self, websocket: WebSocket, payload: dict[str, Any]
    ) -> bool: ...

    def disconnect(self, websocket: WebSocket) -> None: ...


class SensorSimulatorProtocol(Protocol):
    """Fronteira usada por `routers/simulator.py` (RNF-29)."""

    @property
    def mode(self) -> SimulatorMode: ...

    @mode.setter
    def mode(self, value: SimulatorMode) -> None: ...


class SensorStreamServiceProtocol(Protocol):
    """Fronteira usada por `routers/stream.py` (RF-12) — pub/sub de
    leituras de sensor."""

    def subscribe(self) -> asyncio.Queue[SensorReading]: ...

    def unsubscribe(self, queue: asyncio.Queue[SensorReading]) -> None: ...
