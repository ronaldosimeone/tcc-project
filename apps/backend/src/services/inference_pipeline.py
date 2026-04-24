"""
InferencePipelineService — fecha o loop entre o SensorSimulator e o modelo ML.

Sem este serviço, as leituras do simulador chegam ao frontend via SSE mas
nunca passam pelo modelo de predição, fazendo com que failure_probability
fique sempre em 0 e nenhum alerta WebSocket seja disparado.

Arquitetura
-----------
Este serviço subscreve ao mesmo SensorStreamService que o endpoint SSE usa,
garantindo que o modelo veja exatamente os mesmos dados que o frontend vê,
incluindo mudanças de modo via PUT /simulator/mode.

Fluxo (roda concorrentemente com o broadcast SSE):

    SensorStreamService._broadcast_loop()
        → queue.put_nowait(reading)      # mesma fila que o SSE usa
                    ↓
    InferencePipelineService._run()
        → queue.get()
        → ModelService.predict(PredictRequest)
        → save_prediction(db, ...)       # RF-09
        → AlertService.process_prediction()
        → WebSocket push se prob > 0.70  # RF-14
"""

from __future__ import annotations

import asyncio

import structlog

from src.core.database import AsyncSessionFactory
from src.core.exceptions import ModelNotAvailableError
from src.schemas.predict import PredictRequest, PredictResponse
from src.schemas.stream import SensorReading
from src.services.alert_service import AlertService
from src.services.model_registry import ModelRegistry
from src.services.prediction_service import save_prediction
from src.services.sensor_stream_service import SensorStreamService

log = structlog.get_logger(__name__)

# Tempo máximo aguardando dado na fila antes de re-verificar cancelamento
_QUEUE_TIMEOUT: float = 5.0


def _reading_to_request(reading: SensorReading) -> PredictRequest:
    """Converte SensorReading (payload SSE) em PredictRequest (input do modelo)."""
    return PredictRequest(
        TP2=reading.TP2,
        TP3=reading.TP3,
        H1=reading.H1,
        DV_pressure=reading.DV_pressure,
        Reservoirs=reading.Reservoirs,
        Motor_current=reading.Motor_current,
        Oil_temperature=reading.Oil_temperature,
        COMP=reading.COMP,
        DV_eletric=reading.DV_eletric,
        Towers=reading.Towers,
        MPG=reading.MPG,
        Oil_level=reading.Oil_level,
    )


class InferencePipelineService:
    """
    Serviço de background: lê do stream SSE → inferência ML → DB → alertas.

    Lifecycle:
        start()  — chamado no lifespan do FastAPI (startup)
        stop()   — chamado no lifespan do FastAPI (shutdown)
    """

    def __init__(
        self,
        stream_service: SensorStreamService,
        registry: ModelRegistry,
        alert_service: AlertService,
    ) -> None:
        self._stream = stream_service
        self._registry = registry
        self._alert = alert_service
        self._task: asyncio.Task[None] | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """Inicia o loop de inferência como asyncio.Task."""
        self._task = asyncio.create_task(self._run(), name="inference-pipeline")
        log.info("inference_pipeline_started")

    async def stop(self) -> None:
        """Cancela o loop e aguarda finalização limpa."""
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        log.info("inference_pipeline_stopped")

    # ── Loop principal ─────────────────────────────────────────────────────

    async def _run(self) -> None:
        queue = self._stream.subscribe()
        log.info("inference_pipeline_subscribed")
        try:
            while True:
                try:
                    reading: SensorReading = await asyncio.wait_for(
                        queue.get(), timeout=_QUEUE_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    # Nenhum dado ainda (ex.: broadcast task reiniciando). Continua.
                    continue

                await self._process(reading)

        except asyncio.CancelledError:
            log.info("inference_pipeline_cancelled")
            raise  # propaga corretamente para o Task ser marcado como cancelado
        finally:
            self._stream.unsubscribe(queue)
            log.info("inference_pipeline_unsubscribed")

    # ── Processamento de uma leitura ──────────────────────────────────────

    async def _process(self, reading: SensorReading) -> None:
        """Converte → infere → persiste → alerta (todos os erros são isolados)."""
        # 1. Obtém o modelo — pula se não carregado ainda
        try:
            model_service = await self._registry.get()
        except ModelNotAvailableError:
            log.debug("inference_skipped_no_model")
            return

        # 2. Inferência (CPU-bound, síncrona mas rápida para RandomForest/XGBoost)
        predict_request = _reading_to_request(reading)
        try:
            result: PredictResponse = model_service.predict(predict_request)
        except Exception as exc:
            log.warning("inference_error", error=str(exc))
            return

        # 3. Persiste no banco — sessão própria por leitura (RF-09)
        async with AsyncSessionFactory() as db:
            try:
                await save_prediction(db, predict_request, result)
                await db.commit()
            except Exception as exc:
                await db.rollback()
                log.warning("inference_save_error", error=str(exc))

        # 4. Dispara alerta WebSocket se probabilidade > 0.70 (RF-14)
        await self._alert.process_prediction(
            {
                "probability": result.failure_probability,
                "predicted_class": result.predicted_class,
                "timestamp": result.timestamp,
            }
        )

        log.debug(
            "inference_completed",
            prob=result.failure_probability,
            cls=result.predicted_class,
        )
