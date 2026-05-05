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

Buffer de histórico (V2 — fix da degradação OOD do MLP)
-------------------------------------------------------
Cada leitura também alimenta um ``SensorBuffer`` (deque thread-safe). Assim
que o buffer tem ``warmup_size`` amostras, este serviço re-aplica o
``MetroPTPreprocessor`` exatamente como no treino, calculando std/ma/lag/roc/
min/max reais e passando a *última* linha pro modelo via
``ModelService.predict_from_features``.  Antes do warmup, mantemos o caminho
stateless original (``predict(request)``) para que a aplicação não bloqueie
durante os primeiros segundos após o startup.

Fluxo (roda concorrentemente com o broadcast SSE):

    SensorStreamService._broadcast_loop()
        → queue.put_nowait(reading)      # mesma fila que o SSE usa
                    ↓
    InferencePipelineService._run()
        → queue.get()
        → buffer.append(reading)
        → if buffer.is_warm():
              preprocessor.transform(buffer.to_dataframe()) → last row
              ModelService.predict_from_features(last_row)
          else:
              ModelService.predict(PredictRequest)        # fallback OOD
        → save_prediction(db, ...)       # RF-09
        → AlertService.process_prediction()
        → WebSocket push se prob > 0.70  # RF-14
"""

from __future__ import annotations

import asyncio

import pandas as pd
import structlog

from src.core.database import AsyncSessionFactory
from src.core.exceptions import ModelNotAvailableError
from src.schemas.predict import PredictRequest, PredictResponse
from src.schemas.stream import SensorReading
from src.services.alert_service import AlertService
from src.services.feature_buffer import SensorBuffer
from src.services.model_registry import ModelRegistry
from src.services.model_service import ModelService
from src.services.prediction_service import save_prediction
from src.services.preprocessing import MetroPTPreprocessor
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


def _reading_to_dict(reading: SensorReading) -> dict[str, float]:
    """
    Extrai apenas os 12 sensores numéricos do SensorReading para o buffer.

    Note que o ``timestamp`` é deliberadamente excluído — o
    ``MetroPTPreprocessor`` trabalha sobre colunas numéricas em ordem
    posicional, e um campo datetime quebraria o ``select_dtypes``.
    """
    return {
        "TP2": reading.TP2,
        "TP3": reading.TP3,
        "H1": reading.H1,
        "DV_pressure": reading.DV_pressure,
        "Reservoirs": reading.Reservoirs,
        "Motor_current": reading.Motor_current,
        "Oil_temperature": reading.Oil_temperature,
        "COMP": reading.COMP,
        "DV_eletric": reading.DV_eletric,
        "Towers": reading.Towers,
        "MPG": reading.MPG,
        "Oil_level": reading.Oil_level,
    }


def _infer_with_history(
    model_service: ModelService,
    buffer_df: pd.DataFrame,
    preprocessor: MetroPTPreprocessor,
) -> PredictResponse:
    """
    Roda o pipeline de feature engineering e a predição numa única chamada
    CPU-bound — empacotada num helper para que ``asyncio.to_thread`` despache
    tudo de uma vez sem ping-pong com o event loop.

    A última linha do DataFrame é a leitura mais recente, e é a única que
    contém todas as rolling/lag features completamente preenchidas.
    """
    engineered = preprocessor.transform(buffer_df)
    # iloc[[-1]] (lista) preserva o tipo DataFrame; iloc[-1] (escalar)
    # devolveria uma Series e quebraria o ``X[self._expected_features]``.
    last_row: pd.DataFrame = engineered.iloc[[-1]].reset_index(drop=True)
    return model_service.predict_from_features(last_row)


class InferencePipelineService:
    """
    Serviço de background: lê do stream SSE → inferência ML → DB → alertas.

    Lifecycle:
        start()  — chamado no lifespan do FastAPI (startup)
        stop()   — chamado no lifespan do FastAPI (shutdown)

    Parameters
    ----------
    stream_service, registry, alert_service:
        Dependências injetadas no lifespan.
    sensor_buffer:
        Buffer FIFO compartilhado.  Default: singleton de módulo
        ``feature_buffer.buffer`` (window=30, warmup=15).  Override em testes
        via DI para isolar fixtures.
    preprocessor:
        Mesmo ``MetroPTPreprocessor`` que o script de treino usa.  Default:
        nova instância com ``enable_v2_features=True`` (compatível com RF V2,
        XGB V2 e MLP V1, que apenas ignoram colunas não-treinadas).
    """

    def __init__(
        self,
        stream_service: SensorStreamService,
        registry: ModelRegistry,
        alert_service: AlertService,
        sensor_buffer: SensorBuffer | None = None,
        preprocessor: MetroPTPreprocessor | None = None,
    ) -> None:
        self._stream = stream_service
        self._registry = registry
        self._alert = alert_service
        self._buffer: SensorBuffer = sensor_buffer or get_sensor_buffer()
        self._preprocessor: MetroPTPreprocessor = (
            preprocessor or MetroPTPreprocessor()
        )
        self._task: asyncio.Task[None] | None = None
        # Loga só uma vez quando o buffer aquece — evita spam por leitura.
        self._warm_logged: bool = False

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """Inicia o loop de inferência como asyncio.Task."""
        self._task = asyncio.create_task(self._run(), name="inference-pipeline")
        log.info(
            "inference_pipeline_started",
            buffer_capacity=self._buffer.capacity,
            warmup_size=self._buffer.warmup_size,
        )

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

        # 2. Atualiza o histórico — ``deque.append`` + lock é O(1) e barato
        # o suficiente para rodar no event loop sem hand-off.
        self._buffer.append(_reading_to_dict(reading))
        warm = self._buffer.is_warm()
        if warm and not self._warm_logged:
            log.info(
                "inference_buffer_warm",
                buffered=len(self._buffer),
                warmup_size=self._buffer.warmup_size,
            )
            self._warm_logged = True

        # 3. Inferência — CPU-bound; despachada para a threadpool para não
        # bloquear o event loop (SSE broadcast, WebSocket, health checks).
        try:
            if warm:
                # Snapshot do buffer fora do thread (lock de leitura curto).
                buffer_df = self._buffer.to_dataframe()
                result: PredictResponse = await asyncio.to_thread(
                    _infer_with_history,
                    model_service,
                    buffer_df,
                    self._preprocessor,
                )
            else:
                # Fallback durante o warmup — features rolling zeradas, mas
                # mantém o pipeline funcional desde o primeiro tick.
                predict_request = _reading_to_request(reading)
                result = await asyncio.to_thread(
                    model_service.predict, predict_request
                )
        except Exception as exc:
            log.warning("inference_error", error=str(exc), warm=warm)
            return

        # 4. Persiste no banco — sessão própria por leitura (RF-09)
        predict_request = _reading_to_request(reading)
        async with AsyncSessionFactory() as db:
            try:
                await save_prediction(db, predict_request, result)
                await db.commit()
            except Exception as exc:
                await db.rollback()
                log.warning("inference_save_error", error=str(exc))

        # 5. Dispara alerta WebSocket se probabilidade > 0.70 (RF-14)
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
            warm=warm,
        )
