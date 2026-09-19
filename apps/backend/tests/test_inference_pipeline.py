"""
Testes unitários para InferencePipelineService.

Cobre:
    _reading_to_request   Conversão de todos os 12 campos.
    start / stop          Lifecycle: task criada e cancelada corretamente.
    _process              Inferência → DB → alerta (caminho feliz).
    _process sem modelo   ModelNotAvailableError → pula silenciosamente.
    _process erro ML      Exceção no predict() → não quebra o loop.
    _process erro DB      Falha no save → rollback, loop continua.
    alerta threshold      prob > 0.70 → broadcast_alert chamado.
    alerta abaixo         prob ≤ 0.70 → broadcast_alert NÃO chamado.
    unsubscribe no stop   Queue removida do SensorStreamService ao parar.

Estratégia de mock:
    - SensorStreamService: stub com subscribe/unsubscribe rastreáveis e uma
      fila em memória controlada pelo teste.
    - ModelRegistry: AsyncMock que retorna um ModelService stub.
    - AlertService: stub com process_prediction rastreável.
    - save_prediction: patchado no módulo inference_pipeline para evitar DB.
    - AsyncSessionFactory: patchado para retornar um context manager em memória.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import pandas as pd

from src.core.exceptions import ModelNotAvailableError
from src.schemas.predict import PredictRequest, PredictResponse
from src.schemas.stream import SensorReading
from src.services.inference_pipeline import (
    _QUEUE_TIMEOUT,
    InferencePipelineService,
    _infer_with_history,
    _reading_to_dict,
    _reading_to_request,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAILURE_READING = SensorReading(
    timestamp=datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
    TP2=3.50,
    TP3=5.20,
    H1=72.0,
    DV_pressure=0.65,
    Reservoirs=5.50,
    Motor_current=6.80,
    Oil_temperature=86.0,
    COMP=0.0,
    DV_eletric=0.0,
    Towers=0.0,
    MPG=0.0,
    Oil_level=0.0,
)

NORMAL_READING = SensorReading(
    timestamp=datetime(2024, 6, 1, 12, 0, 1, tzinfo=timezone.utc),
    TP2=5.90,
    TP3=3.40,
    H1=46.0,
    DV_pressure=0.05,
    Reservoirs=7.05,
    Motor_current=2.80,
    Oil_temperature=57.0,
    COMP=1.0,
    DV_eletric=1.0,
    Towers=1.0,
    MPG=1.0,
    Oil_level=1.0,
)


def _make_predict_response(prob: float) -> PredictResponse:
    return PredictResponse(
        predicted_class=1 if prob > 0.5 else 0,
        failure_probability=prob,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _make_stream_service(readings: list[SensorReading]) -> MagicMock:
    """
    Stub de SensorStreamService.

    subscribe() → asyncio.Queue pré-carregada com `readings`.
    unsubscribe() → rastreável (MagicMock).
    """
    queue: asyncio.Queue[SensorReading] = asyncio.Queue()
    for r in readings:
        queue.put_nowait(r)

    service = MagicMock()
    service.subscribe.return_value = queue
    service.unsubscribe = MagicMock()
    return service


def _make_registry(prob: float) -> MagicMock:
    """ModelRegistry stub que retorna uma predição com probabilidade `prob`."""
    model_service = MagicMock()
    model_service.predict.return_value = _make_predict_response(prob)

    registry = MagicMock()
    registry.get = AsyncMock(return_value=model_service)
    return registry


def _make_registry_unavailable() -> MagicMock:
    """ModelRegistry stub que simula modelo não carregado."""
    registry = MagicMock()
    registry.get = AsyncMock(side_effect=ModelNotAvailableError())
    return registry


def _make_alert_service() -> MagicMock:
    service = MagicMock()
    service.process_prediction = AsyncMock()
    return service


# ---------------------------------------------------------------------------
# _reading_to_request
# ---------------------------------------------------------------------------


def test_queue_timeout_is_5_seconds() -> None:
    assert _QUEUE_TIMEOUT == 5.0


def test_reading_to_dict_extracts_the_exact_12_sensor_keys() -> None:
    """RNF-64: comparação de dict EXATA (não campo a campo) — mata qualquer
    typo de chave introduzido em qualquer um dos 12 sensores de uma vez."""
    assert _reading_to_dict(FAILURE_READING) == {
        "TP2": FAILURE_READING.TP2,
        "TP3": FAILURE_READING.TP3,
        "H1": FAILURE_READING.H1,
        "DV_pressure": FAILURE_READING.DV_pressure,
        "Reservoirs": FAILURE_READING.Reservoirs,
        "Motor_current": FAILURE_READING.Motor_current,
        "Oil_temperature": FAILURE_READING.Oil_temperature,
        "COMP": FAILURE_READING.COMP,
        "DV_eletric": FAILURE_READING.DV_eletric,
        "Towers": FAILURE_READING.Towers,
        "MPG": FAILURE_READING.MPG,
        "Oil_level": FAILURE_READING.Oil_level,
    }


class _FakePreprocessor:
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["engineered_marker"] = out["TP2"] * 100
        return out


class _SpyModelService:
    def __init__(self) -> None:
        self.received_row: pd.DataFrame | None = None

    def predict_from_features(self, X: pd.DataFrame) -> str:
        self.received_row = X
        return "sentinel-response"


def test_infer_with_history_feeds_the_model_only_the_last_engineered_row() -> None:
    """RNF-64: `_infer_with_history` deve pegar a ÚLTIMA linha (mais
    recente) do DataFrame, com o índice resetado — não a primeira nem uma
    linha do meio, e não a Series de `iloc[-1]` escalar."""
    buffer_df = pd.DataFrame({"TP2": [1.0, 2.0, 3.0]})
    model_service = _SpyModelService()

    result = _infer_with_history(model_service, buffer_df, _FakePreprocessor())  # type: ignore[arg-type]

    assert result == "sentinel-response"
    assert model_service.received_row is not None
    assert len(model_service.received_row) == 1
    assert model_service.received_row["TP2"].iloc[0] == 3.0
    assert model_service.received_row["engineered_marker"].iloc[0] == 300.0
    assert list(model_service.received_row.index) == [0]  # reset_index(drop=True)


class TestReadingToRequest:
    def test_converte_todos_os_12_campos(self) -> None:
        req = _reading_to_request(FAILURE_READING)

        assert isinstance(req, PredictRequest)
        assert req.TP2 == FAILURE_READING.TP2
        assert req.TP3 == FAILURE_READING.TP3
        assert req.H1 == FAILURE_READING.H1
        assert req.DV_pressure == FAILURE_READING.DV_pressure
        assert req.Reservoirs == FAILURE_READING.Reservoirs
        assert req.Motor_current == FAILURE_READING.Motor_current
        assert req.Oil_temperature == FAILURE_READING.Oil_temperature
        assert req.COMP == FAILURE_READING.COMP
        assert req.DV_eletric == FAILURE_READING.DV_eletric
        assert req.Towers == FAILURE_READING.Towers
        assert req.MPG == FAILURE_READING.MPG
        assert req.Oil_level == FAILURE_READING.Oil_level

    def test_converte_leitura_normal(self) -> None:
        req = _reading_to_request(NORMAL_READING)
        assert req.TP2 == NORMAL_READING.TP2
        assert req.Motor_current == NORMAL_READING.Motor_current


# ---------------------------------------------------------------------------
# Lifecycle: start / stop
# ---------------------------------------------------------------------------


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_start_cria_task(self) -> None:
        stream = _make_stream_service([])
        registry = _make_registry(0.1)
        alert = _make_alert_service()

        svc = InferencePipelineService(stream, registry, alert)
        svc.start()

        assert svc._task is not None
        assert not svc._task.done()

        await svc.stop()

    @pytest.mark.asyncio
    async def test_stop_cancela_task(self) -> None:
        stream = _make_stream_service([])
        registry = _make_registry(0.1)
        alert = _make_alert_service()

        svc = InferencePipelineService(stream, registry, alert)
        svc.start()
        await svc.stop()

        assert svc._task is not None
        assert svc._task.done()

    @pytest.mark.asyncio
    async def test_stop_chama_unsubscribe(self) -> None:
        stream = _make_stream_service([])
        registry = _make_registry(0.1)
        alert = _make_alert_service()

        svc = InferencePipelineService(stream, registry, alert)
        svc.start()
        # Yielda ao event loop para a task executar subscribe() antes de cancelar
        await asyncio.sleep(0)
        await svc.stop()

        stream.unsubscribe.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_idempotente_sem_task(self) -> None:
        """stop() antes de start() não deve lançar exceção."""
        stream = _make_stream_service([])
        registry = _make_registry(0.1)
        alert = _make_alert_service()

        svc = InferencePipelineService(stream, registry, alert)
        await svc.stop()  # sem start() antes — não deve explodir


# ---------------------------------------------------------------------------
# _process: caminho feliz
# ---------------------------------------------------------------------------


class TestProcessHappyPath:
    @pytest.mark.asyncio
    async def test_chama_modelo_com_dados_corretos(self) -> None:
        stream = _make_stream_service([FAILURE_READING])
        registry = _make_registry(0.90)
        alert = _make_alert_service()

        with (
            patch(
                "src.services.inference_pipeline.save_prediction",
                new_callable=AsyncMock,
            ),
            patch(
                "src.services.inference_pipeline.AsyncSessionFactory"
            ) as mock_factory,
        ):
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(
                return_value=MagicMock(commit=AsyncMock(), rollback=AsyncMock())
            )
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_factory.return_value = mock_ctx

            svc = InferencePipelineService(stream, registry, alert)
            await svc._process(FAILURE_READING)

        model_service = await registry.get()
        model_service.predict.assert_called_once()
        called_req: PredictRequest = model_service.predict.call_args[0][0]
        assert called_req.TP2 == FAILURE_READING.TP2
        assert called_req.Motor_current == FAILURE_READING.Motor_current

    @pytest.mark.asyncio
    async def test_chama_process_prediction_apos_inferencia(self) -> None:
        stream = _make_stream_service([FAILURE_READING])
        registry = _make_registry(0.90)
        alert = _make_alert_service()

        with (
            patch(
                "src.services.inference_pipeline.save_prediction",
                new_callable=AsyncMock,
            ),
            patch(
                "src.services.inference_pipeline.AsyncSessionFactory"
            ) as mock_factory,
        ):
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(
                return_value=MagicMock(commit=AsyncMock(), rollback=AsyncMock())
            )
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_factory.return_value = mock_ctx

            svc = InferencePipelineService(stream, registry, alert)
            await svc._process(FAILURE_READING)

        alert.process_prediction.assert_awaited_once()
        call_kwargs = alert.process_prediction.call_args[0][0]
        assert call_kwargs["probability"] == 0.90


# ---------------------------------------------------------------------------
# _process: modelo não disponível
# ---------------------------------------------------------------------------


class TestProcessNoModel:
    @pytest.mark.asyncio
    async def test_pula_sem_modelo(self) -> None:
        """ModelNotAvailableError → process_prediction NÃO deve ser chamado."""
        stream = _make_stream_service([FAILURE_READING])
        registry = _make_registry_unavailable()
        alert = _make_alert_service()

        svc = InferencePipelineService(stream, registry, alert)
        await svc._process(FAILURE_READING)

        alert.process_prediction.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_pula_sem_modelo_nao_lanca_excecao(self) -> None:
        stream = _make_stream_service([FAILURE_READING])
        registry = _make_registry_unavailable()
        alert = _make_alert_service()

        svc = InferencePipelineService(stream, registry, alert)
        # Não deve lançar exceção
        await svc._process(FAILURE_READING)


# ---------------------------------------------------------------------------
# _process: erros de inferência
# ---------------------------------------------------------------------------


class TestProcessInferenceError:
    @pytest.mark.asyncio
    async def test_erro_predict_nao_quebra_loop(self) -> None:
        """Exceção no predict() → alert NÃO chamado, mas sem crash."""
        model_service = MagicMock()
        model_service.predict.side_effect = RuntimeError("modelo corrompido")

        registry = MagicMock()
        registry.get = AsyncMock(return_value=model_service)

        alert = _make_alert_service()
        stream = _make_stream_service([FAILURE_READING])

        svc = InferencePipelineService(stream, registry, alert)
        await svc._process(FAILURE_READING)

        alert.process_prediction.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_erro_save_faz_rollback_e_continua(self) -> None:
        """Falha no save_prediction → rollback; alert ainda chamado."""
        registry = _make_registry(0.90)
        alert = _make_alert_service()
        stream = _make_stream_service([FAILURE_READING])

        mock_db = MagicMock()
        mock_db.commit = AsyncMock()
        mock_db.rollback = AsyncMock()

        with (
            patch(
                "src.services.inference_pipeline.save_prediction",
                new_callable=AsyncMock,
                side_effect=Exception("DB connection lost"),
            ),
            patch(
                "src.services.inference_pipeline.AsyncSessionFactory"
            ) as mock_factory,
        ):
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_db)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_factory.return_value = mock_ctx

            svc = InferencePipelineService(stream, registry, alert)
            await svc._process(FAILURE_READING)

        mock_db.rollback.assert_awaited_once()
        # Alert ainda é chamado mesmo com falha no DB
        alert.process_prediction.assert_awaited_once()


# ---------------------------------------------------------------------------
# Threshold de alerta (RF-14)
# ---------------------------------------------------------------------------


class TestAlertThreshold:
    @pytest.mark.asyncio
    async def test_alerta_disparado_quando_prob_acima_threshold(self) -> None:
        """prob = 0.90 > 0.70 → process_prediction recebe dado com prob correto."""
        registry = _make_registry(0.90)
        alert = _make_alert_service()
        stream = _make_stream_service([FAILURE_READING])

        with (
            patch(
                "src.services.inference_pipeline.save_prediction",
                new_callable=AsyncMock,
            ),
            patch(
                "src.services.inference_pipeline.AsyncSessionFactory"
            ) as mock_factory,
        ):
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(
                return_value=MagicMock(commit=AsyncMock(), rollback=AsyncMock())
            )
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_factory.return_value = mock_ctx

            svc = InferencePipelineService(stream, registry, alert)
            await svc._process(FAILURE_READING)

        payload = alert.process_prediction.call_args[0][0]
        assert payload["probability"] == 0.90

    @pytest.mark.asyncio
    async def test_process_prediction_chamado_mesmo_prob_baixo(self) -> None:
        """process_prediction é SEMPRE chamado (AlertService decide o threshold internamente)."""
        registry = _make_registry(0.10)
        alert = _make_alert_service()
        stream = _make_stream_service([NORMAL_READING])

        with (
            patch(
                "src.services.inference_pipeline.save_prediction",
                new_callable=AsyncMock,
            ),
            patch(
                "src.services.inference_pipeline.AsyncSessionFactory"
            ) as mock_factory,
        ):
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(
                return_value=MagicMock(commit=AsyncMock(), rollback=AsyncMock())
            )
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_factory.return_value = mock_ctx

            svc = InferencePipelineService(stream, registry, alert)
            await svc._process(NORMAL_READING)

        # AlertService.process_prediction é sempre chamado; ele decide internamente
        # se vai fazer broadcast_alert ou não.
        alert.process_prediction.assert_awaited_once()
        payload = alert.process_prediction.call_args[0][0]
        assert payload["probability"] == 0.10

    @pytest.mark.asyncio
    async def test_process_prediction_payload_has_exactly_the_4_expected_keys(
        self,
    ) -> None:
        """RNF-64: as 4 chaves do payload comparadas como CONJUNTO — mata
        qualquer typo isolado em qualquer uma delas (`predicted_class`,
        `timestamp`, `inference_latency_ms`), não só `probability`."""
        registry = _make_registry(0.90)
        alert = _make_alert_service()
        stream = _make_stream_service([FAILURE_READING])

        with (
            patch(
                "src.services.inference_pipeline.save_prediction",
                new_callable=AsyncMock,
            ),
            patch(
                "src.services.inference_pipeline.AsyncSessionFactory"
            ) as mock_factory,
        ):
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(
                return_value=MagicMock(commit=AsyncMock(), rollback=AsyncMock())
            )
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_factory.return_value = mock_ctx

            svc = InferencePipelineService(stream, registry, alert)
            await svc._process(FAILURE_READING)

        payload = alert.process_prediction.call_args[0][0]
        assert set(payload.keys()) == {
            "probability",
            "predicted_class",
            "timestamp",
            "inference_latency_ms",
        }


class TestSensorBufferInjection:
    def test_explicitly_injected_empty_buffer_is_used_not_the_global_singleton(
        self,
    ) -> None:
        """Bug real achado nesta task (RNF-64): `sensor_buffer or
        get_sensor_buffer()` descartava um buffer recém-criado (vazio, logo
        falsy via `__len__`) e usava o singleton global em seu lugar —
        quebrando o isolamento que passar `sensor_buffer=` deveria garantir."""
        from src.services.feature_buffer import SensorBuffer, get_sensor_buffer

        fresh_buffer = SensorBuffer(window_size=5, warmup_size=1)
        stream = _make_stream_service([])
        registry = _make_registry(0.1)
        alert = _make_alert_service()

        svc = InferencePipelineService(
            stream, registry, alert, sensor_buffer=fresh_buffer
        )

        assert svc._buffer is fresh_buffer
        assert svc._buffer is not get_sensor_buffer()


class TestWarmLoggedFlag:
    @pytest.mark.asyncio
    async def test_warm_logged_flips_to_true_once_buffer_is_warm(self) -> None:
        """RNF-64: `_warm_logged` deve virar `True` (não `False`/`None`)
        assim que o buffer aquece, e é lido depois pra evitar log duplicado."""
        from src.services.feature_buffer import SensorBuffer

        registry = _make_registry(0.10)
        alert = _make_alert_service()
        stream = _make_stream_service([NORMAL_READING])
        tiny_buffer = SensorBuffer(window_size=1, warmup_size=1)

        svc = InferencePipelineService(
            stream, registry, alert, sensor_buffer=tiny_buffer
        )
        assert svc._warm_logged is False

        await svc._process(NORMAL_READING)

        assert svc._warm_logged is True
