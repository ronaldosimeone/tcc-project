"""
Testes de `POST /predict/batch` e `ModelService.predict_batch` — RNF-73.

Duas camadas:

1. `TestModelServicePredictBatch` — contra o artefato REAL `random_forest`
   (joblib): garante que o batch é genuinamente vetorizado (uma única
   chamada a `predict_proba`) e produz o MESMO resultado, amostra a
   amostra, que chamar `predict()` em loop — ordem preservada,
   `len(inputs) == len(outputs)`.

2. `TestPredictBatchEndpoint` — HTTP, modelo mockado (mesmo padrão de
   `test_predict_endpoint.py`): contrato (200/422), tamanhos de lote (1, 2,
   10, 100), casos de borda (vazio, >100, amostra inválida, dtype inválido).
"""

from __future__ import annotations

from typing import AsyncGenerator
from unittest.mock import MagicMock

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.main import create_app
from src.schemas.predict import PredictRequest
from src.services.model_service import (
    ModelService,
    get_model_service,
    load_model_by_name,
)

_VALID_PAYLOAD: dict = {
    "TP2": 5.02,
    "TP3": 9.21,
    "H1": 8.97,
    "DV_pressure": 2.10,
    "Reservoirs": 8.85,
    "Motor_current": 4.5,
    "Oil_temperature": 72.3,
    "COMP": 1.0,
    "DV_eletric": 0.0,
    "Towers": 1.0,
    "MPG": 1.0,
    "Oil_level": 1.0,
}


# ---------------------------------------------------------------------------
# 1. ModelService.predict_batch — artefato real (random_forest, joblib)
# ---------------------------------------------------------------------------


class TestModelServicePredictBatch:
    @staticmethod
    @pytest.fixture(scope="class")
    def service() -> ModelService:
        return load_model_by_name("random_forest")

    @staticmethod
    def _distinct_requests(n: int) -> list[PredictRequest]:
        """N requests distinguíveis (TP2 varia) — não N cópias idênticas,
        para garantir que uma implementação "errada" (ex.: só processa a
        1ª linha e repete) seria pega pelos testes de ordem/equivalência."""
        return [
            PredictRequest(**{**_VALID_PAYLOAD, "TP2": _VALID_PAYLOAD["TP2"] + i * 0.1})
            for i in range(n)
        ]

    def test_empty_batch_returns_empty_list(self, service: ModelService) -> None:
        assert service.predict_batch([]) == []

    @pytest.mark.parametrize("n", [1, 2, 10, 100])
    def test_batch_length_matches_input_length(
        self, service: ModelService, n: int
    ) -> None:
        requests = self._distinct_requests(n)
        results = service.predict_batch(requests)
        assert len(results) == n

    def test_batch_matches_individual_predict_calls_exactly(
        self, service: ModelService
    ) -> None:
        """RNF-73 §6 — predict_batch([s1..s100]) deve ser equivalente a
        [predict(s1), ..., predict(s100)]. Mesma computação determinística
        (mesmo modelo, mesmas features stateless) — espera-se igualdade
        EXATA, não apenas tolerância numérica."""
        requests = self._distinct_requests(100)
        batch_results = service.predict_batch(requests)
        individual_results = [service.predict(r) for r in requests]

        assert len(batch_results) == len(individual_results)
        for i, (b, ind) in enumerate(zip(batch_results, individual_results)):
            assert b.predicted_class == ind.predicted_class, f"index {i}"
            assert b.failure_probability == ind.failure_probability, f"index {i}"

    def test_output_order_matches_input_order(self, service: ModelService) -> None:
        """Posição i do output corresponde à posição i do input — sensores
        bem diferentes entre si (não só TP2) para que qualquer
        reordenação/mistura entre linhas mude o resultado observável."""
        extreme_low = {**_VALID_PAYLOAD, "TP2": 0.01, "TP3": 0.01, "H1": 0.01}
        extreme_high = {**_VALID_PAYLOAD, "TP2": 15.0, "TP3": 15.0, "H1": 15.0}
        requests = [
            PredictRequest(**extreme_low),
            PredictRequest(**extreme_high),
            PredictRequest(**extreme_low),
        ]
        results = service.predict_batch(requests)

        assert results[0].failure_probability == results[2].failure_probability
        # As duas amostras extremas e opostas não podem ter dado a mesma
        # probabilidade por coincidência real (modelo real, não mock).
        assert results[0].failure_probability != results[1].failure_probability

    def test_batch_never_calls_model_predict_proba_more_than_once(self) -> None:
        """Prova direta de vetorização — não um loop de N chamadas. Mocka
        só `predict_proba` (mantém `feature_names_in_` real) e conta
        chamadas para N=100 amostras."""
        real_model = load_model_by_name("random_forest")._model
        wrapped = MagicMock(wraps=real_model)
        wrapped.feature_names_in_ = real_model.feature_names_in_
        service = ModelService(model=wrapped)

        requests = self._distinct_requests(100)
        service.predict_batch(requests)

        assert wrapped.predict_proba.call_count == 1

    def test_predict_batch_reraises_the_original_error_never_swallows_it(self) -> None:
        """Mesma garantia de `predict_from_features` (ver
        test_model_service_unit.py::test_predict_from_features_reraises_the_original_error_never_swallows_it)
        aplicada ao caminho batch — erro do modelo propaga, não vira 500
        genérico silencioso nem lista vazia."""

        class _RaisingModel:
            feature_names_in_ = ["TP2"]

            def predict_proba(self, X):
                raise RuntimeError("boom")

        service = ModelService(model=_RaisingModel())
        with pytest.raises(RuntimeError, match="boom"):
            service.predict_batch(self._distinct_requests(3))


# ---------------------------------------------------------------------------
# 2. POST /predict/batch — HTTP, modelo mockado
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_service() -> ModelService:
    """MagicMock determinístico — probabilidade varia com TP2 (soma
    simples) para que testes de ordem consigam distinguir amostras."""
    mock_model = MagicMock()

    def _predict_proba(X):
        tp2_values = X["TP2"].to_numpy()
        probs = np.clip(tp2_values / 20.0, 0.0, 1.0)
        return np.column_stack([1 - probs, probs])

    mock_model.predict_proba.side_effect = _predict_proba
    mock_model.feature_names_in_ = np.array(
        list(_VALID_PAYLOAD.keys())
        + ["TP2_delta", "TP2_std_5", "TP2_ma_5", "TP2_ma_15"]
        + [
            f"{c}_std_5"
            for c in [
                "TP3",
                "H1",
                "DV_pressure",
                "Reservoirs",
                "Motor_current",
                "Oil_temperature",
            ]
        ]
        + [
            f"{c}_ma_5"
            for c in [
                "TP3",
                "H1",
                "DV_pressure",
                "Reservoirs",
                "Motor_current",
                "Oil_temperature",
            ]
        ]
        + [
            f"{c}_ma_15"
            for c in [
                "TP3",
                "H1",
                "DV_pressure",
                "Reservoirs",
                "Motor_current",
                "Oil_temperature",
            ]
        ]
        + ["TP2_TP3_diff", "TP2_TP3_ratio", "work_per_pressure", "reservoir_drop"],
        dtype=object,
    )
    return ModelService(model=mock_model)


@pytest.fixture()
def app_with_mock_model(mock_service: ModelService):
    application = create_app()
    application.dependency_overrides[get_model_service] = lambda: mock_service
    return application


@pytest_asyncio.fixture()
async def async_client(app_with_mock_model) -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=app_with_mock_model)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


class TestPredictBatchEndpoint:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("n", [1, 2, 10, 100])
    async def test_batch_sizes_return_200_with_matching_count(
        self, async_client: AsyncClient, n: int
    ) -> None:
        samples = [dict(_VALID_PAYLOAD) for _ in range(n)]
        response = await async_client.post("/predict/batch", json={"samples": samples})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["count"] == n
        assert len(body["predictions"]) == n

    @pytest.mark.asyncio
    async def test_empty_batch_returns_422(self, async_client: AsyncClient) -> None:
        response = await async_client.post("/predict/batch", json={"samples": []})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_101_samples_returns_422(self, async_client: AsyncClient) -> None:
        samples = [dict(_VALID_PAYLOAD) for _ in range(101)]
        response = await async_client.post("/predict/batch", json={"samples": samples})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_missing_field_in_one_sample_returns_422(
        self, async_client: AsyncClient
    ) -> None:
        bad_sample = {k: v for k, v in _VALID_PAYLOAD.items() if k != "TP2"}
        samples = [dict(_VALID_PAYLOAD), bad_sample]
        response = await async_client.post("/predict/batch", json={"samples": samples})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_invalid_dtype_in_one_sample_returns_422(
        self, async_client: AsyncClient
    ) -> None:
        bad_sample = {**_VALID_PAYLOAD, "TP2": "not-a-number"}
        samples = [dict(_VALID_PAYLOAD), bad_sample]
        response = await async_client.post("/predict/batch", json={"samples": samples})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_missing_samples_key_returns_422(
        self, async_client: AsyncClient
    ) -> None:
        response = await async_client.post("/predict/batch", json={})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_output_order_matches_input_order(
        self, async_client: AsyncClient
    ) -> None:
        """3 amostras com TP2 bem diferentes — a ordem da resposta precisa
        bater com a ordem do request (mock varia a probabilidade com TP2,
        então isso prova posição i -> i, não só contagem)."""
        samples = [
            {**_VALID_PAYLOAD, "TP2": 1.0},
            {**_VALID_PAYLOAD, "TP2": 10.0},
            {**_VALID_PAYLOAD, "TP2": 19.0},
        ]
        response = await async_client.post("/predict/batch", json={"samples": samples})
        assert response.status_code == 200
        probs = [p["failure_probability"] for p in response.json()["predictions"]]
        assert probs[0] < probs[1] < probs[2]

    @pytest.mark.asyncio
    async def test_returns_503_when_model_not_loaded(self) -> None:
        from src.core.exceptions import ModelNotAvailableError

        app_no_model = create_app()
        app_no_model.dependency_overrides[get_model_service] = lambda: (
            _ for _ in ()
        ).throw(ModelNotAvailableError())

        transport = ASGITransport(app=app_no_model)
        async with AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            response = await client.post(
                "/predict/batch", json={"samples": [dict(_VALID_PAYLOAD)]}
            )

        assert response.status_code == 503
