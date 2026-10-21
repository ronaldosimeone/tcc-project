"""
Testes de integração real — `load_model_by_name` + adapters ONNX (RNF-57).

Carrega de verdade os 7 artefatos SERVIDOS (geração v3, protocolo aninhado
sem vazamento, `apps/ml/src/train_production.py`) — ONNX Runtime e card reais,
sem mock — e confere que cada um produz uma predição válida. Gerações
anteriores (v1 split aleatório + SMOTE, v2 árvores) foram retiradas do
serving e precisam ser recusadas.
"""

from __future__ import annotations

import pytest

from src.core.config import settings
from src.schemas.predict import PredictRequest, PredictResponse
from src.services.model_service import SERVED_MODELS, ModelService, load_model_by_name

# Repetida (não importada) para o teste falhar se a lista servida mudar sem
# cobertura de carregamento real.
_SERVED: tuple[str, ...] = (
    "random_forest_v3",
    "xgboost_v3",
    "mlp_v3",
    "tcn_v3",
    "bilstm_v3",
    "patchtst_v3",
    "autoencoder_v3",
)

# Snapshot realista — operação normal, não uma janela de falha.
_SAMPLE_REQUEST = PredictRequest(
    TP2=5.02,
    TP3=9.21,
    H1=8.97,
    DV_pressure=2.10,
    Reservoirs=8.85,
    Motor_current=4.5,
    Oil_temperature=72.3,
    COMP=1.0,
    DV_eletric=0.0,
    Towers=1.0,
    MPG=1.0,
    Oil_level=1.0,
)


def test_served_list_is_the_v3_generation() -> None:
    assert SERVED_MODELS == _SERVED


@pytest.mark.parametrize("model_name", _SERVED)
def test_load_model_by_name_loads_real_artifact_and_predicts(model_name: str) -> None:
    if not (settings.models_dir / f"{model_name}_card.json").exists():
        pytest.skip(f"{model_name} not trained yet (run src.train_production)")
    service = load_model_by_name(model_name)
    assert isinstance(service, ModelService)
    # RNF-76 — label `model` das métricas reflete o nome REAL carregado.
    assert service._model_name == model_name  # noqa: SLF001 — integração
    assert 0.0 < service.decision_threshold < 1.0

    response = service.predict(_SAMPLE_REQUEST)
    assert isinstance(response, PredictResponse)
    assert response.predicted_class in (0, 1)
    assert 0.0 <= response.failure_probability <= 1.0


@pytest.mark.parametrize(
    "legacy", ["random_forest", "random_forest_v2", "xgboost_v2", "tcn", "mlp_v2"]
)
def test_legacy_generations_are_not_served(legacy: str) -> None:
    with pytest.raises(ValueError):
        load_model_by_name(legacy)
