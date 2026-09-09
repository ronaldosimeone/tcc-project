"""
Testes de integração real — `load_model_by_name` + adapters ONNX (RNF-57).

Auditoria (RNF-56/57): `load_model_by_name` e os 4 adapters ONNX
(`OnnxMlpAdapter`, `OnnxTreeAdapter`, `OnnxSequenceAdapter`,
`OnnxAutoencoderAdapter`) tinham cobertura baixa/zero (`model_service.py`
35%, adapters 0-35%) porque TODOS os testes existentes (`test_model_registry.py`,
`test_predict_endpoint.py`) fazem mock de `load_model_by_name`/do modelo em
si — nenhum teste carregava os artefatos ONNX/joblib REAIS de ponta a
ponta. Este arquivo fecha essa lacuna com os 9 artefatos REAIS já
versionados no repositório (`git ls-files apps/ml/models/` confirma: os 9
existem — nenhuma dependência externa nova, nenhum dado inventado).

Não mocka o Evidently/Ollama/MCP/Telegram (não relacionados); não mocka o
ONNX Runtime nem os artefatos — carrega e executa de verdade.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.config import settings
from src.schemas.predict import PredictRequest, PredictResponse
from src.services.model_service import ModelService, load_model, load_model_by_name

# Os 9 nomes conhecidos (RF-10) — mesma lista de `ModelRegistry.KNOWN_MODELS`,
# repetida aqui (não importada) para o teste falhar explicitamente se um
# nome novo for adicionado ao registry sem cobertura de carregamento real.
#
# "xgboost" (V1, joblib nativo) marcado xfail(strict=True) — achado REAL,
# pré-existente, descoberto por este teste (nenhum teste anterior carregava
# o artefato de verdade, todos mockavam `load_model_by_name`): o joblib
# `xgboost_v1.joblib` foi salvo sem feature names embutidos no booster
# (`XGBClassifier.get_booster().feature_names is None`), então
# `ModelService.__init__` (`list(self._model.feature_names_in_)`) levanta
# `AttributeError` — ou seja, `ACTIVE_MODEL=xgboost` quebraria a aplicação
# no startup HOJE, em produção real. Não corrigido aqui (exigiria
# re-treinar/re-exportar o artefato — fora do escopo de RNF-56/57, que
# proíbe alterar o modelo preditivo); documentado em PENDENCIAS.md.
_ALL_MODEL_NAMES: list[str] = [
    "random_forest",
    pytest.param(
        "xgboost",
        marks=pytest.mark.xfail(
            reason=(
                "Pré-existente: xgboost_v1.joblib sem feature_names_in_ "
                "utilizável — ver PENDENCIAS.md"
            ),
            strict=True,
        ),
    ),
    "mlp",
    "random_forest_v2",
    "xgboost_v2",
    "tcn",
    "bilstm",
    "patchtst",
    "autoencoder",
]

# Snapshot realista — mesmos valores de `_VALID_PAYLOAD` em
# `test_predictions_endpoint.py` (operação normal, não uma janela de falha).
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


# ---------------------------------------------------------------------------
# `load_model_by_name` — os 9 modelos reais, artefatos versionados no repo
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_name", _ALL_MODEL_NAMES)
def test_load_model_by_name_loads_real_artifact_and_predicts(
    model_name: str,
) -> None:
    """Cada um dos 9 modelos conhecidos carrega de verdade (ONNX Runtime ou
    joblib) e produz uma predição válida a partir de um snapshot real."""
    service = load_model_by_name(model_name)
    assert isinstance(service, ModelService)

    response = service.predict(_SAMPLE_REQUEST)
    assert isinstance(response, PredictResponse)
    assert response.predicted_class in (0, 1)
    assert 0.0 <= response.failure_probability <= 1.0


def test_load_model_by_name_random_forest_v2_uses_80_features() -> None:
    """RF V2 (cross-features + lags) — confirma o card real de 80 features
    (mesmo número visto no log de boot do container, RF-27)."""
    service = load_model_by_name("random_forest_v2")
    assert len(service._expected_features) == 80  # noqa: SLF001 — teste de integração


def test_load_model_by_name_xgboost_v2_uses_zipmap_output() -> None:
    """XGB V2 (onnxmltools/ZipMap) exercita o Pass 2 de
    `OnnxTreeAdapter._extract_probabilities` (lista de dicts), diferente do
    Pass 1 (ndarray) exercitado por `random_forest_v2` acima — as duas
    branches reais do parser são cobertas por artefatos genuinamente
    diferentes, não por um mock construído para forçar cada branch."""
    service = load_model_by_name("xgboost_v2")
    response = service.predict(_SAMPLE_REQUEST)
    assert 0.0 <= response.failure_probability <= 1.0


def test_load_model_by_name_unknown_name_falls_back_to_default_registry() -> None:
    """`_MODEL_REGISTRY.get(name, settings.model_path)` — nome desconhecido
    não levanta erro, cai para o Random Forest V1 default (comportamento
    real documentado no código, não assumido)."""
    service = load_model_by_name("this-model-does-not-exist")
    response = service.predict(_SAMPLE_REQUEST)
    assert response.predicted_class in (0, 1)


def test_load_model_missing_artifact_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "does-not-exist.joblib")


def test_load_model_falls_back_to_alternate_filename(tmp_path: Path) -> None:
    """`load_model` tenta `random_forest_final.joblib` quando o path
    configurado não existe — mesmo artefato real usado em produção,
    apontado por um path alternativo forjado no teste."""
    import shutil

    fake_dir = tmp_path
    shutil.copy(settings.model_path, fake_dir / "random_forest_final.joblib")

    service = load_model(fake_dir / "nome-que-nao-existe.joblib")
    response = service.predict(_SAMPLE_REQUEST)
    assert response.predicted_class in (0, 1)
