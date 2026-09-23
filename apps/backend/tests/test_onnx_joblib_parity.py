"""
Paridade Joblib x ONNX — RNF-72 Fase 4.

Auditoria (ver benchmark_onnx_vs_joblib.py e RELATORIO-RNF-72-RNF-73.md):
``random_forest_v2.onnx``/``xgboost_v2.onnx`` NÃO são modelos diferentes —
são o MESMO ``RandomForestClassifier``/``XGBClassifier`` treinado por
``train_random_forest.py``/``train_xgboost.py``, exportado nos dois
formatos na MESMA execução (mesmo ``model_card.json``, mesmo
``feature_count``/``feature_names``). Estes testes comparam as predições
dos dois formatos sobre amostras REAIS do dataset MetroPT-3 (não
sintéticas), com tolerância numérica documentada (não igualdade textual).

Tolerância (rtol=1e-3, atol=1e-4) — justificativa
--------------------------------------------------
ONNX Runtime computa em float32 (ver ``OnnxTreeAdapter._extract_probabilities``,
dtype=np.float32); sklearn/xgboost computam em float64. Medido empiricamente
neste módulo (500 amostras reais, ``test_random_forest_max_diff_is_small``/
``test_xgboost_max_diff_is_small``): diferença máxima absoluta observada
~1.3e-7 (RF) e ~6.6e-8 (XGB) — consistente com o epsilon de máquina do
float32 (~1.19e-7), não com um erro estrutural de conversão. rtol=1e-3/
atol=1e-4 dá ~1000x de margem sobre o que foi observado — generoso o
bastante para não quebrar por ruído de plataforma, apertado o bastante para
pegar uma divergência REAL de conversão.

XGBoost V1 (joblib) — achado pré-existente (já documentado em
test_model_service_real_artifacts.py e PENDENCIAS.md, RE-CONFIRMADO aqui,
não descoberto por esta task): ``xgboost_v1.joblib`` foi treinado em
ndarray puro (sem feature_names, exigência do conversor onnxmltools —
ver train_xgboost.py) e não tem ``feature_names_in_`` — ``ModelService``
não o instancia. Os testes de XGBoost aqui carregam o joblib diretamente
(``joblib.load``) e reordenam colunas pelo `xgboost_v1_card.json`
(mesma ordem gravada no treino) — não reimplementa preprocessing, reusa
``ModelService._build_feature_row`` (função pura).
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.core.config import settings
from src.schemas.predict import PredictRequest
from src.services.model_service import load_model_by_name

_RAW_SENSOR_COLS: list[str] = [
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Motor_current",
    "Oil_temperature",
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "Oil_level",
]

_RTOL: float = 1e-3
_ATOL: float = 1e-4
_N_SAMPLES: int = 500


def _load_real_requests(n: int) -> list[PredictRequest]:
    """``n`` linhas reais, igualmente espaçadas (determinístico, sem
    embaralhar) — mesma técnica de benchmark_onnx_vs_joblib.py. Usa
    ``settings.simulator_parquet_path`` (mesmo path canônico já usado em
    produção e por outros testes, ex. test_drift_monitor.py) em vez de
    recalcular o caminho manualmente a partir de ``__file__``."""
    df = pd.read_parquet(settings.simulator_parquet_path, columns=_RAW_SENSOR_COLS)
    idx = np.linspace(0, len(df) - 1, num=n, dtype=int)
    rows = df.iloc[idx].reset_index(drop=True)
    return [
        PredictRequest(**{col: float(row[col]) for col in _RAW_SENSOR_COLS})
        for _, row in rows.iterrows()
    ]


@pytest.fixture(scope="module")
def real_requests() -> list[PredictRequest]:
    return _load_real_requests(_N_SAMPLES)


@pytest.fixture(scope="module")
def edge_case_requests() -> dict[str, PredictRequest]:
    """Casos de borda pedidos pelo RNF-72 §Fase 4 — valores normais já
    cobertos por `real_requests`; aqui: próximo ao threshold e extremos
    válidos dentro do schema (Pydantic já rejeita NaN/inf — ver
    test_predict_endpoint.py::test_predict_non_numeric_value_returns_422,
    então "aceitar ou rejeitar" está coberto lá, não duplicado aqui)."""
    base = {
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
    return {
        "all_zero_digital": PredictRequest(
            **{
                **base,
                "COMP": 0.0,
                "DV_eletric": 0.0,
                "Towers": 0.0,
                "MPG": 0.0,
                "Oil_level": 0.0,
            }
        ),
        "all_one_digital": PredictRequest(
            **{
                **base,
                "COMP": 1.0,
                "DV_eletric": 1.0,
                "Towers": 1.0,
                "MPG": 1.0,
                "Oil_level": 1.0,
            }
        ),
        "high_pressure": PredictRequest(
            **{**base, "TP2": 10.5, "TP3": 11.2, "H1": 10.9}
        ),
        "low_pressure": PredictRequest(
            **{**base, "TP2": 0.01, "TP3": 0.02, "H1": 0.01}
        ),
        "zero_current": PredictRequest(**{**base, "Motor_current": 0.0}),
    }


# ---------------------------------------------------------------------------
# Random Forest — caminho limpo (ModelService funciona para os dois formatos)
# ---------------------------------------------------------------------------


class TestRandomForestParity:
    @staticmethod
    @pytest.fixture(scope="class")
    def joblib_service():
        return load_model_by_name("random_forest")

    @staticmethod
    @pytest.fixture(scope="class")
    def onnx_service():
        return load_model_by_name("random_forest_v2")

    def test_predicted_class_matches_on_all_real_samples(
        self, joblib_service, onnx_service, real_requests: list[PredictRequest]
    ) -> None:
        mismatches = [
            i
            for i, req in enumerate(real_requests)
            if joblib_service.predict(req).predicted_class
            != onnx_service.predict(req).predicted_class
        ]
        assert mismatches == [], (
            f"{len(mismatches)}/{len(real_requests)} amostras com classe "
            f"divergente (índices: {mismatches[:10]}...)"
        )

    def test_probabilities_allclose_on_all_real_samples(
        self, joblib_service, onnx_service, real_requests: list[PredictRequest]
    ) -> None:
        probs_joblib = np.array(
            [joblib_service.predict(r).failure_probability for r in real_requests]
        )
        probs_onnx = np.array(
            [onnx_service.predict(r).failure_probability for r in real_requests]
        )
        np.testing.assert_allclose(probs_joblib, probs_onnx, rtol=_RTOL, atol=_ATOL)

    def test_random_forest_max_diff_is_small(
        self, joblib_service, onnx_service, real_requests: list[PredictRequest]
    ) -> None:
        """Evidência empírica que justifica a tolerância documentada no
        docstring do módulo — falha se a divergência real crescer muito
        além do que foi medido (regressão de precisão da conversão)."""
        probs_joblib = np.array(
            [joblib_service.predict(r).failure_probability for r in real_requests]
        )
        probs_onnx = np.array(
            [onnx_service.predict(r).failure_probability for r in real_requests]
        )
        max_diff = float(np.max(np.abs(probs_joblib - probs_onnx)))
        assert (
            max_diff < 1e-4
        ), f"max_diff={max_diff} — divergência maior que o esperado"

    def test_edge_cases_predicted_class_matches(
        self,
        joblib_service,
        onnx_service,
        edge_case_requests: dict[str, PredictRequest],
    ) -> None:
        for name, req in edge_case_requests.items():
            c_joblib = joblib_service.predict(req).predicted_class
            c_onnx = onnx_service.predict(req).predicted_class
            assert c_joblib == c_onnx, f"caso '{name}': joblib={c_joblib} onnx={c_onnx}"

    def test_edge_cases_probabilities_allclose(
        self,
        joblib_service,
        onnx_service,
        edge_case_requests: dict[str, PredictRequest],
    ) -> None:
        for name, req in edge_case_requests.items():
            p_joblib = joblib_service.predict(req).failure_probability
            p_onnx = onnx_service.predict(req).failure_probability
            assert p_joblib == pytest.approx(p_onnx, rel=_RTOL, abs=_ATOL), name


# ---------------------------------------------------------------------------
# XGBoost — V1 joblib carregado direto (ver docstring do módulo)
# ---------------------------------------------------------------------------


class TestXGBoostParity:
    @staticmethod
    @pytest.fixture(scope="class")
    def raw_joblib_model():
        return joblib.load(settings.xgboost_model_path)

    @staticmethod
    @pytest.fixture(scope="class")
    def feature_names() -> list[str]:
        card_path = settings.model_path.parent / "xgboost_v1_card.json"
        card = json.loads(card_path.read_text(encoding="utf-8"))
        return card["feature_names"]

    @staticmethod
    @pytest.fixture(scope="class")
    def onnx_service():
        return load_model_by_name("xgboost_v2")

    @staticmethod
    @pytest.fixture(scope="class")
    def feature_builder():
        # `_build_feature_row` é pure function (não usa self._model) —
        # reusa uma instância RF só como "dono" do método, mesmo padrão do
        # benchmark_onnx_vs_joblib.py.
        return load_model_by_name("random_forest")

    def _raw_probability(
        self, raw_model, feature_builder, feature_names: list[str], req: PredictRequest
    ) -> float:
        """
        Só a probabilidade bruta — NÃO compara `predicted_class` aqui: o
        joblib V1 bruto não tem o `decision_threshold` tunado do model card
        (esse threshold é aplicado por `ModelService`, que não conseguimos
        instanciar para este artefato — ver docstring do módulo). Comparar
        classes usando thresholds diferentes (0.5 bruto vs. tunado do
        onnx_service) seria uma comparação inválida, não uma paridade real.
        """
        X = feature_builder._build_feature_row(req)
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0.0
        X = X[feature_names].to_numpy(dtype=np.float32)
        return float(raw_model.predict_proba(X)[0][1])

    def test_probabilities_allclose_on_all_real_samples(
        self,
        raw_joblib_model,
        feature_builder,
        feature_names: list[str],
        onnx_service,
        real_requests: list[PredictRequest],
    ) -> None:
        probs_joblib = np.array(
            [
                self._raw_probability(
                    raw_joblib_model, feature_builder, feature_names, r
                )
                for r in real_requests
            ]
        )
        probs_onnx = np.array(
            [onnx_service.predict(r).failure_probability for r in real_requests]
        )
        np.testing.assert_allclose(probs_joblib, probs_onnx, rtol=_RTOL, atol=_ATOL)

    def test_xgboost_max_diff_is_small(
        self,
        raw_joblib_model,
        feature_builder,
        feature_names: list[str],
        onnx_service,
        real_requests: list[PredictRequest],
    ) -> None:
        probs_joblib = np.array(
            [
                self._raw_probability(
                    raw_joblib_model, feature_builder, feature_names, r
                )
                for r in real_requests
            ]
        )
        probs_onnx = np.array(
            [onnx_service.predict(r).failure_probability for r in real_requests]
        )
        max_diff = float(np.max(np.abs(probs_joblib - probs_onnx)))
        assert (
            max_diff < 1e-4
        ), f"max_diff={max_diff} — divergência maior que o esperado"
