"""
Testes unitários dos helpers puros de `model_service` (RNF-64/RNF-65).

Auditoria desta task: 83% de cobertura, concentrada em três áreas nunca
exercitadas diretamente — `test_model_service_real_artifacts.py` só cobre o
caminho feliz via artefatos reais:

  1. `_read_model_card`/`_resolve_threshold`/`_resolve_feature_names` — os
     fallbacks quando o card falta, tem JSON inválido, ou tem
     `decision_threshold`/`feature_names` malformado (fora de (0,1),
     não-numérico, não-lista, vazio).
  2. `_load_sequential_model`/`_load_autoencoder_model` — os ramos que leem
     `window_size`/`feature_names`/`mse_threshold` do card vs. os fallbacks
     quando o card falta ou o campo está ausente/mal-formado.
  3. `ModelService.decision_threshold` (property) e o `except Exception` de
     `predict_from_features` (relança após logar — nunca engole o erro).
  4. `get_model_service` — as 3 ramificações da resolução via
     `app.state.model_registry` vs. `app.state.model_service` (legado) vs.
     nenhum dos dois (`ModelNotAvailableError`).

Usa `tmp_path` + monkeypatch em `settings.model_path` pra isolar
`_read_model_card` do disco real, e fakes mínimos (não mocks genéricos) pros
adapters ONNX — nenhum destes testes toca ONNX Runtime.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import src.services.model_service as model_service_module
from src.core import metrics as metrics_module
from src.core.config import settings
from src.core.exceptions import ModelNotAvailableError
from src.schemas.predict import PredictRequest
from src.services.model_service import (
    ModelService,
    _load_autoencoder_model,
    _load_sequential_model,
    _read_model_card,
    _resolve_feature_names,
    _resolve_threshold,
    get_model_service,
    load_model_by_name,
)


def _point_model_path_at(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        model_service_module.settings, "model_path", tmp_path / "model.joblib"
    )


# ---------------------------------------------------------------------------
# Constantes de módulo — RNF-64: mutmut mutou cada chave/valor destes dicts
# e listas individualmente (ex.: "xgboost" -> "XXxgboostXX") e sobreviveu,
# porque nenhum teste comparava o dict/lista INTEIRO. Uma comparação exata
# (não "contém"/"len ==") mata todas as variações de uma vez: qualquer
# chave/valor trocado, removido ou com typo faz a igualdade falhar.
# ---------------------------------------------------------------------------


def test_binary_cols_is_the_exact_expected_list() -> None:
    assert model_service_module._BINARY_COLS == [  # noqa: SLF001
        "COMP",
        "DV_eletric",
        "Towers",
        "MPG",
        "Oil_level",
    ]


def test_model_cards_maps_every_model_name_to_its_exact_card_filename() -> None:
    assert model_service_module._MODEL_CARDS == {  # noqa: SLF001
        "random_forest": "model_card.json",
        "xgboost": "xgboost_v1_card.json",
        "mlp": "mlp_v1_card.json",
        "random_forest_v2": "model_card.json",
        "xgboost_v2": "xgboost_v1_card.json",
        "tcn": "tcn_v1_card.json",
        "bilstm": "bilstm_v1_card.json",
        "patchtst": "patchtst_v1_card.json",
        "autoencoder": "autoencoder_v1_card.json",
    }


def test_model_registry_maps_v1_names_to_the_configured_artefact_paths() -> None:
    assert model_service_module._MODEL_REGISTRY == {  # noqa: SLF001
        "random_forest": settings.model_path,
        "xgboost": settings.xgboost_model_path,
    }


# ---------------------------------------------------------------------------
# _read_model_card
# ---------------------------------------------------------------------------


def test_read_model_card_returns_none_for_unknown_model_name() -> None:
    assert _read_model_card("does-not-exist") is None


def test_read_model_card_returns_none_when_file_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _point_model_path_at(monkeypatch, tmp_path)
    assert _read_model_card("random_forest") is None


def test_read_model_card_returns_none_on_malformed_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _point_model_path_at(monkeypatch, tmp_path)
    (tmp_path / "model_card.json").write_text("{not valid json", encoding="utf-8")
    assert _read_model_card("random_forest") is None


def test_read_model_card_parses_valid_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _point_model_path_at(monkeypatch, tmp_path)
    (tmp_path / "model_card.json").write_text(
        json.dumps({"decision_threshold": 0.3}), encoding="utf-8"
    )
    assert _read_model_card("random_forest") == {"decision_threshold": 0.3}


# ---------------------------------------------------------------------------
# _resolve_threshold
# ---------------------------------------------------------------------------


def test_resolve_threshold_defaults_to_0_5_when_card_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _point_model_path_at(monkeypatch, tmp_path)
    assert _resolve_threshold("random_forest") == 0.5


@pytest.mark.parametrize("raw", ["not-a-number", None, [1, 2]])
def test_resolve_threshold_falls_back_when_value_not_coercible_to_float(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, raw: Any
) -> None:
    _point_model_path_at(monkeypatch, tmp_path)
    (tmp_path / "model_card.json").write_text(
        json.dumps({"decision_threshold": raw}), encoding="utf-8"
    )
    assert _resolve_threshold("random_forest") == 0.5


@pytest.mark.parametrize("raw", [0.0, 1.0, -0.1, 1.5])
def test_resolve_threshold_clamps_out_of_range_values_to_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, raw: float
) -> None:
    """Fronteira: a regra real é `0.0 < threshold < 1.0` (exclusiva nas duas
    pontas) — 0.0 e 1.0 são REJEITADOS, não aceitos como limite."""
    _point_model_path_at(monkeypatch, tmp_path)
    (tmp_path / "model_card.json").write_text(
        json.dumps({"decision_threshold": raw}), encoding="utf-8"
    )
    assert _resolve_threshold("random_forest") == 0.5


def test_resolve_threshold_accepts_value_strictly_between_0_and_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _point_model_path_at(monkeypatch, tmp_path)
    (tmp_path / "model_card.json").write_text(
        json.dumps({"decision_threshold": 0.37}), encoding="utf-8"
    )
    assert _resolve_threshold("random_forest") == 0.37


def test_resolve_threshold_uses_default_when_field_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _point_model_path_at(monkeypatch, tmp_path)
    (tmp_path / "model_card.json").write_text(json.dumps({}), encoding="utf-8")
    assert _resolve_threshold("random_forest") == 0.5


# ---------------------------------------------------------------------------
# _resolve_feature_names
# ---------------------------------------------------------------------------


def test_resolve_feature_names_raises_file_not_found_when_card_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Mensagem EXATA (não `match=` parcial): RNF-64 achado — o mutador de
    string do mutmut embrulha a mensagem inteira em `"XX...XX"`, o que
    preserva qualquer substring que um `match=` parcial procurasse. Só uma
    comparação de string completa mata essa classe de mutante."""
    _point_model_path_at(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError) as exc_info:
        _resolve_feature_names("random_forest_v2")
    assert str(exc_info.value) == (
        "Model card for 'random_forest_v2' is missing — cannot resolve "
        "feature_names for the ONNX adapter."
    )


@pytest.mark.parametrize("bad_names", [None, "not-a-list", [], 42])
def test_resolve_feature_names_raises_value_error_when_field_unusable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bad_names: Any
) -> None:
    _point_model_path_at(monkeypatch, tmp_path)
    (tmp_path / "model_card.json").write_text(
        json.dumps({"feature_names": bad_names}), encoding="utf-8"
    )
    with pytest.raises(ValueError) as exc_info:
        _resolve_feature_names("random_forest_v2")
    assert str(exc_info.value) == (
        "Model card for 'random_forest_v2' has no usable 'feature_names' field."
    )


def test_resolve_feature_names_returns_names_as_strings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _point_model_path_at(monkeypatch, tmp_path)
    (tmp_path / "model_card.json").write_text(
        json.dumps({"feature_names": ["TP2", "TP3"]}), encoding="utf-8"
    )
    assert _resolve_feature_names("random_forest_v2") == ["TP2", "TP3"]


# ---------------------------------------------------------------------------
# _load_sequential_model / _load_autoencoder_model — ramos guiados pelo card
# ---------------------------------------------------------------------------


class _FakeAdapter:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.feature_names_in_ = kwargs.get("channel_names") or ["TP2"]


def test_load_sequential_model_uses_card_window_size_and_channel_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        model_service_module,
        "_read_model_card",
        lambda name: {
            "inference": {"window_size": 42},
            "feature_names": ["TP2", "TP3"],
        },
    )
    monkeypatch.setattr(
        "src.services.onnx_sequence_adapter.OnnxSequenceAdapter", _FakeAdapter
    )
    service = _load_sequential_model("tcn", threshold=0.5)
    assert service._model.kwargs["window_size"] == 42  # noqa: SLF001
    assert service._model.kwargs["channel_names"] == ["TP2", "TP3"]  # noqa: SLF001


def test_load_sequential_model_falls_back_to_default_window_size_when_card_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_service_module, "_read_model_card", lambda name: None)
    monkeypatch.setattr(
        "src.services.onnx_sequence_adapter.OnnxSequenceAdapter", _FakeAdapter
    )
    service = _load_sequential_model("bilstm", threshold=0.5)
    assert service._model.kwargs["window_size"] == 60  # noqa: SLF001
    assert service._model.kwargs["channel_names"] is None  # noqa: SLF001


def test_load_sequential_model_ignores_non_list_feature_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        model_service_module,
        "_read_model_card",
        lambda name: {"feature_names": "not-a-list"},
    )
    monkeypatch.setattr(
        "src.services.onnx_sequence_adapter.OnnxSequenceAdapter", _FakeAdapter
    )
    service = _load_sequential_model("patchtst", threshold=0.5)
    assert service._model.kwargs["channel_names"] is None  # noqa: SLF001


def test_load_sequential_model_ignores_an_empty_feature_names_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fronteira RNF-64: `isinstance(x, list) and x` — uma lista VAZIA é
    `isinstance` True mas falsy, então a condição real deve ser False (não
    setar `channel_names=[]`). Um mutante `and`->`or` setaria `[]` aqui,
    porque `isinstance([], list)` sozinho já é True e o `or` nunca avalia o
    segundo operando."""
    monkeypatch.setattr(
        model_service_module, "_read_model_card", lambda name: {"feature_names": []}
    )
    monkeypatch.setattr(
        "src.services.onnx_sequence_adapter.OnnxSequenceAdapter", _FakeAdapter
    )
    service = _load_sequential_model("patchtst", threshold=0.5)
    assert service._model.kwargs["channel_names"] is None  # noqa: SLF001


def test_load_autoencoder_model_raises_when_card_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_service_module, "_read_model_card", lambda name: None)
    with pytest.raises(FileNotFoundError) as exc_info:
        _load_autoencoder_model(threshold=0.5)
    assert str(exc_info.value) == (
        "autoencoder_v1_card.json not found — run `python src/train_autoencoder.py` first."
    )


def test_load_autoencoder_model_raises_when_mse_threshold_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_service_module, "_read_model_card", lambda name: {})
    with pytest.raises(ValueError) as exc_info:
        _load_autoencoder_model(threshold=0.5)
    assert str(exc_info.value) == (
        "autoencoder_v1_card.json is missing 'mse_threshold'. "
        "Re-run `python src/train_autoencoder.py` to regenerate the card."
    )


def test_load_autoencoder_model_falls_back_to_default_window_size_when_card_has_no_inference_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fronteira RNF-64: o card SÓ precisa de `mse_threshold` pra ser válido —
    sem a chave `inference`, `window_size` deve cair no default 60 (não 0,
    não None, não outro número)."""
    monkeypatch.setattr(
        model_service_module, "_read_model_card", lambda name: {"mse_threshold": 0.02}
    )
    monkeypatch.setattr(
        "src.services.onnx_autoencoder_adapter.OnnxAutoencoderAdapter", _FakeAdapter
    )
    service = _load_autoencoder_model(threshold=0.5)
    assert service._model.kwargs["window_size"] == 60  # noqa: SLF001
    assert service._model.kwargs["channel_names"] is None  # noqa: SLF001


def test_load_autoencoder_model_uses_card_window_size_and_channel_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        model_service_module,
        "_read_model_card",
        lambda name: {
            "mse_threshold": 0.02,
            "inference": {"window_size": 30},
            "feature_names": ["TP2"],
        },
    )
    monkeypatch.setattr(
        "src.services.onnx_autoencoder_adapter.OnnxAutoencoderAdapter", _FakeAdapter
    )
    service = _load_autoencoder_model(threshold=0.5)
    assert service._model.kwargs["window_size"] == 30  # noqa: SLF001
    assert service._model.kwargs["mse_threshold"] == 0.02  # noqa: SLF001
    assert service._model.kwargs["channel_names"] == ["TP2"]  # noqa: SLF001


# ---------------------------------------------------------------------------
# load_model_by_name — dispatch por nome (RNF-64: cada ramo `if model_name
# == "..."` / `in {...}` tem seu literal testado individualmente; um typo
# em QUALQUER um deles faz cair no ramo errado ou no fallback V1).
# ---------------------------------------------------------------------------


def test_load_model_by_name_dispatches_mlp_to_the_mlp_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_service_module, "_resolve_threshold", lambda name: 0.5)
    monkeypatch.setattr("src.services.mlp_adapter.OnnxMlpAdapter", _FakeAdapter)
    service = load_model_by_name("mlp")
    assert set(service._model.kwargs) == {"onnx_path", "scaler_path"}  # noqa: SLF001


@pytest.mark.parametrize(
    ("model_name", "expected_onnx_path"),
    [
        ("random_forest_v2", settings.rf_v2_onnx_path),
        ("xgboost_v2", settings.xgboost_v2_onnx_path),
    ],
)
def test_load_model_by_name_dispatches_v2_tree_models_to_the_right_onnx_path(
    monkeypatch: pytest.MonkeyPatch, model_name: str, expected_onnx_path: Path
) -> None:
    monkeypatch.setattr(model_service_module, "_resolve_threshold", lambda name: 0.5)
    monkeypatch.setattr(
        model_service_module, "_resolve_feature_names", lambda name: ["TP2"]
    )
    monkeypatch.setattr("src.services.onnx_tree_adapter.OnnxTreeAdapter", _FakeAdapter)
    service = load_model_by_name(model_name)
    assert service._model.kwargs["onnx_path"] == expected_onnx_path  # noqa: SLF001


@pytest.mark.parametrize("model_name", ["tcn", "bilstm", "patchtst"])
def test_load_model_by_name_dispatches_sequential_models_by_exact_name(
    monkeypatch: pytest.MonkeyPatch, model_name: str
) -> None:
    monkeypatch.setattr(model_service_module, "_resolve_threshold", lambda name: 0.5)
    calls: list[str] = []

    def _fake_load_sequential(name: str, threshold: float) -> ModelService:
        calls.append(name)
        return ModelService(_MinimalModel())

    monkeypatch.setattr(
        model_service_module, "_load_sequential_model", _fake_load_sequential
    )
    load_model_by_name(model_name)
    assert calls == [model_name]


def test_load_model_by_name_dispatches_autoencoder_by_exact_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_service_module, "_resolve_threshold", lambda name: 0.5)
    calls: list[str] = []

    def _fake_load_autoencoder(threshold: float) -> ModelService:
        calls.append("autoencoder")
        return ModelService(_MinimalModel())

    monkeypatch.setattr(
        model_service_module, "_load_autoencoder_model", _fake_load_autoencoder
    )
    load_model_by_name("autoencoder")
    assert calls == ["autoencoder"]


def test_load_model_by_name_logs_only_when_threshold_differs_from_default(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """RNF-64: `if threshold != _DEFAULT_THRESHOLD` só tem efeito observável
    via log (o valor retornado não muda) — a fronteira `!=`/`==` só é
    detectável inspecionando se a linha de log foi de fato emitida."""
    monkeypatch.setattr(
        model_service_module, "_MODEL_REGISTRY", {"random_forest": settings.model_path}
    )
    monkeypatch.setattr(
        model_service_module,
        "load_model",
        lambda *a, **kw: ModelService(_MinimalModel()),
    )

    monkeypatch.setattr(model_service_module, "_resolve_threshold", lambda name: 0.5)
    with caplog.at_level("INFO", logger="src.services.model_service"):
        caplog.clear()
        load_model_by_name("random_forest")
    assert not any("usará threshold" in r.message for r in caplog.records)

    monkeypatch.setattr(model_service_module, "_resolve_threshold", lambda name: 0.3)
    with caplog.at_level("INFO", logger="src.services.model_service"):
        caplog.clear()
        load_model_by_name("random_forest")
    assert any("usará threshold" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# ModelService.decision_threshold / predict_from_features — exceção relançada
# ---------------------------------------------------------------------------


class _MinimalModel:
    feature_names_in_ = ["TP2"]


class _RaisingModel:
    feature_names_in_ = ["TP2"]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise RuntimeError("boom")


def test_decision_threshold_property_reflects_constructor_value() -> None:
    service = ModelService(_MinimalModel(), decision_threshold=0.37)
    assert service.decision_threshold == 0.37


def test_predict_from_features_reraises_the_original_error_never_swallows_it() -> None:
    service = ModelService(_RaisingModel(), decision_threshold=0.5)
    with pytest.raises(RuntimeError, match="boom"):
        service.predict_from_features(pd.DataFrame([{"TP2": 1.0}]))


# ---------------------------------------------------------------------------
# predict_from_features — threshold boundary, precisão do round, e
# preenchimento de colunas ausentes (RNF-64)
# ---------------------------------------------------------------------------


class _ProbaModel:
    """Modelo fake cujo `predict_proba` INSPECIONA o `X` recebido — mata os
    mutantes de `_build_feature_row`'s fallback fill (1.0 vs 0.0 vs None,
    `in`/`not in`, cada literal de coluna) verificando os valores exatos que
    o alinhamento dinâmico de `predict_from_features` de fato produziu."""

    feature_names_in_ = [
        "TP2",
        "LPS",
        "Pressure_switch",
        "Caudal_impulses",
        "Reservoirs",
    ]

    def __init__(self, positive_proba: float) -> None:
        self._positive_proba = positive_proba

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert X["LPS"].iloc[0] == 1.0
        assert X["Pressure_switch"].iloc[0] == 1.0
        assert X["Caudal_impulses"].iloc[0] == 1.0
        assert X["Reservoirs"].iloc[0] == 0.0
        return np.array([[1.0 - self._positive_proba, self._positive_proba]])


def test_predict_from_features_fills_legacy_ok_columns_with_1_and_others_with_0() -> (
    None
):
    service = ModelService(_ProbaModel(positive_proba=0.5), decision_threshold=0.5)
    service.predict_from_features(pd.DataFrame([{"TP2": 1.0}]))


def test_predict_from_features_predicted_class_is_1_exactly_at_the_threshold() -> None:
    """Fronteira: a regra real é `>=` — `failure_probability == threshold`
    deve classificar como falha (1), não como normal (0)."""
    service = ModelService(_ProbaModel(positive_proba=0.5), decision_threshold=0.5)
    response = service.predict_from_features(pd.DataFrame([{"TP2": 1.0}]))
    assert response.predicted_class == 1


def test_predict_from_features_rounds_probability_to_exactly_6_decimal_places() -> None:
    service = ModelService(
        _ProbaModel(positive_proba=0.1234567891), decision_threshold=0.5
    )
    response = service.predict_from_features(pd.DataFrame([{"TP2": 1.0}]))
    assert response.failure_probability == 0.123457


# ---------------------------------------------------------------------------
# _build_feature_row — exatidão das 12 chaves brutas + 4 cross-features V2
# ---------------------------------------------------------------------------


def _predict_request(**overrides: Any) -> PredictRequest:
    kwargs = dict(
        TP2=8.0,
        TP3=4.0,
        H1=8.5,
        DV_pressure=2.1,
        Reservoirs=8.7,
        Motor_current=6.0,
        Oil_temperature=68.5,
        COMP=1.0,
        DV_eletric=0.0,
        Towers=1.0,
        MPG=1.0,
        Oil_level=1.0,
    )
    kwargs.update(overrides)
    return PredictRequest(**kwargs)


def test_build_feature_row_computes_the_exact_v2_cross_sensor_formulas() -> None:
    service = ModelService(_MinimalModel(), decision_threshold=0.5)
    row = service._build_feature_row(_predict_request())  # noqa: SLF001

    eps = 1e-6
    assert row["TP2_TP3_diff"].iloc[0] == pytest.approx(8.0 - 4.0)
    assert row["TP2_TP3_ratio"].iloc[0] == pytest.approx(8.0 / (4.0 + eps))
    assert row["work_per_pressure"].iloc[0] == pytest.approx(6.0 / (8.0 + eps))
    assert row["reservoir_drop"].iloc[0] == pytest.approx(8.7 - 4.0)


def test_build_feature_row_tp2_tp3_ratio_uses_exactly_1e_minus_6_as_epsilon() -> None:
    """RNF-64: com TP3 == 0.0 (denominador zerado de propósito), o valor de
    `eps` deixa de ser desprezível — `1e-6` vs `2e-6` produzem resultados
    bem diferentes, ao contrário do teste acima (TP3=4.0 torna qualquer
    variação de eps irrelevante dentro da tolerância padrão do
    `pytest.approx`)."""
    service = ModelService(_MinimalModel(), decision_threshold=0.5)
    row = service._build_feature_row(_predict_request(TP2=8.0, TP3=0.0))  # noqa: SLF001
    assert row["TP2_TP3_ratio"].iloc[0] == pytest.approx(8.0 / 1e-6, rel=1e-9)


def test_build_feature_row_work_per_pressure_uses_exactly_1e_minus_6_as_epsilon() -> (
    None
):
    """Mesma fronteira, mas para `work_per_pressure = Motor_current / (TP2 +
    eps)` — precisa de TP2 == 0.0 (não TP3) pra tornar eps observável, e
    mata especificamente o mutante que troca `+eps` por `-eps`."""
    service = ModelService(_MinimalModel(), decision_threshold=0.5)
    row = service._build_feature_row(
        _predict_request(TP2=0.0, Motor_current=6.0)
    )  # noqa: SLF001
    assert row["work_per_pressure"].iloc[0] == pytest.approx(6.0 / 1e-6, rel=1e-9)


def test_build_feature_row_keeps_the_exact_raw_sensor_and_binary_keys() -> None:
    service = ModelService(_MinimalModel(), decision_threshold=0.5)
    row = service._build_feature_row(_predict_request())  # noqa: SLF001

    for col in [
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
    ]:
        assert col in row.columns


# ---------------------------------------------------------------------------
# get_model_service — as 3 ramificações de resolução da dependência
# ---------------------------------------------------------------------------


class _FakeState:
    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


class _FakeApp:
    def __init__(self, state: _FakeState) -> None:
        self.state = state


class _FakeRequest:
    def __init__(self, state: _FakeState) -> None:
        self.app = _FakeApp(state)


class _FakeRegistry:
    def __init__(self, service: Any) -> None:
        self._service = service

    async def get(self) -> Any:
        return self._service


async def test_get_model_service_prefers_registry_when_present() -> None:
    sentinel = object()
    request = _FakeRequest(_FakeState(model_registry=_FakeRegistry(sentinel)))
    result = await get_model_service(request)  # type: ignore[arg-type]
    assert result is sentinel


async def test_get_model_service_falls_back_to_legacy_service_attribute() -> None:
    sentinel = object()
    request = _FakeRequest(_FakeState(model_registry=None, model_service=sentinel))
    result = await get_model_service(request)  # type: ignore[arg-type]
    assert result is sentinel


async def test_get_model_service_raises_when_neither_registry_nor_legacy_service_exists() -> (
    None
):
    request = _FakeRequest(_FakeState(model_registry=None, model_service=None))
    with pytest.raises(ModelNotAvailableError):
        await get_model_service(request)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# RNF-76 — métricas de ML (predict_from_features / predict_batch).
#
# `_counter_value`/`_histogram_count` leem o valor ATUAL do child de label
# via a API interna do prometheus_client (`._value.get()`) — os Counters em
# `src.core.metrics` são singletons de módulo, compartilhados por toda a
# suíte no mesmo processo; por isso todo teste aqui mede o DELTA
# antes/depois da chamada, nunca o valor absoluto (evita acoplamento à
# ordem de execução dos outros testes).
# ---------------------------------------------------------------------------


def _counter_value(counter: Any, **labels: str) -> float:
    return float(counter.labels(**labels)._value.get())  # noqa: SLF001


def _histogram_count(histogram: Any, **labels: str) -> float:
    return float(histogram.labels(**labels)._sum.get())  # noqa: SLF001


class _BatchProbaModel:
    feature_names_in_ = ["TP2"]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([[0.9, 0.1]] * len(X))


class _BatchRaisingModel:
    feature_names_in_ = ["TP2"]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise RuntimeError("batch boom")


def _sample_request() -> PredictRequest:
    return PredictRequest(
        TP2=1.0,
        TP3=1.0,
        H1=1.0,
        DV_pressure=1.0,
        Reservoirs=1.0,
        Motor_current=1.0,
        Oil_temperature=1.0,
        COMP=1,
        DV_eletric=1,
        Towers=1,
        MPG=1,
        Oil_level=1,
    )


def test_predict_from_features_success_records_inference_and_prediction_metrics() -> (
    None
):
    service = ModelService(
        _ProbaModel(positive_proba=0.9), decision_threshold=0.5, model_name="_test_rf"
    )
    before_total = _counter_value(
        metrics_module.INFERENCE_TOTAL, model="_test_rf", status="success"
    )
    before_pred = _counter_value(
        metrics_module.PREDICTIONS_TOTAL, model="_test_rf", prediction_class="1"
    )
    before_error = _counter_value(
        metrics_module.INFERENCE_TOTAL, model="_test_rf", status="error"
    )

    service.predict_from_features(pd.DataFrame([{"TP2": 1.0}]))

    assert (
        _counter_value(
            metrics_module.INFERENCE_TOTAL, model="_test_rf", status="success"
        )
        == before_total + 1
    )
    assert (
        _counter_value(
            metrics_module.PREDICTIONS_TOTAL,
            model="_test_rf",
            prediction_class="1",
        )
        == before_pred + 1
    )
    # Sucesso NUNCA incrementa o contador de erro — mata o mutante que
    # trocaria "success"/"error" ou removeria o `except`.
    assert (
        _counter_value(metrics_module.INFERENCE_TOTAL, model="_test_rf", status="error")
        == before_error
    )


def test_predict_from_features_error_records_error_status_and_no_prediction() -> None:
    service = ModelService(
        _RaisingModel(), decision_threshold=0.5, model_name="_test_err"
    )
    before_error = _counter_value(
        metrics_module.INFERENCE_TOTAL, model="_test_err", status="error"
    )
    before_success = _counter_value(
        metrics_module.INFERENCE_TOTAL, model="_test_err", status="success"
    )

    with pytest.raises(RuntimeError, match="boom"):
        service.predict_from_features(pd.DataFrame([{"TP2": 1.0}]))

    assert (
        _counter_value(
            metrics_module.INFERENCE_TOTAL, model="_test_err", status="error"
        )
        == before_error + 1
    )
    assert (
        _counter_value(
            metrics_module.INFERENCE_TOTAL, model="_test_err", status="success"
        )
        == before_success
    )


def test_predict_from_features_records_positive_duration() -> None:
    """Limite superior de 1s (folgado para uma predict_proba fake instantânea)
    mata o mutante `time.perf_counter() - start` -> `+ start`: a soma de dois
    `perf_counter()` dá um valor na casa dos milhares/milhões de segundos
    (tempo de monotonic clock, não epoch), muito acima de qualquer duração
    real — só a SUBTRAÇÃO produz um delta pequeno e plausível."""
    service = ModelService(
        _ProbaModel(positive_proba=0.1), decision_threshold=0.5, model_name="_test_dur"
    )
    before = _histogram_count(
        metrics_module.INFERENCE_DURATION_SECONDS, model="_test_dur", status="success"
    )
    service.predict_from_features(pd.DataFrame([{"TP2": 1.0}]))
    after = _histogram_count(
        metrics_module.INFERENCE_DURATION_SECONDS, model="_test_dur", status="success"
    )
    delta = after - before
    assert 0.0 < delta < 1.0


def test_predict_from_features_error_records_bounded_positive_duration() -> None:
    """Mesmo racional do teste acima, para o ramo de erro (`except` de
    predict_from_features) — mata o mutante equivalente na chamada de
    `record_inference` dentro do `except`."""
    service = ModelService(
        _RaisingModel(), decision_threshold=0.5, model_name="_test_err_dur"
    )
    before = _histogram_count(
        metrics_module.INFERENCE_DURATION_SECONDS, model="_test_err_dur", status="error"
    )
    with pytest.raises(RuntimeError, match="boom"):
        service.predict_from_features(pd.DataFrame([{"TP2": 1.0}]))
    after = _histogram_count(
        metrics_module.INFERENCE_DURATION_SECONDS, model="_test_err_dur", status="error"
    )
    delta = after - before
    assert 0.0 < delta < 1.0


def test_predict_batch_success_records_batch_request_and_sample_counts() -> None:
    service = ModelService(
        _BatchProbaModel(), decision_threshold=0.5, model_name="_test_batch"
    )
    requests = [_sample_request(), _sample_request(), _sample_request()]

    before_requests = _counter_value(
        metrics_module.BATCH_REQUESTS_TOTAL, model="_test_batch", status="success"
    )
    before_samples = _counter_value(
        metrics_module.BATCH_SAMPLES_TOTAL, model="_test_batch"
    )

    service.predict_batch(requests)

    assert (
        _counter_value(
            metrics_module.BATCH_REQUESTS_TOTAL, model="_test_batch", status="success"
        )
        == before_requests + 1
    )
    assert (
        _counter_value(metrics_module.BATCH_SAMPLES_TOTAL, model="_test_batch")
        == before_samples + 3
    )


def test_predict_batch_error_records_error_status_and_no_sample_count() -> None:
    service = ModelService(
        _BatchRaisingModel(), decision_threshold=0.5, model_name="_test_batch_err"
    )
    before_error = _counter_value(
        metrics_module.BATCH_REQUESTS_TOTAL, model="_test_batch_err", status="error"
    )
    before_samples = _counter_value(
        metrics_module.BATCH_SAMPLES_TOTAL, model="_test_batch_err"
    )

    with pytest.raises(RuntimeError, match="batch boom"):
        service.predict_batch([_sample_request(), _sample_request()])

    assert (
        _counter_value(
            metrics_module.BATCH_REQUESTS_TOTAL, model="_test_batch_err", status="error"
        )
        == before_error + 1
    )
    # Amostras de um batch que falhou NUNCA contam como processadas.
    assert (
        _counter_value(metrics_module.BATCH_SAMPLES_TOTAL, model="_test_batch_err")
        == before_samples
    )


class _BatchThresholdModel:
    """Fake cujo `predict_proba` devolve a probabilidade EXATA do threshold
    para a 1ª amostra — mata o mutante `>=` -> `>` em `predict_batch` (o
    equivalente do teste de fronteira já existente para
    `predict_from_features`, `test_predict_from_features_predicted_class_is_1_exactly_at_the_threshold`,
    que não cobre o loop de `predict_batch`, lógica duplicada e separada)."""

    feature_names_in_ = ["TP2"]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([[0.5, 0.5]] * len(X))


def test_predict_batch_predicted_class_is_1_exactly_at_the_threshold() -> None:
    service = ModelService(_BatchThresholdModel(), decision_threshold=0.5)
    responses = service.predict_batch([_sample_request()])
    assert responses[0].predicted_class == 1


def test_predict_batch_empty_list_records_no_metrics() -> None:
    service = ModelService(
        _BatchProbaModel(), decision_threshold=0.5, model_name="_test_batch_empty"
    )
    before = _counter_value(
        metrics_module.BATCH_REQUESTS_TOTAL, model="_test_batch_empty", status="success"
    )
    assert service.predict_batch([]) == []
    assert (
        _counter_value(
            metrics_module.BATCH_REQUESTS_TOTAL,
            model="_test_batch_empty",
            status="success",
        )
        == before
    )


def test_model_service_defaults_to_unknown_model_name_label() -> None:
    service = ModelService(_ProbaModel(positive_proba=0.1), decision_threshold=0.5)
    before = _counter_value(
        metrics_module.INFERENCE_TOTAL, model="unknown", status="success"
    )
    service.predict_from_features(pd.DataFrame([{"TP2": 1.0}]))
    assert (
        _counter_value(
            metrics_module.INFERENCE_TOTAL, model="unknown", status="success"
        )
        == before + 1
    )
