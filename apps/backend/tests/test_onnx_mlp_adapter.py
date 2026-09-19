"""
Testes de `OnnxMlpAdapter` (RF-10/RNF-24, RNF-64/RNF-65).

Auditoria desta task: 87% de cobertura, zero teste dedicado — só
exercitado indiretamente via `test_model_service_real_artifacts.py`
(caminho feliz). `_FEATURE_NAMES` (34 entradas) e `_softmax` (staticmethod
puro) nunca tinham asserção de valor exato.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import pandas as pd

from src.services.mlp_adapter import _FEATURE_NAMES, OnnxMlpAdapter
from src.services.model_service import load_model_by_name

# ---------------------------------------------------------------------------
# _FEATURE_NAMES — as 34 colunas, na ordem exata de treino
# ---------------------------------------------------------------------------


def test_feature_names_is_the_exact_expected_34_column_list_in_order() -> None:
    assert _FEATURE_NAMES == [
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
        "TP2_delta",
        "TP2_std_5",
        "TP3_std_5",
        "H1_std_5",
        "DV_pressure_std_5",
        "Reservoirs_std_5",
        "Oil_temperature_std_5",
        "Motor_current_std_5",
        "TP2_ma_5",
        "TP2_ma_15",
        "TP3_ma_5",
        "TP3_ma_15",
        "H1_ma_5",
        "H1_ma_15",
        "DV_pressure_ma_5",
        "DV_pressure_ma_15",
        "Reservoirs_ma_5",
        "Reservoirs_ma_15",
        "Oil_temperature_ma_5",
        "Oil_temperature_ma_15",
        "Motor_current_ma_5",
        "Motor_current_ma_15",
    ]


# ---------------------------------------------------------------------------
# Construtor — validação ANTES do ONNX Runtime
# ---------------------------------------------------------------------------


def test_constructor_raises_file_not_found_for_missing_onnx(tmp_path: Path) -> None:
    fake_scaler = tmp_path / "scaler.joblib"
    fake_scaler.write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="MLP ONNX artefact not found"):
        OnnxMlpAdapter(tmp_path / "missing.onnx", fake_scaler)


def test_constructor_raises_file_not_found_for_missing_scaler(tmp_path: Path) -> None:
    fake_onnx = tmp_path / "model.onnx"
    fake_onnx.write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="MLP scaler artefact not found"):
        OnnxMlpAdapter(fake_onnx, tmp_path / "missing.joblib")


# ---------------------------------------------------------------------------
# _softmax — staticmethod puro, numericamente estável
# ---------------------------------------------------------------------------


def test_softmax_matches_the_textbook_formula_on_known_logits() -> None:
    logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    probs = OnnxMlpAdapter._softmax(logits)  # noqa: SLF001

    shifted = logits - logits.max(axis=1, keepdims=True)
    expected = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    np.testing.assert_allclose(probs, expected, rtol=1e-6)


def test_softmax_rows_each_sum_to_exactly_1() -> None:
    logits = np.array([[10.0, -5.0], [0.0, 0.0], [3.0, 3.0001]], dtype=np.float32)
    probs = OnnxMlpAdapter._softmax(logits)  # noqa: SLF001
    np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0, 1.0], rtol=1e-6)


def test_softmax_is_invariant_to_a_constant_shift_added_to_every_logit() -> None:
    """Prova que a subtração de `logits.max(axis=1, keepdims=True)` é uma
    estabilização numérica, não muda o resultado — kills a mutação
    `+` no lugar de `-`, que QUEBRARIA essa invariância."""
    logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    shifted_logits = logits + 1000.0
    np.testing.assert_allclose(
        OnnxMlpAdapter._softmax(logits),  # noqa: SLF001
        OnnxMlpAdapter._softmax(shifted_logits),  # noqa: SLF001
        rtol=1e-4,
    )


def test_softmax_on_a_non_square_batch_matches_exact_row_wise_values() -> None:
    """RNF-64: matriz (2, 3) — retangular de propósito. `keepdims=False` em
    `.max(...)`/`.sum(...)` produziria shape (2,) que NÃO faz broadcast
    válido contra (2, 3) (dimensão final 2 != 3): o mutante levanta
    `ValueError` aqui, o que já mata o mutante; a asserção de valor exato
    cobre o caso em que a forma calha de ser compatível."""
    logits = np.array([[1.0, 2.0, 3.0], [30.0, 20.0, 10.0]], dtype=np.float32)
    probs = OnnxMlpAdapter._softmax(logits)  # noqa: SLF001

    shifted = logits - logits.max(axis=1, keepdims=True)
    expected = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    np.testing.assert_allclose(probs, expected, rtol=1e-6)
    assert probs.shape == (2, 3)


# ---------------------------------------------------------------------------
# predict() — threshold 0.5 sobre a coluna 1, artefato real
# ---------------------------------------------------------------------------


def test_predict_applies_0_5_threshold_on_column_1_of_predict_proba() -> None:
    service = load_model_by_name("mlp")
    adapter = service._model  # noqa: SLF001
    assert isinstance(adapter, OnnxMlpAdapter)

    features = pd.DataFrame(
        [[0.0] * len(adapter.feature_names_in_)],
        columns=list(adapter.feature_names_in_),
    )
    probs = adapter.predict_proba(features)
    labels = adapter.predict(features)

    assert labels.dtype == np.int64
    expected = (probs[:, 1] >= 0.5).astype(np.int64)
    np.testing.assert_array_equal(labels, expected)
