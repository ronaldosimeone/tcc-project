"""
Testes de `OnnxSequenceAdapter` (RF-10/RNF-24 — TCN/BiLSTM/PatchTST,
RNF-64/RNF-65).

Auditoria desta task: 89% de cobertura, zero teste dedicado — só
exercitado indiretamente via `test_model_service_real_artifacts.py`
(caminho feliz, um único artefato por vez). Mesma estrutura de
`OnnxAutoencoderAdapter` (janela + scaler + softmax) — mesmas classes de
mutante sobrevivente: lista de canais, fronteira exata da janela, padding
por replicação, e a fórmula do softmax.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.services.model_service import load_model_by_name
from src.services.onnx_sequence_adapter import _RAW_CHANNELS, OnnxSequenceAdapter

# ---------------------------------------------------------------------------
# Constantes de módulo
# ---------------------------------------------------------------------------


def test_raw_channels_is_the_exact_expected_12_column_list() -> None:
    assert _RAW_CHANNELS == [
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


# ---------------------------------------------------------------------------
# Construtor — validação ANTES do ONNX Runtime
# ---------------------------------------------------------------------------


def test_constructor_raises_file_not_found_for_missing_onnx(tmp_path: Path) -> None:
    fake_scaler = tmp_path / "scaler.joblib"
    fake_scaler.write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="Sequential ONNX artefact not found"):
        OnnxSequenceAdapter(tmp_path / "missing.onnx", fake_scaler)


def test_constructor_raises_file_not_found_for_missing_scaler(tmp_path: Path) -> None:
    fake_onnx = tmp_path / "model.onnx"
    fake_onnx.write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="Sequential scaler artefact not found"):
        OnnxSequenceAdapter(fake_onnx, tmp_path / "missing.joblib")


# ---------------------------------------------------------------------------
# _softmax — staticmethod puro (mesma fórmula do OnnxMlpAdapter)
# ---------------------------------------------------------------------------


def test_softmax_matches_the_textbook_formula_on_known_logits() -> None:
    logits = np.array([[1.0, 2.0, 3.0], [30.0, 20.0, 10.0]], dtype=np.float32)
    probs = OnnxSequenceAdapter._softmax(logits)  # noqa: SLF001

    shifted = logits - logits.max(axis=1, keepdims=True)
    expected = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    np.testing.assert_allclose(probs, expected, rtol=1e-6)
    assert probs.shape == (2, 3)


def test_softmax_rows_each_sum_to_exactly_1() -> None:
    logits = np.array([[10.0, -5.0], [0.0, 0.0]], dtype=np.float32)
    probs = OnnxSequenceAdapter._softmax(logits)  # noqa: SLF001
    np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0], rtol=1e-6)


def test_softmax_is_invariant_to_a_constant_shift_added_to_every_logit() -> None:
    logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    np.testing.assert_allclose(
        OnnxSequenceAdapter._softmax(logits),  # noqa: SLF001
        OnnxSequenceAdapter._softmax(logits + 1000.0),  # noqa: SLF001
        rtol=1e-4,
    )


# ---------------------------------------------------------------------------
# _build_window / predict — via artefato real (TCN)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_adapter() -> OnnxSequenceAdapter:
    service = load_model_by_name("tcn")
    adapter = service._model  # noqa: SLF001
    assert isinstance(adapter, OnnxSequenceAdapter)
    return adapter


def test_build_window_uses_the_last_window_size_rows_when_history_is_longer(
    real_adapter: OnnxSequenceAdapter,
) -> None:
    t = real_adapter._window_size  # noqa: SLF001
    n = t + 5
    df = pd.DataFrame(
        {
            c: [float(i) for i in range(n)] for c in real_adapter._channels
        }  # noqa: SLF001
    )
    window = real_adapter._build_window(df)  # noqa: SLF001
    assert window.shape == (1, t, len(real_adapter._channels))  # noqa: SLF001
    assert window[0, 0, 0] == float(n - t)
    assert window[0, -1, 0] == float(n - 1)


def test_build_window_boundary_at_exactly_window_size_rows_uses_all_of_them(
    real_adapter: OnnxSequenceAdapter,
) -> None:
    t = real_adapter._window_size  # noqa: SLF001
    df = pd.DataFrame(
        {
            c: [float(i) for i in range(t)] for c in real_adapter._channels
        }  # noqa: SLF001
    )
    window = real_adapter._build_window(df)  # noqa: SLF001
    assert window[0, 0, 0] == 0.0
    assert window[0, -1, 0] == float(t - 1)


def test_build_window_pads_by_replicating_only_the_oldest_row(
    real_adapter: OnnxSequenceAdapter,
) -> None:
    df = pd.DataFrame(
        {c: [10.0, 20.0, 30.0] for c in real_adapter._channels}  # noqa: SLF001
    )
    window = real_adapter._build_window(df)  # noqa: SLF001
    t = real_adapter._window_size  # noqa: SLF001
    assert (window[0, : t - 3, 0] == 10.0).all()
    assert list(window[0, t - 3 :, 0]) == [10.0, 20.0, 30.0]


def test_predict_applies_0_5_threshold_on_column_1_of_predict_proba(
    real_adapter: OnnxSequenceAdapter,
) -> None:
    df = pd.DataFrame(
        {
            c: [0.0] * real_adapter._window_size for c in real_adapter._channels
        }  # noqa: SLF001
    )
    probs = real_adapter.predict_proba(df)
    labels = real_adapter.predict(df)
    expected = (probs[:, 1] >= 0.5).astype(np.int64)
    np.testing.assert_array_equal(labels, expected)
