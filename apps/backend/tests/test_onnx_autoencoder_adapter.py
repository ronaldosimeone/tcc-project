"""
Testes de `OnnxAutoencoderAdapter` (RF-10, RNF-64/RNF-65).

Auditoria desta task: 84% de cobertura mas mutation score baixo — os testes
existentes (via `test_model_service_real_artifacts.py`) só exercitam o
caminho feliz ponta-a-ponta, sem verificar os VALORES exatos da fórmula do
sigmoide (documentada no próprio docstring do módulo: score(threshold)=0.5,
score(2×threshold)≈0.95, score(0)≈0.05) nem os limites exatos do
padding/janela de `_build_window`. Usa o artefato REAL (`autoencoder`, via
`load_model_by_name`) para chamar os métodos privados diretamente — mais
preciso que recriar um `OnnxAutoencoderAdapter` fake, e sem custo real
(ONNX Runtime já carregado nesse artefato para outros testes).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.services.model_service import load_model_by_name
from src.services.onnx_autoencoder_adapter import (
    _RAW_CHANNELS,
    _SIGMOID_SCALE_FACTOR,
    OnnxAutoencoderAdapter,
)

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


def test_sigmoid_scale_factor_is_exactly_3() -> None:
    assert _SIGMOID_SCALE_FACTOR == 3.0


# ---------------------------------------------------------------------------
# Construtor — validação ANTES de tocar o ONNX Runtime
# ---------------------------------------------------------------------------


def test_constructor_raises_file_not_found_for_missing_onnx(tmp_path: Path) -> None:
    fake_scaler = tmp_path / "scaler.joblib"
    fake_scaler.write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="Autoencoder ONNX artefact not found"):
        OnnxAutoencoderAdapter(
            tmp_path / "missing.onnx", fake_scaler, mse_threshold=0.1
        )


def test_constructor_raises_file_not_found_for_missing_scaler(tmp_path: Path) -> None:
    fake_onnx = tmp_path / "model.onnx"
    fake_onnx.write_bytes(b"x")
    with pytest.raises(
        FileNotFoundError, match="Autoencoder scaler artefact not found"
    ):
        OnnxAutoencoderAdapter(
            fake_onnx, tmp_path / "missing.joblib", mse_threshold=0.1
        )


@pytest.mark.parametrize("bad_threshold", [0.0, -0.1, -5.0])
def test_constructor_raises_value_error_for_non_positive_mse_threshold(
    tmp_path: Path, bad_threshold: float
) -> None:
    """Fronteira: `mse_threshold <= 0.0` rejeita 0.0 e negativos — só valores
    estritamente positivos são válidos."""
    fake_onnx = tmp_path / "model.onnx"
    fake_onnx.write_bytes(b"x")
    fake_scaler = tmp_path / "scaler.joblib"
    fake_scaler.write_bytes(b"x")
    with pytest.raises(ValueError, match="must be positive"):
        OnnxAutoencoderAdapter(fake_onnx, fake_scaler, mse_threshold=bad_threshold)


# ---------------------------------------------------------------------------
# _sigmoid_score — a fórmula documentada no docstring do módulo, testada
# com os 3 pontos que o próprio docstring cita como contrato.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_adapter() -> OnnxAutoencoderAdapter:
    service = load_model_by_name("autoencoder")
    adapter = service._model  # noqa: SLF001
    assert isinstance(adapter, OnnxAutoencoderAdapter)
    return adapter


def test_sigmoid_score_at_exact_threshold_is_the_decision_boundary_0_5(
    real_adapter: OnnxAutoencoderAdapter,
) -> None:
    score = real_adapter._sigmoid_score(real_adapter._mse_threshold)  # noqa: SLF001
    assert score == pytest.approx(0.5, abs=1e-9)


def test_sigmoid_score_at_2x_threshold_is_approximately_0_95(
    real_adapter: OnnxAutoencoderAdapter,
) -> None:
    score = real_adapter._sigmoid_score(2 * real_adapter._mse_threshold)  # noqa: SLF001
    assert score == pytest.approx(0.95, abs=0.01)


def test_sigmoid_score_at_zero_mse_is_approximately_0_05(
    real_adapter: OnnxAutoencoderAdapter,
) -> None:
    score = real_adapter._sigmoid_score(0.0)  # noqa: SLF001
    assert score == pytest.approx(0.05, abs=0.01)


def test_sigmoid_score_is_strictly_monotonic_in_mse(
    real_adapter: OnnxAutoencoderAdapter,
) -> None:
    threshold = real_adapter._mse_threshold  # noqa: SLF001
    low = real_adapter._sigmoid_score(0.0)  # noqa: SLF001
    mid = real_adapter._sigmoid_score(threshold)  # noqa: SLF001
    high = real_adapter._sigmoid_score(2 * threshold)  # noqa: SLF001
    very_high = real_adapter._sigmoid_score(10 * threshold)  # noqa: SLF001
    assert low < mid < high < very_high


# ---------------------------------------------------------------------------
# _build_window — janela completa vs. padding por replicação da linha mais
# antiga (RF-10, cold-start).
# ---------------------------------------------------------------------------


def test_build_window_uses_the_last_window_size_rows_when_history_is_longer(
    real_adapter: OnnxAutoencoderAdapter,
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
    assert window[0, 0, 0] == float(
        n - t
    )  # primeira linha da janela = a t-ésima do fim
    assert window[0, -1, 0] == float(n - 1)  # última linha = a mais recente


def test_build_window_boundary_at_exactly_window_size_rows_uses_all_of_them(
    real_adapter: OnnxAutoencoderAdapter,
) -> None:
    """Fronteira: `len(arr) == window_size` deve usar TODAS as linhas
    diretamente (`>=`), não cair no ramo de padding (`>` seria o mutante)."""
    t = real_adapter._window_size  # noqa: SLF001
    df = pd.DataFrame(
        {
            c: [float(i) for i in range(t)] for c in real_adapter._channels
        }  # noqa: SLF001
    )
    window = real_adapter._build_window(df)  # noqa: SLF001
    assert window[0, 0, 0] == 0.0  # NÃO reescrito por padding
    assert window[0, -1, 0] == float(t - 1)


def test_build_window_pads_by_replicating_only_the_oldest_row(
    real_adapter: OnnxAutoencoderAdapter,
) -> None:
    """Histórico curto (3 linhas << window_size): o padding replica
    EXATAMENTE a primeira linha real (10.0), nunca uma média ou outra
    linha, e os 3 valores reais aparecem na ordem certa no final."""
    df = pd.DataFrame(
        {c: [10.0, 20.0, 30.0] for c in real_adapter._channels}  # noqa: SLF001
    )
    window = real_adapter._build_window(df)  # noqa: SLF001
    t = real_adapter._window_size  # noqa: SLF001
    assert (window[0, : t - 3, 0] == 10.0).all()
    assert list(window[0, t - 3 :, 0]) == [10.0, 20.0, 30.0]


# ---------------------------------------------------------------------------
# predict_proba / predict — threshold 0.5 aplicado sobre a coluna certa
# ---------------------------------------------------------------------------


def test_predict_applies_0_5_threshold_on_column_1_of_predict_proba(
    real_adapter: OnnxAutoencoderAdapter,
) -> None:
    df = pd.DataFrame(
        {
            c: [0.0] * real_adapter._window_size for c in real_adapter._channels
        }  # noqa: SLF001
    )
    probs = real_adapter.predict_proba(df)
    labels = real_adapter.predict(df)
    assert probs.shape == (1, 2)
    assert labels.shape == (1,)
    expected = (probs[:, 1] >= 0.5).astype(np.int64)
    np.testing.assert_array_equal(labels, expected)
    # As duas colunas somam 1 (probabilidade complementar real, não um
    # valor arbitrário na posição 0).
    assert probs[0, 0] + probs[0, 1] == pytest.approx(1.0)
