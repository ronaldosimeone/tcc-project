"""
Testes de `OnnxTreeAdapter` (RF-10, RNF-64/RNF-65).

Auditoria desta task: 67% de cobertura — faltavam os DOIS erros de
construtor (artefato ausente / feature_names vazio), o método `predict()`
(threshold 0.5) chamado diretamente, e o Pass 2 (ZipMap) de
`_extract_probabilities`. `test_model_service_real_artifacts.py` já cobre o
Pass 1 (ndarray) via `random_forest_v2` real; aqui o `_extract_probabilities`
— `@staticmethod`, puro, sem ONNX Runtime — é testado isoladamente com as DUAS
formas reais de saída que ele precisa distinguir (não um mock genérico: os
formatos exatos documentados no próprio docstring do adapter).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.services.model_service import load_model_by_name
from src.services.onnx_tree_adapter import OnnxTreeAdapter

# ---------------------------------------------------------------------------
# Construtor — validação ANTES de tocar o ONNX Runtime
# ---------------------------------------------------------------------------


def test_constructor_raises_file_not_found_for_missing_artifact(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        OnnxTreeAdapter(tmp_path / "does-not-exist.onnx", feature_names=["TP2"])


def test_constructor_raises_value_error_for_empty_feature_names(tmp_path: Path) -> None:
    fake_onnx = tmp_path / "model.onnx"
    fake_onnx.write_bytes(b"not a real onnx file")  # nunca chega a ser lido
    with pytest.raises(ValueError, match="feature_names must be non-empty"):
        OnnxTreeAdapter(fake_onnx, feature_names=[])


# ---------------------------------------------------------------------------
# _extract_probabilities — as DUAS formas reais de saída (staticmethod puro)
# ---------------------------------------------------------------------------


def test_extract_probabilities_pass1_ndarray_shape_n_by_2() -> None:
    """skl2onnx com zipmap=False — formato usado por random_forest_v2."""
    outputs = [
        np.array([0, 1], dtype=np.int64),  # "label" — ignorado por este método
        np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32),  # "probabilities"
    ]
    probs = OnnxTreeAdapter._extract_probabilities(outputs, n_rows=2)
    np.testing.assert_allclose(probs, [[0.9, 0.1], [0.2, 0.8]])
    assert probs.dtype == np.float32


def test_extract_probabilities_pass2_zipmap_list_of_dicts_int_keys() -> None:
    """onnxmltools com ZipMap (default) — chaves inteiras."""
    outputs = [
        np.array([0, 1], dtype=np.int64),
        [{0: 0.7, 1: 0.3}, {0: 0.1, 1: 0.9}],
    ]
    probs = OnnxTreeAdapter._extract_probabilities(outputs, n_rows=2)
    np.testing.assert_allclose(probs, [[0.7, 0.3], [0.1, 0.9]])


def test_extract_probabilities_pass2_zipmap_list_of_dicts_string_keys() -> None:
    """Alguns conversores emitem as chaves do ZipMap como string — o
    adapter precisa aceitar ambas as formas (`k0 = 0 if 0 in first else "0"`)."""
    outputs = [[{"0": 0.6, "1": 0.4}]]
    probs = OnnxTreeAdapter._extract_probabilities(outputs, n_rows=1)
    np.testing.assert_allclose(probs, [[0.6, 0.4]])


def test_extract_probabilities_prefers_ndarray_when_both_shapes_present() -> None:
    """Se por algum motivo a sessão emitir os dois formatos, o ndarray (Pass 1)
    tem prioridade — documentado como "Pass 1" no próprio código."""
    ndarray_out = np.array([[0.5, 0.5]], dtype=np.float32)
    zipmap_out = [{0: 0.99, 1: 0.01}]
    probs = OnnxTreeAdapter._extract_probabilities([zipmap_out, ndarray_out], n_rows=1)
    np.testing.assert_allclose(probs, [[0.5, 0.5]])


def test_extract_probabilities_raises_runtime_error_when_no_known_shape_matches() -> (
    None
):
    outputs = [np.array([1, 2, 3]), "algo completamente inesperado"]
    with pytest.raises(RuntimeError, match="Could not find a probability output"):
        OnnxTreeAdapter._extract_probabilities(outputs, n_rows=3)


def test_extract_probabilities_rejects_ndarray_with_wrong_row_count() -> None:
    """Um ndarray (n=3, 2) não deve ser aceito quando `n_rows=2` foi pedido —
    a checagem de shape é exata, não "parece uma matriz de probabilidades"."""
    outputs = [np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]], dtype=np.float32)]
    with pytest.raises(RuntimeError):
        OnnxTreeAdapter._extract_probabilities(outputs, n_rows=2)


# ---------------------------------------------------------------------------
# predict() — threshold 0.5, artefato real (random_forest_v2)
# ---------------------------------------------------------------------------


def test_predict_applies_0_5_threshold_on_real_artifact() -> None:
    """Chama `adapter.predict()` DIRETAMENTE (não via `ModelService`, que só
    usa `predict_proba`) — a única forma de exercitar a linha do threshold
    0.5 em `OnnxTreeAdapter.predict` com um artefato ONNX real."""
    service = load_model_by_name("random_forest_v2")
    adapter = service._model  # noqa: SLF001 — acesso de teste ao adapter real
    assert isinstance(adapter, OnnxTreeAdapter)

    features = pd.DataFrame(
        [[0.0] * len(adapter.feature_names_in_)],
        columns=list(adapter.feature_names_in_),
    )
    probs = adapter.predict_proba(features)
    labels = adapter.predict(features)

    assert labels.dtype == np.int64
    # A regra é exatamente a mesma usada por `predict_proba` — >= 0.5 -> 1.
    expected = (probs[:, 1] >= 0.5).astype(np.int64)
    np.testing.assert_array_equal(labels, expected)


def test_predict_classifies_as_1_exactly_at_the_0_5_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fronteira exata (RNF-64): não dá pra forçar um artefato real a
    devolver EXATAMENTE 0.5 — substitui `predict_proba` por um valor
    controlado no adapter real (sem tocar o ONNX Runtime) pra testar só a
    regra `>= 0.5`, não `> 0.5`."""
    service = load_model_by_name("random_forest_v2")
    adapter = service._model  # noqa: SLF001
    monkeypatch.setattr(
        adapter, "predict_proba", lambda X: np.array([[0.5, 0.5]], dtype=np.float32)
    )
    features = pd.DataFrame(
        [[0.0] * len(adapter.feature_names_in_)],
        columns=list(adapter.feature_names_in_),
    )
    labels = adapter.predict(features)
    assert labels[0] == 1
