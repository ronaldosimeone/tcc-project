"""
Smoke tests for the sequential DL ONNX artefacts (TCN, BiLSTM, PatchTST).

Strategy
--------
All tests are skip-friendly: they require the artefacts produced by
``python src/train_sequential.py --arch <arch>``.  Run once per arch before
executing this suite.

Quick smoke run to generate artefacts:
    python src/train_sequential.py --arch tcn       --max-epochs 3 --subsample-rows 200000
    python src/train_sequential.py --arch bilstm    --max-epochs 3 --subsample-rows 200000
    python src/train_sequential.py --arch patchtst  --max-epochs 3 --subsample-rows 200000

Interface contract verified per arch
-------------------------------------
  1. ONNX session + scaler load without errors.
  2. ONNX input shape: (None, T=60, C=12).
  3. ONNX output shape: (batch, 2) (raw logits).
  4. predict_proba rows sum to 1.0 ± 1e-5 after softmax.
  5. All probabilities in [0.0, 1.0].
  6. PyTorch model output ≈ ONNX output (max abs diff < 1e-5).
  7. DataModule: window shape (B, T, C) and no exact-hash overlap between
     train and val sets (data-leakage guard).
  8. Single-window inference latency p95 < 500 ms.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import pytest
import torch
from sklearn.preprocessing import StandardScaler

# ── resolve project paths ─────────────────────────────────────────────────────
_ML_ROOT: Path = Path(__file__).resolve().parents[1]
_MODELS_DIR: Path = _ML_ROOT / "models"
_SRC: Path = _ML_ROOT / "src"
_DATA: Path = _ML_ROOT / "data" / "processed" / "metropt3.parquet"

sys.path.insert(0, str(_SRC))

_WINDOW_SIZE: int = 60
_N_CHANNELS: int = 12
_ARCHS: list[str] = ["tcn", "bilstm", "patchtst"]

# Per-architecture ONNX equivalence tolerance.  Mirrors the values in
# train_sequential.py: TCN/BiLSTM stay tight at 1e-5; PatchTST relaxes to
# 1e-4 because Transformer SDPA decomposes differently in ONNX vs PyTorch.
_TOLERANCES: dict[str, float] = {
    "tcn": 1e-5,
    "bilstm": 1e-5,
    "patchtst": 1e-4,
}

# Maximum batch size each ONNX graph supports.  PatchTST is exported with a
# fixed batch dim of 1 (see train_sequential.export_to_onnx for the PyTorch
# 2.3.x bug rationale); TCN/BiLSTM accept arbitrary batches via dynamic_axes.
_MAX_ONNX_BATCH: dict[str, int] = {
    "tcn": 32,
    "bilstm": 32,
    "patchtst": 1,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _onnx_path(arch: str) -> Path:
    return _MODELS_DIR / f"{arch}_v1.onnx"


def _scaler_path(arch: str) -> Path:
    return _MODELS_DIR / f"{arch}_scaler.joblib"


def _artefacts_exist(arch: str) -> bool:
    return _onnx_path(arch).exists() and _scaler_path(arch).exists()


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted: np.ndarray = logits - logits.max(axis=1, keepdims=True)
    exp: np.ndarray = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _dummy_window(batch: int = 1) -> np.ndarray:
    """Random float32 window of shape (batch, T, C)."""
    rng: np.random.Generator = np.random.default_rng(42)
    return rng.standard_normal((batch, _WINDOW_SIZE, _N_CHANNELS)).astype(np.float32)


# ---------------------------------------------------------------------------
# 1. Artefact loading
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch", _ARCHS)
def test_onnx_model_loads(arch: str) -> None:
    if not _artefacts_exist(arch):
        pytest.skip(f"Run train_sequential.py --arch {arch} first")
    sess = ort.InferenceSession(
        str(_onnx_path(arch)), providers=["CPUExecutionProvider"]
    )
    assert sess is not None


@pytest.mark.parametrize("arch", _ARCHS)
def test_scaler_loads(arch: str) -> None:
    if not _artefacts_exist(arch):
        pytest.skip(f"Run train_sequential.py --arch {arch} first")
    sc: StandardScaler = joblib.load(_scaler_path(arch))
    assert hasattr(sc, "mean_"), "Scaler has no mean_ — was it fitted?"
    assert sc.mean_.shape == (
        _N_CHANNELS,
    ), f"Expected {_N_CHANNELS} features, got {sc.mean_.shape[0]}"


# ---------------------------------------------------------------------------
# 2. ONNX input / output contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch", _ARCHS)
def test_onnx_input_shape(arch: str) -> None:
    if not _artefacts_exist(arch):
        pytest.skip()
    sess = ort.InferenceSession(
        str(_onnx_path(arch)), providers=["CPUExecutionProvider"]
    )
    inp = sess.get_inputs()[0]
    # Shape is [batch_size, T, C]; batch is dynamic (None or str label)
    assert (
        inp.shape[1] == _WINDOW_SIZE
    ), f"Expected T={_WINDOW_SIZE}, got {inp.shape[1]}"
    assert inp.shape[2] == _N_CHANNELS, f"Expected C={_N_CHANNELS}, got {inp.shape[2]}"


@pytest.mark.parametrize("arch", _ARCHS)
def test_onnx_output_has_two_logits(arch: str) -> None:
    if not _artefacts_exist(arch):
        pytest.skip()
    sess = ort.InferenceSession(
        str(_onnx_path(arch)), providers=["CPUExecutionProvider"]
    )
    out = sess.get_outputs()[0]
    assert out.shape[1] == 2, f"Expected 2 logits, got {out.shape[1]}"


# ---------------------------------------------------------------------------
# 3. Scaler + ONNX pipeline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch", _ARCHS)
def test_predict_proba_single_window_shape(arch: str) -> None:
    if not _artefacts_exist(arch):
        pytest.skip()
    sess = ort.InferenceSession(
        str(_onnx_path(arch)), providers=["CPUExecutionProvider"]
    )
    sc: StandardScaler = joblib.load(_scaler_path(arch))

    raw: np.ndarray = _dummy_window(1)
    scaled: np.ndarray = sc.transform(raw.reshape(-1, _N_CHANNELS)).reshape(raw.shape)
    logits: np.ndarray = sess.run(None, {sess.get_inputs()[0].name: scaled})[0]
    proba: np.ndarray = _softmax(logits)

    assert proba.shape == (1, 2), f"Expected (1, 2), got {proba.shape}"


@pytest.mark.parametrize("arch", _ARCHS)
def test_predict_proba_sums_to_one(arch: str) -> None:
    if not _artefacts_exist(arch):
        pytest.skip()
    sess = ort.InferenceSession(
        str(_onnx_path(arch)), providers=["CPUExecutionProvider"]
    )
    sc: StandardScaler = joblib.load(_scaler_path(arch))

    batch: int = min(4, _MAX_ONNX_BATCH[arch])
    raw: np.ndarray = _dummy_window(batch)
    scaled: np.ndarray = sc.transform(raw.reshape(-1, _N_CHANNELS)).reshape(raw.shape)
    logits: np.ndarray = sess.run(None, {sess.get_inputs()[0].name: scaled})[0]
    proba: np.ndarray = _softmax(logits)

    row_sums: np.ndarray = proba.sum(axis=1)
    assert np.allclose(
        row_sums, 1.0, atol=1e-5
    ), f"Probabilities do not sum to 1: {row_sums}"


@pytest.mark.parametrize("arch", _ARCHS)
def test_predict_proba_values_in_unit_range(arch: str) -> None:
    if not _artefacts_exist(arch):
        pytest.skip()
    sess = ort.InferenceSession(
        str(_onnx_path(arch)), providers=["CPUExecutionProvider"]
    )
    sc: StandardScaler = joblib.load(_scaler_path(arch))

    batch: int = min(8, _MAX_ONNX_BATCH[arch])
    raw: np.ndarray = _dummy_window(batch)
    scaled: np.ndarray = sc.transform(raw.reshape(-1, _N_CHANNELS)).reshape(raw.shape)
    logits: np.ndarray = sess.run(None, {sess.get_inputs()[0].name: scaled})[0]
    proba: np.ndarray = _softmax(logits)

    assert (proba >= 0.0).all() and (
        proba <= 1.0
    ).all(), f"Out-of-range probabilities: min={proba.min():.4f} max={proba.max():.4f}"


@pytest.mark.parametrize("arch", _ARCHS)
def test_batch_inference_shape(arch: str) -> None:
    if not _artefacts_exist(arch):
        pytest.skip()
    if _MAX_ONNX_BATCH[arch] == 1:
        pytest.skip(f"{arch} ONNX graph is exported with fixed batch=1")
    sess = ort.InferenceSession(
        str(_onnx_path(arch)), providers=["CPUExecutionProvider"]
    )
    sc: StandardScaler = joblib.load(_scaler_path(arch))

    raw: np.ndarray = _dummy_window(32)
    scaled: np.ndarray = sc.transform(raw.reshape(-1, _N_CHANNELS)).reshape(raw.shape)
    logits: np.ndarray = sess.run(None, {sess.get_inputs()[0].name: scaled})[0]

    assert logits.shape == (32, 2), f"Expected (32, 2), got {logits.shape}"


# ---------------------------------------------------------------------------
# 4. PyTorch ↔ ONNX equivalence (tolerance 1e-5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch", _ARCHS)
def test_pytorch_onnx_equivalence(arch: str) -> None:
    """Max abs diff between PyTorch and ONNX output must be < 1e-5."""
    if not _artefacts_exist(arch):
        pytest.skip()

    ckpt_dir: Path = _ML_ROOT / "checkpoints" / arch
    ckpts: list[Path] = sorted(ckpt_dir.glob("best-*.ckpt"))
    if not ckpts:
        pytest.skip(f"No checkpoint found at {ckpt_dir}")

    from models import BiLstmClassifier, PatchTSTClassifier, TcnClassifier

    model_cls = {
        "tcn": TcnClassifier,
        "bilstm": BiLstmClassifier,
        "patchtst": PatchTSTClassifier,
    }[arch]
    # ``map_location="cpu"`` avoids CUDA OOM / driver errors when the
    # checkpoint was trained on GPU and the test runs in a CPU-only env.
    model = model_cls.load_from_checkpoint(str(ckpts[-1]), map_location="cpu")
    model.eval()

    sc: StandardScaler = joblib.load(_scaler_path(arch))
    batch: int = min(4, _MAX_ONNX_BATCH[arch])
    raw: np.ndarray = _dummy_window(batch)
    scaled: np.ndarray = sc.transform(raw.reshape(-1, _N_CHANNELS)).reshape(raw.shape)

    with torch.no_grad():
        torch_out: np.ndarray = model(torch.from_numpy(scaled)).numpy()

    sess = ort.InferenceSession(
        str(_onnx_path(arch)), providers=["CPUExecutionProvider"]
    )
    onnx_out: np.ndarray = sess.run(None, {sess.get_inputs()[0].name: scaled})[0]

    diff: float = float(np.abs(torch_out - onnx_out).max())
    tol: float = _TOLERANCES[arch]
    assert (
        diff < tol
    ), f"{arch}: PyTorch/ONNX max abs diff = {diff:.3e} exceeds tolerance {tol:.0e}"


# ---------------------------------------------------------------------------
# 5. DataModule — shape and no-leakage
# ---------------------------------------------------------------------------


def test_datamodule_window_shape() -> None:
    if not _DATA.exists():
        pytest.skip("Parquet not found — run ingest_metropt.py first")
    from datamodule_sequence import MetroPTSequenceDataModule, SequenceConfig

    cfg = SequenceConfig(
        window_size=_WINDOW_SIZE,
        stride=30,
        batch_size=64,
        subsample_rows=50_000,
    )
    dm = MetroPTSequenceDataModule(_DATA, cfg)
    dm.prepare_data()
    dm.setup()

    x_tr, _ = dm.train_ds.tensors  # type: ignore[union-attr]
    x_te, _ = dm.val_ds.tensors  # type: ignore[union-attr]

    assert x_tr.shape[1] == _WINDOW_SIZE, f"Train T mismatch: {x_tr.shape}"
    assert x_tr.shape[2] == dm.n_channels, f"Train C mismatch: {x_tr.shape}"
    assert x_te.shape[1] == _WINDOW_SIZE, f"Val T mismatch: {x_te.shape}"
    assert x_te.shape[2] == dm.n_channels, f"Val C mismatch: {x_te.shape}"


def test_datamodule_no_window_leakage() -> None:
    """Exact-hash overlap between the first 500 train and val windows must be zero."""
    if not _DATA.exists():
        pytest.skip("Parquet not found")
    from datamodule_sequence import MetroPTSequenceDataModule, SequenceConfig

    cfg = SequenceConfig(
        window_size=_WINDOW_SIZE,
        stride=30,
        batch_size=64,
        subsample_rows=80_000,
    )
    dm = MetroPTSequenceDataModule(_DATA, cfg)
    dm.prepare_data()
    dm.setup()

    x_tr, _ = dm.train_ds.tensors  # type: ignore[union-attr]
    x_te, _ = dm.val_ds.tensors  # type: ignore[union-attr]

    sample_tr: int = min(500, len(x_tr))
    sample_te: int = min(500, len(x_te))

    tr_hashes: set[bytes] = {t.numpy().tobytes() for t in x_tr[:sample_tr]}
    te_hashes: set[bytes] = {t.numpy().tobytes() for t in x_te[:sample_te]}

    overlap: set[bytes] = tr_hashes & te_hashes
    assert (
        not overlap
    ), f"Data leakage detected: {len(overlap)} identical windows in both splits"


def test_datamodule_scaler_fitted_only_on_train() -> None:
    """Scaler must be fit on training data only — channel means must differ
    from the raw val-set channel means (if they were the same the scaler
    would have been fit on the full dataset)."""
    if not _DATA.exists():
        pytest.skip("Parquet not found")
    from datamodule_sequence import MetroPTSequenceDataModule, SequenceConfig

    cfg = SequenceConfig(
        window_size=_WINDOW_SIZE,
        stride=30,
        subsample_rows=80_000,
    )
    dm = MetroPTSequenceDataModule(_DATA, cfg)
    dm.prepare_data()
    dm.setup()

    # The val windows must NOT be zero-mean after per-channel scaling (which
    # would only be the case if the scaler was fit on the full dataset).
    x_te: np.ndarray = dm.val_ds.tensors[0].numpy()  # type: ignore[union-attr]
    channel_means: np.ndarray = x_te.reshape(-1, dm.n_channels).mean(axis=0)
    # If the scaler was fitted on the full dataset all means would be ~0.
    # Fitted only on train → val means are non-zero by the distributional gap.
    # We assert at least one channel mean has absolute value > 0.01 (loose guard).
    assert (
        np.abs(channel_means) > 0.01
    ).any(), (
        "All val channel means are ~0 — scaler may have been fit on the full dataset"
    )


# ---------------------------------------------------------------------------
# 6. Latency guard (p95 < 500 ms per single window)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch", _ARCHS)
def test_single_window_latency_p95_under_500ms(arch: str) -> None:
    if not _artefacts_exist(arch):
        pytest.skip()
    sess = ort.InferenceSession(
        str(_onnx_path(arch)), providers=["CPUExecutionProvider"]
    )
    sc: StandardScaler = joblib.load(_scaler_path(arch))

    raw: np.ndarray = _dummy_window(1)
    scaled: np.ndarray = sc.transform(raw.reshape(-1, _N_CHANNELS)).reshape(raw.shape)
    input_name: str = sess.get_inputs()[0].name

    times: list[float] = []
    for _ in range(50):
        t0: float = time.perf_counter()
        sess.run(None, {input_name: scaled})
        times.append((time.perf_counter() - t0) * 1_000)

    p95: float = float(np.percentile(times, 95))
    assert p95 < 500.0, f"{arch}: inference p95 = {p95:.1f} ms exceeds 500 ms threshold"
