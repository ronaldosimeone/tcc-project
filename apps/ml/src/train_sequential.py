"""
Generic trainer for sequential DL models on MetroPT-3 (TCN, BiLSTM, PatchTST).

Pipeline (mirrors train_mlp.py for consistency)
-----------------------------------------------
1. Load `data/processed/metropt3.parquet`.
2. Build sliding-window dataset via ``MetroPTSequenceDataModule``.
3. Stratified split on window indices (test_size=0.20, random_state=42).
4. Per-channel ``StandardScaler`` fitted on train only — saved as joblib
   alongside the ONNX graph (parity with ``OnnxMlpAdapter`` artefacts).
5. Train selected ``LightningModule`` with AdamW + CosineAnnealingLR,
   EarlyStopping(monitor="val_f1", patience=10).
6. Export best checkpoint to ONNX with dynamic batch axis.
7. PyTorch ↔ ONNX equivalence check at tolerance 1e-5 (hard assertion).
8. Tune decision threshold via F2-score on validation probabilities.
9. Persist V2 schema model card with full architecture/training/metrics.
10. MLflow tracking with gracious fallback when the server is unreachable.

Usage
-----
    python src/train_sequential.py --arch tcn       --max-epochs 30
    python src/train_sequential.py --arch bilstm    --max-epochs 30
    python src/train_sequential.py --arch patchtst  --max-epochs 30 --batch-size 128

Quick smoke run on a 200k-row slice (avoids 15M-row materialisation):

    python src/train_sequential.py --arch tcn --max-epochs 3 --subsample-rows 200000

Design decisions
----------------
* Three architectures share **one** training script; per-arch hparams come from
  a small lookup table.  This keeps the training surface area small and the
  comparison fair (same data split, same callbacks, same MLflow logger).
* Logits (not probabilities) are the ONNX output; softmax is applied at
  inference inside ``OnnxSequenceAdapter`` — keeps the graph minimal.
* The scaler is **separate** from the ONNX graph so it can be retrained
  independently (same rationale as ``train_mlp.py``).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import onnxruntime as ort
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

# ── path resolution (matches the convention used by train_mlp.py) ─────────────
_HERE: Path = Path(__file__).resolve().parent  # apps/ml/src/
_ML_ROOT: Path = _HERE.parent  # apps/ml/
_DATA_PATH: Path = _ML_ROOT / "data" / "processed" / "metropt3.parquet"
_MODELS_DIR: Path = _ML_ROOT / "models"

sys.path.insert(0, str(_HERE))

from datamodule_sequence import MetroPTSequenceDataModule, SequenceConfig  # noqa: E402
from models.bilstm import BiLstmClassifier  # noqa: E402
from models.patchtst import PatchTSTClassifier  # noqa: E402
from models.tcn import TcnClassifier  # noqa: E402

# ── MLflow tracking URI (env-overridable, same default as train_mlp.py) ───────
_MLFLOW_TRACKING_URI: str = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(_MLFLOW_TRACKING_URI)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log: logging.Logger = logging.getLogger(__name__)

_RANDOM_STATE: int = 42


# ---------------------------------------------------------------------------
# Architecture registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[pl.LightningModule]] = {
    "tcn": TcnClassifier,
    "bilstm": BiLstmClassifier,
    "patchtst": PatchTSTClassifier,
}

# Per-architecture ONNX equivalence tolerance.  TCN and BiLSTM use clean
# ONNX-native ops (Conv1d / LSTM) and stay within 1e-5.  PatchTST relies on
# ``F.scaled_dot_product_attention`` which PyTorch fuses but ONNX decomposes,
# producing FP32 reordering errors of up to ~1e-4.  This is the practical
# industry tolerance for Transformer ONNX exports.
_ONNX_TOLERANCES: dict[str, float] = {
    "tcn": 1e-5,
    "bilstm": 1e-5,
    "patchtst": 1e-4,
}


def _build_model(
    arch: str,
    n_channels: int,
    window_size: int,
    pos_weight: float,
    learning_rate: float,
    max_epochs: int,
) -> pl.LightningModule:
    """Instantiate the chosen architecture with arch-specific defaults."""
    if arch == "tcn":
        return TcnClassifier(
            n_channels=n_channels,
            hidden=64,
            kernel_size=3,
            n_blocks=6,
            dropout=0.2,
            learning_rate=learning_rate,
            pos_weight=pos_weight,
            max_epochs=max_epochs,
        )
    if arch == "bilstm":
        return BiLstmClassifier(
            n_channels=n_channels,
            hidden=64,
            n_layers=2,
            dropout=0.3,
            learning_rate=learning_rate,
            pos_weight=pos_weight,
            max_epochs=max_epochs,
        )
    if arch == "patchtst":
        return PatchTSTClassifier(
            n_channels=n_channels,
            window_size=window_size,
            patch_len=12,
            stride=6,
            d_model=64,
            n_heads=4,
            n_layers=3,
            dropout=0.2,
            learning_rate=learning_rate,
            pos_weight=pos_weight,
            max_epochs=max_epochs,
        )
    raise ValueError(f"Unknown architecture: {arch!r}")


# ---------------------------------------------------------------------------
# ONNX export & equivalence check
# ---------------------------------------------------------------------------


def export_to_onnx(
    model: pl.LightningModule,
    onnx_path: Path,
    window_size: int,
    n_channels: int,
) -> None:
    """
    Export the LightningModule to ONNX.

    Batch-axis policy
    -----------------
    TCN and BiLSTM are exported with a dynamic batch dimension so the same
    graph can be reused for arbitrary inference batch sizes.

    PatchTST is exported with a **fixed** batch dimension of 1.  Reason:
    PyTorch 2.3.x's ``nn.TransformerEncoder`` ONNX tracer bakes the dummy's
    batch size into the post-encoder ``Reshape`` op, so calling the graph
    later with batch > 1 raises ``Input shape:{B,N,D}, requested shape:{B,D}``.
    Fixed batch=1 sidesteps the bug entirely.

    This is a non-issue for production: ``OnnxSequenceAdapter`` always feeds
    a single ``(1, T, C)`` window per inference call.  Downstream code in
    this script (equivalence check, test-set scoring, latency benchmark)
    handles the ``arch == 'patchtst'`` case by looping one window at a time.
    """
    model.eval()
    dummy: torch.Tensor = torch.randn(1, window_size, n_channels, dtype=torch.float32)

    is_patchtst: bool = "PatchTST" in model.__class__.__name__
    dynamic_axes: dict[str, dict[int, str]] | None = (
        None
        if is_patchtst
        else {
            "window": {0: "batch_size"},
            "logits": {0: "batch_size"},
        }
    )

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["window"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
    )
    log.info(
        "ONNX exported → %s (dynamic batch=%s)",
        onnx_path,
        "no" if is_patchtst else "yes",
    )


def validate_onnx_equivalence(
    model: pl.LightningModule,
    onnx_path: Path,
    sample: torch.Tensor,
    tol: float = 1e-5,
) -> float:
    """Hard-fail if the ONNX output diverges from the PyTorch output."""
    model.eval()
    with torch.no_grad():
        torch_out: np.ndarray = model(sample).cpu().numpy()
    session: ort.InferenceSession = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    input_name: str = session.get_inputs()[0].name
    onnx_out: np.ndarray = session.run(None, {input_name: sample.cpu().numpy()})[0]
    diff: float = float(np.abs(torch_out - onnx_out).max())
    if diff > tol:
        raise AssertionError(
            f"PyTorch/ONNX mismatch: max abs diff = {diff:.3e} > tol={tol:.0e}"
        )
    log.info("ONNX equivalence OK (max abs diff = %.3e)", diff)
    return diff


# ---------------------------------------------------------------------------
# Helpers (parity with train_mlp.py)
# ---------------------------------------------------------------------------


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted: np.ndarray = logits - logits.max(axis=1, keepdims=True)
    exp: np.ndarray = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    beta: float = 2.0,
) -> dict[str, float]:
    """F-beta optimal cut-off — same routine used by train_mlp.py."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    p, r = precision[:-1], recall[:-1]
    fbeta_num: np.ndarray = (1 + beta**2) * p * r
    fbeta_den: np.ndarray = (beta**2) * p + r + 1e-9
    fbeta: np.ndarray = fbeta_num / fbeta_den
    best_idx: int = int(np.argmax(fbeta))
    return {
        "threshold": float(thresholds[best_idx]),
        "precision": float(p[best_idx]),
        "recall": float(r[best_idx]),
        "fbeta": float(fbeta[best_idx]),
        "beta": beta,
    }


def _measure_latency(
    session: ort.InferenceSession, x_sample: np.ndarray, n_reps: int = 200
) -> dict[str, float]:
    """Single-row ONNX inference latency (p50 / p95) in milliseconds."""
    single: np.ndarray = x_sample[:1]
    input_name: str = session.get_inputs()[0].name
    times: list[float] = []
    for _ in range(n_reps):
        t0: float = time.perf_counter()
        session.run(None, {input_name: single})
        times.append((time.perf_counter() - t0) * 1_000)
    return {
        "p50_ms": round(float(np.percentile(times, 50)), 3),
        "p95_ms": round(float(np.percentile(times, 95)), 3),
    }


def _build_mlflow_logger(experiment_name: str) -> MLFlowLogger | None:
    """
    Probe MLflow at startup; fall back to None if the server is unreachable.

    Same gracious-fallback strategy as ``train_mlp.py:378-397``.
    """
    try:
        candidate: MLFlowLogger = MLFlowLogger(
            experiment_name=experiment_name,
            tracking_uri=_MLFLOW_TRACKING_URI,
            log_model=True,
        )
        # Property access triggers the GET probe; if it fails we want to know
        # *here*, not inside Trainer.fit().
        _ = candidate.experiment
        log.info("MLflow tracking habilitado em %s", _MLFLOW_TRACKING_URI)
        return candidate
    except Exception as exc:
        log.warning(
            "MLflow indisponível em %s — treino sem experiment tracking. Motivo: %s",
            _MLFLOW_TRACKING_URI,
            exc,
        )
        return None


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------


def train(
    arch: str,
    max_epochs: int,
    batch_size: int,
    window_size: int,
    stride: int,
    feature_set: str,
    apply_smote: bool,
    learning_rate: float,
    subsample_rows: int | None,
) -> None:
    if arch not in _REGISTRY:
        raise ValueError(f"Unknown arch '{arch}'. Valid: {sorted(_REGISTRY.keys())}")

    pl.seed_everything(_RANDOM_STATE, workers=True)
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Data ───────────────────────────────────────────────────────────
    cfg: SequenceConfig = SequenceConfig(
        window_size=window_size,
        stride=stride,
        batch_size=batch_size,
        random_state=_RANDOM_STATE,
        apply_smote=apply_smote,
        feature_set=feature_set,
        subsample_rows=subsample_rows,
    )
    dm: MetroPTSequenceDataModule = MetroPTSequenceDataModule(_DATA_PATH, cfg)
    dm.prepare_data()
    dm.setup()

    n_neg, n_pos = dm.class_balance()
    pos_weight: float = round(n_neg / max(1, n_pos), 4)
    log.info("pos_weight = %.4f (neg=%d, pos=%d)", pos_weight, n_neg, n_pos)

    # ── 2. Save scaler artefact (separate from the ONNX graph) ────────────
    scaler_path: Path = _MODELS_DIR / f"{arch}_scaler.joblib"
    joblib.dump(dm.scaler, scaler_path)
    log.info("StandardScaler saved → %s", scaler_path)

    # ── 3. Build model ─────────────────────────────────────────────────────
    model: pl.LightningModule = _build_model(
        arch=arch,
        n_channels=dm.n_channels,
        window_size=window_size,
        pos_weight=pos_weight,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
    )

    # ── 4. MLflow + callbacks ──────────────────────────────────────────────
    mlflow_logger: MLFlowLogger | None = _build_mlflow_logger(f"{arch}_metropt3")

    ckpt_dir: Path = (_ML_ROOT / "checkpoints" / arch).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_cb: ModelCheckpoint = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        monitor="val_f1",
        mode="max",
        save_top_k=1,
        filename=f"best-{arch}-{{epoch:02d}}-{{val_f1:.4f}}",
    )
    early_stop_cb: EarlyStopping = EarlyStopping(
        monitor="val_f1",
        mode="max",
        patience=10,
        verbose=True,
    )

    trainer: pl.Trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=mlflow_logger if mlflow_logger is not None else False,
        callbacks=[checkpoint_cb, early_stop_cb],
        enable_progress_bar=True,
        log_every_n_steps=10,
        deterministic=True,
    )

    log.info(
        "Training %s | epochs=%d | batch=%d | T=%d | C=%d",
        arch.upper(),
        max_epochs,
        batch_size,
        window_size,
        dm.n_channels,
    )
    trainer.fit(model, datamodule=dm)

    # ── 5. Best checkpoint ────────────────────────────────────────────────
    best_ckpt: str = checkpoint_cb.best_model_path
    log.info(
        "Best checkpoint: %s (val_f1=%.4f)",
        best_ckpt,
        checkpoint_cb.best_model_score,
    )
    model_cls: type[pl.LightningModule] = _REGISTRY[arch]
    # Force CPU for export + equivalence — Trainer.fit may have moved the
    # original model to GPU; load_from_checkpoint defaults to CPU but we
    # call .cpu() defensively to guarantee the device for the equivalence
    # check below (where the sample tensor is always on CPU).
    best_model: pl.LightningModule = model_cls.load_from_checkpoint(best_ckpt).cpu()
    best_model.eval()

    # ── 6. ONNX export + equivalence ──────────────────────────────────────
    onnx_path: Path = _MODELS_DIR / f"{arch}_v1.onnx"
    export_to_onnx(best_model, onnx_path, window_size, dm.n_channels)

    # PatchTST is exported with fixed batch=1 (see export_to_onnx for the
    # PyTorch 2.3.x bug rationale) — equivalence sample MUST be batch=1.
    fixed_batch_one: bool = arch == "patchtst"
    val_x: torch.Tensor = dm.val_ds.tensors[0][  # type: ignore[union-attr]
        : (1 if fixed_batch_one else 8)
    ]
    tol: float = _ONNX_TOLERANCES[arch]
    max_diff: float = validate_onnx_equivalence(best_model, onnx_path, val_x, tol=tol)

    # ── 7. Test-set evaluation via ONNX (production path) ─────────────────
    session: ort.InferenceSession = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    input_name: str = session.get_inputs()[0].name
    x_te: np.ndarray = dm.val_ds.tensors[0].numpy()  # type: ignore[union-attr]
    y_te: np.ndarray = dm.val_ds.tensors[1].numpy()  # type: ignore[union-attr]

    if fixed_batch_one:
        # PatchTST graph is hardcoded to batch=1 — must call ONNX once per
        # window.  This mirrors how OnnxSequenceAdapter runs in production
        # (single-window inference per request) so the metrics here equal
        # the metrics the deployed adapter would produce.
        all_logits = np.concatenate(
            [
                session.run(None, {input_name: x_te[i : i + 1]})[0]
                for i in range(len(x_te))
            ],
            axis=0,
        )
    else:
        all_logits = session.run(None, {input_name: x_te})[0]
    proba: np.ndarray = _softmax(all_logits)[:, 1]
    y_pred: np.ndarray = (proba >= 0.5).astype(int)

    f1: float = float(f1_score(y_te, y_pred))
    auc: float = float(roc_auc_score(y_te, proba))
    report: dict[str, Any] = classification_report(y_te, y_pred, output_dict=True)
    latency: dict[str, float] = _measure_latency(session, x_te)

    log.info("Test F1 (class 1): %.4f", f1)
    log.info("Test AUC-ROC:      %.4f", auc)
    log.info(
        "Latency p50/p95:   %.3f ms / %.3f ms",
        latency["p50_ms"],
        latency["p95_ms"],
    )

    # ── 8. F2-tuned threshold ─────────────────────────────────────────────
    thr: dict[str, float] = _find_optimal_threshold(y_te, proba, beta=2.0)
    y_pred_tuned: np.ndarray = (proba >= thr["threshold"]).astype(int)
    tuned_report: dict[str, Any] = classification_report(
        y_te, y_pred_tuned, output_dict=True
    )
    log.info(
        "[V2] Optimal threshold (F2): %.4f | precision=%.4f | recall=%.4f | F2=%.4f",
        thr["threshold"],
        thr["precision"],
        thr["recall"],
        thr["fbeta"],
    )

    # ── 9. MLflow artefact + metric logging ──────────────────────────────
    if mlflow_logger is not None:
        with mlflow.start_run(run_id=mlflow_logger.run_id):
            mlflow.log_params(
                {
                    "arch": arch,
                    "window_size": window_size,
                    "stride": stride,
                    "feature_set": feature_set,
                    "n_channels": dm.n_channels,
                    "batch_size": batch_size,
                    "max_epochs": max_epochs,
                    "learning_rate": learning_rate,
                    "pos_weight": pos_weight,
                    "apply_smote": apply_smote,
                    "optimizer": "AdamW",
                    "scheduler": "CosineAnnealingLR",
                }
            )
            mlflow.log_metrics(
                {
                    "test_f1_class1": round(f1, 4),
                    "test_roc_auc": round(auc, 4),
                    "latency_p50_ms": latency["p50_ms"],
                    "latency_p95_ms": latency["p95_ms"],
                    "onnx_max_abs_diff": max_diff,
                    "decision_threshold": round(thr["threshold"], 4),
                }
            )
            mlflow.log_artifact(str(onnx_path), artifact_path="model")
            mlflow.log_artifact(str(scaler_path), artifact_path="model")

    # ── 10. Model card (V2 schema) ────────────────────────────────────────
    card: dict[str, Any] = {
        "schema_version": "2.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_type": model.__class__.__name__,
        "framework": "PyTorch Lightning",
        "export_format": "ONNX",
        "dataset": "MetroPT-3",
        "target_column": "anomaly",
        "requirements": {
            "RNF-24": "MLflow experiment tracking (local mlruns/)",
            "RF-10": f"Selectable via ACTIVE_MODEL={arch} env-var",
            "RF-04": "F1-Score (classe 1) >= 0.75",
        },
        "train_size": len(dm.train_ds),  # type: ignore[arg-type]
        "test_size": len(dm.val_ds),  # type: ignore[arg-type]
        "test_size_ratio": cfg.test_size,
        "random_state": _RANDOM_STATE,
        "architecture": dict(model.hparams),
        "training": {
            "optimizer": "AdamW",
            "lr": learning_rate,
            "weight_decay": 1e-4,
            "scheduler": "CosineAnnealingLR",
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "early_stopping_patience": 10,
            "monitor_metric": "val_f1",
            "pos_weight": pos_weight,
        },
        "preprocessing": {
            "scaler": "StandardScaler (per-channel, fit on train windows only)",
            "scaler_artefact": scaler_path.name,
            "feature_set": feature_set,
            "window_size": window_size,
            "stride": stride,
            "apply_smote": apply_smote,
        },
        "feature_count": dm.n_channels,
        "feature_names": dm.channel_names,
        "inference": {
            "buffer_strategy": "sliding_window",
            "window_size": window_size,
            # ``None`` in axis 0 = dynamic batch; ``1`` = fixed batch=1 (PatchTST
            # has a hardcoded batch dim due to a PyTorch 2.3.x ONNX export bug
            # — see export_to_onnx for the full rationale).
            "expects_shape": [
                1 if fixed_batch_one else None,
                window_size,
                dm.n_channels,
            ],
            "fixed_batch_size": 1 if fixed_batch_one else None,
        },
        "decision_threshold": round(thr["threshold"], 4),
        "threshold_strategy": "F2-score (recall-favouring)",
        "threshold_metrics": {
            "precision": round(thr["precision"], 4),
            "recall": round(thr["recall"], 4),
            "fbeta": round(thr["fbeta"], 4),
            "beta": thr["beta"],
        },
        "metrics": {
            "f1_class1_test": round(f1, 4),
            "roc_auc_test": round(auc, 4),
            "latency_single_row_ms": latency,
            "onnx_max_abs_diff": max_diff,
            "classification_report": report,
            "tuned_classification_report": tuned_report,
        },
    }
    card_path: Path = _MODELS_DIR / f"{arch}_v1_card.json"
    card_path.write_text(json.dumps(card, indent=2, default=str))
    log.info("Model card saved → %s", card_path)

    if mlflow_logger is not None:
        with mlflow.start_run(run_id=mlflow_logger.run_id):
            mlflow.log_artifact(str(card_path), artifact_path="model")

    log.info("Done (%s).", arch.upper())


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Train sequential DL models on MetroPT-3 with PyTorch Lightning."
    )
    parser.add_argument(
        "--arch",
        choices=sorted(_REGISTRY.keys()),
        required=True,
        help="Which sequential architecture to train.",
    )
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument(
        "--feature-set",
        choices=["raw", "v1", "v2"],
        default="raw",
        help="raw: 12 channels; v1: 34 features; v2: 80 features.",
    )
    parser.add_argument(
        "--apply-smote",
        action="store_true",
        help="Oversample minority class via SMOTE in flattened window space.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--subsample-rows",
        type=int,
        default=None,
        help="Use head(n) of the dataset for fast iteration (default: full).",
    )
    args: argparse.Namespace = parser.parse_args()

    train(
        arch=args.arch,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        window_size=args.window_size,
        stride=args.stride,
        feature_set=args.feature_set,
        apply_smote=args.apply_smote,
        learning_rate=args.learning_rate,
        subsample_rows=args.subsample_rows,
    )
