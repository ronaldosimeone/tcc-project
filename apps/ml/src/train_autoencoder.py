"""
train_autoencoder.py — Train the Conv1D Autoencoder for unsupervised
anomaly detection on the MetroPT-3 compressor dataset.

MLOps checklist
---------------
✓ Trains on healthy windows only (label=0) — no target leakage
✓ Validates on both classes — monitors healthy / fault MSE separation
✓ Per-channel StandardScaler fitted on healthy training windows only
✓ MSE threshold = 99th percentile of healthy val windows (after final epoch)
✓ Exports ONNX artefact with dynamic batch axis (opset 17)
✓ ONNX parity check: max abs diff between PyTorch and ORT outputs
✓ Saves model card (schema_version 2.0) with full training metadata

Usage
-----
    cd apps/ml
    python src/train_autoencoder.py [--epochs N] [--batch-size B] [--lr LR]
                                    [--subsample N] [--percentile P]

Outputs (apps/ml/models/)
--------------------------
  autoencoder_v1.onnx          ONNX graph (input → reconstruction)
  autoencoder_scaler.joblib    Per-channel StandardScaler artefact
  autoencoder_v1_card.json     Model card (mse_threshold, metrics, …)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# Resolve project root so imports work regardless of cwd.
_SRC: Path = Path(__file__).resolve().parent
_ROOT: Path = _SRC.parent
_MODELS_DIR: Path = _ROOT / "models"
_DATA_DIR: Path = _ROOT / "data" / "processed"
sys.path.insert(0, str(_SRC))

from datamodule_unsupervised import MetroPTUnsupervisedDataModule, UnsupervisedConfig
from models.autoencoder import Conv1DAutoencoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Conv1D Autoencoder on MetroPT-3 healthy data."
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--kernel-size", type=int, default=5)
    p.add_argument(
        "--percentile",
        type=float,
        default=99.0,
        help="Percentile of healthy val MSE used as the anomaly threshold.",
    )
    p.add_argument(
        "--subsample",
        type=int,
        default=None,
        help="Use only the first N rows (fast iteration / smoke test).",
    )
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Threshold calibration
# ---------------------------------------------------------------------------


def _compute_mse_threshold(
    model: Conv1DAutoencoder,
    val_loader: torch.utils.data.DataLoader,
    percentile: float,
) -> tuple[float, dict[str, float]]:
    """
    Run one full pass over the validation set and compute the anomaly threshold.

    Returns
    -------
    threshold : float
        ``percentile``-th percentile of per-sample MSE on healthy val windows.
    stats : dict
        Summary statistics logged to the model card.
    """
    model.eval()
    healthy_mses: list[float] = []
    anomaly_mses: list[float] = []

    with torch.no_grad():
        for x, y in val_loader:
            reconstruction: torch.Tensor = model(x)
            per_sample: torch.Tensor = F.mse_loss(
                reconstruction, x, reduction="none"
            ).mean(dim=(1, 2))
            healthy_mask = y == 0
            anomaly_mask = y == 1
            healthy_mses.extend(per_sample[healthy_mask].cpu().tolist())
            anomaly_mses.extend(per_sample[anomaly_mask].cpu().tolist())

    if not healthy_mses:
        raise RuntimeError(
            "No healthy windows in the validation set — cannot compute threshold."
        )

    healthy_arr = np.array(healthy_mses)
    threshold = float(np.percentile(healthy_arr, percentile))

    stats: dict[str, float] = {
        "healthy_mse_mean": float(healthy_arr.mean()),
        "healthy_mse_std": float(healthy_arr.std()),
        f"healthy_mse_p{int(percentile)}": threshold,
    }

    if anomaly_mses:
        anomaly_arr = np.array(anomaly_mses)
        stats["anomaly_mse_mean"] = float(anomaly_arr.mean())
        stats["anomaly_mse_std"] = float(anomaly_arr.std())
        stats["separation_sigma"] = float(
            (anomaly_arr.mean() - healthy_arr.mean()) / (healthy_arr.std() + 1e-8)
        )

    logger.info(
        "MSE threshold (p%.0f of healthy val) = %.6f | "
        "healthy_mean=%.6f | anomaly_mean=%.6f | separation=%.1f σ",
        percentile,
        threshold,
        stats["healthy_mse_mean"],
        stats.get("anomaly_mse_mean", float("nan")),
        stats.get("separation_sigma", float("nan")),
    )
    return threshold, stats


# ---------------------------------------------------------------------------
# ONNX export and parity check
# ---------------------------------------------------------------------------


def _export_onnx(
    model: Conv1DAutoencoder, onnx_path: Path, window_size: int, n_channels: int
) -> None:
    """Export the autoencoder to ONNX with a dynamic batch axis."""
    model.eval()
    dummy: torch.Tensor = torch.zeros(1, window_size, n_channels)

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["reconstruction"],
        dynamic_axes={"input": {0: "batch"}, "reconstruction": {0: "batch"}},
        opset_version=17,
    )
    logger.info("ONNX exported → %s", onnx_path)


def _check_onnx_parity(
    model: Conv1DAutoencoder, onnx_path: Path, window_size: int, n_channels: int
) -> float:
    """
    Verify numerical parity between PyTorch and ONNX Runtime outputs.

    Returns the maximum absolute difference across all elements.
    """
    model.eval()
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((4, window_size, n_channels)).astype(np.float32)
    x_pt = torch.from_numpy(x_np)

    with torch.no_grad():
        pt_out: np.ndarray = model(x_pt).numpy()

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(onnx_path), sess_options=sess_opts)
    ort_out: np.ndarray = session.run(None, {"input": x_np})[0]

    max_diff: float = float(np.abs(pt_out - ort_out).max())
    logger.info("ONNX parity check — max_abs_diff = %.2e", max_diff)
    if max_diff > 1e-4:
        logger.warning("ONNX parity gap > 1e-4; check opset or custom ops.")
    return max_diff


# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------


def _benchmark_latency(
    onnx_path: Path, window_size: int, n_channels: int, n_reps: int = 200
) -> dict[str, float]:
    """Measure single-window inference latency via ONNX Runtime."""
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_opts.intra_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), sess_options=sess_opts)

    x_np = np.zeros((1, window_size, n_channels), dtype=np.float32)
    # Warm-up
    for _ in range(10):
        session.run(None, {"input": x_np})

    latencies: list[float] = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        session.run(None, {"input": x_np})
        latencies.append((time.perf_counter() - t0) * 1_000)

    arr = np.array(latencies)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    pl.seed_everything(args.seed, workers=True)
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)

    parquet_path: Path = _DATA_DIR / "metropt3.parquet"
    cfg = UnsupervisedConfig(
        batch_size=args.batch_size,
        subsample_rows=args.subsample,
    )
    dm = MetroPTUnsupervisedDataModule(parquet_path, config=cfg)
    dm.setup()

    model = Conv1DAutoencoder(
        n_channels=dm.n_channels,
        base_channels=args.base_channels,
        kernel_size=args.kernel_size,
        learning_rate=args.lr,
        max_epochs=args.epochs,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(_MODELS_DIR / "checkpoints"),
        filename="autoencoder_v1-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint_cb, early_stop_cb],
        log_every_n_steps=20,
        enable_model_summary=True,
    )

    logger.info(
        "Starting training — healthy windows only (%d samples)",
        dm.healthy_train_count(),
    )
    trainer.fit(model, dm)

    # Load best checkpoint for threshold calibration and export.
    best_ckpt: str | None = checkpoint_cb.best_model_path
    if best_ckpt:
        logger.info("Loading best checkpoint: %s", best_ckpt)
        model = Conv1DAutoencoder.load_from_checkpoint(best_ckpt)

    # ── Threshold calibration ────────────────────────────────────────────
    mse_threshold, mse_stats = _compute_mse_threshold(
        model, dm.val_dataloader(), percentile=args.percentile
    )

    # ── Scaler export ────────────────────────────────────────────────────
    scaler_path: Path = _MODELS_DIR / "autoencoder_scaler.joblib"
    joblib.dump(dm.scaler, scaler_path)
    logger.info("Scaler saved → %s", scaler_path)

    # ── ONNX export + parity ─────────────────────────────────────────────
    onnx_path: Path = _MODELS_DIR / "autoencoder_v1.onnx"
    _export_onnx(model, onnx_path, cfg.window_size, dm.n_channels)
    max_diff = _check_onnx_parity(model, onnx_path, cfg.window_size, dm.n_channels)

    # ── Latency benchmark ─────────────────────────────────────────────────
    latency = _benchmark_latency(onnx_path, cfg.window_size, dm.n_channels)
    logger.info(
        "Latency: p50=%.3f ms  p95=%.3f ms", latency["p50_ms"], latency["p95_ms"]
    )

    # ── Model card ────────────────────────────────────────────────────────
    val_n_healthy, val_n_anomaly = dm.val_class_balance()
    card: dict = {
        "schema_version": "2.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "Conv1DAutoencoder",
        "framework": "PyTorch Lightning",
        "export_format": "ONNX",
        "dataset": "MetroPT-3",
        "target_column": "anomaly",
        "anomaly_detection_strategy": "reconstruction_error",
        "requirements": {
            "RF-10": "Selectable via ACTIVE_MODEL=autoencoder env-var",
        },
        "train_size": dm.healthy_train_count(),
        "val_size": val_n_healthy + val_n_anomaly,
        "val_healthy": val_n_healthy,
        "val_anomaly": val_n_anomaly,
        "test_size_ratio": cfg.test_size,
        "random_state": cfg.random_state,
        "architecture": {
            "type": "Conv1D Autoencoder",
            "input_shape": [1, cfg.window_size, dm.n_channels],
            "latent_shape": [1, args.base_channels * 4, 8],
            "encoder_channels": [
                dm.n_channels,
                args.base_channels,
                args.base_channels * 2,
                args.base_channels * 4,
            ],
            "decoder_channels": [
                args.base_channels * 4,
                args.base_channels * 2,
                args.base_channels,
                dm.n_channels,
            ],
            "kernel_size": args.kernel_size,
            "stride": 2,
            "activation": "GELU",
            "bottleneck_size": args.base_channels * 4 * 8,
        },
        "training": {
            "loss": "MSE (healthy windows only)",
            "optimizer": "AdamW",
            "lr": args.lr,
            "weight_decay": 1e-4,
            "scheduler": "CosineAnnealingLR",
            "batch_size": args.batch_size,
            "max_epochs": args.epochs,
            "early_stopping_patience": args.patience,
            "monitor_metric": "val_loss",
            "training_data": "healthy_only (label=0)",
            "actual_epochs": trainer.current_epoch,
        },
        "preprocessing": {
            "scaler": "StandardScaler (per-channel, fitted on healthy train windows)",
            "scaler_artefact": "autoencoder_scaler.joblib",
            "window_size": cfg.window_size,
            "stride": cfg.stride,
        },
        "feature_count": dm.n_channels,
        "feature_names": dm.channel_names,
        "mse_threshold": mse_threshold,
        "mse_threshold_percentile": args.percentile,
        "mse_threshold_source": f"percentile_{int(args.percentile)}_healthy_val_windows",
        "mse_calibration_stats": mse_stats,
        "decision_threshold": 0.5,
        "threshold_strategy": (
            "sigmoid((mse - mse_threshold) / (mse_threshold / 3)) ≥ 0.5 "
            "↔ mse ≥ mse_threshold"
        ),
        "inference": {
            "buffer_strategy": "sliding_window",
            "expects_shape": [1, cfg.window_size, dm.n_channels],
            "window_size": cfg.window_size,
            "adapter": "OnnxAutoencoderAdapter",
        },
        "metrics": {
            "onnx_max_abs_diff": max_diff,
            "latency_single_window_ms": latency,
        },
    }

    card_path: Path = _MODELS_DIR / "autoencoder_v1_card.json"
    card_path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    logger.info("Model card saved → %s", card_path)
    logger.info(
        "Done.  MSE threshold = %.6f | ONNX = %s | Scaler = %s",
        mse_threshold,
        onnx_path.name,
        scaler_path.name,
    )


if __name__ == "__main__":
    main()
