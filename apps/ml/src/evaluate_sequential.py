"""
5-fold cross-validation for sequential DL models on MetroPT-3.

Two evaluation strategies (the TCC comparison table requires both)
------------------------------------------------------------------
A) ``StratifiedKFold`` — the same split used by RF and XGBoost baselines.
   Adjacent windows share T-stride samples → known data leakage on folds.
   Reported for **apples-to-apples comparison with the baseline models**.

B) ``GroupKFold`` grouped by operational day (derived from the parquet's
   ``timestamp`` column) — no window crosses a day boundary, so no leakage.
   This is the **honest metric for the TCC dissertation**.

Both results are written to ``models/eval_<arch>_cv.json``.

Usage
-----
    python src/evaluate_sequential.py --arch tcn   --max-epochs 10
    python src/evaluate_sequential.py --arch bilstm --max-epochs 10
    python src/evaluate_sequential.py --arch patchtst --max-epochs 10

    # Run all three architectures in one shot:
    python src/evaluate_sequential.py --arch all   --max-epochs 10

Reduce to a dataset slice for rapid iteration:
    python src/evaluate_sequential.py --arch tcn --max-epochs 5 --subsample-rows 300000

Design notes
------------
* A fresh model (randomly initialised) is trained from scratch for each fold —
  no weight sharing between folds (avoids optimistic bias).
* The ``DataModule.setup()`` path is replicated here to allow explicit fold
  assignment.  The DataModule itself is NOT used because GroupKFold requires
  access to the raw window-to-day mapping before the standard train/val split.
* ONNX export is intentionally *skipped* per fold — we care about fold metrics
  only; the final production ONNX artefact comes from ``train_sequential.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

_HERE: Path = Path(__file__).resolve().parent
_ML_ROOT: Path = _HERE.parent
_DATA_PATH: Path = _ML_ROOT / "data" / "processed" / "metropt3.parquet"
_MODELS_DIR: Path = _ML_ROOT / "models"

sys.path.insert(0, str(_HERE))

from datamodule_sequence import MetroPTSequenceDataModule, SequenceConfig  # noqa: E402
from models import BiLstmClassifier, PatchTSTClassifier, TcnClassifier  # noqa: E402
from train_sequential import (
    _build_model,
    _find_optimal_threshold,
    _softmax,
)  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log: logging.Logger = logging.getLogger(__name__)

_N_SPLITS: int = 5
_RANDOM_STATE: int = 42
_BATCH_SIZE: int = 256

_ALL_ARCHS: list[str] = ["tcn", "bilstm", "patchtst"]


# ---------------------------------------------------------------------------
# Window construction (mirrors DataModule internals for fold-level control)
# ---------------------------------------------------------------------------


def _make_windows_with_groups(
    x: np.ndarray,
    y: np.ndarray,
    groups_raw: np.ndarray,
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sliding windows and assign each window the group of its first row.

    Returns
    -------
    x_win : (n, T, C) float32
    y_win : (n,)     int64
    g_win : (n,)     object  — group label (day string) of the window start
    """
    n: int = (len(x) - window_size) // stride + 1
    c: int = x.shape[1]

    x_win: np.ndarray = np.lib.stride_tricks.as_strided(
        x,
        shape=(n, window_size, c),
        strides=(x.strides[0] * stride, x.strides[0], x.strides[1]),
        writeable=False,
    )
    x_win = np.ascontiguousarray(x_win, dtype=np.float32)

    y_strided: np.ndarray = np.lib.stride_tricks.as_strided(
        y,
        shape=(n, window_size),
        strides=(y.strides[0] * stride, y.strides[0]),
        writeable=False,
    )
    y_win: np.ndarray = y_strided.any(axis=1).astype(np.int64)

    g_win: np.ndarray = groups_raw[np.arange(n) * stride]

    return x_win, y_win, g_win


def _run_fold(
    arch: str,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    n_channels: int,
    window_size: int,
    max_epochs: int,
    learning_rate: float,
    fold_idx: int,
) -> dict[str, Any]:
    """Train one fold and return fold metrics."""
    # Per-channel scaler fit on this fold's train split only
    scaler: StandardScaler = StandardScaler()
    b, t, c = x_tr.shape
    flat_tr: np.ndarray = scaler.fit_transform(x_tr.reshape(-1, c))
    flat_te: np.ndarray = scaler.transform(x_te.reshape(-1, c))
    x_tr_s: np.ndarray = flat_tr.reshape(b, t, c).astype(np.float32)
    x_te_s: np.ndarray = flat_te.reshape(x_te.shape[0], t, c).astype(np.float32)

    pos_weight: float = round(
        int((y_tr == 0).sum()) / max(1, int((y_tr == 1).sum())), 4
    )

    train_ds: TensorDataset = TensorDataset(
        torch.from_numpy(x_tr_s), torch.from_numpy(y_tr)
    )
    val_ds: TensorDataset = TensorDataset(
        torch.from_numpy(x_te_s), torch.from_numpy(y_te)
    )
    train_loader: DataLoader = DataLoader(
        train_ds, batch_size=_BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader: DataLoader = DataLoader(
        val_ds, batch_size=_BATCH_SIZE * 2, shuffle=False, num_workers=0
    )

    model: pl.LightningModule = _build_model(
        arch=arch,
        n_channels=n_channels,
        window_size=window_size,
        pos_weight=pos_weight,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
    )

    from pytorch_lightning.callbacks import (
        EarlyStopping,
    )  # local to avoid top-level dep

    trainer: pl.Trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[
            EarlyStopping(monitor="val_f1", mode="max", patience=7, verbose=False)
        ],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=20,
        deterministic=True,
    )
    trainer.fit(model, train_loader, val_loader)

    # Trainer.fit may have moved the model to GPU; force CPU before manual
    # inference loop so xb (CPU tensor) and the model live on the same device.
    model = model.cpu()
    model.eval()
    all_proba: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in val_loader:
            logits: torch.Tensor = model(xb)
            proba: np.ndarray = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_proba.append(proba)

    y_proba: np.ndarray = np.concatenate(all_proba)
    y_pred: np.ndarray = (y_proba >= 0.5).astype(int)

    fold_f1: float = float(f1_score(y_te, y_pred))
    fold_auc: float = float(roc_auc_score(y_te, y_proba))
    thr: dict[str, float] = _find_optimal_threshold(y_te, y_proba, beta=2.0)
    y_pred_t: np.ndarray = (y_proba >= thr["threshold"]).astype(int)

    log.info(
        "  fold %d: F1=%.4f AUC=%.4f | F2-threshold=%.4f (F1-tuned=%.4f)",
        fold_idx,
        fold_f1,
        fold_auc,
        thr["threshold"],
        float(f1_score(y_te, y_pred_t)),
    )

    return {
        "fold": fold_idx,
        "n_train_windows": len(x_tr),
        "n_val_windows": len(x_te),
        "pos_weight": pos_weight,
        "f1_class1_default": round(fold_f1, 4),
        "roc_auc": round(fold_auc, 4),
        "decision_threshold_f2": round(thr["threshold"], 4),
        "f1_class1_tuned": round(float(f1_score(y_te, y_pred_t)), 4),
        "classification_report": classification_report(y_te, y_pred, output_dict=True),
    }


# ---------------------------------------------------------------------------
# Evaluation routine
# ---------------------------------------------------------------------------


def evaluate(
    arch: str,
    max_epochs: int,
    window_size: int,
    stride: int,
    learning_rate: float,
    subsample_rows: int | None,
) -> None:
    pl.seed_everything(_RANDOM_STATE, workers=True)
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading parquet from %s …", _DATA_PATH)
    raw: pd.DataFrame = pd.read_parquet(_DATA_PATH)
    if subsample_rows is not None:
        raw = raw.head(int(subsample_rows)).reset_index(drop=True)
        log.info("Subsampled to %d rows", len(raw))

    # Derive operational day from timestamp for GroupKFold
    if "timestamp" in raw.columns:
        groups_raw: np.ndarray = (
            pd.to_datetime(raw["timestamp"]).dt.date.astype(str).to_numpy()
        )
    else:
        # Fallback: synthetic day groups based on 86400-sample blocks (1 Hz assumption)
        groups_raw = (np.arange(len(raw)) // 86_400).astype(str)
        log.warning(
            "No 'timestamp' column — using synthetic day groups (86400-row blocks)."
        )

    # Identify raw channels (12) — same selection as DataModule "raw" mode
    _RAW_COLS: list[str] = [
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
    channels: list[str] = [c for c in _RAW_COLS if c in raw.columns]
    n_channels: int = len(channels)
    log.info("Using %d channels: %s", n_channels, channels)

    x_full: np.ndarray = raw[channels].astype(np.float32).to_numpy()
    y_full: np.ndarray = raw["anomaly"].astype(np.int64).to_numpy()

    x_win, y_win, g_win = _make_windows_with_groups(
        x_full, y_full, groups_raw, window_size, stride
    )
    log.info(
        "Built %d windows | class dist: neg=%d, pos=%d | unique days: %d",
        len(x_win),
        int((y_win == 0).sum()),
        int((y_win == 1).sum()),
        len(np.unique(g_win)),
    )

    results: dict[str, Any] = {
        "schema_version": "2.0",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "arch": arch,
        "window_size": window_size,
        "stride": stride,
        "n_channels": n_channels,
        "channel_names": channels,
        "max_epochs_per_fold": max_epochs,
        "n_splits": _N_SPLITS,
    }

    # ── Strategy A: StratifiedKFold (parity with baselines) ───────────────
    log.info("[A] StratifiedKFold (n_splits=%d) — %s", _N_SPLITS, arch.upper())
    skf: StratifiedKFold = StratifiedKFold(
        n_splits=_N_SPLITS, shuffle=True, random_state=_RANDOM_STATE
    )
    strat_folds: list[dict[str, Any]] = []
    for fold_i, (idx_tr, idx_te) in enumerate(skf.split(x_win, y_win)):
        log.info("[A] fold %d/%d", fold_i + 1, _N_SPLITS)
        strat_folds.append(
            _run_fold(
                arch,
                x_win[idx_tr],
                y_win[idx_tr],
                x_win[idx_te],
                y_win[idx_te],
                n_channels,
                window_size,
                max_epochs,
                learning_rate,
                fold_i + 1,
            )
        )

    f1_strat: list[float] = [f["f1_class1_default"] for f in strat_folds]
    auc_strat: list[float] = [f["roc_auc"] for f in strat_folds]
    results["stratified_kfold"] = {
        "note": "Contains window-level leakage — use for comparison with RF/XGB baselines only",
        "f1_mean": round(float(np.mean(f1_strat)), 4),
        "f1_std": round(float(np.std(f1_strat)), 4),
        "auc_mean": round(float(np.mean(auc_strat)), 4),
        "auc_std": round(float(np.std(auc_strat)), 4),
        "folds": strat_folds,
    }

    # ── Strategy B: GroupKFold by operational day (honest split) ──────────
    log.info("[B] GroupKFold (n_splits=%d) — %s", _N_SPLITS, arch.upper())
    gkf: GroupKFold = GroupKFold(n_splits=_N_SPLITS)
    group_folds: list[dict[str, Any]] = []
    for fold_i, (idx_tr, idx_te) in enumerate(gkf.split(x_win, y_win, g_win)):
        log.info("[B] fold %d/%d", fold_i + 1, _N_SPLITS)
        group_folds.append(
            _run_fold(
                arch,
                x_win[idx_tr],
                y_win[idx_tr],
                x_win[idx_te],
                y_win[idx_te],
                n_channels,
                window_size,
                max_epochs,
                learning_rate,
                fold_i + 1,
            )
        )

    f1_group: list[float] = [f["f1_class1_default"] for f in group_folds]
    auc_group: list[float] = [f["roc_auc"] for f in group_folds]
    results["group_kfold_by_day"] = {
        "note": "Honest split — no window leakage across fold boundaries",
        "f1_mean": round(float(np.mean(f1_group)), 4),
        "f1_std": round(float(np.std(f1_group)), 4),
        "auc_mean": round(float(np.mean(auc_group)), 4),
        "auc_std": round(float(np.std(auc_group)), 4),
        "folds": group_folds,
    }

    out_path: Path = _MODELS_DIR / f"eval_{arch}_cv.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    log.info("Evaluation results saved → %s", out_path)

    log.info(
        "\n%-10s  %-22s  %-22s",
        arch.upper(),
        "StratifiedKFold (biased)",
        "GroupKFold (honest)",
    )
    log.info(
        "%-10s  F1 = %.4f ± %.4f      F1 = %.4f ± %.4f",
        "",
        results["stratified_kfold"]["f1_mean"],
        results["stratified_kfold"]["f1_std"],
        results["group_kfold_by_day"]["f1_mean"],
        results["group_kfold_by_day"]["f1_std"],
    )
    log.info(
        "%-10s  AUC= %.4f ± %.4f      AUC= %.4f ± %.4f",
        "",
        results["stratified_kfold"]["auc_mean"],
        results["stratified_kfold"]["auc_std"],
        results["group_kfold_by_day"]["auc_mean"],
        results["group_kfold_by_day"]["auc_std"],
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="5-fold CV for sequential DL models on MetroPT-3."
    )
    parser.add_argument(
        "--arch",
        choices=_ALL_ARCHS + ["all"],
        required=True,
        help="Architecture to evaluate, or 'all' to run all three sequentially.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=10,
        help="Max epochs per fold (default: 10). Use 3-5 for smoke runs.",
    )
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--subsample-rows", type=int, default=None)

    args: argparse.Namespace = parser.parse_args()
    archs_to_run: list[str] = _ALL_ARCHS if args.arch == "all" else [args.arch]

    for a in archs_to_run:
        log.info("=" * 60)
        log.info("Evaluating: %s", a.upper())
        log.info("=" * 60)
        evaluate(
            arch=a,
            max_epochs=args.max_epochs,
            window_size=args.window_size,
            stride=args.stride,
            learning_rate=args.learning_rate,
            subsample_rows=args.subsample_rows,
        )
