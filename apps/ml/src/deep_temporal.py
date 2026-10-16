"""
Leakage-free training and scoring for the deep models (MLP, TCN, BiLSTM,
PatchTST, Conv1D autoencoder) — the same rules as the tree models.

Used by ``evaluate_all_models`` (walk-forward + confusion matrices) and
``train_deep_leakfree`` (production artefacts).

Rules
-----
* Harmonised sensors only (``Reservoirs``/``Oil_level`` dropped). Windows of
  T = 60 rows never cross the 2020 → 2022 boundary.
* Row-level labels: a window ENDING at row r has the label of row r, and every
  test row gets a score from the window that ends on it. This makes the deep
  models comparable, row by row and with the same k = 6 persistence, with the
  tree models (the legacy datamodule labelled a window positive when ANY row
  inside it failed and evaluated one window per 10 rows).
* Scaling statistics come from the TRAINING rows of each fold only.
* Class balance ONLY in training: every 2nd positive window + every 20th
  negative window, plus the positive-class weight in the cross-entropy. The
  autoencoder trains on healthy windows only (no failure row in the window).
* Fixed epoch budget with a cosine schedule — no early stopping, so no
  validation set is needed and the test fold is never looked at.
* Threshold: cost-optimal (RN-02, k = 6) on a TEMPORAL validation — an inner
  model trained before the most recent training failure and scored on it.
  (Tree models use leave-one-failure-out OOF; one extra training per fold is
  the affordable equivalent for networks that take minutes to train.)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from src.evaluate_temporal import EVAL_KW, PERSISTENCE_K, PURGE_S
from src.event_metrics import cost_optimal_threshold
from src.models.autoencoder import Conv1DAutoencoder
from src.models.bilstm import BiLstmClassifier
from src.models.patchtst import PatchTSTClassifier
from src.models.tcn import TcnClassifier
from src.temporal_validation import FailureEvent
from src.train_mlp import MlpClassifier

log = logging.getLogger("deep_temporal")

ARCHS: tuple[str, ...] = ("mlp", "tcn", "bilstm", "patchtst", "autoencoder")
WINDOW: int = 60
POS_EVERY: int = 2
NEG_EVERY: int = 20
MLP_NEG_STRIDE: int = 5
EPOCHS: int = 8
BATCH: int = 512
SCORE_BATCH: int = 8192
LR: float = 1e-3
PROB_GRID: list[float] = [
    *np.round(np.arange(0.05, 0.95, 0.05), 2).tolist(),
    *np.round(np.arange(0.95, 0.996, 0.005), 3).tolist(),
]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class DeepData:
    """Everything the deep models need, aligned row by row with ``ts``/``y``."""

    raw: np.ndarray  # (N, C) harmonised sensor channels, float32
    feats: np.ndarray  # (N, F) harmonised tabular features (MLP), float32
    y: np.ndarray  # (N,) int
    ts: pd.Series  # (N,) timestamps
    valid_end: np.ndarray  # (N,) bool — a full window ends here
    clean_window: np.ndarray  # (N,) bool — no failure row inside the window
    channels: list[str]
    feature_names: list[str]


def build_data(
    raw_df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, ts: pd.Series, is22: np.ndarray
) -> DeepData:
    channels = [c for c in raw_df.columns if c not in ("timestamp", "anomaly")]
    raw = raw_df[channels].to_numpy(np.float32)
    n = len(raw)
    seg_start = np.where(is22, int(np.argmax(is22)), 0)
    valid_end = (np.arange(n) - seg_start) >= WINDOW - 1
    yv = y.to_numpy().astype(int)
    csum = np.concatenate([[0], np.cumsum(yv)])
    lo = np.maximum(np.arange(n) - WINDOW + 1, 0)
    clean = (csum[np.arange(n) + 1] - csum[lo]) == 0
    return DeepData(
        raw=raw,
        feats=X.to_numpy(np.float32),
        y=yv,
        ts=ts.reset_index(drop=True),
        valid_end=valid_end,
        clean_window=clean & valid_end,
        channels=channels,
        feature_names=list(X.columns),
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def make_model(arch: str, n_in: int, pos_weight: float) -> nn.Module:
    if arch == "mlp":
        return MlpClassifier(input_dim=n_in, pos_weight=pos_weight, max_epochs=EPOCHS)
    if arch == "tcn":
        return TcnClassifier(n_channels=n_in, pos_weight=pos_weight, max_epochs=EPOCHS)
    if arch == "bilstm":
        return BiLstmClassifier(
            n_channels=n_in, pos_weight=pos_weight, max_epochs=EPOCHS
        )
    if arch == "patchtst":
        return PatchTSTClassifier(
            n_channels=n_in,
            window_size=WINDOW,
            patch_len=12,
            stride=6,
            d_model=64,
            n_heads=4,
            n_layers=3,
            pos_weight=pos_weight,
            max_epochs=EPOCHS,
        )
    if arch == "autoencoder":
        return Conv1DAutoencoder(n_channels=n_in, max_epochs=EPOCHS)
    raise ValueError(arch)


@dataclass
class Fitted:
    arch: str
    model: nn.Module
    mean: np.ndarray
    std: np.ndarray


def _windows(raw_scaled: np.ndarray, ends: np.ndarray) -> np.ndarray:
    offs = np.arange(-WINDOW + 1, 1)
    return raw_scaled[ends[:, None] + offs[None, :]]


def _batch_inputs(
    data: DeepData, arch: str, rows: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> torch.Tensor:
    if arch == "mlp":
        x = (data.feats[rows] - mean) / std
    else:
        x = (_windows(data.raw, rows) - mean) / std
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(DEVICE)


def _train_rows(
    data: DeepData, arch: str, train_idx: np.ndarray, seed: int
) -> np.ndarray:
    if arch == "mlp":
        pos = train_idx[data.y[train_idx] == 1]
        neg = train_idx[data.y[train_idx] == 0][::MLP_NEG_STRIDE]
        return np.sort(np.concatenate([pos, neg]))
    ends = train_idx[data.valid_end[train_idx]]
    if arch == "autoencoder":
        return ends[data.clean_window[ends]][:: NEG_EVERY // 2]
    pos = ends[data.y[ends] == 1][::POS_EVERY]
    neg = ends[data.y[ends] == 0][::NEG_EVERY]
    return np.sort(np.concatenate([pos, neg]))


def fit(data: DeepData, arch: str, train_idx: np.ndarray, seed: int) -> Fitted:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if arch == "mlp":
        src = data.feats[train_idx]
    else:
        src = data.raw[train_idx]
    mean = src.mean(axis=0)
    std = src.std(axis=0) + 1e-6
    rows = _train_rows(data, arch, train_idx, seed)
    yv = data.y[rows]
    pos_weight = float((len(yv) - yv.sum()) / max(int(yv.sum()), 1))
    n_in = data.feats.shape[1] if arch == "mlp" else data.raw.shape[1]
    model = make_model(arch, n_in, pos_weight).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    steps = EPOCHS * math.ceil(len(rows) / BATCH)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))
    rng = np.random.default_rng(seed)
    weights = (
        None
        if arch == "autoencoder"
        else torch.tensor([1.0, pos_weight], dtype=torch.float32, device=DEVICE)
    )
    model.train()
    for epoch in range(EPOCHS):
        order = rng.permutation(rows)
        total = 0.0
        for i in range(0, len(order), BATCH):
            b = order[i : i + BATCH]
            x = _batch_inputs(data, arch, b, mean, std)
            if arch == "autoencoder":
                loss = F.mse_loss(model(x), x)
            else:
                yb = torch.from_numpy(data.y[b]).long().to(DEVICE)
                loss = F.cross_entropy(model(x), yb, weight=weights)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss) * len(b)
        log.info("%s epoch %d/%d loss %.4f", arch, epoch + 1, EPOCHS, total / len(rows))
    model.eval()
    return Fitted(arch, model, mean, std)


@torch.no_grad()
def score(data: DeepData, fitted: Fitted, rows: np.ndarray) -> np.ndarray:
    """P(failure) for classifiers, reconstruction MSE for the autoencoder.
    Rows without a full window (first 59 rows of a dataset) score 0."""
    out = np.zeros(len(rows), dtype=np.float64)
    ok = rows if fitted.arch == "mlp" else rows[data.valid_end[rows]]
    pos = np.searchsorted(rows, ok)
    for i in range(0, len(ok), SCORE_BATCH):
        b = ok[i : i + SCORE_BATCH]
        x = _batch_inputs(data, fitted.arch, b, fitted.mean, fitted.std)
        y = fitted.model(x)
        if fitted.arch == "autoencoder":
            s = ((y - x) ** 2).mean(dim=(1, 2))
        else:
            s = torch.softmax(y, dim=1)[:, 1]
        out[pos[i : i + len(b)]] = s.float().cpu().numpy()
    return out


# ---------------------------------------------------------------------------
# Threshold on a temporal validation (never the test fold)
# ---------------------------------------------------------------------------


def temporal_threshold(
    data: DeepData,
    arch: str,
    train_idx: np.ndarray,
    events: list[FailureEvent],
    seed: int,
) -> tuple[float, str]:
    ts = data.ts
    t_lo, t_hi = ts.iloc[train_idx].min(), ts.iloc[train_idx].max()
    inside = [e for e in events if e.start >= t_lo and e.end <= t_hi]
    if len(inside) < 2:
        return (
            _fallback(data, arch, train_idx, seed),
            "fallback (<2 failures in training)",
        )
    last, prev = inside[-1], inside[-2]
    cut = prev.end + (last.start - prev.end) / 2  # segment boundary, as in RF-28
    t_train = ts.iloc[train_idx]
    inner_train = train_idx[(t_train < cut - pd.Timedelta(seconds=PURGE_S)).to_numpy()]
    val = train_idx[(t_train >= cut).to_numpy()]
    if data.y[inner_train].sum() == 0 or val.size == 0:
        return _fallback(data, arch, train_idx, seed), "fallback (empty inner split)"
    inner = fit(data, arch, inner_train, seed)
    s = score(data, inner, val)
    grid = (
        np.quantile(
            score(data, inner, inner_train[data.y[inner_train] == 0][::50]),
            [0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995],
        ).tolist()
        if arch == "autoencoder"
        else PROB_GRID
    )
    thr, _ = cost_optimal_threshold(
        ts.iloc[val].reset_index(drop=True),
        s,
        [last],
        k=PERSISTENCE_K,
        candidates=grid,
        **EVAL_KW,
    )
    return float(thr), "temporal validation on the most recent training failure"


def _fallback(data: DeepData, arch: str, train_idx: np.ndarray, seed: int) -> float:
    if arch != "autoencoder":
        return 0.5
    # Autoencoder: 99th percentile of healthy training reconstruction error.
    fitted = fit(data, arch, train_idx, seed)
    healthy = train_idx[data.clean_window[train_idx]][::50]
    return float(np.quantile(score(data, fitted, healthy), 0.99))


def describe() -> dict[str, Any]:
    return {
        "window": WINDOW,
        "epochs": EPOCHS,
        "batch": BATCH,
        "lr": LR,
        "train_windows": f"every {POS_EVERY}nd positive + every {NEG_EVERY}th negative",
        "mlp_rows": f"all positives + every {MLP_NEG_STRIDE}th negative",
        "labels": "row-level: window ending at row r has the label of row r",
        "threshold": "cost-optimal on temporal validation (most recent training "
        "failure); fallback 0.5 / AE 99th pct healthy MSE when <2 failures",
        "device": str(DEVICE),
    }
