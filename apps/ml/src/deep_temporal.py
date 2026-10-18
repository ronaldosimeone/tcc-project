"""
Leakage-free training and scoring for the deep models (MLP, TCN, BiLSTM,
PatchTST, Conv1D autoencoder) — the same rules as the tree models.

Used by ``nested_protocol`` (evaluation and production training).

Rules
-----
* Harmonised sensors only (``Reservoirs``/``Oil_level`` dropped). Windows of
  T = 60 rows never cross the 2020 → 2022 boundary.
* Row-level labels: a window ENDING at row r has the label of row r, and every
  test row gets a score from the window that ends on it — the same rows and
  the same k = 6 persistence as the tree models.
* Scaling statistics come from the TRAINING rows only.
* Class balance ONLY in training: every 2nd positive window + every 20th
  negative window, plus the positive-class weight in the cross-entropy. The
  autoencoder trains on healthy windows only (no failure row in the window).
* Fixed epoch budget with a cosine schedule (epochs and learning rate are
  hyper-parameters chosen by the nested search) — no early stopping, so the
  test fold is never looked at.
* Scores are raw and unbounded: the logit margin (l1 − l0) for classifiers,
  the reconstruction MSE for the autoencoder. Calibration and thresholds are
  learned downstream on out-of-fold scores (``nested_protocol``).
* Data lives on the GPU when available; windows are gathered there.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from src.models.autoencoder import Conv1DAutoencoder
from src.models.bilstm import BiLstmClassifier
from src.models.patchtst import PatchTSTClassifier
from src.models.tcn import TcnClassifier
from src.train_mlp import MlpClassifier

log = logging.getLogger("deep_temporal")

ARCHS: tuple[str, ...] = ("mlp", "tcn", "bilstm", "patchtst", "autoencoder")
WINDOW: int = 60
POS_EVERY: int = 2
NEG_EVERY: int = 20
MLP_NEG_STRIDE: int = 5
BATCH: int = 512
MLP_BATCH: int = 4096  # tabular rows are cheap: larger batches keep the GPU busy
SCORE_BATCH: int = 16384
DEFAULT_PARAMS: dict[str, float] = {"epochs": 8, "lr": 1e-3}
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
    _device: dict[str, torch.Tensor] = field(default_factory=dict, repr=False)

    def on_device(self, arch: str) -> torch.Tensor:
        key = "feats" if arch == "mlp" else "raw"
        if key not in self._device:
            src = self.feats if key == "feats" else self.raw
            self._device[key] = torch.from_numpy(src).to(DEVICE)
        return self._device[key]


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


def make_model(arch: str, n_in: int, pos_weight: float, epochs: int) -> nn.Module:
    if arch == "mlp":
        return MlpClassifier(input_dim=n_in, pos_weight=pos_weight, max_epochs=epochs)
    if arch == "tcn":
        return TcnClassifier(n_channels=n_in, pos_weight=pos_weight, max_epochs=epochs)
    if arch == "bilstm":
        return BiLstmClassifier(
            n_channels=n_in, pos_weight=pos_weight, max_epochs=epochs
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
            max_epochs=epochs,
        )
    if arch == "autoencoder":
        return Conv1DAutoencoder(n_channels=n_in, max_epochs=epochs)
    raise ValueError(arch)


@dataclass
class Fitted:
    arch: str
    model: nn.Module
    mean: np.ndarray
    std: np.ndarray


_OFFSETS: dict[str, torch.Tensor] = {}


def _offsets() -> torch.Tensor:
    if "o" not in _OFFSETS:
        _OFFSETS["o"] = torch.arange(-WINDOW + 1, 1, device=DEVICE)
    return _OFFSETS["o"]


def batch_inputs(
    data: DeepData, arch: str, rows: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> torch.Tensor:
    """Scaled model inputs for ``rows``, built on the device."""
    src = data.on_device(arch)
    idx = torch.from_numpy(np.asarray(rows, dtype=np.int64)).to(DEVICE)
    m = torch.from_numpy(mean.astype(np.float32)).to(DEVICE)
    s = torch.from_numpy(std.astype(np.float32)).to(DEVICE)
    if arch == "mlp":
        return (src[idx] - m) / s
    return (src[idx[:, None] + _offsets()[None, :]] - m) / s


def _train_rows(data: DeepData, arch: str, train_idx: np.ndarray) -> np.ndarray:
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


def fit(
    data: DeepData,
    arch: str,
    train_idx: np.ndarray,
    seed: int,
    params: dict[str, Any] | None = None,
) -> Fitted:
    p = {**DEFAULT_PARAMS, **(params or {})}
    epochs, lr = int(p["epochs"]), float(p["lr"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    src = data.feats[train_idx] if arch == "mlp" else data.raw[train_idx]
    mean = src.mean(axis=0)
    std = src.std(axis=0) + 1e-6
    rows = _train_rows(data, arch, train_idx)
    yv = data.y[rows]
    pos_weight = float((len(yv) - yv.sum()) / max(int(yv.sum()), 1))
    n_in = data.feats.shape[1] if arch == "mlp" else data.raw.shape[1]
    model = make_model(arch, n_in, pos_weight, epochs).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    batch = MLP_BATCH if arch == "mlp" else BATCH
    steps = epochs * math.ceil(len(rows) / batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))
    rng = np.random.default_rng(seed)
    weights = (
        None
        if arch == "autoencoder"
        else torch.tensor([1.0, pos_weight], dtype=torch.float32, device=DEVICE)
    )
    y_dev = torch.from_numpy(data.y).to(DEVICE)
    model.train()
    for epoch in range(epochs):
        order = rng.permutation(rows)
        total = 0.0
        for i in range(0, len(order), batch):
            b = order[i : i + batch]
            x = batch_inputs(data, arch, b, mean, std)
            if arch == "autoencoder":
                loss = F.mse_loss(model(x), x)
            else:
                yb = y_dev[torch.from_numpy(b).to(DEVICE)].long()
                loss = F.cross_entropy(model(x), yb, weight=weights)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss.detach()) * len(b)
        log.debug(
            "%s epoch %d/%d loss %.4f", arch, epoch + 1, epochs, total / len(rows)
        )
    model.eval()
    return Fitted(arch, model, mean, std)


@torch.no_grad()
def score(data: DeepData, fitted: Fitted, rows: np.ndarray) -> np.ndarray:
    """Raw score: logit margin (classifiers) or reconstruction MSE (autoencoder).
    Rows without a full window (first 59 rows of a dataset) get the minimum
    observed score, i.e. "no evidence of failure"."""
    out = np.full(len(rows), np.nan, dtype=np.float64)
    ok = rows if fitted.arch == "mlp" else rows[data.valid_end[rows]]
    pos = np.searchsorted(rows, ok)
    for i in range(0, len(ok), SCORE_BATCH):
        b = ok[i : i + SCORE_BATCH]
        x = batch_inputs(data, fitted.arch, b, fitted.mean, fitted.std)
        y = fitted.model(x)
        if fitted.arch == "autoencoder":
            s = ((y - x) ** 2).mean(dim=(1, 2))
        else:
            s = y[:, 1] - y[:, 0]
        out[pos[i : i + len(b)]] = s.float().cpu().numpy()
    if np.isnan(out).any():
        out[np.isnan(out)] = np.nanmin(out) if np.isfinite(out).any() else 0.0
    return out


class CalibratedClassifier(nn.Module):
    """Classifier + Platt calibration baked into the exported graph.

    Outputs logits ``[0, a·(l1 − l0) + b]``: the serving adapters apply a
    softmax, so they return the CALIBRATED probability with no code change.
    """

    def __init__(self, base: nn.Module, a: float, b: float) -> None:
        super().__init__()
        self.base = base
        self.register_buffer("a", torch.tensor(float(a)))
        self.register_buffer("b", torch.tensor(float(b)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.base(x)
        z = self.a * (logits[:, 1] - logits[:, 0]) + self.b
        return torch.stack([torch.zeros_like(z), z], dim=1)


def describe() -> dict[str, Any]:
    return {
        "window": WINDOW,
        "batch": BATCH,
        "default_params": DEFAULT_PARAMS,
        "train_windows": f"every {POS_EVERY}nd positive + every {NEG_EVERY}th negative",
        "mlp_rows": f"all positives + every {MLP_NEG_STRIDE}th negative",
        "labels": "row-level: window ending at row r has the label of row r",
        "raw_score": "logit margin (classifiers) / reconstruction MSE (autoencoder)",
        "device": str(DEVICE),
    }
