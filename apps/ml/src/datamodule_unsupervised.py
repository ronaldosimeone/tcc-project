"""
MetroPTUnsupervisedDataModule — sliding-window DataModule for the Conv1D Autoencoder.

Key difference from MetroPTSequenceDataModule
---------------------------------------------
The autoencoder is trained exclusively on **healthy** windows (label=0) so
that it learns the manifold of normal compressor operation.  Reconstruction
error then serves as an anomaly score at inference — fault windows fall
outside the learned manifold and yield high MSE.

Split strategy
--------------
1. Build all sliding windows from the full parquet (identical to the
   supervised DataModule, preserving temporal ordering).
2. Stratified 80/20 split on *all* windows (same seed as supervised DM for
   reproducibility).
3. Fit a per-channel ``StandardScaler`` on **training healthy windows only**
   (no target leakage, no contamination from fault statistics).
4. Apply scaler to all splits.
5. **Train DataLoader** — yields only class-0 windows.
   **Val  DataLoader** — yields all windows (both classes) so the
   LightningModule can track the MSE separation between healthy and fault
   windows during training.

This asymmetry is intentional:
* Training on healthy-only forces the network to specialise on the normal
  manifold; fault windows seen at validation are *held-out signals* that
  probe whether the threshold generalises.
* The scaler is still fitted on healthy training windows only (never on fault
  data) to avoid injecting fault-period statistics into the normalisation.

Output tensors
--------------
* X : float32 of shape (B, T, C)
* y : int64   of shape (B,)  — 0 for healthy, 1 for fault
  (Training loader always emits y=0; included for interface parity with the
   supervised DataModule so the same `validation_step` signature works.)

Unchanged from MetroPTSequenceDataModule
-----------------------------------------
* Vectorised stride-tricks window construction
* Recall-favouring window label (any fault sample inside the window → label=1)
* Per-channel StandardScaler artefact (``autoencoder_scaler.joblib``)
* SequenceConfig frozen dataclass for full MLflow logging compatibility
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

logger: logging.Logger = logging.getLogger(__name__)

_RAW_CHANNELS: list[str] = [
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


@dataclass(frozen=True)
class UnsupervisedConfig:
    """
    Static configuration for the unsupervised DataModule.

    Frozen dataclass — fully hashable and safe to log to MLflow as-is.
    ``apply_smote`` is absent: class-balancing makes no sense for a
    reconstruction-based training objective.
    """

    window_size: int = 60
    stride: int = 10
    test_size: float = 0.20
    random_state: int = 42
    batch_size: int = 256
    num_workers: int = 0
    subsample_rows: int | None = None
    channels: tuple[str, ...] = field(default_factory=lambda: tuple(_RAW_CHANNELS))


class MetroPTUnsupervisedDataModule(pl.LightningDataModule):
    """
    LightningDataModule for the Conv1D Autoencoder.

    Training split: healthy-only windows (label=0).
    Validation split: all windows — used for MSE separation tracking.
    """

    def __init__(
        self,
        parquet_path: Path,
        config: UnsupervisedConfig | None = None,
    ) -> None:
        super().__init__()
        self.parquet_path: Path = parquet_path
        self.cfg: UnsupervisedConfig = (
            config if config is not None else UnsupervisedConfig()
        )
        self.scaler: StandardScaler | None = None
        self._channels: list[str] = []

        self.train_ds: TensorDataset | None = None
        self.val_ds: TensorDataset | None = None

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def prepare_data(self) -> None:
        if not self.parquet_path.exists():
            raise FileNotFoundError(
                f"Parquet not found at {self.parquet_path}. "
                "Run `python src/ingest_metropt.py` first."
            )

    def setup(self, stage: str | None = None) -> None:
        raw: pd.DataFrame = pd.read_parquet(self.parquet_path)
        if self.cfg.subsample_rows is not None:
            raw = raw.head(int(self.cfg.subsample_rows)).reset_index(drop=True)
            logger.info("Subsampled to %d rows", len(raw))

        self._channels = [c for c in self.cfg.channels if c in raw.columns]
        if not self._channels:
            raise ValueError(
                f"No channels resolved. Available columns: {list(raw.columns)[:20]}…"
            )

        x_full: np.ndarray = raw[self._channels].astype(np.float32).to_numpy()
        y_full: np.ndarray = raw["anomaly"].astype(np.int64).to_numpy()

        x_win, y_win = self._make_windows(x_full, y_full)
        n_healthy: int = int((y_win == 0).sum())
        n_fault: int = int((y_win == 1).sum())
        logger.info(
            "Windows: total=%d  healthy=%d  fault=%d  shape=(T=%d, C=%d)",
            len(x_win),
            n_healthy,
            n_fault,
            self.cfg.window_size,
            len(self._channels),
        )

        # Stratified split preserves the same class ratio in both folds.
        idx = np.arange(len(x_win))
        idx_tr, idx_te = train_test_split(
            idx,
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state,
            stratify=y_win,
        )

        # Scaler fitted *only* on training healthy windows — no leakage of
        # fault-period statistics into the normalisation parameters.
        healthy_train_mask: np.ndarray = y_win[idx_tr] == 0
        healthy_train_windows: np.ndarray = x_win[idx_tr][healthy_train_mask]

        self.scaler = StandardScaler()
        self.scaler.fit(healthy_train_windows.reshape(-1, len(self._channels)))

        logger.info(
            "Scaler fitted on %d healthy training windows (%d raw rows)",
            healthy_train_windows.shape[0],
            healthy_train_windows.shape[0] * self.cfg.window_size,
        )

        x_tr: np.ndarray = self._scale(x_win[idx_tr])
        x_te: np.ndarray = self._scale(x_win[idx_te])
        y_tr: np.ndarray = y_win[idx_tr]
        y_te: np.ndarray = y_win[idx_te]

        # Training dataset: healthy windows ONLY.
        healthy_mask: np.ndarray = y_tr == 0
        x_tr_healthy: np.ndarray = x_tr[healthy_mask]
        y_tr_healthy: np.ndarray = y_tr[healthy_mask]

        logger.info(
            "Train set (healthy-only): %d windows | Val set (all): %d windows",
            len(x_tr_healthy),
            len(x_te),
        )

        self.train_ds = TensorDataset(
            torch.from_numpy(x_tr_healthy),
            torch.from_numpy(y_tr_healthy),
        )
        self.val_ds = TensorDataset(
            torch.from_numpy(x_te),
            torch.from_numpy(y_te),
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_ds is None:
            raise RuntimeError("setup() must be called before train_dataloader()")
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            persistent_workers=self.cfg.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_ds is None:
            raise RuntimeError("setup() must be called before val_dataloader()")
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.batch_size * 2,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            persistent_workers=self.cfg.num_workers > 0,
        )

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def n_channels(self) -> int:
        return len(self._channels)

    @property
    def channel_names(self) -> list[str]:
        return list(self._channels)

    def healthy_train_count(self) -> int:
        if self.train_ds is None:
            raise RuntimeError("setup() must be called first")
        return len(self.train_ds)

    def val_class_balance(self) -> tuple[int, int]:
        if self.val_ds is None:
            raise RuntimeError("setup() must be called first")
        y = self.val_ds.tensors[1].numpy()
        return int((y == 0).sum()), int((y == 1).sum())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _make_windows(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        t: int = self.cfg.window_size
        s: int = self.cfg.stride
        n: int = (len(x) - t) // s + 1
        if n <= 0:
            raise ValueError(
                f"Not enough rows ({len(x)}) for window_size={t}, stride={s}"
            )

        c: int = x.shape[1]
        x_strided: np.ndarray = np.lib.stride_tricks.as_strided(
            x,
            shape=(n, t, c),
            strides=(x.strides[0] * s, x.strides[0], x.strides[1]),
            writeable=False,
        )
        x_win: np.ndarray = np.ascontiguousarray(x_strided, dtype=np.float32)

        y_strided: np.ndarray = np.lib.stride_tricks.as_strided(
            y,
            shape=(n, t),
            strides=(y.strides[0] * s, y.strides[0]),
            writeable=False,
        )
        # Recall-favouring: any fault sample inside the window flags the window.
        y_win: np.ndarray = y_strided.any(axis=1).astype(np.int64)

        return x_win, y_win

    def _scale(self, x_win: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            raise RuntimeError("Scaler not fitted — call setup() first")
        b, t, c = x_win.shape
        flat: np.ndarray = self.scaler.transform(x_win.reshape(-1, c))
        return flat.reshape(b, t, c).astype(np.float32)
