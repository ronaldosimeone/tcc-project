"""
MetroPTSequenceDataModule — sliding-window DataModule for sequential DL models
(TCN, BiLSTM, PatchTST).

Pipeline contract
-----------------
1. Load `data/processed/metropt3.parquet`.
2. (Optional) MetroPTPreprocessor — only required when ``feature_set != "raw"``.
   In ``"raw"`` mode the 12 base channels are used as-is, which is the academic
   point of the DL experiment: no manual feature engineering.
3. Sliding window of length ``T = window_size`` with configurable stride.
   Window label = ``y.any()`` over the window — recall-favouring for industrial
   fault detection (a single fault sample inside the window flips the label).
4. Stratified split on **window indices** (the test set asserts disjointness).
5. Per-channel ``StandardScaler`` fit on training windows only — saved as a
   joblib artefact alongside the ONNX graph (parity with ``OnnxMlpAdapter``).
6. (Optional) SMOTE in flattened ``[T*C]`` space; defaults off because the
   loss-side ``pos_weight`` already handles imbalance and SMOTE doubles RAM.

Output tensors
--------------
* ``X`` : float32 of shape ``(B, T, C)``
* ``y`` : int64   of shape ``(B,)``

Note on splitting
-----------------
The default ``StratifiedKFold``-style split shares some context between
neighbouring windows (each pair of windows starting ``stride`` apart shares
``T - stride`` rows).  ``evaluate_sequential.py`` provides a complementary
``GroupKFold`` evaluation grouped by operational day — that is the honest
metric for the TCC comparison; this DataModule is the production split for
single-shot training.
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


# 12 raw channels — 7 analogue sensors + 5 binary indicators.  This is the
# "feature_set=raw" contract that DL models consume directly: no feature
# engineering, the network learns its own temporal representation.
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
class SequenceConfig:
    """
    Static configuration for the DataModule.

    The frozen dataclass is fully hashable (all fields are immutable) and can
    be logged to MLflow as-is.  ``channels`` is a ``tuple`` (not ``list``)
    precisely to preserve hashability — a ``list`` field would silently
    disable the auto-generated ``__hash__``.
    """

    window_size: int = 60
    stride: int = 10
    test_size: float = 0.20
    random_state: int = 42
    batch_size: int = 256
    num_workers: int = 0
    apply_smote: bool = False
    feature_set: str = "raw"  # "raw" (12) | "v1" (34) | "v2" (80)
    subsample_rows: int | None = (
        None  # None → full dataset; int → head(n) for fast iteration
    )
    channels: tuple[str, ...] = field(default_factory=lambda: tuple(_RAW_CHANNELS))


class MetroPTSequenceDataModule(pl.LightningDataModule):
    """LightningDataModule that materialises sliding windows from MetroPT-3."""

    def __init__(
        self,
        parquet_path: Path,
        config: SequenceConfig | None = None,
    ) -> None:
        super().__init__()
        self.parquet_path: Path = parquet_path
        self.cfg: SequenceConfig = config if config is not None else SequenceConfig()
        self.scaler: StandardScaler | None = None
        self._channels: list[str] = []
        self.train_ds: TensorDataset | None = None
        self.val_ds: TensorDataset | None = None

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def prepare_data(self) -> None:
        """Validate that the input parquet exists — fails fast in fit()."""
        if not self.parquet_path.exists():
            raise FileNotFoundError(
                f"Parquet not found at {self.parquet_path}. "
                "Run `python src/ingest_metropt.py` first."
            )

    def setup(self, stage: str | None = None) -> None:
        raw: pd.DataFrame = pd.read_parquet(self.parquet_path)
        if self.cfg.subsample_rows is not None:
            raw = raw.head(int(self.cfg.subsample_rows)).reset_index(drop=True)
            logger.info("Subsampled to %d rows (head)", len(raw))

        df: pd.DataFrame = self._materialise_features(raw)

        self._channels = self._select_channels(df)
        if not self._channels:
            raise ValueError(
                f"No channels resolved for feature_set={self.cfg.feature_set!r}. "
                f"Available columns: {list(df.columns)[:20]}…"
            )

        x_full: np.ndarray = df[self._channels].astype(np.float32).to_numpy()
        y_full: np.ndarray = raw["anomaly"].astype(np.int64).to_numpy()

        x_win, y_win = self._make_windows(x_full, y_full)
        logger.info(
            "Built %d windows of shape (T=%d, C=%d) | class dist: %s",
            len(x_win),
            self.cfg.window_size,
            len(self._channels),
            dict(zip(*np.unique(y_win, return_counts=True))),
        )

        idx = np.arange(len(x_win))
        idx_tr, idx_te = train_test_split(
            idx,
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state,
            stratify=y_win,
        )

        # Per-channel scaler fit on *training windows only* (no leakage).
        self.scaler = StandardScaler()
        flat_train: np.ndarray = x_win[idx_tr].reshape(-1, len(self._channels))
        self.scaler.fit(flat_train)

        x_tr: np.ndarray = self._scale(x_win[idx_tr])
        x_te: np.ndarray = self._scale(x_win[idx_te])
        y_tr: np.ndarray = y_win[idx_tr]
        y_te: np.ndarray = y_win[idx_te]

        if self.cfg.apply_smote:
            x_tr, y_tr = self._smote_on_windows(x_tr, y_tr)
            logger.info("After SMOTE: %d train windows", len(x_tr))

        self.train_ds = TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr))
        self.val_ds = TensorDataset(torch.from_numpy(x_te), torch.from_numpy(y_te))

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
    # Public accessors used by training/evaluation scripts
    # ------------------------------------------------------------------

    @property
    def n_channels(self) -> int:
        return len(self._channels)

    @property
    def channel_names(self) -> list[str]:
        return list(self._channels)

    def class_balance(self) -> tuple[int, int]:
        """Return (n_negative, n_positive) on the train fold."""
        if self.train_ds is None:
            raise RuntimeError("setup() must be called before class_balance()")
        y = self.train_ds.tensors[1].numpy()
        return int((y == 0).sum()), int((y == 1).sum())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _materialise_features(self, raw: pd.DataFrame) -> pd.DataFrame:
        """
        Apply MetroPTPreprocessor only when the requested feature_set demands it.

        For ``"raw"`` we deliberately skip preprocessing — the academic claim
        of the DL experiment is that the network learns its own dynamics.
        """
        if self.cfg.feature_set == "raw":
            return raw

        # Lazy import keeps the DataModule importable in environments that
        # don't have the full ML stack (e.g., backend test harness).
        from preprocessing import MetroPTPreprocessor  # noqa: WPS433

        return MetroPTPreprocessor().transform(raw)

    def _select_channels(self, df: pd.DataFrame) -> list[str]:
        if self.cfg.feature_set == "raw":
            return [c for c in self.cfg.channels if c in df.columns]

        if self.cfg.feature_set == "v1":
            # Same 34 features that train_mlp.py consumes — preserved for ablation.
            v1: list[str] = [
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
                "TP2_delta",
                "TP2_std_5",
                "TP3_std_5",
                "H1_std_5",
                "DV_pressure_std_5",
                "Reservoirs_std_5",
                "Oil_temperature_std_5",
                "Motor_current_std_5",
                "TP2_ma_5",
                "TP2_ma_15",
                "TP3_ma_5",
                "TP3_ma_15",
                "H1_ma_5",
                "H1_ma_15",
                "DV_pressure_ma_5",
                "DV_pressure_ma_15",
                "Reservoirs_ma_5",
                "Reservoirs_ma_15",
                "Oil_temperature_ma_5",
                "Oil_temperature_ma_15",
                "Motor_current_ma_5",
                "Motor_current_ma_15",
            ]
            return [c for c in v1 if c in df.columns]

        if self.cfg.feature_set == "v2":
            # All numeric columns minus the target/timestamp.
            blocked: set[str] = {"timestamp", "anomaly"}
            return [
                c
                for c in df.columns
                if c not in blocked and pd.api.types.is_numeric_dtype(df[c])
            ]

        raise ValueError(f"Unknown feature_set: {self.cfg.feature_set!r}")

    def _make_windows(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Vectorised sliding window via stride tricks — avoids the O(n*T) Python
        loop that would dominate runtime on 15M-row datasets.
        """
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
        # Force a contiguous copy — strided views break torch.from_numpy on
        # subsequent operations and would silently leak the original buffer.
        x_win: np.ndarray = np.ascontiguousarray(x_strided, dtype=np.float32)

        y_strided: np.ndarray = np.lib.stride_tricks.as_strided(
            y,
            shape=(n, t),
            strides=(y.strides[0] * s, y.strides[0]),
            writeable=False,
        )
        # Recall-favouring window label: any fault sample inside the window
        # flags the entire window as anomalous.
        y_win: np.ndarray = (y_strided.any(axis=1)).astype(np.int64)

        return x_win, y_win

    def _scale(self, x_win: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            raise RuntimeError("Scaler not fitted — call setup() first")
        b, t, c = x_win.shape
        flat: np.ndarray = self.scaler.transform(x_win.reshape(-1, c))
        return flat.reshape(b, t, c).astype(np.float32)

    def _smote_on_windows(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Oversample minority class by SMOTE in flattened [T*C] space.

        Synthetic samples interpolate linearly between two neighbouring
        windows of the minority class — physically plausible for sensor data
        because pressure/temperature curves are locally linear at 1 Hz.

        Memory: peak at ~2 × |train| × T × C × 4 bytes.  Guard with
        `subsample_rows` for fast iteration.
        """
        from imblearn.over_sampling import SMOTE  # lazy — heavy import

        b, t, c = x.shape
        flat: np.ndarray = x.reshape(b, t * c)
        flat_res, y_res = SMOTE(random_state=self.cfg.random_state).fit_resample(
            flat, y
        )
        return (
            flat_res.reshape(-1, t, c).astype(np.float32),
            y_res.astype(np.int64),
        )
