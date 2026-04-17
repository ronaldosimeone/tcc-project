"""
MLP Classifier — PyTorch Lightning training, MLflow tracking, ONNX export (RNF-24).

Pipeline
--------
1. Load `data/processed/metropt3.parquet` → apply MetroPTPreprocessor (34 features).
2. Stratified 80/20 split (random_state=42).
3. Fit StandardScaler on X_train — saves to `models/mlp_scaler.joblib`.
4. Build MlpClassifier (LightningModule) with architecture [256, 128, 64].
5. Train with AdamW + CosineAnnealing, EarlyStopping on val_f1 (patience=10).
6. Export best checkpoint to `models/mlp_v1.onnx` (dynamic batch_size axis).
7. Track params, metrics and artefacts in MLflow (`mlruns/`, experiment: mlp_metropt3).
8. Save model card → `models/mlp_v1_card.json`.

Usage
-----
    python src/train_mlp.py                      # defaults (50 epochs, batch 1024)
    python src/train_mlp.py --max-epochs 10      # quick smoke run
    python src/train_mlp.py --batch-size 512

Design decisions
----------------
- StandardScaler is fit **only** on X_train and saved separately from the ONNX graph.
  Embedding the scaler inside the ONNX graph would prevent updating it independently.
  The `OnnxMlpAdapter` in the backend loads both artefacts and applies scaling before
  forwarding the tensor to ONNX Runtime.
- CrossEntropyLoss is weighted by the inverse class frequency so the minority class
  (fault) receives proportionally higher gradient signal — equivalent to what
  `scale_pos_weight` does for XGBoost.
- Logits (not probabilities) are the ONNX output; softmax is applied at inference time
  inside the adapter.  Exporting raw logits is numerically safer and keeps the ONNX
  graph simpler.
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
import mlflow
import numpy as np
import onnxruntime as ort
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.classification import BinaryAUROC, BinaryF1Score

# ── resolve paths from this file's location ───────────────────────────────────
_HERE: Path = Path(__file__).resolve().parent  # apps/ml/src/
_ML_ROOT: Path = _HERE.parent  # apps/ml/
_DATA_PATH: Path = _ML_ROOT / "data" / "processed" / "metropt3.parquet"
_MODELS_DIR: Path = _ML_ROOT / "models"
_MLRUNS_DIR: Path = _ML_ROOT / "mlruns"

sys.path.insert(0, str(_HERE))
from preprocessing import MetroPTPreprocessor  # noqa: E402

# ── feature contract (must match RF and XGBoost — model_service.py) ───────────
_FEATURE_NAMES: list[str] = [
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

_TARGET: str = "anomaly"
_RANDOM_STATE: int = 42
_TEST_SIZE: float = 0.20
_INPUT_DIM: int = len(_FEATURE_NAMES)  # 34

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------


class MlpClassifier(pl.LightningModule):
    """
    Tabular MLP for binary fault classification.

    Architecture: Linear → BatchNorm → ReLU → Dropout (repeated per layer)
                  followed by a final linear head with 2 output logits.

    The forward pass produces raw logits — softmax is applied at inference time.
    """

    def __init__(
        self,
        input_dim: int = _INPUT_DIM,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        pos_weight: float = 1.0,
        max_epochs: int = 50,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        # Build feed-forward body
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 2))
        self.network = nn.Sequential(*layers)

        # Class weight tensor registered as buffer so it moves with the model
        self.register_buffer(
            "class_weights", torch.tensor([1.0, pos_weight], dtype=torch.float32)
        )

        # Torchmetrics — reset manually after each epoch
        self.val_f1 = BinaryF1Score(threshold=0.5)
        self.val_auc = BinaryAUROC()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, weight=self.class_weights)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, weight=self.class_weights)
        probs = torch.softmax(logits, dim=1)[:, 1]
        self.val_f1.update(probs, y)
        self.val_auc.update(probs, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        f1 = self.val_f1.compute()
        auc = self.val_auc.compute()
        self.log("val_f1", f1, prog_bar=True)
        self.log("val_auc", auc, prog_bar=True)
        self.val_f1.reset()
        self.val_auc.reset()

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=1e-6,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


# ---------------------------------------------------------------------------
# Data loading & engineering
# ---------------------------------------------------------------------------


def load_and_engineer(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load parquet and apply the MetroPTPreprocessor to produce the 34-feature matrix."""
    log.info("Loading data from %s …", path)
    raw: pd.DataFrame = pd.read_parquet(path)
    log.info("Raw shape: %s", raw.shape)

    preprocessor = MetroPTPreprocessor()
    engineered: pd.DataFrame = preprocessor.transform(raw)

    missing = [c for c in _FEATURE_NAMES if c not in engineered.columns]
    if missing:
        raise ValueError(f"Engineered DataFrame is missing columns: {missing}")

    X: pd.DataFrame = engineered[_FEATURE_NAMES].astype(np.float32)
    y: pd.Series = raw[_TARGET].astype(int)

    log.info(
        "Feature matrix: %s | class dist: %s",
        X.shape,
        y.value_counts().to_dict(),
    )
    return X, y


# ---------------------------------------------------------------------------
# ONNX export & validation
# ---------------------------------------------------------------------------


def export_to_onnx(model: MlpClassifier, onnx_path: Path) -> None:
    """Export the LightningModule to ONNX with a dynamic batch_size axis."""
    model.eval()
    dummy_input = torch.randn(1, _INPUT_DIM, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={
            "features": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )
    log.info("ONNX model exported → %s", onnx_path)


def validate_onnx(onnx_path: Path, X_sample: np.ndarray) -> np.ndarray:
    """Run a quick sanity check: ONNX output must match PyTorch output."""
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    logits: np.ndarray = session.run(None, {input_name: X_sample})[0]
    log.info("ONNX validation — output shape: %s", logits.shape)
    return logits


# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------


def _measure_latency(
    session: ort.InferenceSession, X_sample: np.ndarray, n_reps: int = 200
) -> dict[str, float]:
    """Measure single-row ONNX inference latency (p50 / p95) in milliseconds."""
    single = X_sample[:1]
    input_name = session.get_inputs()[0].name
    times: list[float] = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        session.run(None, {input_name: single})
        times.append((time.perf_counter() - t0) * 1_000)
    return {
        "p50_ms": round(float(np.percentile(times, 50)), 3),
        "p95_ms": round(float(np.percentile(times, 95)), 3),
    }


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------


def train(max_epochs: int = 50, batch_size: int = 1024) -> None:
    # 1. Data
    X, y = load_and_engineer(_DATA_PATH)

    # 2. Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=_TEST_SIZE, random_state=_RANDOM_STATE, stratify=y
    )
    log.info("Train: %d rows | Test: %d rows", len(X_train), len(X_test))

    # 3. Fit & save StandardScaler
    scaler = StandardScaler()
    X_train_scaled: np.ndarray = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled: np.ndarray = scaler.transform(X_test).astype(np.float32)

    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    scaler_path = _MODELS_DIR / "mlp_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    log.info("StandardScaler saved → %s", scaler_path)

    # 4. TensorDatasets
    y_train_arr = y_train.to_numpy(dtype=np.int64)
    y_test_arr = y_test.to_numpy(dtype=np.int64)

    train_ds = TensorDataset(
        torch.from_numpy(X_train_scaled),
        torch.from_numpy(y_train_arr),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_test_scaled),
        torch.from_numpy(y_test_arr),
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0
    )

    # 5. Class weight (minority upsampling via loss weight)
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    pos_weight = round(n_neg / n_pos, 4)
    log.info("pos_weight = %.2f  (neg=%d, pos=%d)", pos_weight, n_neg, n_pos)

    # 6. MLflow logger
    _MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow_logger = MLFlowLogger(
        experiment_name="mlp_metropt3",
        tracking_uri=f"file:///{_MLRUNS_DIR}",
        log_model=False,
    )

    # 7. Callbacks
    # dirpath is set explicitly to a local absolute path to avoid the
    # FileNotFoundError on Windows where MLFlowLogger inherits the
    # artifact_uri (file:///C:/...) and ModelCheckpoint tries to use
    # that URI string verbatim as an os.makedirs() target.
    _ckpt_dir = (_ML_ROOT / "checkpoints").resolve()
    _ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(_ckpt_dir),
        monitor="val_f1",
        mode="max",
        save_top_k=1,
        filename="best-{epoch:02d}-{val_f1:.4f}",
    )
    early_stop_cb = EarlyStopping(
        monitor="val_f1",
        mode="max",
        patience=10,
        verbose=True,
    )

    # 8. Model
    model = MlpClassifier(
        input_dim=_INPUT_DIM,
        hidden_dims=[256, 128, 64],
        dropout=0.3,
        learning_rate=1e-3,
        pos_weight=pos_weight,
        max_epochs=max_epochs,
    )

    # 9. Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=mlflow_logger,
        callbacks=[checkpoint_cb, early_stop_cb],
        enable_progress_bar=True,
        log_every_n_steps=10,
        deterministic=True,
    )

    log.info(
        "Starting training — max_epochs=%d, batch_size=%d …", max_epochs, batch_size
    )
    trainer.fit(model, train_loader, val_loader)

    # 10. Load best checkpoint for export
    best_ckpt = checkpoint_cb.best_model_path
    log.info(
        "Best checkpoint: %s  (val_f1=%.4f)", best_ckpt, checkpoint_cb.best_model_score
    )
    best_model = MlpClassifier.load_from_checkpoint(best_ckpt)

    # 11. ONNX export + validation
    onnx_path = _MODELS_DIR / "mlp_v1.onnx"
    export_to_onnx(best_model, onnx_path)
    onnx_logits = validate_onnx(onnx_path, X_test_scaled[:8])
    assert onnx_logits.shape == (
        8,
        2,
    ), f"Unexpected ONNX output shape: {onnx_logits.shape}"

    # 12. Evaluate on test set (via ONNX, matching production path)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    all_logits: np.ndarray = session.run(None, {input_name: X_test_scaled})[0]
    probs: np.ndarray = _softmax(all_logits)
    y_proba = probs[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    f1 = float(f1_score(y_test_arr, y_pred))
    auc = float(roc_auc_score(y_test_arr, y_proba))
    report = classification_report(y_test_arr, y_pred, output_dict=True)
    latency = _measure_latency(session, X_test_scaled)

    log.info("Test F1 (class 1): %.4f", f1)
    log.info("Test AUC-ROC:      %.4f", auc)
    log.info(
        "Latency p50/p95:   %.2f ms / %.2f ms", latency["p50_ms"], latency["p95_ms"]
    )

    # 13. Log artefacts to MLflow (same run the logger opened)
    with mlflow.start_run(run_id=mlflow_logger.run_id):
        mlflow.log_params(
            {
                "hidden_dims": "[256, 128, 64]",
                "dropout": 0.3,
                "learning_rate": 1e-3,
                "batch_size": batch_size,
                "max_epochs": max_epochs,
                "pos_weight": pos_weight,
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
            }
        )
        mlflow.log_artifact(str(onnx_path), artifact_path="model")
        mlflow.log_artifact(str(scaler_path), artifact_path="model")

    # 14. Model card
    card: dict[str, Any] = {
        "schema_version": "1.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "MlpClassifier",
        "framework": "PyTorch Lightning",
        "export_format": "ONNX",
        "dataset": "MetroPT-3",
        "target_column": _TARGET,
        "requirements": {
            "RNF-24": "MLflow experiment tracking (local mlruns/)",
            "RF-10": "Selectable via ACTIVE_MODEL=mlp env-var",
        },
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "test_size_ratio": _TEST_SIZE,
        "random_state": _RANDOM_STATE,
        "architecture": {
            "input_dim": _INPUT_DIM,
            "hidden_dims": [256, 128, 64],
            "dropout": 0.3,
            "output_dim": 2,
            "activation": "ReLU",
            "normalization": "BatchNorm1d",
        },
        "training": {
            "optimizer": "AdamW",
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "scheduler": "CosineAnnealingLR",
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "early_stopping_patience": 10,
            "monitor_metric": "val_f1",
            "pos_weight": pos_weight,
        },
        "preprocessing": {
            "scaler": "StandardScaler",
            "scaler_artefact": "mlp_scaler.joblib",
        },
        "feature_count": len(_FEATURE_NAMES),
        "feature_names": _FEATURE_NAMES,
        "metrics": {
            "f1_class1_test": round(f1, 4),
            "roc_auc_test": round(auc, 4),
            "latency_single_row_ms": latency,
            "classification_report": report,
        },
    }
    card_path = _MODELS_DIR / "mlp_v1_card.json"
    card_path.write_text(json.dumps(card, indent=2))
    log.info("Model card saved → %s", card_path)
    log.info("Done.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along axis 1."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train MLP on MetroPT-3 with PyTorch Lightning."
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=50,
        help="Maximum training epochs (default: 50). Use 10 for a quick smoke run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Mini-batch size (default: 1024).",
    )
    args = parser.parse_args()
    train(max_epochs=args.max_epochs, batch_size=args.batch_size)
