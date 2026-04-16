"""
XGBoost training script with Optuna hyperparameter optimisation (RF-10, RNF-23).

Pipeline
--------
1. Load `data/processed/metropt3.parquet`.
2. Apply MetroPTPreprocessor → same 34-feature matrix used by the Random Forest.
3. Stratified 80/20 split (random_state=42).
4. Optuna study (SQLite-persistent, RNF-23) tunes:
     n_estimators, max_depth, learning_rate, subsample, colsample_bytree.
   Trials evaluate F1-score (class 1 / fault) via 3-fold CV on a 200 k-row
   stratified sample to keep each trial under ~30 s.
5. Retrain final model on the *full* training set with best params.
6. Evaluate on hold-out test set.
7. Save model → `models/xgboost_v1.joblib`.
8. Save model card → `models/xgboost_v1_card.json`.

Usage
-----
    # Full 100-trial run (recommended, ~40–90 min depending on hardware)
    python src/train_xgboost.py

    # Quick smoke run (5 trials, ~2 min — generates the artefact immediately)
    python src/train_xgboost.py --n-trials 5

    # Resume an interrupted study (Optuna skips completed trials automatically)
    python src/train_xgboost.py --n-trials 100

Design decisions
----------------
- `scale_pos_weight = majority / minority` replaces SMOTE for XGBoost: the
  algorithm handles imbalance natively by upweighting the minority class during
  gradient computation — faster and just as effective on tabular data.
- `tree_method="hist"` uses the approximate histogram algorithm (GBT-style),
  which is 5–10× faster than the exact method on large datasets.
- The Optuna study is saved to `data/optuna/xgboost_study.db` (SQLite) so
  interrupted runs can be resumed without losing progress (RNF-23).
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
import optuna
import pandas as pd
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

# ── resolve paths from this file's location ───────────────────────────────────
_HERE: Path = Path(__file__).resolve().parent  # apps/ml/src/
_ML_ROOT: Path = _HERE.parent  # apps/ml/
_DATA_PATH: Path = _ML_ROOT / "data" / "processed" / "metropt3.parquet"
_MODELS_DIR: Path = _ML_ROOT / "models"
_OPTUNA_DIR: Path = _ML_ROOT / "data" / "optuna"
_STUDY_DB: Path = _OPTUNA_DIR / "xgboost_study.db"

# Add ml/src to sys.path so we can import the preprocessing module
sys.path.insert(0, str(_HERE))
from preprocessing import MetroPTPreprocessor  # noqa: E402

# ── feature contract (must match RF model_card.json and model_service.py) ─────
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
_CV_FOLDS: int = 3
_OPTUNA_SAMPLE_SIZE: int = 200_000  # rows used per Optuna trial (speed trade-off)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading & feature engineering
# ---------------------------------------------------------------------------


def load_and_engineer(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load parquet and apply the same 34-feature pipeline used by the RF."""
    log.info("Loading data from %s …", path)
    raw: pd.DataFrame = pd.read_parquet(path)
    log.info("Raw shape: %s", raw.shape)

    preprocessor = MetroPTPreprocessor()
    engineered: pd.DataFrame = preprocessor.transform(raw)

    # Keep only the 34 contractual features (same columns RF was trained on)
    missing = [c for c in _FEATURE_NAMES if c not in engineered.columns]
    if missing:
        raise ValueError(f"Engineered DataFrame is missing columns: {missing}")

    X: pd.DataFrame = engineered[_FEATURE_NAMES].astype(np.float32)
    y: pd.Series = raw[_TARGET].astype(int)

    log.info("Feature matrix: %s | class dist: %s", X.shape, y.value_counts().to_dict())
    return X, y


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------


def _build_objective(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_pos_weight: float,
) -> object:
    """Return the Optuna objective closure (keeps dataset in local scope)."""

    # Subsample for faster trial evaluation
    if len(X_train) > _OPTUNA_SAMPLE_SIZE:
        idx = np.random.default_rng(_RANDOM_STATE).choice(
            len(X_train), size=_OPTUNA_SAMPLE_SIZE, replace=False
        )
        X_s = X_train.iloc[idx]
        y_s = y_train.iloc[idx]
    else:
        X_s, y_s = X_train, y_train

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            # Fixed infrastructure params
            "scale_pos_weight": scale_pos_weight,
            "tree_method": "hist",
            "eval_metric": "logloss",
            "random_state": _RANDOM_STATE,
            "n_jobs": -1,
            "verbosity": 0,
        }
        model = XGBClassifier(**params)
        cv = StratifiedKFold(
            n_splits=_CV_FOLDS, shuffle=True, random_state=_RANDOM_STATE
        )
        scores = cross_val_score(model, X_s, y_s, cv=cv, scoring="f1", n_jobs=1)
        return float(scores.mean())

    return objective


# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------


def _measure_latency(
    model: XGBClassifier, X_test: pd.DataFrame, n_reps: int = 200
) -> dict:
    """Measure single-row inference latency (p50 / p95) in milliseconds."""
    row = X_test.iloc[:1]
    times: list[float] = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        model.predict_proba(row)
        times.append((time.perf_counter() - t0) * 1_000)
    return {
        "p50_ms": round(float(np.percentile(times, 50)), 3),
        "p95_ms": round(float(np.percentile(times, 95)), 3),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def train(n_trials: int = 100) -> None:
    # 1. Data
    X, y = load_and_engineer(_DATA_PATH)

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=_TEST_SIZE, random_state=_RANDOM_STATE, stratify=y
    )
    log.info("Train: %d rows | Test: %d rows", len(X_train), len(X_test))

    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    log.info("scale_pos_weight = %.2f", scale_pos_weight)

    # 3. Optuna study (SQLite-persistent, RNF-23)
    _OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{_STUDY_DB}"
    study = optuna.create_study(
        study_name="xgboost_metropt3",
        direction="maximize",
        storage=storage,
        load_if_exists=True,  # resume interrupted study automatically
    )
    completed_before = len(study.trials)
    remaining = n_trials - completed_before
    if remaining <= 0:
        log.info(
            "Study already has %d trials — skipping optimisation.", completed_before
        )
    else:
        log.info(
            "Starting %d Optuna trial(s) (%d already completed) …",
            remaining,
            completed_before,
        )
        objective = _build_objective(X_train, y_train, scale_pos_weight)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=remaining, show_progress_bar=True)

    best_params = study.best_params
    log.info("Best params: %s", best_params)
    log.info("Best CV F1:  %.4f", study.best_value)

    # 4. Final model — retrain on FULL training set
    log.info("Retraining final model on full training set …")
    final_params = {
        **best_params,
        "scale_pos_weight": scale_pos_weight,
        "tree_method": "hist",
        "eval_metric": "logloss",
        "random_state": _RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": 0,
    }
    final_model = XGBClassifier(**final_params)
    final_model.fit(X_train, y_train)

    # 5. Evaluate on test set
    y_pred = final_model.predict(X_test)
    y_proba = final_model.predict_proba(X_test)[:, 1]

    f1 = float(f1_score(y_test, y_pred))
    auc = float(roc_auc_score(y_test, y_proba))
    report = classification_report(y_test, y_pred, output_dict=True)
    latency = _measure_latency(final_model, X_test)

    log.info("Test F1 (class 1): %.4f", f1)
    log.info("Test AUC-ROC:      %.4f", auc)
    log.info(
        "Latency p50/p95:   %.2f ms / %.2f ms", latency["p50_ms"], latency["p95_ms"]
    )

    # 6. Save model artefact
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = _MODELS_DIR / "xgboost_v1.joblib"
    joblib.dump(final_model, model_path)
    log.info("Model saved → %s", model_path)

    # 7. Save model card
    card = {
        "schema_version": "1.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "XGBClassifier",
        "dataset": "MetroPT-3",
        "target_column": _TARGET,
        "requirements": {
            "RF-04": "F1-Score (class 1 / fault) >= 0.75",
            "RF-10": "Selectable via ACTIVE_MODEL=xgboost env-var",
            "RNF-10": "Model versioned with structured metadata",
            "RNF-23": "Optuna study persisted in SQLite",
        },
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "test_size_ratio": _TEST_SIZE,
        "random_state": _RANDOM_STATE,
        "optuna_trials": n_trials,
        "best_cv_f1": round(study.best_value, 4),
        "best_hyperparameters": best_params,
        "scale_pos_weight": round(scale_pos_weight, 2),
        "feature_count": len(_FEATURE_NAMES),
        "feature_names": _FEATURE_NAMES,
        "metrics": {
            "f1_class1_test": round(f1, 4),
            "roc_auc_test": round(auc, 4),
            "latency_single_row_ms": latency,
            "classification_report": report,
        },
    }
    card_path = _MODELS_DIR / "xgboost_v1_card.json"
    card_path.write_text(json.dumps(card, indent=2))
    log.info("Model card saved → %s", card_path)
    log.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train XGBoost on MetroPT-3 with Optuna."
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of Optuna trials (default: 100). Use 5 for a quick smoke run.",
    )
    args = parser.parse_args()
    train(n_trials=args.n_trials)
