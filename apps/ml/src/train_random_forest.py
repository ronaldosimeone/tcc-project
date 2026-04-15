"""
Random Forest Training Pipeline — RF-04 / RNF-10.

Entry point:
    python -m src.train_random_forest

Pipeline:
    1. Load processed Parquet (must contain 'anomaly' column from ingest step).
    2. Select the 12 authoritative sensor features.
    3. Apply MetroPTPreprocessor (rolling features on the 7 analogue sensors).
    4. Stratified Train/Test split BEFORE any resampling (anti-leakage).
    5. GridSearchCV over an imblearn Pipeline [SMOTE → RF]:
       - SMOTE is applied INSIDE each CV fold → zero leakage during hyperparameter search.
       - X_train is passed RAW (not pre-balanced) so GridSearchCV controls resampling.
    6. Refit a bare RandomForestClassifier with the best hyperparameters on the
       full SMOTE-balanced training set.  Fitting on a DataFrame preserves
       feature_names_in_ so ModelService can use X[model.feature_names_in_].
    7. Evaluate the final model on the untouched test set.
    8. Persist the model to models/random_forest_final.joblib.
    9. Persist metadata + best_params_ to models/model_card.json.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE  # type: ignore
from imblearn.pipeline import Pipeline as ImbPipeline  # type: ignore
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV

_ML_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ML_ROOT))

from src.balancing import MetroPTBalancer  # noqa: E402
from src.preprocessing import MetroPTPreprocessor  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Binary target created by label_anomalies() during ingestion.
TARGET_COL: str = "anomaly"

# Exactly 12 authoritative sensors — analogue first, then digital.
# LPS, Pressure_switch and Caudal_impulses are intentionally excluded
# (near-zero variance, negative bias on model performance).
FEATURE_COLS: list[str] = [
    # Analogue
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Motor_current",
    "Oil_temperature",
    # Digital
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "Oil_level",
]

PARQUET_PATH: Path = _ML_ROOT / "data" / "processed" / "metropt3.parquet"
MODELS_DIR: Path = _ML_ROOT / "models"
MODEL_PATH: Path = MODELS_DIR / "random_forest_final.joblib"
MODEL_CARD_PATH: Path = MODELS_DIR / "model_card.json"

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
CV_FOLDS: int = 3

# Hyperparameter search space — applied to the 'rf' step inside the pipeline.
PARAM_GRID: dict[str, list[Any]] = {
    "rf__n_estimators": [100, 200],
    "rf__max_depth": [10, 15, 20],
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    stream=sys.stdout,
)
logger: logging.Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def load_and_preprocess(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the processed Parquet, select the 12 authoritative sensors,
    and apply rolling-window feature engineering.

    Returns
    -------
    X : pd.DataFrame
        Engineered feature matrix (MetroPTPreprocessor output).
    y : pd.Series
        Binary anomaly target (0 = normal, 1 = fault).
    """
    logger.info("Carregando Parquet: %s", path)
    df: pd.DataFrame = pd.read_parquet(path)

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Coluna alvo '{TARGET_COL}' não encontrada no Parquet. "
            "Execute o script de ingestão (ingest_metropt.py) primeiro."
        )

    missing_features: list[str] = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_features:
        raise ValueError(f"Features ausentes no Parquet: {missing_features}")

    y: pd.Series = df[TARGET_COL].astype(int).rename("anomaly")
    X_raw: pd.DataFrame = df[FEATURE_COLS].copy()

    logger.info(
        "Aplicando MetroPTPreprocessor (rolling features nos sensores analógicos)..."
    )
    preprocessor = MetroPTPreprocessor()
    X: pd.DataFrame = preprocessor.fit_transform(X_raw)

    logger.info(
        "Dataset pronto — shape: %s | proporção de falhas: %.4f%%",
        X.shape,
        y.mean() * 100,
    )
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified train/test split.

    SMOTE is NOT applied here — it is handled internally by the imblearn
    Pipeline during GridSearchCV so that each fold is resampled independently,
    preventing leakage of synthetic samples into validation folds.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    logger.info("Split estratificado (test_size=%.0f%%)...", TEST_SIZE * 100)
    splits = MetroPTBalancer.train_test_split_safe(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_train: pd.DataFrame = splits["X_train"]  # type: ignore[assignment]
    X_test: pd.DataFrame = splits["X_test"]  # type: ignore[assignment]
    y_train: pd.Series = splits["y_train"]  # type: ignore[assignment]
    y_test: pd.Series = splits["y_test"]  # type: ignore[assignment]

    logger.info(
        "Split concluído — treino: %d linhas | teste: %d linhas",
        len(X_train),
        len(X_test),
    )
    return X_train, X_test, y_train, y_test


def run_grid_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> GridSearchCV:
    """
    GridSearchCV over an imblearn Pipeline [SMOTE → RandomForestClassifier].

    The pipeline applies SMOTE inside each CV fold, so synthetic minority
    samples are never present in validation folds — zero data leakage.

    X_train is passed raw (not pre-balanced); the pipeline controls resampling.

    Returns
    -------
    GridSearchCV
        Fitted searcher with best_estimator_ (Pipeline) and best_params_.
    """
    pipeline: ImbPipeline = ImbPipeline(
        steps=[
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("rf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
        ]
    )

    n_combinations: int = 1
    for values in PARAM_GRID.values():
        n_combinations *= len(values)

    logger.info(
        "GridSearchCV: %d combinações × %d folds = %d fits",
        n_combinations,
        CV_FOLDS,
        n_combinations * CV_FOLDS,
    )

    grid_search: GridSearchCV = GridSearchCV(
        estimator=pipeline,
        param_grid=PARAM_GRID,
        scoring="f1",
        cv=CV_FOLDS,
        n_jobs=-1,
        verbose=2,
        refit=True,
    )
    grid_search.fit(X_train, y_train)

    logger.info("Melhores hiperparâmetros: %s", grid_search.best_params_)
    logger.info("Melhor CV F1 (class 1): %.4f", grid_search.best_score_)
    return grid_search


def build_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: dict[str, Any],
) -> tuple[RandomForestClassifier, int]:
    """
    Refit a bare RandomForestClassifier with the best hyperparameters on the
    full SMOTE-balanced training set.

    Fitting on a DataFrame (not ndarray) ensures sklearn sets
    feature_names_in_ on the estimator — required by ModelService.

    Parameters
    ----------
    X_train : pd.DataFrame
        Raw (unbalanced) training features.
    y_train : pd.Series
        Raw training target.
    best_params : dict
        best_params_ from GridSearchCV (keys prefixed with 'rf__').

    Returns
    -------
    tuple[RandomForestClassifier, int]
        Fitted estimator and number of training rows after SMOTE.
    """
    # Strip pipeline prefix to get RF constructor kwargs
    rf_kwargs: dict[str, Any] = {
        k.replace("rf__", ""): v for k, v in best_params.items()
    }
    logger.info("Refitando RF final com melhores hiperparâmetros: %s", rf_kwargs)

    # Apply SMOTE on the full training set (not per-fold — this is the final model)
    smote: SMOTE = SMOTE(random_state=RANDOM_STATE)
    X_arr, y_arr = smote.fit_resample(X_train.values, y_train.values)

    # Reconstruct DataFrame to preserve column names → feature_names_in_
    X_bal: pd.DataFrame = pd.DataFrame(X_arr, columns=X_train.columns)
    y_bal: pd.Series = pd.Series(y_arr, name=y_train.name)

    logger.info(
        "Conjunto balanceado para treino final: %d linhas (classe 1: %d)",
        len(X_bal),
        int(y_bal.sum()),
    )

    final_rf: RandomForestClassifier = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **rf_kwargs,
    )
    final_rf.fit(X_bal, y_bal)
    logger.info("Modelo final treinado.")
    return final_rf, len(X_bal)


def evaluate(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """
    Compute and print evaluation metrics on the test set.

    Checks [RF-04]: F1-Score on class 1 (fault) must be >= 0.75.

    Returns
    -------
    dict with keys: f1_class1, roc_auc, report_dict.
    """
    y_pred: np.ndarray = model.predict(X_test)
    y_proba: np.ndarray = model.predict_proba(X_test)[:, 1]

    report_str: str = classification_report(y_test, y_pred, digits=4)
    report_dict: dict[str, Any] = classification_report(
        y_test, y_pred, output_dict=True
    )
    cm: np.ndarray = confusion_matrix(y_test, y_pred)
    roc_auc: float = roc_auc_score(y_test, y_proba)
    f1_class1: float = report_dict["1"]["f1-score"]

    separator = "=" * 62
    print(f"\n{separator}")
    print("CLASSIFICATION REPORT")
    print(separator)
    print(report_str)
    print(separator)
    print("CONFUSION MATRIX")
    print(separator)
    print(cm)
    print(f"\nROC-AUC:              {roc_auc:.4f}")
    print(f"F1-Score (class 1):   {f1_class1:.4f}")
    print(separator)

    if f1_class1 >= 0.75:
        logger.info("[RF-04] PASSED — F1 class 1 = %.4f >= 0.75", f1_class1)
    else:
        logger.warning("[RF-04] FAILED — F1 class 1 = %.4f < 0.75", f1_class1)

    return {
        "f1_class1": f1_class1,
        "roc_auc": roc_auc,
        "report_dict": report_dict,
    }


def save_artefacts(
    model: RandomForestClassifier,
    metrics: dict[str, Any],
    best_params: dict[str, Any],
    best_cv_score: float,
    n_train_balanced: int,
    n_test: int,
    feature_names: list[str],
) -> None:
    """
    Persist the fitted model and a structured model card.

    Satisfies [RNF-10]: model versioned with structured metadata.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    logger.info("Modelo salvo: %s", MODEL_PATH)

    model_card: dict[str, Any] = {
        "schema_version": "1.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_type": type(model).__name__,
        "dataset": "MetroPT-3",
        "target_column": TARGET_COL,
        "requirements": {
            "RF-04": "F1-Score (class 1 / fault) >= 0.75",
            "RNF-10": "Model versioned with structured metadata",
        },
        "train_size": n_train_balanced,
        "test_size": n_test,
        "test_size_ratio": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "cv_folds": CV_FOLDS,
        "param_grid_used": PARAM_GRID,
        "best_hyperparameters": best_params,
        "best_cv_f1": round(best_cv_score, 4),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "metrics": {
            "f1_class1_test": round(metrics["f1_class1"], 4),
            "roc_auc_test": round(metrics["roc_auc"], 4),
            "classification_report": metrics["report_dict"],
        },
    }

    with MODEL_CARD_PATH.open("w", encoding="utf-8") as fh:
        json.dump(model_card, fh, indent=2, default=str)

    logger.info("Model card salvo: %s", MODEL_CARD_PATH)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def train() -> None:
    """Execute the full training pipeline end-to-end."""
    X, y = load_and_preprocess(PARQUET_PATH)

    X_train, X_test, y_train, y_test = split_data(X, y)

    # Hyperparameter search: SMOTE inside pipeline per CV fold (anti-leakage)
    grid_search = run_grid_search(X_train, y_train)

    # Final model: bare RF with best params, fitted on full SMOTE-balanced train set
    final_model, n_train_balanced = build_final_model(
        X_train, y_train, grid_search.best_params_
    )

    metrics = evaluate(final_model, X_test, y_test)

    save_artefacts(
        model=final_model,
        metrics=metrics,
        best_params=grid_search.best_params_,
        best_cv_score=grid_search.best_score_,
        n_train_balanced=n_train_balanced,
        n_test=len(X_test),
        feature_names=list(X.columns),
    )


if __name__ == "__main__":
    train()
