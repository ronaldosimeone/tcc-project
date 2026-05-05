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
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split

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

# Memory-aware hyperparameter search:
# 1.2 M rows × 80 features × SMOTE (×2) × 3 folds × N param combos estourou a RAM
# do container. A busca passa a usar uma amostra estratificada do treino — o
# refit final continua usando 100% do dataset.
GRID_SEARCH_SAMPLE_FRAC: float = 0.2
GRID_SEARCH_N_JOBS: int = 2  # Cap workers paralelos do GridSearchCV (memória)

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

    Memory note: the search runs on a stratified ``GRID_SEARCH_SAMPLE_FRAC``
    subsample of the training set. With 1.2 M rows × 80 features, SMOTE +
    parallel folds estouravam a RAM do container. A amostragem preserva a
    proporção de falhas (stratify=y_train) e o refit final em
    ``build_final_model`` usa 100 % do treino.

    Worker caps:
      - GridSearchCV n_jobs = ``GRID_SEARCH_N_JOBS`` (default 2).
      - RF *interna* ao pipeline n_jobs = 1 — evita oversubscription
        (2 workers × N_cores árvores explodiriam a memória de novo).
      - O ``final_rf`` em ``build_final_model`` continua com n_jobs=-1.

    Returns
    -------
    GridSearchCV
        Fitted searcher with best_estimator_ (Pipeline) and best_params_.
    """
    # Stratified subsample for hyperparameter search only.
    X_search, _, y_search, _ = train_test_split(
        X_train,
        y_train,
        train_size=GRID_SEARCH_SAMPLE_FRAC,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )
    logger.info(
        "Amostragem para GridSearch: %d linhas (%.0f%% do treino) | "
        "proporção de falhas: %.4f%% (treino completo: %.4f%%)",
        len(X_search),
        GRID_SEARCH_SAMPLE_FRAC * 100,
        y_search.mean() * 100,
        y_train.mean() * 100,
    )

    pipeline: ImbPipeline = ImbPipeline(
        steps=[
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            (
                "rf",
                # V2 — class_weight="balanced_subsample" re-pondera a perda
                # por bootstrap, complementando o SMOTE para imbalanced classes.
                # n_jobs=1 aqui: o paralelismo vive no GridSearchCV (n_jobs=2);
                # deixar -1 também causaria oversubscription e OOM.
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )

    n_combinations: int = 1
    for values in PARAM_GRID.values():
        n_combinations *= len(values)

    logger.info(
        "GridSearchCV: %d combinações × %d folds = %d fits (n_jobs=%d)",
        n_combinations,
        CV_FOLDS,
        n_combinations * CV_FOLDS,
        GRID_SEARCH_N_JOBS,
    )

    grid_search: GridSearchCV = GridSearchCV(
        estimator=pipeline,
        param_grid=PARAM_GRID,
        scoring="f1",
        cv=CV_FOLDS,
        n_jobs=GRID_SEARCH_N_JOBS,
        verbose=2,
        refit=True,
    )
    grid_search.fit(X_search, y_search)

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
        class_weight="balanced_subsample",  # V2 — paridade com o GridSearch
        **rf_kwargs,
    )
    final_rf.fit(X_bal, y_bal)
    logger.info("Modelo final treinado.")
    return final_rf, len(X_bal)


def find_optimal_threshold(
    y_true: pd.Series,
    y_proba: np.ndarray,
    beta: float = 2.0,
) -> dict[str, float]:
    """
    Pick the probability cut-off that maximises the F-beta score.

    Industrial fault detection: a missed fault (FN) costs orders of magnitude
    more than a false alarm (FP).  ``beta=2`` weights recall 4× as heavily as
    precision in the harmonic mean — appropriate for the MetroPT-3 imbalance
    (~2 % fault prevalence) and the user-facing alert pipeline that operators
    can dismiss at low cost.

    Returns
    -------
    dict
        ``threshold`` — selected cut-off in [0, 1].
        ``precision`` / ``recall`` / ``fbeta`` at that cut-off.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # ``thresholds`` is shorter than precision/recall by 1 — slice to align.
    p, r = precision[:-1], recall[:-1]
    fbeta_num = (1 + beta**2) * p * r
    fbeta_den = (beta**2) * p + r + 1e-9
    fbeta = fbeta_num / fbeta_den

    best_idx = int(np.argmax(fbeta))
    return {
        "threshold": float(thresholds[best_idx]),
        "precision": float(p[best_idx]),
        "recall": float(r[best_idx]),
        "fbeta": float(fbeta[best_idx]),
        "beta": beta,
    }


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
    dict with keys: f1_class1, roc_auc, report_dict, threshold_info.
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

    # V2 — Threshold tuning via F2-score on PR curve
    threshold_info = find_optimal_threshold(y_test, y_proba, beta=2.0)
    y_pred_tuned = (y_proba >= threshold_info["threshold"]).astype(int)
    tuned_report: dict[str, Any] = classification_report(
        y_test, y_pred_tuned, output_dict=True
    )

    separator = "=" * 62
    print(f"\n{separator}")
    print("CLASSIFICATION REPORT (threshold=0.5)")
    print(separator)
    print(report_str)
    print(separator)
    print("CONFUSION MATRIX (threshold=0.5)")
    print(separator)
    print(cm)
    print(f"\nROC-AUC:              {roc_auc:.4f}")
    print(f"F1-Score (class 1):   {f1_class1:.4f}")
    print(separator)
    print(
        f"\n[V2] Optimal threshold (F{threshold_info['beta']:.0f}-score):"
        f" {threshold_info['threshold']:.4f}"
    )
    print(
        f"     precision={threshold_info['precision']:.4f}  "
        f"recall={threshold_info['recall']:.4f}  "
        f"F2={threshold_info['fbeta']:.4f}"
    )
    print(f"     F1 (class 1) tuned: {tuned_report['1']['f1-score']:.4f}")
    print(separator)

    if f1_class1 >= 0.75:
        logger.info("[RF-04] PASSED — F1 class 1 = %.4f >= 0.75", f1_class1)
    else:
        logger.warning("[RF-04] FAILED — F1 class 1 = %.4f < 0.75", f1_class1)

    return {
        "f1_class1": f1_class1,
        "roc_auc": roc_auc,
        "report_dict": report_dict,
        "threshold_info": threshold_info,
        "tuned_report": tuned_report,
    }


def export_to_onnx(
    model: RandomForestClassifier,
    feature_names: list[str],
    onnx_path: Path,
) -> None:
    """
    Convert the fitted RandomForest to ONNX (V2 — paridade com MLP/XGBoost).

    Required deps: ``skl2onnx>=1.17``.  Disables the ZipMap output so that
    ONNX Runtime returns a plain ndarray of probabilities.
    """
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        logger.warning(
            "skl2onnx não instalado — pulando export ONNX. "
            "Instale com: pip install skl2onnx>=1.17"
        )
        return

    initial_type = [("features", FloatTensorType([None, len(feature_names)]))]
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=17,
        options={id(model): {"zipmap": False}},
    )
    onnx_path.write_bytes(onnx_model.SerializeToString())
    logger.info(
        "[V2] ONNX salvo: %s (%.1f MB)",
        onnx_path,
        onnx_path.stat().st_size / 1e6,
    )


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

    threshold_info = metrics.get("threshold_info", {})
    tuned_report = metrics.get("tuned_report", {})

    model_card: dict[str, Any] = {
        "schema_version": "2.0",
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
        "class_balancing": "SMOTE + class_weight='balanced_subsample'",
        # V2 — threshold tuning consumed by ModelService at load time
        "decision_threshold": round(
            threshold_info.get("threshold", 0.5), 4
        ),
        "threshold_strategy": "F2-score (recall-favouring)",
        "threshold_metrics": {
            "precision": round(threshold_info.get("precision", 0.0), 4),
            "recall": round(threshold_info.get("recall", 0.0), 4),
            "fbeta": round(threshold_info.get("fbeta", 0.0), 4),
            "beta": threshold_info.get("beta", 2.0),
        },
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "metrics": {
            "f1_class1_test": round(metrics["f1_class1"], 4),
            "roc_auc_test": round(metrics["roc_auc"], 4),
            "classification_report": metrics["report_dict"],
            "tuned_classification_report": tuned_report,
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

    # V2 — Export ONNX para paridade com MLP/XGBoost
    export_to_onnx(
        final_model,
        feature_names=list(X.columns),
        onnx_path=MODELS_DIR / "random_forest_v2.onnx",
    )


if __name__ == "__main__":
    train()
