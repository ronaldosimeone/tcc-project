# Model Comparison — Random Forest vs XGBoost

**Dataset**: MetroPT-3 Air Compressor (Porto Metro)  
**Task**: Binary fault detection (`anomaly` column, class 1 = fault)  
**Split**: 80 % train / 20 % test — stratified, `random_state=42`  
**Feature set**: 34 engineered features (12 raw sensors + 1 pressure delta + 21 rolling stats)

> **⚠️ Superseded metrics (RF-28).** Every number in the hold-out tables
> below was obtained with a *stratified random* 80/20 split over 10-second
> samples whose features are rolling windows — consecutive, almost identical
> rows land in train and test, which leaks the test set (AUC = 1.0000).
> The leakage-free results (purged walk-forward by failure event, 5 seeds,
> event-level metrics, 95 % CIs) are in
> [`temporal_evaluation.md`](temporal_evaluation.md). Summary:
>
> | Model | PR-AUC legacy split | PR-AUC walk-forward [95 % CI] | Failures detected (walk-forward) | Expected cost RN-02 |
> |---|---|---|---|---|
> | Random Forest | 0.9992 | 0.587 [0.428, 0.739] | 10/15 runs | 50.4 [30.7, 70.3] |
> | XGBoost | 0.9996 | 0.400 [0.283, 0.543] | 10/15 runs | 35.3 [14.9, 61.1] |
> | Logistic Regression | 0.7090 | 0.331 [0.155, 0.547] | 10/15 runs | 38.0 [17.8, 63.2] |
>
> No model meets RN-01 (lead time ≥ 2 h; observed +0.1 h to +0.9 h), and with
> 3 walk-forward folds no comparison can reach significance (RN-05 gate:
> minimum achievable p = 0.125). The original text is kept below for the
> history of the project.

---

## Metrics on Hold-Out Test Set

| Metric                         | Random Forest | XGBoost (100 trials) |
| ------------------------------ | ------------- | -------------------- |
| **F1-Score (class 1 / fault)** | **0.9703**    | 0.9302               |
| **Precision (class 1)**        | 0.9447        | —                    |
| **Recall (class 1)**           | 0.9973        | —                    |
| **AUC-ROC**                    | **1.0000**    | **1.0000**           |
| **Accuracy**                   | 0.9988        | —                    |
| Inference p50 (single row)     | ~2–5 ms       | **1.67 ms**          |
| Inference p95 (single row)     | ~5–15 ms      | **1.98 ms**          |
| Artefact size                  | ~120 MB       | ~35 MB               |

---

## Hyperparameter Optimisation (Optuna)

**Storage**: `apps/ml/data/optuna/xgboost_study.db` (SQLite, RNF-23)  
Interrupted studies resume automatically (`load_if_exists=True`).

### Search Space & Best Parameters

| Hyperparameter     | Range             | Best (100 trials) |
| ------------------ | ----------------- | ----------------- |
| `n_estimators`     | 100 – 600         | 598               |
| `max_depth`        | 3 – 10            | 7                 |
| `learning_rate`    | 0.01 – 0.30 (log) | 0.1877            |
| `subsample`        | 0.60 – 1.00       | 0.9201            |
| `colsample_bytree` | 0.60 – 1.00       | 0.7351            |
| `min_child_weight` | 1 – 10            | 1                 |
| `gamma`            | 0.0 – 1.0         | 0.0023            |

**Best CV F1 (100 trials)**: 0.9840  
**scale_pos_weight**: 49.64 (majority/minority ratio — replaces SMOTE for XGBoost)

---

## Class Imbalance Strategy

| Model         | Technique                  | Minority samples in train |
| ------------- | -------------------------- | ------------------------- |
| Random Forest | SMOTE (`imbalanced-learn`) | ~1.19 M (synthetic)       |
| XGBoost       | `scale_pos_weight = 49.64` | 23 963 (original only)    |

XGBoost handles imbalance natively by upweighting minority-class gradients.  
This eliminates SMOTE's memory overhead (~1 GB extra RAM) and halves training time.

---

## Model Selection (RF-10)

**Selection criterion (revised).** Point-wise F1 is no longer the criterion.
A model is selected by the **expected operational cost** of its alarm policy
(RN-02: missed failure = 100 × false alarm, threshold chosen on training
out-of-fold scores, never on the test fold), with event lead time (RN-01) as
tie-breaker, and it replaces the incumbent only if it passes the statistical
promotion gate (RN-05, `apps/ml/src/promotion_gate.py`: one-sided paired
Wilcoxon over temporal folds, p < 0.05). On MetroPT-3 the gate currently
refuses every swap as underpowered — the honest outcome with 3 folds.

Both models are loaded by the same `ModelService` class — no backend code change required.

```bash
# Default: Random Forest
ACTIVE_MODEL=random_forest

# Switch to XGBoost (no redeploy)
ACTIVE_MODEL=xgboost
```

The registry in `model_service.py` maps names to artefact paths:

```text
random_forest  →  apps/ml/models/random_forest_final.joblib
xgboost        →  apps/ml/models/xgboost_v1.joblib
```

---

## Recommendation

| Scenario                                      | Recommended model                                      |
| --------------------------------------------- | ------------------------------------------------------ |
| Maximum fault recall (safety-critical)        | **Random Forest** (recall 0.9973)                      |
| Ultra-low latency / Resource-constrained edge | **XGBoost** (p95 1.98 ms, 3× smaller memory footprint) |

---

## Running More Trials

If further optimisation is desired, the study can be resumed. Optuna will automatically read the existing 100 trials from SQLite and continue exploring the search space:

```bash
cd apps/ml
.venv/Scripts/python src/train_xgboost.py --n-trials 50
```

---
