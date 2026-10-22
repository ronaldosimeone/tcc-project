"""
Nested, leakage-free protocol shared by evaluation and production training.

For one TRAINING set (an outer walk-forward fold, or all data for production):

1. Inner folds = leave-one-failure-out over the failures inside the training
   period (purge 300 s, embargo 1 h) — never touches the outer test fold.
2. Hyper-parameter search: every candidate in the learner's space is scored
   on inner out-of-fold (OOF) predictions; objective = RN-02 expected cost at
   the cost-optimal threshold (k = 6), tie-break = OOF PR-AUC. This replaces
   the legacy Optuna parameters, which were chosen on a random split.
3. Calibration: Platt scaling fitted on the OOF raw scores of the chosen
   candidate (raw score = logit(p) for trees/LogReg, logit margin for neural
   classifiers, log MSE for the autoencoder). Monotonic → alarms unchanged,
   probabilities meaningful (ECE reported before/after).
4. Threshold: cost-optimal on the CALIBRATED OOF probabilities.
5. Final model: chosen candidate refitted on the whole training set.

With fewer than two failures in training there is nothing to validate on:
the first candidate is used, uncalibrated, with the a-priori threshold
(p = 0.5; autoencoder: 99th percentile of healthy training MSE; CUSUM/EWMA:
99th percentile of the healthy training statistic) — recorded
as ``fallback``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import src.deep_temporal as dt
import src.statistical_baselines as sb
from src.evaluate_temporal import EMBARGO_S, EVAL_KW, PERSISTENCE_K, PURGE_S, _subsample
from src.event_metrics import cost_optimal_threshold
from src.temporal_validation import FailureEvent, PurgedEventBlockKFold

log = logging.getLogger("nested_protocol")

NEG_STRIDE: int = 5
EPS: float = 1e-6
# Calibrated probabilities sit near the failure prevalence (~2 %), so the grid
# has to reach low values, not only 0.05…0.95.
THRESHOLD_GRID: list[float] = sorted(
    {0.001, 0.002, 0.005, *np.round(np.arange(0.01, 1.0, 0.01), 2).tolist()}
)

# Search spaces — small on purpose (compute), fixed BEFORE looking at any
# result. The first candidate is the a-priori default (used in fallback).
SPACES: dict[str, list[dict[str, Any]]] = {
    "random_forest": [
        {"max_depth": 20, "min_samples_leaf": 1, "max_features": "sqrt"},
        {"max_depth": 10, "min_samples_leaf": 1, "max_features": "sqrt"},
        {"max_depth": None, "min_samples_leaf": 20, "max_features": "sqrt"},
        {"max_depth": 20, "min_samples_leaf": 5, "max_features": 0.3},
    ],
    "xgboost": [
        {
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 300,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
        },
        {
            "max_depth": 4,
            "learning_rate": 0.05,
            "n_estimators": 600,
            "subsample": 0.8,
            "colsample_bytree": 0.6,
        },
        {
            "max_depth": 8,
            "learning_rate": 0.1,
            "n_estimators": 300,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
        },
        {
            "max_depth": 6,
            "learning_rate": 0.2,
            "n_estimators": 150,
            "subsample": 0.7,
            "colsample_bytree": 0.5,
        },
    ],
    "logistic_regression": [{"C": 1.0}, {"C": 0.1}, {"C": 0.01}, {"C": 10.0}],
    **{
        arch: [{"epochs": 8, "lr": 1e-3}, {"epochs": 16, "lr": 3e-4}]
        for arch in dt.ARCHS
    },
}

# Imbalance handling is selected by the same inner validation, not fixed:
# two variants of the a-priori default per model — denser sampling of the
# normal class (all normal rows for tabular learners) and a softer class
# weight (0.25 x balanced). The autoencoder sees no failure rows, so only
# its sampling varies.
DENSE_NEG_STRIDE: dict[str, int] = {
    "random_forest": 1,
    "xgboost": 1,
    "logistic_regression": 1,
    "mlp": 1,
    "tcn": 5,
    "bilstm": 5,
    "patchtst": 5,
    "autoencoder": 2,
}
SOFT_POS_WEIGHT: float = 0.25
for _name, _space in SPACES.items():
    _space.append({**_space[0], "neg_stride": DENSE_NEG_STRIDE[_name]})
    if _name != "autoencoder":
        _space.append({**_space[0], "pos_weight_scale": SOFT_POS_WEIGHT})

# Statistical baselines (CUSUM/EWMA): no class balance to tune — the space
# is the detector's own parameters (see ``statistical_baselines``).
SPACES.update(sb.SPACES)

# Keys that configure sampling/weighting, not the learner itself.
_BALANCE_KEYS: frozenset[str] = frozenset({"neg_stride", "pos_weight_scale"})


@dataclass
class Context:
    X: pd.DataFrame
    y: pd.Series
    ts: pd.Series
    events: list[FailureEvent]
    data: dt.DeepData


# ---------------------------------------------------------------------------
# Learners — fit on training rows, return RAW scores (monotone in risk)
# ---------------------------------------------------------------------------


def _xgb_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover
        return "cpu"


def _tree_model(
    name: str, params: dict[str, Any], seed: int, pos_weight: float, scale: float
) -> Any:
    # scale == 1 keeps the balanced weighting; otherwise an explicit weight
    # pos_weight * scale on the failure class (pos_weight is the balanced
    # ratio on the rows actually trained on).
    explicit = {0: 1.0, 1: pos_weight * scale}
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample" if scale == 1.0 else explicit,
            n_jobs=-1,
            random_state=seed,
            **params,
        )
    if name == "xgboost":
        return XGBClassifier(
            **params,
            scale_pos_weight=pos_weight * scale,
            tree_method="hist",
            device=_xgb_device(),
            n_jobs=-1,
            random_state=seed,
            eval_metric="logloss",
        )
    if name == "logistic_regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=params["C"],
                class_weight="balanced" if scale == 1.0 else explicit,
                max_iter=2000,
                random_state=seed,
            ),
        )
    raise ValueError(name)


def fit_model(
    ctx: Context, name: str, train_idx: np.ndarray, seed: int, params: dict[str, Any]
) -> Any:
    if name in sb.BASELINES:
        return sb.fit(name, params, ctx.X, ctx.y, ctx.ts, train_idx)
    if name in dt.ARCHS:
        return dt.fit(ctx.data, name, train_idx, seed, params)
    tr = _subsample(train_idx, ctx.y, int(params.get("neg_stride", NEG_STRIDE)))
    y_tr = ctx.y.to_numpy()[tr]
    pos_weight = float((len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1))
    learner = {k: v for k, v in params.items() if k not in _BALANCE_KEYS}
    scale = float(params.get("pos_weight_scale", 1.0))
    model = _tree_model(name, learner, seed, pos_weight, scale)
    model.fit(ctx.X.iloc[tr].to_numpy(np.float32), y_tr)
    return model


def raw_score(ctx: Context, name: str, model: Any, rows: np.ndarray) -> np.ndarray:
    if name in sb.BASELINES:
        return sb.raw_score(model, ctx.X, ctx.ts, rows)
    if name in dt.ARCHS:
        s = dt.score(ctx.data, model, rows)
        return np.log(np.maximum(s, 1e-12)) if name == "autoencoder" else s
    p = model.predict_proba(ctx.X.iloc[rows].to_numpy(np.float32))[:, 1]
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def native_threshold(name: str, raw_thr: float) -> float:
    """Raw-score threshold → the scale the serving adapter compares against:
    probability for trees/LogReg, MSE for the autoencoder (neural classifiers
    serve calibrated probabilities, so they use the calibrated threshold)."""
    if name == "autoencoder":
        return float(np.exp(raw_thr))
    return float(1.0 / (1.0 + np.exp(-raw_thr)))


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


@dataclass
class Platt:
    a: float = 1.0
    b: float = 0.0

    @classmethod
    def fit(cls, raw: np.ndarray, y: np.ndarray) -> "Platt":
        lr = LogisticRegression(C=1e6, max_iter=1000)
        lr.fit(raw.reshape(-1, 1), y)
        return cls(a=float(lr.coef_[0, 0]), b=float(lr.intercept_[0]))

    def prob(self, raw: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(self.a * raw + self.b)))

    def raw_at(self, p: float) -> float:
        """Raw score whose calibrated probability is ``p``."""
        p = min(max(p, EPS), 1 - EPS)
        return float((np.log(p / (1 - p)) - self.b) / self.a)


# ---------------------------------------------------------------------------
# Inner OOF, selection, threshold
# ---------------------------------------------------------------------------


def inner_splits(
    ctx: Context, train_idx: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    ts_tr = ctx.ts.iloc[train_idx].reset_index(drop=True)
    lo, hi = ts_tr.min(), ts_tr.max()
    inner_events = [e for e in ctx.events if e.start >= lo and e.end <= hi]
    if len(inner_events) < 2:
        return []
    out = []
    y = ctx.y.to_numpy()
    for f in PurgedEventBlockKFold(inner_events, PURGE_S, EMBARGO_S).split(ts_tr):
        tr, te = train_idx[f.train_idx], train_idx[f.test_idx]
        if te.size and y[tr].sum() > 0:
            out.append((tr, te))
    return out


def oof_scores(
    ctx: Context,
    name: str,
    splits: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
    params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenated (rows, raw scores) of the inner held-out failures."""
    rows, scores = [], []
    for tr, te in splits:
        model = fit_model(ctx, name, tr, seed, params)
        rows.append(te)
        scores.append(raw_score(ctx, name, model, te))
    order = np.argsort(np.concatenate(rows))
    return np.concatenate(rows)[order], np.concatenate(scores)[order]


def cost_threshold(
    ctx: Context,
    rows: np.ndarray,
    prob: np.ndarray,
    eval_kw: dict[str, Any] | None = None,
) -> tuple[float, float]:
    ts = ctx.ts.iloc[rows].reset_index(drop=True)
    lo, hi = ts.min(), ts.max()
    events = [e for e in ctx.events if e.end >= lo and e.start <= hi]
    return cost_optimal_threshold(
        ts,
        prob,
        events,
        k=PERSISTENCE_K,
        candidates=THRESHOLD_GRID,
        **(EVAL_KW if eval_kw is None else eval_kw),
    )


@dataclass
class Selection:
    params: dict[str, Any]
    calibrator: Platt | None
    threshold_cal: float  # on calibrated probability (or 0.5 fallback)
    raw_threshold: float  # same decision on the raw score scale
    source: str
    search: list[dict[str, Any]]
    # OOF of the chosen candidate (empty in fallback) — kept for sensitivity
    # analyses that re-select the threshold under other cost/horizon settings.
    oof_rows: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    oof_raw: np.ndarray = field(default_factory=lambda: np.empty(0))


def select(
    ctx: Context,
    name: str,
    train_idx: np.ndarray,
    seed: int,
    tune: bool = True,
    fixed_params: dict[str, Any] | None = None,
    eval_kw: dict[str, Any] | None = None,
) -> Selection:
    """Steps 1–4. ``tune=False`` reuses ``fixed_params`` (extra seeds).
    ``eval_kw`` overrides the event-evaluation rule used by the cost (e.g. no
    early horizon for the anticipation study)."""
    space = SPACES[name] if tune else [fixed_params or SPACES[name][0]]
    splits = inner_splits(ctx, train_idx)
    if not splits:
        return _fallback(ctx, name, train_idx, seed, space[0])

    y = ctx.y.to_numpy()
    best: tuple[float, float, int] | None = None
    cache: dict[int, tuple[np.ndarray, np.ndarray, Platt, float]] = {}
    search: list[dict[str, Any]] = []
    for i, params in enumerate(space):
        rows, raw = oof_scores(ctx, name, splits, seed, params)
        cal = Platt.fit(raw, y[rows])
        thr, cost = cost_threshold(ctx, rows, cal.prob(raw), eval_kw)
        ap = float(average_precision_score(y[rows], raw)) if y[rows].any() else 0.0
        search.append(
            {"params": params, "oof_cost": cost, "oof_pr_auc": ap, "threshold": thr}
        )
        log.info(
            "%s seed=%d %s → oof cost %.1f, PR-AUC %.3f", name, seed, params, cost, ap
        )
        key = (cost, -ap, i)
        if best is None or key < best:
            best = key
        cache[i] = (rows, raw, cal, thr)
    assert best is not None
    i = best[2]
    rows_i, raw_i, cal, thr = cache[i]
    return Selection(
        params=space[i],
        calibrator=cal,
        threshold_cal=thr,
        raw_threshold=cal.raw_at(thr),
        source=f"nested LOFO OOF ({len(splits)} inner folds)",
        search=search,
        oof_rows=rows_i,
        oof_raw=raw_i,
    )


def _fallback(
    ctx: Context, name: str, train_idx: np.ndarray, seed: int, params: dict[str, Any]
) -> Selection:
    if name == "autoencoder" or name in sb.BASELINES:
        model = fit_model(ctx, name, train_idx, seed, params)
        healthy = (
            train_idx[ctx.y.to_numpy()[train_idx] == 0][::50]
            if name in sb.BASELINES
            else train_idx[ctx.data.clean_window[train_idx]][::50]
        )
        raw_thr = float(np.quantile(raw_score(ctx, name, model, healthy), 0.99))
    else:
        raw_thr = 0.0  # p = 0.5 (logit 0)
    return Selection(
        params=params,
        calibrator=None,
        threshold_cal=0.5,
        raw_threshold=raw_thr,
        source="fallback (<2 failures in training: a-priori threshold, uncalibrated)",
        search=[],
    )


def calibrated(sel: Selection, raw: np.ndarray, name: str) -> np.ndarray:
    """Calibrated probability; fallback = sigmoid of the raw score (AE: rank-
    free 0/1 relative to its threshold, only used for ECE bookkeeping)."""
    if sel.calibrator is not None:
        return sel.calibrator.prob(raw)
    if name == "autoencoder" or name in sb.BASELINES:
        return (raw >= sel.raw_threshold).astype(float)
    return 1.0 / (1.0 + np.exp(-raw))


def uncalibrated_prob(name: str, raw: np.ndarray) -> np.ndarray:
    """What the model would report without calibration (for the ECE 'before')."""
    if name == "autoencoder" or name in sb.BASELINES:
        return np.full_like(raw, np.nan)
    return 1.0 / (1.0 + np.exp(-raw))


Scorer = Callable[[np.ndarray], np.ndarray]
