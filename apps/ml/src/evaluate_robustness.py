"""
Robustness of the detectors to sensor faults and adversarial bias — RNF-81.

Entry point:
    python -m src.evaluate_robustness [--search 60]

Setting: walk-forward fold 1 (train before ~9 May, test segment containing the
29-30 May failure) — the fold where both tree models detect the failure, so a
degradation is measurable. Models are trained once on clean data (seed 0).

Perturbations are applied to the RAW test-segment sensors and the 80
features are then recomputed, so faults propagate through the rolling
windows exactly as they would in the field:

* noise_α       i.i.d. Gaussian noise, std = α·σ_train, on the 7 analogue sensors
* frozen_<s>    sensor s stuck at its first value of the segment
* missing_<s>   sensor s replaced by its training median (imputation)
* adv_ε         universal calibration bias δ ∈ [−ε, ε]^7 (units of σ_train)
                chosen by random search to MINIMISE P(failure) inside the
                failure window — "how much miscalibration hides a failure?"

Metrics: PR-AUC, event detected (threshold 0.5, persistence k = 6, RF-29),
false alarms / day, mean P(failure) inside the failure.

Output: models/eval_robustness.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

_ML_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ML_ROOT))

from src.evaluate_temporal import (  # noqa: E402
    EMBARGO_S,
    EVAL_KW,
    MODELS,
    PERSISTENCE_K,
    PURGE_S,
    _subsample,
)
from src.event_metrics import evaluate_events, persistent_alarms  # noqa: E402
from src.ingest_metropt import FAILURE_INTERVALS  # noqa: E402
from src.preprocessing import MetroPTPreprocessor  # noqa: E402
from src.temporal_validation import (  # noqa: E402
    PurgedWalkForwardSplit,
    events_from_intervals,
)  # noqa: E402
from src.train_random_forest import FEATURE_COLS, TARGET_COL  # noqa: E402

PARQUET: Path = _ML_ROOT / "data" / "processed" / "metropt3.parquet"
OUT_JSON: Path = _ML_ROOT / "models" / "eval_robustness.json"
ANALOG: tuple[str, ...] = (
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
)
WARMUP: int = 20  # > longest rolling window (15)
NOISE_LEVELS: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5)
ADV_EPS: tuple[float, ...] = (0.25, 0.5, 1.0)


def _features(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        MetroPTPreprocessor().fit_transform(raw[FEATURE_COLS].copy()).astype(np.float32)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--search", type=int, default=60, help="random-search candidates per ε"
    )
    parser.add_argument("--neg-stride", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    df = pd.read_parquet(PARQUET).sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    y = df[TARGET_COL].astype(int)
    ts = df["timestamp"]
    X = _features(df)
    events = events_from_intervals(FAILURE_INTERVALS)
    fold = next(
        f
        for f in PurgedWalkForwardSplit(events, PURGE_S, EMBARGO_S).split(ts)
        if f.fold == 1
    )

    tr = _subsample(fold.train_idx, y, args.neg_stride)
    sigma = df.iloc[fold.train_idx][list(ANALOG)].std().to_numpy()
    median = df.iloc[fold.train_idx][FEATURE_COLS].median()

    lo = max(int(fold.test_idx[0]) - WARMUP, 0)
    seg_raw = df.iloc[lo : int(fold.test_idx[-1]) + 1].reset_index(drop=True)
    keep = slice(int(fold.test_idx[0]) - lo, None)
    y_te = y.iloc[fold.test_idx].to_numpy()
    ts_te = ts.iloc[fold.test_idx].reset_index(drop=True)
    ev = fold.test_event
    in_fail = ((ts_te >= ev.start) & (ts_te <= ev.end)).to_numpy()
    # Search objective restricted to the failure ±1 day (faster than the whole segment).
    near = (
        (ts_te >= ev.start - pd.Timedelta(days=1))
        & (ts_te <= ev.end + pd.Timedelta(days=1))
    ).to_numpy()

    result: dict[str, Any] = {
        "schema_version": "1.0",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "requirements": ["RNF-81"],
        "fold": 1,
        "test_event": [str(ev.start), str(ev.end)],
        "sigma_train": dict(zip(ANALOG, map(float, sigma))),
        "models": {},
    }

    for name in ("random_forest", "xgboost"):
        y_tr = y.to_numpy()[tr]
        model = MODELS[name](args.seed, (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1))
        model.fit(X.iloc[tr], y_tr)

        def evaluate(raw: pd.DataFrame, m: Any = model) -> dict[str, float]:
            p = m.predict_proba(_features(raw).iloc[keep])[:, 1]
            rep = evaluate_events(
                ts_te, persistent_alarms(p, 0.5, PERSISTENCE_K), events, **EVAL_KW
            )
            return {
                "pr_auc": float(average_precision_score(y_te, p)),
                "event_detected": float(rep.event_recall),
                "false_alarms_per_day": float(rep.false_alarms_per_day),
                "mean_p_in_failure": float(p[in_fail].mean()),
            }

        rows: dict[str, Any] = {"clean": evaluate(seg_raw)}
        for a in NOISE_LEVELS:
            noisy = seg_raw.copy()
            noisy[list(ANALOG)] = (
                noisy[list(ANALOG)].to_numpy()
                + rng.normal(0, 1, (len(noisy), len(ANALOG))) * a * sigma
            )
            rows[f"noise_{a}"] = evaluate(noisy)
        for s in ANALOG:
            frozen = seg_raw.copy()
            frozen[s] = frozen[s].iloc[keep.start]
            rows[f"frozen_{s}"] = evaluate(frozen)
            missing = seg_raw.copy()
            missing[s] = median[s]
            rows[f"missing_{s}"] = evaluate(missing)

        # Universal adversarial bias by random search on the failure window.
        near_raw_idx = np.flatnonzero(near) + keep.start
        win = seg_raw.iloc[
            max(near_raw_idx[0] - WARMUP, 0) : near_raw_idx[-1] + 1
        ].reset_index(drop=True)
        win_keep = slice(near_raw_idx[0] - max(near_raw_idx[0] - WARMUP, 0), None)
        win_fail = in_fail[near]

        def p_fail(delta: np.ndarray, m: Any = model) -> float:
            w = win.copy()
            w[list(ANALOG)] = w[list(ANALOG)].to_numpy() + delta * sigma
            return float(
                m.predict_proba(_features(w).iloc[win_keep])[:, 1][win_fail].mean()
            )

        for eps in ADV_EPS:
            best_delta = np.zeros(len(ANALOG))
            best = p_fail(best_delta)
            for _ in range(args.search):
                cand = rng.uniform(-eps, eps, len(ANALOG))
                val = p_fail(cand)
                if val < best:
                    best, best_delta = val, cand
            attacked = seg_raw.copy()
            attacked[list(ANALOG)] = (
                attacked[list(ANALOG)].to_numpy() + best_delta * sigma
            )
            rows[f"adv_{eps}"] = {
                **evaluate(attacked),
                "delta_sigma": dict(zip(ANALOG, map(float, best_delta))),
            }

        result["models"][name] = rows
        print(
            name,
            json.dumps(
                {
                    k: {m: round(v, 3) for m, v in r.items() if isinstance(v, float)}
                    for k, r in rows.items()
                }
            ),
            flush=True,
        )

    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
