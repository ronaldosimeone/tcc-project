"""
Hybrid physics + ML vs ML-only vs physics-only — digital-twin study.

Entry point:
    python -m src.evaluate_twin [--seeds 3]

Same purged walk-forward protocol, event metrics and threshold policy as
``evaluate_temporal`` (RF-28/29, RN-01/02/03). Per fold the grey-box twin
(``digital_twin.PhysicsTwin``) is fitted ONLY on healthy rows of that fold's
training period, then its 4 physical residuals are computed for the whole
series. Variants:

    ml_only        XGBoost on the 80 statistical features
    hybrid         XGBoost on 80 features + 4 physics residuals
    physics_only   logistic regression on the 4 physics residuals
    fusion         late fusion: alarm when ml_only OR physics_only alarms,
                   each with its own threshold chosen on training OOF scores
                   (no threshold sees the test fold)

Also reported per fold: mean residual z-score inside the real failure vs the
healthy training distribution (does the physics actually move when the
compressor fails?).

Output: models/eval_twin.json
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

_ML_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ML_ROOT))

from src.digital_twin import TWIN_FEATURES, PhysicsTwin  # noqa: E402
from src.evaluate_temporal import (  # noqa: E402
    EMBARGO_S,
    EVAL_KW,
    PERSISTENCE_K,
    PARQUET_PATH,
    PURGE_S,
    evaluate_fold,
    load_features,
)
from src.event_metrics import evaluate_events, persistent_alarms  # noqa: E402
from src.ingest_metropt import FAILURE_INTERVALS  # noqa: E402
from src.stat_tests import bootstrap_ci, wilcoxon_paired  # noqa: E402
from src.temporal_validation import (  # noqa: E402
    PurgedWalkForwardSplit,
    events_from_intervals,
)  # noqa: E402

OUT_JSON: Path = _ML_ROOT / "models" / "eval_twin.json"
VARIANTS: dict[str, str] = {
    "ml_only": "xgboost",
    "hybrid": "xgboost",
    "physics_only": "logistic_regression",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--neg-stride", type=int, default=5)
    args = parser.parse_args()

    X, y, ts = load_features()
    raw = pd.read_parquet(PARQUET_PATH).sort_values("timestamp").reset_index(drop=True)
    events = events_from_intervals(FAILURE_INTERVALS)

    runs: list[dict[str, Any]] = []
    physics_signal: list[dict[str, Any]] = []
    for fold in PurgedWalkForwardSplit(events, PURGE_S, EMBARGO_S).split(ts):
        train_normal = raw.iloc[fold.train_idx]
        train_normal = train_normal[train_normal["anomaly"] == 0]
        twin = PhysicsTwin().fit(train_normal)
        resid = twin.transform(raw).astype(np.float32)
        resid.index = X.index

        mu = resid.iloc[fold.train_idx][y.iloc[fold.train_idx].to_numpy() == 0].mean()
        sd = (
            resid.iloc[fold.train_idx][y.iloc[fold.train_idx].to_numpy() == 0].std()
            + 1e-9
        )
        in_fail = fold.test_idx[
            (
                (ts.iloc[fold.test_idx] >= fold.test_event.start)
                & (ts.iloc[fold.test_idx] <= fold.test_event.end)
            ).to_numpy()
        ]
        z = ((resid.iloc[in_fail] - mu) / sd).mean()
        physics_signal.append(
            {
                "fold": fold.fold,
                "mean_z_in_failure": {k: float(v) for k, v in z.items()},
            }
        )
        print(
            f"fold {fold.fold} physics z in failure:",
            {k: round(float(v), 2) for k, v in z.items()},
            flush=True,
        )

        matrices = {
            "ml_only": X,
            "hybrid": pd.concat([X, resid], axis=1),
            "physics_only": resid,
        }
        for seed in range(args.seeds):
            fold_rows: dict[str, dict[str, Any]] = {}
            for variant, model in VARIANTS.items():
                full = evaluate_fold(
                    model, seed, fold, matrices[variant], y, ts, events, args.neg_stride
                )
                fold_rows[variant] = full
                row = {k: v for k, v in full.items() if not k.startswith("_")}
                row["variant"] = variant
                runs.append(row)
            # Late fusion: union of the two alarm streams, thresholds from training only.
            ml, ph = fold_rows["ml_only"], fold_rows["physics_only"]
            alarms = persistent_alarms(
                ml["_scores"], ml["event_inner"]["threshold"], PERSISTENCE_K
            ) | persistent_alarms(
                ph["_scores"], ph["event_inner"]["threshold"], PERSISTENCE_K
            )
            rep_fusion = evaluate_events(ml["_ts"], alarms, events, **EVAL_KW).to_dict()
            runs.append(
                {
                    "model": "ml_only|physics_only",
                    "seed": seed,
                    "fold": fold.fold,
                    "variant": "fusion",
                    "point": {"pr_auc": float("nan")},
                    "event_inner": rep_fusion,
                }
            )
            print(
                f"fold={fold.fold} fusion seed={seed} recall_ev={rep_fusion['event_recall']:.2f} "
                f"lead={rep_fusion['mean_lead_time_s']} fa/day={rep_fusion['false_alarms_per_day']:.2f}",
                flush=True,
            )

    summary: dict[str, Any] = {}
    for variant in [*VARIANTS, "fusion"]:
        rs = [r for r in runs if r["variant"] == variant]
        summary[variant] = {
            "pr_auc": bootstrap_ci(
                np.array([r["point"]["pr_auc"] for r in rs])
            ).to_dict(),
            "event_recall_inner": bootstrap_ci(
                np.array([r["event_inner"]["event_recall"] for r in rs])
            ).to_dict(),
            "false_alarms_per_day_inner": bootstrap_ci(
                np.array([r["event_inner"]["false_alarms_per_day"] for r in rs])
            ).to_dict(),
            "expected_cost_inner": bootstrap_ci(
                np.array([r["event_inner"]["expected_cost"] for r in rs])
            ).to_dict(),
            "lead_time_s_inner": bootstrap_ci(
                np.array(
                    [
                        o["lead_time_s"]
                        for r in rs
                        for o in r["event_inner"]["outcomes"]
                        if o["lead_time_s"] is not None
                    ]
                )
            ).to_dict(),
        }

    def per_fold(variant: str, getter: Any) -> np.ndarray:
        folds = sorted({r["fold"] for r in runs})
        return np.array(
            [
                np.mean(
                    [
                        getter(r)
                        for r in runs
                        if r["variant"] == variant and r["fold"] == f
                    ]
                )
                for f in folds
            ]
        )

    paired = {
        metric: dict(
            zip(
                ("statistic", "p_value"),
                wilcoxon_paired(per_fold("hybrid", g), per_fold("ml_only", g)),
            )
        )
        for metric, g in {
            "pr_auc": lambda r: r["point"]["pr_auc"],
            "expected_cost_inner": lambda r: r["event_inner"]["expected_cost"],
        }.items()
    }

    out = {
        "schema_version": "1.0",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "twin_features": list(TWIN_FEATURES),
        "config": {"seeds": args.seeds, "neg_stride": args.neg_stride},
        "physics_signal": physics_signal,
        "summary": summary,
        "wilcoxon_hybrid_vs_ml_only_over_folds": paired,
        "fusion_vs_ml_only_expected_cost_over_folds": dict(
            zip(
                ("statistic", "p_value"),
                wilcoxon_paired(
                    per_fold("fusion", lambda r: r["event_inner"]["expected_cost"]),
                    per_fold("ml_only", lambda r: r["event_inner"]["expected_cost"]),
                ),
            )
        ),
        "note": "3 walk-forward folds: no paired test can reach p < 0.05 (minimum one-sided p = 0.125).",
        "runs": runs,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
