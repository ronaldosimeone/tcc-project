"""
Explain the real failures with TreeSHAP — RF-32 (evidence for RF-35).

Entry point:
    python -m src.explain_events [--max-rows 600]

For every walk-forward fold (same splitter, purge and subsampling as
``evaluate_temporal``), a model trained only on the past explains the samples
INSIDE the test failure. Outputs, per model and fold:

* sensor ranking (mean signed SHAP over failure samples, Shapley additivity
  makes the feature -> sensor sum exact);
* deletion faithfulness: drop in P(failure) when the top-k features are
  replaced by the normal-operation median, vs k random features;
* stability: Kendall τ between the sensor rankings of different folds — an
  explanation that changes completely from one failure to the next is not a
  usable diagnostic signal for the LLM in RF-35.

Outputs: models/explain_events.json, docs/figures-ml/shap_*.png
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

_ML_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ML_ROOT))

from src.evaluate_temporal import (  # noqa: E402
    EMBARGO_S,
    MODELS,
    PURGE_S,
    _subsample,
    load_features,
)
from src.explainability import (  # noqa: E402
    RAW_SENSORS,
    TreeAlertExplainer,
    aggregate_by_sensor,
    deletion_faithfulness,
)
from src.ingest_metropt import FAILURE_INTERVALS  # noqa: E402
from src.temporal_validation import (  # noqa: E402
    PurgedWalkForwardSplit,
    events_from_intervals,
)  # noqa: E402

OUT_JSON: Path = _ML_ROOT / "models" / "explain_events.json"
FIG_DIR: Path = _ML_ROOT.parent.parent / "docs" / "figures-ml"
TREE_MODELS: tuple[str, ...] = ("random_forest", "xgboost")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log: logging.Logger = logging.getLogger("explain_events")


def _plot_sensor_bars(result: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for model, folds in result["models"].items():
        n: int = len(folds)
        fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, fold in zip(axes, folds):
            items = sorted(fold["sensor_mean_shap"].items(), key=lambda kv: kv[1])
            ax.barh(
                [k for k, _ in items],
                [v for _, v in items],
                color=["tab:red" if v > 0 else "tab:blue" for _, v in items],
            )
            ax.axvline(0, color="0.3", lw=0.8)
            ax.set_title(
                f"fold {fold['fold']} — {fold['test_event'][0][:10]}", fontsize=9
            )
            ax.set_xlabel("mean SHAP (→ failure)")
        fig.suptitle(
            f"{model}: per-sensor contribution inside the real failure", fontsize=10
        )
        fig.tight_layout()
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_DIR / f"shap_sensors_{model}.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 3.6))
        for fold in folds:
            f = fold["faithfulness"]
            ax.plot(
                f["k"],
                f["explained_drop"],
                marker="o",
                label=f"fold {fold['fold']} SHAP top-k",
            )
            ax.plot(f["k"], f["random_drop"], ls="--", color="0.6")
        ax.plot([], [], ls="--", color="0.6", label="random k features")
        ax.set_xlabel("k features replaced by normal median")
        ax.set_ylabel("mean drop in P(failure)")
        ax.set_title(f"{model}: deletion faithfulness", fontsize=10)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"shap_faithfulness_{model}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-rows", type=int, default=600)
    parser.add_argument("--neg-stride", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    started: float = time.perf_counter()
    X, y, ts = load_features()
    events = events_from_intervals(FAILURE_INTERVALS)
    rng: np.random.Generator = np.random.default_rng(args.seed)
    normal_median: pd.Series = X[y.to_numpy() == 0].iloc[::50].median()

    result: dict[str, Any] = {
        "schema_version": "1.0",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "requirements": ["RF-32", "RF-35"],
        "config": {
            "max_rows": args.max_rows,
            "neg_stride": args.neg_stride,
            "seed": args.seed,
            "purge_s": PURGE_S,
        },
        "models": {},
        "stability_kendall_tau": {},
    }

    for name in TREE_MODELS:
        folds_out: list[dict[str, Any]] = []
        for fold in PurgedWalkForwardSplit(events, PURGE_S, EMBARGO_S).split(ts):
            tr: np.ndarray = _subsample(fold.train_idx, y, args.neg_stride)
            y_tr: np.ndarray = y.to_numpy()[tr]
            pos_weight: float = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
            model = MODELS[name](args.seed, pos_weight)
            model.fit(X.iloc[tr], y_tr)

            ev = fold.test_event
            in_fail: np.ndarray = fold.test_idx[
                (
                    (ts.iloc[fold.test_idx] >= ev.start)
                    & (ts.iloc[fold.test_idx] <= ev.end)
                ).to_numpy()
            ]
            pick: np.ndarray = np.sort(
                rng.choice(
                    in_fail, size=min(args.max_rows, in_fail.size), replace=False
                )
            )
            Xf: pd.DataFrame = X.iloc[pick]

            explainer = TreeAlertExplainer(model)
            values, base = explainer.shap_values(Xf)
            sensor_mean: dict[str, float] = aggregate_by_sensor(
                values.mean(axis=0), list(X.columns)
            )
            feat_order: np.ndarray = np.argsort(-np.abs(values).mean(axis=0))[:10]
            faith = deletion_faithfulness(
                lambda d, m=model: m.predict_proba(d)[:, 1],
                Xf,
                values,
                normal_median,
                n_random=10,
                seed=args.seed,
            )
            prob: np.ndarray = model.predict_proba(Xf)[:, 1]
            folds_out.append(
                {
                    "fold": fold.fold,
                    "test_event": [str(ev.start), str(ev.end)],
                    "n_explained": int(pick.size),
                    "mean_probability_in_failure": float(prob.mean()),
                    "base_value": base,
                    "sensor_mean_shap": sensor_mean,
                    "sensor_ranking": [
                        s for s, _ in sorted(sensor_mean.items(), key=lambda kv: -kv[1])
                    ],
                    "top_features_mean_abs": {
                        X.columns[j]: float(np.abs(values[:, j]).mean())
                        for j in feat_order
                    },
                    "faithfulness": faith,
                }
            )
            log.info(
                "%s fold=%d p_in_failure=%.3f top sensors=%s",
                name,
                fold.fold,
                prob.mean(),
                folds_out[-1]["sensor_ranking"][:3],
            )
        result["models"][name] = folds_out

        taus: list[dict[str, Any]] = []
        for a, b in combinations(folds_out, 2):
            va = [a["sensor_mean_shap"][s] for s in RAW_SENSORS]
            vb = [b["sensor_mean_shap"][s] for s in RAW_SENSORS]
            tau, p = kendalltau(va, vb)
            taus.append(
                {
                    "folds": [a["fold"], b["fold"]],
                    "tau": float(tau),
                    "p_value": float(p),
                }
            )
        result["stability_kendall_tau"][name] = taus

    _plot_sensor_bars(result)
    result["elapsed_s"] = round(time.perf_counter() - started, 1)
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log.info("Wrote %s (%.0fs)", OUT_JSON, result["elapsed_s"])


if __name__ == "__main__":
    main()
