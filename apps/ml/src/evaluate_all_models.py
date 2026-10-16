"""
Every servable model under ONE leakage-free protocol + confusion matrices.

Entry point:
    python -m src.evaluate_all_models [--seed 0] [--models rf xgb ...]

Protocol: pooled MetroPT-3 + MetroPT 2022, harmonised sensors, purged
walk-forward by failure event (7 failures → 6 test folds). Each fold trains
only on the past, picks its threshold only from training data, and is tested
on a failure it never saw. Tree models: ``evaluate_temporal.evaluate_fold``
(OOF threshold). Deep models: ``deep_temporal`` (temporal-validation
threshold). Same test rows, same k = 6 persistence for all.

Confusion matrices — summed over the 6 out-of-time test folds:
* ``samples``  — score ≥ threshold vs label, per 10 s row;
* ``alarms``   — after k = 6 persistence (what the operator sees);
* ``events``   — failures detected / missed (early horizon 6 h, grace 1 h)
                 and false-alarm episodes.

A confusion matrix of the FINAL production model on its own training data
would be leakage, so none is reported.

Outputs: models/eval_all_models.json, docs/figures-ml/confusion_*.png
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

import src.deep_temporal as dt  # noqa: E402
import src.evaluate_cross_dataset as xd  # noqa: E402
from src.evaluate_temporal import (  # noqa: E402
    EMBARGO_S,
    EVAL_KW,
    PERSISTENCE_K,
    PURGE_S,
    evaluate_fold,
)
from src.event_metrics import evaluate_events, persistent_alarms  # noqa: E402
from src.ingest_metropt import FAILURE_INTERVALS  # noqa: E402
from src.ingest_metropt2022 import FAILURE_INTERVALS_2022  # noqa: E402
from src.temporal_validation import (  # noqa: E402
    PurgedWalkForwardSplit,
    events_from_intervals,
)

_ML_ROOT: Path = Path(__file__).resolve().parent.parent
OUT_JSON: Path = _ML_ROOT / "models" / "eval_all_models.json"
FIG_DIR: Path = _ML_ROOT.parent.parent / "docs" / "figures-ml"
DROPPED: list[str] = ["Reservoirs", "Oil_level"]

TREES: dict[str, str] = {
    "random_forest": "random_forest",
    "xgboost": "xgboost",
    "logistic_regression": "logistic_regression",
}
LABEL: dict[str, str] = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "logistic_regression": "LogReg",
    "mlp": "MLP",
    "tcn": "TCN",
    "bilstm": "BiLSTM",
    "patchtst": "PatchTST",
    "autoencoder": "Autoencoder",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("evaluate_all_models")


def _cm(pred: np.ndarray, y: np.ndarray) -> dict[str, int]:
    return {
        "tn": int(((pred == 0) & (y == 0)).sum()),
        "fp": int(((pred == 1) & (y == 0)).sum()),
        "fn": int(((pred == 0) & (y == 1)).sum()),
        "tp": int(((pred == 1) & (y == 1)).sum()),
    }


def _add(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    return {k: a.get(k, 0) + v for k, v in b.items()}


def fold_report(
    scores: np.ndarray,
    y: np.ndarray,
    ts: pd.Series,
    thr: float,
    events: list[Any],
) -> dict[str, Any]:
    alarms = persistent_alarms(scores, thr, PERSISTENCE_K)
    ev = evaluate_events(ts.reset_index(drop=True), alarms, events, **EVAL_KW).to_dict()
    n_ev = len(ev["outcomes"])
    detected = sum(1 for o in ev["outcomes"] if o["detected"])
    return {
        "threshold": float(thr),
        "pr_auc": float(average_precision_score(y, scores)) if y.any() else None,
        "samples": _cm((scores >= thr).astype(int), y),
        "alarms": _cm(alarms.astype(int), y),
        "events": {
            "failures": n_ev,
            "detected": detected,
            "missed": n_ev - detected,
            "false_alarm_episodes": int(ev["n_false_alarm_episodes"]),
            "false_alarms_per_day": ev["false_alarms_per_day"],
            "mean_lead_time_s": ev["mean_lead_time_s"],
            "expected_cost": ev["expected_cost"],
        },
    }


def _metrics(cm: dict[str, int]) -> dict[str, float | None]:
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else 0.0
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "specificity": tn / (tn + fp) if tn + fp else None,
    }


def plot(results: dict[str, Any], level: str, path: Path) -> None:
    names = list(results)
    cols = 4
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.2 * rows))
    for ax, name in zip(np.ravel(axes), names):
        cm = results[name]["total"][level]
        m = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]], dtype=float)
        norm = m / np.maximum(m.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(m):
            ax.text(
                j,
                i,
                f"{int(v):,}\n{norm[i, j]:.1%}".replace(",", "."),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if norm[i, j] > 0.5 else "black",
            )
        ax.set_xticks([0, 1], ["prev. normal", "prev. falha"], fontsize=7)
        ax.set_yticks([0, 1], ["real normal", "real falha"], fontsize=7)
        ev = results[name]["total"]["events"]
        ax.set_title(
            f"{LABEL[name]} — {ev['detected']}/{ev['failures']} falhas", fontsize=9
        )
    for ax in np.ravel(axes)[len(names) :]:
        ax.axis("off")
    title = {"samples": "por amostra (limiar)", "alarms": "por amostra após k = 6"}[
        level
    ]
    fig.suptitle(
        f"Matrizes de confusão fora do tempo — {title} — walk-forward conjunto, 6 folds",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--models", nargs="*", default=[*TREES, *dt.ARCHS], choices=[*TREES, *dt.ARCHS]
    )
    parser.add_argument("--neg-stride", type=int, default=5)
    parser.add_argument(
        "--epochs", type=int, default=dt.EPOCHS, help="deep models (smoke tests only)"
    )
    parser.add_argument("--out", type=Path, default=OUT_JSON)
    args = parser.parse_args()
    dt.EPOCHS = args.epochs
    started = time.perf_counter()

    xd.configure(DROPPED)
    X, y, ts, is22 = xd.load_pooled()
    raw_df = pd.concat(
        [xd._read(xd.P2020, []), xd._read(xd.P2022, [])], ignore_index=True
    )
    raw_df = raw_df[["timestamp", *[c for c in xd.FEATURES], "anomaly"]]
    data = dt.build_data(raw_df, X, y, ts, is22)
    events = events_from_intervals([*FAILURE_INTERVALS, *FAILURE_INTERVALS_2022])
    folds = list(PurgedWalkForwardSplit(events, PURGE_S, EMBARGO_S).split(ts))

    previous: dict[str, Any] = {}
    if args.out.exists():  # keep models not re-run
        previous = json.loads(args.out.read_text(encoding="utf-8")).get("models", {})
    results: dict[str, Any] = dict(previous)

    for name in args.models:
        per_fold: list[dict[str, Any]] = []
        for fold in folds:
            t0 = time.perf_counter()
            y_te = y.to_numpy()[fold.test_idx]
            ts_te = ts.iloc[fold.test_idx]
            if name in TREES:
                r = evaluate_fold(
                    TREES[name], args.seed, fold, X, y, ts, events, args.neg_stride
                )
                scores, thr = r["_scores"], r["event_inner"]["threshold"]
                src = r["event_inner"].get("threshold_source", "inner OOF")
            else:
                thr, src = dt.temporal_threshold(
                    data, name, fold.train_idx, events, args.seed
                )
                fitted = dt.fit(data, name, fold.train_idx, args.seed)
                scores = dt.score(data, fitted, fold.test_idx)
            rep = fold_report(scores, y_te, ts_te, thr, events)
            rep.update(
                {
                    "fold": fold.fold,
                    "test_dataset": "2022" if is22[fold.test_idx].all() else "2020",
                    "threshold_source": src,
                    "seconds": round(time.perf_counter() - t0, 1),
                }
            )
            per_fold.append(rep)
            log.info(
                "%s fold=%d thr=%.4g detected=%d/%d fa/day=%.2f pr_auc=%s (%.0fs)",
                name,
                fold.fold,
                thr,
                rep["events"]["detected"],
                rep["events"]["failures"],
                rep["events"]["false_alarms_per_day"],
                None if rep["pr_auc"] is None else round(rep["pr_auc"], 3),
                rep["seconds"],
            )
        total: dict[str, Any] = {"samples": {}, "alarms": {}, "events": {}}
        for r in per_fold:
            total["samples"] = _add(total["samples"], r["samples"])
            total["alarms"] = _add(total["alarms"], r["alarms"])
            total["events"] = _add(
                total["events"],
                {
                    k: r["events"][k]
                    for k in ("failures", "detected", "missed", "false_alarm_episodes")
                },
            )
        total["metrics_samples"] = _metrics(total["samples"])
        total["metrics_alarms"] = _metrics(total["alarms"])
        total["expected_cost_mean"] = float(
            np.mean([r["events"]["expected_cost"] for r in per_fold])
        )
        results[name] = {"folds": per_fold, "total": total}
        _write(results, args, started)

    ordered = {k: results[k] for k in [*TREES, *dt.ARCHS] if k in results}
    if args.out == OUT_JSON:
        for level in ("samples", "alarms"):
            plot(ordered, level, FIG_DIR / f"confusion_{level}.png")
    _write(ordered, args, started)


def _write(results: dict[str, Any], args: argparse.Namespace, started: float) -> None:
    args.out.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "evaluated_at": datetime.now(timezone.utc).isoformat(),
                "requirements": ["RF-28", "RF-29", "RF-33", "RNF-79"],
                "protocol": "pooled purged walk-forward, 6 folds, harmonised sensors",
                "seed": args.seed,
                "deep_epochs": args.epochs,
                "dropped_sensors": DROPPED,
                "deep_config": dt.describe(),
                "elapsed_s": round(time.perf_counter() - started, 1),
                "models": results,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
