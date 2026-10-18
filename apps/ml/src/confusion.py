"""
Confusion matrices and event bookkeeping shared by the evaluations.

Three levels, all on out-of-time test rows:
* ``samples`` — score ≥ threshold vs label, per 10 s row;
* ``alarms``  — after the k = 6 persistence (what the operator sees);
* ``events``  — failures detected / missed and false-alarm episodes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

from src.evaluate_temporal import EVAL_KW, PERSISTENCE_K  # noqa: E402
from src.event_metrics import evaluate_events, persistent_alarms  # noqa: E402

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


def cm(pred: np.ndarray, y: np.ndarray) -> dict[str, int]:
    return {
        "tn": int(((pred == 0) & (y == 0)).sum()),
        "fp": int(((pred == 1) & (y == 0)).sum()),
        "fn": int(((pred == 0) & (y == 1)).sum()),
        "tp": int(((pred == 1) & (y == 1)).sum()),
    }


def add(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
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
        "samples": cm((scores >= thr).astype(int), y),
        "alarms": cm(alarms.astype(int), y),
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


def metrics(cm: dict[str, int]) -> dict[str, float | None]:
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
