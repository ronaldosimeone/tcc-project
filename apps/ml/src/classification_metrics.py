"""
Classic classification metrics of every model under the leak-free protocol.

Entry point:
    python -m src.classification_metrics

Source: the nested evaluation (``models/nested/<model>.json`` and the saved
per-fold test scores, 6 out-of-time folds × 3 seeds). Two decision levels:

* **reading** — each 10 s reading classified at the model's calibrated
  decision threshold (``samples`` confusion matrix);
* **alarm** — after the RN-03 persistence rule (k = 6 consecutive readings),
  i.e. what actually raises an alarm (``alarms`` confusion matrix).

Confusion counts are pooled over the 18 runs; F1 / F2 / MCC are also given as
mean ± sd over runs. ROC-AUC and PR-AUC are threshold-free (mean over runs).
Accuracy is reported but is NOT informative here: ~98 % of readings are
normal, so "always normal" scores ~0.98.

Outputs: models/eval_classification_metrics.json and
docs/classification_metrics.md.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import src.evaluate_cross_dataset as xd

_ML_ROOT: Path = Path(__file__).resolve().parent.parent
NESTED: Path = _ML_ROOT / "models" / "nested"
SCORES: Path = NESTED / "scores"
OUT_JSON: Path = _ML_ROOT / "models" / "eval_classification_metrics.json"
OUT_MD: Path = _ML_ROOT.parent.parent / "docs" / "classification_metrics.md"
MODELS: list[str] = [
    "random_forest",
    "xgboost",
    "logistic_regression",
    "mlp",
    "tcn",
    "bilstm",
    "patchtst",
    "autoencoder",
]
LABEL: dict[str, str] = {
    "random_forest": "Random Forest (champion)",
    "xgboost": "XGBoost",
    "logistic_regression": "Regressão logística (baseline)",
    "mlp": "MLP",
    "tcn": "TCN",
    "bilstm": "BiLSTM",
    "patchtst": "PatchTST",
    "autoencoder": "Autoencoder",
}


def metrics(cm: dict[str, int]) -> dict[str, float | None]:
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
    n = tp + fp + fn + tn

    def div(a: float, b: float) -> float | None:
        return a / b if b else None

    prec = div(tp, tp + fp)
    rec = div(tp, tp + fn)
    spec = div(tn, tn + fp)

    def fbeta(beta: float) -> float | None:
        # Undefined only when the fold has no failure rows. A model that
        # raises nothing on a fold WITH failures scores 0 (not "undefined"):
        # dropping those runs would inflate the per-run mean.
        if rec is None:
            return None
        if not prec or not rec:
            return 0.0
        b2 = beta * beta
        return (1 + b2) * prec * rec / (b2 * prec + rec)

    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    # No predicted (or no actual) positives: MCC is conventionally 0.
    mcc = (tp * tn - fp * fn) / denom if denom else (0.0 if tp + fn else None)
    acc = div(tp + tn, n)
    p_exp = div((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp), n * n)
    kappa = (
        (acc - p_exp) / (1 - p_exp)
        if acc is not None and p_exp is not None and p_exp < 1
        else None
    )
    return {
        "accuracy": acc,
        "balanced_accuracy": (
            (rec + spec) / 2 if rec is not None and spec is not None else None
        ),
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "f1": fbeta(1.0),
        "f2": fbeta(2.0),
        "f0_5": fbeta(0.5),
        "mcc": mcc,
        "cohen_kappa": kappa,
        "false_positive_rate": div(fp, fp + tn),
        "false_negative_rate": div(fn, fn + tp),
    }


def _labels() -> np.ndarray:
    xd.configure(["Reservoirs", "Oil_level"])
    d20, d22 = xd._read(xd.P2020, []), xd._read(xd.P2022, [])
    return (
        pd.concat([d20[xd.TARGET_COL], d22[xd.TARGET_COL]], ignore_index=True)
        .astype(int)
        .to_numpy()
    )


def _sum(cms: list[dict[str, int]]) -> dict[str, int]:
    return {k: int(sum(c[k] for c in cms)) for k in ("tp", "fp", "fn", "tn")}


def _mean_sd(vals: Sequence[float | None]) -> dict[str, float | None]:
    v = [x for x in vals if x is not None]
    if not v:
        return {"mean": None, "sd": None}
    return {
        "mean": float(np.mean(v)),
        "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
    }


def analyse() -> dict[str, Any]:
    y = _labels()
    out: dict[str, Any] = {}
    for m in MODELS:
        runs = json.loads((NESTED / f"{m}.json").read_text(encoding="utf-8"))["runs"]
        rocs: list[float] = []
        for r in runs:
            f = SCORES / f"{m}_s{r['seed']}_f{r['fold']}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            yt = y[z["test_rows"]]
            if yt.any() and not yt.all():
                rocs.append(float(roc_auc_score(yt, z["test_raw"])))
        block: dict[str, Any] = {
            "runs": len(runs),
            "roc_auc": _mean_sd(rocs),
            "pr_auc": _mean_sd([r.get("pr_auc") for r in runs]),
        }
        for level, key in (("reading", "samples"), ("alarm", "alarms")):
            pooled = _sum([r[key] for r in runs])
            per_run = [metrics(r[key]) for r in runs]
            block[level] = {
                "confusion": pooled,
                "pooled": metrics(pooled),
                "per_run": {
                    k: _mean_sd([p[k] for p in per_run]) for k in ("f1", "f2", "mcc")
                },
            }
        ev = [r["events"] for r in runs]
        block["events"] = {
            "detected_mean_per_seed": float(
                np.mean(
                    [
                        sum(e["detected"] for e, r in zip(ev, runs) if r["seed"] == s)
                        for s in sorted({r["seed"] for r in runs})
                    ]
                )
            ),
            "false_alarms_per_day_mean": float(
                np.mean([e["false_alarms_per_day"] for e in ev])
            ),
            "expected_cost_mean": float(np.mean([e["expected_cost"] for e in ev])),
        }
        out[m] = block
    return out


def _f(v: float | None, fmt: str = "{:.3f}") -> str:
    return "—" if v is None else fmt.format(v)


def render(res: dict[str, Any]) -> str:
    L = [
        "# Classification metrics — every model, leak-free protocol",
        "",
        "> Generated by `python -m src.classification_metrics` — do not edit by hand.",
        "",
        "Out-of-time evaluation: purged walk-forward (6 folds, each testing a failure unseen by "
        "that training), inner leave-one-failure-out selection, 3 seeds (18 runs per model). "
        "Confusion counts pooled over the runs. **Accuracy is not informative**: about 98 % of "
        'readings are normal, so a model that always says "normal" scores ≈ 0.98 — use '
        "balanced accuracy, F2 (recall-weighted, the business priority), MCC and the event "
        "metrics.",
        "",
    ]
    for level, title, note in (
        (
            "alarm",
            "## Alarm level (after persistence k = 6) — what raises an alarm",
            "",
        ),
        (
            "reading",
            "## Reading level (each 10 s reading at the calibrated threshold)",
            "",
        ),
    ):
        L += [
            title,
            "",
            "| Model | Accuracy | Balanced acc. | Precision | Recall | Specificity | F1 | **F2** | F0.5 | MCC | Kappa |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for m in MODELS:
            p = res[m][level]["pooled"]
            L.append(
                f"| {LABEL[m]} | {_f(p['accuracy'])} | {_f(p['balanced_accuracy'])} | "
                f"{_f(p['precision'])} | {_f(p['recall'])} | {_f(p['specificity'])} | "
                f"{_f(p['f1'])} | **{_f(p['f2'])}** | {_f(p['f0_5'])} | {_f(p['mcc'])} | "
                f"{_f(p['cohen_kappa'])} |"
            )
        L += [
            "",
            "Mean ± sd over the 18 runs (per-run variability):",
            "",
            "| Model | F1 | F2 | MCC |",
            "|---|---|---|---|",
        ]
        for m in MODELS:
            pr = res[m][level]["per_run"]
            L.append(
                f"| {LABEL[m]} | "
                + " | ".join(
                    f"{_f(pr[k]['mean'])} ± {_f(pr[k]['sd'])}"
                    for k in ("f1", "f2", "mcc")
                )
                + " |"
            )
        L.append("")
    L += [
        "## Threshold-free ranking and event level",
        "",
        "| Model | ROC-AUC | PR-AUC | Failures detected (of 6) | False alarms / day | Expected cost (RN-02) |",
        "|---|---|---|---|---|---|",
    ]
    for m in MODELS:
        r = res[m]
        L.append(
            f"| {LABEL[m]} | {_f(r['roc_auc']['mean'])} ± {_f(r['roc_auc']['sd'])} | "
            f"{_f(r['pr_auc']['mean'])} ± {_f(r['pr_auc']['sd'])} | "
            f"{r['events']['detected_mean_per_seed']:.1f} | "
            f"{r['events']['false_alarms_per_day_mean']:.2f} | "
            f"{r['events']['expected_cost_mean']:.1f} |"
        )
    L += [
        "",
        "PR-AUC baseline (random ranking) ≈ failure prevalence of the test folds (~0.02–0.07). "
        "Confusion matrices: [`model_evaluation_leakfree.md`](model_evaluation_leakfree.md).",
    ]
    return "\n".join(L) + "\n"


def main() -> None:
    res = analyse()
    OUT_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")
    OUT_MD.write_text(render(res), encoding="utf-8")
    print(f"Wrote {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()
