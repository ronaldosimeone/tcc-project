"""
Regenerate every table and CD figure of the temporal evaluation — RNF-78.

Entry point:
    python -m src.render_temporal_report

Reads ``models/eval_temporal.json`` (written by ``src.evaluate_temporal``) and
writes ``docs/temporal_evaluation.md`` plus ``docs/figures-ml/cd_*.png``. No
model is trained here: the report is a pure function of the JSON, so the
thesis tables can be rebuilt byte-for-byte from the committed results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.stat_tests import FriedmanResult, plot_cd_diagram

_ML_ROOT: Path = Path(__file__).resolve().parent.parent
IN_JSON: Path = _ML_ROOT / "models" / "eval_temporal.json"
REPO_ROOT: Path = _ML_ROOT.parent.parent
OUT_MD: Path = REPO_ROOT / "docs" / "temporal_evaluation.md"
FIG_DIR: Path = REPO_ROOT / "docs" / "figures-ml"

_LABELS: dict[str, str] = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "logistic_regression": "Logistic Regression",
}

_METRIC_LABELS: dict[str, str] = {
    "pr_auc": "PR-AUC",
    "roc_auc": "ROC-AUC",
    "event_recall_inner": "Event recall",
    "range_f1_inner": "Range-F1",
    "false_alarms_per_day_inner": "False alarms / day",
    "expected_cost_inner": "Expected cost (RN-02)",
}


def _ci(c: dict[str, Any], digits: int = 3) -> str:
    if c["n"] == 0 or c["mean"] != c["mean"]:
        return "—"
    return f"{c['mean']:.{digits}f} [{c['low']:.{digits}f}, {c['high']:.{digits}f}]"


def _hours(c: dict[str, Any]) -> str:
    if c["n"] == 0 or c["mean"] != c["mean"]:
        return "—"
    return f"{c['mean'] / 3600:+.2f} h [{c['low'] / 3600:+.2f}, {c['high'] / 3600:+.2f}] (n={c['n']})"


def _summary_table(summary: dict[str, Any]) -> list[str]:
    metrics: list[str] = list(_METRIC_LABELS)
    lines: list[str] = [
        "| Model | "
        + " | ".join(_METRIC_LABELS[m] for m in metrics)
        + " | Lead time |",
        "|---|" + "---|" * (len(metrics) + 1),
    ]
    for name, data in summary["per_model"].items():
        cells: list[str] = [
            _ci(data[m], 2 if "cost" in m or "day" in m else 3) for m in metrics
        ]
        lines.append(
            f"| {_LABELS.get(name, name)} | "
            + " | ".join(cells)
            + f" | {_hours(data['lead_time_s_inner'])} |"
        )
    return lines


def _event_table(runs: list[dict[str, Any]]) -> list[str]:
    """Per (fold, model): fraction of seeds that detected the failure, mean
    lead time, mean false alarms/day — under the three threshold policies."""
    folds: list[int] = sorted({r["fold"] for r in runs})
    models: list[str] = list(dict.fromkeys(r["model"] for r in runs))
    lines: list[str] = [
        "| Fold (test failure) | Model | Detected (fixed 0.5) | Detected (inner) | Lead time (inner) | FA/day (inner) | Threshold (inner) | Detected (oracle) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for f in folds:
        for m in models:
            rs: list[dict[str, Any]] = [
                r for r in runs if r["fold"] == f and r["model"] == m
            ]
            if not rs:
                continue
            ev: str = rs[0]["test_event"][0][:16]

            def det(key: str) -> str:
                hits: int = sum(r[key]["n_detected"] for r in rs)
                return f"{hits}/{sum(r[key]['n_events'] for r in rs)}"

            leads: list[float] = [
                o["lead_time_s"]
                for r in rs
                for o in r["event_inner"]["outcomes"]
                if o["lead_time_s"] is not None
            ]
            lead: str = f"{np.mean(leads) / 3600:+.2f} h" if leads else "—"
            fa: float = float(
                np.mean([r["event_inner"]["false_alarms_per_day"] for r in rs])
            )
            thr: str = ", ".join(
                sorted({f"{r['event_inner']['threshold']:.2f}" for r in rs})
            )
            lines.append(
                f"| {f} ({ev}) | {_LABELS.get(m, m)} | {det('event_fixed')} | {det('event_inner')} | "
                f"{lead} | {fa:.2f} | {thr} | {det('event_oracle_upper_bound')} |"
            )
    return lines


def _uncertainty_table(runs: list[dict[str, Any]], alpha: float) -> list[str]:
    """RF-31: calibration and conformal coverage, mean over (fold, seed)."""
    models: list[str] = list(dict.fromkeys(r["model"] for r in runs))
    lines: list[str] = [
        f"Target coverage per class: ≥ {1 - alpha:.2f} (Mondrian conformal, α = {alpha}). "
        "Calibration set = training out-of-fold scores; runs without one are excluded from the conformal columns.",
        "",
        "| Model | ECE (all runs) | ECE after isotonic (runs with calibration set) | Brier | Coverage normal | Coverage failure | Empty sets | Conformal runs |",
        "|---|---|---|---|---|---|---|---|",
    ]

    def m(vals: list[float]) -> str:
        clean = [v for v in vals if v == v]
        return f"{np.mean(clean):.3f}" if clean else "—"

    for name in models:
        rs = [
            r["uncertainty"] for r in runs if r["model"] == name and "uncertainty" in r
        ]
        conf = [
            u["conformal"]
            for u in rs
            if u.get("conformal") and "coverage" in u["conformal"]
        ]
        lines.append(
            f"| {_LABELS.get(name, name)} | {m([u['ece'] for u in rs])} | "
            f"{m([u['ece_isotonic'] for u in rs if 'ece_isotonic' in u])} | {m([u['brier'] for u in rs])} | "
            f"{m([c['coverage_class_0'] for c in conf])} | {m([c['coverage_class_1'] for c in conf])} | "
            f"{m([c['empty_rate'] for c in conf])} | {len(conf)}/{len(rs)} |"
        )
    return lines


def _friedman_lines(protocol: str, friedman: dict[str, Any]) -> list[str]:
    lines: list[str] = [
        "| Metric | Friedman χ² | p | CD | Average ranks |",
        "|---|---|---|---|---|",
    ]
    for metric, res in friedman.items():
        if "error" in res:
            lines.append(
                f"| {_METRIC_LABELS.get(metric, metric)} | — | — | — | {res['error']} |"
            )
            continue
        ranks: str = ", ".join(
            f"{_LABELS.get(k, k)} {v:.2f}"
            for k, v in sorted(res["avg_ranks"].items(), key=lambda kv: kv[1])
        )
        stat: str = (
            "nan" if res["statistic"] != res["statistic"] else f"{res['statistic']:.2f}"
        )
        p: str = "nan" if res["p_value"] != res["p_value"] else f"{res['p_value']:.3f}"
        lines.append(
            f"| {_METRIC_LABELS.get(metric, metric)} | {stat} | {p} | {res['critical_difference']:.2f} | {ranks} |"
        )
        if metric in ("pr_auc", "expected_cost_inner"):
            plot_cd_diagram(
                FriedmanResult(**res),
                FIG_DIR / f"cd_{protocol}_{metric}.png",
                f"{protocol} — {_METRIC_LABELS.get(metric, metric)}",
            )
    return lines


def render(data: dict[str, Any]) -> str:
    cfg: dict[str, Any] = data["config"]
    out: list[str] = [
        "# Temporal evaluation — RF-28 / RF-29 / RNF-79",
        "",
        "> Generated by `python -m src.render_temporal_report` from "
        "`apps/ml/models/eval_temporal.json` — do not edit by hand.",
        "",
        f"Evaluated at `{data['evaluated_at']}` · {cfg['n_features']} features · "
        f"seeds {cfg['seeds']} · negatives subsampled 1/{cfg['neg_stride']} in training only · "
        f"purge {cfg['purge_s']:.0f} s · embargo {cfg['embargo_s']:.0f} s · persistence k={cfg['persistence_k']} "
        f"(RN-03) · early horizon {cfg['early_horizon_s'] / 3600:.0f} h · min lead {cfg['min_lead_s'] / 3600:.0f} h (RN-01) · "
        f"C_FN={cfg['cost_fn']:.0f}, C_FP={cfg['cost_fp']:.0f} (RN-02).",
        "",
        "Values are `mean [95 % bootstrap CI]` over every (fold, seed) run.",
        "",
    ]

    if "legacy_random_split" in data:
        out += [
            "## Legacy protocol (stratified random 80/20 — leaks)",
            "",
            "| Model | ROC-AUC | PR-AUC | F1@0.5 |",
            "|---|---|---|---|",
        ]
        for r in data["legacy_random_split"]:
            out.append(
                f"| {_LABELS.get(r['model'], r['model'])} | {r['roc_auc']:.4f} | {r['pr_auc']:.4f} | {r['f1_at_0.5']:.4f} |"
            )
        out.append("")

    for protocol, pdata in data["protocols"].items():
        title: str = {
            "walk_forward": "Walk-forward with purge (PRIMARY — never trains on the future)",
            "blocked": "Leave-one-event-segment-out with purge + embargo (secondary)",
        }.get(protocol, protocol)
        out += [f"## {title}", "", f"{pdata['n_folds']} folds.", "", "### Summary", ""]
        out += _summary_table(pdata["summary"])
        out += ["", "### Per-failure detection", ""]
        out += _event_table(pdata["runs"])
        if any("uncertainty" in r for r in pdata["runs"]):
            out += ["", "### Calibration and conformal coverage (RF-31)", ""]
            out += _uncertainty_table(pdata["runs"], cfg.get("conformal_alpha", 0.1))
        out += ["", "### Friedman + Nemenyi (blocks = folds, seeds averaged)", ""]
        out += _friedman_lines(protocol, pdata["summary"]["friedman"])
        out += [
            "",
            f"![CD PR-AUC](figures-ml/cd_{protocol}_pr_auc.png)",
            f"![CD expected cost](figures-ml/cd_{protocol}_expected_cost_inner.png)",
            "",
            "Wilcoxon RF vs XGBoost (paired over folds):",
            "",
        ]
        for metric, w in pdata["summary"]["wilcoxon_rf_vs_xgb"].items():
            out.append(
                f"- {_METRIC_LABELS.get(metric, metric)}: p = {w['p_value']:.3f}"
            )
        out.append("")
        folds = sorted({r["fold"] for r in pdata["runs"]})
        out += [f"![fold {f}](figures-ml/scores_{protocol}_fold{f}.png)" for f in folds]
        out.append("")

    return "\n".join(out) + "\n"


def main() -> None:
    data: dict[str, Any] = json.loads(IN_JSON.read_text(encoding="utf-8"))
    OUT_MD.write_text(render(data), encoding="utf-8")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
