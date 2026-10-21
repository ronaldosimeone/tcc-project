"""
Anticipation study — can the platform warn BEFORE a failure starts?

Entry point:
    python -m src.evaluate_anticipation [--horizons 30 60 120] [--models xgboost ...]

The main detector is trained on the reported failure INTERVAL, so it mostly
learns a failure in progress. Here the target is anticipation: for a horizon
H, a row is positive when a failure STARTS within the next H. Everything else
follows the main nested protocol, so the two are comparable:

* rows inside a failure and in the hour after it (repair / restart) are
  AMBIGUOUS for this task and excluded from training, selection and scoring;
* outer purged walk-forward over the pooled 2020 + 2022 timeline, inner
  leave-one-failure-out for hyper-parameters (same search spaces), Platt
  calibration on out-of-fold scores, cost-optimal threshold (RN-02) — now
  with NO early horizon: only alarms inside [start − H, start) count;
* alarms are the persistent ones (k = 6 consecutive readings).

Per horizon and model: failures anticipated (a persistent alarm before the
start), lead before the start, false-alarm episodes per day, cost — and the
same numbers for the MAIN detector, read from its saved test scores, under
the same "before the start" rule.

Scope: XGBoost (best model of the main protocol) and logistic regression
(baseline), seed 0 — the deep models stay out to bound compute.

Outputs: models/eval_anticipation.json and docs/anticipation_study.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import src.nested_protocol as npc
from src.evaluate_nested import SCORES_DIR, build_context
from src.evaluate_temporal import EMBARGO_S, EVAL_KW, PERSISTENCE_K, PURGE_S
from src.event_metrics import evaluate_events, persistent_alarms
from src.fold_chronology import _label
from src.temporal_validation import FailureEvent, PurgedWalkForwardSplit

_ML_ROOT: Path = Path(__file__).resolve().parent.parent
OUT_JSON: Path = _ML_ROOT / "models" / "eval_anticipation.json"
OUT_MD: Path = _ML_ROOT.parent.parent / "docs" / "anticipation_study.md"
# Per-fold test alarms (gitignored, regenerable) for re-analysis.
ALARMS_DIR: Path = _ML_ROOT / "models" / "anticipation_alarms"
# Reference windows for the chance control stay this far from any failure.
CHANCE_GUARD_BEFORE: pd.Timedelta = pd.Timedelta(hours=6)
HORIZONS_MIN: list[int] = [30, 60, 120]
MODELS: list[str] = ["xgboost", "logistic_regression"]
POST_FAILURE_EXCLUDED: pd.Timedelta = pd.Timedelta(hours=1)
# Only alarms inside [start − H, start) are anticipation; nothing after.
ANTICIPATION_KW: dict[str, float] = {
    **EVAL_KW,
    "early_horizon_s": 0.0,
    "grace_s": 0.0,
    "min_lead_s": 0.0,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("evaluate_anticipation")


def anticipation_task(
    ctx: npc.Context, horizon: pd.Timedelta
) -> tuple[npc.Context, np.ndarray, list[FailureEvent]]:
    """(context restricted to unambiguous rows with the anticipation label,
    positions of those rows in the original context, anticipation windows)."""
    ts = ctx.ts.reset_index(drop=True)
    y = np.zeros(len(ts), int)
    keep = np.ones(len(ts), bool)
    windows: list[FailureEvent] = []
    for ev in ctx.events:
        pre = (ts >= ev.start - horizon) & (ts < ev.start)
        y[pre.to_numpy()] = 1
        ambiguous = (ts >= ev.start) & (ts <= ev.end + POST_FAILURE_EXCLUDED)
        keep[ambiguous.to_numpy()] = False
        windows.append(
            FailureEvent(
                start=ev.start - horizon, end=ev.start - pd.Timedelta(seconds=10)
            )
        )
    pos = np.flatnonzero(keep)
    sub = replace(
        ctx,
        X=ctx.X.iloc[pos].reset_index(drop=True),
        y=pd.Series(y[pos]),
        ts=ts.iloc[pos].reset_index(drop=True),
        events=windows,
    )
    return sub, pos, windows


def outcomes(
    ts: pd.Series, alarms: np.ndarray, windows: list[FailureEvent]
) -> dict[str, Any]:
    """Anticipated failures, lead BEFORE the failure start, false alarms."""
    ts = ts.reset_index(drop=True)
    rep = evaluate_events(ts, alarms, windows, **ANTICIPATION_KW).to_dict()
    per: list[dict[str, Any]] = []
    for w in windows:
        inside = ((ts >= w.start) & (ts <= w.end)).to_numpy()
        a = alarms.astype(bool)
        m = inside & a
        first = ts[m].min() if m.any() else None
        failure_start = w.end + pd.Timedelta(seconds=10)
        # An alarm that was already on before the window opened does not show
        # the model reacting to the coming failure: require an OFF->ON onset
        # inside the window for the stricter count.
        onset = inside & a & ~np.r_[False, a[:-1]]
        per.append(
            {
                "failure_start": str(failure_start),
                "anticipated": first is not None,
                "onset_in_window": bool(onset.any()),
                "lead_before_start_min": (
                    None
                    if first is None
                    else round((failure_start - first).total_seconds() / 60, 1)
                ),
            }
        )
    return {
        "anticipated": sum(p["anticipated"] for p in per),
        "anticipated_onset_in_window": sum(p["onset_in_window"] for p in per),
        "failures": len(per),
        "per_failure": per,
        "false_alarm_episodes": rep["n_false_alarm_episodes"],
        "false_alarms_per_day": rep["false_alarms_per_day"],
        "expected_cost": rep["expected_cost"],
        **chance_control(ts, alarms, windows),
    }


def chance_control(
    ts: pd.Series, alarms: np.ndarray, windows: list[FailureEvent]
) -> dict[str, Any]:
    """How often would the SAME alarms "anticipate" a random window?

    The test timeline is tiled into consecutive windows of length H; windows
    within 6 h before / 1 h after any failure are discarded (they are not
    normal time), as are windows with less than half their readings. Hit =
    any alarm inside. Also: share of normal time spent in alarm."""
    if not windows:
        return {"reference_windows": 0, "reference_hits": 0, "alarm_time_share": None}
    ts = ts.reset_index(drop=True)
    horizon = windows[0].end + pd.Timedelta(seconds=10) - windows[0].start
    excl = np.zeros(len(ts), bool)
    for w in windows:
        start = w.end + pd.Timedelta(seconds=10)
        excl |= (
            (ts >= start - horizon - CHANCE_GUARD_BEFORE)
            & (ts <= start + POST_FAILURE_EXCLUDED)
        ).to_numpy()
    edges = pd.date_range(ts.min(), ts.max() + horizon, freq=horizon)
    bins = np.searchsorted(edges.to_numpy(), ts.to_numpy(), side="right") - 1
    df = pd.DataFrame({"bin": bins, "alarm": alarms.astype(bool), "excl": excl})
    g = df.groupby("bin").agg(
        alarm=("alarm", "any"), excl=("excl", "any"), n=("alarm", "size")
    )
    full = horizon.total_seconds() / 10.0
    ok = g[(~g["excl"]) & (g["n"] >= 0.5 * full)]
    normal = ~excl
    return {
        "reference_windows": int(len(ok)),
        "reference_hits": int(ok["alarm"].sum()),
        "alarm_time_share": (
            float(alarms.astype(bool)[normal].mean()) if normal.any() else None
        ),
    }


def run_trained(
    ctx: npc.Context, name: str, horizon_min: int, seed: int = 0
) -> dict[str, Any]:
    sub, _, windows = anticipation_task(ctx, pd.Timedelta(minutes=horizon_min))
    folds = list(PurgedWalkForwardSplit(windows, PURGE_S, EMBARGO_S).split(sub.ts))
    per_fold: list[dict[str, Any]] = []
    for fold in folds:
        t0 = time.perf_counter()
        sel = npc.select(sub, name, fold.train_idx, seed, eval_kw=ANTICIPATION_KW)
        model = npc.fit_model(sub, name, fold.train_idx, seed, sel.params)
        raw = npc.raw_score(sub, name, model, fold.test_idx)
        alarms = persistent_alarms(
            (raw >= sel.raw_threshold).astype(float), 0.5, PERSISTENCE_K
        )
        ts_te = sub.ts.iloc[fold.test_idx]
        test_windows = [fold.test_event]
        out = outcomes(ts_te, alarms, test_windows)
        ALARMS_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            ALARMS_DIR / f"{name}_H{horizon_min}_f{fold.fold}.npz",
            ts=ts_te.to_numpy().astype("datetime64[s]").astype(np.int64),
            alarms=alarms.astype(np.int8),
        )
        fid, _ = _label(
            FailureEvent(
                start=fold.test_event.end + pd.Timedelta(seconds=10),
                end=fold.test_event.end,
            )
        )
        per_fold.append(
            {
                "fold": fold.fold,
                "test_failure": fid,
                "params": sel.params,
                "threshold_source": sel.source,
                **out,
                "seconds": round(time.perf_counter() - t0, 1),
            }
        )
        log.info(
            "H=%dmin %s fold %d (%s): anticipated=%d lead=%s fa/day=%.2f (%.0fs)",
            horizon_min,
            name,
            fold.fold,
            fid,
            out["anticipated"],
            out["per_failure"][0]["lead_before_start_min"],
            out["false_alarms_per_day"],
            per_fold[-1]["seconds"],
        )
    return _summary(per_fold)


def run_detector(
    ctx: npc.Context, name: str, horizon_min: int
) -> dict[str, Any] | None:
    """The MAIN detector (trained on the failure interval), seed 0, scored by
    the anticipation rule on the same unambiguous rows."""
    files = sorted(SCORES_DIR.glob(f"{name}_s0_f*.npz"))
    if not files:
        return None
    _, keep_pos, windows = anticipation_task(ctx, pd.Timedelta(minutes=horizon_min))
    keep = np.zeros(len(ctx.ts), bool)
    keep[keep_pos] = True
    per_fold = []
    for f in files:
        z = np.load(f)
        rows, raw = z["test_rows"], z["test_raw"]
        m = keep[rows]
        rows, raw = rows[m], raw[m]
        alarms = persistent_alarms(
            (raw >= float(z["raw_threshold"])).astype(float), 0.5, PERSISTENCE_K
        )
        ts_te = ctx.ts.iloc[rows].reset_index(drop=True)
        lo, hi = ts_te.min(), ts_te.max()
        test_windows = [w for w in windows if w.end >= lo and w.start <= hi]
        out = outcomes(ts_te, alarms, test_windows)
        fold = int(str(f.stem).split("_f")[-1])
        per_fold.append({"fold": fold, **out})
    return _summary(sorted(per_fold, key=lambda r: r["fold"]))


def _summary(per_fold: list[dict[str, Any]]) -> dict[str, Any]:
    leads = [
        p["lead_before_start_min"]
        for r in per_fold
        for p in r["per_failure"]
        if p["lead_before_start_min"] is not None
    ]
    ref = sum(r.get("reference_windows", 0) for r in per_fold)
    hits = sum(r.get("reference_hits", 0) for r in per_fold)
    shares = [
        r["alarm_time_share"] for r in per_fold if r.get("alarm_time_share") is not None
    ]
    return {
        "anticipated": sum(r["anticipated"] for r in per_fold),
        "anticipated_onset_in_window": sum(
            r.get("anticipated_onset_in_window", 0) for r in per_fold
        ),
        "chance_hit_rate": hits / ref if ref else None,
        "reference_windows": ref,
        "alarm_time_share_mean": float(np.mean(shares)) if shares else None,
        "failures": sum(r["failures"] for r in per_fold),
        "false_alarm_episodes": sum(r["false_alarm_episodes"] for r in per_fold),
        "false_alarms_per_day_mean": float(
            np.mean([r["false_alarms_per_day"] for r in per_fold])
        ),
        "median_lead_before_start_min": float(np.median(leads)) if leads else None,
        "per_fold": per_fold,
    }


def render(result: dict[str, Any]) -> str:
    lines = [
        "# Anticipation study — warning BEFORE the failure starts",
        "",
        "> Generated by `python -m src.evaluate_anticipation` — do not edit by hand.",
        "",
        "Target: a failure starts within the next *H*. Rows inside a failure and in the "
        "hour after it are excluded (ambiguous for this task). Same nested temporal "
        "protocol as the main study (outer purged walk-forward, inner leave-one-failure-"
        "out, Platt calibration, RN-02 threshold); an alarm counts only if it is "
        "persistent (k = 6) and inside [start − H, start). The *detector* rows score the "
        "main models (trained on the failure interval) under the same rule, from their "
        "saved test scores. Seed 0. Seven failures, six tested: counts, not rates.",
        "",
        "| Horizon | Model | Trained for | Anticipated | Onset inside window | Chance (random H-windows hit) | Time in alarm | False alarms / day |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for h, block in result["horizons"].items():
        for name, res in block.items():
            for kind in ("anticipation", "detector"):
                r = res.get(kind)
                if r is None:
                    continue
                ch = r.get("chance_hit_rate")
                share = r.get("alarm_time_share_mean")
                lines.append(
                    f"| {h} min | {name} | {'anticipation (H)' if kind == 'anticipation' else 'failure interval (main)'} | "
                    f"{r['anticipated']}/{r['failures']} | "
                    f"{r.get('anticipated_onset_in_window', 0)}/{r['failures']} | "
                    f"{'—' if ch is None else f'{ch:.0%}'} ({r.get('reference_windows', 0)}) | "
                    f"{'—' if share is None else f'{share:.1%}'} | "
                    f"{r['false_alarms_per_day_mean']:.2f} |"
                )
    lines += [
        "",
        "*Anticipated* = a persistent alarm inside [start − H, start). *Onset inside window* "
        "is stricter: the alarm must switch ON inside the window (an alarm already on before "
        "it opened does not show a reaction to the coming failure). *Chance* is the share of "
        "random H-long windows of normal operation (≥ 6 h before / 1 h after any failure) "
        "that the SAME alarms hit — the rate expected if the model knew nothing about the "
        "coming failure (number of reference windows in brackets). Anticipation is evidence "
        "only where it clearly exceeds chance.",
        "",
        "Per-failure detail is in `models/eval_anticipation.json`.",
        "",
        "**Limitation.** With six tested failures, this is exploratory evidence about "
        "early warning, not a validation of failure prediction; it complements — does not "
        "replace — the main detector results.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--horizons", nargs="+", type=int, default=HORIZONS_MIN)
    parser.add_argument("--models", nargs="+", default=MODELS)
    args = parser.parse_args()
    ctx, _ = build_context()
    result: dict[str, Any] = (
        json.loads(OUT_JSON.read_text(encoding="utf-8")) if OUT_JSON.exists() else {}
    )
    result.setdefault("horizons", {})
    for h in args.horizons:
        block = result["horizons"].setdefault(str(h), {})
        for name in args.models:
            t0 = time.perf_counter()
            block[name] = {
                "anticipation": run_trained(ctx, name, h),
                "detector": run_detector(ctx, name, h),
                "elapsed_s": round(time.perf_counter() - t0, 1),
            }
            OUT_JSON.write_text(
                json.dumps(result, indent=1, default=str), encoding="utf-8"
            )
    result.update(
        {
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "excluded_after_failure_h": POST_FAILURE_EXCLUDED.total_seconds() / 3600,
            "persistence_k": PERSISTENCE_K,
        }
    )
    OUT_JSON.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    OUT_MD.write_text(render(result), encoding="utf-8")
    log.info("wrote %s", OUT_MD)


if __name__ == "__main__":
    main()
