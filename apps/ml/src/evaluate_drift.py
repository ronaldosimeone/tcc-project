"""
Controlled drift-injection study — RF-34.

Entry point:
    python -m src.evaluate_drift [--trials 30] [--calib-trials 30]

Data: MetroPT-3 NORMAL operation only (failure intervals ±6 h removed),
the 7 analogue sensors, raw 10 s samples.

Detectors (alarm if ANY sensor alarms):
    psi_daily     production rule family (RF-27): at each midnight, PSI of the
                  last 24 h of RAW samples vs the reference, alarm if > θ
    page_hinkley  two-sided, on hourly residuals
    adwin         confidence δ, window ≤ 14 days, on hourly residuals
Hourly residual = hourly mean − reference mean for that hour of day, divided
by the reference residual std (removes the daily operating cycle, which
otherwise violates the detectors' independence assumption).

Reference policies (the "adaptive" part of RF-34):
    fixed     first 30 days of the dataset (what RF-27 does today)
    rolling   the 30 days ending 7 days before the monitored stretch

Protocol:
1. False alarms of the production rule (PSI > 0.25, fixed reference) on null
   (no-injection) 20-day stretches.
2. For each (detector, reference policy): calibrate the most sensitive
   threshold with ≤ 1 false alarm / 30 days on null stretches.
3. Inject drift on raw data of a random sensor (abrupt: +k·σ from t0;
   gradual: linear ramp to k·σ over 72 h), k ∈ {0.5, 1, 2}, σ = reference
   std of that sensor, and measure detection within 7 days + delay.
   Only calibrated detectors are compared: a detector that alarms every
   day "detects" any drift instantly.

Output: models/eval_drift.json
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

from src.drift_detectors import Adwin, TwoSidedPageHinkley, psi  # noqa: E402
from src.ingest_metropt import FAILURE_INTERVALS  # noqa: E402
from src.stat_tests import bootstrap_ci  # noqa: E402

PARQUET: Path = _ML_ROOT / "data" / "processed" / "metropt3.parquet"
OUT_JSON: Path = _ML_ROOT / "models" / "eval_drift.json"
SENSORS: tuple[str, ...] = (
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
)
REF_DAYS: int = 30
ROLLING_GAP_DAYS: int = 7
STRETCH_DAYS: int = 20
RAMP_H: float = 72.0
DETECT_WINDOW_H: int = 7 * 24
FA_BUDGET_PER_30D: float = 1.0
PRODUCTION_PSI: float = 0.25
REF_SAMPLE: int = 20_000

GRIDS: dict[str, list[float]] = {
    "psi_daily": [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
    "page_hinkley": [15.0, 30.0, 60.0, 120.0, 250.0, 500.0, 1000.0, 2000.0],
    "adwin": [2e-3, 1e-5, 1e-10, 1e-20, 1e-40, 1e-80],
}
POLICIES: tuple[str, ...] = ("fixed", "rolling")


def normal_raw() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET, columns=["timestamp", *SENSORS])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    bad = np.zeros(len(df), dtype=bool)
    for s, e in FAILURE_INTERVALS:
        bad |= (df["timestamp"] >= pd.Timestamp(s) - pd.Timedelta(hours=6)) & (
            df["timestamp"] <= pd.Timestamp(e) + pd.Timedelta(hours=6)
        )
    return df[~bad].set_index("timestamp").sort_index()


class Reference:
    """Reference sample + hour-of-day profile for one policy and stretch."""

    def __init__(self, raw: pd.DataFrame, rng: np.random.Generator) -> None:
        vals = raw[list(SENSORS)].to_numpy()
        take = rng.choice(len(vals), size=min(REF_SAMPLE, len(vals)), replace=False)
        self.sample: np.ndarray = vals[take]
        self.sigma: np.ndarray = vals.std(axis=0)
        hourly = raw[list(SENSORS)].resample("1h").mean().dropna()
        self.profile: pd.DataFrame = hourly.groupby(hourly.index.hour).mean()
        resid = hourly.to_numpy() - self.profile.loc[hourly.index.hour].to_numpy()
        self.resid_std: np.ndarray = resid.std(axis=0) + 1e-9


def reference_for(
    raw: pd.DataFrame,
    policy: str,
    stretch_start: pd.Timestamp,
    rng: np.random.Generator,
) -> Reference:
    t0 = raw.index[0]
    if policy == "fixed":
        window = raw[raw.index < t0 + pd.Timedelta(days=REF_DAYS)]
    else:
        end = stretch_start - pd.Timedelta(days=ROLLING_GAP_DAYS)
        window = raw[
            (raw.index >= end - pd.Timedelta(days=REF_DAYS)) & (raw.index < end)
        ]
    return Reference(window, rng)


def alarms(
    detector: str, param: float, stretch: pd.DataFrame, ref: Reference
) -> list[float]:
    """Alarm times in HOURS since the stretch start (any sensor)."""
    start = stretch.index[0]
    out: set[float] = set()
    if detector == "psi_daily":
        days = pd.date_range(
            start.normalize() + pd.Timedelta(days=1), stretch.index[-1], freq="1D"
        )
        for day in days:
            window = stretch[
                (stretch.index >= day - pd.Timedelta(days=1)) & (stretch.index < day)
            ]
            if len(window) < 500:
                continue
            vals = window[list(SENSORS)].to_numpy()
            if any(
                psi(ref.sample[:, j], vals[:, j]) > param for j in range(len(SENSORS))
            ):
                out.add((day - start).total_seconds() / 3600)
        return sorted(out)

    hourly = stretch[list(SENSORS)].resample("1h").mean().dropna()
    z = (
        hourly.to_numpy() - ref.profile.loc[hourly.index.hour].to_numpy()
    ) / ref.resid_std
    hours = (hourly.index - start).total_seconds() / 3600
    for j in range(z.shape[1]):
        det: Any = (
            TwoSidedPageHinkley(delta=0.1, threshold=param)
            if detector == "page_hinkley"
            else Adwin(delta=param, max_window=24 * 14)
        )
        for i in range(z.shape[0]):
            if det.update(z[i, j]):
                out.add(float(hours[i]))
    return sorted(out)


def sample_stretch(raw: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    earliest = raw.index[0] + pd.Timedelta(days=REF_DAYS + ROLLING_GAP_DAYS)
    latest = raw.index[-1] - pd.Timedelta(days=STRETCH_DAYS)
    span = (latest - earliest).total_seconds()
    start = earliest + pd.Timedelta(seconds=float(rng.uniform(0, span)))
    return raw[
        (raw.index >= start) & (raw.index < start + pd.Timedelta(days=STRETCH_DAYS))
    ]


def fa_per_30d(
    detector: str,
    param: float,
    nulls: list[tuple[pd.DataFrame, dict[str, Reference]]],
    policy: str,
) -> float:
    counts = [len(alarms(detector, param, s, refs[policy])) for s, refs in nulls]
    return float(np.mean(counts)) * 30 / STRETCH_DAYS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials", type=int, default=30, help="injected trials per (kind, k) cell"
    )
    parser.add_argument("--calib-trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    raw = normal_raw()

    nulls: list[tuple[pd.DataFrame, dict[str, Reference]]] = []
    for _ in range(args.calib_trials):
        s = sample_stretch(raw, rng)
        nulls.append((s, {p: reference_for(raw, p, s.index[0], rng) for p in POLICIES}))

    production = {
        p: fa_per_30d("psi_daily", PRODUCTION_PSI, nulls, p) for p in POLICIES
    }
    print(f"production PSI>{PRODUCTION_PSI} false alarms/30d: {production}", flush=True)

    calibration: dict[str, dict[str, Any]] = {}
    for d in GRIDS:
        for p in POLICIES:
            curve: list[dict[str, float]] = []
            chosen: float | None = None
            for param in GRIDS[d]:
                fa = fa_per_30d(d, param, nulls, p)
                curve.append({"param": param, "false_alarms_per_30d": fa})
                print(f"  calib {d}/{p} param={param:g} FA/30d={fa:.2f}", flush=True)
                if fa <= FA_BUDGET_PER_30D:
                    chosen = param
                    break
            calibration[f"{d}/{p}"] = {"chosen": chosen, "curve": curve}

    usable = {k: v["chosen"] for k, v in calibration.items() if v["chosen"] is not None}
    results: dict[str, Any] = {}
    for kind in ("abrupt", "gradual"):
        for k in (0.5, 1.0, 2.0):
            per: dict[str, list[tuple[bool, float | None]]] = {
                key: [] for key in usable
            }
            for _ in range(args.trials):
                s = sample_stretch(raw, rng).copy()
                refs = {p: reference_for(raw, p, s.index[0], rng) for p in POLICIES}
                t0_h = float(rng.uniform(STRETCH_DAYS * 24 / 4, STRETCH_DAYS * 24 / 2))
                j = int(rng.integers(0, len(SENSORS)))
                el_h = (s.index - s.index[0]).total_seconds().to_numpy() / 3600 - t0_h
                ramp = (
                    (el_h >= 0).astype(float)
                    if kind == "abrupt"
                    else np.clip(el_h / RAMP_H, 0.0, 1.0)
                )
                s[SENSORS[j]] = (
                    s[SENSORS[j]].to_numpy() + k * refs["fixed"].sigma[j] * ramp
                )
                for key, param in usable.items():
                    d, p = key.split("/")
                    hits = [
                        h
                        for h in alarms(d, param, s, refs[p])
                        if t0_h <= h < t0_h + DETECT_WINDOW_H
                    ]
                    per[key].append((bool(hits), (hits[0] - t0_h) if hits else None))
            cell: dict[str, Any] = {}
            for key, vals in per.items():
                det = np.array([v[0] for v in vals], dtype=float)
                delays = np.array([v[1] for v in vals if v[1] is not None], dtype=float)
                cell[key] = {
                    "detection_rate": bootstrap_ci(det).to_dict(),
                    "delay_h": bootstrap_ci(delays).to_dict(),
                }
            results[f"{kind}_{k}"] = cell
            print(
                kind,
                k,
                json.dumps(
                    {
                        key: [
                            round(v["detection_rate"]["mean"], 2),
                            round(v["delay_h"]["mean"], 1),
                        ]
                        for key, v in cell.items()
                    }
                ),
                flush=True,
            )

    out = {
        "schema_version": "3.0",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "requirements": ["RF-34"],
        "config": {
            "trials_per_cell": args.trials,
            "calibration_trials": args.calib_trials,
            "seed": args.seed,
            "reference_days": REF_DAYS,
            "rolling_gap_days": ROLLING_GAP_DAYS,
            "stretch_days": STRETCH_DAYS,
            "ramp_hours": RAMP_H,
            "detection_window_hours": DETECT_WINDOW_H,
            "false_alarm_budget_per_30d": FA_BUDGET_PER_30D,
            "sensors": list(SENSORS),
        },
        "production_rule_false_alarms_per_30d": production,
        "calibration": calibration,
        "results": results,
    }
    OUT_JSON.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
