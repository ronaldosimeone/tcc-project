"""
RAG diagnosis cases from the REAL failures (RF-35 / RF-36, honest version).

Entry point:
    python -m src.build_diagnosis_cases

The original RAG benchmark (apps/backend/benchmarks/rag_cases_v1.json) uses
operator-written symptoms and hand-picked sensors. Here the evidence comes
from the data itself, for every documented failure of the three datasets
(MetroPT-3 2020 ×4, MetroPT 2022 ×3, MetroPT2 ×2), so the question becomes:
given only what the platform measures, does retrieval reach the manual
section of the component that actually failed?

Evidence per failure, on the ONSET window [start, start + 30 min] (fixed a
priori, never tuned):

* ``occlusion``  — exactly the online RCA of production
  (apps/backend/src/services/root_cause_service.py): each sensor's raw
  channel is replaced by its normal median, features are recomputed and the
  drop in P(failure) of the ACTIVE served model (random_forest_v3) is the
  contribution. ``p_failure_onset`` and ``alarm_in_onset`` say whether
  production would have opened an event (RCA only runs on events).
* ``deviation``  — model-agnostic signature: |z| of the onset mean against the
  7 healthy days before the failure (same asset). Signed z is kept so the
  operator-style symptom can say "acima/abaixo do normal".

Output: apps/backend/benchmarks/rag_diagnosis_cases.json (consumed by
apps/backend/benchmarks/benchmark_rag_diagnosis.py).
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import src.evaluate_cross_dataset as xd
from src.evaluate_metropt2 import score_served
from src.ingest_metropt import FAILURE_INTERVALS
from src.ingest_metropt2 import (
    FAILURE_INTERVALS_MT2,
    FAILURE_TYPES_MT2,
    TRAINING_END,
)
from src.ingest_metropt2022 import FAILURE_INTERVALS_2022

_ML_ROOT: Path = Path(__file__).resolve().parent.parent
DATA: Path = _ML_ROOT / "data" / "processed"
OUT_JSON: Path = _ML_ROOT.parent / "backend" / "benchmarks" / "rag_diagnosis_cases.json"

ACTIVE_MODEL: str = "random_forest"  # backend ACTIVE_MODEL default (random_forest_v3)
ONSET: pd.Timedelta = pd.Timedelta(minutes=30)
HISTORY: pd.Timedelta = pd.Timedelta(hours=6)  # rolling-feature warm-up
BASELINE: pd.Timedelta = pd.Timedelta(days=7)
DROP: list[str] = ["Reservoirs", "Oil_level"]  # harmonisation (not served)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("build_diagnosis_cases")


@dataclass(frozen=True)
class Failure:
    id: str
    dataset: str
    parquet: str
    start: pd.Timestamp
    end: pd.Timestamp
    failure_type: str  # air_leak | air_dryer | oil_leak
    source: str


def failures() -> list[Failure]:
    out: list[Failure] = []
    # MetroPT-3 data descriptor (Veloso et al., 2022): the four are air leaks.
    for i, (s, e) in enumerate(FAILURE_INTERVALS, 1):
        out.append(
            Failure(
                f"F20_{i}",
                "MetroPT-3",
                "metropt3.parquet",
                pd.Timestamp(s),
                pd.Timestamp(e),
                "air_leak",
                "MetroPT-3 incident report",
            )
        )
    for i, ((s, e), kind) in enumerate(
        zip(FAILURE_INTERVALS_2022, ["air_leak", "air_dryer", "oil_leak"]), 1
    ):
        out.append(
            Failure(
                f"F22_{i}",
                "MetroPT 2022",
                "metropt2022.parquet",
                pd.Timestamp(s),
                pd.Timestamp(e),
                kind,
                "MetroPT 2022 maintenance report",
            )
        )
    for i, ((s, e), kind) in enumerate(
        zip(FAILURE_INTERVALS_MT2, FAILURE_TYPES_MT2), 1
    ):
        out.append(
            Failure(
                f"FM2_{i}",
                "MetroPT2",
                "metropt2.parquet",
                pd.Timestamp(s),
                pd.Timestamp(e),
                kind,
                "arXiv:2502.07394 Table 1",
            )
        )
    return out


def _segment(df: pd.DataFrame, f: Failure) -> pd.DataFrame:
    ts = df["timestamp"]
    lo, hi = f.start - HISTORY, min(f.start + ONSET, f.end)
    return df[(ts >= lo) & (ts <= hi)].reset_index(drop=True)


def _baseline(df: pd.DataFrame, f: Failure) -> pd.DataFrame:
    ts = df["timestamp"]
    m = (ts >= f.start - BASELINE) & (ts < f.start) & (df["anomaly"] == 0)
    return df[m]


def _p_onset(
    seg: pd.DataFrame, onset_rows: np.ndarray
) -> tuple[float, np.ndarray, float]:
    feats = xd._features(seg[["timestamp", *xd.FEATURES, "anomaly"]], [])
    score, thr = score_served(ACTIVE_MODEL, seg, feats, onset_rows)
    return float(score.mean()), score, thr


def evidence(df: pd.DataFrame, f: Failure) -> dict[str, Any]:
    seg = _segment(df, f)
    base = _baseline(df, f)
    onset = np.flatnonzero((seg["timestamp"] >= f.start).to_numpy())
    p_real, score, thr = _p_onset(seg, onset)

    medians = base[xd.FEATURES].median()
    occl: list[dict[str, float | str]] = []
    for sensor in xd.FEATURES:
        neutral = seg.copy()
        neutral[sensor] = medians[sensor]
        p_neutral, _, _ = _p_onset(neutral, onset)
        occl.append({"sensor": sensor, "contribution": round(p_real - p_neutral, 6)})
    occl.sort(key=lambda c: -abs(float(c["contribution"])))

    on = seg.iloc[onset]
    z = (on[xd.FEATURES].mean() - base[xd.FEATURES].mean()) / (
        base[xd.FEATURES].std() + 1e-9
    )
    total = float(z.abs().sum()) or 1.0
    dev = sorted(
        (
            {
                "sensor": s,
                "z": round(float(z[s]), 3),
                "contribution": round(float(abs(z[s])) / total, 6),
            }
            for s in xd.FEATURES
        ),
        key=lambda c: -float(c["contribution"]),
    )
    return {
        "p_failure_onset": round(p_real, 4),
        "threshold": thr,
        "alarm_in_onset": bool((score >= thr).any()),
        "onset_rows": int(len(onset)),
        "baseline_rows": int(len(base)),
        "occlusion": occl,
        "deviation": dev,
    }


def main() -> None:
    xd.configure(DROP)
    cache: dict[str, pd.DataFrame] = {}
    cases: list[dict[str, Any]] = []
    for f in failures():
        if f.parquet not in cache:
            df = pd.read_parquet(DATA / f.parquet)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            cache[f.parquet] = df.sort_values("timestamp").reset_index(drop=True)
        ev = evidence(cache[f.parquet], f)
        log.info(
            "%s %s p=%.3f alarm=%s occl=%s dev=%s",
            f.id,
            f.failure_type,
            ev["p_failure_onset"],
            ev["alarm_in_onset"],
            [c["sensor"] for c in ev["occlusion"][:3]],
            [c["sensor"] for c in ev["deviation"][:3]],
        )
        cases.append(
            {
                "id": f.id,
                "dataset": f.dataset,
                "failure_type": f.failure_type,
                "start": str(f.start),
                "end": str(f.end),
                "seen_in_training": bool(f.end <= pd.Timestamp(TRAINING_END)),
                "label_source": f.source,
                **ev,
            }
        )
    OUT_JSON.write_text(
        json.dumps(
            {
                "version": "1.0",
                "requirements": ["RF-35", "RF-36", "RN-04"],
                "active_model": "random_forest_v3",
                "onset_minutes": ONSET.total_seconds() / 60,
                "baseline_days": BASELINE.days,
                "sensors": list(xd.FEATURES),
                "cases": cases,
            },
            indent=1,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    log.info("wrote %s (%d cases)", OUT_JSON, len(cases))


if __name__ == "__main__":
    main()
