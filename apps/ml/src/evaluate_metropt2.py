"""
Held-out test on MetroPT2 — failures that no model has ever seen.

Entry point:
    python -m src.evaluate_metropt2 [--no-oil-models] [--seed 0]

Test set: MetroPT2 rows AFTER 2022-06-02 15:49:50 (end of every training
set), two failures from maintenance reports — air leak 04/06 and oil leak
11–14/07 (``ingest_metropt2``). Nothing in this script trains on them.

Variants
--------
* ``production``  — the served artefacts (``models/*_v3|_v2`` + cards),
  scored exactly as the backend does: ONNX, scalers, card thresholds, k = 6.
  They were trained on all data, which contains ONE oil leak (MetroPT 2022).
* ``no_oil``      — same nested recipe, trained only on data BEFORE the 2022
  oil leak (minus purge): models that have never seen an oil failure. The
  pair answers whether one example of a failure type is enough to recognise
  the next one.
* ``physics``     — grey-box twin calibrated on MetroPT2's own NORMAL weeks
  (28/04 → 02/06, no labels): per-asset calibration, no failure examples.

Metrics per variant and failure: detected (alarm inside [start − 6 h, end]),
lead time vs the failure start AND vs the train's own LPS low-pressure alarm,
false-alarm episodes per day, expected cost (RN-02), confusion matrices.

Output: models/eval_metropt2.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd

import src.evaluate_cross_dataset as xd
import src.nested_protocol as npc
from src.confusion import fold_report
from src.digital_twin import PhysicsTwin
from src.evaluate_nested import build_context
from src.evaluate_temporal import EVAL_KW, PERSISTENCE_K, PURGE_S
from src.event_metrics import evaluate_events, persistent_alarms
from src.ingest_metropt2 import (
    FAILURE_INTERVALS_MT2,
    FAILURE_TYPES_MT2,
    LPS_ALARMS_MT2,
    OUT_PARQUET,
    TRAINING_END,
)
from src.temporal_validation import events_from_intervals

_ML_ROOT: Path = Path(__file__).resolve().parent.parent
MODELS_DIR: Path = _ML_ROOT / "models"
OUT_JSON: Path = MODELS_DIR / "eval_metropt2.json"
# Per-row scores of every variant (test rows only) for re-analysis.
OUT_SCORES: Path = MODELS_DIR / "eval_metropt2_scores.npz"
SCORES: dict[str, np.ndarray] = {}
ARTEFACTS: dict[str, str] = {
    "random_forest": "random_forest_v3",
    "xgboost": "xgboost_v3",
    "mlp": "mlp_v2",
    "tcn": "tcn_v2",
    "bilstm": "bilstm_v2",
    "patchtst": "patchtst_v2",
    "autoencoder": "autoencoder_v2",
}
OIL_2022_START: pd.Timestamp = pd.Timestamp("2022-05-30 12:00")
WINDOW: int = 60
BATCH: int = 8192

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("evaluate_metropt2")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_mt2() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Harmonised features and raw channels for the WHOLE MetroPT2 series
    (rolling windows need the history), plus the test mask (after training)."""
    df = pd.read_parquet(OUT_PARQUET).sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    feats = xd._features(df[["timestamp", *xd.FEATURES, "anomaly"]], [])
    test = (df["timestamp"] > TRAINING_END).to_numpy()
    return df, feats, test


def _session(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _softmax1(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True))[:, 1]


def _windows(raw: np.ndarray, ends: np.ndarray) -> np.ndarray:
    offs = np.arange(-WINDOW + 1, 1)
    return raw[ends[:, None] + offs[None, :]]


def score_served(
    name: str, df: pd.DataFrame, feats: pd.DataFrame, rows: np.ndarray
) -> tuple[np.ndarray, float]:
    """(score, threshold) exactly as the backend serves the artefact."""
    artefact = ARTEFACTS[name]
    card = json.loads(
        (MODELS_DIR / f"{artefact}_card.json").read_text(encoding="utf-8")
    )
    serving = card["serving"]
    sess = _session(MODELS_DIR / serving["onnx"])
    names: list[str] = card["feature_names"]
    if serving["adapter"] == "tree":
        X = feats[names].to_numpy(np.float32)[rows]
        out = []
        for i in range(0, len(X), BATCH):
            p = sess.run(["probabilities"], {"features": X[i : i + BATCH]})[0]
            if isinstance(p, list):
                p = np.array([[d[0], d[1]] for d in p])
            out.append(np.asarray(p)[:, 1])
        return np.concatenate(out), float(card["decision_threshold"])

    scaler = joblib.load(MODELS_DIR / serving["scaler"])
    if serving["adapter"] == "mlp":
        X = scaler.transform(feats[names].to_numpy(np.float32)[rows]).astype(np.float32)
        logits = np.concatenate(
            [
                sess.run(None, {"input": X[i : i + BATCH]})[0]
                for i in range(0, len(X), BATCH)
            ]
        )
        return _softmax1(logits), float(card["decision_threshold"])

    raw = df[names].to_numpy(np.float32)
    scaled = ((raw - scaler.mean_) / scaler.scale_).astype(np.float32)
    ends = np.clip(rows, WINDOW - 1, None)
    out = []
    for i in range(0, len(ends), BATCH):
        w = _windows(scaled, ends[i : i + BATCH])
        y = sess.run(None, {"input": w})[0]
        if serving["adapter"] == "autoencoder":
            out.append(((y - w) ** 2).mean(axis=(1, 2)))
        else:
            out.append(_softmax1(y))
    score = np.concatenate(out)
    thr = (
        float(serving["mse_threshold"])
        if serving["adapter"] == "autoencoder"
        else float(card["decision_threshold"])
    )
    return score, thr


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


# A "sustained" warning is the first alarm episode (gaps ≤ EPISODE_GAP) that
# lasts ≥ SUSTAINED_MIN: an isolated 1-minute blip hours before a failure is
# not a warning an operator would act on. Declared before looking at the
# per-failure lead of the no-oil variant.
EPISODE_GAP: pd.Timedelta = pd.Timedelta(minutes=5)
SUSTAINED_MIN: pd.Timedelta = pd.Timedelta(minutes=10)


def _minutes(ref: pd.Timestamp, t: pd.Timestamp | None) -> float | None:
    return None if t is None else round((ref - t).total_seconds() / 60, 1)


def sustained_onset(alarm_ts: pd.Series) -> pd.Timestamp | None:
    if alarm_ts.empty:
        return None
    episode = (alarm_ts.diff() > EPISODE_GAP).cumsum()
    spans = alarm_ts.groupby(episode).agg(["min", "max"])
    long = spans[(spans["max"] - spans["min"]) >= SUSTAINED_MIN]
    return None if long.empty else pd.Timestamp(long["min"].iloc[0])


def lead_report(ts: pd.Series, alarms: np.ndarray) -> list[dict[str, Any]]:
    """Per failure: first alarm in [start − early horizon, end] and its lead
    vs the failure start and vs the train's LPS alarm (positive = earlier),
    plus the same leads for the first SUSTAINED episode."""
    out = []
    horizon = pd.Timedelta(seconds=EVAL_KW["early_horizon_s"])
    for (s, e), kind, lps in zip(
        FAILURE_INTERVALS_MT2, FAILURE_TYPES_MT2, LPS_ALARMS_MT2
    ):
        start, end, lps_t = pd.Timestamp(s), pd.Timestamp(e), pd.Timestamp(lps)
        m = ((ts >= start - horizon) & (ts <= end)).to_numpy() & alarms.astype(bool)
        first = ts[m].min() if m.any() else None
        sustained = sustained_onset(ts[m].reset_index(drop=True))
        out.append(
            {
                "type": kind,
                "detected": first is not None,
                "first_alarm": None if first is None else str(first),
                "lead_vs_start_min": _minutes(start, first),
                "lead_vs_lps_min": _minutes(lps_t, first),
                "sustained_detected": sustained is not None,
                "sustained_alarm": None if sustained is None else str(sustained),
                "sustained_lead_vs_start_min": _minutes(start, sustained),
                "sustained_lead_vs_lps_min": _minutes(lps_t, sustained),
            }
        )
    return out


def report(
    ts: pd.Series, y: np.ndarray, score: np.ndarray, thr: float
) -> dict[str, Any]:
    events = events_from_intervals(FAILURE_INTERVALS_MT2)
    rep = fold_report((score >= thr).astype(float), y, ts, 0.5, events)
    alarms = persistent_alarms((score >= thr).astype(float), 0.5, PERSISTENCE_K)
    ev = evaluate_events(ts.reset_index(drop=True), alarms, events, **EVAL_KW).to_dict()
    rep["failures"] = lead_report(ts.reset_index(drop=True), alarms)
    rep["false_alarms_per_day"] = ev["false_alarms_per_day"]
    rep["threshold"] = thr
    return rep


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------


def variant_production(
    df: pd.DataFrame, feats: pd.DataFrame, test: np.ndarray
) -> dict[str, Any]:
    rows = np.flatnonzero(test)
    ts, y = df["timestamp"].iloc[rows], df["anomaly"].to_numpy()[rows]
    out = {}
    for name in ARTEFACTS:
        score, thr = score_served(name, df, feats, rows)
        SCORES[f"production/{name}"] = (score >= thr).astype(np.float32)
        out[name] = report(ts, y, score, thr)
        log.info(
            "production %s: %s", name, [f["detected"] for f in out[name]["failures"]]
        )
    return out


def variant_no_oil(
    df: pd.DataFrame, feats: pd.DataFrame, test: np.ndarray, seed: int
) -> dict[str, Any]:
    """Nested recipe trained only before the 2022 oil leak, scored on MT2."""
    ctx, _ = build_context()
    cut = OIL_2022_START - pd.Timedelta(seconds=PURGE_S)
    train_idx = np.flatnonzero((ctx.ts < cut).to_numpy())
    rows = np.flatnonzero(test)
    ts, y = df["timestamp"].iloc[rows], df["anomaly"].to_numpy()[rows]
    # Score MT2 with models fitted in the pooled context: append MT2 rows to it.
    mt2_ctx, offset = _append_mt2(ctx, df, feats)
    out = {}
    for name in ARTEFACTS:
        sel = npc.select(mt2_ctx, name, train_idx, seed, tune=True)
        model = npc.fit_model(mt2_ctx, name, train_idx, seed, sel.params)
        raw = npc.raw_score(mt2_ctx, name, model, rows + offset)
        SCORES[f"no_oil/{name}"] = (raw >= sel.raw_threshold).astype(np.float32)
        out[name] = report(ts, y, (raw >= sel.raw_threshold).astype(float), 0.5)
        out[name]["params"] = sel.params
        log.info("no_oil %s: %s", name, [f["detected"] for f in out[name]["failures"]])
    return out


def _append_mt2(
    ctx: npc.Context, df: pd.DataFrame, feats: pd.DataFrame
) -> tuple[npc.Context, int]:
    """Context whose rows are [pooled training data | MetroPT2]; MT2 rows are
    only ever scored, never trained on (they are after every training cut)."""
    import src.deep_temporal as dt

    offset = len(ctx.y)
    X = pd.concat([ctx.X, feats[ctx.X.columns]], ignore_index=True)
    y = pd.concat([ctx.y, df["anomaly"].astype(int)], ignore_index=True)
    ts = pd.concat([ctx.ts, df["timestamp"]], ignore_index=True)
    is_new = np.r_[np.zeros(offset, bool), np.ones(len(df), bool)]
    raw_df = pd.concat(
        [
            pd.DataFrame(ctx.data.raw, columns=ctx.data.channels),
            df[ctx.data.channels].reset_index(drop=True),
        ],
        ignore_index=True,
    )
    raw_df.insert(0, "timestamp", ts)
    raw_df["anomaly"] = y
    data = dt.build_data(raw_df, X, y, ts, is_new)
    # build_data marks valid windows from the start of the LAST segment; keep
    # the pooled context's own validity for its rows.
    data.valid_end[:offset] = ctx.data.valid_end
    data.clean_window[:offset] = ctx.data.clean_window
    return npc.Context(X=X, y=y, ts=ts, events=ctx.events, data=data), offset


def variant_physics(df: pd.DataFrame, test: np.ndarray) -> dict[str, Any]:
    """Twin calibrated on MetroPT2's own normal weeks; alarm when the
    standardised residual magnitude exceeds its 99.9th healthy percentile."""
    calib = (~test) & (df["anomaly"].to_numpy() == 0)
    twin = PhysicsTwin(pressure_col="TP3").fit(df[calib])
    resid = twin.transform(df)
    mu, sd = resid[calib].mean(), resid[calib].std() + 1e-9
    z = ((resid - mu) / sd).abs().max(axis=1).to_numpy()
    thr = float(np.quantile(z[calib], 0.999))
    rows = np.flatnonzero(test)
    rep = report(
        df["timestamp"].iloc[rows], df["anomaly"].to_numpy()[rows], z[rows], thr
    )
    rep["calibration"] = "MetroPT2 normal rows 28/04→02/06, no labels"
    return {"physics_twin": rep}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--no-oil-models", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    xd.configure(["Reservoirs", "Oil_level"])
    df, feats, test = load_mt2()
    result: dict[str, Any] = {}
    if OUT_JSON.exists():
        result = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    result.update(
        {
            "schema_version": "1.0",
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "requirements": ["RF-33", "RN-01", "RN-03"],
            "test_period": [
                str(df["timestamp"][test].min()),
                str(df["timestamp"][test].max()),
            ],
            "failures": [
                {"start": s, "end": e, "type": t, "lps_alarm": lps}
                for (s, e), t, lps in zip(
                    FAILURE_INTERVALS_MT2, FAILURE_TYPES_MT2, LPS_ALARMS_MT2
                )
            ],
            "persistence_k": PERSISTENCE_K,
            "early_horizon_h": EVAL_KW["early_horizon_s"] / 3600,
        }
    )
    result["production"] = variant_production(df, feats, test)
    result["physics"] = variant_physics(df, test)
    if args.no_oil_models:
        result["no_oil"] = variant_no_oil(df, feats, test, args.seed)
    OUT_JSON.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    previous = dict(np.load(OUT_SCORES)) if OUT_SCORES.exists() else {}
    previous.pop("timestamp", None)
    np.savez_compressed(
        OUT_SCORES,
        timestamp=df["timestamp"][test].astype("int64").to_numpy(),
        **(previous | SCORES),
    )
    log.info("wrote %s", OUT_JSON)


if __name__ == "__main__":
    main()
