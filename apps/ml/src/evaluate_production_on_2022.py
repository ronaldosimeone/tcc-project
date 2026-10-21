"""
The deployed model on a recording period it never saw — RF-33 baseline.

Entry point:
    python -m src.evaluate_production_on_2022

``models/random_forest_v2.onnx`` (the model the API serves) was trained on
MetroPT-3 with a random 80/20 split and SMOTE (model_card.json). It is scored
here, WITHOUT retraining, on MetroPT 2022 — a genuinely out-of-period test.

Inputs
------
* ``as_deployed``         2022 columns fed by name, exactly as a naive
                          integration would (``Reservoirs`` and ``Oil_level``
                          carry a different signal in 2022).
* ``harmonized_at_input`` serving-time fix decided from NORMAL data only
                          (docs/cross_dataset_evaluation.md): ``Reservoirs``
                          := ``TP3`` (they are the same channel in 2020,
                          r = 1.000) and ``Oil_level`` := 1 (its mode on 2020
                          healthy rows). No failure row informs either rule.
* ``in_sample_2020``      the same model on MetroPT-3 — CONTAMINATED (it was
                          trained on these rows); reported only to show the size
                          of the drop.

Alarms: persistence k = 6 (RN-03) at the model's own threshold (0.6598,
model_card) and at the backend EVENT_THRESHOLD (0.7). Event metrics as in
RF-29. Output: models/eval_production_2022.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluate_temporal import EVAL_KW, PERSISTENCE_K
from src.event_metrics import evaluate_events, persistent_alarms
from src.ingest_metropt import FAILURE_INTERVALS
from src.ingest_metropt2022 import FAILURE_INTERVALS_2022
from src.preprocessing import MetroPTPreprocessor
from src.temporal_validation import events_from_intervals
from src.train_random_forest import FEATURE_COLS, TARGET_COL

_ML_ROOT: Path = Path(__file__).resolve().parent.parent
MODELS_DIR: Path = _ML_ROOT / "models"
P2020: Path = _ML_ROOT / "data" / "processed" / "metropt3.parquet"
P2022: Path = _ML_ROOT / "data" / "processed" / "metropt2022.parquet"
OUT_JSON: Path = MODELS_DIR / "eval_production_2022.json"
BACKEND_EVENT_THRESHOLD: float = 0.7
BATCH: int = 50_000


def _probabilities(sess: ort.InferenceSession, X: np.ndarray) -> np.ndarray:
    out: list[np.ndarray] = []
    for i in range(0, len(X), BATCH):
        probs = sess.run(["probabilities"], {"features": X[i : i + BATCH]})[0]
        if isinstance(probs, list):  # ZipMap output: list of {class: p}
            probs = np.array([[p[0], p[1]] for p in probs])
        out.append(np.asarray(probs)[:, 1])
    return np.concatenate(out)


def _score(
    df: pd.DataFrame, sess: ort.InferenceSession, feature_names: list[str]
) -> np.ndarray:
    feats = MetroPTPreprocessor().fit_transform(df[FEATURE_COLS].copy())
    return _probabilities(sess, feats[feature_names].to_numpy(np.float32))


def _report(
    ts: pd.Series,
    y: np.ndarray,
    p: np.ndarray,
    events: list[Any],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "prevalence": float(y.mean()),
    }
    for name, thr in thresholds.items():
        alarms = persistent_alarms(p, thr, PERSISTENCE_K)
        rep = evaluate_events(
            ts.reset_index(drop=True), alarms, events, **EVAL_KW
        ).to_dict()
        out[f"events_at_{name}"] = {
            "threshold": thr,
            "event_recall": rep["event_recall"],
            "false_alarms_per_day": rep["false_alarms_per_day"],
            "mean_lead_time_s": rep["mean_lead_time_s"],
            "expected_cost": rep["expected_cost"],
            "detected": {str(o["event_start"]): o["detected"] for o in rep["outcomes"]},
        }
    return out


def main() -> None:
    card = json.loads(
        (MODELS_DIR / "legacy" / "model_card.json").read_text(encoding="utf-8")
    )
    names: list[str] = card["feature_names"]
    thresholds = {
        "model_threshold": float(card["decision_threshold"]),
        "backend_event_threshold": BACKEND_EVENT_THRESHOLD,
    }
    sess = ort.InferenceSession(
        str(MODELS_DIR / "legacy" / "random_forest_v2.onnx"),
        providers=["CPUExecutionProvider"],
    )

    d20 = pd.read_parquet(P2020).sort_values("timestamp").reset_index(drop=True)
    d22 = pd.read_parquet(P2022).sort_values("timestamp").reset_index(drop=True)
    d20["timestamp"] = pd.to_datetime(d20["timestamp"])
    oil_mode = float(d20.loc[d20[TARGET_COL] == 0, "Oil_level"].mode().iloc[0])
    fixed = d22.copy()
    fixed["Reservoirs"] = fixed["TP3"]
    fixed["Oil_level"] = oil_mode

    ev22 = events_from_intervals(FAILURE_INTERVALS_2022)
    ev20 = events_from_intervals(FAILURE_INTERVALS)
    y22 = d22[TARGET_COL].to_numpy(int)
    result: dict[str, Any] = {
        "schema_version": "1.0",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "requirements": ["RF-33"],
        "model": "random_forest_v2.onnx",
        "model_training": {
            "split": f"random {card['test_size_ratio']:.0%} hold-out",
            "class_balancing": card["class_balancing"],
            "trained_at": card["trained_at"],
        },
        "serving_fix": {"Reservoirs": "TP3", "Oil_level": oil_mode},
        "as_deployed": _report(
            d22["timestamp"], y22, _score(d22, sess, names), ev22, thresholds
        ),
        "harmonized_at_input": _report(
            d22["timestamp"], y22, _score(fixed, sess, names), ev22, thresholds
        ),
        "in_sample_2020_contaminated": _report(
            d20["timestamp"],
            d20[TARGET_COL].to_numpy(int),
            _score(d20, sess, names),
            ev20,
            thresholds,
        ),
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    for key in ("as_deployed", "harmonized_at_input", "in_sample_2020_contaminated"):
        r = result[key]
        e = r["events_at_model_threshold"]
        print(
            f"{key}: pr_auc={r['pr_auc']:.3f} recall_ev={e['event_recall']:.2f} "
            f"fa/day={e['false_alarms_per_day']:.2f} detected={e['detected']}",
            flush=True,
        )
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
