"""
Challenger model trained on MetroPT-3 + MetroPT 2022 (harmonised) — RF-33/RN-05.

Entry point:
    python -m src.train_challenger [--seed 0] [--neg-stride 5]

What it produces (served by the backend as ``xgboost_v3``):
    models/xgboost_v3.onnx        ONNX (onnxmltools), input "features"
    models/xgboost_v3_card.json   feature list, threshold and the evidence

Recipe — the one evaluated in docs/cross_dataset_evaluation.md (harmonised):
* Data: both recording periods, ``Reservoirs`` and ``Oil_level`` excluded
  (same name, different signal across datasets). Rolling features computed
  per dataset (no window crosses the 2020→2022 gap).
* Class balance ONLY in training: every failure row + every ``neg_stride``-th
  normal row, plus ``scale_pos_weight`` = negatives / positives. No SMOTE:
  synthetic minority samples interpolated across a time series leak
  neighbouring rows (the deployed RF v2 used SMOTE + a random split and
  scores PR-AUC 0.058 on 2022, see models/eval_production_2022.json).
* Threshold: cost-optimal (RN-02, C_FN = 100·C_FP, persistence k = 6) on
  leave-one-failure-out OUT-OF-FOLD scores over the 7 failures (purge 300 s,
  embargo 1 h) — never on in-sample scores.
* Status: CHALLENGER. In the pooled 6-fold walk-forward XGBoost does not beat
  Random Forest with significance (RN-05 gate p = 0.438), so the gate does not
  allow promotion; the model goes to shadow/canary to collect live evidence.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from src import evaluate_cross_dataset as xd
from src.evaluate_temporal import MODELS, _inner_threshold, _subsample
from src.ingest_metropt import FAILURE_INTERVALS
from src.ingest_metropt2022 import FAILURE_INTERVALS_2022
from src.temporal_validation import events_from_intervals

_ML_ROOT: Path = Path(__file__).resolve().parent.parent
MODELS_DIR: Path = _ML_ROOT / "models"
NAME: str = "xgboost_v3"
DROPPED: list[str] = ["Reservoirs", "Oil_level"]
EVIDENCE_JSON: Path = MODELS_DIR / "eval_cross_dataset_harmonized.json"
# The default grid (0.05…0.95) put the optimum on its upper edge in the first
# run, so the top is refined. Chosen on TRAINING out-of-fold scores only.
THRESHOLD_GRID: list[float] = [
    *np.round(np.arange(0.05, 0.95, 0.05), 2).tolist(),
    *np.round(np.arange(0.95, 0.996, 0.005), 3).tolist(),
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log: logging.Logger = logging.getLogger("train_challenger")


def export_onnx(model: Any, n_features: int, path: Path) -> None:
    from onnxmltools.convert import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    onnx_model = convert_xgboost(
        model,
        initial_types=[("features", FloatTensorType([None, n_features]))],
        target_opset=15,
    )
    path.write_bytes(onnx_model.SerializeToString())


def onnx_probabilities(path: Path, X: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    probs = sess.run(["probabilities"], {"features": X})[0]
    if isinstance(probs, list):  # ZipMap
        probs = np.array([[p[0], p[1]] for p in probs])
    return np.asarray(probs)[:, 1]


def evidence() -> dict[str, Any]:
    """Leakage-free evidence for this recipe, copied into the card."""
    if not EVIDENCE_JSON.exists():
        return {"missing": str(EVIDENCE_JSON.name)}
    data = json.loads(EVIDENCE_JSON.read_text(encoding="utf-8"))
    pooled = data["pooled_walk_forward"]
    xgb = pooled["summary"]["per_model"]["xgboost"]
    return {
        "source": f"models/{EVIDENCE_JSON.name}",
        "protocol": f"pooled purged walk-forward, {pooled['n_folds']} folds, 7 failures",
        "expected_cost_inner": xgb["expected_cost_inner"],
        "event_recall_inner": xgb["event_recall_inner"],
        "false_alarms_per_day_inner": xgb["false_alarms_per_day_inner"],
        "pr_auc": xgb["pr_auc"],
        "rn05_gate_vs_random_forest": pooled["promotion_gate_rn05"][
            "xgboost_vs_random_forest"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--neg-stride", type=int, default=5)
    args = parser.parse_args()

    xd.configure(DROPPED)
    X, y, ts, is22 = xd.load_pooled()
    events = events_from_intervals([*FAILURE_INTERVALS, *FAILURE_INTERVALS_2022])
    all_idx = np.arange(len(y))
    log.info("data %s, %d positives, %d failures", X.shape, int(y.sum()), len(events))

    thr, source, oof, _ = _inner_threshold(
        "xgboost",
        args.seed,
        X,
        y,
        ts,
        all_idx,
        events,
        args.neg_stride,
        candidates=THRESHOLD_GRID,
    )
    log.info(
        "threshold %.4f (%s, %d OOF scores)",
        thr,
        source,
        0 if oof is None else len(oof),
    )

    tr = _subsample(all_idx, y, args.neg_stride)
    y_tr = y.to_numpy()[tr]
    pos_weight = float((len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1))
    model = MODELS["xgboost"](args.seed, pos_weight)
    model.fit(X.iloc[tr].to_numpy(np.float32), y_tr)

    onnx_path = MODELS_DIR / f"{NAME}.onnx"
    export_onnx(model, X.shape[1], onnx_path)
    sample = X.iloc[np.linspace(0, len(X) - 1, 5000).astype(int)].to_numpy(np.float32)
    diff = float(
        np.abs(
            onnx_probabilities(onnx_path, sample) - model.predict_proba(sample)[:, 1]
        ).max()
    )
    if diff > 1e-4:
        raise SystemExit(f"ONNX parity failed: max |Δp| = {diff:.2e}")
    log.info("ONNX parity ok (max |Δp| = %.1e)", diff)

    card = {
        "schema_version": "2.0",
        "model_name": NAME,
        "status": "challenger",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "XGBClassifier",
        "datasets": {
            "metropt3_2020": {"failures": len(FAILURE_INTERVALS)},
            "metropt_2022": {
                "failures": len(FAILURE_INTERVALS_2022),
                "source": "doi:10.5281/zenodo.6854240",
            },
        },
        "dropped_sensors": DROPPED,
        "class_balancing": (
            f"training only: all positives + every {args.neg_stride}th negative, "
            f"scale_pos_weight={pos_weight:.2f}; no SMOTE"
        ),
        "n_train_rows": int(tr.size),
        "n_train_positives": int(y_tr.sum()),
        "decision_threshold": round(thr, 4),
        "threshold_strategy": f"RN-02 cost-optimal on out-of-fold scores ({source})",
        "seed": args.seed,
        "onnx_parity_max_abs_diff": diff,
        "feature_count": int(X.shape[1]),
        "feature_names": list(X.columns),
        "evidence": evidence(),
        "promotion": "not promoted: RN-05 gate requires p < 0.05 over the pooled "
        "folds; serve as CHALLENGER (shadow + canary).",
    }
    (MODELS_DIR / f"{NAME}_card.json").write_text(
        json.dumps(card, indent=2, default=str), encoding="utf-8"
    )
    log.info("wrote %s and %s_card.json", onnx_path.name, NAME)


if __name__ == "__main__":
    main()
