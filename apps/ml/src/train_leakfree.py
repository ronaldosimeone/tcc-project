"""
Leakage-free production training for the tree models — RF-33 / RN-05.

Entry points:
    python -m src.train_leakfree --model random_forest --role champion
    python -m src.train_leakfree --model xgboost --role challenger

Artefacts (served by the backend):
    random_forest -> models/random_forest_v3.onnx + random_forest_v3_card.json
    xgboost       -> models/xgboost_v3.onnx       + xgboost_v3_card.json

The recipe is EXACTLY the one evaluated without leakage in
docs/cross_dataset_evaluation.md (pooled MetroPT-3 + MetroPT 2022, harmonised):

* Data: both recording periods; ``Reservoirs`` and ``Oil_level`` excluded
  (same name, different signal across datasets). Rolling features computed
  per dataset — no window crosses the 2020 → 2022 gap.
* Class balance ONLY in training: every failure row + every ``neg_stride``-th
  normal row, plus class weights (RF ``balanced_subsample``, XGBoost
  ``scale_pos_weight``). No SMOTE: synthetic minority rows interpolated across
  a time series leak neighbouring samples (the legacy RF v2 used SMOTE + a
  random split: PR-AUC 1.000 on rows it saw, 0.058 and 0/3 failures on 2022).
* Threshold: cost-optimal (RN-02, C_FN = 100·C_FP, persistence k = 6) on
  leave-one-failure-out OUT-OF-FOLD scores over the 7 failures (purge 300 s,
  embargo 1 h). The grid is refined at the top because the optimum sat on the
  default grid's upper edge — decided on training scores only.
* The card stores the walk-forward evidence of the recipe and the role.

Roles
-----
* ``champion``   — the legacy champion (RF v2) has no valid evidence, so it is
  replaced by the SAME algorithm trained with the validated recipe. This is a
  corrective rebuild, not a promotion between candidates; RN-05 still governs
  any change of algorithm (e.g. xgboost_v3 vs random_forest_v3).
* ``challenger`` — served in shadow/canary until the RN-05 gate approves it.
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

import src.evaluate_cross_dataset as xd
from src.evaluate_temporal import MODELS, _inner_threshold, _subsample
from src.ingest_metropt import FAILURE_INTERVALS
from src.ingest_metropt2022 import FAILURE_INTERVALS_2022
from src.temporal_validation import events_from_intervals

_ML_ROOT: Path = Path(__file__).resolve().parent.parent
MODELS_DIR: Path = _ML_ROOT / "models"
DROPPED: list[str] = ["Reservoirs", "Oil_level"]
EVIDENCE_JSON: Path = MODELS_DIR / "eval_cross_dataset_harmonized.json"
ARTEFACT: dict[str, str] = {
    "random_forest": "random_forest_v3",
    "xgboost": "xgboost_v3",
}
THRESHOLD_GRID: list[float] = [
    *np.round(np.arange(0.05, 0.95, 0.05), 2).tolist(),
    *np.round(np.arange(0.95, 0.996, 0.005), 3).tolist(),
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log: logging.Logger = logging.getLogger("train_leakfree")


def export_onnx(model_kind: str, model: Any, n_features: int, path: Path) -> None:
    if model_kind == "xgboost":
        from onnxmltools.convert import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType

        onnx_model = convert_xgboost(
            model,
            initial_types=[("features", FloatTensorType([None, n_features]))],
            target_opset=15,
        )
    else:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType as SkFloat

        onnx_model = convert_sklearn(
            model,
            initial_types=[("features", SkFloat([None, n_features]))],
            target_opset=17,
            options={id(model): {"zipmap": False}},
        )
    path.write_bytes(onnx_model.SerializeToString())


def onnx_probabilities(path: Path, X: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    probs = sess.run(["probabilities"], {"features": X})[0]
    if isinstance(probs, list):  # ZipMap
        probs = np.array([[p[0], p[1]] for p in probs])
    return np.asarray(probs)[:, 1]


def evidence(model_kind: str) -> dict[str, Any]:
    """Leakage-free walk-forward evidence of THIS recipe, copied into the card."""
    if not EVIDENCE_JSON.exists():
        return {"missing": EVIDENCE_JSON.name}
    pooled = json.loads(EVIDENCE_JSON.read_text(encoding="utf-8"))[
        "pooled_walk_forward"
    ]
    m = pooled["summary"]["per_model"][model_kind]
    out: dict[str, Any] = {
        "source": f"models/{EVIDENCE_JSON.name}",
        "protocol": f"pooled purged walk-forward, {pooled['n_folds']} folds, 7 failures",
        "expected_cost_inner": m["expected_cost_inner"],
        "event_recall_inner": m["event_recall_inner"],
        "false_alarms_per_day_inner": m["false_alarms_per_day_inner"],
        "pr_auc": m["pr_auc"],
    }
    key = "xgboost_vs_random_forest"
    out["rn05_gate_xgboost_vs_random_forest"] = pooled["promotion_gate_rn05"][key]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--model", choices=sorted(ARTEFACT), required=True)
    parser.add_argument("--role", choices=["champion", "challenger"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--neg-stride", type=int, default=5)
    args = parser.parse_args()
    name = ARTEFACT[args.model]

    xd.configure(DROPPED)
    X, y, ts, is22 = xd.load_pooled()
    events = events_from_intervals([*FAILURE_INTERVALS, *FAILURE_INTERVALS_2022])
    all_idx = np.arange(len(y))
    log.info(
        "%s: data %s, %d positives, %d failures",
        name,
        X.shape,
        int(y.sum()),
        len(events),
    )

    thr, source, oof, _ = _inner_threshold(
        args.model,
        args.seed,
        X,
        y,
        ts,
        all_idx,
        events,
        args.neg_stride,
        candidates=THRESHOLD_GRID,
    )
    log.info("threshold %.4f (%s)", thr, source)

    tr = _subsample(all_idx, y, args.neg_stride)
    y_tr = y.to_numpy()[tr]
    pos_weight = float((len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1))
    model = MODELS[args.model](args.seed, pos_weight)
    model.fit(X.iloc[tr].to_numpy(np.float32), y_tr)

    onnx_path = MODELS_DIR / f"{name}.onnx"
    export_onnx(args.model, model, X.shape[1], onnx_path)
    sample = X.iloc[np.linspace(0, len(X) - 1, 5000).astype(int)].to_numpy(np.float32)
    diff = float(
        np.abs(
            onnx_probabilities(onnx_path, sample) - model.predict_proba(sample)[:, 1]
        ).max()
    )
    if diff > 1e-4:
        raise SystemExit(f"ONNX parity failed: max |dp| = {diff:.2e}")
    log.info("ONNX parity ok (max |dp| = %.1e)", diff)

    balancing = (
        f"training only: all positives + every {args.neg_stride}th negative, "
        + (
            f"scale_pos_weight={pos_weight:.2f}"
            if args.model == "xgboost"
            else "class_weight='balanced_subsample'"
        )
        + "; no SMOTE"
    )
    card = {
        "schema_version": "2.0",
        "model_name": name,
        "status": args.role,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_type": type(model).__name__,
        "validation": {
            "leakage_free": True,
            "protocol": "purged walk-forward by failure event (RF-28), "
            "pooled MetroPT-3 + MetroPT 2022",
            "threshold_selection": "training out-of-fold only",
        },
        "datasets": {
            "metropt3_2020": {"failures": len(FAILURE_INTERVALS)},
            "metropt_2022": {
                "failures": len(FAILURE_INTERVALS_2022),
                "source": "doi:10.5281/zenodo.6854240",
            },
        },
        "dropped_sensors": DROPPED,
        "class_balancing": balancing,
        "n_train_rows": int(tr.size),
        "n_train_positives": int(y_tr.sum()),
        "decision_threshold": round(thr, 4),
        "threshold_strategy": f"RN-02 cost-optimal on out-of-fold scores ({source})",
        "seed": args.seed,
        "onnx_parity_max_abs_diff": diff,
        "feature_count": int(X.shape[1]),
        "feature_names": list(X.columns),
        # Read by the backend's card-driven loader (model_service).
        "serving": {"adapter": "tree", "onnx": f"{name}.onnx"},
        "evidence": evidence(args.model),
        "role_rationale": (
            "corrective rebuild of the legacy champion (random split + SMOTE, "
            "no valid evidence) with the validated recipe"
            if args.role == "champion"
            else "not promoted: RN-05 gate requires p < 0.05 over the pooled folds; "
            "served in shadow + canary"
        ),
    }
    (MODELS_DIR / f"{name}_card.json").write_text(
        json.dumps(card, indent=2, default=str), encoding="utf-8"
    )
    log.info("wrote %s and %s_card.json", onnx_path.name, name)


if __name__ == "__main__":
    main()
