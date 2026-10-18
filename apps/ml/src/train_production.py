"""
Production artefacts for every servable model — same nested protocol as the
evaluation (``nested_protocol``), applied to ALL data.

Entry point:
    python -m src.train_production [--models random_forest tcn ...]

Per model, on MetroPT-3 + MetroPT 2022 (harmonised sensors):
1. hyper-parameters chosen by leave-one-failure-out OOF over the 7 failures
   (RN-02 expected cost), never on test data;
2. Platt calibration fitted on those OOF scores;
3. RN-02 threshold on the calibrated OOF probabilities;
4. refit on all data, ONNX export + parity check, card with ``serving``.

How each model serves the calibrated decision (no backend change needed):
* neural classifiers — the Platt map is baked into the ONNX graph
  (``deep_temporal.CalibratedClassifier``): the adapter returns the calibrated
  probability and ``decision_threshold`` is on that scale;
* trees — the ONNX returns the raw probability; ``decision_threshold`` is the
  raw-probability equivalent of the calibrated threshold (same alarms);
* autoencoder — ``mse_threshold`` is the MSE equivalent (adapter: 0.5 there).

Artefact names: random_forest_v3 (champion), xgboost_v3 (challenger),
mlp_v2 / tcn_v2 / bilstm_v2 / patchtst_v2 / autoencoder_v2 (candidates).
Evidence (out-of-time, 3 seeds) is copied from models/nested/<model>.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import onnxruntime as ort
import torch
from sklearn.preprocessing import StandardScaler

import src.deep_temporal as dt
import src.nested_protocol as npc
from src.evaluate_nested import DROPPED, build_context

_ML_ROOT: Path = Path(__file__).resolve().parent.parent
MODELS_DIR: Path = _ML_ROOT / "models"
NESTED_DIR: Path = MODELS_DIR / "nested"
ARTEFACT: dict[str, str] = {
    "random_forest": "random_forest_v3",
    "xgboost": "xgboost_v3",
    **{a: f"{a}_v2" for a in dt.ARCHS},
}
ROLE: dict[str, str] = {"random_forest": "champion", "xgboost": "challenger"}
ADAPTER: dict[str, str] = {
    "random_forest": "tree",
    "xgboost": "tree",
    "mlp": "mlp",
    "tcn": "sequence",
    "bilstm": "sequence",
    "patchtst": "sequence",
    "autoencoder": "autoencoder",
}
TOLERANCE: float = 5e-4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("train_production")


def _export_tree(name: str, model: Any, n_features: int, path: Path) -> None:
    if name == "xgboost":
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


def _tree_parity(path: Path, model: Any, X: np.ndarray) -> float:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    probs = sess.run(["probabilities"], {"features": X})[0]
    if isinstance(probs, list):
        probs = np.array([[p[0], p[1]] for p in probs])
    return float(np.abs(np.asarray(probs)[:, 1] - model.predict_proba(X)[:, 1]).max())


def _export_deep(module: torch.nn.Module, sample: torch.Tensor, path: Path) -> float:
    module = module.to("cpu").eval()
    sample = sample.cpu()
    torch.onnx.export(
        module,
        sample,
        str(path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    with torch.no_grad():
        ref = module(sample).numpy()
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"input": sample.numpy()})[0]
    return float(np.abs(ref - got).max())


def _scaler(mean: np.ndarray, std: np.ndarray, n: int) -> StandardScaler:
    sc = StandardScaler()
    sc.mean_ = mean.astype(np.float64)
    sc.scale_ = std.astype(np.float64)
    sc.var_ = sc.scale_**2
    sc.n_features_in_ = int(mean.shape[0])
    sc.n_samples_seen_ = n
    return sc


def _evidence(name: str) -> dict[str, Any]:
    path = NESTED_DIR / f"{name}.json"
    if not path.exists():
        return {"missing": f"models/nested/{name}.json"}
    s = json.loads(path.read_text(encoding="utf-8"))["summary"]
    return {
        "source": f"models/nested/{name}.json",
        "protocol": "outer purged walk-forward (6 folds) × inner leave-one-failure-out, "
        "3 seeds, out-of-time",
        "failures_detected_mean_of_6": s["detected_mean"],
        "failures_detected_range": [s["detected_min"], s["detected_max"]],
        "false_alarm_episodes_mean": s["false_alarm_episodes_mean"],
        "expected_cost_mean": s["expected_cost_mean"],
        "confusion_alarms_seed0": s["per_seed"][0]["alarms"],
    }


def train(ctx: npc.Context, name: str, seed: int) -> None:
    artefact = ARTEFACT[name]
    all_idx = np.arange(len(ctx.y))
    sel = npc.select(ctx, name, all_idx, seed, tune=True)
    model = npc.fit_model(ctx, name, all_idx, seed, sel.params)
    log.info(
        "%s: params=%s thr_cal=%.4g (%s)",
        artefact,
        sel.params,
        sel.threshold_cal,
        sel.source,
    )
    onnx_path = MODELS_DIR / f"{artefact}.onnx"
    serving: dict[str, Any] = {"adapter": ADAPTER[name], "onnx": onnx_path.name}
    data = ctx.data

    if name in ("random_forest", "xgboost"):
        X = ctx.X.to_numpy(np.float32)
        _export_tree(name, model, X.shape[1], onnx_path)
        sample = X[np.linspace(0, len(X) - 1, 5000).astype(int)]
        parity = _tree_parity(onnx_path, model, sample)
        decision_threshold = npc.native_threshold(name, sel.raw_threshold)
        feature_names = list(ctx.X.columns)
    else:
        rows = all_idx[data.valid_end][:: max(len(all_idx) // 64, 1)][:64]
        sample_t = dt.batch_inputs(data, name, rows, model.mean, model.std)
        if name == "autoencoder":
            module: torch.nn.Module = model.model
            serving["mse_threshold"] = npc.native_threshold(name, sel.raw_threshold)
            decision_threshold = 0.5
        else:
            cal = sel.calibrator or npc.Platt()
            module = dt.CalibratedClassifier(model.model, cal.a, cal.b)
            decision_threshold = sel.threshold_cal
        parity = _export_deep(module, sample_t, onnx_path)
        scaler_name = f"{artefact}_scaler.joblib"
        joblib.dump(
            _scaler(model.mean, model.std, len(all_idx)), MODELS_DIR / scaler_name
        )
        serving["scaler"] = scaler_name
        if name != "mlp":
            serving["window_size"] = dt.WINDOW
        feature_names = data.feature_names if name == "mlp" else data.channels
    if parity > TOLERANCE:
        raise SystemExit(f"{artefact}: ONNX parity failed (max |d| = {parity:.2e})")

    card = {
        "schema_version": "3.0",
        "model_name": artefact,
        "status": ROLE.get(name, "candidate"),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": name,
        "validation": {
            "leakage_free": True,
            "protocol": "nested: outer purged walk-forward × inner leave-one-failure-out",
            "hyperparameters": "selected on inner OOF expected cost (RN-02)",
            "threshold_selection": sel.source,
        },
        "dropped_sensors": DROPPED,
        "hyperparameters": sel.params,
        "search": sel.search,
        "calibration": (
            {"method": "platt", **asdict(sel.calibrator)}
            if sel.calibrator is not None
            else None
        ),
        "threshold_calibrated_probability": sel.threshold_cal,
        "decision_threshold": round(float(decision_threshold), 6),
        "seed": seed,
        "onnx_parity_max_abs_diff": parity,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "serving": serving,
        "evidence": _evidence(name),
    }
    (MODELS_DIR / f"{artefact}_card.json").write_text(
        json.dumps(card, indent=2, default=str), encoding="utf-8"
    )
    log.info("wrote %s (+card); parity %.1e", onnx_path.name, parity)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--models", nargs="+", default=list(ARTEFACT), choices=list(ARTEFACT)
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--evidence-only",
        action="store_true",
        help="refresh the out-of-time evidence in existing cards",
    )
    args = parser.parse_args()
    if args.evidence_only:
        for name in args.models:
            path = MODELS_DIR / f"{ARTEFACT[name]}_card.json"
            card = json.loads(path.read_text(encoding="utf-8"))
            card["evidence"] = _evidence(name)
            path.write_text(json.dumps(card, indent=2, default=str), encoding="utf-8")
        return
    ctx, _ = build_context()
    for name in args.models:
        train(ctx, name, args.seed)


if __name__ == "__main__":
    main()
