"""
Leakage-free production artefacts for the deep models.

Entry point:
    python -m src.train_deep_leakfree [--archs mlp tcn ...] [--seed 0]

Produces, per architecture (served by the backend's card-driven loader):
    models/<arch>_v2.onnx, <arch>_v2_scaler.joblib, <arch>_v2_card.json

Recipe = ``deep_temporal`` exactly as evaluated by ``evaluate_all_models``:
pooled MetroPT-3 + MetroPT 2022, harmonised sensors, row-level labels,
training-only scaling and class balance, fixed epochs, threshold from the
temporal validation on the most recent failure (the 2022 oil leak). The
legacy v1 artefacts (stratified random split, threshold tuned on the test
split) are left on disk but are no longer the recommended models.
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
import torch
from sklearn.preprocessing import StandardScaler

import src.deep_temporal as dt
import src.evaluate_cross_dataset as xd
from src.ingest_metropt import FAILURE_INTERVALS
from src.ingest_metropt2022 import FAILURE_INTERVALS_2022
from src.temporal_validation import events_from_intervals

_ML_ROOT: Path = Path(__file__).resolve().parent.parent
MODELS_DIR: Path = _ML_ROOT / "models"
DROPPED: list[str] = ["Reservoirs", "Oil_level"]
EVIDENCE_JSON: Path = MODELS_DIR / "eval_all_models.json"
TOLERANCE: dict[str, float] = {
    "mlp": 1e-4,
    "tcn": 1e-4,
    "bilstm": 1e-4,
    "patchtst": 5e-4,
    "autoencoder": 1e-4,
}
ADAPTER: dict[str, str] = {
    "mlp": "mlp",
    "tcn": "sequence",
    "bilstm": "sequence",
    "patchtst": "sequence",
    "autoencoder": "autoencoder",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("train_deep_leakfree")


def _scaler(mean: np.ndarray, std: np.ndarray, n: int) -> StandardScaler:
    sc = StandardScaler()
    sc.mean_ = mean.astype(np.float64)
    sc.scale_ = std.astype(np.float64)
    sc.var_ = sc.scale_**2
    sc.n_features_in_ = int(mean.shape[0])
    sc.n_samples_seen_ = n
    return sc


def _export(arch: str, fitted: dt.Fitted, sample: torch.Tensor, path: Path) -> None:
    model = fitted.model.to("cpu").eval()
    torch.onnx.export(
        model,
        sample,
        str(path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )


def _parity(arch: str, fitted: dt.Fitted, sample: torch.Tensor, path: Path) -> float:
    with torch.no_grad():
        ref = fitted.model(sample).numpy()
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"input": sample.numpy()})[0]
    return float(np.abs(ref - got).max())


def _evidence(arch: str) -> dict[str, Any]:
    if not EVIDENCE_JSON.exists():
        return {"missing": EVIDENCE_JSON.name}
    m = json.loads(EVIDENCE_JSON.read_text(encoding="utf-8"))["models"].get(arch)
    if m is None:
        return {"missing": f"{arch} in {EVIDENCE_JSON.name}"}
    total = m["total"]
    return {
        "source": f"models/{EVIDENCE_JSON.name}",
        "protocol": "pooled purged walk-forward, 6 folds, out-of-time",
        "events": total["events"],
        "confusion_alarms": total["alarms"],
        "metrics_alarms": total["metrics_alarms"],
        "expected_cost_mean": total["expected_cost_mean"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--archs", nargs="*", default=list(dt.ARCHS), choices=dt.ARCHS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--evidence-only",
        action="store_true",
        help="only refresh the walk-forward evidence in existing cards",
    )
    args = parser.parse_args()
    if args.evidence_only:
        for arch in args.archs:
            path = MODELS_DIR / f"{arch}_v2_card.json"
            card = json.loads(path.read_text(encoding="utf-8"))
            card["evidence"] = _evidence(arch)
            path.write_text(json.dumps(card, indent=2, default=str), encoding="utf-8")
            log.info("evidence refreshed in %s", path.name)
        return

    xd.configure(DROPPED)
    X, y, ts, is22 = xd.load_pooled()
    raw_df = pd.concat(
        [xd._read(xd.P2020, []), xd._read(xd.P2022, [])], ignore_index=True
    )[["timestamp", *xd.FEATURES, "anomaly"]]
    data = dt.build_data(raw_df, X, y, ts, is22)
    events = events_from_intervals([*FAILURE_INTERVALS, *FAILURE_INTERVALS_2022])
    all_idx = np.arange(len(y))

    for arch in args.archs:
        name = f"{arch}_v2"
        thr, source = dt.temporal_threshold(data, arch, all_idx, events, args.seed)
        fitted = dt.fit(data, arch, all_idx, args.seed)
        log.info("%s threshold %.6g (%s)", name, thr, source)

        rows = all_idx[data.valid_end][:: max(len(all_idx) // 64, 1)][:64]
        sample = dt._batch_inputs(data, arch, rows, fitted.mean, fitted.std).cpu()
        onnx_path = MODELS_DIR / f"{name}.onnx"
        _export(arch, fitted, sample, onnx_path)
        diff = _parity(arch, fitted, sample, onnx_path)
        if diff > TOLERANCE[arch]:
            raise SystemExit(f"{name}: ONNX parity failed (max |d| = {diff:.2e})")
        scaler_name = f"{name}_scaler.joblib"
        joblib.dump(
            _scaler(fitted.mean, fitted.std, len(all_idx)), MODELS_DIR / scaler_name
        )

        serving: dict[str, Any] = {
            "adapter": ADAPTER[arch],
            "onnx": onnx_path.name,
            "scaler": scaler_name,
        }
        if arch != "mlp":
            serving["window_size"] = dt.WINDOW
        if arch == "autoencoder":
            serving["mse_threshold"] = thr
        card = {
            "schema_version": "2.0",
            "model_name": name,
            "status": "candidate",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "architecture": arch,
            "validation": {
                "leakage_free": True,
                "protocol": "purged walk-forward by failure event (RF-28), "
                "pooled MetroPT-3 + MetroPT 2022",
                "threshold_selection": source,
            },
            "dropped_sensors": DROPPED,
            "training": dt.describe(),
            "seed": args.seed,
            # Autoencoder: the adapter maps MSE → 0.5 at mse_threshold.
            "decision_threshold": 0.5 if arch == "autoencoder" else round(thr, 4),
            "onnx_parity_max_abs_diff": diff,
            "feature_count": len(
                data.feature_names if arch == "mlp" else data.channels
            ),
            "feature_names": data.feature_names if arch == "mlp" else data.channels,
            "serving": serving,
            "evidence": _evidence(arch),
        }
        (MODELS_DIR / f"{name}_card.json").write_text(
            json.dumps(card, indent=2, default=str), encoding="utf-8"
        )
        log.info("wrote %s (+ scaler, card); parity %.1e", onnx_path.name, diff)


if __name__ == "__main__":
    main()
