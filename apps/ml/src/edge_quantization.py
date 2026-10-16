"""
Edge deployment study: INT8 quantization and single-thread latency — RNF-80.

Entry point:
    python -m src.edge_quantization

No Raspberry Pi / Jetson is available in this project, so the study uses a
documented PROXY: ONNX Runtime pinned to ONE intra-op thread on the dev CPU
(an edge SoC runs 4 slower cores; single-thread latency is the bound that
matters for one inference stream). Absolute numbers will differ on real
hardware; the FP32 → INT8 ratios and the accuracy deltas transfer.

Models:
* mlp_v1 (34 features) and tcn_v1 (60 × 12 window) — dynamic INT8
  quantization of MatMul/Gemm weights (``quantize_dynamic``). The TCN's
  Conv1d layers stay FP32: ORT's CPU provider has no ConvInteger kernel.
* random_forest_v2 (production) — tree ensembles in ONNX-ML have no INT8
  kernels; only its FP32 single-thread latency is measured.

Evaluation slice: 2020-05-27 → 2020-06-01 (contains the 29-30 May failure),
so agreement is measured on both normal and failure inputs.

Outputs: models/edge_quantization.json, models/*_int8.onnx (not versioned)
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from onnxruntime.quantization import QuantType, quantize_dynamic
from sklearn.metrics import average_precision_score

_ML_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ML_ROOT))

from src.preprocessing import MetroPTPreprocessor  # noqa: E402
from src.train_random_forest import FEATURE_COLS  # noqa: E402

MODELS_DIR: Path = _ML_ROOT / "models"
PARQUET: Path = _ML_ROOT / "data" / "processed" / "metropt3.parquet"
OUT_JSON: Path = MODELS_DIR / "edge_quantization.json"
SLICE: tuple[str, str] = ("2020-05-27", "2020-06-01")
WINDOW: int = 60
WINDOW_STRIDE: int = 10
LATENCY_RUNS: int = 500


def _session(path: Path, threads: int = 1) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(path), sess_options=opts, providers=["CPUExecutionProvider"]
    )


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _latency_ms(
    sess: ort.InferenceSession, feed: dict[str, np.ndarray]
) -> dict[str, float]:
    for _ in range(20):
        sess.run(None, feed)
    times: list[float] = []
    for _ in range(LATENCY_RUNS):
        t0 = time.perf_counter()
        sess.run(None, feed)
        times.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times)
    return {"p50": float(np.percentile(arr, 50)), "p95": float(np.percentile(arr, 95))}


def _size_mb(path: Path) -> float:
    data = path.with_name(path.name + ".data")
    return (path.stat().st_size + (data.stat().st_size if data.exists() else 0)) / 1e6


def _slice() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df[(df["timestamp"] >= SLICE[0]) & (df["timestamp"] < SLICE[1])].reset_index(
        drop=True
    )


def _compare(
    name: str, fp32: Path, feed_all: np.ndarray, y: np.ndarray, input_name: str
) -> dict[str, Any]:
    int8 = MODELS_DIR / f"{name}_int8.onnx"
    # The Lightning export keeps stale intermediate shapes in value_info that
    # contradict ONNX shape inference (e.g. 34 vs 256): drop them first.
    model = onnx.load(str(fp32), load_external_data=True)
    del model.graph.value_info[:]
    clean = MODELS_DIR / f"{name}_clean_tmp.onnx"
    onnx.save(model, str(clean))
    try:
        # Only MatMul/Gemm: the CPU execution provider has no ConvInteger
        # kernel, so the TCN's Conv1d layers stay FP32 (partial quantization).
        quantize_dynamic(
            str(clean),
            str(int8),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Gemm"],
        )
    finally:
        clean.unlink(missing_ok=True)

    out: dict[str, Any] = {"input": list(feed_all.shape[1:])}
    probs: dict[str, np.ndarray] = {}
    for tag, path in (("fp32", fp32), ("int8", int8)):
        sess = _session(path, threads=1)
        logits = np.concatenate(
            [
                sess.run(None, {input_name: feed_all[i : i + 512]})[0]
                for i in range(0, len(feed_all), 512)
            ]
        )
        probs[tag] = _softmax(logits)[:, 1]
        out[tag] = {
            "size_mb": _size_mb(path),
            "latency_single_ms_1thread": _latency_ms(sess, {input_name: feed_all[:1]}),
            "latency_batch256_ms_1thread": _latency_ms(
                sess, {input_name: feed_all[:256]}
            ),
            "pr_auc": float(average_precision_score(y, probs[tag])),
        }
    out["agreement"] = {
        "decision_agreement_at_0.5": float(
            np.mean((probs["fp32"] >= 0.5) == (probs["int8"] >= 0.5))
        ),
        "mean_abs_prob_diff": float(np.mean(np.abs(probs["fp32"] - probs["int8"]))),
        "max_abs_prob_diff": float(np.max(np.abs(probs["fp32"] - probs["int8"]))),
    }
    out["speedup_single"] = (
        out["fp32"]["latency_single_ms_1thread"]["p50"]
        / out["int8"]["latency_single_ms_1thread"]["p50"]
    )
    out["size_ratio"] = out["int8"]["size_mb"] / out["fp32"]["size_mb"]
    print(
        name,
        json.dumps({k: out[k] for k in ("agreement", "speedup_single", "size_ratio")}),
        flush=True,
    )
    return out


def main() -> None:
    df = _slice()
    y_rows = df["anomaly"].astype(int).to_numpy()
    result: dict[str, Any] = {
        "schema_version": "1.0",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "requirements": ["RNF-80"],
        "proxy": "ONNX Runtime, 1 intra-op thread, dev CPU — not a Raspberry Pi/Jetson measurement",
        "onnxruntime": ort.__version__,
        "slice": list(SLICE),
        "n_rows": int(len(df)),
        "n_positive_rows": int(y_rows.sum()),
        "models": {},
    }

    # MLP — 34 V1 features, per-feature StandardScaler.
    feats_v1 = MetroPTPreprocessor(enable_v2_features=False).fit_transform(
        df[FEATURE_COLS].copy()
    )
    card = json.loads((MODELS_DIR / "mlp_v1_card.json").read_text(encoding="utf-8"))
    mlp_scaler = joblib.load(MODELS_DIR / "mlp_scaler.joblib")
    x_mlp = mlp_scaler.transform(feats_v1[card["feature_names"]].to_numpy()).astype(
        np.float32
    )
    result["models"]["mlp_v1"] = _compare(
        "mlp_v1", MODELS_DIR / "mlp_v1.onnx", x_mlp, y_rows, "features"
    )

    # TCN — windows of 60 × 12 raw channels, per-channel StandardScaler.
    tcn_card = json.loads((MODELS_DIR / "tcn_v1_card.json").read_text(encoding="utf-8"))
    raw = df[tcn_card["feature_names"]].to_numpy(dtype=np.float32)
    tcn_scaler = joblib.load(MODELS_DIR / "tcn_scaler.joblib")
    scaled = tcn_scaler.transform(raw).astype(np.float32)
    starts = np.arange(0, len(scaled) - WINDOW, WINDOW_STRIDE)
    windows = np.stack([scaled[s : s + WINDOW] for s in starts]).astype(np.float32)
    y_win = y_rows[starts + WINDOW - 1]
    result["models"]["tcn_v1"] = _compare(
        "tcn_v1", MODELS_DIR / "tcn_v1.onnx", windows, y_win, "window"
    )

    # Production RF — FP32 only (no INT8 kernels for TreeEnsemble).
    feats_v2 = (
        MetroPTPreprocessor()
        .fit_transform(df[FEATURE_COLS].copy())
        .to_numpy(dtype=np.float32)
    )
    rf = _session(MODELS_DIR / "random_forest_v2.onnx", threads=1)
    result["models"]["random_forest_v2"] = {
        "size_mb": _size_mb(MODELS_DIR / "random_forest_v2.onnx"),
        "latency_single_ms_1thread": _latency_ms(rf, {"features": feats_v2[:1]}),
        "int8": "not applicable — ONNX-ML TreeEnsemble has no quantized kernels",
    }
    print(
        "random_forest_v2", json.dumps(result["models"]["random_forest_v2"]), flush=True
    )

    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
