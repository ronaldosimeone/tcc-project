"""
Energy and carbon footprint of the PredictIQ pipeline — RNF-83.

Entry point:
    python -m src.evaluate_energy [--reps 3] [--llm-reps 5] [--skip-llm]

What is measured (CodeCarbon, offline mode, Brazilian grid mix)
--------------------------------------------------------------
* ``idle``      — machine at rest; its power is subtracted from every other
                  phase so the numbers are the *marginal* cost of the work,
                  not of keeping the workstation on.
* ``training``  — fitting RF / XGBoost / LogReg on the walk-forward training
                  budget (every positive + every 5th negative of MetroPT-3).
* ``inference`` — ONNX Runtime, batch 1 (the online path of the API),
                  RF v2, XGBoost v2, MLP fp32 and MLP INT8 (RNF-80), one
                  session per core so the load is measurable.
* ``llm``       — one maintenance-plan generation by the local LLM through
                  Ollama (fixed prompt, temperature 0, fixed output budget).

Measurement method, stated so the thesis can quote it
-----------------------------------------------------
* CPU: Windows Energy Meter Interface (real counter, not a TDP estimate).
* GPU: NVML (real counter). Ollama runs on the GPU.
* RAM: CodeCarbon's constant model (W per GB) — an ESTIMATE, reported apart.
* Tracking mode "machine": every process is counted, which is why the idle
  baseline is subtracted. Ollama is a separate process, so "process" mode
  would miss the LLM entirely.
* No network: OfflineEmissionsTracker never calls an external API.

Output: models/eval_energy.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import pandas as pd
import requests  # type: ignore[import-untyped]
from codecarbon import OfflineEmissionsTracker

from src.evaluate_temporal import MODELS, PARQUET_PATH, _subsample
from src.preprocessing import MetroPTPreprocessor
from src.train_random_forest import FEATURE_COLS, TARGET_COL

_ML_ROOT: Path = Path(__file__).resolve().parent.parent
MODELS_DIR: Path = _ML_ROOT / "models"
OUT_JSON: Path = MODELS_DIR / "eval_energy.json"

COUNTRY: str = "BRA"
IDLE_SECONDS: float = 30.0
# Inference phases loop until at least this long, so the 1 s power sampling
# integrates enough samples for a stable energy figure.
MIN_PHASE_SECONDS: float = 20.0
NEG_STRIDE: int = 5
# One MetroPT asset at 0.1 Hz for a year.
PREDICTIONS_PER_ASSET_YEAR: int = int(365 * 24 * 3600 / 10)

OLLAMA_URL: str = "http://localhost:11434"
LLM_MODEL: str = "llama3.2:3b"
LLM_NUM_PREDICT: int = 400

# Representative RAG prompt (shape and length of a real one: system rules, an
# alert with XAI contributions and three manual excerpts). Synthetic text, so
# no third-party manual content is stored in the repository.
LLM_SYSTEM: str = (
    "Você é um assistente técnico de manutenção industrial. Responda em "
    "Markdown com: diagnóstico provável, procedimento numerado, peças e "
    "ferramentas, e referências citando arquivo e página. Use apenas os "
    "trechos fornecidos; se não houver base, diga que não há."
)
LLM_USER: str = (
    "Alerta: probabilidade de falha 0,91 no compressor da APU (Trem 042).\n"
    "Sensores que mais contribuíram: DV_pressure (+0,31), Oil_temperature "
    "(+0,22), TP2 (+0,15), Motor_current (+0,09).\n\n"
    + "\n\n".join(
        f"[Trecho {i} — manual_apu.pdf, p. {40 + i}, score 0,7{i}]\n"
        "Verifique a pressão de descarga e o funcionamento da válvula de "
        "dreno do secador de ar. Uma queda lenta de pressão com o compressor "
        "em carga indica vazamento na linha de alimentação dos consumidores. "
        "Inspecione conexões, mangueiras e o pressostato; meça a temperatura "
        "do óleo após 10 minutos de operação contínua e compare com o limite "
        "de 85 °C. Substitua o elemento separador se houver arraste de óleo."
        for i in range(1, 4)
    )
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log: logging.Logger = logging.getLogger("evaluate_energy")
for _noisy in ("codecarbon", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)


@dataclass(frozen=True)
class Reading:
    seconds: float
    cpu_kwh: float
    gpu_kwh: float
    ram_kwh: float
    co2_kg: float

    @property
    def measured_kwh(self) -> float:
        """CPU + GPU: the two components with a real hardware counter."""
        return self.cpu_kwh + self.gpu_kwh


def measure(work: Callable[[], Any]) -> tuple[Reading, Any]:
    tracker = OfflineEmissionsTracker(
        country_iso_code=COUNTRY,
        measure_power_secs=1,
        tracking_mode="machine",
        save_to_file=False,
        log_level="error",
    )
    tracker.start()
    result = work()
    tracker.stop()
    d = tracker.final_emissions_data
    reading = Reading(
        seconds=float(d.duration),
        cpu_kwh=float(d.cpu_energy),
        gpu_kwh=float(d.gpu_energy),
        ram_kwh=float(d.ram_energy),
        co2_kg=float(d.emissions),
    )
    return reading, result


@dataclass(frozen=True)
class Baseline:
    cpu_w: float
    gpu_w: float
    kg_per_kwh: float

    def net_kwh(self, r: Reading) -> float:
        idle = (self.cpu_w + self.gpu_w) * r.seconds / 3.6e6
        return max(r.measured_kwh - idle, 0.0)


def idle_baseline() -> Baseline:
    reading, _ = measure(lambda: time.sleep(IDLE_SECONDS))
    total = reading.cpu_kwh + reading.gpu_kwh + reading.ram_kwh
    return Baseline(
        cpu_w=reading.cpu_kwh * 3.6e6 / reading.seconds,
        gpu_w=reading.gpu_kwh * 3.6e6 / reading.seconds,
        kg_per_kwh=reading.co2_kg / total if total > 0 else float("nan"),
    )


def _summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n": int(arr.size),
    }


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------


def training_phase(base: Baseline, reps: int) -> dict[str, Any]:
    df = pd.read_parquet(PARQUET_PATH).sort_values("timestamp").reset_index(drop=True)
    X = MetroPTPreprocessor().fit_transform(df[FEATURE_COLS].copy()).astype(np.float32)
    y = df[TARGET_COL].astype(int)
    idx = _subsample(np.arange(len(y)), y, NEG_STRIDE)
    X_tr, y_tr = X.iloc[idx], y.to_numpy()[idx]
    pos_weight = float((len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1))
    out: dict[str, Any] = {"n_rows": int(idx.size), "n_features": int(X.shape[1])}
    for name, factory in MODELS.items():
        net_wh: list[float] = []
        secs: list[float] = []
        for rep in range(reps):
            model = factory(rep, pos_weight)
            reading, _ = measure(partial(model.fit, X_tr, y_tr))
            net_wh.append(base.net_kwh(reading) * 1000)
            secs.append(reading.seconds)
            log.info(
                "train %s rep %d: %.2f Wh net, %.1f s", name, rep, net_wh[-1], secs[-1]
            )
        out[name] = {
            "net_wh": _summary(net_wh),
            "seconds": _summary(secs),
            "gco2": _summary([w * base.kg_per_kwh for w in net_wh]),
        }
    return out


def _session(path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(path), sess_options=opts, providers=["CPUExecutionProvider"]
    )


def _inference_inputs() -> dict[str, np.ndarray]:
    df = pd.read_parquet(PARQUET_PATH).sort_values("timestamp").iloc[:20_000]
    v2 = MetroPTPreprocessor().fit_transform(df[FEATURE_COLS].copy())
    v1 = MetroPTPreprocessor(enable_v2_features=False).fit_transform(
        df[FEATURE_COLS].copy()
    )
    return {
        "random_forest_v2": v2.to_numpy(np.float32),
        "xgboost_v2": v1.to_numpy(np.float32),
        "mlp_v1": v1.to_numpy(np.float32),
        "mlp_v1_int8": v1.to_numpy(np.float32),
    }


def inference_phase(base: Baseline, reps: int) -> dict[str, Any]:
    """Batch-1 inference with one ONNX session per core, all cores busy.

    A single thread draws ~1 W on a 16-thread CPU idling at ~85 W, which is
    below the resolution of machine-level metering (first run: 0.00-0.06
    mWh/1k predictions, an 8x spread). Saturating every core lifts the signal
    well above the idle noise; the figure is the per-prediction energy of a
    fully loaded inference server, i.e. a lower bound for a lightly loaded one.
    """
    workers = os.cpu_count() or 1
    inputs = _inference_inputs()
    out: dict[str, Any] = {
        "batch_size": 1,
        "workers": workers,
        "threads_per_session": 1,
    }
    for name, rows in inputs.items():
        sessions = [
            _session(MODELS_DIR / "legacy" / f"{name}.onnx") for _ in range(workers)
        ]
        feed_name = sessions[0].get_inputs()[0].name
        for sess in sessions:  # warm-up outside the measurement
            for i in range(20):
                sess.run(None, {feed_name: rows[i : i + 1]})

        def worker(sess: ort.InferenceSession, offset: int) -> int:
            n, t0 = 0, time.perf_counter()
            while time.perf_counter() - t0 < MIN_PHASE_SECONDS:
                i = (offset + n) % len(rows)
                sess.run(None, {feed_name: rows[i : i + 1]})
                n += 1
            return n

        def run_all() -> int:
            # ONNX Runtime releases the GIL inside run(), so threads scale.
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(worker, sess, k * 997)
                    for k, sess in enumerate(sessions)
                ]
                return sum(f.result() for f in futures)

        mwh_per_k: list[float] = []
        for rep in range(reps):
            reading, n_pred = measure(run_all)
            mwh_per_k.append(base.net_kwh(reading) * 1e6 / n_pred * 1000)
            log.info(
                "infer %s rep %d: %.4f mWh / 1k preds (%d preds)",
                name,
                rep,
                mwh_per_k[-1],
                n_pred,
            )
        mean_wh_year = float(np.mean(mwh_per_k)) / 1e6 * PREDICTIONS_PER_ASSET_YEAR
        out[name] = {
            "net_mwh_per_1k_predictions": _summary(mwh_per_k),
            "asset_year_wh": mean_wh_year,
            "asset_year_gco2": mean_wh_year * base.kg_per_kwh,
        }
    return out


def _ollama_generate(client: requests.Session) -> dict[str, Any]:
    resp = client.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": LLM_MODEL,
            "system": LLM_SYSTEM,
            "prompt": LLM_USER,
            "stream": False,
            "options": {"temperature": 0, "seed": 0, "num_predict": LLM_NUM_PREDICT},
        },
        timeout=300,
    )
    resp.raise_for_status()
    body: dict[str, Any] = resp.json()
    return body


def llm_phase(base: Baseline, reps: int) -> dict[str, Any]:
    with requests.Session() as client:
        _ollama_generate(client)  # load weights into VRAM outside the measurement
        wh: list[float] = []
        wh_per_1k_tok: list[float] = []
        secs: list[float] = []
        tokens: list[int] = []
        for rep in range(reps):
            reading, body = measure(lambda: _ollama_generate(client))
            n_tok = int(body.get("eval_count", 0)) + int(
                body.get("prompt_eval_count", 0)
            )
            net = base.net_kwh(reading) * 1000
            wh.append(net)
            secs.append(reading.seconds)
            tokens.append(n_tok)
            wh_per_1k_tok.append(net / max(n_tok, 1) * 1000)
            log.info(
                "llm rep %d: %.3f Wh net, %.1f s, %d tokens",
                rep,
                net,
                reading.seconds,
                n_tok,
            )
    return {
        "model": LLM_MODEL,
        "num_predict": LLM_NUM_PREDICT,
        "net_wh_per_suggestion": _summary(wh),
        "net_wh_per_1k_tokens": _summary(wh_per_1k_tok),
        "seconds": _summary(secs),
        "tokens_per_call": _summary([float(t) for t in tokens]),
        "gco2_per_suggestion": float(np.mean(wh)) * base.kg_per_kwh,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--llm-reps", type=int, default=5)
    parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args()

    log.info("idle baseline (%.0f s)…", IDLE_SECONDS)
    base = idle_baseline()
    log.info(
        "idle: CPU %.1f W, GPU %.1f W, %.0f gCO2/kWh",
        base.cpu_w,
        base.gpu_w,
        base.kg_per_kwh * 1000,
    )

    result: dict[str, Any] = {
        "schema_version": "1.0",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "requirements": ["RNF-83"],
        "method": {
            "tool": "codecarbon (OfflineEmissionsTracker, tracking_mode=machine)",
            "country": COUNTRY,
            "cpu": "Windows Energy Meter Interface (measured)",
            "gpu": "NVML (measured)",
            "ram": "CodeCarbon constant model — estimate, excluded from net figures",
            "net": "measured CPU+GPU energy minus idle power × duration",
        },
        "idle_baseline": asdict(base),
        "training": training_phase(base, args.reps),
        "inference": inference_phase(base, args.reps),
    }
    if not args.skip_llm:
        try:
            result["llm"] = llm_phase(base, args.llm_reps)
        except requests.RequestException as exc:
            result["llm"] = {"skipped": f"Ollama unavailable: {exc}"}
    OUT_JSON.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("saved %s", OUT_JSON)


if __name__ == "__main__":
    main()
