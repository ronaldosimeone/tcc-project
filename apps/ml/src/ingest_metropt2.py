"""
MetroPT2 ingestion — the held-out test set for failures never seen in training.

Source: Zenodo doi:10.5281/zenodo.7766691 (version V2 of the MetroPT record),
file ``MetroPT2.csv`` (CC BY 4.0, md5 056e04fd1874eaef4697d6b6657502c3),
28/04/2022 → 28/07/2022, 1 Hz, Porto metro APU compressor.

Ground truth: the record is unlabeled; two failures are known from
maintenance reports (Table 1 of arXiv:2502.07394, Interpretable Rules for
Online Failure Prediction: A Case Study on the Metro do Porto dataset):

    air leak  2022-06-04 10:19:24 → 14:22:39   (train LPS alarm 11:26:01)
    oil leak  2022-07-11 10:10:18 → 2022-07-14 10:22:08 (LPS 2022-07-13 19:43:52)

Both start AFTER the last row used anywhere in training (MetroPT 2022 ends on
2022-06-02 15:49:50), so every production model is scored on them without any
contact with the test period. The LPS times are the train's own low-pressure
alarm: they let the thesis ask whether the system warns BEFORE the train does.

Same harmonisation as ``ingest_metropt2022``: MetroPT-3 sensor names, GPS
dropped, ``Flowmeter`` kept only for ablations, point-sampled to 10 s, gaps
dropped. The overlap 28/04 → 02/06 with MetroPT 2022 is used only to check
whether both files come from the same APU (they do not share sensor
semantics: see ``same_apu_check`` in the metadata) and is excluded from any
test (``overlap_with_training`` flag).

Entry point:
    python -m src.ingest_metropt2
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ingest_metropt import EXPECTED_COLUMNS
from src.ingest_metropt2022 import OUT_PARQUET as P2022
from src.ingest_metropt2022 import label, md5sum, point_sample

_ML_ROOT: Path = Path(__file__).resolve().parent.parent
RAW_CSV: Path = _ML_ROOT / "data" / "raw" / "metropt2" / "MetroPT2.csv"
OUT_PARQUET: Path = _ML_ROOT / "data" / "processed" / "metropt2.parquet"
OUT_META: Path = _ML_ROOT / "models" / "metropt2_ingest.json"

SOURCE_MD5: str = "056e04fd1874eaef4697d6b6657502c3"
EXTRA_COLUMNS: tuple[str, ...] = ("Flowmeter", "LPS")
TRAINING_END: pd.Timestamp = pd.Timestamp("2022-06-02 15:49:50")

FAILURE_INTERVALS_MT2: list[tuple[str, str]] = [
    ("2022-06-04 10:19:24", "2022-06-04 14:22:39"),
    ("2022-07-11 10:10:18", "2022-07-14 10:22:08"),
]
FAILURE_TYPES_MT2: list[str] = ["air_leak", "oil_leak"]
LPS_ALARMS_MT2: list[str] = ["2022-06-04 11:26:01", "2022-07-13 19:43:52"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log: logging.Logger = logging.getLogger("ingest_metropt2")


def same_apu_check(mt2: pd.DataFrame) -> dict[str, Any]:
    """Compare MetroPT2 with MetroPT 2022 on their common period (10 s bins)."""
    if not P2022.exists():
        return {"skipped": "metropt2022.parquet missing"}
    other = pd.read_parquet(P2022)
    lo = max(mt2["timestamp"].min(), other["timestamp"].min())
    hi = min(mt2["timestamp"].max(), other["timestamp"].max())
    a = mt2[(mt2["timestamp"] >= lo) & (mt2["timestamp"] <= hi)].set_index("timestamp")
    b = other[(other["timestamp"] >= lo) & (other["timestamp"] <= hi)].set_index(
        "timestamp"
    )
    joined = a[["TP3", "Reservoirs", "Oil_level"]].join(
        b[["TP3", "Reservoirs", "Oil_level"]],
        lsuffix="_mt2",
        rsuffix="_2022",
        how="inner",
    )
    out: dict[str, Any] = {
        "common_period": [str(lo), str(hi)],
        "aligned_rows": len(joined),
    }
    if len(joined) > 100:
        out["tp3_correlation"] = float(joined["TP3_mt2"].corr(joined["TP3_2022"]))
        out["tp3_mean_abs_diff_bar"] = float(
            (joined["TP3_mt2"] - joined["TP3_2022"]).abs().mean()
        )
    out["reservoirs_mean"] = {
        "metropt2": float(a["Reservoirs"].mean()),
        "metropt_2022": float(b["Reservoirs"].mean()),
    }
    out["oil_level_mean"] = {
        "metropt2": float(a["Oil_level"].mean()),
        "metropt_2022": float(b["Oil_level"].mean()),
    }
    out["reservoirs_tp3_correlation_metropt2"] = float(
        mt2["Reservoirs"].corr(mt2["TP3"])
    )
    return out


def main() -> None:
    if not RAW_CSV.exists():
        raise SystemExit(f"{RAW_CSV} não encontrado — baixe do Zenodo (ver docstring).")
    digest = md5sum(RAW_CSV)
    if digest != SOURCE_MD5:
        raise SystemExit(f"md5 {digest} != publicado {SOURCE_MD5}: arquivo corrompido.")

    sensors = [c for c in EXPECTED_COLUMNS if c != "timestamp"]
    usecols = ["timestamp", *sensors, *EXTRA_COLUMNS]
    df = pd.read_csv(
        RAW_CSV,
        usecols=usecols,
        dtype={c: np.float32 for c in usecols if c != "timestamp"},
        parse_dates=["timestamp"],
    )
    n_raw = len(df)
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    sampled = point_sample(df)
    sampled["anomaly"] = label(sampled, FAILURE_INTERVALS_MT2)
    sampled["overlap_with_training"] = sampled["timestamp"] <= TRAINING_END
    sampled["dataset"] = "metropt2"
    sampled.to_parquet(OUT_PARQUET, index=False)

    failures = []
    for (s, e), kind, lps in zip(
        FAILURE_INTERVALS_MT2, FAILURE_TYPES_MT2, LPS_ALARMS_MT2
    ):
        m = (sampled["timestamp"] >= pd.Timestamp(s)) & (
            sampled["timestamp"] <= pd.Timestamp(e)
        )
        failures.append(
            {
                "start": s,
                "end": e,
                "type": kind,
                "lps_alarm": lps,
                "rows_10s": int(m.sum()),
            }
        )
    meta: dict[str, Any] = {
        "source": "Zenodo 10.5281/zenodo.7766691 (MetroPT2); labels: arXiv:2502.07394 Table 1",
        "license": "CC BY 4.0",
        "md5": digest,
        "rows_raw_1hz": int(n_raw),
        "rows_10s": int(len(sampled)),
        "period": [str(sampled["timestamp"].min()), str(sampled["timestamp"].max())],
        "training_end": str(TRAINING_END),
        "rows_after_training_end": int((~sampled["overlap_with_training"]).sum()),
        "failures": failures,
        "same_apu_check": same_apu_check(sampled),
    }
    OUT_META.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    log.info("%d linhas 1 Hz -> %d linhas 10 s; %s", n_raw, len(sampled), OUT_PARQUET)
    log.info("falhas: %s", [(f["type"], f["rows_10s"]) for f in failures])


if __name__ == "__main__":
    main()
