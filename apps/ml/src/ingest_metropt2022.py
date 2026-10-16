"""
MetroPT (2022) ingestion — second dataset for RF-33 (cross-dataset).

Source: Veloso et al., "The MetroPT dataset for predictive maintenance",
Scientific Data 9, 764 (2022), doi:10.1038/s41597-022-01877-3.
Data: Zenodo doi:10.5281/zenodo.6854240, file ``dataset_train.csv``
(CC BY 4.0, 1.6 GB, md5 83c713c0f8a6c2f24418cca999b2a8eb).

Same APU compressor family as MetroPT-3 (Porto metro), another period
(Jan–Jun 2022 vs Feb–Sep 2020) and another acquisition rate (1 Hz vs ~0.1 Hz).

Harmonisation with MetroPT-3 (decisions, so the thesis can cite them)
--------------------------------------------------------------------
* Columns: the 15 MetroPT-3 sensors keep their exact names. ``gpsLong``,
  ``gpsLat``, ``gpsSpeed``, ``gpsQuality`` are dropped (position/route, not
  compressor health — a model could memorise the route). ``Flowmeter`` is KEPT
  in the parquet only for the ablation study; it never enters a model that is
  compared with MetroPT-3, which does not have it.
* Rate: point-sampled to 10 s (first reading of each 10 s bin), the same way
  MetroPT-3 was acquired. Averaging would smooth the signal and change the
  distribution of every rolling-std feature.
* Gaps (train switched off) are dropped, not interpolated — as in MetroPT-3.
* Labels: Table 2 of the paper (maintenance reports), dates DD-MM-YY.

Entry point:
    python -m src.ingest_metropt2022
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ingest_metropt import EXPECTED_COLUMNS

_ML_ROOT: Path = Path(__file__).resolve().parent.parent
RAW_CSV: Path = _ML_ROOT / "data" / "raw" / "metropt2022" / "dataset_train.csv"
OUT_PARQUET: Path = _ML_ROOT / "data" / "processed" / "metropt2022.parquet"
OUT_META: Path = _ML_ROOT / "models" / "metropt2022_ingest.json"

SOURCE_MD5: str = "83c713c0f8a6c2f24418cca999b2a8eb"
RESAMPLE: str = "10s"
DROPPED_COLUMNS: tuple[str, ...] = ("gpsLong", "gpsLat", "gpsSpeed", "gpsQuality")
EXTRA_COLUMNS: tuple[str, ...] = ("Flowmeter",)

# Table 2 — "Failures disclosed on Maintenance Reports".
FAILURE_INTERVALS_2022: list[tuple[str, str]] = [
    ("2022-02-28 21:53", "2022-03-01 02:00"),  # air leak — clients
    ("2022-03-23 14:54", "2022-03-23 15:24"),  # air leak — air dryer
    ("2022-05-30 12:00", "2022-06-02 06:18"),  # oil leak — compressor
]
FAILURE_COMPONENTS_2022: list[str] = [
    "air_leak_clients",
    "air_leak_air_dryer",
    "oil_leak_compressor",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log: logging.Logger = logging.getLogger("ingest_metropt2022")


def md5sum(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()  # noqa: S324 — integrity check against the published md5
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def point_sample(df: pd.DataFrame, rule: str = RESAMPLE) -> pd.DataFrame:
    """First reading of each ``rule`` bin; empty bins (gaps) are dropped."""
    out = (
        df.set_index("timestamp")
        .resample(rule, label="left", closed="left")
        .first()
        .dropna(how="all")
        .reset_index()
    )
    return out


def label(df: pd.DataFrame, intervals: list[tuple[str, str]]) -> pd.Series:
    y = np.zeros(len(df), dtype=np.int8)
    ts = df["timestamp"]
    for start, end in intervals:
        y[((ts >= pd.Timestamp(start)) & (ts <= pd.Timestamp(end))).to_numpy()] = 1
    return pd.Series(y, index=df.index, name="anomaly")


def main() -> None:
    if not RAW_CSV.exists():
        raise SystemExit(f"{RAW_CSV} não encontrado — baixe do Zenodo (ver docstring).")
    digest = md5sum(RAW_CSV)
    if digest != SOURCE_MD5:
        raise SystemExit(f"md5 {digest} != publicado {SOURCE_MD5}: arquivo corrompido.")

    sensors: list[str] = [c for c in EXPECTED_COLUMNS if c != "timestamp"]
    usecols: list[str] = ["timestamp", *sensors, *EXTRA_COLUMNS]
    header = pd.read_csv(RAW_CSV, nrows=0).columns.tolist()
    missing = sorted(set(usecols) - set(header))
    if missing:
        raise SystemExit(f"Colunas ausentes no CSV: {missing}")

    log.info("lendo %s (1 Hz)…", RAW_CSV.name)
    df = pd.read_csv(
        RAW_CSV,
        usecols=usecols,
        dtype={c: np.float32 for c in usecols if c != "timestamp"},
        parse_dates=["timestamp"],
    )
    n_raw = len(df)
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    sampled = point_sample(df)
    sampled["anomaly"] = label(sampled, FAILURE_INTERVALS_2022)
    sampled["dataset"] = "metropt2022"
    sampled.to_parquet(OUT_PARQUET, index=False)

    gaps = sampled["timestamp"].diff().dt.total_seconds()
    failures: list[dict[str, Any]] = [
        {
            "start": s,
            "end": e,
            "component": c,
            "rows_10s": int(
                (
                    (sampled["timestamp"] >= pd.Timestamp(s))
                    & (sampled["timestamp"] <= pd.Timestamp(e))
                ).sum()
            ),
        }
        for (s, e), c in zip(FAILURE_INTERVALS_2022, FAILURE_COMPONENTS_2022)
    ]
    meta: dict[str, Any] = {
        "source": "Veloso et al. 2022, Sci Data 9:764; Zenodo 10.5281/zenodo.6854240",
        "license": "CC BY 4.0",
        "md5": digest,
        "rows_raw_1hz": int(n_raw),
        "rows_10s": int(len(sampled)),
        "period": [str(sampled["timestamp"].min()), str(sampled["timestamp"].max())],
        "dropped_columns": list(DROPPED_COLUMNS),
        "extra_columns_kept_for_ablation": list(EXTRA_COLUMNS),
        "resample": f"point sample, first reading per {RESAMPLE} bin",
        "gaps_over_60s": int((gaps > 60).sum()),
        "failures": failures,
        "positive_rows": int(sampled["anomaly"].sum()),
    }
    OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("%d linhas 1 Hz -> %d linhas 10 s; %s", n_raw, len(sampled), OUT_PARQUET)
    log.info("falhas: %s", [f["rows_10s"] for f in failures])


if __name__ == "__main__":
    main()
