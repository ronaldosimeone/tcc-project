"""
Chronology of every validation fold — auditable proof of the temporal split.

Entry point:
    python -m src.fold_chronology

Rebuilds the EXACT outer folds of the nested protocol (``evaluate_nested``:
``PurgedWalkForwardSplit`` over the pooled MetroPT-3 + MetroPT 2022 timeline,
purge 300 s, embargo 3600 s) and, for each one, the inner leave-one-failure-
out folds that select hyper-parameters, calibration and threshold. Reports
real timestamps (never row indices), the failure under test, the purged
interval, and the positive/negative row counts of every partition.

Outputs: models/fold_chronology.json and docs/fold_chronology.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import src.evaluate_cross_dataset as xd
from src.evaluate_temporal import EMBARGO_S, PURGE_S
from src.ingest_metropt import FAILURE_INTERVALS
from src.ingest_metropt2022 import FAILURE_INTERVALS_2022
from src.temporal_validation import (
    FailureEvent,
    PurgedEventBlockKFold,
    PurgedWalkForwardSplit,
    events_from_intervals,
)

_ML_ROOT: Path = Path(__file__).resolve().parent.parent
OUT_JSON: Path = _ML_ROOT / "models" / "fold_chronology.json"
OUT_MD: Path = _ML_ROOT.parent.parent / "docs" / "fold_chronology.md"

# Identity of the 7 failures (report source, component).
_LABELS: dict[str, tuple[str, str]] = {
    **{s: (f"F20_{i}", "air_leak") for i, (s, _) in enumerate(FAILURE_INTERVALS, 1)},
    **{
        s: (f"F22_{i}", kind)
        for i, ((s, _), kind) in enumerate(
            zip(FAILURE_INTERVALS_2022, ["air_leak", "air_dryer", "oil_leak"]), 1
        )
    },
}


def _label(ev: FailureEvent) -> tuple[str, str]:
    for start, lab in _LABELS.items():
        if pd.Timestamp(start) == ev.start:
            return lab
    raise KeyError(ev.start)


def _span(ts: pd.Series, idx: np.ndarray) -> list[str] | None:
    if idx.size == 0:
        return None
    t = ts.iloc[idx]
    return [str(t.min()), str(t.max())]


def _counts(y: np.ndarray, idx: np.ndarray) -> dict[str, int]:
    pos = int(y[idx].sum())
    return {
        "rows": int(idx.size),
        "failure_rows": pos,
        "normal_rows": int(idx.size) - pos,
    }


def chronology() -> dict[str, Any]:
    xd.configure(["Reservoirs", "Oil_level"])
    d20, d22 = xd._read(xd.P2020, []), xd._read(xd.P2022, [])
    ts = pd.concat([d20["timestamp"], d22["timestamp"]], ignore_index=True)
    y = (
        pd.concat([d20[xd.TARGET_COL], d22[xd.TARGET_COL]], ignore_index=True)
        .astype(int)
        .to_numpy()
    )
    events = events_from_intervals([*FAILURE_INTERVALS, *FAILURE_INTERVALS_2022])
    folds: list[dict[str, Any]] = []
    for f in PurgedWalkForwardSplit(events, PURGE_S, EMBARGO_S).split(ts):
        fid, kind = _label(f.test_event)
        lo, hi = pd.Timestamp(ts.iloc[f.train_idx].min()), pd.Timestamp(
            ts.iloc[f.train_idx].max()
        )
        train_events = [e for e in events if e.start >= lo and e.end <= hi]
        inner = []
        ts_tr = ts.iloc[f.train_idx].reset_index(drop=True)
        if len(train_events) >= 2:
            for g in PurgedEventBlockKFold(train_events, PURGE_S, EMBARGO_S).split(
                ts_tr
            ):
                iid, _ = _label(g.test_event)
                tr = f.train_idx[g.train_idx]
                te = f.train_idx[g.test_idx]
                inner.append(
                    {
                        "held_out_failure": iid,
                        "validation_segment": [str(g.test_start), str(g.test_end)],
                        "train": _counts(y, tr),
                        "validation": _counts(y, te),
                        "trains_on_later_failures": any(
                            e.start > g.test_event.end for e in train_events
                        ),
                    }
                )
        folds.append(
            {
                "fold": f.fold,
                "test_failure": fid,
                "failure_type": kind,
                "failure_interval": [str(f.test_event.start), str(f.test_event.end)],
                "train_span": _span(ts, f.train_idx),
                "train_failures": [_label(e)[0] for e in train_events],
                "purge": [
                    str(f.test_start - pd.Timedelta(seconds=PURGE_S)),
                    str(f.test_start),
                ],
                "test_segment": [str(f.test_start), str(f.test_end)],
                "test_span": _span(ts, f.test_idx),
                "train": _counts(y, f.train_idx),
                "test": _counts(y, f.test_idx),
                "selection": (
                    "nested LOFO"
                    if len(inner) >= 2
                    else "fallback (a-priori, uncalibrated)"
                ),
                "inner_folds": inner,
                "no_train_row_after_test_start": bool(
                    ts.iloc[f.train_idx].max()
                    < f.test_start - pd.Timedelta(seconds=PURGE_S)
                ),
            }
        )
    return {
        "protocol": "PurgedWalkForwardSplit (outer) x PurgedEventBlockKFold (inner)",
        "purge_s": PURGE_S,
        "embargo_s": EMBARGO_S,
        "n_failures": len(events),
        "failures": [
            {
                "id": _label(e)[0],
                "type": _label(e)[1],
                "start": str(e.start),
                "end": str(e.end),
            }
            for e in events
        ],
        "folds": folds,
    }


def _md(c: dict[str, Any]) -> str:
    lines = [
        "# Fold chronology — nested temporal protocol",
        "",
        "> Generated by `python -m src.fold_chronology` — do not edit by hand.",
        "",
        f"Outer: purged walk-forward (train only before `test start − {c['purge_s']:.0f} s`). "
        f"Inner (selection of hyper-parameters, calibration and threshold): leave-one-failure-out "
        f"over the failures INSIDE the outer training period, purge {c['purge_s']:.0f} s and "
        f"embargo {c['embargo_s']:.0f} s around each held-out segment. The outer test segment "
        "is never seen by any choice.",
        "",
        "## The failures",
        "",
        "| ID | Type | Start | End |",
        "|---|---|---|---|",
        *[
            f"| {f['id']} | {f['type']} | {f['start']} | {f['end']} |"
            for f in c["failures"]
        ],
        "",
        "## Outer folds",
        "",
        "| Fold | Test failure | Train span | Failures in train | Purged | Test segment | Train rows (failure / normal) | Test rows (failure / normal) | Selection |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for f in c["folds"]:
        lines.append(
            f"| {f['fold']} | {f['test_failure']} ({f['failure_type']}) | "
            f"{f['train_span'][0]} → {f['train_span'][1]} | {', '.join(f['train_failures'])} | "
            f"{f['purge'][0]} → {f['purge'][1]} | {f['test_segment'][0]} → {f['test_segment'][1]} | "
            f"{f['train']['failure_rows']:,} / {f['train']['normal_rows']:,} | "
            f"{f['test']['failure_rows']:,} / {f['test']['normal_rows']:,} | {f['selection']} |"
        )
    lines += [
        "",
        "Every fold satisfies `last training row < test start − purge` "
        f"({'yes' if all(f['no_train_row_after_test_start'] for f in c['folds']) else '**NO**'}).",
        "",
        "## Inner folds (selection)",
        "",
        "| Outer fold | Held-out failure | Validation segment | Inner-train rows (failure / normal) | Validation rows (failure / normal) | Inner train uses later failures |",
        "|---|---|---|---|---|---|",
    ]
    for f in c["folds"]:
        if not f["inner_folds"]:
            lines.append(
                f"| {f['fold']} | — | only {len(f['train_failures'])} failure(s) in training: "
                "a-priori hyper-parameters, uncalibrated, a-priori threshold | — | — | — |"
            )
        for g in f["inner_folds"]:
            lines.append(
                f"| {f['fold']} | {g['held_out_failure']} | {g['validation_segment'][0]} → "
                f"{g['validation_segment'][1]} | {g['train']['failure_rows']:,} / "
                f"{g['train']['normal_rows']:,} | {g['validation']['failure_rows']:,} / "
                f"{g['validation']['normal_rows']:,} | {'yes' if g['trains_on_later_failures'] else 'no'} |"
            )
    lines += [
        "",
        "**Known limitation (documented, not hidden).** The inner leave-one-failure-out may "
        "train on failures that occur LATER than the held-out one, always inside the outer "
        "training period (never on the outer test). A strictly forward inner split would leave "
        "most inner folds without any failure to learn from: with seven failures in total, the "
        "first outer folds would have no usable selection at all.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    c = chronology()
    OUT_JSON.write_text(json.dumps(c, indent=1), encoding="utf-8")
    OUT_MD.write_text(_md(c), encoding="utf-8")
    print(f"Wrote {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()
