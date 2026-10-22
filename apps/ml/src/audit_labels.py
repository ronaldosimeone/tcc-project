"""
Label audit — event boundaries first, then false alarms (pre-registered in
docs/label_audit_protocol.md; criteria fixed before looking at any alarm).

Entry point:
    python -m src.audit_labels [--models ...]

Stage A: physical start/end of each labelled failure from sensor evidence
(E1 pressure drop, E2 abnormal duty cycle in the physics twin, E3 LPS), in
10-min blocks, ±48 h around the label.
Stage B: false-alarm episodes of every nested-evaluation model (seed 0,
out-of-time test folds), clustered across models and classified as
boundary / unreported-failure candidate / weak evidence / no evidence.
Secondary analysis: false alarms and RN-02 cost with the alarm time inside
boundary and candidate clusters treated as ambiguous. The original labels stay
the primary metric; nothing here feeds back into any model.

Output: models/label_audit.json and docs/label_audit.md.
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
import pandas as pd

import src.evaluate_cross_dataset as xd
from src.digital_twin import PhysicsTwin
from src.evaluate_nested import ALL_MODELS, DROPPED, SCORES_DIR
from src.evaluate_temporal import EARLY_HORIZON_S, EVAL_KW, GRACE_S, PERSISTENCE_K
from src.event_metrics import alarm_episodes, evaluate_events, persistent_alarms
from src.ingest_metropt import FAILURE_INTERVALS
from src.ingest_metropt2022 import FAILURE_INTERVALS_2022
from src.statistical_baselines import BASELINES
from src.temporal_validation import FailureEvent, events_from_intervals

_ML_ROOT: Path = Path(__file__).resolve().parent.parent
REPO_ROOT: Path = _ML_ROOT.parent.parent
OUT_JSON: Path = _ML_ROOT / "models" / "label_audit.json"
OUT_MD: Path = REPO_ROOT / "docs" / "label_audit.md"

BLOCK: str = "10min"
BLOCK_TD: pd.Timedelta = pd.Timedelta(BLOCK)
HEALTHY_MARGIN: pd.Timedelta = pd.Timedelta(hours=24)
SEARCH: pd.Timedelta = pd.Timedelta(hours=48)
MAINT_PROXIMITY: pd.Timedelta = pd.Timedelta(hours=24)
CLUSTER_GAP: pd.Timedelta = pd.Timedelta(hours=1)
MIN_BLOCKS_STRONG: int = 3
SENSOR_EVIDENCE: tuple[str, ...] = ("E1", "E2", "E3")
CLASSES: tuple[str, ...] = ("boundary", "candidate", "weak", "none")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("audit_labels")


# ---------------------------------------------------------------------------
# Evidence per 10-min block
# ---------------------------------------------------------------------------


def _near_failure(ts: pd.Series, events: list[FailureEvent]) -> np.ndarray:
    near = np.zeros(len(ts), bool)
    for ev in events:
        near |= (
            (ts >= ev.start - HEALTHY_MARGIN) & (ts <= ev.end + HEALTHY_MARGIN)
        ).to_numpy()
    return near


def evidence_blocks(
    df: pd.DataFrame, events: list[FailureEvent], e2_inclusive: bool = False
) -> pd.DataFrame:
    """E1–E4 per 10-min block of one dataset (``df``: raw sensors, sorted).
    ``e2_inclusive`` = declared deviation (``duty >= p99``, see render)."""
    ts = pd.to_datetime(df["timestamp"]).reset_index(drop=True)
    healthy = (df["anomaly"].to_numpy() == 0) & ~_near_failure(ts, events)
    twin = PhysicsTwin(pressure_col="TP3").fit(df.loc[healthy].reset_index(drop=True))
    res = twin.transform(df.reset_index(drop=True))
    rows = pd.DataFrame(
        {
            "block": ts.dt.floor(BLOCK),
            "tp3": df["TP3"].to_numpy(),
            "fill": res["fill_residual"].to_numpy(),
            "duty": res["duty_residual"].to_numpy(),
            "lps": df["LPS"].to_numpy(),
            "healthy": healthy,
        }
    )
    b = rows.groupby("block").agg(
        tp3=("tp3", "mean"),
        fill=("fill", "mean"),
        duty=("duty", "mean"),
        lps=("lps", "max"),
        healthy=("healthy", "all"),
    )
    ref = b[b["healthy"]]
    thr = {
        "tp3_p01": float(ref["tp3"].quantile(0.01)),
        "fill_p01": float(ref["fill"].quantile(0.01)),
        "duty_p99": float(ref["duty"].quantile(0.99)),
    }
    b["E1"] = (b["tp3"] < thr["tp3_p01"]) | (b["fill"] < thr["fill_p01"])
    b["E2"] = (
        (b["duty"] >= thr["duty_p99"])
        if e2_inclusive
        else (b["duty"] > thr["duty_p99"])
    )
    b["E3"] = b["lps"] >= 0.5
    idx = b.index.to_series()
    b["E4"] = False
    for ev in events:
        b.loc[
            ((idx >= ev.start - MAINT_PROXIMITY) & (idx <= ev.end + MAINT_PROXIMITY)),
            "E4",
        ] = True
    b.attrs["thresholds"] = thr
    b.attrs["healthy_blocks"] = int(b["healthy"].sum())
    # Share of healthy blocks at the duty-cycle ceiling (compressor loading
    # for the whole last hour) — the signature of the 2020 labelled leaks.
    duty_max = float(b["duty"].max())
    b.attrs["healthy_at_duty_ceiling"] = float((ref["duty"] >= duty_max - 1e-9).mean())
    return b


# ---------------------------------------------------------------------------
# Stage A — boundaries
# ---------------------------------------------------------------------------


def _stretches(on: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Runs of evidence blocks allowing a single missing block in between;
    each run is (first block start, last block end)."""
    times = on.index[on.to_numpy()]
    if len(times) == 0:
        return []
    out = []
    s = p = times[0]
    for t in times[1:]:
        if t - p > 2 * BLOCK_TD:
            out.append((s, p + BLOCK_TD))
            s = t
        p = t
    out.append((s, p + BLOCK_TD))
    return out


def boundaries(
    blocks: pd.DataFrame, events: list[FailureEvent], ids: list[str]
) -> list[dict[str, Any]]:
    any_ev = blocks[list(SENSOR_EVIDENCE)].any(axis=1)
    out = []
    for fid, ev in zip(ids, events):
        win = any_ev[
            (any_ev.index >= ev.start - SEARCH) & (any_ev.index <= ev.end + SEARCH)
        ]
        touching = [(a, b) for a, b in _stretches(win) if b >= ev.start and a <= ev.end]
        in_label = blocks[
            (blocks.index >= ev.start.floor(BLOCK)) & (blocks.index <= ev.end)
        ]
        shares = {
            e: float(in_label[e].mean()) if len(in_label) else 0.0
            for e in SENSOR_EVIDENCE
        }
        if not touching:
            out.append(
                {
                    "id": fid,
                    "label": [str(ev.start), str(ev.end)],
                    "physical": None,
                    "evidence": "none",
                    "start_shift_h": 0.0,
                    "end_shift_h": 0.0,
                    "evidence_share_in_label": shares,
                }
            )
            continue
        p0 = min(a for a, _ in touching)
        p1 = max(b for _, b in touching)
        out.append(
            {
                "id": fid,
                "label": [str(ev.start), str(ev.end)],
                "physical": [str(p0), str(p1)],
                "evidence": "sensors",
                # negative = physical start BEFORE the label
                "start_shift_h": (p0 - ev.start).total_seconds() / 3600,
                "end_shift_h": (p1 - ev.end).total_seconds() / 3600,
                "evidence_share_in_label": shares,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Stage B — false alarms
# ---------------------------------------------------------------------------


def _windows(events: list[FailureEvent]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    h, g = pd.Timedelta(seconds=EARLY_HORIZON_S), pd.Timedelta(seconds=GRACE_S)
    return [(ev.start - h, ev.end + g) for ev in events]


def _outside(
    a: pd.Timestamp, b: pd.Timestamp, windows: list[tuple[pd.Timestamp, pd.Timestamp]]
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    segs = [(a, b)]
    for w0, w1 in windows:
        kept = []
        for s, e in segs:
            if w1 < s or w0 > e:
                kept.append((s, e))
                continue
            if s < w0:
                kept.append((s, w0))
            if e > w1:
                kept.append((w1, e))
        segs = kept
    return segs


def model_alarms(name: str, ts: pd.Series) -> list[dict[str, Any]]:
    """Per fold (seed 0): test rows and the persisted alarm vector."""
    out = []
    for f in range(1, 7):
        p = SCORES_DIR / f"{name}_s0_f{f}.npz"
        if not p.exists():
            return []
        z = np.load(p)
        rows = z["test_rows"]
        alarms = persistent_alarms(
            (z["test_raw"] >= float(z["raw_threshold"])).astype(float),
            0.5,
            PERSISTENCE_K,
        )
        out.append(
            {
                "fold": f,
                "rows": rows,
                "ts": ts.iloc[rows].reset_index(drop=True),
                "alarms": alarms,
            }
        )
    return out


def false_alarm_spans(
    folds: list[dict[str, Any]], windows: list[tuple[pd.Timestamp, pd.Timestamp]]
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    spans = []
    for f in folds:
        for ep in alarm_episodes(f["ts"], f["alarms"]):
            spans.extend(_outside(ep.start, ep.end, windows))
    return spans


def cluster(
    spans: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]],
) -> list[dict[str, Any]]:
    flat = sorted((a, b, m) for m, ss in spans.items() for a, b in ss)
    out: list[dict[str, Any]] = []
    for a, b, m in flat:
        if out and a - out[-1]["end"] <= CLUSTER_GAP:
            out[-1]["end"] = max(out[-1]["end"], b)
            out[-1]["models"].add(m)
        else:
            out.append({"start": a, "end": b, "models": {m}})
    return out


def classify(
    c: dict[str, Any],
    blocks: pd.DataFrame,
    physical_windows: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> dict[str, Any]:
    sel = blocks[(blocks.index >= c["start"].floor(BLOCK)) & (blocks.index <= c["end"])]
    counts = {e: int(sel[e].sum()) for e in (*SENSOR_EVIDENCE, "E4")}
    touches = any(c["end"] >= w0 and c["start"] <= w1 for w0, w1 in physical_windows)
    present = [e for e in SENSOR_EVIDENCE if counts[e] > 0]
    if touches or counts["E4"] > 0:
        cls = "boundary"
    elif len(present) >= 2 and max(counts[e] for e in present) >= MIN_BLOCKS_STRONG:
        cls = "candidate"
    elif len(present) == 1:
        cls = "weak"
    elif len(present) >= 2:
        cls = "weak"  # two kinds of evidence but each shorter than 30 min
    else:
        cls = "none"
    return {
        "start": str(c["start"]),
        "end": str(c["end"]),
        "hours": (c["end"] - c["start"]).total_seconds() / 3600,
        "models": sorted(c["models"]),
        "blocks": int(len(sel)),
        "evidence_blocks": counts,
        "class": cls,
    }


def secondary(
    folds: list[dict[str, Any]],
    events: list[FailureEvent],
    ambiguous: list[tuple[pd.Timestamp, pd.Timestamp]],
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> dict[str, Any]:
    """Original vs audited false alarms / cost (alarm samples inside an
    ambiguous span AND outside every event window are dropped)."""
    tot = {
        "original": {"fa": 0, "cost": 0.0, "detected": 0},
        "audited": {"fa": 0, "cost": 0.0, "detected": 0},
    }
    for f in folds:
        ts = f["ts"]
        rep = evaluate_events(ts, f["alarms"], events, **EVAL_KW)
        in_win = np.zeros(len(ts), bool)
        for w0, w1 in windows:
            in_win |= ((ts >= w0) & (ts <= w1)).to_numpy()
        amb = np.zeros(len(ts), bool)
        for a, b in ambiguous:
            amb |= ((ts >= a) & (ts <= b)).to_numpy()
        alarms2 = f["alarms"] & ~(amb & ~in_win)
        rep2 = evaluate_events(ts, alarms2, events, **EVAL_KW)
        for key, r in (("original", rep), ("audited", rep2)):
            tot[key]["fa"] += r.n_false_alarm_episodes
            tot[key]["cost"] += r.expected_cost
            tot[key]["detected"] += r.n_detected
    for key in tot:
        tot[key]["cost"] /= max(len(folds), 1)  # mean per fold, as in the reports
    return tot


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(
    d20: pd.DataFrame,
    d22: pd.DataFrame,
    models: list[str],
    e2_inclusive: bool,
) -> dict[str, Any]:
    ev20 = events_from_intervals(FAILURE_INTERVALS)
    ev22 = events_from_intervals(FAILURE_INTERVALS_2022)
    ids = [
        *(f"F20_{i}" for i in range(1, len(ev20) + 1)),
        *(f"F22_{i}" for i in range(1, len(ev22) + 1)),
    ]
    events = [*ev20, *ev22]
    ts = pd.concat([d20["timestamp"], d22["timestamp"]], ignore_index=True)

    b20 = evidence_blocks(d20, ev20, e2_inclusive)
    b22 = evidence_blocks(d22, ev22, e2_inclusive)
    blocks = pd.concat([b20, b22]).sort_index()

    stage_a = [
        *boundaries(b20, ev20, ids[: len(ev20)]),
        *boundaries(b22, ev22, ids[len(ev20) :]),
    ]
    h, g = pd.Timedelta(seconds=EARLY_HORIZON_S), pd.Timedelta(seconds=GRACE_S)
    physical_windows = []
    for a, ev in zip(stage_a, events):
        p0, p1 = (
            (pd.Timestamp(a["physical"][0]), pd.Timestamp(a["physical"][1]))
            if a["physical"]
            else (ev.start, ev.end)
        )
        physical_windows.append((min(p0, ev.start) - h, max(p1, ev.end) + g))

    windows = _windows(events)
    per_model, spans = {}, {}
    for name in models:
        folds = model_alarms(name, ts)
        if not folds:
            log.warning("no scores for %s — skipped", name)
            continue
        per_model[name] = folds
        spans[name] = false_alarm_spans(folds, windows)
    clusters = [classify(c, blocks, physical_windows) for c in cluster(spans)]
    ambiguous = [
        (pd.Timestamp(c["start"]), pd.Timestamp(c["end"]))
        for c in clusters
        if c["class"] in ("boundary", "candidate")
    ]
    sec = {n: secondary(f, events, ambiguous, windows) for n, f in per_model.items()}
    return {
        "thresholds": {
            "2020": b20.attrs["thresholds"],
            "2022": b22.attrs["thresholds"],
        },
        "healthy_blocks": {
            "2020": b20.attrs["healthy_blocks"],
            "2022": b22.attrs["healthy_blocks"],
        },
        "healthy_at_duty_ceiling": {
            "2020": b20.attrs["healthy_at_duty_ceiling"],
            "2022": b22.attrs["healthy_at_duty_ceiling"],
        },
        "stage_a": stage_a,
        "clusters": clusters,
        "class_counts": {
            k: sum(1 for c in clusters if c["class"] == k) for k in CLASSES
        },
        "class_hours": {
            k: round(sum(c["hours"] for c in clusters if c["class"] == k), 1)
            for k in CLASSES
        },
        "secondary": sec,
        "models": list(per_model),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--models", nargs="+", default=[*ALL_MODELS, *BASELINES])
    args = parser.parse_args()

    xd.configure(DROPPED)
    d20 = xd._read(xd.P2020, ["LPS"])
    d22 = xd._read(xd.P2022, ["LPS"])
    log.info("pre-registered criteria …")
    primary = run(d20, d22, args.models, e2_inclusive=False)
    log.info("declared deviation (E2 >= p99) …")
    deviation = run(d20, d22, args.models, e2_inclusive=True)
    result = {
        "schema_version": "1.0",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "docs/label_audit_protocol.md",
        "primary": primary,
        "deviation_e2_inclusive": deviation,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    OUT_MD.write_text(render(result), encoding="utf-8")
    log.info("wrote %s and %s", OUT_JSON.name, OUT_MD.name)


def _h(x: float) -> str:
    return f"{x:+.1f}"


def _section(r: dict[str, Any], hx: str) -> list[str]:
    lines = [
        f"{hx} Etapa A — fronteiras dos eventos",
        "",
        "Deslocamento negativo = a evidência física começa **antes** do rótulo.",
        "",
        "| Falha | Rótulo | Intervalo físico (E1–E3) | Δ início (h) | Δ fim (h) | % blocos do rótulo com E1 / E2 / E3 |",
        "|---|---|---|---|---|---|",
    ]
    for a in r["stage_a"]:
        sh = a["evidence_share_in_label"]
        share = " / ".join(f"{sh[e]:.0%}" for e in SENSOR_EVIDENCE)
        if a["physical"] is None:
            lines.append(
                f"| {a['id']} | {a['label'][0]} → {a['label'][1]} | sem evidência nos sensores | — | — | {share} |"
            )
        else:
            lines.append(
                f"| {a['id']} | {a['label'][0]} → {a['label'][1]} | {a['physical'][0]} → {a['physical'][1]} "
                f"| {_h(a['start_shift_h'])} | {_h(a['end_shift_h'])} | {share} |"
            )
    cc, ch = r["class_counts"], r["class_hours"]
    names = {
        "boundary": "Fronteira (perto de falha rotulada)",
        "candidate": "Candidato a falha não reportada",
        "weak": "Evidência fraca",
        "none": "Sem evidência (alarme falso de fato)",
    }
    lines += [
        "",
        f"{hx} Etapa B — falsos alarmes",
        "",
        f"Modelos: {', '.join(f'`{m}`' for m in r['models'])} (seed 0, 6 folds fora do tempo). "
        "Episódios fora das janelas da métrica, agrupados entre modelos (≤ 1 h).",
        "",
        "| Classe | Agrupamentos | Horas de alarme |",
        "|---|---|---|",
        *[f"| {names[k]} | {cc[k]} | {ch[k]} |" for k in CLASSES],
        "",
        f"{hx}# Candidatos a falha não reportada",
        "",
        "| Início | Fim | Horas | Modelos | Blocos E1 / E2 / E3 |",
        "|---|---|---|---|---|",
    ]
    cands = [c for c in r["clusters"] if c["class"] == "candidate"]
    for c in cands:
        e = c["evidence_blocks"]
        lines.append(
            f"| {c['start']} | {c['end']} | {c['hours']:.1f} | {len(c['models'])} | {e['E1']} / {e['E2']} / {e['E3']} |"
        )
    if not cands:
        lines.append("| — | — | — | — | — |")
    lines += [
        "",
        f"{hx} Análise secundária — falsos alarmes e custo com rótulos auditados",
        "",
        "Tempo de alarme dentro de agrupamentos *fronteira* e *candidato* tratado "
        "como ambíguo (nem acerto nem alarme falso). Detecções inalteradas.",
        "",
        "| Modelo | Detectadas | FA original | FA auditado | Custo original | Custo auditado |",
        "|---|---|---|---|---|---|",
    ]
    for m, s in sorted(
        r["secondary"].items(), key=lambda kv: kv[1]["original"]["cost"]
    ):
        o, a = s["original"], s["audited"]
        lines.append(
            f"| `{m}` | {o['detected']}/6 | {o['fa']} | {a['fa']} | {o['cost']:.1f} | {a['cost']:.1f} |"
        )
    lines.append("")
    return lines


def render(result: dict[str, Any]) -> str:
    p, d = result["primary"], result["deviation_e2_inclusive"]
    ceil = p["healthy_at_duty_ceiling"]
    lines = [
        "# Auditoria dos rótulos — resultado",
        "",
        "> Gerado por `python -m src.audit_labels` a partir de "
        "`apps/ml/models/label_audit.json`. Não edite à mão. Protocolo "
        "pré-registrado: [`label_audit_protocol.md`](label_audit_protocol.md).",
        "",
        "**Rótulos originais = métrica principal. Rótulos auditados = análise "
        "secundária. Nenhum modelo foi retreinado nem ajustado com a auditoria.** "
        "Custos aqui são da seed 0 (média por fold); os relatórios principais "
        "fazem a média de 3 seeds.",
        "",
        "## 1. Critérios pré-registrados (resultado principal da auditoria)",
        "",
        *_section(p, "###"),
        "## 2. Desvio declarado — E2 com `≥` em vez de `>`",
        "",
        "A etapa A mostrou um defeito do critério E2: o resíduo de ciclo de "
        "trabalho tem teto (compressor em carga a hora inteira = 1 − média "
        f"saudável). Em 2020, **{ceil['2020']:.1%} dos blocos rotulados como "
        f"saudáveis** (2022: {ceil['2022']:.1%}) já estão nesse teto, então o "
        "percentil 99 da referência coincide com o máximo e `duty > p99` nunca "
        "dispara, nem dentro de F20_1/F20_2, onde o compressor fica em carga "
        "contínua por horas. Esses blocos saudáveis no teto têm a **mesma "
        "assinatura das falhas rotuladas de 2020**, o que já é evidência de "
        'vazamentos não reportados no período "saudável".',
        "",
        "Pelo protocolo, o critério principal não é corrigido. A variante abaixo "
        "(`duty ≥ p99`) foi decidida **depois** de ver a etapa A e **antes** de "
        "ver a etapa B com ela; é análise de sensibilidade, não substitui a "
        "seção 1.",
        "",
        *_section(d, "###"),
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
