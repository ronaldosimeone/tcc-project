"""
RAG diagnosis on the REAL failures — RF-35 / RF-36 / RN-04 (honest version).

The original benchmark (benchmark_rag.py) asks operator-written questions.
This one asks what production can actually ask: the evidence measured on the
onset of every documented failure (MetroPT-3 ×4, MetroPT 2022 ×3, MetroPT2 ×2,
built by ``apps/ml/src/build_diagnosis_cases.py``), and checks whether
retrieval lands on the maintenance tasks of the component that failed.

Stages:
    1. queries   python benchmarks/benchmark_rag_diagnosis.py queries
                 -> benchmarks/out/rag_diag_queries.json
    2. retrieve  (apps/mcp-server image) python benchmark_rag_retrieval.py
                 --queries /bench/out/rag_diag_queries.json
                 -> benchmarks/out/rag_diag_retrieval[_<embedder>].json
    3. generate  python benchmarks/benchmark_rag_diagnosis.py generate   (Ollama)
                 -> benchmarks/out/rag_diag_generation.json
    4. report    python benchmarks/benchmark_rag_diagnosis.py report
                 -> docs/rag_diagnosis_benchmark.md + out/rag_diag_metrics.json

Query regimes (what the assistant is told):
    xai_occlusion  equipment + top-3 sensors of the online RCA (production path,
                   no human input)
    xai_deviation  equipment + top-3 |z| deviations (model-agnostic evidence)
    operator       an operator reading the panel: "<sensor> acima/abaixo do
                   normal" for the top-3 deviations + the RCA sensors. The
                   failure name is NEVER given.
    oracle         the failure name itself (upper bound: retrieval quality
                   when the diagnosis is already known)

Relevance (PROVISIONAL, pending maintenance-specialist validation, like
rag_cases_v1): WAP-5 task codes per component family —
    oil_leak   601.x tasks about compressor oil (check, inspect/leakage,
               change, filters, breather)
    air_dryer  602.x (air drier)
    air_leak   pneumatic circuit: delivery hose, outlet copper gasket,
               reservoir pipes, drain valves, check valves, safety valves
A chunk is relevant when it contains a task code of the family. Identification
= rank-weighted vote (1/rank) of the task families in the top-5 chunks.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
import numpy as np

_BACKEND: Path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_rag import (  # noqa: E402
    MCP_CUT,
    MCP_TOP_K,
    ReplayMCP,
    TimedOllama,
    _norm,
)
from src.schemas.maintenance import (  # noqa: E402
    MaintenanceSuggestionRequest,
    SensorContribution,
)
from src.services.grounding import extract_citations  # noqa: E402
from src.services.maintenance_suggestion_service import (  # noqa: E402
    MaintenanceSuggestionService,
)
from src.services.sensor_semantics import SENSOR_MEANINGS  # noqa: E402

BENCH: Path = Path(__file__).resolve().parent
OUT: Path = BENCH / "out"
CASES: Path = BENCH / "rag_diagnosis_cases.json"
REPO: Path = _BACKEND.parents[1]
REPORT_MD: Path = REPO / "docs" / "rag_diagnosis_benchmark.md"

EQUIP: str = "Compressor de ar da unidade de produção de ar (APU)"
CONFIGS: tuple[str, ...] = ("xai_occlusion", "xai_deviation", "operator", "oracle")
TYPES: tuple[str, ...] = ("air_leak", "air_dryer", "oil_leak")
TAU: float = 0.6  # RN-04 grounding threshold (production default)

ORACLE_SYMPTOM: dict[str, str] = {
    "air_leak": "vazamento de ar no circuito pneumático (air leak)",
    "air_dryer": "falha no secador de ar (air drier fault)",
    "oil_leak": "vazamento de óleo do compressor (compressor oil leak)",
}

# WAP-5 task codes (Vol. B, sec. 6.6, pp. 170-177) per failure family.
TASK_FAMILIES: dict[str, set[str]] = {
    "oil_leak": {
        "601.1W.02",
        "601.1W.03",
        "601.1M.02",
        "601.3M.05",
        "601.3M.06",
        "601.1Y.02",
        "601.1Y.03",
        "601.1Y.05",
    },
    "air_dryer": {
        "602.1W.01",
        "602.1M.01",
        "602.3M.01",
        "602.3M.02",
        "602.1Y.01",
        "602.1Y.02",
        "602.1Y.03",
        "602.1Y.04",
        "602.2Y.01",
        "602.4Y.01",
        "602.4Y.02",
        "602.5Y.01",
    },
    "air_leak": {
        "601.3M.02",
        "601.5Y.02",
        "601.1Y.06",
        "601.4Y.04",
        "603.1W.01",
        "603.3M.01",
        "603.3M.02",
        "603.6M.03",
        "603.1Y.01",
        "603.4Y.02",
        "603.4Y.05",
        "603.4Y.06",
    },
}
GOLD_PAGES: dict[str, set[int]] = {
    "oil_leak": {170, 171, 172},
    "air_dryer": {173, 174},
    "air_leak": {170, 172, 173, 174, 175, 176},
}
WAP5: str = "wap5-scheduled-maintenance-vol-b.pdf"

# Words that name the component in a generated plan (accent/case-insensitive).
COMPONENT_WORDS: dict[str, list[str]] = {
    "air_leak": [
        "vazamento de ar",
        "air leak",
        "tubulac",
        "pneumatic pipe",
        "valvula de retencao",
        "check valve",
        "dreno",
    ],
    "air_dryer": ["secador", "drier", "dryer", "dessecante", "desiccant", "torre"],
    "oil_leak": [
        "vazamento de oleo",
        "oil leak",
        "nivel de oleo",
        "oil level",
        "oleo do compressor",
        "compressor oil",
    ],
}

_CODE = re.compile(r"60[1-5]\.\d{1,2}[WMY]\.[0-9O]{2}")


def task_codes(text: str) -> set[str]:
    compact = re.sub(r"\s+", "", text)
    return {c[:-2] + c[-2:].replace("O", "0") for c in _CODE.findall(compact)}


def chunk_family(text: str) -> set[str]:
    codes = task_codes(text)
    return {fam for fam, fam_codes in TASK_FAMILIES.items() if codes & fam_codes}


# ---------------------------------------------------------------------------
# Stage 1 — queries
# ---------------------------------------------------------------------------


def _contribs(items: list[dict[str, Any]]) -> list[SensorContribution]:
    return [
        SensorContribution(sensor=c["sensor"], contribution=float(c["contribution"]))
        for c in items
        if c["sensor"] in SENSOR_MEANINGS
    ]


def operator_symptom(deviation: list[dict[str, Any]], k: int = 3) -> str:
    parts = []
    for c in deviation[:k]:
        word = "acima" if float(c["z"]) > 0 else "abaixo"
        parts.append(f"{SENSOR_MEANINGS[c['sensor']].pt} {word} do normal")
    return "; ".join(parts)


def requests_for(case: dict[str, Any]) -> dict[str, MaintenanceSuggestionRequest]:
    # Diagnosis is evaluated GIVEN that the assistant was asked: the service
    # skips suggestions below its own probability gate, so use the same 0.9
    # as rag_cases_v1. Whether production would have asked = alarm_in_onset.
    p = 0.9
    occl = _contribs(case["occlusion"])
    dev = _contribs(case["deviation"])
    return {
        "xai_occlusion": MaintenanceSuggestionRequest(
            failure_probability=p, equipment_name=EQUIP, sensor_contributions=occl
        ),
        "xai_deviation": MaintenanceSuggestionRequest(
            failure_probability=p, equipment_name=EQUIP, sensor_contributions=dev
        ),
        "operator": MaintenanceSuggestionRequest(
            failure_probability=p,
            equipment_name=EQUIP,
            symptom_description=operator_symptom(case["deviation"]),
            sensor_contributions=occl,
        ),
        "oracle": MaintenanceSuggestionRequest(
            failure_probability=p,
            equipment_name=EQUIP,
            symptom_description=ORACLE_SYMPTOM[case["failure_type"]],
        ),
    }


def stage_queries() -> None:
    cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
    out = {
        c["id"]: {
            cfg: MaintenanceSuggestionService._build_query(req)  # noqa: SLF001
            for cfg, req in requests_for(c).items()
        }
        for c in cases
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "rag_diag_queries.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {len(out)} cases x {len(CONFIGS)} queries")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _relevant(r: dict[str, Any], kind: str) -> bool:
    return kind in chunk_family(r["text"])


def _page_relevant(r: dict[str, Any], kind: str) -> bool:
    md = r["metadata"]
    return md["file_name"] == WAP5 and int(md["page"]) in GOLD_PAGES[kind]


def identify(ranked: list[dict[str, Any]], k: int = MCP_TOP_K) -> str | None:
    votes: dict[str, float] = {}
    for i, r in enumerate(ranked[:k]):
        for fam in chunk_family(r["text"]):
            votes[fam] = votes.get(fam, 0.0) + 1.0 / (i + 1)
    if not votes:
        return None
    best = max(votes.values())
    winners = sorted(f for f, v in votes.items() if v == best)
    return winners[0] if len(winners) == 1 else None  # tie = unidentified


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    from scipy.stats import beta

    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def random_hit_at_k(n_total: int, n_rel: int, k: int) -> float:
    """P(at least one relevant in a random top-k) — hypergeometric."""
    if n_rel == 0:
        return 0.0
    return 1.0 - math.comb(n_total - n_rel, k) / math.comb(n_total, k)


def corpus_families(doc: dict[str, Any]) -> dict[str, int]:
    """Relevant chunks per family in the indexed corpus (from any result list:
    each case keeps the top-20 of a 25-chunk corpus, union over cases)."""
    seen: dict[tuple[str, int, int], str] = {}
    for per_case in doc["results"].values():
        for ranked in per_case.values():
            for r in ranked:
                md = r["metadata"]
                seen[(md["file_name"], int(md["page"]), int(md["chunk_index"]))] = r[
                    "text"
                ]
    return {
        kind: sum(1 for t in seen.values() if kind in chunk_family(t)) for kind in TYPES
    } | {"_n_chunks": int(doc["n_chunks"])}


def metrics(
    cases: list[dict[str, Any]], doc: dict[str, Any], cfg: str
) -> dict[str, Any]:
    rows = []
    for c in cases:
        ranked = doc["results"][c["id"]][cfg]
        kind = c["failure_type"]
        rel = [_relevant(r, kind) for r in ranked]
        prel = [_page_relevant(r, kind) for r in ranked]
        first = next((i for i, v in enumerate(rel) if v), None)
        top1 = ranked[0]["score"] if ranked else -1.0
        kept = [
            r for r in ranked[:MCP_TOP_K] if r["score"] > MCP_CUT and r["score"] >= TAU
        ]
        rows.append(
            {
                "id": c["id"],
                "type": kind,
                "seen": c["seen_in_training"],
                "alarm": c["alarm_in_onset"],
                "hit@1": float(rel[0]) if rel else 0.0,
                "hit@5": float(any(rel[:5])),
                "page_hit@5": float(any(prel[:5])),
                "rr": 0.0 if first is None else 1.0 / (first + 1),
                "identified": identify(ranked),
                "top1_score": top1,
                "answers": bool(kept),
                "answers_with_relevant": any(_relevant(r, kind) for r in kept),
            }
        )
    n = len(rows)
    correct = sum(1 for r in rows if r["identified"] == r["type"])
    confusion = {
        t: Counter(str(r["identified"]) for r in rows if r["type"] == t) for t in TYPES
    }
    return {
        "n": n,
        "hit@1": float(np.mean([r["hit@1"] for r in rows])),
        "hit@5": float(np.mean([r["hit@5"] for r in rows])),
        "page_hit@5": float(np.mean([r["page_hit@5"] for r in rows])),
        "mrr": float(np.mean([r["rr"] for r in rows])),
        "identification_accuracy": correct / n,
        "identification_ci95": clopper_pearson(correct, n),
        # Macro recall over the failure types present: a constant guess of
        # the majority type scores 1 / n_types here, whatever its accuracy.
        "balanced_identification": float(
            np.mean(
                [
                    np.mean([r["identified"] == t for r in rows if r["type"] == t])
                    for t in TYPES
                    if any(r["type"] == t for r in rows)
                ]
            )
        ),
        "minority_identified": sum(
            1 for r in rows if r["type"] != "air_leak" and r["identified"] == r["type"]
        ),
        "minority_n": sum(1 for r in rows if r["type"] != "air_leak"),
        "answer_rate": float(np.mean([r["answers"] for r in rows])),
        "answer_with_relevant_rate": float(
            np.mean([r["answers_with_relevant"] for r in rows])
        ),
        "confusion": {t: dict(v) for t, v in confusion.items()},
        "per_case": rows,
    }


# ---------------------------------------------------------------------------
# Stage 3 — generation (optional, needs Ollama)
# ---------------------------------------------------------------------------


async def stage_generate(
    model: str, retrieval_file: str, mcp_cut: float, tau: float, tag: str
) -> None:
    cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
    retrieval = json.loads((OUT / retrieval_file).read_text(encoding="utf-8"))[
        "results"
    ]
    out_path = OUT / "rag_diag_generation.json"
    rows: list[dict[str, Any]] = (
        json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else []
    )
    done = {(r["id"], r["config"], r["model"], r["setup"]) for r in rows}
    async with httpx.AsyncClient() as http:
        for c in cases:
            for cfg, req in requests_for(c).items():
                if (c["id"], cfg, model, tag) in done:
                    continue
                ollama = TimedOllama(model, http)
                service = MaintenanceSuggestionService(
                    mcp_client=ReplayMCP(retrieval[c["id"]][cfg], mcp_cut),  # type: ignore[arg-type]
                    ollama_client=ollama,  # type: ignore[arg-type]
                    model=model,
                    grounding_threshold=tau,
                )
                resp = await service.suggest(req)
                rows.append(
                    {
                        "id": c["id"],
                        "config": cfg,
                        "model": model,
                        "setup": tag,
                        "markdown": resp.markdown,
                        "references": [r.model_dump() for r in resp.references],
                        "grounding": (
                            resp.grounding.model_dump() if resp.grounding else None
                        ),
                        "seconds": round(ollama.last_seconds, 2),
                    }
                )
                out_path.write_text(
                    json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8"
                )
                print(f"{model} {cfg} {c['id']} {rows[-1]['seconds']}s", flush=True)


def _diagnosis_section(markdown: str) -> str:
    text = _norm(markdown)
    m = re.search(r"##\s*diagnostico[^\n]*\n(.*?)(\n##\s|\Z)", text, re.S)
    return m.group(1) if m else ""


def generation_metrics(
    cases: dict[str, dict[str, Any]], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    named: list[float] = []
    named_wrong: list[float] = []
    abstained: list[float] = []
    cite_rel: list[float] = []
    fabricated: list[float] = []
    for r in rows:
        c = cases[r["id"]]
        md = r["markdown"] or ""
        diag = _diagnosis_section(md) or _norm(md)
        hits = {t: any(w in diag for w in COMPONENT_WORDS[t]) for t in TYPES}
        named.append(float(hits[c["failure_type"]]))
        named_wrong.append(
            float(any(v for t, v in hits.items() if t != c["failure_type"]))
        )
        g = r["grounding"]
        abstained.append(0.0 if (g is None or g.get("llm_called", True)) else 1.0)
        cited = {(x.file_name, x.page) for x in extract_citations(md)}
        retrieved = {(x["file_name"], x["page"]) for x in r["references"]}
        gold = {(WAP5, p) for p in GOLD_PAGES[c["failure_type"]]}
        cite_rel.append(float(bool(cited & gold)))
        fabricated.append(float(bool(cited - retrieved)))
    m = lambda v: float(np.mean(v)) if v else None  # noqa: E731
    return {
        "component_named": m(named),
        "other_component_named": m(named_wrong),
        "abstained_rn04": m(abstained),
        "cites_gold_page": m(cite_rel),
        "fabricated_citation_rate": m(fabricated),
        "n": len(rows),
    }


# ---------------------------------------------------------------------------
# Stage 4 — report
# ---------------------------------------------------------------------------


def _retrieval_files() -> list[Path]:
    base = OUT / "rag_diag_retrieval.json"
    others = sorted(OUT.glob("rag_diag_retrieval_*.json"))
    return [base, *others] if base.exists() else others


def _f(v: float | None) -> str:
    return "—" if v is None else f"{v:.2f}"


def stage_report() -> None:
    data = json.loads(CASES.read_text(encoding="utf-8"))
    cases = data["cases"]
    by_id = {c["id"]: c for c in cases}
    docs = [json.loads(p.read_text(encoding="utf-8")) for p in _retrieval_files()]
    summary: dict[str, Any] = {"cases": len(cases), "embedders": {}}

    lines = [
        "# RAG diagnosis on the real failures — RF-35 / RF-36 / RN-04",
        "",
        "> Generated by `python benchmarks/benchmark_rag_diagnosis.py report` — do not edit by hand.",
        "",
        f"{len(cases)} documented failures (MetroPT-3 ×4, MetroPT 2022 ×3, MetroPT2 ×2). "
        "Evidence measured on the first 30 min of each failure "
        f"(`apps/ml/src/build_diagnosis_cases.py`, active model `{data['active_model']}`). "
        "The failure name is never given except in the `oracle` upper bound.",
        "",
        "Relevance is PROVISIONAL (WAP-5 task codes per component family, pending "
        "maintenance-specialist validation). n = 9: confidence intervals are wide by "
        "construction; the table is evidence about behaviour, not a population estimate.",
        "",
        "## Evidence per failure",
        "",
        "| Case | Dataset | Type | Seen in training | P(failure) onset | Alarm in onset | RCA top-3 (occlusion) | Deviation top-3 (z) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in cases:
        occl = ", ".join(
            f"{x['sensor']} {x['contribution']:+.3f}" for x in c["occlusion"][:3]
        )
        dev = ", ".join(f"{x['sensor']} {x['z']:+.1f}" for x in c["deviation"][:3])
        lines.append(
            f"| {c['id']} | {c['dataset']} | {c['failure_type']} | {'yes' if c['seen_in_training'] else '**no**'} | "
            f"{c['p_failure_onset']:.3f} | {'yes' if c['alarm_in_onset'] else 'no'} | {occl} | {dev} |"
        )

    for doc in docs:
        emb = doc["embedding_model"].split("/")[-1]
        fams = corpus_families(doc)
        n_total = fams["_n_chunks"]
        chance = {t: random_hit_at_k(n_total, fams[t], MCP_TOP_K) for t in TYPES}
        chance_mean = float(np.mean([chance[c["failure_type"]] for c in cases]))
        summary["embedders"][emb] = {
            "corpus_chunks": n_total,
            "relevant_chunks": fams,
            "random_hit@5": chance_mean,
        }
        lines += [
            "",
            f"## Retrieval — {emb}",
            "",
            f"Corpus: {n_total} chunks; relevant chunks per family: "
            + ", ".join(f"{t} {fams[t]}" for t in TYPES)
            + f". Random-ranking hit@5 baseline (mean over cases): {chance_mean:.2f}. "
            f"Baselines for identification: always `air_leak` (majority) = "
            f"{sum(c['failure_type'] == 'air_leak' for c in cases) / len(cases):.2f} accuracy "
            "but 0.33 balanced; uniform guess = 0.33.",
            "",
            "| Query | hit@1 | hit@5 | Page hit@5 | MRR | Identification (95% CI) | Balanced identification | Dryer/oil identified | RN-04 answers | Answers with relevant chunk |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        summary["embedders"][emb]["configs"] = {}
        for cfg in CONFIGS:
            m = metrics(cases, doc, cfg)
            summary["embedders"][emb]["configs"][cfg] = m
            lo, hi = m["identification_ci95"]
            lines.append(
                f"| {cfg} | {m['hit@1']:.2f} | {m['hit@5']:.2f} | {m['page_hit@5']:.2f} | {m['mrr']:.2f} | "
                f"{m['identification_accuracy']:.2f} ({lo:.2f}–{hi:.2f}) | "
                f"{m['balanced_identification']:.2f} | {m['minority_identified']}/{m['minority_n']} | "
                f"{m['answer_rate']:.2f} | "
                f"{m['answer_with_relevant_rate']:.2f} |"
            )
        lines += [
            "",
            "Identified family per case (rank-weighted vote of the top-5; `None` = no task code or tie):",
            "",
            "| Case | Type | " + " | ".join(CONFIGS) + " |",
            "|---|---|" + "---|" * len(CONFIGS),
        ]
        per = {
            cfg: {
                r["id"]: r
                for r in summary["embedders"][emb]["configs"][cfg]["per_case"]
            }
            for cfg in CONFIGS
        }
        for c in cases:
            cells = []
            for cfg in CONFIGS:
                r = per[cfg][c["id"]]
                mark = "✓" if r["identified"] == c["failure_type"] else "✗"
                cells.append(f"{mark} {r['identified']} ({r['top1_score']:.2f})")
            lines.append(
                f"| {c['id']} | {c['failure_type']} | " + " | ".join(cells) + " |"
            )

    gen_path = OUT / "rag_diag_generation.json"
    if gen_path.exists():
        rows = json.loads(gen_path.read_text(encoding="utf-8"))
        lines += [
            "",
            "## Generation (Ollama, temperature 0)",
            "",
            "| Model | Setup | Query | Names failed component | Names another component | RN-04 abstained | Cites gold page | Fabricated citation | n |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        summary["generation"] = {}
        for model in sorted({r["model"] for r in rows}):
            for setup in sorted({r["setup"] for r in rows}):
                for cfg in CONFIGS:
                    sub = [
                        r
                        for r in rows
                        if r["model"] == model
                        and r["setup"] == setup
                        and r["config"] == cfg
                    ]
                    if not sub:
                        continue
                    g = generation_metrics(by_id, sub)
                    summary["generation"][f"{model}|{setup}|{cfg}"] = g
                    lines.append(
                        f"| {model} | {setup} | {cfg} | {_f(g['component_named'])} | {_f(g['other_component_named'])} | "
                        f"{_f(g['abstained_rn04'])} | {_f(g['cites_gold_page'])} | {_f(g['fabricated_citation_rate'])} | {g['n']} |"
                    )

    lines += [
        "",
        "## How to read this",
        "",
        "* **Identification vs majority class.** Six of the nine failures are air leaks, "
        "so a retriever that always lands on the generic main-compressor chunk "
        "(WAP-5 p. 170, which happens to contain the delivery-hose task 601.3M.02) "
        "scores the majority accuracy without identifying anything. Balanced "
        "identification and the dryer/oil column are the honest measures.",
        "* **Why sensor evidence cannot name the failure here.** The online RCA blames "
        "`Oil_temperature` / `Motor_current` / `DV_pressure` for air leaks, the dryer "
        "fault and oil leaks alike: the sensors are shared by every component, so "
        "the sensor→component vocabulary (RF-35) maps all of them to the same "
        "compressor chunks. The query carries no type information to retrieve.",
        "* **Why the corpus cannot either.** The indexed corpus is a *scheduled* "
        "maintenance schedule (25 chunks, 3–6 tasks each): it says what to inspect "
        "periodically, not which symptom points to which fault. There is no "
        "fault-finding table to retrieve, so even the `oracle` queries rarely clear "
        "the similarity cut.",
        "* **RN-04 is doing its job.** In the production setup the assistant abstains "
        "on these cases instead of producing a confident procedure from irrelevant "
        "context, and no citation is fabricated. Abstention is the correct output "
        "for a corpus that does not contain the answer; it is not a diagnosis.",
        "* **Consequence for the platform.** Failure *detection* is the ML layer's job "
        "(see docs/metropt2_unseen_failures.md); failure *type* identification is not "
        "supported by the current corpus and must not be claimed. Closing the gap "
        "needs (a) a fault-finding / troubleshooting document for the APU and (b) "
        "task-level chunking, both re-evaluated with this same benchmark.",
    ]

    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "rag_diag_metrics.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(f"Wrote {REPORT_MD}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("queries")
    g = sub.add_parser("generate")
    g.add_argument("--model", default="llama3.2:3b")
    g.add_argument("--retrieval-file", default="rag_diag_retrieval.json")
    g.add_argument("--mcp-cut", type=float, default=MCP_CUT)
    g.add_argument("--tau", type=float, default=TAU)
    g.add_argument("--tag", default="production")
    sub.add_parser("report")
    args = parser.parse_args()
    if args.stage == "queries":
        stage_queries()
    elif args.stage == "generate":
        asyncio.run(
            stage_generate(
                args.model, args.retrieval_file, args.mcp_cut, args.tau, args.tag
            )
        )
    else:
        stage_report()


if __name__ == "__main__":
    main()
