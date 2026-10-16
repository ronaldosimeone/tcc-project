"""
RAG benchmark and ablation — RF-36 / RF-37 / RN-04 / RNF-82.

Three stages, so retrieval (needs the embedding model, runs in the
mcp-server image) and generation (needs Ollama, runs on the host) can be
reproduced independently and cached:

    1. queries   python benchmarks/benchmark_rag.py queries
                 -> benchmarks/out/rag_queries.json
                 Real ``MaintenanceSuggestionService._build_query`` for each
                 case, with and without the XAI sensor evidence (RF-35).

    2. retrieve  (apps/mcp-server) python benchmark_rag_retrieval.py
                 -> benchmarks/out/rag_retrieval.json  (top-20, unfiltered)

    3. generate  python benchmarks/benchmark_rag.py generate --models llama3.2:3b qwen2.5:3b-instruct
                 Real ``MaintenanceSuggestionService`` + local Ollama. The MCP
                 client is replaced by a replay of stage 2 that applies the
                 mcp-server's own cut (score > 0.6, top 5), so the service
                 sees exactly what production would.
                 -> benchmarks/out/rag_generation.json

    4. report    python benchmarks/benchmark_rag.py report
                 -> docs/rag_benchmark.md

Ablation configurations (RF-37):
    llm_only   no retrieval, permissive prompt (the model's own knowledge)
    rag        query = equipment + symptom (baseline RF-22)
    rag_xai    query = equipment + symptom + top-3 sensor vocabulary (RF-35)

Metrics — all deterministic, no LLM-as-judge:
    retrieval   hit@1, hit@5, MRR, context precision@5 vs gold pages
    grounding   τ sweep: answerable coverage vs unanswerable false-grounding
    generation  fact recall (gold facts present, any accepted surface form),
                citation validity (cites only retrieved pages), citation
                accuracy (cites a gold page), hallucination:
                  answerable   -> any fabricated citation
                  unanswerable -> a procedure was produced instead of "Limitações"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

import httpx
import numpy as np

_BACKEND: Path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))

from src.schemas.maintenance import MaintenanceSuggestionRequest  # noqa: E402
from src.services.grounding import extract_citations  # noqa: E402
from src.services.maintenance_suggestion_service import (  # noqa: E402
    MaintenanceSuggestionService,
)

BENCH: Path = Path(__file__).resolve().parent
OUT: Path = BENCH / "out"
CASES: Path = BENCH / "rag_cases_v1.json"
REPO: Path = _BACKEND.parents[1]
REPORT_MD: Path = REPO / "docs" / "rag_benchmark.md"

MCP_CUT: float = 0.6  # apps/mcp-server/semantic_search.py SCORE_THRESHOLD (score > 0.6)
MCP_TOP_K: int = 5
OLLAMA_URL: str = "http://localhost:11434"

LLM_ONLY_SYSTEM: str = (
    "Você é um assistente técnico de manutenção industrial. Responda em "
    "português, em Markdown, com as seções: # Plano de Manutenção, "
    "## Diagnóstico provável, ## Procedimento recomendado, ## Referências."
)


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def _request(case: dict[str, Any], with_xai: bool) -> MaintenanceSuggestionRequest:
    return MaintenanceSuggestionRequest(
        failure_probability=0.9,
        equipment_name=case["equipment_name"],
        symptom_description=case["symptom_description"],
        sensor_contributions=(
            case["sensor_contributions"]
            if with_xai and case["sensor_contributions"]
            else None
        ),
    )


# ---------------------------------------------------------------------------
# Stage 1 — queries
# ---------------------------------------------------------------------------


def stage_queries() -> None:
    cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
    out: dict[str, dict[str, str]] = {}
    for c in cases:
        out[c["id"]] = {
            "rag": MaintenanceSuggestionService._build_query(
                _request(c, False)
            ),  # noqa: SLF001
            "rag_xai": MaintenanceSuggestionService._build_query(
                _request(c, True)
            ),  # noqa: SLF001
        }
    OUT.mkdir(exist_ok=True)
    (OUT / "rag_queries.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {len(out)} cases x 2 queries")


# ---------------------------------------------------------------------------
# Stage 3 — generation
# ---------------------------------------------------------------------------


class ReplayMCP:
    """Serves stage-2 results with the mcp-server's own filtering."""

    def __init__(
        self, retrieved: list[dict[str, Any]], mcp_cut: float = MCP_CUT
    ) -> None:
        self._retrieved = retrieved
        self._cut = mcp_cut

    async def search_maintenance_manual(self, query: str) -> dict[str, Any]:
        kept = [r for r in self._retrieved if r["score"] > self._cut][:MCP_TOP_K]
        return {"query": query, "results": kept}


class TimedOllama:
    """Minimal non-streaming Ollama chat client that records latency."""

    def __init__(self, model: str, client: httpx.AsyncClient) -> None:
        self.model = model
        self._client = client
        self.last_seconds: float = 0.0

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        t0 = time.perf_counter()
        resp = await self._client.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "options": {"temperature": 0.0, "seed": 42},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=600,
        )
        resp.raise_for_status()
        self.last_seconds = time.perf_counter() - t0
        return str(resp.json()["message"]["content"])


async def _run_case(
    case: dict[str, Any],
    config: str,
    model: str,
    retrieval: dict[str, Any],
    http: httpx.AsyncClient,
    setup: dict[str, Any],
) -> dict[str, Any]:
    ollama = TimedOllama(model, http)
    row: dict[str, Any] = {
        "id": case["id"],
        "config": config,
        "model": model,
        "answerable": case["answerable"],
        "setup": setup["tag"],
    }

    if config == "llm_only":
        user = f"Equipamento: {case['equipment_name']}\nSintoma: {case['symptom_description']}\nGere o plano de manutenção."
        markdown = await ollama.generate(LLM_ONLY_SYSTEM, user)
        row.update(markdown=markdown, references=[], grounding=None)
    else:
        retrieved = retrieval[case["id"]][config]
        service = MaintenanceSuggestionService(
            mcp_client=ReplayMCP(retrieved, setup["mcp_cut"]),  # type: ignore[arg-type]
            ollama_client=ollama,  # type: ignore[arg-type]
            model=model,
            grounding_threshold=setup["tau"],
        )
        resp = await service.suggest(_request(case, config == "rag_xai"))
        row.update(
            markdown=resp.markdown,
            references=[r.model_dump() for r in resp.references],
            grounding=resp.grounding.model_dump() if resp.grounding else None,
        )
    row["seconds"] = round(ollama.last_seconds, 2)
    return row


async def stage_generate(
    models: list[str], configs: list[str], limit: int | None, setup: dict[str, Any]
) -> None:
    cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"][:limit]
    retrieval = json.loads((OUT / setup["retrieval_file"]).read_text(encoding="utf-8"))[
        "results"
    ]
    out_path = OUT / "rag_generation.json"
    rows: list[dict[str, Any]] = (
        json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else []
    )
    done = {
        (r["id"], r["config"], r["model"], r.get("setup", "production")) for r in rows
    }
    async with httpx.AsyncClient() as http:
        for model in models:
            for config in configs:
                for case in cases:
                    # llm_only does not depend on retrieval: run it once.
                    tag = "none" if config == "llm_only" else setup["tag"]
                    key = (case["id"], config, model, tag)
                    if key in done:
                        continue
                    row = await _run_case(
                        case, config, model, retrieval, http, {**setup, "tag": tag}
                    )
                    rows.append(row)
                    out_path.write_text(
                        json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8"
                    )
                    print(
                        f"{model} {config} {case['id']} {row['seconds']}s", flush=True
                    )


# ---------------------------------------------------------------------------
# Stage 4 — metrics + report
# ---------------------------------------------------------------------------


def _gold(case: dict[str, Any]) -> set[tuple[str, int]]:
    return {(g["file_name"], g["page"]) for g in case["gold_refs"]}


def retrieval_metrics(
    cases: list[dict[str, Any]], retrieval: dict[str, Any], config: str
) -> dict[str, float]:
    hit1: list[float] = []
    hit5: list[float] = []
    rr: list[float] = []
    prec5: list[float] = []
    for c in cases:
        if not c["answerable"]:
            continue
        gold = _gold(c)
        ranked = [
            (r["metadata"]["file_name"], int(r["metadata"]["page"]))
            for r in retrieval[c["id"]][config]
        ]
        rel = [1.0 if k in gold else 0.0 for k in ranked]
        hit1.append(rel[0] if rel else 0.0)
        hit5.append(1.0 if any(rel[:5]) else 0.0)
        first = next((i for i, v in enumerate(rel) if v), None)
        rr.append(0.0 if first is None else 1.0 / (first + 1))
        prec5.append(float(np.mean(rel[:5])) if rel else 0.0)
    return {
        "hit@1": float(np.mean(hit1)),
        "hit@5": float(np.mean(hit5)),
        "mrr": float(np.mean(rr)),
        "context_precision@5": float(np.mean(prec5)),
        "n": len(hit1),
    }


def tau_sweep(
    cases: list[dict[str, Any]], retrieval: dict[str, Any], config: str
) -> list[dict[str, float]]:
    """RN-04 calibration: for each τ, share of answerable cases that keep ≥ 1
    GOLD chunk (useful coverage) vs share of unanswerable cases that keep any
    chunk (false grounding)."""
    out: list[dict[str, float]] = []
    for tau in np.round(np.arange(0.30, 0.81, 0.05), 2):
        cov: list[float] = []
        false_g: list[float] = []
        for c in cases:
            kept = [
                r for r in retrieval[c["id"]][config][:MCP_TOP_K] if r["score"] >= tau
            ]
            if c["answerable"]:
                gold = _gold(c)
                cov.append(
                    1.0
                    if any(
                        (r["metadata"]["file_name"], int(r["metadata"]["page"])) in gold
                        for r in kept
                    )
                    else 0.0
                )
            else:
                false_g.append(1.0 if kept else 0.0)
        out.append(
            {
                "tau": float(tau),
                "gold_coverage": float(np.mean(cov)),
                "false_grounding": float(np.mean(false_g)),
            }
        )
    return out


def _has_procedure(markdown: str | None) -> bool:
    if not markdown:
        return False
    return "## limitacoes" not in _norm(markdown) or "## procedimento" in _norm(
        markdown
    )


def generation_metrics(
    cases: dict[str, dict[str, Any]], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    facts: list[float] = []
    novel_facts: list[float] = []
    cite_valid: list[float] = []
    cite_gold: list[float] = []
    halluc_ans: list[float] = []
    halluc_unans: list[float] = []
    secs: list[float] = []
    for r in rows:
        c = cases[r["id"]]
        md = r["markdown"] or ""
        cited = {(x.file_name, x.page) for x in extract_citations(md)}
        retrieved = {(x["file_name"], x["page"]) for x in r["references"]}
        if r["seconds"]:
            secs.append(r["seconds"])
        if c["answerable"]:
            text = _norm(md)
            hits = [any(_norm(alt) in text for alt in fact) for fact in c["gold_facts"]]
            facts.append(float(np.mean(hits)) if hits else 0.0)
            # Novel facts: none of their surface forms appear in the question,
            # so the answer cannot score them by echoing the query.
            question = _norm(c["symptom_description"])
            novel = [
                any(_norm(alt) in text for alt in fact)
                for fact in c["gold_facts"]
                if not any(_norm(alt) in question for alt in fact)
            ]
            if novel:
                novel_facts.append(float(np.mean(novel)))
            cite_valid.append(1.0 if cited and cited <= retrieved else 0.0)
            cite_gold.append(1.0 if cited & _gold(c) else 0.0)
            halluc_ans.append(1.0 if cited - retrieved else 0.0)
        else:
            halluc_unans.append(1.0 if _has_procedure(md) else 0.0)

    def m(v: list[float]) -> float | None:
        return float(np.mean(v)) if v else None

    return {
        "fact_recall": m(facts),
        "novel_fact_recall": m(novel_facts),
        "citation_validity": m(cite_valid),
        "citation_accuracy": m(cite_gold),
        "fabricated_citation_rate": m(halluc_ans),
        "unanswerable_hallucination_rate": m(halluc_unans),
        "median_seconds": float(np.median(secs)) if secs else None,
        "n": len(rows),
    }


def _fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:.3f}"


def _reciprocal_ranks(
    cases_list: list[dict[str, Any]], retrieval: dict[str, Any], cfg: str
) -> list[float]:
    vals: list[float] = []
    for c in cases_list:
        if not c["answerable"]:
            continue
        gold = _gold(c)
        ranked = [
            (r["metadata"]["file_name"], int(r["metadata"]["page"]))
            for r in retrieval[c["id"]][cfg]
        ]
        first = next((i for i, k in enumerate(ranked) if k in gold), None)
        vals.append(0.0 if first is None else 1.0 / (first + 1))
    return vals


def grounding_auc(
    cases_list: list[dict[str, Any]], retrieval: dict[str, Any], cfg: str
) -> float:
    """RN-04 separability: ROC-AUC of the top-1 score for telling answerable
    cases apart from unanswerable ones. 0.5 = no τ can separate them."""
    from sklearn.metrics import roc_auc_score

    y: list[int] = []
    s: list[float] = []
    for c in cases_list:
        res = retrieval[c["id"]][cfg]
        y.append(1 if c["answerable"] else 0)
        s.append(res[0]["score"] if res else -1.0)
    return float(roc_auc_score(y, s))


def _retrieval_files() -> list[Path]:
    base = OUT / "rag_retrieval.json"
    others = sorted(p for p in OUT.glob("rag_retrieval_*.json"))
    return [base, *others] if base.exists() else others


def stage_report() -> None:
    from scipy.stats import wilcoxon

    data = json.loads(CASES.read_text(encoding="utf-8"))
    cases_list = data["cases"]
    cases = {c["id"]: c for c in cases_list}
    gen_path = OUT / "rag_generation.json"
    rows: list[dict[str, Any]] = (
        json.loads(gen_path.read_text(encoding="utf-8")) if gen_path.exists() else []
    )
    docs = [json.loads(f.read_text(encoding="utf-8")) for f in _retrieval_files()]

    lines: list[str] = [
        "# RAG benchmark — RF-36 / RF-37 / RN-04",
        "",
        "> Generated by `python benchmarks/benchmark_rag.py report` — do not edit by hand.",
        "",
        f"Cases: {data['n_answerable']} answerable + {data['n_unanswerable']} unanswerable "
        f"(annotation: {data['annotation']}). Corpus: {', '.join(data['corpus'])}. "
        "Production embedder (first row): `paraphrase-multilingual-MiniLM-L12-v2`.",
        "",
        "## Retrieval (answerable cases, no LLM)",
        "",
        "| Embedder | Query | hit@1 | hit@5 | MRR | Context precision@5 | Grounding AUC¹ |",
        "|---|---|---|---|---|---|---|",
    ]
    for doc in docs:
        emb = doc["embedding_model"].split("/")[-1]
        for cfg in ("rag", "rag_xai"):
            rm = retrieval_metrics(cases_list, doc["results"], cfg)
            lines.append(
                f"| {emb} | {cfg} | {rm['hit@1']:.3f} | {rm['hit@5']:.3f} | {rm['mrr']:.3f} | "
                f"{rm['context_precision@5']:.3f} | {grounding_auc(cases_list, doc['results'], cfg):.3f} |"
            )
    lines += [
        "",
        "¹ ROC-AUC of the top-1 cosine score separating answerable from unanswerable cases "
        "(0.5 = no threshold τ can separate them).",
        "",
        "### H7 — XAI-augmented query (Wilcoxon on per-case reciprocal rank, rag_xai vs rag)",
        "",
        "| Embedder | p | improved | worse | tied |",
        "|---|---|---|---|---|",
    ]
    for doc in docs:
        a = _reciprocal_ranks(cases_list, doc["results"], "rag")
        b = _reciprocal_ranks(cases_list, doc["results"], "rag_xai")
        diff = np.array(b) - np.array(a)
        p = (
            float(wilcoxon(b, a, zero_method="zsplit").pvalue)
            if np.any(diff != 0)
            else 1.0
        )
        lines.append(
            f"| {doc['embedding_model'].split('/')[-1]} | {p:.4f} | {int((diff > 0).sum())} | "
            f"{int((diff < 0).sum())} | {int((diff == 0).sum())} |"
        )

    lines += ["", "## RN-04 — τ calibration (top-5 kept after τ)", ""]
    for doc in docs:
        lines += [
            f"### {doc['embedding_model'].split('/')[-1]}",
            "",
            "| τ | Gold coverage (rag) | False grounding (rag) | Gold coverage (rag_xai) | False grounding (rag_xai) |",
            "|---|---|---|---|---|",
        ]
        for s1, s2 in zip(
            tau_sweep(cases_list, doc["results"], "rag"),
            tau_sweep(cases_list, doc["results"], "rag_xai"),
        ):
            lines.append(
                f"| {s1['tau']:.2f} | {s1['gold_coverage']:.3f} | {s1['false_grounding']:.3f} | "
                f"{s2['gold_coverage']:.3f} | {s2['false_grounding']:.3f} |"
            )
        lines.append("")

    if rows:
        lines += [
            "",
            "## Generation ablation (RF-37)",
            "",
            "| Model | Setup | Config | Fact recall | Novel fact recall | Citation validity | Citation accuracy | Fabricated citations | Unanswerable hallucination | LLM called | Median s | Runs |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        setups = sorted({r.get("setup", "production") for r in rows})
        for model in sorted({r["model"] for r in rows}):
            for setup_tag in setups:
                for cfg in ("llm_only", "rag", "rag_xai"):
                    sub = [
                        r
                        for r in rows
                        if r["model"] == model
                        and r["config"] == cfg
                        and r.get("setup", "production") == setup_tag
                    ]
                    if not sub:
                        continue
                    g = generation_metrics(cases, sub)
                    called = [
                        (
                            1.0
                            if (r["grounding"] is None or r["grounding"]["llm_called"])
                            else 0.0
                        )
                        for r in sub
                    ]
                    lines.append(
                        f"| {model} | {setup_tag} | {cfg} | {_fmt(g['fact_recall'])} | {_fmt(g['novel_fact_recall'])} | {_fmt(g['citation_validity'])} | "
                        f"{_fmt(g['citation_accuracy'])} | {_fmt(g['fabricated_citation_rate'])} | "
                        f"{_fmt(g['unanswerable_hallucination_rate'])} | {np.mean(called):.2f} | "
                        f"{_fmt(g['median_seconds'])} | {g['n']} |"
                    )
        lines += [
            "",
            "Fact recall = share of gold facts present in the plan (any accepted surface form, accent/case-insensitive). "
            "Novel fact recall = same, counting only facts absent from the question (cannot be scored by echoing it). "
            "Citation validity = every cited file/page was retrieved. Citation accuracy = cites at least one gold page. "
            "Unanswerable hallucination = a procedure was produced for a question the corpus cannot answer.",
        ]

    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {REPORT_MD}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("queries")
    g = sub.add_parser("generate")
    g.add_argument("--models", nargs="+", default=["llama3.2:3b"])
    g.add_argument("--configs", nargs="+", default=["llm_only", "rag", "rag_xai"])
    g.add_argument("--limit", type=int, default=None)
    g.add_argument("--retrieval-file", default="rag_retrieval.json")
    g.add_argument("--mcp-cut", type=float, default=MCP_CUT)
    g.add_argument("--tau", type=float, default=0.6)
    g.add_argument("--tag", default="production")
    sub.add_parser("report")
    args = parser.parse_args()

    if args.stage == "queries":
        stage_queries()
    elif args.stage == "generate":
        setup = {
            "retrieval_file": args.retrieval_file,
            "mcp_cut": args.mcp_cut,
            "tau": args.tau,
            "tag": args.tag,
        }
        asyncio.run(stage_generate(args.models, args.configs, args.limit, setup))
    else:
        stage_report()


if __name__ == "__main__":
    main()
