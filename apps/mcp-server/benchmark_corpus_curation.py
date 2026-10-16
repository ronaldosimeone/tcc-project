"""
Benchmark de curadoria do corpus RAG — insumo do RF-37 (ablação).

Pergunta: adicionar um manual real mas fora do domínio exato (Scheduled
Maintenance Manual da locomotiva WAP-5) melhora ou piora a recuperação para
as falhas do compressor do MetroPT-3? E a curadoria via sidecar (só a seção
6.6, sem remissões "Refer to ...") muda o resultado?

Três corpora, cada um indexado num vector store TEMPORÁRIO (nunca o de dev):
    A  baseline  — manuais atuais de `MANUALS_DIR`, sem o WAP-5
    B  raw       — A + WAP-5 inteiro (224 páginas, sem sidecar)
    C  curated   — A + WAP-5 com sidecar (p. 170-177, remissões removidas)

Para cada consulta de falha, registra os top-5 (arquivo, página, score) e
quantos resultados são "remissão pura" ou vêm de páginas fora da seção de ar
comprimido. Sem LLM, sem rede além do download do modelo de embeddings.

Uso (dentro da imagem do mcp-server):
    python benchmark_corpus_curation.py --wap5 /path/VOL-B.pdf
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import index_manuals as im
from semantic_search import SemanticSearchService

QUERIES: list[str] = [
    "Compressor de ar: vazamento de ar e queda de pressão no reservatório",
    "Compressor de ar: temperatura do óleo elevada",
    "Compressor de ar: nível de óleo baixo",
    "Compressor de ar: corrente do motor acima do normal",
    "Secador de ar: dessecante das torres contaminado, umidade no ar comprimido",
    "Air compressor: air leak, main reservoir pressure drop",
    "Air dryer twin tower desiccant contaminated with oil or water",
    "Compressor safety valve and drain valve test",
]

WAP5_SECTION: tuple[int, int] = (170, 177)
WAP5_PROFILE: dict[str, Any] = {
    "pages": list(WAP5_SECTION),
    "strip_cross_references": True,
    "min_chunk_chars": 120,
    "metadata": {
        "domain": "locomotive",
        "doc_type": "scheduled_maintenance",
        "equipment": "air_supply_pneumatic",
        "source_title": "Indian Railways WAP-5 Scheduled Maintenance Manual Vol. B, sec. 6.6",
    },
}


def _build_corpus(
    root: Path,
    base_manuals: list[Path],
    wap5: Path | None,
    profile: dict[str, Any] | None,
) -> Path:
    manuals = root / "manuals"
    manuals.mkdir(parents=True)
    for pdf in base_manuals:
        shutil.copy2(pdf, manuals / pdf.name)
    if wap5 is not None:
        target = manuals / "wap5-scheduled-maintenance-vol-b.pdf"
        shutil.copy2(wap5, target)
        if profile is not None:
            im.profile_path_for(target).write_text(
                json.dumps(profile), encoding="utf-8"
            )
    return manuals


def _is_reference_only(text: str) -> bool:
    cleaned = " ".join(im.strip_cross_references(text).split())
    return im._useful_chars(cleaned) < 120


def run(wap5: Path, base_dir: Path) -> dict[str, Any]:
    base_manuals = sorted(p for p in base_dir.glob("*.pdf"))
    model = im.load_embedding_model(im.EMBEDDING_MODEL)
    corpora: dict[str, tuple[Path | None, dict[str, Any] | None]] = {
        "A_baseline": (None, None),
        "B_wap5_raw": (wap5, None),
        "C_wap5_curated": (wap5, WAP5_PROFILE),
    }
    out: dict[str, Any] = {
        "embedding_model": im.EMBEDDING_MODEL,
        "chunk_size": im.CHUNK_SIZE,
        "chunk_overlap": im.CHUNK_OVERLAP,
        "base_manuals": [p.name for p in base_manuals],
        "wap5_profile": WAP5_PROFILE,
        "corpora": {},
    }

    for name, (pdf, profile) in corpora.items():
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manuals = _build_corpus(root, base_manuals, pdf, profile)
            stats = im.run_indexing(
                manuals,
                root / "store",
                im.EMBEDDING_MODEL,
                im.CHUNK_SIZE,
                im.CHUNK_OVERLAP,
            )
            collection = im.get_collection(root / "store")
            service = SemanticSearchService(collection, model)
            # Mesmo ranking, sem o corte `score > 0.6` da RF-21: separa
            # "o ranking está errado" de "o limiar está alto demais".
            unfiltered = SemanticSearchService(collection, model, score_threshold=-1.0)

            per_query: list[dict[str, Any]] = []
            for q in QUERIES:
                results = service.search(q)
                rows = []
                for r in results:
                    page = int(r.metadata.get("page", 0))
                    is_wap5 = r.metadata.get("file_name", "").startswith("wap5")
                    rows.append(
                        {
                            "file": r.metadata.get("file_name"),
                            "page": page,
                            "score": round(r.score, 4),
                            "wap5_off_section": bool(
                                is_wap5
                                and not (WAP5_SECTION[0] <= page <= WAP5_SECTION[1])
                            ),
                            "reference_only": _is_reference_only(r.text),
                            "preview": " ".join(r.text.split())[:160],
                        }
                    )
                top_unfiltered = [
                    {
                        "file": r.metadata.get("file_name"),
                        "page": int(r.metadata.get("page", 0)),
                        "score": round(r.score, 4),
                        "preview": " ".join(r.text.split())[:120],
                    }
                    for r in unfiltered.search(q)
                ]
                per_query.append(
                    {
                        "query": q,
                        "n_results": len(rows),
                        "results": rows,
                        "top5_no_threshold": top_unfiltered,
                    }
                )

            flat = [row for pq in per_query for row in pq["results"]]
            out["corpora"][name] = {
                "n_chunks": collection.count(),
                "indexing": stats.summary(),
                "summary": {
                    "queries_with_zero_results": sum(
                        1 for pq in per_query if pq["n_results"] == 0
                    ),
                    "total_results": len(flat),
                    "wap5_results": sum(
                        1 for r in flat if str(r["file"]).startswith("wap5")
                    ),
                    "wap5_off_section_results": sum(
                        1 for r in flat if r["wap5_off_section"]
                    ),
                    "reference_only_results": sum(
                        1 for r in flat if r["reference_only"]
                    ),
                    "mean_top1_score_no_threshold": round(
                        sum(pq["top5_no_threshold"][0]["score"] for pq in per_query)
                        / len(per_query),
                        4,
                    ),
                    "mean_top1_score": round(
                        sum(
                            pq["results"][0]["score"]
                            for pq in per_query
                            if pq["results"]
                        )
                        / max(sum(1 for pq in per_query if pq["results"]), 1),
                        4,
                    ),
                },
                "queries": per_query,
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wap5", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, default=im.MANUALS_DIR)
    parser.add_argument(
        "--out", type=Path, default=Path("corpus_curation_benchmark.json")
    )
    args = parser.parse_args()

    result = run(args.wap5, args.base_dir)
    args.out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for name, data in result["corpora"].items():
        print(name, data["n_chunks"], json.dumps(data["summary"]))


if __name__ == "__main__":
    main()
