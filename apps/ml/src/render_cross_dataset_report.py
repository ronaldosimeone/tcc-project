"""
Renders docs/cross_dataset_evaluation.md from models/eval_cross_dataset.json
(RF-33). No training — only formatting.

Entry point:
    python -m src.render_cross_dataset_report
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

_ML_ROOT: Path = Path(__file__).resolve().parent.parent
REPO_ROOT: Path = _ML_ROOT.parent.parent
IN_JSON: Path = _ML_ROOT / "models" / "eval_cross_dataset.json"
INGEST_JSON: Path = _ML_ROOT / "models" / "metropt2022_ingest.json"
HARMONIZED_JSON: Path = _ML_ROOT / "models" / "eval_cross_dataset_harmonized.json"
OUT_MD: Path = REPO_ROOT / "docs" / "cross_dataset_evaluation.md"

MODEL_LABEL: dict[str, str] = {
    "random_forest": "RF",
    "xgboost": "XGBoost",
    "logistic_regression": "LogReg",
}


def _ci(d: dict[str, Any], digits: int = 2) -> str:
    return f"{d['mean']:.{digits}f} [{d['low']:.{digits}f}; {d['high']:.{digits}f}]"


def _shift(data: dict[str, Any]) -> list[str]:
    s = data["shift"]
    adv = s["adversarial_validation"]
    lines = [
        "### Q1 — Quão diferentes são os dois períodos?",
        "",
        f"**Validação adversarial**: um RF treinado para distinguir linhas *normais* de "
        f"2020 e de 2022 atinge ROC-AUC **{adv['auc_mean']:.3f}** em folds bloqueados "
        "no tempo (0,5 = indistinguíveis). Sensores que mais separam os períodos: "
        + ", ".join(
            f"`{f['feature']}` ({f['importance']:.2f})" for f in adv["top_features"]
        )
        + ".",
        "",
        "| Sensor | média ± dp 2020 | média ± dp 2022 | PSI | KS |",
        "|---|---|---|---|---|",
    ]
    for col, v in s["per_sensor_normal_rows"].items():
        psi = "—" if v["psi"] is None else f"{v['psi']:.2f}"
        lines.append(
            f"| `{col}` | {v['mean_2020']:.2f} ± {v['std_2020']:.2f} | "
            f"{v['mean_2022']:.2f} ± {v['std_2022']:.2f} | {psi} | {v['ks']:.2f} |"
        )
    return lines + [""]


def _cross(data: dict[str, Any]) -> list[str]:
    lines = [
        "### Q2 — Um modelo de um período funciona no outro?",
        "",
        "Treino no período inteiro de origem, teste no período inteiro de destino; "
        "limiar escolhido **só** no treino (OOF interno, RN-02). Média [IC 95 %] "
        "sobre 3 seeds.",
        "",
        "| Direção | Modelo | PR-AUC | Recall de evento | FA/dia | Custo esperado |",
        "|---|---|---|---|---|---|",
    ]
    for direction, block in data["cross"].items():
        label = direction.replace("_to_", " → ")
        for name, m in block["per_model"].items():
            lines.append(
                f"| {label} | {MODEL_LABEL[name]} | {_ci(m['pr_auc'], 3)} | "
                f"{_ci(m['event_recall_inner'])} | "
                f"{_ci(m['false_alarms_per_day_inner'])} | "
                f"{_ci(m['expected_cost_inner'], 1)} |"
            )
    return lines + [""]


def _pooled(data: dict[str, Any]) -> list[str]:
    p = data["pooled_walk_forward"]
    pm = p["summary"]["per_model"]
    lines = [
        "### Q3 — Vale a pena somar os dois?",
        "",
        f"Walk-forward purgado na linha do tempo conjunta: 7 falhas → "
        f"**{p['n_folds']} folds** de teste. Cada fold treina só com o passado.",
        "",
        "| Modelo | PR-AUC | Recall de evento | FA/dia | Custo esperado |",
        "|---|---|---|---|---|",
    ]
    for name, m in pm.items():
        lines.append(
            f"| {MODEL_LABEL[name]} | {_ci(m['pr_auc'], 3)} | "
            f"{_ci(m['event_recall_inner'])} | "
            f"{_ci(m['false_alarms_per_day_inner'])} | "
            f"{_ci(m['expected_cost_inner'], 1)} |"
        )
    lines += [
        "",
        "Custo esperado por período de teste:",
        "",
        "| Modelo | folds de 2020 | folds de 2022 |",
        "|---|---|---|",
    ]
    for name in pm:
        a = p["per_test_dataset"]["2020"][name]
        b = p["per_test_dataset"]["2022"][name]
        lines.append(f"| {MODEL_LABEL[name]} | {_ci(a, 1)} | {_ci(b, 1)} |")

    if p.get("pooled_vs_target_only"):
        lines += [
            "",
            "**Conjunto × só 2022** — mesmos folds de 2022 que já têm uma falha de "
            "2022 no passado, treinados (a) com 2020 + 2022 e (b) só com 2022:",
            "",
            "| Modelo | folds | Custo conjunto | Custo só 2022 | PR-AUC conjunto | "
            "PR-AUC só 2022 | Wilcoxon p (custo) |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, v in p["pooled_vs_target_only"].items():
            lines.append(
                f"| {MODEL_LABEL[name]} | {v['folds']} | {v['pooled_cost_mean']:.1f} | "
                f"{v['target_only_cost_mean']:.1f} | {v['pooled_pr_auc_mean']:.3f} | "
                f"{v['target_only_pr_auc_mean']:.3f} | "
                f"{v['wilcoxon_cost']['p_value']:.3f} |"
            )

    lines += [
        "",
        "**Gate RN-05** (Wilcoxon unilateral pareado por fold, custo esperado):",
        "",
        "| Candidato × base | folds | melhora mediana | p | p mínimo atingível | "
        "promove? |",
        "|---|---|---|---|---|---|",
    ]
    for key, g in p["promotion_gate_rn05"].items():
        lines.append(
            f"| {MODEL_LABEL[g['candidate']]} × {MODEL_LABEL[g['baseline']]} | "
            f"{g['n_blocks']} | {g['median_improvement']:.1f} | {g['p_value']:.3f} | "
            f"{g['min_achievable_p']:.3f} | {'sim' if g['promote'] else 'não'} |"
        )
    fr = p["summary"]["friedman"].get("expected_cost_inner", {})
    if "p_value" in fr:
        lines += [
            "",
            f"Friedman (custo esperado, {p['n_folds']} folds): p = {fr['p_value']:.3f}.",
        ]
    return lines + [""]


def _flowmeter(data: dict[str, Any]) -> list[str]:
    f = data["flowmeter_ablation"]
    lines = [
        "### Q4 — Quanto custa descartar o `Flowmeter`?",
        "",
        f"Só 2022 (o MetroPT-3 não tem medidor de vazão), {f['protocol']}. "
        "Diferença pareada por fold×seed, com − sem (IC 95 %):",
        "",
        "Leitura: só com o 2022 (no máximo 2 falhas de treino por fold) nenhum "
        "modelo detecta a falha de teste, com ou sem o sensor (recall 0). A "
        "ablação fica **inconclusiva por efeito piso** e não prova que o "
        "`Flowmeter` seja inútil.",
        "",
        "| Modelo | Métrica | com | sem | Δ (com − sem) |",
        "|---|---|---|---|---|",
    ]
    for name, metrics in f["deltas"].items():
        for metric, v in metrics.items():
            lines.append(
                f"| {MODEL_LABEL[name]} | {metric} | {v['with']:.3f} | "
                f"{v['without']:.3f} | {_ci(v['delta_with_minus_without'], 3)} |"
            )
    return lines + [""]


SEMANTIC_NOTE: list[str] = [
    "## Harmonização semântica — mesmo nome, outro sinal",
    "",
    "O bloco Q1 mostrou que casar as colunas pelo nome não basta:",
    "",
    "* **`Reservoirs`** — no MetroPT-3 é uma cópia do `TP3` (correlação 1,0000, "
    "diferença máxima 0,18 bar); no 2022 é outra grandeza (≈ 1,6 bar, correlação "
    "−0,001 com o `TP3`).",
    "* **`Oil_level`** — no MetroPT-3 vale 1 em 90 % das linhas normais e em 100 % "
    "das de falha; no 2022 é constante 0.",
    "",
    "Os dois respondem pela maior parte da separação adversarial. A **versão "
    "harmonizada** abaixo repete Q1–Q3 sem esses dois sensores "
    "(`--drop-sensors Reservoirs Oil_level`). A linha de base com as 12 colunas "
    "fica acima, e a diferença entre as duas versões mede o custo de uma "
    "harmonização ingênua.",
    "",
]

SECTIONS: tuple[tuple[str, Callable[[dict[str, Any]], list[str]]], ...] = (
    ("shift", _shift),
    ("cross", _cross),
    ("pooled_walk_forward", _pooled),
    ("flowmeter_ablation", _flowmeter),
)


def _sections(data: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for key, fn in SECTIONS:
        if key in data:
            out += fn(data)
    return out


def render(
    data: dict[str, Any],
    ingest: dict[str, Any],
    harmonized: dict[str, Any] | None = None,
) -> str:
    fails = ", ".join(
        f"{x['component']} ({x['start']} → {x['end']}, {x['rows_10s']} linhas)"
        for x in ingest["failures"]
    )
    head = [
        "# Avaliação cross-dataset — RF-33",
        "",
        "> Gerado por `python -m src.render_cross_dataset_report` a partir de "
        "`apps/ml/models/eval_cross_dataset.json`. Não edite à mão.",
        "",
        "**Dados.** MetroPT-3 (fev–set/2020, 4 falhas) + MetroPT 2022 "
        "(Veloso et al., *Scientific Data* 9:764, 2022; Zenodo "
        "10.5281/zenodo.6854240, CC BY 4.0). O 2022 foi harmonizado: as 4 colunas "
        "de GPS foram descartadas, o `Flowmeter` ficou só para a ablação, e a "
        f"amostragem pontual a 10 s levou {ingest['rows_raw_1hz']:,} linhas a 1 Hz a "
        f"{ingest['rows_10s']:,} linhas. Falhas de 2022 (Tabela 2 do artigo): {fails}.",
        "",
    ]
    body = ["## Linha de base — 12 sensores casados pelo nome", ""]
    body += _sections(data) + SEMANTIC_NOTE
    if harmonized is not None:
        dropped = ", ".join(f"`{c}`" for c in harmonized["config"]["dropped_sensors"])
        body += [f"## Versão harmonizada — sem {dropped}", ""]
        body += _sections(harmonized)
    return "\n".join(head + body)


def main() -> None:
    data = json.loads(IN_JSON.read_text(encoding="utf-8"))
    ingest = json.loads(INGEST_JSON.read_text(encoding="utf-8"))
    harmonized = (
        json.loads(HARMONIZED_JSON.read_text(encoding="utf-8"))
        if HARMONIZED_JSON.exists()
        else None
    )
    OUT_MD.write_text(render(data, ingest, harmonized) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
