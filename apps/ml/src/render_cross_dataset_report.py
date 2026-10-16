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


PRODUCTION_JSON: Path = _ML_ROOT / "models" / "eval_production_2022.json"
TWIN_JSON: Path = _ML_ROOT / "models" / "eval_twin_cross.json"
CHALLENGER_CARD: Path = _ML_ROOT / "models" / "xgboost_v3_card.json"

TWIN_LABEL: dict[str, str] = {
    "ml_2020_only": "ML (XGBoost) só 2020",
    "ml_only": "ML (XGBoost) 2020 + semanas normais 2022",
    "hybrid": "Híbrido (features + resíduos físicos)",
    "physics_only": "Física, twin calibrado por período",
    "physics_uncalibrated": "Física, twin de 2020 sem calibrar",
    "fusion": "Fusão (ML ou física)",
}


POOLED_LABEL: dict[str, str] = {
    "ml_only": "ML (XGBoost)",
    "physics_only": "Física (twin calibrado por período)",
    "hybrid": "Híbrido (features + resíduos físicos)",
    "fusion": "Fusão (ML ou física)",
}


def _production(p: dict[str, Any]) -> list[str]:
    lines = [
        "## Q5 — E o modelo que está em produção?",
        "",
        f"`{p['model']}` foi treinado com {p['model_training']['split']} e "
        f"{p['model_training']['class_balancing']}. Avaliado **sem retreino**:",
        "",
        "| Entrada | PR-AUC | Falhas detectadas (limiar do modelo) | FA/dia |",
        "|---|---|---|---|",
    ]
    for key, label in (
        ("in_sample_2020_contaminated", "2020 — dados de treino (contaminado)"),
        ("as_deployed", "2022 — como implantado"),
        ("harmonized_at_input", "2022 — `Reservoirs`:=`TP3`, `Oil_level`:=moda"),
    ):
        r = p[key]
        e = r["events_at_model_threshold"]
        hits = sum(e["detected"].values())
        lines.append(
            f"| {label} | {r['pr_auc']:.3f} | {hits}/{len(e['detected'])} | "
            f"{e['false_alarms_per_day']:.2f} |"
        )
    lines += [
        "",
        "Perfeito nos dados que viu e cego num período novo: é a assinatura de "
        "sobreajuste do split aleatório + SMOTE. Corrigir os sensores na entrada "
        "não resolve, então o problema está no treino e não na harmonização.",
        "",
    ]
    return lines


def _twin(t: dict[str, Any]) -> list[str]:
    lines = ["## Q6 — A física generaliza onde o ML não generaliza?", ""]
    tr = t.get("transfer")
    if tr:
        lines += [
            "**Transferência sem rótulo de falha de 2022.** Treino = 2020 + "
            f"semanas normais de 2022 ({tr['calibration_window'][0]} → "
            f"{tr['calibration_window'][1]}); teste = resto de 2022.",
            "",
            "| Variante | Recall de evento | FA/dia | Custo | Detecção por falha |",
            "|---|---|---|---|---|",
        ]
        for v, s in tr["summary"].items():
            det = " · ".join(
                f"{k[:10]}: {x:.0%}" for k, x in s["detected_per_event"].items()
            )
            lines.append(
                f"| {TWIN_LABEL.get(v, v)} | {_ci(s['event_recall'])} | "
                f"{_ci(s['false_alarms_per_day'])} | {_ci(s['expected_cost'], 1)} | {det} |"
            )
        lines += ["", "Resíduos físicos (z médio) dentro de cada falha de 2022:", ""]
        for ev, z in tr["physics_z_in_2022_failures"].items():
            lines.append(
                f"* {ev}: " + ", ".join(f"`{k}` {v:+.1f}" for k, v in z.items())
            )
        lines.append("")
    pw = t.get("pooled_walk_forward")
    if pw:
        lines += [
            f"**Walk-forward conjunto ({pw['n_folds']} folds).** Twin calibrado por "
            "período só com as linhas normais do treino de cada fold.",
            "",
            "| Variante | Recall (todos) | Custo (todos) | Recall (folds 2022) | "
            "Custo (folds 2022) |",
            "|---|---|---|---|---|",
        ]
        for v, s in pw["summary"].items():
            s22 = pw["summary_2022_folds"].get(v)
            lines.append(
                f"| {POOLED_LABEL.get(v, v)} | {_ci(s['event_recall'])} | "
                f"{_ci(s['expected_cost'], 1)} | "
                + (
                    f"{_ci(s22['event_recall'])} | {_ci(s22['expected_cost'], 1)} |"
                    if s22
                    else "— | — |"
                )
            )
        lines += ["", "Gate RN-05 (custo esperado, pareado por fold):", ""]
        for key, g in pw["promotion_gate_rn05"].items():
            lines.append(
                f"* {key.replace('_vs_', ' × ')}: melhora mediana "
                f"{g['median_improvement']:+.1f}, p = {g['p_value']:.3f} → "
                f"{'promove' if g['promote'] else 'não promove'}"
            )
        lines.append("")
    return lines


def _challenger(c: dict[str, Any]) -> list[str]:
    g = c["evidence"]["rn05_gate_vs_random_forest"]
    return [
        "## Q7 — O que fica em produção",
        "",
        f"O candidato `{c['model_name']}` foi treinado com 2020 + 2022 "
        f"({c['feature_count']} features, sem {', '.join(c['dropped_sensors'])}). O "
        f"balanceamento foi feito só no treino ({c['class_balancing']}), e o limiar "
        f"{c['decision_threshold']} saiu de {c['threshold_strategy']}. "
        f"Contra o RF no walk-forward conjunto, o gate RN-05 deu p = {g['p_value']:.3f}, "
        "e por isso o modelo **não é promovido**. Ele foi registrado no MLflow como "
        "`challenger` (shadow + canary, `CHALLENGER_MODEL=xgboost_v3`), e o "
        "`random_forest_v2` continua `champion` até haver evidência significativa.",
        "",
    ]


def main() -> None:
    data = json.loads(IN_JSON.read_text(encoding="utf-8"))
    ingest = json.loads(INGEST_JSON.read_text(encoding="utf-8"))
    harmonized = (
        json.loads(HARMONIZED_JSON.read_text(encoding="utf-8"))
        if HARMONIZED_JSON.exists()
        else None
    )
    text = render(data, ingest, harmonized)
    for path, fn in (
        (PRODUCTION_JSON, _production),
        (TWIN_JSON, _twin),
        (CHALLENGER_CARD, _challenger),
    ):
        if path.exists():
            text += "\n" + "\n".join(fn(json.loads(path.read_text(encoding="utf-8"))))
    OUT_MD.write_text(text + "\n", encoding="utf-8")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
