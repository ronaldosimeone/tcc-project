"""
Renders docs/statistical_baselines.md: CUSUM/EWMA under the nested protocol
and the RN-05 gate in both directions (does any ML/DL model beat the
statistical baseline? does the baseline beat the champion?).

Entry point:
    python -m src.render_baselines_report
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluate_nested import ALL_MODELS
from src.promotion_gate import decide
from src.statistical_baselines import BASELINES

_ML_ROOT: Path = Path(__file__).resolve().parent.parent
REPO_ROOT: Path = _ML_ROOT.parent.parent
NESTED_DIR: Path = _ML_ROOT / "models" / "nested"
OUT_MD: Path = REPO_ROOT / "docs" / "statistical_baselines.md"
CHAMPION: str = "random_forest"


def _load(names: list[str]) -> dict[str, Any]:
    out = {}
    for name in names:
        p = NESTED_DIR / f"{name}.json"
        if p.exists():
            out[name] = json.loads(p.read_text(encoding="utf-8"))
    return out


def _fold_costs(m: dict[str, Any]) -> dict[int, float]:
    by_fold: dict[int, list[float]] = {}
    for r in m["runs"]:
        by_fold.setdefault(r["fold"], []).append(r["events"]["expected_cost"])
    return {f: float(np.mean(c)) for f, c in by_fold.items()}


def _gate(models: dict[str, Any], cand: str, base: str) -> dict[str, Any]:
    runs = [
        {
            "model": name,
            "seed": 0,
            "fold": fold,
            "event_inner": {"expected_cost": cost},
            "point": {"pr_auc": 0.0},
        }
        for name in (cand, base)
        for fold, cost in _fold_costs(models[name]).items()
    ]
    ev = {"protocols": {"nested": {"runs": runs}}}
    return decide(ev, cand, base, protocol="nested").to_dict()


def _row(name: str, m: dict[str, Any]) -> str:
    s = m["summary"]
    fa_day = np.mean([r["events"]["false_alarms_per_day"] for r in m["runs"]])
    return (
        f"| `{name}` | {s['detected_mean']:.1f}/6 | "
        f"{s['false_alarm_episodes_mean']:.1f} | {fa_day:.2f} | "
        f"{s['expected_cost_mean']:.1f} |"
    )


def _params(p: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in p.items())


def render(models: dict[str, Any]) -> str:
    base = [b for b in BASELINES if b in models]
    ml = [n for n in ALL_MODELS if n in models]
    lines = [
        "# Baselines estatísticos (CUSUM / EWMA) no protocolo aninhado",
        "",
        "> Gerado por `python -m src.render_baselines_report` a partir de "
        "`apps/ml/models/nested/*.json`. Não edite à mão.",
        "",
        "## Pergunta",
        "",
        "A complexidade dos modelos de ML/DL se paga com 7 falhas? O benchmark "
        "público do MetroPT-3 recomenda um CUSUM na corrente do motor + z-score "
        "da temperatura do óleo (4/4 falhas, 0,71 FA/semana), mas escolheu os "
        "detectores olhando as mesmas 4 falhas que testou.",
        "",
        "## Protocolo (o mesmo dos 8 modelos)",
        "",
        "* Walk-forward purgado externo (6 folds fora do tempo, MetroPT-3 + 2022).",
        "* Tudo o que é livre é escolhido **só no treino**, por leave-one-failure-out "
        "interno e custo RN-02: sensor (corrente do motor, temperatura do óleo, "
        "TP2, ou corrente + óleo), `k` do CUSUM ∈ {0,5; 1,0} ou `λ` do EWMA ∈ "
        "{0,01; 0,1}, referência (estática = média/desvio do treino saudável; "
        "móvel = 7 dias anteriores, causal, com 1 h de defasagem), e o limiar "
        "`h` (Platt + ótimo de custo nos OOF internos).",
        "* Sinal = média móvel causal de 10 min dentro de cada segmento contínuo "
        "(lacuna > 60 s reinicia), estatística reiniciada em cada segmento.",
        "* Fold 1 tem só 1 falha no treino: limiar a priori (percentil 99 da "
        "estatística no treino saudável), parâmetros a priori = escolha do "
        "benchmark (corrente + óleo, k = 0,5 / λ = 0,1, referência móvel). "
        "Ressalva: essa escolha a priori veio de um trabalho que viu as falhas "
        "do MetroPT-3, inclusive a do fold 1; os folds 2–6 não dependem dela.",
        "* Determinísticos: 1 seed. Espaço de busca fixado antes de rodar.",
        "",
        "## Resultado (rótulos originais)",
        "",
        "| Detector / modelo | Falhas detectadas | Alarmes falsos (total) | FA/dia (média dos folds) | Custo esperado médio |",
        "|---|---|---|---|---|",
        *[_row(n, models[n]) for n in base],
        *[_row(n, models[n]) for n in ml],
        "",
        "## Por fold",
        "",
        "| Fold | Teste | "
        + " | ".join(f"`{b}` det / FA / custo" for b in base)
        + f" | `{CHAMPION}` custo |",
        "|---|---|" + "---|" * len(base) + "---|",
    ]
    champ = _fold_costs(models[CHAMPION]) if CHAMPION in models else {}
    folds = sorted({r["fold"] for b in base for r in models[b]["runs"]})
    for f in folds:
        cells = []
        test = ""
        for b in base:
            r = next(r for r in models[b]["runs"] if r["fold"] == f)
            test = r["test_dataset"]
            e = r["events"]
            cells.append(
                f"{e['detected']}/{e['failures']} / {e['false_alarm_episodes']} / "
                f"{e['expected_cost']:.0f}"
            )
        lines.append(
            f"| {f} | {test} | "
            + " | ".join(cells)
            + f" | {champ.get(f, float('nan')):.0f} |"
        )
    lines += ["", "## Parâmetros escolhidos no loop interno", ""]
    for b in base:
        lines += [
            f"**`{b}`**",
            "",
            "| Fold | Parâmetros | Origem do limiar |",
            "|---|---|---|",
        ]
        for r in models[b]["runs"]:
            lines.append(
                f"| {r['fold']} | {_params(r['params'])} | {r['threshold_source']} |"
            )
        lines.append("")
    lines += [
        "## Gate RN-05",
        "",
        "Wilcoxon unilateral exato, pareado por fold (6 blocos → menor p possível "
        "1/64 ≈ 0,016), custo esperado RN-02, α = 0,05.",
        "",
        "| Candidato | Baseline | Melhora mediana do custo | p | Promove? |",
        "|---|---|---|---|---|",
    ]
    pairs = [(b, CHAMPION) for b in base if CHAMPION in models]
    pairs += [(n, b) for b in base for n in ml]
    for c, bl in pairs:
        g = _gate(models, c, bl)
        lines.append(
            f"| `{c}` | `{bl}` | {g['median_improvement']:+.1f} | "
            f"{g['p_value']:.3f} | {'sim' if g['promote'] else 'não'} |"
        )
    lines += ["", *_without_fold1(models, [*base, *ml])]
    return "\n".join(lines)


def _without_fold1(models: dict[str, Any], names: list[str]) -> list[str]:
    """Descriptive sensitivity: fold 1 reported apart (its baseline
    parameters came from a benchmark that saw the MetroPT-3 failures)."""
    lines = [
        "## Análise secundária descritiva — fold 1 separado",
        "",
        "**Não é um novo resultado principal.** O resultado acima (6 folds) "
        "continua sendo o oficial. Esta separação foi motivada depois de ver os "
        "resultados, por isso os folds 2–6 **não** são um novo teste "
        "independente, e nenhum gate é recalculado aqui. Ela só mostra se a "
        "ordem dos métodos depende do fold 1, cujos parâmetros a priori do "
        "CUSUM/EWMA vêm de um benchmark que viu as falhas do MetroPT-3. "
        'Parâmetros "fisicamente justificados" escolhidos agora não foram '
        "testados: seriam escolhidos já conhecendo os resultados.",
        "",
        "| Detector / modelo | Fold 1: falha / FA / custo | Folds 2–6: falhas | Folds 2–6: FA | Folds 2–6: FA/dia (média) | Folds 2–6: custo médio |",
        "|---|---|---|---|---|---|",
    ]
    rows = []
    for n in names:
        by_fold: dict[int, list[dict[str, Any]]] = {}
        for r in models[n]["runs"]:
            by_fold.setdefault(r["fold"], []).append(r["events"])

        def mean(f: int, k: str) -> float:
            return float(np.mean([e[k] for e in by_fold[f]]))

        rest = [f for f in sorted(by_fold) if f != 1]
        rows.append(
            (
                np.mean([mean(f, "expected_cost") for f in rest]),
                f"| `{n}` | {mean(1, 'detected'):.0f} / "
                f"{mean(1, 'false_alarm_episodes'):.0f} / "
                f"{mean(1, 'expected_cost'):.0f} | "
                f"{sum(mean(f, 'detected') for f in rest):.1f}/{len(rest)} | "
                f"{sum(mean(f, 'false_alarm_episodes') for f in rest):.0f} | "
                f"{np.mean([mean(f, 'false_alarms_per_day') for f in rest]):.2f} | "
                f"{np.mean([mean(f, 'expected_cost') for f in rest]):.1f} |",
            )
        )
    lines += [r for _, r in sorted(rows, key=lambda x: x[0])]
    lines += [
        "",
        "Leitura: menos alarmes falsos não basta para dizer que um método é "
        "melhor; o custo junta falhas perdidas e alarmes. Nos folds 2–6 as "
        "árvores e os melhores modelos sequenciais (XGBoost, RF, BiLSTM, "
        "PatchTST) quase não alarmam fora das falhas, e a maior parte dos "
        "alarmes falsos deles fica no fold 1 (maio de 2020), onde estão as "
        "anomalias não confirmadas da [auditoria](label_audit.md). Essa "
        "concentração é compatível com falhas não documentadas, mas também com "
        "mudança operacional ou de instrumentação; não comprova nenhuma delas.",
        "",
    ]
    return lines


def main() -> None:
    models = _load([*ALL_MODELS, *BASELINES])
    OUT_MD.write_text(render(models), encoding="utf-8")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
