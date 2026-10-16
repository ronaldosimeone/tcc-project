"""
Renders docs/model_evaluation_leakfree.md from models/eval_all_models.json —
confusion matrices of every servable model under the leakage-free protocol.

Entry point:
    python -m src.render_all_models_report
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_ML_ROOT: Path = Path(__file__).resolve().parent.parent
REPO_ROOT: Path = _ML_ROOT.parent.parent
IN_JSON: Path = _ML_ROOT / "models" / "eval_all_models.json"
OUT_MD: Path = REPO_ROOT / "docs" / "model_evaluation_leakfree.md"

LABEL: dict[str, str] = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "logistic_regression": "LogReg (baseline)",
    "mlp": "MLP",
    "tcn": "TCN",
    "bilstm": "BiLSTM",
    "patchtst": "PatchTST",
    "autoencoder": "Autoencoder",
}
ARTEFACT: dict[str, str] = {
    "random_forest": "random_forest_v3",
    "xgboost": "xgboost_v3",
    "logistic_regression": "—",
    "mlp": "mlp_v2",
    "tcn": "tcn_v2",
    "bilstm": "bilstm_v2",
    "patchtst": "patchtst_v2",
    "autoencoder": "autoencoder_v2",
}


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x:.1%}"


def _n(x: int) -> str:
    return f"{x:,}".replace(",", ".")


def render(data: dict[str, Any]) -> str:
    models: dict[str, Any] = data["models"]
    lines = [
        "# Avaliação sem vazamento de todos os modelos — matrizes de confusão",
        "",
        "> Gerado por `python -m src.render_all_models_report` a partir de "
        "`apps/ml/models/eval_all_models.json`. Não edite à mão.",
        "",
        "**Protocolo.** MetroPT-3 (2020) + MetroPT 2022, sensores harmonizados (sem "
        "`Reservoirs`/`Oil_level`), walk-forward purgado por evento: 7 falhas → "
        "**6 folds fora do tempo**. Cada fold treina só com o passado, escolhe o "
        "limiar só com dados de treino e é testado numa falha que o modelo nunca "
        "viu. Balanceamento de classes **apenas no treino**; o teste mantém a "
        "proporção real (≈ 2 % de amostras de falha). Árvores: limiar por custo "
        "em predições out-of-fold; redes: limiar por custo na falha de treino "
        f"mais recente (validação temporal). Seed {data['seed']}.",
        "",
        "As matrizes somam os 6 folds de teste. A matriz do modelo final medida "
        "nos próprios dados de treino seria vazamento e não é reportada.",
        "",
        "## Por evento — o que importa para a manutenção",
        "",
        "| Modelo | Artefato | Falhas detectadas | Falhas perdidas | Episódios de "
        "falso alarme | Custo esperado médio |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in models.items():
        ev = m["total"]["events"]
        lines.append(
            f"| {LABEL.get(name, name)} | `{ARTEFACT.get(name, '—')}` | "
            f"**{ev['detected']}/{ev['failures']}** | {ev['missed']} | "
            f"{ev['false_alarm_episodes']} | {m['total']['expected_cost_mean']:.1f} |"
        )
    lines += [
        "",
        "## Por amostra, após a persistência k = 6 (alarme visto pelo operador)",
        "",
        "![Matrizes de confusão — alarmes](figures-ml/confusion_alarms.png)",
        "",
        "| Modelo | TP | FP | FN | TN | Precisão | Recall | Especificidade | F1 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in models.items():
        cm, mt = m["total"]["alarms"], m["total"]["metrics_alarms"]
        lines.append(
            f"| {LABEL.get(name, name)} | {_n(cm['tp'])} | {_n(cm['fp'])} | "
            f"{_n(cm['fn'])} | {_n(cm['tn'])} | {_pct(mt['precision'])} | "
            f"{_pct(mt['recall'])} | {_pct(mt['specificity'])} | {_pct(mt['f1'])} |"
        )
    lines += [
        "",
        "## Por amostra, no limiar (sem persistência)",
        "",
        "![Matrizes de confusão — amostras](figures-ml/confusion_samples.png)",
        "",
        "| Modelo | TP | FP | FN | TN | Precisão | Recall | F1 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, m in models.items():
        cm, mt = m["total"]["samples"], m["total"]["metrics_samples"]
        lines.append(
            f"| {LABEL.get(name, name)} | {_n(cm['tp'])} | {_n(cm['fp'])} | "
            f"{_n(cm['fn'])} | {_n(cm['tn'])} | {_pct(mt['precision'])} | "
            f"{_pct(mt['recall'])} | {_pct(mt['f1'])} |"
        )
    lines += [
        "",
        "## Detecção por falha (cada fold testa uma falha)",
        "",
        "| Modelo | " + " | ".join(f"fold {f}" for f in range(1, 7)) + " |",
        "|---|" + "---|" * 6,
    ]
    for name, m in models.items():
        cells = []
        for r in m["folds"]:
            ok = "✅" if r["events"]["detected"] else "❌"
            cells.append(f"{ok} ({r['test_dataset']}, limiar {r['threshold']:.3g})")
        lines.append(f"| {LABEL.get(name, name)} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Folds 1–3 testam as falhas 2–4 de 2020; folds 4–6 testam as 3 falhas de "
        "2022 (vazamento de ar nos consumidores, válvula do secador de ar, "
        "vazamento de óleo).",
        "",
        "## Gate RN-05 contra o champion (`random_forest_v3`)",
        "",
        "Wilcoxon unilateral pareado por fold, custo esperado; p mínimo atingível "
        "com 6 folds = 0,016.",
        "",
        "| Candidato | Melhora mediana | p | Promove? |",
        "|---|---|---|---|",
    ]
    for g in _gates(models):
        lines.append(
            f"| {LABEL.get(g['candidate'], g['candidate'])} | "
            f"{g['median_improvement']:+.1f} | {g['p_value']:.3f} | "
            f"{'sim' if g['promote'] else 'não'} |"
        )
    lines += [
        "",
        "## Leitura",
        "",
        "* **Nenhum modelo detecta as falhas 2 e 3 de 2022** (folds 5 e 6: válvula "
        "do secador de ar, 30 min; vazamento de óleo). São modos de falha "
        "ausentes do treino. O teto atual por evento é 3 de 6 falhas.",
        "* **Nenhum candidato supera o Random Forest com significância**; por "
        "isso o champion continua `random_forest_v3`. BiLSTM e TCN têm o menor "
        "custo médio e menos episódios de falso alarme, mas a diferença por fold "
        "é nula na mediana.",
        "* **A regressão logística tem o maior recall por amostra**: acompanha a "
        "longa falha de óleo amostra a amostra, mas sem disparar o evento dentro "
        "do horizonte de antecedência.",
        "* **Redes superconfiantes**: TCN, BiLSTM, MLP e PatchTST escolheram o "
        "limiar no topo da grade (0,995) na maioria dos folds; os logits chegam "
        "a centenas, e a probabilidade fica quase binária. A calibração (por "
        "exemplo, *temperature scaling* na validação temporal) é trabalho "
        "futuro.",
        "* **Cards legados** (F1 ≈ 0,96–0,98) vinham de split aleatório, com o "
        "limiar ajustado no próprio teste. Os números acima são a medida honesta "
        "do que cada modelo faz numa falha futura.",
        "",
    ]
    return "\n".join(lines)


def _gates(models: dict[str, Any]) -> list[dict[str, Any]]:
    from src.promotion_gate import decide

    runs = [
        {
            "model": m,
            "seed": 0,
            "fold": f["fold"],
            "event_inner": {"expected_cost": f["events"]["expected_cost"]},
            "point": {"pr_auc": f["pr_auc"] or 0.0},
        }
        for m, v in models.items()
        for f in v["folds"]
    ]
    ev = {"protocols": {"all_models": {"runs": runs}}}
    return [
        decide(
            ev, m, "random_forest", metric="expected_cost_inner", protocol="all_models"
        ).to_dict()
        for m in models
        if m != "random_forest"
    ]


def main() -> None:
    data = json.loads(IN_JSON.read_text(encoding="utf-8"))
    OUT_MD.write_text(render(data) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
