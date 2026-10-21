"""
Renders docs/model_evaluation_leakfree.md and the confusion-matrix figures
from the nested evaluation (models/nested/*.json) plus the sensitivity,
conformal (ACI) and pooled RUL results.

Entry point:
    python -m src.render_nested_report
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.confusion import LABEL, plot
from src.evaluate_nested import ALL_MODELS
from src.promotion_gate import decide

_ML_ROOT: Path = Path(__file__).resolve().parent.parent
REPO_ROOT: Path = _ML_ROOT.parent.parent
NESTED_DIR: Path = _ML_ROOT / "models" / "nested"
FIG_DIR: Path = REPO_ROOT / "docs" / "figures-ml"
OUT_MD: Path = REPO_ROOT / "docs" / "model_evaluation_leakfree.md"
SENS_JSON: Path = _ML_ROOT / "models" / "eval_sensitivity.json"
ACI_JSON: Path = _ML_ROOT / "models" / "eval_conformal_aci.json"
RUL_JSON: Path = _ML_ROOT / "models" / "eval_rul_pooled.json"

ARTEFACT: dict[str, str] = {
    "random_forest": "random_forest_v3",
    "xgboost": "xgboost_v3",
    "logistic_regression": "—",
    "mlp": "mlp_v3",
    "tcn": "tcn_v3",
    "bilstm": "bilstm_v3",
    "patchtst": "patchtst_v3",
    "autoencoder": "autoencoder_v3",
}


def _n(x: float) -> str:
    return f"{int(x):,}".replace(",", ".")


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x:.1%}"


def _f(x: float | None, d: int = 3) -> str:
    return "—" if x is None else f"{x:.{d}f}"


def _load() -> dict[str, Any]:
    out = {}
    for name in ALL_MODELS:
        p = NESTED_DIR / f"{name}.json"
        if p.exists():
            out[name] = json.loads(p.read_text(encoding="utf-8"))
    return out


def _gate_rows(models: dict[str, Any]) -> list[dict[str, Any]]:
    """RN-05 vs the champion, pairing = fold, cost averaged over seeds."""
    runs = []
    for name, m in models.items():
        by_fold: dict[int, list[float]] = {}
        for r in m["runs"]:
            by_fold.setdefault(r["fold"], []).append(r["events"]["expected_cost"])
        for fold, costs in by_fold.items():
            runs.append(
                {
                    "model": name,
                    "seed": 0,
                    "fold": fold,
                    "event_inner": {"expected_cost": float(np.mean(costs))},
                    "point": {"pr_auc": 0.0},
                }
            )
    ev = {"protocols": {"nested": {"runs": runs}}}
    return [
        decide(
            ev, n, "random_forest", metric="expected_cost_inner", protocol="nested"
        ).to_dict()
        for n in models
        if n != "random_forest"
    ]


def render(models: dict[str, Any]) -> str:
    lines = [
        "# Avaliação de doutorado de todos os modelos — protocolo aninhado",
        "",
        "> Gerado por `python -m src.render_nested_report` a partir de "
        "`apps/ml/models/nested/*.json`. Não edite à mão.",
        "",
        "## Protocolo",
        "",
        "* **Dados**: MetroPT-3 (2020) + MetroPT 2022, sensores harmonizados (sem "
        "`Reservoirs`/`Oil_level`, que mudam de significado entre os datasets).",
        "* **Loop externo**: walk-forward purgado por evento, 7 falhas → **6 folds "
        "fora do tempo**; cada fold treina só com o passado e é testado numa falha "
        "que o modelo nunca viu.",
        "* **Loop interno** (só nos dados de treino): leave-one-failure-out para "
        "(i) escolher os hiperparâmetros pelo custo esperado RN-02, (ii) calibrar "
        "por Platt e (iii) escolher o limiar sobre a probabilidade calibrada. Os "
        "parâmetros antigos do Optuna (escolhidos num split aleatório) foram "
        "descartados.",
        "* **Mesmo protocolo para os 8 modelos**, árvores e redes; **3 seeds** (a "
        "seed 0 escolhe os hiperparâmetros, as seeds 1–2 medem a variância do "
        "treino). Balanceamento de classes só no treino; teste na prevalência "
        "real.",
        "* **Fold 1** tem uma única falha de treino: não há o que validar "
        "internamente, então usa os hiperparâmetros padrão e limiar a priori "
        "(registrado como *fallback*).",
        "",
        "## Por evento — 6 falhas fora do tempo, média de 3 seeds",
        "",
        "| Modelo | Artefato | Falhas detectadas (média [mín; máx]) | Episódios de "
        "falso alarme | Custo esperado médio | ECE bruto → calibrado |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in models.items():
        s = m["summary"]
        e_raw = [
            p["ece_raw_mean"] for p in s["per_seed"] if p["ece_raw_mean"] is not None
        ]
        e_cal = [
            p["ece_calibrated_mean"]
            for p in s["per_seed"]
            if p["ece_calibrated_mean"] is not None
        ]
        ece = f"{np.mean(e_raw):.3f} → {np.mean(e_cal):.3f}" if e_raw and e_cal else "—"
        lines.append(
            f"| {LABEL.get(name, name)} | `{ARTEFACT[name]}` | "
            f"**{s['detected_mean']:.1f}** [{s['detected_min']}; {s['detected_max']}] "
            f"| {s['false_alarm_episodes_mean']:.0f} | {s['expected_cost_mean']:.1f} | {ece} |"
        )

    seed0 = {n: {"total": m["summary"]["per_seed"][0]} for n, m in models.items()}
    for level in ("alarms", "samples"):
        plot(seed0, level, FIG_DIR / f"confusion_{level}.png")
    lines += [
        "",
        "## Matrizes de confusão — por amostra após k = 6 (seed 0)",
        "",
        "![Matrizes de confusão — alarmes](figures-ml/confusion_alarms.png)",
        "",
        "| Modelo | TP | FP | FN | TN | Precisão | Recall | Especificidade | F1 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, t in seed0.items():
        cm, mt = t["total"]["alarms"], t["total"]["metrics_alarms"]
        lines.append(
            f"| {LABEL.get(name, name)} | {_n(cm['tp'])} | {_n(cm['fp'])} | "
            f"{_n(cm['fn'])} | {_n(cm['tn'])} | {_pct(mt['precision'])} | "
            f"{_pct(mt['recall'])} | {_pct(mt['specificity'])} | {_pct(mt['f1'])} |"
        )
    lines += [
        "",
        "Por amostra no limiar, sem persistência: "
        "![Matrizes de confusão — amostras](figures-ml/confusion_samples.png)",
        "",
        "## Detecção por falha (seed 0) e hiperparâmetros escolhidos",
        "",
        "| Modelo | " + " | ".join(f"fold {f}" for f in range(1, 7)) + " |",
        "|---|" + "---|" * 6,
    ]
    for name, m in models.items():
        cells = []
        for r in sorted(
            (r for r in m["runs"] if r["seed"] == 0), key=lambda r: r["fold"]
        ):
            ok = "✅" if r["events"]["detected"] else "❌"
            params = ", ".join(f"{k}={v}" for k, v in r["params"].items())
            cells.append(f"{ok} `{params}`")
        lines.append(f"| {LABEL.get(name, name)} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Folds 1–3: falhas 2–4 de 2020. Folds 4–6: as 3 falhas de 2022 (vazamento "
        "de ar nos consumidores, válvula do secador de ar, vazamento de óleo).",
        "",
        "## Gate RN-05 contra o champion (`random_forest_v3`)",
        "",
        "Wilcoxon unilateral pareado por fold (custo médio das 3 seeds); p mínimo "
        "atingível com 6 folds = 0,016.",
        "",
        "| Candidato | Melhora mediana | p | Promove? |",
        "|---|---|---|---|",
    ]
    if "random_forest" in models:
        for g in _gate_rows(models):
            lines.append(
                f"| {LABEL.get(g['candidate'], g['candidate'])} | "
                f"{g['median_improvement']:+.1f} | {g['p_value']:.3f} | "
                f"{'sim' if g['promote'] else 'não'} |"
            )
    lines += _sensitivity() + _aci() + _rul()
    return "\n".join(lines) + "\n"


def _sensitivity() -> list[str]:
    if not SENS_JSON.exists():
        return []
    d = json.loads(SENS_JSON.read_text(encoding="utf-8"))
    stab = d["ranking_stability"]
    taus = [v["kendall_tau_vs_reference"] for v in stab.values()]
    best: dict[str, int] = {}
    for v in stab.values():
        best[v["best_model"]] = best.get(v["best_model"], 0) + 1
    lines = [
        "",
        "## Sensibilidade às escolhas de avaliação (k, horizonte, razão de custo)",
        "",
        f"Grade: k ∈ {d['grid']['k']}, horizonte ∈ {d['grid']['horizon_h']} h, "
        f"C_FN/C_FP ∈ {d['grid']['cost_ratio']} — {len(stab)} configurações; o "
        "limiar é re-escolhido nas predições out-of-fold calibradas de cada "
        "configuração (sem retreino).",
        "",
        f"* Estabilidade do ranking por custo (τ de Kendall contra a referência "
        f"k = 6, 6 h, 100): mediana **{np.median(taus):.2f}**, mínimo "
        f"{np.min(taus):.2f}.",
        "* Melhor modelo por configuração: "
        + ", ".join(
            f"{LABEL.get(k, k)} em {v}"
            for k, v in sorted(best.items(), key=lambda t: -t[1])
        )
        + ".",
        "",
        "| Modelo | Falhas detectadas (mín–máx entre configurações) |",
        "|---|---|",
    ]
    for name, settings in d["models"].items():
        det = [v["detected_mean"] for v in settings.values()]
        lines.append(f"| {LABEL.get(name, name)} | {min(det):.1f} – {max(det):.1f} |")
    return lines


def _aci() -> list[str]:
    if not ACI_JSON.exists():
        return []
    d = json.loads(ACI_JSON.read_text(encoding="utf-8"))
    target = 1 - d["alpha"]
    lines = [
        "",
        f"## Incerteza — conformal estático × adaptativo (ACI), alvo {target:.0%}",
        "",
        "Conjunto de calibração = predições out-of-fold calibradas (só treino); "
        "fluxo de teste fora do tempo. Hipótese: o rótulo de cada linha fica "
        "conhecido logo depois (na operação, chega com o veredito da OS).",
        "",
        "| Modelo | Cobertura da classe falha: estático → ACI | Cobertura normal: "
        "estático → ACI | Tamanho médio do conjunto: estático → ACI |",
        "|---|---|---|---|",
    ]
    for name, m in d["models"].items():
        s, a = m["static"], m["aci"]
        lines.append(
            f"| {LABEL.get(name, name)} | {s['coverage_class_1']:.2f} → "
            f"**{a['coverage_class_1']:.2f}** | {s['coverage_class_0']:.2f} → "
            f"{a['coverage_class_0']:.2f} | {s['set_size']:.2f} → {a['set_size']:.2f} |"
        )
    return lines


def _rul() -> list[str]:
    if not RUL_JSON.exists():
        return []
    d = json.loads(RUL_JSON.read_text(encoding="utf-8"))
    s = d["summary"]
    return [
        "",
        "## RUL com as 7 falhas (6 folds)",
        "",
        f"* XGBoost AFT: C-index {s['xgb_aft']['c_index']['mean']:.2f} "
        f"[{s['xgb_aft']['c_index']['low']:.2f}; {s['xgb_aft']['c_index']['high']:.2f}].",
        f"* Baseline de renovação: C-index {s['renewal']['c_index']['mean']:.2f} "
        f"[{s['renewal']['c_index']['low']:.2f}; {s['renewal']['c_index']['high']:.2f}].",
        "",
        "Com 7 falhas o tempo até a falha continua não aprendível pelas features "
        "(C-index ≈ 0,5): o resultado negativo do RF-30 se mantém com o dobro de "
        "folds.",
    ]


def main() -> None:
    models = _load()
    OUT_MD.write_text(render(models), encoding="utf-8")
    print(f"Wrote {OUT_MD} ({len(models)} models)")


if __name__ == "__main__":
    main()
