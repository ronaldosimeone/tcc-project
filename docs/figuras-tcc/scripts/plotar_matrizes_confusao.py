"""Matrizes de confusão das figuras do TCC — protocolo oficial (congelado).

Fonte: apps/ml/models/nested/<detector>.json (walk-forward purgado × LOFO
interno, 6 folds de teste, seed 0) — os mesmos números de
docs/model_evaluation_leakfree.md e docs/statistical_baselines.md.

Duas leituras por detector:
* alarmes (principal): amostra prevista "falha" só depois de k = 6 leituras
  seguidas acima do limiar calibrado (RN-03), como o sistema alarma;
* por amostra: cada leitura no limiar calibrado, sem persistência.

As matrizes medem acerto POR LEITURA. A métrica principal do TCC é por
EVENTO (falhas detectadas de 6): ela vai no subtítulo de cada figura.

Uso (na raiz do repositório):
    apps/ml/.venv/Scripts/python docs/figuras-tcc/scripts/plotar_matrizes_confusao.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
NESTED = ROOT / "apps" / "ml" / "models" / "nested"
OUT = ROOT / "docs" / "figuras-tcc" / "matrizes-confusao"

DETECTORS: dict[str, str] = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "logistic_regression": "Regressão Logística",
    "mlp": "MLP",
    "tcn": "TCN",
    "bilstm": "BiLSTM",
    "patchtst": "PatchTST",
    "autoencoder": "Autoencoder",
    "cusum": "CUSUM",
    "ewma": "EWMA",
}
LEVELS: dict[str, str] = {
    "alarms": "alarmes (k = 6 leituras seguidas)",
    "samples": "por leitura, no limiar calibrado",
}


def br(n: float, dec: int = 0) -> str:
    s = f"{n:,.{dec}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def cm_of(counts: dict[str, int]) -> np.ndarray:
    return np.array(
        [[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]], dtype=np.int64
    )


def scores(cm: np.ndarray) -> tuple[float, float, float]:
    tn, fp, fn, tp = cm.ravel()
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def draw(ax: plt.Axes, cm: np.ndarray, title: str, subtitle: str) -> None:
    norm = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1.15)
    labels = [["VN", "FP"], ["FN", "VP"]]
    for i in range(2):
        for j in range(2):
            ink = "white" if norm[i, j] > 0.55 else "#1b2a41"
            ax.text(
                j,
                i - 0.12,
                br(cm[i, j]),
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
                color=ink,
            )
            ax.text(
                j,
                i + 0.17,
                f"{br(norm[i, j] * 100, 1)}% da classe real",
                ha="center",
                va="center",
                fontsize=8.5,
                color=ink,
            )
            ax.text(
                j - 0.46,
                i - 0.42,
                labels[i][j],
                ha="left",
                va="center",
                fontsize=8,
                color=ink,
                alpha=0.8,
            )
    ax.set_xticks([0, 1], ["Normal", "Falha"], fontsize=11)
    ax.set_yticks([0, 1], ["Normal", "Falha"], fontsize=11, rotation=90, va="center")
    ax.set_xlabel("Classe prevista", fontsize=11, labelpad=8)
    ax.set_ylabel("Classe real", fontsize=11, labelpad=8)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=44)
    ax.text(
        0.5,
        1.025,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#4a5568",
    )
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=3)
    ax.tick_params(which="minor", length=0)


def subtitle(row: dict, level: str) -> str:
    p, r, f1 = scores(np.array(row[level]))
    return (
        f"{LEVELS[level]} · falhas detectadas {row['detected']}/6\n"
        f"Precisão {br(p * 100, 1)}% · Recall {br(r * 100, 1)}% · F1 {br(f1 * 100, 1)}%"
    )


def slug(name: str) -> str:
    return name.lower().replace(" ", "-").replace("ã", "a").replace("í", "i")


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "por-leitura").mkdir(parents=True)

    data: dict[str, dict] = {}
    for key, name in DETECTORS.items():
        seed0 = json.loads((NESTED / f"{key}.json").read_text(encoding="utf-8"))[
            "summary"
        ]["per_seed"][0]
        detected = int(seed0["events"]["detected"])
        data[name] = {
            "alarms": cm_of(seed0["alarms"]).tolist(),
            "samples": cm_of(seed0["samples"]).tolist(),
            "detected": detected,
        }

    for name, row in data.items():
        for level, folder in (("alarms", OUT), ("samples", OUT / "por-leitura")):
            fig, ax = plt.subplots(figsize=(6, 5.6), dpi=200)
            draw(
                ax,
                np.array(row[level]),
                f"Matriz de confusão — {name}",
                subtitle(row, level),
            )
            fig.tight_layout()
            suffix = "" if level == "alarms" else "-por-leitura"
            fig.savefig(
                folder / f"matriz-confusao-{slug(name)}{suffix}.png",
                dpi=300,
                facecolor="white",
                bbox_inches="tight",
            )
            plt.close(fig)

    fig, axes = plt.subplots(2, 5, figsize=(27, 11.5), dpi=150)
    for ax, (name, row) in zip(axes.ravel(), data.items()):
        draw(ax, np.array(row["alarms"]), name, subtitle(row, "alarms"))
    fig.suptitle(
        "Matrizes de confusão — walk-forward aninhado (MetroPT-3 + MetroPT 2022, "
        "6 folds), alarmes com k = 6",
        fontsize=17,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=3)
    fig.savefig(
        OUT / "matrizes-confusao-comparativo.png",
        dpi=200,
        facecolor="white",
        bbox_inches="tight",
    )
    plt.close(fig)

    (OUT / "matrizes_confusao.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    rows = [
        "# Matrizes de confusão — protocolo oficial (walk-forward aninhado)",
        "",
        "> Gerado por `docs/figuras-tcc/scripts/plotar_matrizes_confusao.py` a partir de",
        "> `apps/ml/models/nested/*.json` (seed 0). Não edite à mão.",
        "",
        "Acerto por leitura nos 6 folds de teste. A métrica principal do TCC é por",
        "evento (falhas detectadas de 6); ver `docs/model_evaluation_leakfree.md`.",
        "",
        "| Detector | Falhas detectadas | Leitura | VN | FP | FN | VP | Precisão | Recall | F1 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in data.items():
        for level, label in (("alarms", "alarmes k = 6"), ("samples", "por leitura")):
            cm = np.array(row[level])
            tn, fp, fn, tp = cm.ravel()
            p, r, f1 = scores(cm)
            rows.append(
                f"| {name} | {row['detected']}/6 | {label} | {br(tn)} | {br(fp)} | "
                f"{br(fn)} | {br(tp)} | {br(p * 100, 1)}% | {br(r * 100, 1)}% | "
                f"{br(f1 * 100, 1)}% |"
            )
    (OUT / "tabela-matrizes-confusao.md").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )
    print("\n".join(rows))


if __name__ == "__main__":
    main()
