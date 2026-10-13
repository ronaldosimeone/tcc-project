import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(
    "C:/Users/Raphael Okuyama/Desktop/projeto-tcc/docs/figuras-tcc/matrizes-confusao"
)
d = json.loads((OUT / "matrizes_confusao.json").read_text(encoding="utf-8"))
ORDER = ["Random Forest", "XGBoost", "MLP", "TCN", "BiLSTM", "PatchTST", "Autoencoder"]


def br(n, dec=0):
    s = f"{n:,.{dec}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def met(cm):
    tn, fp, fn, tp = np.array(cm).ravel()
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * p * r / (p + r) if p + r else 0
    return p, r, f1


def draw(ax, cm, title, subtitle):
    cm = np.array(cm)
    norm = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1.15)
    labs = [["VN", "FP"], ["FN", "VP"]]
    for i in range(2):
        for j in range(2):
            col = "white" if norm[i, j] > 0.55 else "#1b2a41"
            ax.text(
                j,
                i - 0.12,
                br(cm[i, j]),
                ha="center",
                va="center",
                fontsize=17,
                fontweight="bold",
                color=col,
            )
            ax.text(
                j,
                i + 0.17,
                f"{br(norm[i, j] * 100, 2)}% da classe real",
                ha="center",
                va="center",
                fontsize=8.5,
                color=col,
            )
            ax.text(
                j - 0.46,
                i - 0.42,
                labs[i][j],
                ha="left",
                va="center",
                fontsize=8,
                color=col,
                alpha=0.8,
            )
    ax.set_xticks([0, 1], ["Normal", "Falha"], fontsize=11)
    ax.set_yticks([0, 1], ["Normal", "Falha"], fontsize=11, rotation=90, va="center")
    ax.set_xlabel("Classe prevista", fontsize=11, labelpad=8)
    ax.set_ylabel("Classe real", fontsize=11, labelpad=8)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=42)
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
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=3)
    ax.tick_params(which="minor", length=0)


def sub(name, key):
    v = d[name]
    cm = v[key]
    p, r, f1 = met(cm)
    unit = "janelas" if v["kind"] == "janelas" else "amostras"
    if v.get("ae"):
        thr = f"limiar de MSE = {br(v['threshold'], 4)}"
    elif key == "cm_default":
        thr = "limiar = 0,5"
    else:
        thr = f"limiar = {br(v['threshold'], 4)}"
    return (
        f"n = {br(v['n'])} {unit} · {thr}\n"
        f"Precisão {br(p * 100, 1)}% · Recall {br(r * 100, 1)}% · F1 {br(f1 * 100, 1)}%"
    )


def single(name, key, path):
    fig, ax = plt.subplots(figsize=(6, 5.6), dpi=200)
    draw(ax, d[name][key], f"Matriz de confusão — {name}", sub(name, key))
    fig.tight_layout()
    fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def slug(n):
    return n.lower().replace(" ", "-")


(OUT / "limiar-0.5").mkdir(exist_ok=True)
for n in ORDER:
    single(n, "cm_tuned", OUT / f"matriz-confusao-{slug(n)}.png")
    if n != "Autoencoder":
        single(
            n,
            "cm_default",
            OUT / "limiar-0.5" / f"matriz-confusao-{slug(n)}-limiar-0.5.png",
        )

# grade comparativa (limiar de produção)
fig, axes = plt.subplots(2, 4, figsize=(21, 11.5), dpi=150)
for ax, n in zip(axes.ravel(), ORDER):
    draw(ax, d[n]["cm_tuned"], n, sub(n, "cm_tuned"))
axes.ravel()[-1].axis("off")
axes.ravel()[-1].text(
    0.5,
    0.5,
    "Cores: proporção da classe real\n(normalizado por linha).\n\nTabulares: conjunto de teste (20%).\nJanelas (TCN, BiLSTM,\nPatchTST, Autoencoder):\nconjunto de validação (20%).",
    ha="center",
    va="center",
    fontsize=11,
    color="#4a5568",
)
fig.suptitle(
    "Matrizes de confusão — modelos avaliados no MetroPT-3",
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

# tabela markdown
rows = [
    "| Modelo | Unidade | n | Limiar | VN | FP | FN | VP | Precisão | Recall | F1 | Verificação |",
    "|---|---|---|---|---|---|---|---|---|---|---|---|",
]
for n in ORDER:
    v = d[n]
    cm = np.array(v["cm_tuned"])
    tn, fp, fn, tp = cm.ravel()
    p, r, f1 = met(cm)
    ver = v.get("verify_tuned")
    vtxt = (
        "origem: recalculado"
        if v.get("ae")
        else (
            "reexecutado, idêntico"
            if ver and ver == v["cm_tuned"]
            else ("reexecutado, ±1 amostra" if ver else "derivado do model card")
        )
    )
    rows.append(
        f"| {n} | {v['kind']} | {v['n']} | {v['threshold']:.4f} | {tn} | {fp} | {fn} | {tp} | {p:.4f} | {r:.4f} | {f1:.4f} | {vtxt} |"
    )
(OUT / "tabela-matrizes-confusao.md").write_text(
    "\n".join(rows) + "\n", encoding="utf-8"
)
print("\n".join(rows))
