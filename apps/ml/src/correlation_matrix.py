"""
Correlation matrix of every attribute in the three MetroPT datasets.

Entry point:
    python -m src.correlation_matrix

For MetroPT-3 (2020), MetroPT 2022 and MetroPT2 it computes the Pearson
correlation between all sensor columns and the failure label (`anomaly`), on
every row at 10 s (no sampling). Bookkeeping columns (timestamp, row index,
dataset tag) are not attributes and are left out. A column that is constant in
a dataset has no defined correlation and is shown as "—".

Output: models/correlation_matrix.json, docs/correlation_matrix.md and one
heatmap per dataset in docs/figures-ml/.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

_ML_ROOT: Path = Path(__file__).resolve().parent.parent
REPO_ROOT: Path = _ML_ROOT.parent.parent
OUT_JSON: Path = _ML_ROOT / "models" / "correlation_matrix.json"
OUT_MD: Path = REPO_ROOT / "docs" / "correlation_matrix.md"
FIG_DIR: Path = REPO_ROOT / "docs" / "figures-ml"

DATASETS: dict[str, tuple[str, Path]] = {
    "metropt3": ("MetroPT-3 (2020)", _ML_ROOT / "data/processed/metropt3.parquet"),
    "metropt2022": ("MetroPT 2022", _ML_ROOT / "data/processed/metropt2022.parquet"),
    "metropt2": ("MetroPT2", _ML_ROOT / "data/processed/metropt2.parquet"),
}
NOT_ATTRIBUTES: frozenset[str] = frozenset(
    {"timestamp", "Unnamed: 0", "dataset", "overlap_with_training"}
)
LABEL: str = "anomaly"
STRONG: float = 0.8

# Diverging scale (blue = negative, gray = 0, red = positive), from the
# validated dataviz reference palette.
CMAP = LinearSegmentedColormap.from_list(
    "corr", ["#2a78d6", "#f0efec", "#e34948"], N=256
)


def attributes(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in NOT_ATTRIBUTES]
    return [c for c in cols if c != LABEL] + ([LABEL] if LABEL in cols else [])


def correlation(path: Path) -> tuple[pd.DataFrame, int, list[str]]:
    df = pd.read_parquet(path)
    cols = attributes(df)
    data = df[cols].astype("float64")
    constant = [c for c in cols if data[c].nunique(dropna=True) <= 1]
    return data.corr(method="pearson"), len(df), constant


def strong_pairs(corr: pd.DataFrame) -> list[tuple[str, str, float]]:
    cols = list(corr.columns)
    out = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) >= STRONG:
                out.append((a, b, float(r)))
    return sorted(out, key=lambda t: -abs(t[2]))


def heatmap(corr: pd.DataFrame, title: str, path: Path) -> None:
    n = len(corr)
    fig, ax = plt.subplots(figsize=(0.62 * n + 3, 0.55 * n + 2), dpi=150)
    values = corr.to_numpy()
    masked = np.ma.masked_invalid(values)
    cmap = CMAP.copy()
    cmap.set_bad("#ffffff")
    im = ax.imshow(masked, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(n), corr.columns, rotation=60, ha="right", fontsize=9)
    ax.set_yticks(range(n), corr.index, fontsize=9)
    # 2 px surface gap between cells
    ax.set_xticks(np.arange(-0.5, n), minor=True)
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which="minor", color="#fcfcfb", linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(n):
        for j in range(n):
            v = values[i, j]
            text = "—" if np.isnan(v) else f"{v:.2f}"
            ink = "#ffffff" if (not np.isnan(v) and abs(v) >= 0.6) else "#1f1f1e"
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color=ink)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Correlação de Pearson (r)", fontsize=9)
    cbar.outline.set_visible(False)
    ax.set_title(title, fontsize=12, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(path, facecolor="#fcfcfb")
    plt.close(fig)


def _fmt(v: float) -> str:
    return "—" if pd.isna(v) else f"{v:.2f}"


def table(corr: pd.DataFrame) -> list[str]:
    cols = list(corr.columns)
    lines = [
        "| | " + " | ".join(f"`{c}`" for c in cols) + " |",
        "|---|" + "---:|" * len(cols),
    ]
    for r in cols:
        lines.append(
            f"| `{r}` | " + " | ".join(_fmt(corr.loc[r, c]) for c in cols) + " |"
        )
    return lines


def render(results: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# Matriz de correlação dos atributos — MetroPT-3, MetroPT 2022 e MetroPT2",
        "",
        "> Gerado por `python -m src.correlation_matrix` (apps/ml). Não edite à mão.",
        "",
        "Correlação de Pearson entre **todas as colunas de atributos** de cada dataset "
        "(sensores analógicos, sinais digitais e o rótulo de falha `anomaly`), "
        "calculada sobre **todas as linhas** a 10 s, sem amostragem. Colunas de "
        "controle (`timestamp`, índice, marcação de dataset) não são atributos e "
        "ficam de fora. Coluna constante no dataset não tem correlação definida: "
        "aparece como **—**.",
        "",
        "Leitura: |r| ≥ 0,8 = correlação forte (atributos quase redundantes); o sinal "
        "indica se sobem juntos (+) ou em sentidos opostos (−). Correlação linear "
        "não mede relação causal nem relação não linear.",
        "",
    ]
    for key, r in results.items():
        corr = pd.DataFrame(r["matrix"])
        lines += [
            f"## {r['name']}",
            "",
            f"{r['rows']:,} linhas · {len(corr)} atributos".replace(",", ".")
            + (
                f" · constantes (sem correlação): {', '.join(f'`{c}`' for c in r['constant'])}"
                if r["constant"]
                else ""
            ),
            "",
            f"![Matriz de correlação — {r['name']}](figures-ml/corr_matrix_{key}.png)",
            "",
            *table(corr),
            "",
            f"**Pares fortemente correlacionados (|r| ≥ {STRONG:.1f}):**",
            "",
        ]
        if r["strong_pairs"]:
            lines += ["| Atributo A | Atributo B | r |", "|---|---|---:|"]
            lines += [f"| `{a}` | `{b}` | {v:+.2f} |" for a, b, v in r["strong_pairs"]]
        else:
            lines.append("Nenhum.")
        if LABEL in corr.columns:
            lab = corr[LABEL].drop(LABEL).dropna()
            top = lab.reindex(lab.abs().sort_values(ascending=False).index).head(5)
            lines += [
                "",
                f"**Atributos mais correlacionados com o rótulo de falha (`{LABEL}`):** "
                + ", ".join(f"`{k}` ({v:+.2f})" for k, v in top.items()),
            ]
        lines.append("")
    best = {
        r["name"]: max(
            (
                (c, v)
                for c, v in r["matrix"][LABEL].items()
                if c != LABEL and v is not None
            ),
            key=lambda t: abs(t[1]),
        )
        for r in results.values()
    }
    lines += [
        "## Observações",
        "",
        "* Atributo mais correlacionado com o rótulo, por dataset: "
        + "; ".join(f"{n}: `{c}` ({v:+.2f})" for n, (c, v) in best.items())
        + ". O sinal que acompanha a falha muda de um dataset para outro: no "
        "MetroPT-3, `DV_pressure` vem de três vazamentos de ar de 2020 (ver "
        "[`event_correlation.md`](event_correlation.md)); no MetroPT2, `Oil_level` "
        "vale 0 em todas as 23.156 linhas em falha e em só 2.880 linhas normais "
        "(é praticamente o rótulo). `Oil_level` fica fora dos modelos porque é constante no "
        "MetroPT 2022 (sem informação para treinar) — usá-lo seria aprender um "
        "sinal que o treino não contém. No MetroPT 2022 nenhum atributo bruto passa "
        "de |r| = 0,12 com o rótulo.",
        "* Os sinais digitais `COMP`, `DV_eletric` e `MPG` e as pressões `TP2`/`H1` "
        "se movem juntos (|r| ≥ 0,9): descrevem o mesmo ciclo carga/alívio do "
        "compressor.",
        "* `Reservoirs` e `TP3` medem a mesma pressão no MetroPT-3 e no MetroPT2 "
        "(r ≈ 1), mas não no MetroPT 2022 — um dos motivos de `Reservoirs` ficar fora "
        "dos modelos (ver [`datasets.md`](datasets.md)).",
        "* `Flowmeter` só existe nos datasets de 2022.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    for key, (name, path) in DATASETS.items():
        corr, rows, constant = correlation(path)
        heatmap(
            corr, f"Matriz de correlação — {name}", FIG_DIR / f"corr_matrix_{key}.png"
        )
        results[key] = {
            "name": name,
            "rows": rows,
            "constant": constant,
            "strong_pairs": strong_pairs(corr),
            "matrix": {
                c: {
                    k: (None if pd.isna(v) else round(float(v), 4))
                    for k, v in corr[c].items()
                }
                for c in corr.columns
            },
        }
        print(f"{name}: {rows} rows, {len(corr)} attributes, constant={constant}")
    OUT_JSON.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "method": "pearson, all rows",
                "datasets": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    OUT_MD.write_text(render(results), encoding="utf-8")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
