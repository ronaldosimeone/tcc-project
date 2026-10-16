"""
Statistical comparison of classifiers — RNF-79 / RN-05.

Implements the protocol recommended by Demšar, "Statistical Comparisons of
Classifiers over Multiple Data Sets", JMLR 7 (2006):

* Bootstrap 95 % confidence interval for a mean score.
* Wilcoxon signed-rank test for two models over paired blocks.
* Friedman test + Nemenyi post-hoc for k models over N blocks, with the
  critical-difference (CD) diagram.

A "block" is one evaluation unit shared by every model (here: one temporal
fold, with scores averaged over seeds). With very few blocks (MetroPT-3 has
only 3-4 failure folds) these tests have low power — the report must say so
rather than claim significance it cannot have.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

# Critical values q_alpha for the two-tailed Nemenyi test (alpha = 0.05),
# indexed by the number of models k. Source: Demšar (2006), Table 5(a).
_Q_ALPHA_005: dict[int, float] = {
    2: 1.960,
    3: 2.343,
    4: 2.569,
    5: 2.728,
    6: 2.850,
    7: 2.949,
    8: 3.031,
    9: 3.102,
    10: 3.164,
}


@dataclass
class ConfidenceInterval:
    mean: float
    low: float
    high: float
    n: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FriedmanResult:
    statistic: float
    p_value: float
    avg_ranks: dict[str, float]
    critical_difference: float
    n_blocks: int
    significant: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 42
) -> ConfidenceInterval:
    """Percentile bootstrap CI for the mean of ``values``."""
    arr: np.ndarray = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return ConfidenceInterval(float("nan"), float("nan"), float("nan"), 0)
    if arr.size == 1:
        v: float = float(arr[0])
        return ConfidenceInterval(v, v, v, 1)
    rng: np.random.Generator = np.random.default_rng(seed)
    means: np.ndarray = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(
        axis=1
    )
    return ConfidenceInterval(
        mean=float(arr.mean()),
        low=float(np.quantile(means, alpha / 2)),
        high=float(np.quantile(means, 1 - alpha / 2)),
        n=int(arr.size),
    )


def wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """
    Two-sided Wilcoxon signed-rank test. Returns ``(statistic, p_value)``;
    ``p_value = 1.0`` when every paired difference is zero.
    """
    diff: np.ndarray = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if np.allclose(diff, 0.0):
        return (0.0, 1.0)
    res = stats.wilcoxon(a, b, zero_method="zsplit", alternative="two-sided")
    return (float(res.statistic), float(res.pvalue))


def nemenyi_cd(k: int, n_blocks: int, alpha: float = 0.05) -> float:
    """Critical difference of average ranks: ``q_alpha * sqrt(k(k+1) / 6N)``."""
    if alpha != 0.05:
        raise ValueError("Only alpha = 0.05 is tabulated.")
    if k not in _Q_ALPHA_005:
        raise ValueError(f"Nemenyi q_alpha not tabulated for k={k}.")
    return _Q_ALPHA_005[k] * float(np.sqrt(k * (k + 1) / (6.0 * n_blocks)))


def friedman_nemenyi(
    scores: dict[str, np.ndarray], higher_is_better: bool = True, alpha: float = 0.05
) -> FriedmanResult:
    """
    Friedman test over ``scores[model] = per-block values`` (same block order
    for every model) plus Nemenyi critical difference on the average ranks.
    """
    names: list[str] = list(scores)
    if len(names) < 3:
        raise ValueError("Friedman needs at least 3 models; use wilcoxon_paired for 2.")
    mat: np.ndarray = np.column_stack(
        [np.asarray(scores[n], dtype=float) for n in names]
    )
    n_blocks: int = mat.shape[0]

    to_rank: np.ndarray = -mat if higher_is_better else mat
    ranks: np.ndarray = np.apply_along_axis(stats.rankdata, 1, to_rank)
    avg: np.ndarray = ranks.mean(axis=0)

    stat, p = stats.friedmanchisquare(*[mat[:, j] for j in range(mat.shape[1])])
    return FriedmanResult(
        statistic=float(stat),
        p_value=float(p),
        avg_ranks={n: float(r) for n, r in zip(names, avg)},
        critical_difference=nemenyi_cd(len(names), n_blocks, alpha),
        n_blocks=n_blocks,
        significant=bool(p < alpha),
    )


def plot_cd_diagram(result: FriedmanResult, out_path: Path, title: str) -> None:
    """Critical-difference diagram: models joined by a bar are not distinguishable."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items: list[tuple[str, float]] = sorted(
        result.avg_ranks.items(), key=lambda kv: kv[1]
    )
    k: int = len(items)
    cd: float = result.critical_difference

    # Demšar layout: rank axis at y = 0, best models labelled on the left,
    # worst on the right, CD scale bar above the axis, cliques below it.
    x_hi: float = max(float(k), 1.0 + cd) + 0.3
    half: int = (k + 1) // 2
    fig, ax = plt.subplots(figsize=(9, 2.0 + 0.4 * half))
    ax.set_xlim(1.0 - 1.6, x_hi + 1.6)
    ax.set_ylim(-(half + 1.4), 1.6)
    ax.axis("off")

    ax.plot([1, k], [0, 0], color="black", lw=1)
    for r in range(1, k + 1):
        ax.plot([r, r], [0, 0.12], color="black", lw=1)
        ax.text(r, 0.22, str(r), ha="center", va="bottom", fontsize=9)

    ax.plot([1, 1 + cd], [1.0, 1.0], color="tab:red", lw=2)
    for x in (1, 1 + cd):
        ax.plot([x, x], [0.92, 1.08], color="tab:red", lw=2)
    ax.text(
        1 + cd / 2,
        1.15,
        f"CD = {cd:.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
        color="tab:red",
    )

    for i, (name, rank) in enumerate(items):
        left: bool = i < half
        row: int = i if left else k - 1 - i
        y: float = -(row + 1.0)
        x_text: float = 1.0 - 0.2 if left else x_hi + 0.2
        ax.plot([rank, rank], [0, y], color="0.35", lw=0.8)
        ax.plot([rank, x_text], [y, y], color="0.35", lw=0.8)
        ax.text(
            x_text + (-0.05 if left else 0.05),
            y,
            f"{name} ({rank:.2f})",
            ha="right" if left else "left",
            va="center",
            fontsize=9,
        )

    # Cliques: maximal runs of models whose rank spread is below CD.
    level: float = -0.35
    last_j: int = -1
    for i in range(k):
        j: int = i
        while j + 1 < k and items[j + 1][1] - items[i][1] < cd:
            j += 1
        if j > i and j > last_j:
            ax.plot(
                [items[i][1] - 0.03, items[j][1] + 0.03],
                [level, level],
                color="black",
                lw=4,
            )
            level -= 0.18
            last_j = j

    ax.set_title(
        f"{title}\nFriedman p = {result.p_value:.3g}, N = {result.n_blocks} blocks "
        "(lower rank is better; bar = not distinguishable)",
        fontsize=10,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
