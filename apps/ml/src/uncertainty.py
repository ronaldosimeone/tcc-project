"""
Calibration and conformal prediction — RF-31.

A failure probability is only actionable if it means what it says: when the
model reports 0.8, roughly 80 % of such alarms should be real failures. This
module measures that (reliability diagram, ECE, Brier score) and provides a
distribution-free alternative with a coverage guarantee:

* ``MondrianConformalClassifier`` — class-conditional (Mondrian) split
  conformal prediction (Vovk et al., 2005; Angelopoulos & Bates, 2021).
  Given calibration scores, it returns for each sample a prediction SET
  ⊆ {normal, failure}. With exchangeable data, every true class is covered
  with probability ≥ 1 − α *per class* — important because a marginal
  guarantee would be satisfied by ignoring the 2 % failure class entirely.
  Sets of size 2 mean "the model cannot tell" — the honest answer that a
  single threshold hides.

Time series are not exchangeable, so the guarantee is only approximate under
drift; ``empirical_coverage`` measures the realised coverage on the test fold
so the report can show how far it deviates (RF-34 links the two).

* ``isotonic_recalibrate`` — monotone recalibration fitted on validation
  scores (never on the test fold).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression


@dataclass
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    mean_predicted: float
    observed_rate: float


@dataclass
class CalibrationReport:
    ece: float
    mce: float
    brier: float
    n: int
    bins: list[ReliabilityBin]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def calibration_report(
    y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10
) -> CalibrationReport:
    """
    Expected (ECE) and maximum (MCE) calibration error over equal-width bins,
    plus the Brier score. Empty bins are skipped (they carry zero weight).
    """
    y: np.ndarray = np.asarray(y_true, dtype=float)
    p: np.ndarray = np.clip(np.asarray(prob, dtype=float), 0.0, 1.0)
    if y.shape != p.shape:
        raise ValueError("y_true and prob must have the same shape.")
    if y.size == 0:
        raise ValueError("Empty input.")

    edges: np.ndarray = np.linspace(0.0, 1.0, n_bins + 1)
    # Right-closed last bin so p == 1.0 is counted.
    idx: np.ndarray = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)

    bins: list[ReliabilityBin] = []
    ece: float = 0.0
    mce: float = 0.0
    for b in range(n_bins):
        mask: np.ndarray = idx == b
        count: int = int(mask.sum())
        if count == 0:
            continue
        conf: float = float(p[mask].mean())
        acc: float = float(y[mask].mean())
        gap: float = abs(acc - conf)
        ece += count / y.size * gap
        mce = max(mce, gap)
        bins.append(
            ReliabilityBin(float(edges[b]), float(edges[b + 1]), count, conf, acc)
        )

    return CalibrationReport(
        ece=float(ece),
        mce=float(mce),
        brier=float(np.mean((p - y) ** 2)),
        n=int(y.size),
        bins=bins,
    )


def isotonic_recalibrate(val_prob: np.ndarray, val_y: np.ndarray) -> IsotonicRegression:
    """Fit a monotone map score -> probability on VALIDATION data only."""
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(np.asarray(val_prob, dtype=float), np.asarray(val_y, dtype=float))
    return iso


class MondrianConformalClassifier:
    """
    Class-conditional split conformal prediction for a binary classifier.

    Nonconformity of a sample for class c: ``1 - P(c | x)``. For each class
    the threshold q_c is the ⌈(n_c + 1)(1 − α)⌉ / n_c empirical quantile of
    the calibration nonconformity scores of that class. A class enters the
    prediction set when its nonconformity is ≤ q_c.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1).")
        self.alpha: float = alpha
        self.q_: dict[int, float] = {}

    @staticmethod
    def _nonconformity(prob_failure: np.ndarray, cls: int) -> np.ndarray:
        p: np.ndarray = np.asarray(prob_failure, dtype=float)
        return 1.0 - (p if cls == 1 else 1.0 - p)

    def fit(
        self, cal_prob: np.ndarray, cal_y: np.ndarray
    ) -> "MondrianConformalClassifier":
        y: np.ndarray = np.asarray(cal_y).astype(int)
        p: np.ndarray = np.asarray(cal_prob, dtype=float)
        for cls in (0, 1):
            scores: np.ndarray = np.sort(self._nonconformity(p[y == cls], cls))
            n: int = scores.size
            if n == 0:
                raise ValueError(f"No calibration samples of class {cls}.")
            rank: int = int(np.ceil((n + 1) * (1 - self.alpha)))
            # rank > n means the guarantee needs an infinite threshold:
            # the class is always included (trivially covered).
            self.q_[cls] = float("inf") if rank > n else float(scores[rank - 1])
        return self

    def predict_sets(self, prob_failure: np.ndarray) -> np.ndarray:
        """Boolean matrix (n, 2): column c is True when class c is in the set."""
        if not self.q_:
            raise RuntimeError("Call fit() first.")
        p: np.ndarray = np.asarray(prob_failure, dtype=float)
        return np.column_stack(
            [self._nonconformity(p, c) <= self.q_[c] for c in (0, 1)]
        )

    @staticmethod
    def empirical_coverage(sets: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
        """Realised per-class coverage, mean set size and share of ambiguous
        (size-2) and empty sets."""
        y: np.ndarray = np.asarray(y_true).astype(int)
        covered: np.ndarray = sets[np.arange(y.size), y]
        size: np.ndarray = sets.sum(axis=1)
        out: dict[str, float] = {
            "coverage": float(covered.mean()),
            "mean_set_size": float(size.mean()),
            "ambiguous_rate": float((size == 2).mean()),
            "empty_rate": float((size == 0).mean()),
        }
        for cls in (0, 1):
            m: np.ndarray = y == cls
            out[f"coverage_class_{cls}"] = (
                float(covered[m].mean()) if m.any() else float("nan")
            )
        return out


def plot_reliability(
    reports: dict[str, CalibrationReport], out_path: Any, title: str
) -> None:
    """Reliability diagram: observed failure rate vs mean predicted probability."""
    import matplotlib

    matplotlib.use("Agg")
    from pathlib import Path

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], ls="--", color="0.5", lw=1, label="perfect calibration")
    for name, rep in reports.items():
        xs: list[float] = [b.mean_predicted for b in rep.bins]
        ys: list[float] = [b.observed_rate for b in rep.bins]
        ax.plot(xs, ys, marker="o", lw=1.2, label=f"{name} (ECE {rep.ece:.3f})")
    ax.set_xlabel("Mean predicted P(failure)")
    ax.set_ylabel("Observed failure rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
