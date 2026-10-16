"""
Unit tests for src/uncertainty.py — RF-31.

Guarantees verified
-------------------
1.  A perfectly calibrated predictor has ECE ≈ 0; a constant overconfident one does not.
2.  Brier score matches its definition.
3.  Mondrian conformal reaches ≥ 1 − α coverage PER CLASS on exchangeable
    data, including the rare class (2 % prevalence).
4.  Uninformative scores produce ambiguous (size-2) sets instead of wrong
    confident answers.
5.  Isotonic recalibration lowers ECE of a miscalibrated predictor.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.uncertainty import (
    MondrianConformalClassifier,
    calibration_report,
    isotonic_recalibrate,
)


def _calibrated_sample(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = rng.uniform(0, 1, n)
    y = (rng.uniform(0, 1, n) < p).astype(int)
    return p, y


def test_calibrated_predictor_has_low_ece() -> None:
    p, y = _calibrated_sample(200_000, 0)
    rep = calibration_report(y, p)
    assert rep.ece < 0.01
    assert rep.brier == pytest.approx(np.mean((p - y) ** 2))


def test_overconfident_predictor_has_high_ece() -> None:
    y = np.array([0, 1] * 500)
    rep = calibration_report(y, np.full(y.size, 0.95))
    assert rep.ece == pytest.approx(0.45)
    assert rep.mce == pytest.approx(0.45)


def test_shape_mismatch_rejected() -> None:
    with pytest.raises(ValueError):
        calibration_report(np.array([0, 1]), np.array([0.5]))


def _imbalanced(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = (rng.uniform(0, 1, n) < 0.02).astype(int)
    # Informative but noisy scores.
    p = np.clip(
        np.where(y == 1, rng.normal(0.7, 0.2, n), rng.normal(0.2, 0.15, n)), 0, 1
    )
    return p, y


def test_mondrian_conformal_per_class_coverage() -> None:
    alpha = 0.1
    cal_p, cal_y = _imbalanced(50_000, 1)
    test_p, test_y = _imbalanced(50_000, 2)
    cp = MondrianConformalClassifier(alpha).fit(cal_p, cal_y)
    cov = cp.empirical_coverage(cp.predict_sets(test_p), test_y)
    assert cov["coverage_class_0"] >= 1 - alpha - 0.01
    assert (
        cov["coverage_class_1"] >= 1 - alpha - 0.03
    )  # ~1000 positives -> sampling noise
    # Mondrian sets may be empty (no class conforms) — legitimate, but rare
    # when scores are informative; ambiguous sets stay a minority.
    assert cov["mean_set_size"] < 1.5
    assert cov["empty_rate"] < 0.1 and cov["ambiguous_rate"] < 0.5


def test_uninformative_scores_give_ambiguous_sets() -> None:
    rng = np.random.default_rng(3)
    y = (rng.uniform(0, 1, 20_000) < 0.5).astype(int)
    p = rng.uniform(0, 1, y.size)  # independent of y
    cp = MondrianConformalClassifier(0.1).fit(p[:10_000], y[:10_000])
    cov = cp.empirical_coverage(cp.predict_sets(p[10_000:]), y[10_000:])
    assert cov["ambiguous_rate"] > 0.5


def test_conformal_requires_fit_and_both_classes() -> None:
    with pytest.raises(RuntimeError):
        MondrianConformalClassifier().predict_sets(np.array([0.5]))
    with pytest.raises(ValueError):
        MondrianConformalClassifier().fit(np.array([0.1, 0.2]), np.array([0, 0]))
    with pytest.raises(ValueError):
        MondrianConformalClassifier(alpha=1.5)


def test_isotonic_reduces_ece() -> None:
    p, y = _calibrated_sample(40_000, 4)
    distorted = p**3  # systematically underconfident
    iso = isotonic_recalibrate(distorted[:20_000], y[:20_000])
    before = calibration_report(y[20_000:], distorted[20_000:]).ece
    after = calibration_report(y[20_000:], iso.predict(distorted[20_000:])).ece
    assert after < before / 3
