"""
Risk relative to the active model's calibrated decision threshold.

Each served model has its own leak-free decision threshold (model card:
MLP 0.06 … XGBoost 0.98), so a raw probability means different things for
different models. The platform expresses risk on one scale, shared with the
dashboard (apps/frontend/lib/risk-thresholds.ts):

    risk = 0.65 · p / thr                     for p < thr
    risk = 0.65 + 0.35 · (p − thr) / (1 − thr)  for p ≥ thr

so ``risk = 0.65`` is exactly the model's decision ("failure" → CRITICAL on
the dashboard), ``0.35`` is where ALERT starts (0.35/0.65 of the threshold)
and ``1.0`` is probability 1. The critical-alert setting (RF-25) is a level
on this scale.
"""

from __future__ import annotations

ALERT_RISK: float = 0.35
CRITICAL_RISK: float = 0.65


def risk_index(probability: float, decision_threshold: float) -> float:
    """Probability → risk on the model-relative scale (monotone, [0, 1])."""
    p = min(max(float(probability), 0.0), 1.0)
    thr = min(max(float(decision_threshold), 1e-9), 1.0 - 1e-9)
    if p < thr:
        return CRITICAL_RISK * p / thr
    return CRITICAL_RISK + (1.0 - CRITICAL_RISK) * (p - thr) / (1.0 - thr)


def probability_for_risk(risk: float, decision_threshold: float) -> float:
    """Inverse of :func:`risk_index`: the probability at which the model
    reaches ``risk`` (shown in the settings screen for the active model)."""
    r = min(max(float(risk), 0.0), 1.0)
    thr = min(max(float(decision_threshold), 1e-9), 1.0 - 1e-9)
    if r < CRITICAL_RISK:
        return r / CRITICAL_RISK * thr
    return thr + (r - CRITICAL_RISK) / (1.0 - CRITICAL_RISK) * (1.0 - thr)
