"""
Alerts follow the calibrated decision threshold of the model that scored the
reading (model card), not a fixed 0.70.

Found live: with ``mlp_v2`` (threshold 0.08) or ``bilstm_v2`` (0.27) the
model flagged real failures (``predicted_class = 1``, RN-03 events opened)
while P(failure) stayed at 0.2–0.5, so the fixed 0.70 push alert and the
dashboard bands never showed them.

1. ``decision_threshold`` present → triggered iff probability ≥ threshold.
2. Absent → legacy RF-14 rule (strictly > 0.70) is kept.
3. ``ConnectionManager.broadcast_alert`` applies the same rule.
4. ``GET /models`` exposes the active model's threshold for the UI.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.core.ws_manager import ConnectionManager
from src.services.alert_service import AlertService


def _service() -> tuple[AlertService, MagicMock]:
    ws = MagicMock()
    ws.broadcast_alert = AsyncMock()
    return AlertService(ws), ws


async def test_low_threshold_model_triggers_below_070() -> None:
    service, ws = _service()
    payload = await service.process_prediction(
        {"probability": 0.30, "decision_threshold": 0.08}
    )
    assert payload["triggered"] is True
    assert payload["decision_threshold"] == 0.08
    ws.broadcast_alert.assert_awaited_once()


async def test_high_threshold_model_does_not_trigger_at_075() -> None:
    service, ws = _service()
    payload = await service.process_prediction(
        {"probability": 0.75, "decision_threshold": 0.905}
    )
    assert payload["triggered"] is False
    ws.broadcast_alert.assert_not_awaited()


async def test_threshold_is_inclusive_like_predicted_class() -> None:
    service, _ = _service()
    payload = await service.process_prediction(
        {"probability": 0.27, "decision_threshold": 0.27}
    )
    assert payload["triggered"] is True


async def test_legacy_rule_without_threshold() -> None:
    service, _ = _service()
    low = await service.process_prediction({"probability": 0.30})
    high = await service.process_prediction({"probability": 0.71})
    assert low["triggered"] is False and high["triggered"] is True
    assert low["decision_threshold"] is None


@pytest.mark.parametrize(
    ("probability", "threshold", "sent"),
    [(0.30, 0.08, True), (0.75, 0.905, False), (0.71, None, True), (0.5, None, False)],
)
async def test_ws_manager_gate_uses_payload_threshold(
    probability: float, threshold: float | None, sent: bool
) -> None:
    manager = ConnectionManager()
    manager.broadcast = AsyncMock()  # type: ignore[method-assign]
    await manager.broadcast_alert(
        {"probability": probability, "decision_threshold": threshold}
    )
    assert manager.broadcast.await_count == (1 if sent else 0)


def test_models_list_exposes_active_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.core.config import settings
    from src.main import create_app

    monkeypatch.setattr(settings, "admin_api_token", "t0ken-for-this-test")
    with TestClient(create_app()) as client:
        body = client.get(
            "/models", headers={"X-Admin-Token": "t0ken-for-this-test"}
        ).json()
    assert "active_threshold" in body
    if body["active_model"]:
        assert 0.0 < body["active_threshold"] < 1.0


def test_predictions_page_carries_active_threshold() -> None:
    """The dashboard already polls /v1/predictions every 5 s: the threshold of
    the active model rides along (no extra request)."""
    from src.schemas.prediction import PredictionPage

    page = PredictionPage(items=[], total=0, page=1, size=1, pages=0)
    assert page.active_threshold is None
    assert "active_threshold" in PredictionPage.model_fields
