"""
Alert service — business logic for RF-14.

Responsible for:
- Enriching raw ML prediction payloads with alert metadata.
- Delegating WebSocket broadcast to ConnectionManager when probability > 0.70.

Deliberately decoupled from the transport layer so it can be tested without
a real WebSocket connection.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import structlog

from src.core.ws_manager import ALERT_PROBABILITY_THRESHOLD, ConnectionManager

log = structlog.get_logger(__name__)


class AlertService:
    """
    Orchestrates alert creation and push delivery.

    Injected into routers via ``fastapi.Depends`` — never instantiated
    directly outside of tests.
    """

    def __init__(self, ws_manager: ConnectionManager) -> None:
        self._manager = ws_manager

    async def process_prediction(self, prediction: dict[str, Any]) -> dict[str, Any]:
        """
        Enrich a raw model prediction and broadcast an alert if RF-14 fires.

        Parameters
        ----------
        prediction:
            Raw output from the ML pipeline.  Expected keys:
            ``probability`` (float), ``label`` (str), ``sensor_id`` (str).

        Returns
        -------
        dict
            Enriched alert payload (also persisted by the caller).
        """
        probability: float = float(prediction.get("probability", 0.0))
        alert_payload: dict[str, Any] = {
            "type": "alert",
            "message_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "probability": probability,
            "label": prediction.get("label", "unknown"),
            "sensor_id": prediction.get("sensor_id"),
            "triggered": probability > ALERT_PROBABILITY_THRESHOLD,
        }

        if alert_payload["triggered"]:
            log.info(
                "alert_triggered",
                message_id=alert_payload["message_id"],
                probability=probability,
                sensor_id=alert_payload["sensor_id"],
            )
            await self._manager.broadcast_alert(alert_payload)
        else:
            log.debug(
                "alert_skipped",
                probability=probability,
                threshold=ALERT_PROBABILITY_THRESHOLD,
            )

        return alert_payload
