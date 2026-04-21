"""
WebSocket router — bidirectional alert channel (S6).

Endpoint : /ws/alerts
Protocol : JSON frames over WebSocket

Server → Client
    {"type": "alert",  "message_id": "...", "probability": 0.92, ...}
    {"type": "ping"}                    ← heartbeat every 30 s (RNF-30)

Client → Server
    {"type": "ack",  "message_id": "..."}   ← delivery confirmation
    {"type": "pong"}                        ← heartbeat reply

RF-14: alerts are pushed immediately when probability > 0.70.
RNF-30: server sends a ping frame every 30 s; dead connections are
         evicted automatically on the next failed send.
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from src.core.ws_manager import ConnectionManager, manager
from src.services.alert_service import AlertService

log = structlog.get_logger(__name__)

router = APIRouter(tags=["websocket"])


# ---------------------------------------------------------------------------
# Dependency injection
# ---------------------------------------------------------------------------


def get_ws_manager() -> ConnectionManager:
    """Return the module-level singleton ConnectionManager."""
    return manager


def get_alert_service(
    ws_manager: ConnectionManager = Depends(get_ws_manager),
) -> AlertService:
    return AlertService(ws_manager)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ack(message_id: str, status: str = "received") -> dict[str, Any]:
    return {"type": "ack", "message_id": message_id, "status": status}


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.websocket("/ws/alerts")
async def websocket_alerts(
    websocket: WebSocket,
    ws_manager: ConnectionManager = Depends(get_ws_manager),
) -> None:
    """
    Bidirectional alert channel.

    The ``finally`` block guarantees that every code path — clean close,
    abrupt disconnect, or unexpected exception — calls ``manager.disconnect``,
    so no WebSocket reference leaks into ``_active``.
    """
    await ws_manager.connect(websocket)
    try:
        while True:
            data: dict[str, Any] = await websocket.receive_json()
            msg_type: str = data.get("type", "")

            if msg_type == "pong":
                log.debug("ws_pong_received", client=str(websocket.client))
                continue

            if msg_type == "ack":
                message_id: str = data.get("message_id", "")
                log.info(
                    "ws_ack_received",
                    message_id=message_id,
                    client=str(websocket.client),
                )
                await ws_manager.send_personal(websocket, _ack(message_id))
                continue

            # Unknown frame type — log and ignore; do NOT close the connection.
            log.warning(
                "ws_unknown_frame",
                frame_type=msg_type,
                client=str(websocket.client),
            )

    except WebSocketDisconnect as exc:
        log.info(
            "ws_client_disconnected",
            code=exc.code,
            client=str(websocket.client),
        )
    except Exception as exc:  # noqa: BLE001
        log.error(
            "ws_unexpected_error",
            error=str(exc),
            client=str(websocket.client),
            exc_info=True,
        )
    finally:
        ws_manager.disconnect(websocket)
