"""
Connection manager for WebSocket alert broadcasting.

Responsibilities:
- Register / deregister active WebSocket connections.
- Broadcast JSON payloads to all connected clients (RF-14).
- Run a periodic heartbeat task every 30 s (RNF-30).
- Remove dead connections immediately — no memory leak.
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog
from fastapi import WebSocket
from starlette.websockets import WebSocketState

log = structlog.get_logger(__name__)

HEARTBEAT_INTERVAL: int = 30  # seconds — RNF-30
ALERT_PROBABILITY_THRESHOLD: float = 0.70  # RF-14 (strict >)


class ConnectionManager:
    """
    Registry of active WebSocket connections.

    Thread-safety: all access happens within a single asyncio event-loop,
    so a plain ``set`` is sufficient — no threading.Lock required.
    """

    def __init__(self) -> None:
        self._active: set[WebSocket] = set()
        self._heartbeat_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register a new connection."""
        await websocket.accept()
        self._active.add(websocket)
        log.info("ws_connected", total_active=len(self._active), client=str(websocket.client))
        self._ensure_heartbeat()

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a connection from the registry (idempotent)."""
        self._active.discard(websocket)
        log.info("ws_disconnected", total_active=len(self._active), client=str(websocket.client))
        if not self._active:
            self._cancel_heartbeat()

    # ------------------------------------------------------------------
    # Send helpers
    # ------------------------------------------------------------------

    async def send_personal(self, websocket: WebSocket, payload: dict[str, Any]) -> bool:
        """
        Send a JSON payload to a single client.

        Returns False and removes the connection on any failure so that
        dead sockets never accumulate in ``_active``.
        """
        if websocket.client_state != WebSocketState.CONNECTED:
            self.disconnect(websocket)
            return False
        try:
            await websocket.send_json(payload)
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("ws_send_failed", client=str(websocket.client), error=str(exc))
            self.disconnect(websocket)
            return False

    async def broadcast(self, payload: dict[str, Any]) -> None:
        """
        Send a JSON payload to every active client.

        Iterates over a snapshot of ``_active`` so that ``send_personal``
        can safely mutate the set during the loop.
        """
        if not self._active:
            return

        dead_count = 0
        for ws in list(self._active):
            ok = await self.send_personal(ws, payload)
            if not ok:
                dead_count += 1

        if dead_count:
            log.debug("ws_dead_connections_removed", count=dead_count)

    async def broadcast_alert(self, payload: dict[str, Any]) -> None:
        """
        RF-14: broadcast immediately when ``probability`` exceeds the threshold.

        Silently skips when probability <= ALERT_PROBABILITY_THRESHOLD.
        """
        probability: float = float(payload.get("probability", 0.0))
        if probability <= ALERT_PROBABILITY_THRESHOLD:
            return

        log.info(
            "ws_alert_broadcast",
            probability=probability,
            threshold=ALERT_PROBABILITY_THRESHOLD,
            active_clients=len(self._active),
        )
        await self.broadcast(payload)

    # ------------------------------------------------------------------
    # Heartbeat — RNF-30
    # ------------------------------------------------------------------

    def _ensure_heartbeat(self) -> None:
        """Start the heartbeat task if it is not already running."""
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(), name="ws-heartbeat"
            )
            log.debug("ws_heartbeat_started")

    def _cancel_heartbeat(self) -> None:
        """Cancel the heartbeat task when there are no more clients."""
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            log.debug("ws_heartbeat_cancelled")

    async def _heartbeat_loop(self) -> None:
        """
        Send a ``{"type": "ping"}`` to all clients every HEARTBEAT_INTERVAL
        seconds.  Clients should reply with ``{"type": "pong"}``.

        Dead connections are detected implicitly on the next
        ``broadcast`` / ``send_personal`` call — no extra bookkeeping needed.
        """
        try:
            while True:
                await asyncio.sleep(HEARTBEAT_INTERVAL)
                if not self._active:
                    break
                await self.broadcast({"type": "ping"})
                log.debug("ws_heartbeat_ping_sent", clients=len(self._active))
        except asyncio.CancelledError:
            log.debug("ws_heartbeat_loop_stopped")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def active_count(self) -> int:
        """Number of currently connected clients."""
        return len(self._active)


# Module-level singleton consumed by the router and AlertService.
manager = ConnectionManager()
