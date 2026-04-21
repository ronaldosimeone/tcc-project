"""
Tests for WebSocket alert channel — S6.

Coverage matrix
---------------
ConnectionManager   connect, disconnect (idempotent), broadcast, dead-socket
                    eviction, send_personal success/failure
RF-14               threshold boundary (strict >), broadcast triggered / skipped
RNF-30              heartbeat task lifecycle, ping emission, 30 s constant
AlertService        payload shape, message_id uniqueness, trigger flag
Endpoint            connect/disconnect, ack round-trip, unknown frame resilience
Memory-leak QA      N connect/disconnect cycles, all-dead broadcast, task count
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketState

from src.core.ws_manager import (
    ALERT_PROBABILITY_THRESHOLD,
    HEARTBEAT_INTERVAL,
    ConnectionManager,
)
from src.routers.alerts_ws import router as ws_router
from src.services.alert_service import AlertService

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def manager() -> ConnectionManager:
    """Fresh, isolated ConnectionManager for every test."""
    return ConnectionManager()


@pytest.fixture()
def app(manager: ConnectionManager) -> FastAPI:
    """Minimal FastAPI app with the WS router, manager injected."""
    import src.routers.alerts_ws as _module

    _module.manager = manager  # override singleton for this test

    application = FastAPI()
    application.include_router(ws_router)
    return application


@pytest.fixture()
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_ws(state: WebSocketState = WebSocketState.CONNECTED) -> MagicMock:
    ws = MagicMock()
    ws.client_state = state
    ws.client = ("127.0.0.1", 9000)
    ws.send_json = AsyncMock()
    ws.accept = AsyncMock()
    return ws


# ===========================================================================
# ConnectionManager
# ===========================================================================


class TestConnectionManager:

    @pytest.mark.asyncio
    async def test_connect_accepts_and_registers(self, manager: ConnectionManager) -> None:
        ws = _make_ws()
        await manager.connect(ws)

        ws.accept.assert_awaited_once()
        assert manager.active_count == 1

    @pytest.mark.asyncio
    async def test_disconnect_removes_connection(self, manager: ConnectionManager) -> None:
        ws = _make_ws()
        await manager.connect(ws)
        manager.disconnect(ws)

        assert manager.active_count == 0

    @pytest.mark.asyncio
    async def test_disconnect_is_idempotent(self, manager: ConnectionManager) -> None:
        ws = _make_ws()
        manager.disconnect(ws)  # not registered — must not raise
        assert manager.active_count == 0

    @pytest.mark.asyncio
    async def test_broadcast_reaches_all_clients(self, manager: ConnectionManager) -> None:
        ws1, ws2, ws3 = _make_ws(), _make_ws(), _make_ws()
        for ws in (ws1, ws2, ws3):
            await manager.connect(ws)

        payload = {"type": "alert", "probability": 0.9}
        await manager.broadcast(payload)

        for ws in (ws1, ws2, ws3):
            ws.send_json.assert_awaited_once_with(payload)

    @pytest.mark.asyncio
    async def test_broadcast_evicts_dead_connections(self, manager: ConnectionManager) -> None:
        ws_ok = _make_ws()
        ws_dead = _make_ws()
        ws_dead.send_json = AsyncMock(side_effect=RuntimeError("broken pipe"))

        await manager.connect(ws_ok)
        await manager.connect(ws_dead)
        assert manager.active_count == 2

        await manager.broadcast({"type": "ping"})

        assert manager.active_count == 1
        ws_ok.send_json.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_broadcast_evicts_disconnected_state(self, manager: ConnectionManager) -> None:
        """WebSocket in DISCONNECTED state is removed without calling send_json."""
        ws = _make_ws(state=WebSocketState.DISCONNECTED)
        manager._active.add(ws)  # bypass connect() intentionally

        await manager.broadcast({"type": "test"})

        ws.send_json.assert_not_awaited()
        assert manager.active_count == 0

    @pytest.mark.asyncio
    async def test_broadcast_with_no_clients_is_silent(self, manager: ConnectionManager) -> None:
        await manager.broadcast({"type": "ping"})  # must not raise

    @pytest.mark.asyncio
    async def test_send_personal_returns_false_on_error(self, manager: ConnectionManager) -> None:
        ws = _make_ws()
        ws.send_json = AsyncMock(side_effect=OSError("reset by peer"))

        result = await manager.send_personal(ws, {"type": "x"})

        assert result is False

    @pytest.mark.asyncio
    async def test_send_personal_returns_true_on_success(self, manager: ConnectionManager) -> None:
        ws = _make_ws()
        result = await manager.send_personal(ws, {"type": "x"})

        assert result is True


# ===========================================================================
# RF-14 — broadcast_alert
# ===========================================================================


class TestRF14:

    @pytest.mark.asyncio
    @pytest.mark.parametrize("probability", [0.71, 0.85, 0.99, 1.0])
    async def test_alert_broadcast_above_threshold(
        self, manager: ConnectionManager, probability: float
    ) -> None:
        ws = _make_ws()
        await manager.connect(ws)

        await manager.broadcast_alert({"type": "alert", "probability": probability})

        ws.send_json.assert_awaited_once()
        assert ws.send_json.call_args[0][0]["probability"] == probability

    @pytest.mark.asyncio
    @pytest.mark.parametrize("probability", [0.0, 0.50, 0.699])
    async def test_alert_not_broadcast_below_threshold(
        self, manager: ConnectionManager, probability: float
    ) -> None:
        ws = _make_ws()
        await manager.connect(ws)

        await manager.broadcast_alert({"type": "alert", "probability": probability})

        ws.send_json.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_exact_threshold_does_not_trigger(self, manager: ConnectionManager) -> None:
        """Boundary: probability == 0.70 must NOT trigger (rule is strict >)."""
        ws = _make_ws()
        await manager.connect(ws)

        await manager.broadcast_alert(
            {"type": "alert", "probability": ALERT_PROBABILITY_THRESHOLD}
        )

        ws.send_json.assert_not_awaited()


# ===========================================================================
# RNF-30 — Heartbeat
# ===========================================================================


class TestHeartbeat:

    @pytest.mark.asyncio
    async def test_heartbeat_task_created_on_first_connect(
        self, manager: ConnectionManager
    ) -> None:
        ws = _make_ws()
        await manager.connect(ws)

        assert manager._heartbeat_task is not None
        assert not manager._heartbeat_task.done()

        manager.disconnect(ws)

    @pytest.mark.asyncio
    async def test_heartbeat_task_cancelled_when_empty(
        self, manager: ConnectionManager
    ) -> None:
        ws = _make_ws()
        await manager.connect(ws)
        task = manager._heartbeat_task

        manager.disconnect(ws)
        await asyncio.sleep(0)  # yield to let the cancellation propagate

        assert task.cancelled() or task.done()

    @pytest.mark.asyncio
    async def test_heartbeat_sends_ping(self, manager: ConnectionManager) -> None:
        ws = _make_ws()
        await manager.connect(ws)

        with patch("src.core.ws_manager.HEARTBEAT_INTERVAL", 0):
            manager._heartbeat_task.cancel()
            manager._heartbeat_task = asyncio.create_task(manager._heartbeat_loop())
            await asyncio.sleep(0.05)

        calls: list[Any] = ws.send_json.call_args_list
        assert any(c[0][0].get("type") == "ping" for c in calls)

        manager.disconnect(ws)

    def test_heartbeat_interval_is_30_seconds(self) -> None:
        assert HEARTBEAT_INTERVAL == 30

    @pytest.mark.asyncio
    async def test_reconnect_does_not_spawn_duplicate_tasks(
        self, manager: ConnectionManager
    ) -> None:
        ws1, ws2 = _make_ws(), _make_ws()

        await manager.connect(ws1)
        task_a = manager._heartbeat_task

        await manager.connect(ws2)
        task_b = manager._heartbeat_task

        # Second connect must reuse the running task, not create a new one.
        assert task_a is task_b or task_a.done()

        manager.disconnect(ws1)
        manager.disconnect(ws2)


# ===========================================================================
# AlertService
# ===========================================================================


class TestAlertService:

    @pytest.mark.asyncio
    async def test_prediction_above_threshold_triggers_broadcast(
        self, manager: ConnectionManager
    ) -> None:
        service = AlertService(manager)
        ws = _make_ws()
        await manager.connect(ws)

        result = await service.process_prediction(
            {"probability": 0.95, "label": "failure", "sensor_id": "S01"}
        )

        assert result["triggered"] is True
        ws.send_json.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_prediction_below_threshold_does_not_broadcast(
        self, manager: ConnectionManager
    ) -> None:
        service = AlertService(manager)
        ws = _make_ws()
        await manager.connect(ws)

        result = await service.process_prediction(
            {"probability": 0.50, "label": "normal", "sensor_id": "S02"}
        )

        assert result["triggered"] is False
        ws.send_json.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_payload_contains_required_fields(self, manager: ConnectionManager) -> None:
        service = AlertService(manager)
        result = await service.process_prediction({"probability": 0.3})

        for field in ("type", "message_id", "timestamp", "probability", "triggered"):
            assert field in result, f"Missing field: {field}"

    @pytest.mark.asyncio
    async def test_message_id_is_unique_per_call(self, manager: ConnectionManager) -> None:
        service = AlertService(manager)
        r1 = await service.process_prediction({"probability": 0.1})
        r2 = await service.process_prediction({"probability": 0.1})

        assert r1["message_id"] != r2["message_id"]


# ===========================================================================
# Endpoint /ws/alerts (integration via TestClient)
# ===========================================================================


class TestWebSocketEndpoint:

    def test_client_connects_successfully(self, client: TestClient) -> None:
        with client.websocket_connect("/ws/alerts"):
            pass  # clean connect + disconnect

    def test_ack_returns_confirmation(self, client: TestClient) -> None:
        with client.websocket_connect("/ws/alerts") as ws:
            ws.send_json({"type": "ack", "message_id": "abc-123"})
            response = ws.receive_json()

            assert response["type"] == "ack"
            assert response["message_id"] == "abc-123"
            assert response["status"] == "received"

    def test_pong_does_not_close_connection(self, client: TestClient) -> None:
        with client.websocket_connect("/ws/alerts") as ws:
            ws.send_json({"type": "pong"})
            # Immediately send a valid ack — connection must still be alive
            ws.send_json({"type": "ack", "message_id": "after-pong"})
            response = ws.receive_json()
            assert response["type"] == "ack"

    def test_unknown_frame_does_not_close_connection(self, client: TestClient) -> None:
        with client.websocket_connect("/ws/alerts") as ws:
            ws.send_json({"type": "mystery_event"})
            ws.send_json({"type": "ack", "message_id": "post-unknown"})
            response = ws.receive_json()
            assert response["type"] == "ack"

    def test_abrupt_disconnect_releases_connection(
        self, client: TestClient, manager: ConnectionManager
    ) -> None:
        with client.websocket_connect("/ws/alerts"):
            assert manager.active_count == 1

        assert manager.active_count == 0


# ===========================================================================
# Memory-leak QA
# ===========================================================================


class TestMemoryLeakQA:

    @pytest.mark.asyncio
    async def test_n_connect_disconnect_cycles_leave_zero_refs(
        self, manager: ConnectionManager
    ) -> None:
        for _ in range(50):
            ws = _make_ws()
            await manager.connect(ws)
            manager.disconnect(ws)

        assert manager.active_count == 0

    @pytest.mark.asyncio
    async def test_broadcast_to_all_dead_empties_active_set(
        self, manager: ConnectionManager
    ) -> None:
        for _ in range(10):
            ws = _make_ws()
            ws.send_json = AsyncMock(side_effect=OSError("reset"))
            await manager.connect(ws)

        assert manager.active_count == 10
        await manager.broadcast({"type": "alert", "probability": 0.95})
        assert manager.active_count == 0
