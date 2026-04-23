"""
Tests for GET /stream/sensors (SSE endpoint).

Coverage matrix
---------------
SensorStreamService   subscribe/unsubscribe lifecycle, subscriber_count,
                      broadcast task start/stop, slow-consumer eviction,
                      _generate_reading field completeness and types.
SSE wire format       event:, data:, id:, retry: lines; valid JSON payload;
                      all 12 sensor fields present.
Response headers      status_code, Content-Type, Cache-Control, X-Accel-Buffering.
Cleanup / no-leak     subscriber_count == 0 after generator exits; idempotent
                      unsubscribe; N connect/disconnect cycles leave no refs.

Note on httpx + ASGITransport
------------------------------
httpx.ASGITransport.handle_async_request() calls ``await self.app(scope, receive,
send)`` and collects ALL body chunks into a list before returning.  This makes it
incompatible with infinite SSE streams — the request would never complete.

Wire-format and cleanup tests therefore invoke ``stream_sensors()`` directly and
drive the async generator via ``response.body_iterator``, using a mock request
whose ``is_disconnected()`` returns True after N iterations.  This lets the
generator exit cleanly, the body iterator is exhausted, and tests run in < 100 ms.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.responses import StreamingResponse

from src.schemas.stream import SensorReading
from src.services.sensor_stream_service import (
    BROADCAST_INTERVAL,
    SensorStreamService,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def service() -> SensorStreamService:
    """Fresh, isolated SensorStreamService for every test."""
    return SensorStreamService()


def _make_mock_request(disconnect_after: int = 1) -> MagicMock:
    """
    Build a mock Starlette ``Request`` whose ``is_disconnected()`` coroutine
    returns False for the first ``disconnect_after`` calls and True thereafter.

    disconnect_after=0  → disconnect immediately (no events yielded)
    disconnect_after=1  → allow one loop iteration  (one SSE event)
    disconnect_after=N  → allow N loop iterations   (N SSE events)
    """
    call_count = 0

    async def is_disconnected() -> bool:
        nonlocal call_count
        call_count += 1
        return call_count > disconnect_after

    mock = MagicMock()
    mock.is_disconnected = is_disconnected
    mock.client = MagicMock()
    mock.client.__str__ = MagicMock(return_value="127.0.0.1:0")  # type: ignore[method-assign]
    return mock


async def _drive_generator(
    service: SensorStreamService,
    disconnect_after: int = 1,
    fast_interval: float = 0.02,
) -> list[str]:
    """
    Invoke stream_sensors() directly and collect the raw SSE chunks the
    generator emits before the mock request disconnects.

    Returns a list of decoded chunk strings (each chunk is one full SSE event,
    i.e. "event: ...\ndata: ...\nid: ...\nretry: 3000\n\n").
    """
    from src.routers.stream import stream_sensors

    mock_request = _make_mock_request(disconnect_after=disconnect_after)

    with patch("src.services.sensor_stream_service.BROADCAST_INTERVAL", fast_interval):
        response: StreamingResponse = await stream_sensors(mock_request, service)
        chunks: list[str] = []
        async for raw in response.body_iterator:
            if raw:
                chunks.append(raw.decode() if isinstance(raw, bytes) else raw)
        return chunks


# ---------------------------------------------------------------------------
# Helper: parse SSE lines from a raw chunk
# ---------------------------------------------------------------------------


def _parse_event_lines(chunk: str) -> list[str]:
    """Return non-empty lines from one SSE event chunk."""
    return [line for line in chunk.split("\n") if line.strip()]


# ===========================================================================
# SensorStreamService — unit tests
# ===========================================================================


class TestSensorStreamService:

    def test_initial_subscriber_count_is_zero(
        self, service: SensorStreamService
    ) -> None:
        assert service.subscriber_count == 0

    @pytest.mark.asyncio
    async def test_subscribe_increments_count(
        self, service: SensorStreamService
    ) -> None:
        q = service.subscribe()
        assert service.subscriber_count == 1
        service.unsubscribe(q)

    @pytest.mark.asyncio
    async def test_unsubscribe_decrements_count(
        self, service: SensorStreamService
    ) -> None:
        q = service.subscribe()
        service.unsubscribe(q)
        assert service.subscriber_count == 0

    @pytest.mark.asyncio
    async def test_unsubscribe_is_idempotent(
        self, service: SensorStreamService
    ) -> None:
        q = service.subscribe()
        service.unsubscribe(q)
        service.unsubscribe(q)  # second call must not raise
        assert service.subscriber_count == 0

    @pytest.mark.asyncio
    async def test_broadcast_task_starts_on_subscribe(
        self, service: SensorStreamService
    ) -> None:
        q = service.subscribe()
        assert service._broadcast_task is not None
        assert not service._broadcast_task.done()
        service.unsubscribe(q)

    @pytest.mark.asyncio
    async def test_broadcast_task_stops_when_empty(
        self, service: SensorStreamService
    ) -> None:
        q = service.subscribe()
        task = service._broadcast_task
        service.unsubscribe(q)
        await asyncio.sleep(0)  # yield so the cancellation propagates
        assert task is not None
        assert task.cancelled() or task.done()

    @pytest.mark.asyncio
    async def test_second_subscribe_reuses_running_task(
        self, service: SensorStreamService
    ) -> None:
        q1 = service.subscribe()
        task_a = service._broadcast_task

        q2 = service.subscribe()
        task_b = service._broadcast_task

        assert task_a is task_b or (task_a is not None and task_a.done())

        service.unsubscribe(q1)
        service.unsubscribe(q2)

    @pytest.mark.asyncio
    async def test_broadcast_delivers_reading_to_queue(
        self, service: SensorStreamService
    ) -> None:
        with patch("src.services.sensor_stream_service.BROADCAST_INTERVAL", 0.02):
            q = service.subscribe()
            reading = await asyncio.wait_for(q.get(), timeout=1.0)
            assert isinstance(reading, SensorReading)
            service.unsubscribe(q)

    @pytest.mark.asyncio
    async def test_multiple_subscribers_each_receive_reading(
        self, service: SensorStreamService
    ) -> None:
        with patch("src.services.sensor_stream_service.BROADCAST_INTERVAL", 0.02):
            queues = [service.subscribe() for _ in range(5)]
            for q in queues:
                reading = await asyncio.wait_for(q.get(), timeout=1.0)
                assert isinstance(reading, SensorReading)
            for q in queues:
                service.unsubscribe(q)

    @pytest.mark.asyncio
    async def test_slow_consumer_is_evicted(self, service: SensorStreamService) -> None:
        """A full queue is discarded so it never blocks the broadcaster."""
        q = service.subscribe()
        # Fill the queue to capacity without consuming
        for _ in range(10):
            try:
                q.put_nowait(service._generate_reading())
            except asyncio.QueueFull:
                break

        with patch("src.services.sensor_stream_service.BROADCAST_INTERVAL", 0.02):
            # Trigger one broadcast cycle; the full queue should be evicted.
            await asyncio.sleep(0.10)

        assert service.subscriber_count == 0

    def test_generate_reading_returns_sensor_reading(
        self, service: SensorStreamService
    ) -> None:
        reading = service._generate_reading()
        assert isinstance(reading, SensorReading)

    def test_generate_reading_has_all_12_fields(
        self, service: SensorStreamService
    ) -> None:
        expected = {
            "TP2",
            "TP3",
            "H1",
            "DV_pressure",
            "Reservoirs",
            "Motor_current",
            "Oil_temperature",
            "COMP",
            "DV_eletric",
            "Towers",
            "MPG",
            "Oil_level",
        }
        reading = service._generate_reading()
        assert expected.issubset(reading.model_dump().keys())

    def test_generate_reading_all_fields_are_float(
        self, service: SensorStreamService
    ) -> None:
        reading = service._generate_reading()
        data = reading.model_dump()
        data.pop("timestamp")
        assert all(isinstance(v, float) for v in data.values())

    def test_generate_reading_binary_sensors_are_0_or_1(
        self, service: SensorStreamService
    ) -> None:
        binary_fields = {"COMP", "DV_eletric", "Towers", "MPG", "Oil_level"}
        for _ in range(50):
            reading = service._generate_reading()
            data = reading.model_dump()
            for field in binary_fields:
                assert data[field] in (0.0, 1.0), f"{field}={data[field]} not binary"

    def test_broadcast_interval_is_one_second(self) -> None:
        assert BROADCAST_INTERVAL == 1.0


# ===========================================================================
# SSE response object — headers and status code
# ===========================================================================


class TestSSEResponseObject:
    """
    Inspect the StreamingResponse object returned by stream_sensors() without
    driving the infinite generator.  Disconnect immediately so the generator
    exits after zero events; body_iterator drains instantly.
    """

    @pytest.mark.asyncio
    async def test_status_code_is_200(self, service: SensorStreamService) -> None:
        from src.routers.stream import stream_sensors

        mock_req = _make_mock_request(disconnect_after=0)
        response = await stream_sensors(mock_req, service)
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_content_type_is_event_stream(
        self, service: SensorStreamService
    ) -> None:
        from src.routers.stream import stream_sensors

        mock_req = _make_mock_request(disconnect_after=0)
        response = await stream_sensors(mock_req, service)
        assert response.media_type == "text/event-stream"

    @pytest.mark.asyncio
    async def test_cache_control_header(self, service: SensorStreamService) -> None:
        from src.routers.stream import stream_sensors

        mock_req = _make_mock_request(disconnect_after=0)
        response = await stream_sensors(mock_req, service)
        # Drain the iterator so the finally block runs and cleanup completes.
        async for _ in response.body_iterator:
            pass
        assert response.headers.get("cache-control") == "no-cache"

    @pytest.mark.asyncio
    async def test_x_accel_buffering_header(self, service: SensorStreamService) -> None:
        from src.routers.stream import stream_sensors

        mock_req = _make_mock_request(disconnect_after=0)
        response = await stream_sensors(mock_req, service)
        async for _ in response.body_iterator:
            pass
        assert response.headers.get("x-accel-buffering") == "no"


# ===========================================================================
# SSE wire format — driven via mock request
# ===========================================================================


class TestSSEWireFormat:
    """
    Verify the SSE wire format (event:, data:, id:, retry:) by driving the
    generator directly through its body_iterator with a fast broadcast interval
    and a mock request that disconnects after the desired number of events.
    """

    @pytest.mark.asyncio
    async def test_first_event_has_event_line(
        self, service: SensorStreamService
    ) -> None:
        chunks = await _drive_generator(service, disconnect_after=1)
        assert len(chunks) >= 1
        lines = _parse_event_lines(chunks[0])
        assert lines[0] == "event: sensor_reading"

    @pytest.mark.asyncio
    async def test_first_event_has_data_line(
        self, service: SensorStreamService
    ) -> None:
        chunks = await _drive_generator(service, disconnect_after=1)
        lines = _parse_event_lines(chunks[0])
        assert lines[1].startswith("data: ")

    @pytest.mark.asyncio
    async def test_first_event_has_numeric_id_line(
        self, service: SensorStreamService
    ) -> None:
        chunks = await _drive_generator(service, disconnect_after=1)
        lines = _parse_event_lines(chunks[0])
        assert lines[2].startswith("id: ")
        epoch_str = lines[2].removeprefix("id: ")
        assert epoch_str.isdigit(), f"id is not numeric: {epoch_str!r}"

    @pytest.mark.asyncio
    async def test_first_event_has_retry_line(
        self, service: SensorStreamService
    ) -> None:
        chunks = await _drive_generator(service, disconnect_after=1)
        lines = _parse_event_lines(chunks[0])
        assert lines[3] == "retry: 3000"

    @pytest.mark.asyncio
    async def test_data_is_valid_json(self, service: SensorStreamService) -> None:
        chunks = await _drive_generator(service, disconnect_after=1)
        lines = _parse_event_lines(chunks[0])
        raw_json = lines[1].removeprefix("data: ")
        payload: dict[str, Any] = json.loads(raw_json)
        assert isinstance(payload, dict)

    @pytest.mark.asyncio
    async def test_payload_contains_all_12_sensor_fields(
        self, service: SensorStreamService
    ) -> None:
        chunks = await _drive_generator(service, disconnect_after=1)
        lines = _parse_event_lines(chunks[0])
        payload = json.loads(lines[1].removeprefix("data: "))

        expected = {
            "timestamp",
            "TP2",
            "TP3",
            "H1",
            "DV_pressure",
            "Reservoirs",
            "Motor_current",
            "Oil_temperature",
            "COMP",
            "DV_eletric",
            "Towers",
            "MPG",
            "Oil_level",
        }
        assert expected.issubset(
            payload.keys()
        ), f"Missing fields: {expected - payload.keys()}"

    @pytest.mark.asyncio
    async def test_payload_parses_as_sensor_reading(
        self, service: SensorStreamService
    ) -> None:
        """Round-trip: raw data line must deserialise into a valid SensorReading."""
        chunks = await _drive_generator(service, disconnect_after=1)
        lines = _parse_event_lines(chunks[0])
        raw_json = lines[1].removeprefix("data: ")
        reading = SensorReading.model_validate_json(raw_json)
        assert isinstance(reading.TP2, float)
        assert isinstance(reading.timestamp.year, int)

    @pytest.mark.asyncio
    async def test_consecutive_events_have_increasing_ids(
        self, service: SensorStreamService
    ) -> None:
        chunks = await _drive_generator(service, disconnect_after=2)
        assert len(chunks) >= 2
        id1 = int(_parse_event_lines(chunks[0])[2].removeprefix("id: "))
        id2 = int(_parse_event_lines(chunks[1])[2].removeprefix("id: "))
        assert id2 >= id1, f"id did not increase: {id1} → {id2}"

    @pytest.mark.asyncio
    async def test_event_ends_with_double_newline(
        self, service: SensorStreamService
    ) -> None:
        """Each SSE event must be terminated with \\n\\n."""
        chunks = await _drive_generator(service, disconnect_after=1)
        assert chunks[0].endswith("\n\n"), repr(chunks[0][-4:])


# ===========================================================================
# Cleanup / no memory-leak
# ===========================================================================


class TestCleanup:

    @pytest.mark.asyncio
    async def test_subscriber_count_drops_to_zero_after_generator_exits(
        self, service: SensorStreamService
    ) -> None:
        """After the generator exhausts (mock disconnect), subscriber_count == 0."""
        await _drive_generator(service, disconnect_after=1)
        # The finally block in event_generator calls service.unsubscribe(queue).
        await asyncio.sleep(0.05)  # let the finally block run
        assert service.subscriber_count == 0

    @pytest.mark.asyncio
    async def test_n_connect_disconnect_cycles_leave_zero_refs(
        self, service: SensorStreamService
    ) -> None:
        for _ in range(50):
            q = service.subscribe()
            service.unsubscribe(q)

        assert service.subscriber_count == 0

    @pytest.mark.asyncio
    async def test_unsubscribe_unknown_queue_does_not_raise(
        self, service: SensorStreamService
    ) -> None:
        alien_q: asyncio.Queue[SensorReading] = asyncio.Queue()
        service.unsubscribe(alien_q)  # must not raise
        assert service.subscriber_count == 0

    @pytest.mark.asyncio
    async def test_multiple_generators_all_cleaned_up(
        self, service: SensorStreamService
    ) -> None:
        """Drive 5 generators concurrently; all queues must be removed on exit."""
        tasks = [
            asyncio.create_task(_drive_generator(service, disconnect_after=1))
            for _ in range(5)
        ]
        await asyncio.gather(*tasks)
        await asyncio.sleep(0.05)
        assert service.subscriber_count == 0
