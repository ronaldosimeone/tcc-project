"""
SSE endpoint — real-time sensor data (RF-12, RNF-28).

GET /stream/sensors  →  exposed externally as /api/stream/sensors via Nginx.

Disconnect handling
-------------------
The async generator checks `request.is_disconnected()` (non-blocking poll via
anyio.move_on_after(0)) before every yield.  When the ASGI layer closes the
response body iterator it also calls aclose() on the generator, guaranteeing
the `finally` block runs and the queue is unregistered — no memory leak.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

import structlog
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from src.schemas.stream import SensorReading
from src.services.sensor_stream_service import (
    SensorStreamService,
    get_sensor_stream_service,
)

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/stream", tags=["stream"])

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",  # disables Nginx buffering (nginx.conf already sets this)
}


@router.get(
    "/sensors",
    summary="Real-time sensor readings via SSE (RF-12)",
    response_description="text/event-stream — one sensor_reading event per second.",
)
async def stream_sensors(
    request: Request,
    service: SensorStreamService = Depends(get_sensor_stream_service),
) -> StreamingResponse:
    """
    RF-12: emits all 12 sensor features every 1 second.
    RNF-28: each client owns an isolated asyncio.Queue; 50+ clients are
            served by a single shared broadcast task.

    SSE wire format::

        event: sensor_reading
        data: {"timestamp": "...", "TP2": 5.87, ...}
        id: <epoch_ms>
        retry: 3000

    """

    async def event_generator() -> AsyncGenerator[str, None]:
        queue: asyncio.Queue[SensorReading] = service.subscribe()
        log.info("sse_connection_opened", client=str(request.client))
        try:
            while True:
                if await request.is_disconnected():
                    log.info("sse_client_disconnected", client=str(request.client))
                    break
                try:
                    reading: SensorReading = await asyncio.wait_for(
                        queue.get(), timeout=2.0
                    )
                except asyncio.TimeoutError:
                    # No reading yet (e.g. broadcast task restarting); loop back.
                    continue

                epoch_ms = int(reading.timestamp.timestamp() * 1000)
                yield (
                    f"event: sensor_reading\n"
                    f"data: {reading.model_dump_json()}\n"
                    f"id: {epoch_ms}\n"
                    f"retry: 3000\n\n"
                )
        finally:
            service.unsubscribe(queue)
            log.info("sse_connection_closed", client=str(request.client))

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )
