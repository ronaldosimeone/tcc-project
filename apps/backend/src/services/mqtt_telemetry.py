"""
MQTT telemetry transport — industrial ingestion path (Industry 4.0).

Topic layout (one topic per asset, wildcard subscription on the backend):

    <prefix>/<asset_id>/telemetry          e.g. predictiq/fleet/APU-Trem-001/telemetry

Payload (UTF-8 JSON, versioned):

    {"schema": "predictiq.telemetry.v1", "asset_id": "APU-Trem-001",
     "reading": {<SensorReading fields, ISO timestamp>}}

Ingestion rules (``parse_message`` is pure and unit-tested):
    * the asset id in the TOPIC and in the PAYLOAD must match (a publisher
      cannot inject readings for another asset by forging the body);
    * the asset must be registered in ``assets`` (unknown ids are dropped,
      not auto-created);
    * schema version and every sensor value are validated by Pydantic.
Rejected messages are counted and logged, never processed.

QoS 1 (at-least-once): a duplicated reading only adds one buffer sample;
losing readings silently would be worse for failure detection.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import aiomqtt
import structlog
from pydantic import ValidationError
from sqlalchemy import select

from src.core.database import AsyncSessionFactory
from src.models.asset import Asset
from src.schemas.stream import SensorReading
from src.services.asset_processor import AssetProcessor

log = structlog.get_logger(__name__)

SCHEMA = "predictiq.telemetry.v1"
_ASSET_RE = re.compile(r"^[A-Za-z0-9-]{1,64}$")
_ASSET_REFRESH_S = 60.0


@dataclass(frozen=True)
class MqttConfig:
    host: str
    port: int = 1883
    username: str | None = None
    password: str | None = None
    prefix: str = "predictiq/fleet"


def topic_for(prefix: str, asset_id: str) -> str:
    return f"{prefix}/{asset_id}/telemetry"


def encode_message(asset_id: str, reading: SensorReading) -> bytes:
    return json.dumps(
        {
            "schema": SCHEMA,
            "asset_id": asset_id,
            "reading": reading.model_dump(mode="json"),
        }
    ).encode("utf-8")


def parse_message(topic: str, payload: bytes, prefix: str) -> tuple[str, SensorReading]:
    """Validate topic + payload. Raises ValueError with the reason."""
    parts = topic.split("/")
    prefix_parts = prefix.split("/")
    if (
        len(parts) != len(prefix_parts) + 2
        or parts[: len(prefix_parts)] != prefix_parts
        or parts[-1] != "telemetry"
    ):
        raise ValueError(f"unexpected topic {topic!r}")
    topic_asset = parts[len(prefix_parts)]
    if not _ASSET_RE.match(topic_asset):
        raise ValueError("invalid asset id in topic")
    try:
        body = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("payload is not UTF-8 JSON") from exc
    if not isinstance(body, dict) or body.get("schema") != SCHEMA:
        raise ValueError("unsupported schema")
    if body.get("asset_id") != topic_asset:
        raise ValueError("asset id in payload does not match topic")
    try:
        reading = SensorReading.model_validate(body.get("reading"))
    except ValidationError as exc:
        raise ValueError(f"invalid reading: {exc.error_count()} error(s)") from exc
    return topic_asset, reading


class MqttTelemetryPublisher:
    """Used by the fleet simulator when TELEMETRY_TRANSPORT=mqtt."""

    def __init__(self, config: MqttConfig) -> None:
        self._config = config
        self._client: aiomqtt.Client | None = None

    async def start(self) -> None:
        self._client = aiomqtt.Client(
            self._config.host,
            self._config.port,
            username=self._config.username,
            password=self._config.password,
            identifier="predictiq-fleet-simulator",
        )
        await self._client.__aenter__()

    async def stop(self) -> None:
        if self._client is not None:
            await self._client.__aexit__(None, None, None)
            self._client = None

    async def publish(self, asset_id: str, reading: SensorReading) -> None:
        if self._client is None:
            raise RuntimeError("publisher not started")
        await self._client.publish(
            topic_for(self._config.prefix, asset_id),
            encode_message(asset_id, reading),
            qos=1,
        )


@dataclass
class IngestStats:
    accepted: int = 0
    rejected: int = 0
    unknown_asset: int = 0
    last_error: str | None = None
    per_asset: dict[str, int] = field(default_factory=dict)


class MqttTelemetryIngestor:
    """Subscribes to ``<prefix>/+/telemetry`` and feeds one AssetProcessor per asset."""

    def __init__(
        self, config: MqttConfig, processor_factory: Callable[[str], AssetProcessor]
    ) -> None:
        self._config = config
        self._factory = processor_factory
        self._processors: dict[str, AssetProcessor] = {}
        self._known: set[str] = set()
        self._known_at: float = 0.0
        self._task: asyncio.Task[None] | None = None
        self.stats = IngestStats()

    async def _known_assets(self) -> set[str]:
        if time.monotonic() - self._known_at > _ASSET_REFRESH_S:
            async with AsyncSessionFactory() as db:
                self._known = set((await db.scalars(select(Asset.id))).all())
            self._known_at = time.monotonic()
        return self._known

    async def handle(self, topic: str, payload: bytes) -> bool:
        try:
            asset_id, reading = parse_message(topic, payload, self._config.prefix)
        except ValueError as exc:
            self.stats.rejected += 1
            self.stats.last_error = str(exc)
            log.warning("mqtt_message_rejected", topic=topic, reason=str(exc))
            return False
        if asset_id not in await self._known_assets():
            self.stats.unknown_asset += 1
            log.warning("mqtt_unknown_asset", asset_id=asset_id)
            return False
        processor = self._processors.get(asset_id)
        if processor is None:
            processor = self._processors[asset_id] = self._factory(asset_id)
        await processor.process(reading)
        self.stats.accepted += 1
        self.stats.per_asset[asset_id] = self.stats.per_asset.get(asset_id, 0) + 1
        return True

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="mqtt-ingestor")

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run(self) -> None:
        backoff = 1.0
        while True:
            try:
                async with aiomqtt.Client(
                    self._config.host,
                    self._config.port,
                    username=self._config.username,
                    password=self._config.password,
                    identifier="predictiq-ingestor",
                ) as client:
                    await client.subscribe(f"{self._config.prefix}/+/telemetry", qos=1)
                    log.info(
                        "mqtt_ingestor_subscribed",
                        host=self._config.host,
                        prefix=self._config.prefix,
                    )
                    backoff = 1.0
                    async for message in client.messages:
                        payload = (
                            message.payload
                            if isinstance(message.payload, bytes)
                            else str(message.payload).encode()
                        )
                        await self.handle(str(message.topic), payload)
            except aiomqtt.MqttError as exc:
                self.stats.last_error = str(exc)
                log.warning(
                    "mqtt_ingestor_disconnected", error=str(exc), retry_in_s=backoff
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
