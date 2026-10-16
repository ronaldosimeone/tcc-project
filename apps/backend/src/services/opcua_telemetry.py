"""
OPC UA telemetry transport (IEC 62541) — alternative to MQTT.

Address space (namespace ``urn:predictiq:fleet``)::

    Objects/
      Fleet/
        <asset_id>/                 one object per asset (browse name = asset id)
          TP2 … Oil_level           Double, one typed variable per sensor
          Timestamp                 DateTime of the sample
          Sequence                  UInt64, incremented AFTER the sample is written

"Trigger node" pattern: the gateway writes the 12 sensors and the timestamp,
then bumps ``Sequence``. The ingestor subscribes only to the ``Sequence``
nodes; on each change it reads the whole sample in ONE Read service call and
feeds the asset's ``AssetProcessor``. Typed variables (not a JSON blob) keep
the address space browsable by any SCADA/HMI client.

Ingestion rules — same contract as MQTT:
    * the asset id comes from the browse path, never from the data;
    * only assets registered in ``assets`` are subscribed;
    * every value is validated by ``SensorReading`` (Pydantic);
    * ``Sequence`` must strictly increase per asset (duplicate notifications
      after a reconnect are dropped).

Security: this module runs the gateway with ``NoSecurity`` and optional
username/password — enough for the simulator inside the Docker network.
A real deployment needs SecurityPolicy Basic256Sha256 with certificates
(IEC 62443 FR1/FR4); listed as a gap in docs/pesquisa/conformidade-normas.md.

``OpcUaGatewayServer`` is the SIMULATED field gateway (the fleet simulator
publishes into it). ``OpcUaTelemetryIngestor`` is the production side: point
``OPCUA_ENDPOINT`` at a real gateway and it works unchanged.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog
from asyncua import Client, Node, Server, ua
from asyncua.server.users import User, UserRole
from pydantic import ValidationError
from sqlalchemy import select

from src.core.database import AsyncSessionFactory
from src.core.metrics import TELEMETRY_MESSAGES_TOTAL
from src.models.asset import Asset
from src.schemas.stream import SensorReading
from src.services.asset_processor import AssetProcessor

log = structlog.get_logger(__name__)

NAMESPACE = "urn:predictiq:fleet"
FLEET_FOLDER = "Fleet"
SENSOR_FIELDS: tuple[str, ...] = (
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
)
_ASSET_REFRESH_S = 60.0


@dataclass(frozen=True)
class OpcUaConfig:
    endpoint: str = "opc.tcp://0.0.0.0:4840/predictiq/"
    username: str | None = None
    password: str | None = None
    publishing_interval_ms: int = 500


def reading_from_values(values: Sequence[Any]) -> SensorReading:
    """``values`` = [Timestamp, *SENSOR_FIELDS] as read from the server.

    Raises ValueError on a missing/invalid value — never returns a partial
    reading.
    """
    if len(values) != len(SENSOR_FIELDS) + 1:
        raise ValueError(f"expected {len(SENSOR_FIELDS) + 1} values, got {len(values)}")
    ts, *sensors = values
    if not isinstance(ts, datetime):
        raise ValueError("Timestamp is not a DateTime")
    if ts.tzinfo is None:  # OPC UA DateTime is UTC
        ts = ts.replace(tzinfo=timezone.utc)
    try:
        return SensorReading(timestamp=ts, **dict(zip(SENSOR_FIELDS, sensors)))
    except ValidationError as exc:
        raise ValueError(f"invalid sample: {exc.errors()[0]['msg']}") from exc


# ---------------------------------------------------------------------------
# Simulated field gateway
# ---------------------------------------------------------------------------


class _SingleUserManager:
    """Accepts exactly one username/password (asyncua UserManager protocol)."""

    def __init__(self, username: str, password: str) -> None:
        self._username, self._password = username, password

    def get_user(
        self,
        iserver: Any,
        username: str | None = None,
        password: str | None = None,
        certificate: Any = None,
    ) -> User | None:
        if username == self._username and password == self._password:
            return User(role=UserRole.User)
        return None


@dataclass
class _AssetNodes:
    variables: dict[str, Node]
    timestamp: Node
    sequence: Node
    counter: int = 0


class OpcUaGatewayServer:
    """Simulated PLC/edge gateway. Implements the fleet's TelemetryPublisher."""

    def __init__(self, config: OpcUaConfig) -> None:
        self._config = config
        auth = config.username is not None and config.password is not None
        self._server = Server(
            user_manager=(
                _SingleUserManager(config.username or "", config.password or "")
                if auth
                else None
            )
        )
        self._idx = 0
        self._fleet: Node | None = None
        self._assets: dict[str, _AssetNodes] = {}
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        await self._server.init()
        self._server.set_endpoint(self._config.endpoint)
        self._server.set_server_name("PredictIQ field gateway (simulated)")
        self._server.set_security_policy([ua.SecurityPolicyType.NoSecurity])
        if self._config.username is not None:
            self._server.set_security_IDs(["Username"])
        self._idx = await self._server.register_namespace(NAMESPACE)
        self._fleet = await self._server.nodes.objects.add_folder(
            self._idx, FLEET_FOLDER
        )
        await self._server.start()
        log.info("opcua_gateway_started", endpoint=self._config.endpoint)

    async def stop(self) -> None:
        await self._server.stop()

    async def _nodes_for(self, asset_id: str) -> _AssetNodes:
        async with self._lock:
            nodes = self._assets.get(asset_id)
            if nodes is not None:
                return nodes
            if self._fleet is None:
                raise RuntimeError("gateway not started")
            obj = await self._fleet.add_object(self._idx, asset_id)
            variables = {
                name: await obj.add_variable(
                    self._idx, name, ua.Variant(0.0, ua.VariantType.Double)
                )
                for name in SENSOR_FIELDS
            }
            ts = await obj.add_variable(
                self._idx,
                "Timestamp",
                ua.Variant(datetime.now(timezone.utc), ua.VariantType.DateTime),
            )
            seq = await obj.add_variable(
                self._idx, "Sequence", ua.Variant(0, ua.VariantType.UInt64)
            )
            nodes = self._assets[asset_id] = _AssetNodes(variables, ts, seq)
            return nodes

    async def publish(self, asset_id: str, reading: SensorReading) -> None:
        nodes = await self._nodes_for(asset_id)
        for name, node in nodes.variables.items():
            await node.write_value(
                ua.Variant(float(getattr(reading, name)), ua.VariantType.Double)
            )
        await nodes.timestamp.write_value(
            ua.Variant(reading.timestamp, ua.VariantType.DateTime)
        )
        nodes.counter += 1  # trigger last: the sample is complete
        await nodes.sequence.write_value(
            ua.Variant(nodes.counter, ua.VariantType.UInt64)
        )


# ---------------------------------------------------------------------------
# Ingestor (production side)
# ---------------------------------------------------------------------------


@dataclass
class OpcUaIngestStats:
    accepted: int = 0
    rejected: int = 0
    duplicates: int = 0
    last_error: str | None = None
    subscribed_assets: int = 0
    per_asset: dict[str, int] = field(default_factory=dict)


@dataclass
class _Subscribed:
    asset_id: str
    sample_nodes: list[Node]  # [Timestamp, *SENSOR_FIELDS]
    last_sequence: int = -1


class OpcUaTelemetryIngestor:
    """Subscribes to every registered asset's ``Sequence`` node."""

    def __init__(
        self, config: OpcUaConfig, processor_factory: Callable[[str], AssetProcessor]
    ) -> None:
        self._config = config
        self._factory = processor_factory
        self._processors: dict[str, AssetProcessor] = {}
        self._by_node: dict[str, _Subscribed] = {}
        self._queue: asyncio.Queue[tuple[str, int]] = asyncio.Queue(maxsize=10_000)
        self._task: asyncio.Task[None] | None = None
        self.stats = OpcUaIngestStats()

    # asyncua SubHandler protocol — called from the client's receive loop.
    def datachange_notification(self, node: Node, val: Any, data: Any) -> None:
        try:
            self._queue.put_nowait((node.nodeid.to_string(), int(val)))
        except asyncio.QueueFull:
            self.stats.rejected += 1
            TELEMETRY_MESSAGES_TOTAL.labels(transport="opcua", result="dropped").inc()

    async def _known_assets(self) -> set[str]:
        async with AsyncSessionFactory() as db:
            return set((await db.scalars(select(Asset.id))).all())

    async def handle(self, client: Client, node_id: str, sequence: int) -> bool:
        sub = self._by_node.get(node_id)
        if sub is None:
            return False
        if sequence <= sub.last_sequence:
            self.stats.duplicates += 1
            return False
        sub.last_sequence = sequence
        try:
            values = await client.read_values(sub.sample_nodes)
            reading = reading_from_values(values)
        except (ValueError, ua.UaError) as exc:
            self.stats.rejected += 1
            self.stats.last_error = str(exc)
            TELEMETRY_MESSAGES_TOTAL.labels(transport="opcua", result="rejected").inc()
            log.warning("opcua_sample_rejected", asset_id=sub.asset_id, reason=str(exc))
            return False
        processor = self._processors.get(sub.asset_id)
        if processor is None:
            processor = self._processors[sub.asset_id] = self._factory(sub.asset_id)
        await processor.process(reading)
        self.stats.accepted += 1
        TELEMETRY_MESSAGES_TOTAL.labels(transport="opcua", result="accepted").inc()
        self.stats.per_asset[sub.asset_id] = (
            self.stats.per_asset.get(sub.asset_id, 0) + 1
        )
        return True

    async def _subscribe_new(self, client: Client, subscription: Any, idx: int) -> None:
        fleet = await client.nodes.objects.get_child([f"{idx}:{FLEET_FOLDER}"])
        known = await self._known_assets()
        seen = {s.asset_id for s in self._by_node.values()}
        new_triggers: list[Node] = []
        for obj in await fleet.get_children():
            asset_id = (await obj.read_browse_name()).Name
            if asset_id in seen:
                continue
            if asset_id not in known:
                TELEMETRY_MESSAGES_TOTAL.labels(
                    transport="opcua", result="unknown_asset"
                ).inc()
                continue
            sample = [await obj.get_child([f"{idx}:Timestamp"])] + [
                await obj.get_child([f"{idx}:{name}"]) for name in SENSOR_FIELDS
            ]
            trigger = await obj.get_child([f"{idx}:Sequence"])
            self._by_node[trigger.nodeid.to_string()] = _Subscribed(asset_id, sample)
            new_triggers.append(trigger)
        if new_triggers:
            await subscription.subscribe_data_change(new_triggers)
            self.stats.subscribed_assets = len(self._by_node)
            log.info("opcua_assets_subscribed", total=len(self._by_node))

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="opcua-ingestor")

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _session(self) -> None:
        url = self._config.endpoint.replace("0.0.0.0", "127.0.0.1")
        client = Client(url=url)
        if self._config.username is not None:
            client.set_user(self._config.username)
            client.set_password(self._config.password or "")
        async with client:
            idx = await client.get_namespace_index(NAMESPACE)
            subscription = await client.create_subscription(
                self._config.publishing_interval_ms, self
            )
            self._by_node.clear()  # fresh session → fresh monitored items
            await self._subscribe_new(client, subscription, idx)
            refreshed = time.monotonic()
            while True:
                try:
                    node_id, seq = await asyncio.wait_for(
                        self._queue.get(), timeout=5.0
                    )
                    await self.handle(client, node_id, seq)
                except asyncio.TimeoutError:
                    await client.check_connection()
                if time.monotonic() - refreshed > _ASSET_REFRESH_S:
                    await self._subscribe_new(client, subscription, idx)
                    refreshed = time.monotonic()

    async def _run(self) -> None:
        backoff = 1.0
        while True:
            started = time.monotonic()
            try:
                await self._session()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # network, auth, server restart
                self.stats.last_error = str(exc)
                if time.monotonic() - started > 30.0:
                    backoff = 1.0  # the session was healthy for a while
                log.warning(
                    "opcua_ingestor_disconnected", error=str(exc), retry_in_s=backoff
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
