"""
OPC UA telemetry (IEC 62541) — gateway simulado + ingestor, em loopback real.

1. `reading_from_values` valida a amostra inteira ou recusa (nunca parcial).
2. Ponta a ponta: o gateway publica, o ingestor assina `Sequence` e entrega
   a leitura completa ao AssetProcessor do ativo certo.
3. Ativo não cadastrado não é assinado.
4. Notificação com `Sequence` repetido (reconexão) é descartada.
5. Senha errada não abre sessão.
"""

from __future__ import annotations

import asyncio
import socket
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from asyncua import Client

import src.services.opcua_telemetry as opcua_module
from src.schemas.stream import SensorReading
from src.services.opcua_telemetry import (
    SENSOR_FIELDS,
    OpcUaConfig,
    OpcUaGatewayServer,
    OpcUaTelemetryIngestor,
    reading_from_values,
)

T0 = datetime(2026, 10, 16, 12, 0, tzinfo=timezone.utc)


def _reading(i: int) -> SensorReading:
    return SensorReading(
        timestamp=T0 + timedelta(seconds=10 * i),
        **{name: float(k + i) for k, name in enumerate(SENSOR_FIELDS)},
    )


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _Recorder:
    def __init__(self, asset_id: str, sink: dict[str, list[SensorReading]]) -> None:
        self._sink = sink.setdefault(asset_id, [])

    async def process(self, reading: SensorReading) -> None:
        self._sink.append(reading)


def test_reading_from_values_roundtrip() -> None:
    r = _reading(1)
    values = [r.timestamp.replace(tzinfo=None)] + [getattr(r, f) for f in SENSOR_FIELDS]
    out = reading_from_values(values)
    assert out == r  # naive OPC UA DateTime is read as UTC


@pytest.mark.parametrize(
    "values",
    [
        [T0] + [1.0] * (len(SENSOR_FIELDS) - 1),  # one sensor missing
        ["not a date"] + [1.0] * len(SENSOR_FIELDS),
        [T0] + [None] * len(SENSOR_FIELDS),
    ],
)
def test_reading_from_values_rejects_incomplete_samples(values: list[Any]) -> None:
    with pytest.raises(ValueError):
        reading_from_values(values)


async def _wait(cond: Any, timeout: float = 10.0) -> None:
    end = asyncio.get_running_loop().time() + timeout
    while not cond():
        if asyncio.get_running_loop().time() > end:
            raise AssertionError("timeout")
        await asyncio.sleep(0.05)


@pytest.fixture()
def known_assets(monkeypatch: pytest.MonkeyPatch) -> set[str]:
    assets = {"APU-Trem-001"}

    async def _known(self: OpcUaTelemetryIngestor) -> set[str]:
        return assets

    monkeypatch.setattr(OpcUaTelemetryIngestor, "_known_assets", _known)
    monkeypatch.setattr(opcua_module, "_ASSET_REFRESH_S", 0.2)
    return assets


async def test_end_to_end_subscription_delivers_complete_samples(
    known_assets: set[str],
) -> None:
    config = OpcUaConfig(
        endpoint=f"opc.tcp://127.0.0.1:{_free_port()}/predictiq/",
        publishing_interval_ms=50,
    )
    gateway = OpcUaGatewayServer(config)
    await gateway.start()
    received: dict[str, list[SensorReading]] = {}
    ingestor = OpcUaTelemetryIngestor(config, lambda a: _Recorder(a, received))  # type: ignore[arg-type, return-value]
    try:
        # Both assets exist in the gateway before the ingestor connects.
        await gateway.publish("APU-Trem-001", _reading(0))
        await gateway.publish("APU-Intruso-999", _reading(0))
        ingestor.start()
        await _wait(lambda: ingestor.stats.subscribed_assets == 1)
        for i in range(1, 4):
            await gateway.publish("APU-Trem-001", _reading(i))
            await gateway.publish("APU-Intruso-999", _reading(i))
            await asyncio.sleep(0.3)  # one sample per publishing interval
        await _wait(lambda: len(received.get("APU-Trem-001", [])) >= 3)
    finally:
        await ingestor.stop()
        await gateway.stop()

    got = received["APU-Trem-001"]
    assert got[-1] == _reading(3)  # every field, not a mix of two samples
    assert "APU-Intruso-999" not in received
    assert ingestor.stats.rejected == 0


async def test_repeated_sequence_is_dropped(known_assets: set[str]) -> None:
    config = OpcUaConfig(endpoint=f"opc.tcp://127.0.0.1:{_free_port()}/predictiq/")
    gateway = OpcUaGatewayServer(config)
    await gateway.start()
    received: dict[str, list[SensorReading]] = {}
    ingestor = OpcUaTelemetryIngestor(config, lambda a: _Recorder(a, received))  # type: ignore[arg-type, return-value]
    try:
        await gateway.publish("APU-Trem-001", _reading(1))
        async with Client(url=config.endpoint) as client:
            idx = await client.get_namespace_index(opcua_module.NAMESPACE)
            sub = await client.create_subscription(1000, ingestor)
            await ingestor._subscribe_new(client, sub, idx)
            node_id = next(iter(ingestor._by_node))
            assert await ingestor.handle(client, node_id, 1) is True
            assert await ingestor.handle(client, node_id, 1) is False
    finally:
        await gateway.stop()
    assert ingestor.stats.duplicates == 1
    assert len(received["APU-Trem-001"]) == 1


async def test_wrong_password_is_refused() -> None:
    config = OpcUaConfig(
        endpoint=f"opc.tcp://127.0.0.1:{_free_port()}/predictiq/",
        username="gateway",
        password="s3nha-correta",
    )
    gateway = OpcUaGatewayServer(config)
    await gateway.start()
    try:
        bad = Client(url=config.endpoint)
        bad.set_user("gateway")
        bad.set_password("errada")
        with pytest.raises(Exception):
            async with bad:
                pass
        good = Client(url=config.endpoint)
        good.set_user("gateway")
        good.set_password("s3nha-correta")
        async with good:
            assert await good.get_namespace_index(opcua_module.NAMESPACE) > 0
    finally:
        await gateway.stop()
