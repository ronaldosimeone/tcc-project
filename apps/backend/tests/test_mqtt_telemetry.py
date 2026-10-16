"""
Ingestão MQTT (src/services/mqtt_telemetry.py).

Garantias verificadas
---------------------
1.  Round-trip encode → parse preserva a leitura.
2.  Rejeita: tópico fora do layout, schema desconhecido, JSON inválido,
    asset do payload ≠ asset do tópico (anti-spoofing), leitura inválida.
3.  Ingestor: só processa ativos cadastrados; um processador por ativo;
    estatísticas de aceitos/rejeitados/desconhecidos.
4.  (Integração, opcional) publish → broker real → ingestor → processador,
    quando MQTT_TEST_HOST estiver definido (ex.: mosquitto local).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

import pytest

import src.services.mqtt_telemetry as mqtt_module
from src.schemas.stream import SensorReading
from src.services.mqtt_telemetry import (
    MqttConfig,
    MqttTelemetryIngestor,
    MqttTelemetryPublisher,
    encode_message,
    parse_message,
    topic_for,
)

PREFIX = "predictiq/fleet"
READING = SensorReading(
    timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    TP2=8.1,
    TP3=8.9,
    H1=-0.01,
    DV_pressure=0.02,
    Reservoirs=8.9,
    Motor_current=5.4,
    Oil_temperature=62.0,
    COMP=0.0,
    DV_eletric=1.0,
    Towers=1.0,
    MPG=1.0,
    Oil_level=1.0,
)


def test_round_trip() -> None:
    topic = topic_for(PREFIX, "APU-Trem-001")
    assert topic == "predictiq/fleet/APU-Trem-001/telemetry"
    asset, reading = parse_message(
        topic, encode_message("APU-Trem-001", READING), PREFIX
    )
    assert asset == "APU-Trem-001" and reading == READING


@pytest.mark.parametrize(
    ("topic", "payload", "reason"),
    [
        ("other/APU-1/telemetry", encode_message("APU-1", READING), "unexpected topic"),
        (
            "predictiq/fleet/APU-1/status",
            encode_message("APU-1", READING),
            "unexpected topic",
        ),
        ("predictiq/fleet/APU 1/telemetry", b"{}", "invalid asset id"),
        ("predictiq/fleet/APU-1/telemetry", b"\xff\xfe", "not UTF-8 JSON"),
        (
            "predictiq/fleet/APU-1/telemetry",
            json.dumps({"schema": "v0"}).encode(),
            "unsupported schema",
        ),
        (
            "predictiq/fleet/APU-1/telemetry",
            encode_message("APU-2", READING),
            "does not match topic",
        ),
        (
            "predictiq/fleet/APU-1/telemetry",
            json.dumps(
                {
                    "schema": "predictiq.telemetry.v1",
                    "asset_id": "APU-1",
                    "reading": {"TP2": "x"},
                }
            ).encode(),
            "invalid reading",
        ),
    ],
)
def test_rejections(topic: str, payload: bytes, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        parse_message(topic, payload, PREFIX)


class _FakeProcessor:
    def __init__(self, asset_id: str) -> None:
        self.asset_id = asset_id
        self.readings: list[SensorReading] = []

    async def process(self, reading: SensorReading) -> float | None:
        self.readings.append(reading)
        return None


def _ingestor(
    known: set[str],
) -> tuple[MqttTelemetryIngestor, dict[str, _FakeProcessor]]:
    created: dict[str, _FakeProcessor] = {}

    def factory(asset_id: str) -> Any:
        created[asset_id] = _FakeProcessor(asset_id)
        return created[asset_id]

    ing = MqttTelemetryIngestor(MqttConfig(host="unused", prefix=PREFIX), factory)

    async def _known() -> set[str]:
        return known

    ing._known_assets = _known  # type: ignore[method-assign]  # noqa: SLF001
    return ing, created


async def test_ingestor_routes_known_assets_only() -> None:
    ing, created = _ingestor({"APU-Trem-001", "APU-Trem-002"})
    for asset in ("APU-Trem-001", "APU-Trem-001", "APU-Trem-002", "APU-Trem-999"):
        await ing.handle(topic_for(PREFIX, asset), encode_message(asset, READING))
    await ing.handle(
        topic_for(PREFIX, "APU-Trem-001"), encode_message("APU-Trem-002", READING)
    )

    assert set(created) == {"APU-Trem-001", "APU-Trem-002"}
    assert len(created["APU-Trem-001"].readings) == 2
    assert (
        ing.stats.accepted == 3
        and ing.stats.unknown_asset == 1
        and ing.stats.rejected == 1
    )
    assert ing.stats.per_asset == {"APU-Trem-001": 2, "APU-Trem-002": 1}


@pytest.mark.skipif(
    not os.environ.get("MQTT_TEST_HOST"), reason="needs a real broker (MQTT_TEST_HOST)"
)
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="aiomqtt needs the selector event loop (run in Linux/Docker)",
)
async def test_end_to_end_with_real_broker(monkeypatch: pytest.MonkeyPatch) -> None:
    config = MqttConfig(
        host=os.environ["MQTT_TEST_HOST"],
        port=int(os.environ.get("MQTT_TEST_PORT", "1883")),
        username=os.environ.get("MQTT_TEST_USERNAME"),
        password=os.environ.get("MQTT_TEST_PASSWORD"),
        prefix=f"test-{os.getpid()}/fleet",
    )
    ing, created = _ingestor({"APU-Trem-001"})
    ing._config = config  # noqa: SLF001
    monkeypatch.setattr(mqtt_module, "AsyncSessionFactory", None)
    ing.start()
    await asyncio.sleep(1.0)
    pub = MqttTelemetryPublisher(config)
    await pub.start()
    for _ in range(5):
        await pub.publish("APU-Trem-001", READING)
    for _ in range(50):
        if ing.stats.accepted == 5:
            break
        await asyncio.sleep(0.1)
    await pub.stop()
    await ing.stop()
    assert ing.stats.accepted == 5 and len(created["APU-Trem-001"].readings) == 5
