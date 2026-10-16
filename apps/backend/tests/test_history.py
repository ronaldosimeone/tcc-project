"""
Histórico real — FailureEventService, HistoryService e router /v1.

Garantias verificadas
---------------------
1.  Episódios (RN-03) persistidos online a partir das predições, um por
    sequência; fechamento com ended_at = última amostra acima do limiar.
2.  Reinício: evento deixado `open` é fechado no primeiro processamento do ativo.
3.  `rebuild` recria os mesmos episódios do stream online e recusa ativo com OS.
4.  Paginação por cursor sem perda nem repetição.
5.  Detalhe traz leituras REAIS da janela (−2 h … +30 min).
6.  OS: máquina de estados validada; fechar exige veredito (causa ou falso alarme).
7.  KPIs só contam falhas confirmadas: MTBF, MTTR, disponibilidade, precisão
    de alarmes; OEE = None (não computável) com justificativa.
8.  Contagem diária por severidade.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.core.database import Base, get_db
from src.main import create_app
from src.models.asset import Asset
from src.models.failure_event import FailureEvent
from src.models.prediction import Prediction
from src.services.failure_event_service import FailureEventService

T0 = datetime(2026, 3, 1, tzinfo=timezone.utc)
ASSET = "APU-Trem-042"
SENSORS = (
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


@pytest.fixture()
async def db() -> AsyncGenerator[AsyncSession, None]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False
    )
    async with factory() as s:
        s.add(
            Asset(
                id=ASSET,
                name="Compressor 042",
                installation="Porto",
                system="APU",
                equipment_unit="Compressor",
                subunit="Compressor + secador",
                criticality=3,
            )
        )
        await s.commit()
        yield s
    await engine.dispose()


def _probs() -> list[float]:
    # episode A (8 high), gap, episode B (6 high, one critical), tail normal
    return (
        [0.1] * 3
        + [0.8] * 8
        + [0.1] * 10
        + [0.75, 0.95, 0.8, 0.8, 0.8, 0.8]
        + [0.1] * 8
    )


async def _stream(
    db: AsyncSession, svc: FailureEventService, probs: list[float]
) -> None:
    for i, p in enumerate(probs):
        ts = T0 + timedelta(seconds=10 * i)
        db.add(
            Prediction(
                asset_id=ASSET,
                timestamp=ts,
                predicted_class=int(p >= 0.5),
                failure_probability=p,
                **{s: float(i) for s in SENSORS},
            )
        )
        await svc.process(db, ASSET, ts, p, "random_forest_v2")
    await db.commit()


async def test_online_episodes(db: AsyncSession) -> None:
    await _stream(db, FailureEventService(k_open=6, k_close=6), _probs())
    events = (
        await db.scalars(select(FailureEvent).order_by(FailureEvent.started_at))
    ).all()
    assert len(events) == 2
    a, b = events
    assert a.started_at.replace(tzinfo=timezone.utc) == T0 + timedelta(seconds=30)
    assert a.ended_at.replace(tzinfo=timezone.utc) == T0 + timedelta(
        seconds=100
    )  # last of 8 highs
    assert a.n_samples == 8 and a.status == "closed" and a.severity == "alerta"
    assert b.severity == "critico" and b.peak_probability == pytest.approx(0.95)


async def test_restart_closes_stale_open_event(db: AsyncSession) -> None:
    db.add(
        FailureEvent(
            asset_id=ASSET,
            started_at=T0,
            status="open",
            severity="alerta",
            peak_probability=0.8,
            mean_probability=0.8,
            n_samples=6,
            threshold=0.7,
        )
    )
    await db.commit()
    await FailureEventService().process(db, ASSET, T0 + timedelta(hours=1), 0.1)
    await db.commit()
    stale = (await db.scalars(select(FailureEvent))).one()
    assert stale.status == "closed" and stale.ended_at is not None


async def test_rebuild_matches_online(db: AsyncSession) -> None:
    await _stream(db, FailureEventService(), _probs())
    online = [
        (e.started_at, e.ended_at, e.n_samples)
        for e in (
            await db.scalars(select(FailureEvent).order_by(FailureEvent.started_at))
        ).all()
    ]
    created = await FailureEventService().rebuild(db, ASSET)
    await db.commit()
    rebuilt = [
        (e.started_at, e.ended_at, e.n_samples)
        for e in (
            await db.scalars(select(FailureEvent).order_by(FailureEvent.started_at))
        ).all()
    ]
    assert created == 2 and rebuilt == online


@pytest.fixture()
async def client(db: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    app: FastAPI = create_app()

    async def _db() -> AsyncGenerator[AsyncSession, None]:
        yield db

    app.dependency_overrides[get_db] = _db
    await _stream(db, FailureEventService(), _probs())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        yield c


async def test_cursor_pagination(client: AsyncClient) -> None:
    p1 = (await client.get("/v1/events", params={"limit": 1})).json()
    p2 = (
        await client.get("/v1/events", params={"limit": 1, "cursor": p1["next_cursor"]})
    ).json()
    assert len(p1["items"]) == 1 and len(p2["items"]) == 1
    assert p1["items"][0]["id"] != p2["items"][0]["id"]
    assert p1["items"][0]["started_at"] > p2["items"][0]["started_at"]  # newest first
    assert p2["next_cursor"] is None
    assert (await client.get("/v1/events", params={"severity": "critico"})).json()[
        "items"
    ][0]["severity"] == "critico"


async def test_detail_has_real_sensor_window(client: AsyncClient) -> None:
    eid = (await client.get("/v1/events")).json()["items"][-1]["id"]
    detail = (await client.get(f"/v1/events/{eid}")).json()
    assert detail["event"]["id"] == eid
    assert len(detail["sensor_window"]) == len(
        _probs()
    )  # whole synthetic stream fits the window
    assert set(detail["sensor_window"][0]["values"]) == set(SENSORS)
    assert (await client.get("/v1/events/9999")).status_code == 404


async def test_work_order_lifecycle_and_kpis(client: AsyncClient) -> None:
    ids = [e["id"] for e in (await client.get("/v1/events")).json()["items"]]
    wo1 = (
        await client.post(
            "/v1/work-orders", json={"event_id": ids[0], "assignee": "tec-1"}
        )
    ).json()
    wo2 = (await client.post("/v1/work-orders", json={"event_id": ids[1]})).json()
    assert (
        await client.post("/v1/work-orders", json={"event_id": ids[0]})
    ).status_code == 409

    assert (
        await client.patch(f"/v1/work-orders/{wo1['id']}", json={"status": "closed"})
    ).status_code == 409
    await client.patch(f"/v1/work-orders/{wo1['id']}", json={"status": "in_progress"})
    no_verdict = await client.patch(
        f"/v1/work-orders/{wo1['id']}", json={"status": "closed"}
    )
    assert no_verdict.status_code == 409
    closed = await client.patch(
        f"/v1/work-orders/{wo1['id']}",
        json={
            "status": "closed",
            "confirmed_cause": "air_leak",
            "parts": [{"part": "junta", "quantity": 1}],
        },
    )
    assert closed.status_code == 200 and closed.json()["is_false_alarm"] is False
    assert (
        await client.patch(f"/v1/work-orders/{wo1['id']}", json={"notes": "x"})
    ).status_code == 409

    await client.patch(
        f"/v1/work-orders/{wo2['id']}",
        json={"is_false_alarm": True, "status": "in_progress"},
    )
    await client.patch(f"/v1/work-orders/{wo2['id']}", json={"status": "closed"})

    k = (
        await client.get(
            "/v1/kpis",
            params={
                "start": (T0 - timedelta(days=1)).isoformat(),
                "end": (T0 + timedelta(days=1)).isoformat(),
            },
        )
    ).json()
    assert (
        k["events_total"] == 2
        and k["confirmed_failures"] == 1
        and k["false_alarms"] == 1
    )
    assert k["alarm_precision"] == pytest.approx(0.5)
    assert k["observed_hours"] == pytest.approx((len(_probs()) - 1) * 10 / 3600)
    assert k["mtbf_hours"] == pytest.approx(k["observed_hours"])
    assert k["mttr_hours"] is not None and k["availability"] is not None
    assert k["oee"] is None and "OEE" in k["oee_note"]

    event = (await client.get(f"/v1/events/{ids[0]}")).json()["event"]
    assert (
        event["confirmed_cause"] == "air_leak"
        and event["work_order_status"] == "closed"
    )


async def test_rebuild_refused_when_labelled(
    client: AsyncClient, db: AsyncSession
) -> None:
    eid = (await client.get("/v1/events")).json()["items"][0]["id"]
    await client.post("/v1/work-orders", json={"event_id": eid})
    from src.core.exceptions import ConflictError

    with pytest.raises(ConflictError):
        await FailureEventService().rebuild(db, ASSET)


async def test_daily_counts(client: AsyncClient) -> None:
    days = (
        await client.get(
            "/v1/events/daily",
            params={
                "start": T0.isoformat(),
                "end": (T0 + timedelta(days=2)).isoformat(),
            },
        )
    ).json()
    assert [d["date"] for d in days] == ["2026-03-01", "2026-03-02"]
    assert days[0] == {"date": "2026-03-01", "alerta": 1, "critico": 1}


@pytest.mark.parametrize(
    ("path", "params"),
    [
        # Schemathesis: default start = end − 30 d overflowed → 500.
        ("/v1/events/daily", {"end": "0001-01-01T18:42:00Z"}),
        ("/v1/kpis", {"end": "0001-01-01T00:00:00Z"}),
        # Millennia of empty daily rows.
        (
            "/v1/events/daily",
            {"start": "2000-01-01T00:00:00Z", "end": "9392-01-01T00:02:17Z"},
        ),
        # Empty / inverted window.
        (
            "/v1/kpis",
            {"start": "2026-03-02T00:00:00Z", "end": "2026-03-01T00:00:00Z"},
        ),
    ],
)
async def test_unusable_windows_are_400_not_500(
    client: AsyncClient, path: str, params: dict[str, str]
) -> None:
    resp = await client.get(path, params=params)
    assert resp.status_code == 400, resp.text
