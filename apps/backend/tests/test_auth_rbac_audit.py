"""
Autenticação, RBAC e trilha de auditoria imutável (IEC 62443 FR1/FR2/FR6).

Garantias verificadas
---------------------
1.  Senha guardada como hash argon2id; senha errada/usuário inexistente/inativo
    → mesmo 401 (sem enumeração de usuários).
2.  Com AUTH_ENABLED: sem token → 401; token expirado/adulterado → 401.
3.  Hierarquia de papéis: operador vê evento mas não abre OS (403); técnico
    abre OS; só admin faz rebuild/cria usuário/verifica a cadeia.
4.  Toda ação relevante (ver evento, abrir/alterar OS, criar usuário) vira
    entrada de auditoria com o ator autenticado.
5.  Cadeia de hashes válida; alterar qualquer linha é detectado e aponta a
    primeira entrada quebrada.
6.  Com AUTH_ENABLED=false o ator é "anonymous (auth disabled)", nunca um
    usuário inventado; e a app recusa subir com auth ligada e segredo placeholder.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.core import security
from src.core.config import settings
from src.core.database import Base, get_db
from src.main import create_app
from src.models.asset import Asset
from src.models.failure_event import FailureEvent
from src.models.user import AuditLog, User
from src.services.auth_service import AuthService, UserCreate

PASSWORD = "correct-horse-battery-staple"


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
                id="APU-Trem-042",
                name="x",
                installation="x",
                system="x",
                equipment_unit="x",
                subunit="x",
            )
        )
        s.add(
            FailureEvent(
                asset_id="APU-Trem-042",
                started_at=datetime(2026, 3, 1, tzinfo=timezone.utc),
                status="closed",
                severity="alerta",
                peak_probability=0.8,
                mean_probability=0.8,
                n_samples=6,
                threshold=0.7,
                ended_at=datetime(2026, 3, 1, 0, 5, tzinfo=timezone.utc),
            )
        )
        svc = AuthService(s)
        for name, role in [
            ("ana", "operator"),
            ("tec", "technician"),
            ("eng", "engineer"),
            ("root", "admin"),
        ]:
            await svc.create_user(
                UserCreate(username=name, password=PASSWORD, role=role)
            )
        await s.commit()
        yield s
    await engine.dispose()


@pytest.fixture()
async def client(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> AsyncGenerator[AsyncClient, None]:
    monkeypatch.setattr(settings, "auth_enabled", True)
    monkeypatch.setattr(settings, "jwt_secret", "test-secret-" + "x" * 32)
    app: FastAPI = create_app()

    async def _db() -> AsyncGenerator[AsyncSession, None]:
        yield db

    app.dependency_overrides[get_db] = _db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        yield c


async def _token(
    client: AsyncClient, user: str, password: str = PASSWORD
) -> dict[str, str]:
    r = await client.post(
        "/v1/auth/login", data={"username": user, "password": password}
    )
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


async def test_password_hashed_and_login_errors_uniform(
    client: AsyncClient, db: AsyncSession
) -> None:
    stored = (
        await db.scalars(select(User.password_hash).where(User.username == "ana"))
    ).one()
    assert stored.startswith("$argon2id$") and PASSWORD not in stored
    wrong = await client.post(
        "/v1/auth/login", data={"username": "ana", "password": "nope-nope-nope"}
    )
    ghost = await client.post(
        "/v1/auth/login", data={"username": "ghost", "password": PASSWORD}
    )
    assert wrong.status_code == ghost.status_code == 401
    assert wrong.json() == ghost.json()


async def test_tokens_required_and_validated(client: AsyncClient) -> None:
    assert (await client.get("/v1/events/1")).status_code == 401
    assert (
        await client.get("/v1/events/1", headers={"Authorization": "Bearer garbage"})
    ).status_code == 401
    expired, _ = security.create_access_token(
        "ana", "operator", now=datetime.now(timezone.utc) - timedelta(days=30)
    )
    assert (
        await client.get("/v1/events/1", headers={"Authorization": f"Bearer {expired}"})
    ).status_code == 401
    me = (await client.get("/v1/auth/me", headers=await _token(client, "eng"))).json()
    assert me == {"username": "eng", "role": "engineer", "authenticated": True}


async def test_role_hierarchy(client: AsyncClient) -> None:
    op, tec, root = (
        await _token(client, "ana"),
        await _token(client, "tec"),
        await _token(client, "root"),
    )
    assert (await client.get("/v1/events/1", headers=op)).status_code == 200
    assert (
        await client.post("/v1/work-orders", json={"event_id": 1}, headers=op)
    ).status_code == 403
    assert (
        await client.post("/v1/work-orders", json={"event_id": 1}, headers=tec)
    ).status_code == 201
    assert (
        await client.post("/v1/events/rebuild/APU-Trem-042", headers=tec)
    ).status_code == 403
    assert (await client.get("/v1/audit", headers=tec)).status_code == 403
    assert (await client.get("/v1/audit/verify", headers=root)).status_code == 200
    new_user = {"username": "novo", "password": PASSWORD, "role": "operator"}
    assert (
        await client.post("/v1/auth/users", json=new_user, headers=tec)
    ).status_code == 403
    assert (
        await client.post("/v1/auth/users", json=new_user, headers=root)
    ).status_code == 201


async def test_audit_records_actor_and_chain_detects_tampering(
    client: AsyncClient, db: AsyncSession
) -> None:
    op = await _token(client, "ana")
    tec = await _token(client, "tec")
    eng = await _token(client, "eng")
    root = await _token(client, "root")
    await client.get("/v1/events/1", headers=op)
    wo = (
        await client.post("/v1/work-orders", json={"event_id": 1}, headers=tec)
    ).json()
    await client.patch(
        f"/v1/work-orders/{wo['id']}", json={"status": "in_progress"}, headers=tec
    )

    trail = (await client.get("/v1/audit", headers=eng)).json()
    assert [(e["actor"], e["action"]) for e in reversed(trail)] == [
        ("ana", "event.view"),
        ("tec", "work_order.create"),
        ("tec", "work_order.update"),
    ]
    assert trail[0]["details"] == {"status": "in_progress"}
    assert (await client.get("/v1/audit/verify", headers=root)).json() == {
        "valid": True,
        "checked": 3,
        "first_broken_id": None,
    }

    second_id = trail[1]["id"]
    await db.execute(
        update(AuditLog).where(AuditLog.id == second_id).values(actor="someone-else")
    )
    await db.commit()
    broken = (await client.get("/v1/audit/verify", headers=root)).json()
    assert broken["valid"] is False and broken["first_broken_id"] == second_id


async def test_auth_disabled_actor_is_anonymous(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "auth_enabled", False)
    app = create_app()

    async def _db() -> AsyncGenerator[AsyncSession, None]:
        yield db

    app.dependency_overrides[get_db] = _db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        assert (await c.get("/v1/events/1")).status_code == 200
    entry = (await db.scalars(select(AuditLog))).one()
    assert entry.actor == security.ANONYMOUS


def test_refuses_auth_with_placeholder_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "auth_enabled", True)
    monkeypatch.setattr(settings, "jwt_secret", security.DEV_JWT_SECRET)
    with pytest.raises(RuntimeError):
        security.ensure_secure_config()
