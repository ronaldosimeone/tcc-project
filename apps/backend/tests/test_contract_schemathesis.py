"""
Teste de contrato gerado do OpenAPI (Schemathesis) — rotas de histórico.

Para cada operação GET do histórico, o Schemathesis gera entradas válidas e
inválidas a partir do próprio schema OpenAPI e verifica:
    * nenhuma resposta 5xx (entrada malformada vira 4xx, nunca crash);
    * todo status devolvido está declarado no OpenAPI (achado desta suíte:
      404/409/401/403 não estavam — corrigido com `responses=` nas rotas, e o
      cursor inválido passou de 409 para 400);
    * content-type e corpo batem com o schema declarado.

Banco: SQLite em arquivo temporário (uma engine por requisição, para não
atravessar event loops), com um ativo e um evento semeados.
"""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from pathlib import Path

import schemathesis
from hypothesis import HealthCheck, settings
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.core.database import Base, get_db
from src.main import create_app
from src.models.asset import Asset
from src.models.failure_event import FailureEvent

_DB = Path(tempfile.mkdtemp()) / "contract.db"
_URL = f"sqlite+aiosqlite:///{_DB.as_posix()}"


async def _seed() -> None:
    engine = create_async_engine(_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with async_sessionmaker(bind=engine, class_=AsyncSession)() as s:
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
                mean_probability=0.75,
                n_samples=6,
                threshold=0.7,
                ended_at=datetime(2026, 3, 1, 0, 5, tzinfo=timezone.utc),
            )
        )
        await s.commit()
    await engine.dispose()


asyncio.run(_seed())

app = create_app()


async def _db() -> AsyncGenerator[AsyncSession, None]:
    engine = create_async_engine(_URL)
    try:
        async with async_sessionmaker(
            bind=engine, class_=AsyncSession, expire_on_commit=False
        )() as s:
            yield s
    finally:
        await engine.dispose()


app.dependency_overrides[get_db] = _db

# Schema gerado em memória (a rota /openapi.json fica desligada fora do modo
# debug, como em produção). `servers` removido: o app é chamado direto via
# ASGI, sem o prefixo /api do Nginx (root_path).
_spec = app.openapi()
_spec.pop("servers", None)
schema = schemathesis.openapi.from_dict(_spec).include(
    method="GET", path_regex=r"^/v1/(events|kpis|assets|work-orders)"
)
schema.app = app


# Checagens excluídas — achados documentados, não regressões:
#  * positive_data_acceptance: o cursor de paginação é OPACO (base64 de
#    started_at|id); um texto arbitrário é "válido pelo schema" mas não é um
#    cursor emitido pela API → 400 intencional.
#  * negative_data_rejection: FastAPI/Pydantic em modo lax aceita número em
#    parâmetro date-time (start=0.5 vira unix timestamp). Coerção benigna.
#  * allow_header_conformance: em 405 o Starlette lista só os métodos da rota
#    casada (GET e POST de /work-orders são rotas distintas). Limitação do
#    framework.
_EXCLUDED = [
    schemathesis.checks.positive_data_acceptance,
    schemathesis.checks.negative_data_rejection,
    schemathesis.checks.allow_header_conformance,
]


@schema.parametrize()
@settings(max_examples=25, deadline=None, suppress_health_check=list(HealthCheck))
def test_history_api_contract(case: schemathesis.Case) -> None:
    case.call_and_validate(excluded_checks=_EXCLUDED)
