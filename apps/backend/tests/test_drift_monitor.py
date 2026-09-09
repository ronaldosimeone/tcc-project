"""
Testes de monitoramento de data drift — RF-27 / RNF-55.

Cobertura (RF-27 §Fase 10):
  1. Sem drift (PSI < 0.25)               -> drift_detected = False
  2. Fronteira PSI == 0.25                -> drift_detected = False (estrito)
  3. Drift (PSI > 0.25)                   -> drift_detected = True
  4. Persistência real (SQLite)           -> resultado aparece no banco
  5. Histórico (GET /monitoring/drift)    -> paginação correta
  6. Dados insuficientes                  -> nenhum PSI inventado
  7. Task Celery                          -> registrada, chama DriftMonitor,
                                              roda sem HTTP
  8. Celery Beat                          -> schedule diário configurado
  9. Idempotência                         -> mesma janela 2x não duplica
  10. Evidently real                      -> integração real (sem mock),
                                              dados determinísticos
                                              (reference≈current -> PSI baixo;
                                              reference≠current -> PSI alto)

Estratégia dos testes de fronteira (1-3): o valor exato de PSI é INJETADO
via um `DriftMonitor` que sobrescreve `calculate_drift()` — a regra de
negócio testada é a COMPARAÇÃO (`psi > 0.25`, nunca `>=`), não a precisão
numérica do Evidently sobre uma distribuição sintética arbitrária (isso é
coberto, sem mock, pelo teste 10 — "PSI real"). Mesma filosofia do
`test_full_pipeline.py::test_between_maintenance_and_critical_threshold...`
(RF-26): isolar a regra de negócio do resto do pipeline quando o objetivo é
provar exatamente a fronteira.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import AsyncGenerator

import numpy as np
import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.core.celery_app import celery_app
from src.core.database import Base, get_db
from src.main import create_app
from src.models.drift_report import DriftReport
from src.models.prediction import Prediction
from src.services.drift_monitor import (
    DRIFT_PSI_THRESHOLD,
    MIN_CURRENT_ROWS,
    MONITORED_FEATURES,
    DriftMonitor,
)
from src.tasks.drift_tasks import TASK_NAME, daily_drift_analysis_task

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

_NOW = datetime(2026, 9, 9, 3, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Fixtures — mesmo padrão de test_predictions_endpoint.py / test_alert_settings.py
# ---------------------------------------------------------------------------


@pytest.fixture()
async def db_engine() -> AsyncGenerator[AsyncEngine, None]:
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture()
def session_factory(
    db_engine: AsyncEngine,
) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=db_engine, class_=AsyncSession, expire_on_commit=False
    )


@pytest.fixture()
def test_app(session_factory: async_sessionmaker[AsyncSession]):
    application = create_app()

    async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    application.dependency_overrides[get_db] = _override_get_db
    return application


@pytest.fixture()
async def client(test_app) -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_ROW: dict[str, float] = {
    "TP2": 8.4,
    "TP3": 9.1,
    "H1": 8.5,
    "DV_pressure": 2.1,
    "Reservoirs": 8.7,
    "Motor_current": 4.2,
    "Oil_temperature": 68.5,
}


async def _seed_predictions(
    factory: async_sessionmaker[AsyncSession],
    count: int,
    now: datetime = _NOW,
) -> None:
    """Insere `count` predições dentro da janela current (últimas 24h a
    partir de `now`), 1 minuto de intervalo — sempre dentro da janela."""
    async with factory() as session:
        for i in range(count):
            session.add(
                Prediction(
                    timestamp=now - timedelta(minutes=i),
                    TP2=_BASE_ROW["TP2"],
                    TP3=_BASE_ROW["TP3"],
                    H1=_BASE_ROW["H1"],
                    DV_pressure=_BASE_ROW["DV_pressure"],
                    Reservoirs=_BASE_ROW["Reservoirs"],
                    Motor_current=_BASE_ROW["Motor_current"],
                    Oil_temperature=_BASE_ROW["Oil_temperature"],
                    COMP=1.0,
                    DV_eletric=0.0,
                    Towers=1.0,
                    MPG=1.0,
                    Oil_level=1.0,
                    predicted_class=0,
                    failure_probability=0.1,
                )
            )
        await session.commit()


class _StubMonitor(DriftMonitor):
    """`DriftMonitor` com `load_reference_data`/`calculate_drift`
    substituídos por valores controlados — usado SÓ pelos testes de
    fronteira (1-3), onde o que se testa é a comparação `psi > 0.25`, não a
    precisão numérica do Evidently."""

    def __init__(self, psi_by_feature: dict[str, float]) -> None:
        super().__init__()
        self._psi_by_feature = psi_by_feature

    def load_reference_data(self) -> pd.DataFrame:
        return pd.DataFrame({col: [0.0, 1.0] for col in MONITORED_FEATURES})

    def calculate_drift(
        self, reference: pd.DataFrame, current: pd.DataFrame
    ) -> dict[str, float]:
        return dict(self._psi_by_feature)


# ---------------------------------------------------------------------------
# 1-3. Fronteira do threshold (RF-27 §Fase 5) — psi > 0.25, nunca >=
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "psi_value,expected_drift",
    [
        (0.2499, False),  # abaixo — sem drift
        (0.25, False),  # exatamente no limiar — ESTRITO, sem drift
        (0.2501, True),  # acima — drift
    ],
)
async def test_psi_threshold_boundary(
    session_factory: async_sessionmaker[AsyncSession],
    psi_value: float,
    expected_drift: bool,
) -> None:
    await _seed_predictions(session_factory, MIN_CURRENT_ROWS)

    async with session_factory() as db:
        monitor = _StubMonitor({"TP2": psi_value})
        result = await monitor.run_daily_analysis(db, now=_NOW)

    assert result.status == "ok"
    assert result.psi == pytest.approx(psi_value)
    assert result.drift_detected is expected_drift


def test_drift_threshold_constant_is_0_25() -> None:
    """Documenta a regra literal do RF-27 — 0.25, nem mais nem menos."""
    assert DRIFT_PSI_THRESHOLD == 0.25


# ---------------------------------------------------------------------------
# 4. Persistência real
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persist_report_writes_real_row(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    await _seed_predictions(session_factory, MIN_CURRENT_ROWS)

    async with session_factory() as db:
        monitor = _StubMonitor({"TP2": 0.9, "TP3": 0.1})
        result = await monitor.run_daily_analysis(db, now=_NOW)
        await monitor.persist_report(db, result, analysis_date=_NOW.date())

    async with session_factory() as db:
        rows = (await db.execute(select(DriftReport))).scalars().all()

    assert len(rows) == 1
    row = rows[0]
    assert row.analysis_date == _NOW.date()
    assert row.status == "ok"
    assert row.psi == pytest.approx(0.9)  # max(0.9, 0.1)
    assert row.drift_detected is True
    assert row.features == {"TP2": 0.9, "TP3": 0.1}
    assert "metropt3_baseline" in row.reference_period


# ---------------------------------------------------------------------------
# 5. Histórico — GET /monitoring/drift
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_drift_history_endpoint_returns_persisted_reports(
    client: AsyncClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    await _seed_predictions(session_factory, MIN_CURRENT_ROWS)

    # `now=_NOW` fixo em toda iteração — a janela current (últimas 24h a
    # partir de `_NOW`) é a MESMA, sempre com dados suficientes (semeados
    # perto de `_NOW`); só `analysis_date` varia, simulando 3 dias
    # DIFERENTES no histórico (decoupled de propósito — ver docstring do
    # arquivo sobre não acoplar `now` da análise ao `analysis_date` persistido).
    #
    # Inserido em ordem CRESCENTE de `day_offset` (2, 1, 0) — o dia mais
    # recente (offset 0) é persistido POR ÚLTIMO, então tem o maior
    # `analyzed_at` (relógio real de `persist_report`) e aparece primeiro
    # na ordenação "mais recente primeiro" — mesmo comportamento de
    # produção (o relatório de hoje é sempre o último gerado).
    for day_offset, psi in [(2, 0.15), (1, 0.9), (0, 0.05)]:
        day = _NOW - timedelta(days=day_offset)
        async with session_factory() as db:
            monitor = _StubMonitor({"TP2": psi})
            result = await monitor.run_daily_analysis(db, now=_NOW)
            await monitor.persist_report(db, result, analysis_date=day.date())

    response = await client.get("/monitoring/drift")
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["total"] == 3
    assert len(body["items"]) == 3
    # Mais recente primeiro (analyzed_at desc).
    assert body["items"][0]["psi"] == pytest.approx(0.05)
    assert body["items"][0]["drift_detected"] is False
    assert body["items"][1]["psi"] == pytest.approx(0.9)
    assert body["items"][1]["drift_detected"] is True


@pytest.mark.asyncio
async def test_drift_history_endpoint_respects_pagination(
    client: AsyncClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    await _seed_predictions(session_factory, MIN_CURRENT_ROWS)

    for day_offset in range(5):
        day = _NOW - timedelta(days=day_offset)
        async with session_factory() as db:
            monitor = _StubMonitor({"TP2": 0.1})
            # `now=_NOW` (não `day`) — mesma janela current com dados
            # suficientes em toda iteração; só `analysis_date` varia.
            result = await monitor.run_daily_analysis(db, now=_NOW)
            await monitor.persist_report(db, result, analysis_date=day.date())

    response = await client.get("/monitoring/drift", params={"page": 1, "size": 2})
    body = response.json()
    assert body["total"] == 5
    assert body["pages"] == 3
    assert len(body["items"]) == 2


@pytest.mark.asyncio
async def test_drift_history_endpoint_never_triggers_analysis(
    client: AsyncClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """RF-27 §Fase 7/9: o endpoint só LÊ — histórico vazio continua vazio,
    nenhuma análise é disparada pela chamada HTTP."""
    response = await client.get("/monitoring/drift")
    assert response.status_code == 200
    body = response.json()
    assert body == {"items": [], "total": 0, "page": 1, "size": 20, "pages": 0}

    async with session_factory() as db:
        rows = (await db.execute(select(DriftReport))).scalars().all()
    assert rows == []


# ---------------------------------------------------------------------------
# 6. Dados insuficientes — nenhum PSI inventado
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_insufficient_current_data_does_not_fabricate_psi(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    # Menos que MIN_CURRENT_ROWS na janela — dados insuficientes.
    await _seed_predictions(session_factory, MIN_CURRENT_ROWS - 1)

    async with session_factory() as db:
        monitor = DriftMonitor()  # SEM stub — prova que nem chega a chamar Evidently
        result = await monitor.run_daily_analysis(db, now=_NOW)

    assert result.status == "insufficient_data"
    assert result.psi is None
    assert result.drift_detected is None
    assert result.features is None
    assert result.error_message is not None
    assert str(MIN_CURRENT_ROWS) in result.error_message


@pytest.mark.asyncio
async def test_insufficient_data_is_still_persisted_and_visible_in_history(
    client: AsyncClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """RF-27 §"SEM DADOS INVENTADOS": o motivo fica registrado no
    histórico, nunca escondido."""
    await _seed_predictions(session_factory, 5)  # bem abaixo do mínimo

    async with session_factory() as db:
        monitor = DriftMonitor()
        result = await monitor.run_daily_analysis(db, now=_NOW)
        await monitor.persist_report(db, result, analysis_date=_NOW.date())

    response = await client.get("/monitoring/drift")
    body = response.json()
    assert body["total"] == 1
    assert body["items"][0]["status"] == "insufficient_data"
    assert body["items"][0]["psi"] is None
    assert body["items"][0]["drift_detected"] is None


@pytest.mark.asyncio
async def test_zero_current_rows_is_insufficient_data_not_a_crash(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Nenhuma predição na janela (banco vazio) — caso extremo de "dados
    insuficientes", não uma exceção não tratada."""
    async with session_factory() as db:
        monitor = DriftMonitor()
        result = await monitor.run_daily_analysis(db, now=_NOW)

    assert result.status == "insufficient_data"


# ---------------------------------------------------------------------------
# 7. Task Celery — registrada, chama DriftMonitor, roda sem HTTP
# ---------------------------------------------------------------------------


def test_task_is_registered_in_celery_app() -> None:
    assert TASK_NAME in celery_app.tasks
    assert celery_app.tasks[TASK_NAME].name == TASK_NAME
    assert callable(daily_drift_analysis_task)


def test_task_runs_directly_without_http(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A task não depende de nenhum request HTTP — só do Celery + Postgres
    (aqui, SQLite via `AsyncSessionFactory` trocado no teste).

    Usa um arquivo SQLite (não `:memory:`): a task real chama
    `engine.dispose()` no `finally` (mesmo padrão de
    `notification_tasks.py`, necessário em produção contra asyncpg/Postgres
    para não reusar conexões de um event loop morto) — com `:memory:`,
    `dispose()` destruiria o próprio banco antes da verificação abaixo
    poder ler de volta; um arquivo persiste normalmente através do dispose,
    igual a um Postgres real.
    """
    import src.tasks.drift_tasks as drift_tasks_module

    db_path = tmp_path / "test_drift_task.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}", echo=False)
    session_factory = async_sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False
    )

    async def _prepare() -> None:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await _seed_predictions(session_factory, MIN_CURRENT_ROWS, now=_NOW)

    import asyncio

    asyncio.run(_prepare())

    monkeypatch.setattr(drift_tasks_module, "AsyncSessionFactory", session_factory)
    monkeypatch.setattr(drift_tasks_module, "engine", engine)

    # Chama a task diretamente (função Python pura, sem `.delay()`/broker).
    daily_drift_analysis_task()

    async def _check() -> list[DriftReport]:
        async with session_factory() as db:
            return list((await db.execute(select(DriftReport))).scalars().all())

    rows = asyncio.run(_check())
    asyncio.run(engine.dispose())

    assert len(rows) == 1
    assert rows[0].status == "ok"  # dados reais suficientes -> Evidently real rodou


# ---------------------------------------------------------------------------
# 8. Celery Beat — schedule diário configurado
# ---------------------------------------------------------------------------


def test_beat_schedule_has_daily_drift_entry() -> None:
    schedule = celery_app.conf.beat_schedule
    assert "daily-drift-analysis" in schedule

    entry = schedule["daily-drift-analysis"]
    assert entry["task"] == TASK_NAME

    # Uma execução por dia: crontab com hora/minuto fixos (não a cada N
    # minutos/segundos) — `crontab.hour`/`.minute` são conjuntos de 1
    # elemento; `*` (todas as horas) teria mais de um elemento.
    cron = entry["schedule"]
    assert len(cron.hour) == 1
    assert len(cron.minute) == 1


def test_celery_app_timezone_is_utc() -> None:
    """Mesma configuração de fuso já usada pelo RNF-50 (RF-27 §Fase 7:
    "usar timezone já adotado pelo projeto")."""
    assert celery_app.conf.timezone == "UTC"
    assert celery_app.conf.enable_utc is True


# ---------------------------------------------------------------------------
# 9. Idempotência — mesma janela 2x não duplica histórico
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_running_same_analysis_date_twice_does_not_duplicate(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    await _seed_predictions(session_factory, MIN_CURRENT_ROWS)

    # Primeira execução — PSI baixo.
    async with session_factory() as db:
        monitor = _StubMonitor({"TP2": 0.05})
        result_1 = await monitor.run_daily_analysis(db, now=_NOW)
        await monitor.persist_report(db, result_1, analysis_date=_NOW.date())

    # Segunda execução do MESMO dia (ex.: erro do scheduler, retry manual) —
    # PSI diferente, simulando um reprocessamento real.
    async with session_factory() as db:
        monitor = _StubMonitor({"TP2": 0.9})
        result_2 = await monitor.run_daily_analysis(db, now=_NOW + timedelta(minutes=5))
        await monitor.persist_report(db, result_2, analysis_date=_NOW.date())

    async with session_factory() as db:
        rows = (await db.execute(select(DriftReport))).scalars().all()

    # Uma ÚNICA linha para o dia — a segunda execução SOBRESCREVEU a primeira.
    assert len(rows) == 1
    assert rows[0].psi == pytest.approx(0.9)
    assert rows[0].drift_detected is True


@pytest.mark.asyncio
async def test_different_analysis_dates_create_separate_rows(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    await _seed_predictions(session_factory, MIN_CURRENT_ROWS)

    for day_offset in range(3):
        day = _NOW - timedelta(days=day_offset)
        async with session_factory() as db:
            monitor = _StubMonitor({"TP2": 0.1})
            result = await monitor.run_daily_analysis(db, now=_NOW)
            await monitor.persist_report(db, result, analysis_date=day.date())

    async with session_factory() as db:
        rows = (await db.execute(select(DriftReport))).scalars().all()

    assert len(rows) == 3
    assert len({r.analysis_date for r in rows}) == 3


# ---------------------------------------------------------------------------
# 10. Evidently REAL (sem mock) — dados determinísticos (RF-27 §Fase 11)
# ---------------------------------------------------------------------------


def test_evidently_real_low_psi_when_distributions_are_similar() -> None:
    """reference + current semelhante -> PSI abaixo do threshold. Nenhum
    mock do Evidently aqui — `Report`/`Dataset`/`ValueDrift` reais."""
    rng = np.random.default_rng(42)
    reference = pd.DataFrame(
        {
            "TP2": rng.normal(8.0, 0.5, 2000),
            "TP3": rng.normal(9.0, 0.3, 2000),
            "H1": rng.normal(8.5, 0.4, 2000),
            "DV_pressure": rng.normal(2.0, 0.2, 2000),
            "Reservoirs": rng.normal(8.5, 0.3, 2000),
            "Oil_temperature": rng.normal(68.0, 2.0, 2000),
            "Motor_current": rng.normal(4.0, 0.5, 2000),
        }
    )
    current = pd.DataFrame(
        {
            "TP2": rng.normal(8.0, 0.5, 200),
            "TP3": rng.normal(9.0, 0.3, 200),
            "H1": rng.normal(8.5, 0.4, 200),
            "DV_pressure": rng.normal(2.0, 0.2, 200),
            "Reservoirs": rng.normal(8.5, 0.3, 200),
            "Oil_temperature": rng.normal(68.0, 2.0, 200),
            "Motor_current": rng.normal(4.0, 0.5, 200),
        }
    )

    monitor = DriftMonitor()
    psi_by_feature = monitor.calculate_drift(reference, current)

    assert set(psi_by_feature.keys()) == set(MONITORED_FEATURES)
    overall_psi = max(psi_by_feature.values())
    assert overall_psi <= DRIFT_PSI_THRESHOLD, (
        f"Distribuições semelhantes deveriam produzir PSI baixo, "
        f"obtido {overall_psi} — {psi_by_feature}"
    )


def test_evidently_real_high_psi_when_distributions_differ_significantly() -> None:
    """reference + current SIGNIFICATIVAMENTE diferente -> PSI acima do
    threshold. Mesma feature (Motor_current) com média completamente
    deslocada — drift real e óbvio, não um valor "só um pouco acima"."""
    rng = np.random.default_rng(7)
    reference = pd.DataFrame(
        {
            "TP2": rng.normal(8.0, 0.5, 2000),
            "TP3": rng.normal(9.0, 0.3, 2000),
            "H1": rng.normal(8.5, 0.4, 2000),
            "DV_pressure": rng.normal(2.0, 0.2, 2000),
            "Reservoirs": rng.normal(8.5, 0.3, 2000),
            "Oil_temperature": rng.normal(68.0, 2.0, 2000),
            "Motor_current": rng.normal(4.0, 0.5, 2000),
        }
    )
    current = pd.DataFrame(
        {
            "TP2": rng.normal(8.0, 0.5, 200),
            "TP3": rng.normal(9.0, 0.3, 200),
            "H1": rng.normal(8.5, 0.4, 200),
            "DV_pressure": rng.normal(2.0, 0.2, 200),
            "Reservoirs": rng.normal(8.5, 0.3, 200),
            "Oil_temperature": rng.normal(68.0, 2.0, 200),
            # Motor_current: deslocamento grande e óbvio (4A -> 12A).
            "Motor_current": rng.normal(12.0, 0.5, 200),
        }
    )

    monitor = DriftMonitor()
    psi_by_feature = monitor.calculate_drift(reference, current)

    assert psi_by_feature["Motor_current"] > DRIFT_PSI_THRESHOLD
    overall_psi = max(psi_by_feature.values())
    assert overall_psi > DRIFT_PSI_THRESHOLD


def test_reference_data_loads_real_baseline_excluding_failure_windows() -> None:
    """`load_reference_data()` real — lê o parquet MetroPT-3 de verdade
    (settings.simulator_parquet_path) e devolve só as colunas monitoradas,
    amostra determinística."""
    monitor = DriftMonitor()
    reference_df = monitor.load_reference_data()

    assert list(reference_df.columns) == MONITORED_FEATURES
    assert len(reference_df) > 0
    assert len(reference_df) <= 5_000

    # Determinístico — chamar de novo produz EXATAMENTE a mesma amostra
    # (mesma seed) — RF-27 §Fase 4 "janela deve ser determinística".
    reference_df_2 = monitor.load_reference_data()
    pd.testing.assert_frame_equal(reference_df, reference_df_2)
