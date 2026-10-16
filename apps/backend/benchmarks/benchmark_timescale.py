"""
TimescaleDB × Postgres puro para a série de predições.

Mesmo esquema nos dois bancos (``Base.metadata`` do backend); no TimescaleDB
aplica-se exatamente o DDL da migração 0009 (``src/core/timescale.py``).

Uso (dois containers descartáveis):
    docker run -d --name bench-pg -e POSTGRES_PASSWORD=bench -p 55432:5432 postgres:15
    docker run -d --name bench-ts -e POSTGRES_PASSWORD=bench -p 55433:5432 \\
        timescale/timescaledb:2.17.2-pg15 postgres -c shared_preload_libraries=timescaledb
    python benchmarks/benchmark_timescale.py \\
        --plain postgresql://postgres:bench@localhost:55432/postgres \\
        --timescale postgresql://postgres:bench@localhost:55433/postgres

Carga: N ativos × D dias a 0,1 Hz terminando "agora" (para a política de
compressão de 7 dias ter chunks elegíveis). Mede ingestão (COPY), tamanho em
disco e latência p50/p95 das consultas que o backend realmente faz:

* ``event_window``      — janela −2 h…+30 min de um evento (``/v1/events/{id}``),
                          em dado recente (chunk descomprimido) e antigo
                          (chunk comprimido).
* ``hourly_30d``        — série horária de um ativo em 30 dias (gráfico).
* ``fleet_daily_60d``   — alarmes por dia da frota inteira em 60 dias (KPIs).
* ``latest_per_asset``  — última leitura de cada ativo (painel da frota).

Saída: benchmarks/timescale_benchmark.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import asyncpg
import numpy as np
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex, CreateTable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.timescale import CAGG_DDL, CAGG_NAME, HYPERTABLE_DDL  # noqa: E402
from src.models.asset import Asset  # noqa: E402
from src.models.prediction import Prediction  # noqa: E402

OUT = ROOT / "benchmarks" / "timescale_benchmark.json"
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
COLUMNS = ("asset_id", "timestamp", *SENSORS, "predicted_class", "failure_probability")
BATCH = 100_000
RUNS = 30


def _schema_sql() -> list[str]:
    dialect = postgresql.dialect()
    stmts: list[str] = []
    for table in (Asset.__table__, Prediction.__table__):
        stmts.append(str(CreateTable(table).compile(dialect=dialect)))
        stmts += [str(CreateIndex(ix).compile(dialect=dialect)) for ix in table.indexes]
    return stmts


def _rows(asset: str, end: datetime, days: int, seed: int) -> list[tuple[Any, ...]]:
    n = days * 8640
    rng = np.random.default_rng(seed)
    ts0 = end - timedelta(days=days)
    sensors = rng.normal(5.0, 1.0, (n, len(SENSORS))).astype(float)
    p = np.clip(rng.beta(1.2, 12.0, n), 0, 1)
    cls = (p > 0.7).astype(int)
    stamps = [ts0 + timedelta(seconds=10 * i) for i in range(n)]
    return [
        (asset, stamps[i], *sensors[i].tolist(), int(cls[i]), float(p[i]))
        for i in range(n)
    ]


async def _setup(conn: asyncpg.Connection, timescale: bool) -> None:
    await conn.execute("DROP MATERIALIZED VIEW IF EXISTS predictions_hourly CASCADE")
    await conn.execute("DROP TABLE IF EXISTS predictions CASCADE")
    await conn.execute("DROP TABLE IF EXISTS assets CASCADE")
    for stmt in _schema_sql():
        await conn.execute(stmt)
    if timescale:
        for stmt in HYPERTABLE_DDL:
            await conn.execute(stmt)
        for stmt in CAGG_DDL:
            await conn.execute(stmt)


async def _load(
    conn: asyncpg.Connection, assets: list[str], end: datetime, days: int
) -> dict[str, float]:
    await conn.executemany(
        "INSERT INTO assets (id, name, installation, system, equipment_unit, "
        "subunit, criticality, created_at) VALUES ($1, $1, 'bench', 'APU', "
        "'compressor', 'air', 2, now())",
        [(a,) for a in assets],
    )
    total, elapsed = 0, 0.0
    for k, asset in enumerate(assets):
        rows = _rows(asset, end, days, seed=k)
        for i in range(0, len(rows), BATCH):
            chunk = rows[i : i + BATCH]
            t0 = time.perf_counter()
            await conn.copy_records_to_table(
                "predictions", records=chunk, columns=COLUMNS
            )
            elapsed += time.perf_counter() - t0
            total += len(chunk)
    await conn.execute("ANALYZE predictions")
    return {"rows": total, "seconds": elapsed, "rows_per_second": total / elapsed}


async def _size_mb(conn: asyncpg.Connection, timescale: bool) -> float:
    sql = (
        "SELECT hypertable_size('predictions')"
        if timescale
        else "SELECT pg_total_relation_size('predictions')"
    )
    return float(await conn.fetchval(sql)) / 1e6


async def _latency(
    fn: Callable[[], Awaitable[Any]], runs: int = RUNS
) -> dict[str, float]:
    for _ in range(3):
        await fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        await fn()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {
        "p50_ms": statistics.median(times),
        "p95_ms": times[int(0.95 * (len(times) - 1))],
    }


async def _queries(
    conn: asyncpg.Connection, timescale: bool, assets: list[str], end: datetime
) -> dict[str, Any]:
    a = assets[0]
    recent, old = end - timedelta(days=1), end - timedelta(days=40)

    def window(t: datetime) -> Callable[[], Awaitable[Any]]:
        return lambda: conn.fetch(
            "SELECT timestamp, failure_probability FROM predictions "
            "WHERE asset_id = $1 AND timestamp BETWEEN $2 AND $3 ORDER BY timestamp",
            a,
            t - timedelta(hours=2),
            t + timedelta(minutes=30),
        )

    if timescale:
        hourly = (
            f"SELECT bucket, max_probability FROM {CAGG_NAME} "
            "WHERE asset_id = $1 AND bucket >= $2 ORDER BY bucket"
        )
        fleet = (
            f"SELECT time_bucket(INTERVAL '1 day', bucket) AS d, sum(n_alarms) "
            f"FROM {CAGG_NAME} WHERE bucket >= $1 GROUP BY d ORDER BY d"
        )
    else:
        hourly = (
            "SELECT date_trunc('hour', timestamp) AS b, max(failure_probability) "
            "FROM predictions WHERE asset_id = $1 AND timestamp >= $2 "
            "GROUP BY b ORDER BY b"
        )
        fleet = (
            "SELECT date_trunc('day', timestamp) AS d, sum(predicted_class) "
            "FROM predictions WHERE timestamp >= $1 GROUP BY d ORDER BY d"
        )
    latest = (
        "SELECT DISTINCT ON (asset_id) asset_id, timestamp, failure_probability "
        "FROM predictions ORDER BY asset_id, timestamp DESC"
    )
    return {
        "event_window_recent": await _latency(window(recent)),
        "event_window_old": await _latency(window(old)),
        "hourly_30d": await _latency(
            lambda: conn.fetch(hourly, a, end - timedelta(days=30))
        ),
        "fleet_daily_60d": await _latency(
            lambda: conn.fetch(fleet, end - timedelta(days=60)), runs=10
        ),
        "latest_per_asset": await _latency(lambda: conn.fetch(latest), runs=10),
    }


async def run(dsn: str, timescale: bool, n_assets: int, days: int) -> dict[str, Any]:
    conn = await asyncpg.connect(dsn)
    try:
        version = await conn.fetchval("SHOW server_version")
        await _setup(conn, timescale)
        assets = [f"APU-Bench-{i:03d}" for i in range(n_assets)]
        end = datetime.now(timezone.utc).replace(microsecond=0)
        ingest = await _load(conn, assets, end, days)
        out: dict[str, Any] = {"server_version": version, "ingest": ingest}
        out["size_mb"] = await _size_mb(conn, timescale)
        if timescale:
            out["timescaledb"] = await conn.fetchval(
                "SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'"
            )
            t0 = time.perf_counter()
            await conn.execute(
                f"CALL refresh_continuous_aggregate('{CAGG_NAME}', NULL, NULL)"
            )
            out["cagg_refresh_s"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            n = await conn.fetchval(
                "SELECT count(compress_chunk(c, if_not_compressed => true)) "
                "FROM show_chunks('predictions', older_than => INTERVAL '7 days') c"
            )
            out["compression"] = {
                "chunks_compressed": int(n),
                "seconds": time.perf_counter() - t0,
            }
            await conn.execute("ANALYZE predictions")
            out["size_mb_after_compression"] = await _size_mb(conn, timescale)
        out["queries"] = await _queries(conn, timescale, assets, end)
        return out
    finally:
        await conn.close()


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--plain", required=True)
    parser.add_argument("--timescale", required=True)
    parser.add_argument("--assets", type=int, default=5)
    parser.add_argument("--days", type=int, default=60)
    args = parser.parse_args()

    result: dict[str, Any] = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "workload": {
            "assets": args.assets,
            "days": args.days,
            "rate_hz": 0.1,
            "rows": args.assets * args.days * 8640,
            "latency_runs": RUNS,
        },
    }
    for name, dsn, ts in (
        ("postgres", args.plain, False),
        ("timescaledb", args.timescale, True),
    ):
        print(f"[{name}] carregando…", flush=True)
        result[name] = await run(dsn, ts, args.assets, args.days)
        print(json.dumps(result[name]["queries"], indent=1), flush=True)

    pg, tsdb = result["postgres"], result["timescaledb"]
    result["speedup_p50"] = {
        q: pg["queries"][q]["p50_ms"] / tsdb["queries"][q]["p50_ms"]
        for q in pg["queries"]
    }
    result["size_ratio_compressed_vs_postgres"] = (
        tsdb["size_mb_after_compression"] / pg["size_mb"]
    )
    OUT.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"salvo em {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
