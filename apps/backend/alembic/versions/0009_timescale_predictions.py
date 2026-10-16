"""predictions as a TimescaleDB hypertable (+ compression, hourly aggregate)

Revision ID: 0009
Revises: 0008
Create Date: 2026-10-16 00:00:00.000000

No-op on SQLite and on a Postgres without the timescaledb extension: the
schema stays exactly as in 0008 and everything keeps working. DDL lives in
src/core/timescale.py (shared with benchmarks/benchmark_timescale.py).
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op  # type: ignore[attr-defined]

from src.core.timescale import (
    CAGG_DDL,
    CAGG_DROP_DDL,
    EXTENSION_AVAILABLE_SQL,
    HYPERTABLE_DDL,
    IS_HYPERTABLE_SQL,
    REVERT_DDL,
)

revision: str = "0009"
down_revision: Union[str, None] = "0008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _timescale_available() -> bool:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return False
    if bind.execute(sa.text(EXTENSION_AVAILABLE_SQL)).first() is None:
        return False
    # Installed but not preloaded (e.g. a volume initialised by plain
    # postgres:15): CREATE EXTENSION would fail, so leave the schema alone.
    preload = bind.execute(sa.text("SHOW shared_preload_libraries")).scalar() or ""
    return "timescaledb" in str(preload)


def _is_hypertable() -> bool:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return False
    has_ext = bind.execute(
        sa.text("SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'")
    ).first()
    return (
        has_ext is not None
        and bind.execute(sa.text(IS_HYPERTABLE_SQL)).first() is not None
    )


def upgrade() -> None:
    if not _timescale_available():
        return
    for stmt in HYPERTABLE_DDL:
        op.execute(stmt)
    with op.get_context().autocommit_block():
        for stmt in CAGG_DDL:
            op.execute(stmt)


def downgrade() -> None:
    if not _is_hypertable():
        return
    with op.get_context().autocommit_block():
        for stmt in CAGG_DROP_DDL:
            op.execute(stmt)
    for stmt in REVERT_DDL:
        op.execute(stmt)
