"""create drift_reports table — RF-27 / RNF-55

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-09 00:00:00.000000

Histórico de relatórios de data drift (Evidently/PSI), gerado uma vez por
dia pela task Celery Beat `monitoring.daily_drift_analysis`
(`src/tasks/drift_tasks.py`). `analysis_date` é UNIQUE — chave de
idempotência (RF-27 §Fase 8): a task faz `INSERT ... ON CONFLICT
(analysis_date) DO UPDATE`, então reprocessar o mesmo dia (erro de
scheduler, retry manual) atualiza o relatório em vez de duplicar o
histórico — ver src/models/drift_report.py.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op  # type: ignore[attr-defined]

# revision identifiers, used by Alembic.
revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "drift_reports",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("analysis_date", sa.Date, nullable=False, unique=True),
        sa.Column("analyzed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("reference_period", sa.String, nullable=False),
        sa.Column("current_period_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("current_period_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("psi", sa.Float, nullable=True),
        sa.Column("drift_detected", sa.Boolean, nullable=True),
        sa.Column("features", sa.JSON, nullable=True),
        sa.Column("status", sa.String, nullable=False),
        sa.Column("error_message", sa.String, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_drift_reports_analyzed_at", "drift_reports", ["analyzed_at"])


def downgrade() -> None:
    op.drop_index("ix_drift_reports_analyzed_at", table_name="drift_reports")
    op.drop_table("drift_reports")
