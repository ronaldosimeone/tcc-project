"""create telegram_alert_locks table — RF-24 / RNF-48

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-08 00:00:00.000000

Rate limit de notificações críticas via Telegram — 1 alerta por equipamento
a cada 15 minutos. Uma linha por `equipment_id`; `TelegramAlertRateLimiter`
faz UPSERT nela via `INSERT ... ON CONFLICT ... WHERE expires_at <= now()`
para adquirir a janela atomicamente (ver
src/services/telegram_alert_rate_limiter.py para o porquê de Postgres em
vez de Redis).
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op  # type: ignore[attr-defined]

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "telegram_alert_locks",
        sa.Column("equipment_id", sa.String, primary_key=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("telegram_alert_locks")
