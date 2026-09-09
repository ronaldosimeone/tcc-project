"""create alert_settings table — RF-25 / RNF-49

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-08 00:00:00.000000

Configuração GLOBAL (single-tenant — não há sistema de usuários no projeto,
auditado explicitamente antes desta task) do limiar de alerta crítico e dos
canais de notificação (Telegram/e-mail). Singleton garantido em nível de
banco: `id` travado em `1` por CHECK constraint (combinado com a PK, nenhuma
segunda linha é possível). O intervalo `0.5 <= alert_threshold <= 0.95` e a
regra "email_enabled implica alert_email preenchido" também são reforçados
por CHECK constraint — ver src/models/alert_settings.py.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op  # type: ignore[attr-defined]

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "alert_settings",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=False),
        sa.Column("alert_threshold", sa.Float, nullable=False),
        sa.Column(
            "telegram_enabled", sa.Boolean, nullable=False, server_default=sa.true()
        ),
        sa.Column(
            "email_enabled", sa.Boolean, nullable=False, server_default=sa.false()
        ),
        sa.Column("alert_email", sa.String, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint("id = 1", name="ck_alert_settings_singleton"),
        sa.CheckConstraint(
            "alert_threshold >= 0.5 AND alert_threshold <= 0.95",
            name="ck_alert_settings_threshold_range",
        ),
        sa.CheckConstraint(
            "(NOT email_enabled) OR (alert_email IS NOT NULL)",
            name="ck_alert_settings_email_required_when_enabled",
        ),
    )


def downgrade() -> None:
    op.drop_table("alert_settings")
