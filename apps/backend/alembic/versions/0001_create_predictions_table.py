"""create predictions table — RF-09

Revision ID: 0001
Revises:
Create Date: 2024-01-01 00:00:00.000000

Creates the `predictions` table that stores every inference result from
POST /predict together with the 12 raw sensor inputs.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op  # type: ignore[attr-defined]

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "predictions",
        # ── Primary key ───────────────────────────────────────────────────
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        # ── Timestamp (indexed for ORDER BY timestamp DESC) ───────────────
        sa.Column(
            "timestamp",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        # ── Analogue sensors ──────────────────────────────────────────────
        sa.Column("TP2", sa.Float, nullable=False),
        sa.Column("TP3", sa.Float, nullable=False),
        sa.Column("H1", sa.Float, nullable=False),
        sa.Column("DV_pressure", sa.Float, nullable=False),
        sa.Column("Reservoirs", sa.Float, nullable=False),
        sa.Column("Motor_current", sa.Float, nullable=False),
        sa.Column("Oil_temperature", sa.Float, nullable=False),
        # ── Digital / switch sensors ──────────────────────────────────────
        sa.Column("COMP", sa.Float, nullable=False),
        sa.Column("DV_eletric", sa.Float, nullable=False),
        sa.Column("Towers", sa.Float, nullable=False),
        sa.Column("MPG", sa.Float, nullable=False),
        sa.Column("Oil_level", sa.Float, nullable=False),
        # ── ML outputs ────────────────────────────────────────────────────
        sa.Column("predicted_class", sa.Integer, nullable=False),
        sa.Column("failure_probability", sa.Float, nullable=False),
    )

    # Separate index call — allows easy removal in downgrade()
    op.create_index(
        op.f("ix_predictions_timestamp"),
        "predictions",
        ["timestamp"],
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_predictions_timestamp"), table_name="predictions")
    op.drop_table("predictions")
