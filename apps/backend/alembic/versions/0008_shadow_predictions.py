"""shadow_predictions — champion vs challenger (shadow deployment)

Revision ID: 0008
Revises: 0007
Create Date: 2026-10-15 00:00:00.000000
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op  # type: ignore[attr-defined]

revision: str = "0008"
down_revision: Union[str, None] = "0007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "shadow_predictions",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("asset_id", sa.String(64), nullable=False),
        sa.Column("champion_model", sa.String(64), nullable=False),
        sa.Column("challenger_model", sa.String(64), nullable=False),
        sa.Column("champion_probability", sa.Float, nullable=False),
        sa.Column("challenger_probability", sa.Float, nullable=False),
        sa.Column("champion_class", sa.Integer, nullable=False),
        sa.Column("challenger_class", sa.Integer, nullable=False),
    )
    op.create_index("ix_shadow_predictions_ts", "shadow_predictions", ["timestamp"])


def downgrade() -> None:
    op.drop_index("ix_shadow_predictions_ts", table_name="shadow_predictions")
    op.drop_table("shadow_predictions")
