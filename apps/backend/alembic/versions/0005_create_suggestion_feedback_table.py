"""create suggestion_feedback table — RF-38

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-14 00:00:00.000000

Feedback do técnico sobre cada plano de manutenção (útil / não útil + voto
por trecho de manual citado). Alimenta o re-ranking da recuperação e vira
rótulo para o benchmark RAG (RF-36) — ver src/models/suggestion_feedback.py.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op  # type: ignore[attr-defined]

# revision identifiers, used by Alembic.
revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "suggestion_feedback",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("equipment_name", sa.String(200), nullable=False),
        sa.Column("symptom_description", sa.String(500), nullable=True),
        sa.Column("query", sa.String(1000), nullable=True),
        sa.Column("model", sa.String(100), nullable=True),
        sa.Column("grounding_status", sa.String(32), nullable=True),
        sa.Column("useful", sa.Boolean, nullable=False),
        sa.Column("reference_votes", sa.JSON, nullable=False),
        sa.Column("comment", sa.String(1000), nullable=True),
    )
    op.create_index(
        "ix_suggestion_feedback_equipment_name",
        "suggestion_feedback",
        ["equipment_name"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_suggestion_feedback_equipment_name", table_name="suggestion_feedback"
    )
    op.drop_table("suggestion_feedback")
