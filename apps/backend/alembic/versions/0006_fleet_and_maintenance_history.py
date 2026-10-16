"""fleet + maintenance history: assets, failure events, work orders, RCA

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-15 00:00:00.000000

Replaces the mocked history of the frontend with real domain tables:

* assets               — fleet registry (ISO 14224 levels); seeded with the
                         historical single asset APU-Trem-042.
* predictions.asset_id — which asset produced each reading (default
                         APU-Trem-042, so existing rows stay valid).
* failure_events       — alarm episodes aggregated with the RN-03 rule.
* work_orders          — maintenance actions + confirmed cause (ground truth).
* root_cause_analyses  — frozen XAI + RAG snapshot per event.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op  # type: ignore[attr-defined]

revision: str = "0006"
down_revision: Union[str, None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

DEFAULT_ASSET = "APU-Trem-042"


def upgrade() -> None:
    assets = op.create_table(
        "assets",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("installation", sa.String(100), nullable=False),
        sa.Column("system", sa.String(100), nullable=False),
        sa.Column("equipment_unit", sa.String(100), nullable=False),
        sa.Column("subunit", sa.String(100), nullable=False),
        sa.Column("manufacturer_model", sa.String(100), nullable=True),
        sa.Column("criticality", sa.Integer, nullable=False, server_default="2"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.bulk_insert(
        assets,
        [
            {
                "id": DEFAULT_ASSET,
                "name": "Compressor de ar da APU — Trem 042",
                "installation": "Metro do Porto",
                "system": "Unidade de produção de ar (APU)",
                "equipment_unit": "Compressor de ar",
                "subunit": "Compressor + secador de torre dupla",
                "manufacturer_model": "MetroPT-3",
                "criticality": 3,
            }
        ],
    )

    with op.batch_alter_table("predictions") as batch:
        batch.add_column(
            sa.Column(
                "asset_id", sa.String(64), nullable=False, server_default=DEFAULT_ASSET
            )
        )
        batch.create_foreign_key("fk_predictions_asset", "assets", ["asset_id"], ["id"])
    op.create_index(
        "ix_predictions_asset_timestamp", "predictions", ["asset_id", "timestamp"]
    )

    op.create_table(
        "failure_events",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "asset_id", sa.String(64), sa.ForeignKey("assets.id"), nullable=False
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("severity", sa.String(16), nullable=False),
        sa.Column("peak_probability", sa.Float, nullable=False),
        sa.Column("mean_probability", sa.Float, nullable=False),
        sa.Column("n_samples", sa.Integer, nullable=False),
        sa.Column("model_name", sa.String(64), nullable=True),
        sa.Column("threshold", sa.Float, nullable=False),
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
    )
    op.create_index("ix_failure_events_started_at", "failure_events", ["started_at"])
    op.create_index(
        "ix_failure_events_asset_started", "failure_events", ["asset_id", "started_at"]
    )

    op.create_table(
        "work_orders",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "event_id",
            sa.Integer,
            sa.ForeignKey("failure_events.id"),
            nullable=False,
            unique=True,
        ),
        sa.Column(
            "asset_id", sa.String(64), sa.ForeignKey("assets.id"), nullable=False
        ),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("assignee", sa.String(100), nullable=True),
        sa.Column(
            "opened_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("parts", sa.JSON, nullable=False),
        sa.Column("confirmed_cause", sa.String(32), nullable=True),
        sa.Column("is_false_alarm", sa.Boolean, nullable=True),
        sa.Column("notes", sa.Text, nullable=True),
    )
    op.create_index("ix_work_orders_asset_id", "work_orders", ["asset_id"])

    op.create_table(
        "root_cause_analyses",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "event_id",
            sa.Integer,
            sa.ForeignKey("failure_events.id"),
            nullable=False,
            unique=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("method", sa.String(32), nullable=False),
        sa.Column("sensor_contributions", sa.JSON, nullable=False),
        sa.Column("suggestion_markdown", sa.Text, nullable=True),
        sa.Column("references", sa.JSON, nullable=False),
        sa.Column("grounding", sa.JSON, nullable=True),
        sa.Column("llm_model", sa.String(100), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("root_cause_analyses")
    op.drop_index("ix_work_orders_asset_id", table_name="work_orders")
    op.drop_table("work_orders")
    op.drop_index("ix_failure_events_asset_started", table_name="failure_events")
    op.drop_index("ix_failure_events_started_at", table_name="failure_events")
    op.drop_table("failure_events")
    op.drop_index("ix_predictions_asset_timestamp", table_name="predictions")
    with op.batch_alter_table("predictions") as batch:
        batch.drop_constraint("fk_predictions_asset", type_="foreignkey")
        batch.drop_column("asset_id")
    op.drop_table("assets")
