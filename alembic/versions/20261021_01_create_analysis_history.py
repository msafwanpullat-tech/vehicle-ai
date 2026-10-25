"""Create PostgreSQL analysis history table.

Revision ID: 20261021_01
Revises:
Create Date: 2026-10-21
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20261021_01"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if "analysis_records" in inspector.get_table_names():
        raise RuntimeError(
            "The PostgreSQL table analysis_records already exists. "
            "Inspect its schema before applying this migration."
        )

    op.create_table(
        "analysis_records",
        sa.Column("id", sa.String(length=32), primary_key=True),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("line_position_percent", sa.Float(), nullable=True),
        sa.Column("total_count", sa.Integer(), nullable=True),
        sa.Column(
            "class_counts",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "tracking_events",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "anpr_records",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "search_results",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "artifact_paths",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("manifest", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("failure_message", sa.String(length=512), nullable=True),
        sa.Column(
            "export_metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
    )
    op.create_index("ix_analysis_records_filename", "analysis_records", ["filename"])
    op.create_index("ix_analysis_records_created_at", "analysis_records", ["created_at"])
    op.create_index("ix_analysis_records_status", "analysis_records", ["status"])
    op.create_index(
        "ix_analysis_records_status_created_at",
        "analysis_records",
        ["status", "created_at"],
    )


def downgrade() -> None:
    raise RuntimeError(
        "Destructive downgrade is disabled to protect persistent vehicle analysis history."
    )
