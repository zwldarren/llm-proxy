"""batch mode: uploaded files and forwarded batch jobs

Revision ID: 9b6d2f0c4e71
Revises: f3a7c1d9e2b4
Create Date: 2026-11-06 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9b6d2f0c4e71"
down_revision: str | None = "f3a7c1d9e2b4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade database schema."""
    op.create_table(
        "batch_files",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("owner", sa.String(length=200), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("purpose", sa.String(length=32), nullable=False),
        sa.Column("bytes", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_batch_files_owner_created",
        "batch_files",
        ["owner", "created_at"],
        unique=False,
    )

    op.create_table(
        "batch_jobs",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("owner", sa.String(length=200), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("provider", sa.String(length=100), nullable=False),
        sa.Column("provider_type", sa.String(length=50), nullable=False),
        sa.Column("upstream_id", sa.String(length=200), nullable=False),
        sa.Column("upstream_file_id", sa.String(length=200), nullable=True),
        sa.Column("model", sa.String(length=300), nullable=False),
        sa.Column("upstream_model", sa.String(length=300), nullable=False),
        sa.Column("endpoint", sa.String(length=100), nullable=False),
        sa.Column("completion_window", sa.String(length=20), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("input_file_id", sa.String(length=64), nullable=False),
        sa.Column("output_file_id", sa.String(length=64), nullable=True),
        sa.Column("error_file_id", sa.String(length=64), nullable=True),
        sa.Column("request_counts", sa.JSON(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("custom_ids", sa.JSON(), nullable=False),
        sa.Column("errors", sa.JSON(), nullable=True),
        sa.Column("usage", sa.JSON(), nullable=True),
        sa.Column("results_materialized_at", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.Integer(), nullable=False),
        sa.Column("in_progress_at", sa.Integer(), nullable=True),
        sa.Column("finalizing_at", sa.Integer(), nullable=True),
        sa.Column("completed_at", sa.Integer(), nullable=True),
        sa.Column("failed_at", sa.Integer(), nullable=True),
        sa.Column("expired_at", sa.Integer(), nullable=True),
        sa.Column("cancelling_at", sa.Integer(), nullable=True),
        sa.Column("cancelled_at", sa.Integer(), nullable=True),
        sa.Column("expires_at", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="SET NULL"),
    )
    op.create_index("ix_batch_jobs_upstream_id", "batch_jobs", ["upstream_id"], unique=False)
    op.create_index("ix_batch_jobs_status", "batch_jobs", ["status"], unique=False)
    op.create_index(
        "ix_batch_jobs_owner_created", "batch_jobs", ["owner", "created_at"], unique=False
    )


def downgrade() -> None:
    """Downgrade database schema."""
    op.drop_index("ix_batch_jobs_owner_created", table_name="batch_jobs")
    op.drop_index("ix_batch_jobs_status", table_name="batch_jobs")
    op.drop_index("ix_batch_jobs_upstream_id", table_name="batch_jobs")
    op.drop_table("batch_jobs")

    op.drop_index("ix_batch_files_owner_created", table_name="batch_files")
    op.drop_table("batch_files")
