"""add provider_oauth_credentials table

Revision ID: d9a1b2c3e4f5
Revises: c8d4e6f1a2b3
Create Date: 2026-11-24 00:55:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d9a1b2c3e4f5"
down_revision: str | None = "c8d4e6f1a2b3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade database schema."""
    op.create_table(
        "provider_oauth_credentials",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("provider_id", sa.Integer(), nullable=False),
        sa.Column("client_id", sa.String(length=255), nullable=False),
        sa.Column("client_secret", sa.Text(), nullable=True),
        sa.Column("ext_agent_host_id", sa.String(length=255), nullable=True),
        sa.Column("access_token", sa.Text(), nullable=False),
        sa.Column("refresh_token", sa.Text(), nullable=True),
        sa.Column("token_type", sa.String(length=50), nullable=False),
        sa.Column("scope", sa.Text(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("account_sub", sa.String(length=255), nullable=True),
        sa.Column("account_email", sa.String(length=320), nullable=True),
        sa.Column("account_name", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["provider_id"], ["providers.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("provider_id"),
    )
    op.create_index(
        "ix_provider_oauth_credentials_provider_id",
        "provider_oauth_credentials",
        ["provider_id"],
        unique=True,
    )


def downgrade() -> None:
    """Downgrade database schema."""
    op.drop_index(
        "ix_provider_oauth_credentials_provider_id",
        table_name="provider_oauth_credentials",
    )
    op.drop_table("provider_oauth_credentials")
