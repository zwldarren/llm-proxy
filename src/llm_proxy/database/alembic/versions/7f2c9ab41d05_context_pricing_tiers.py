"""context_pricing_tiers

Revision ID: 7f2c9ab41d05
Revises: 35c396a461bc
Create Date: 2026-09-12 23:59:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "7f2c9ab41d05"
down_revision: str | None = "35c396a461bc"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade database schema."""
    with op.batch_alter_table("models", schema=None) as batch_op:
        batch_op.add_column(sa.Column("pricing_tiers", sa.JSON(), nullable=True))

    with op.batch_alter_table("model_providers", schema=None) as batch_op:
        batch_op.add_column(sa.Column("pricing_tiers", sa.JSON(), nullable=True))


def downgrade() -> None:
    """Downgrade database schema."""
    with op.batch_alter_table("model_providers", schema=None) as batch_op:
        batch_op.drop_column("pricing_tiers")

    with op.batch_alter_table("models", schema=None) as batch_op:
        batch_op.drop_column("pricing_tiers")
