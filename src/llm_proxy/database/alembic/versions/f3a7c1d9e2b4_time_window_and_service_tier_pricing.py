"""time window, reasoning and service tier pricing

Revision ID: f3a7c1d9e2b4
Revises: e1f0a2b3c4d5
Create Date: 2026-10-31 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f3a7c1d9e2b4"
down_revision: str | None = "e1f0a2b3c4d5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: The two price lists that carry the new overlay dimensions, and the columns
#: each one gains. Both tables keep the whole list as JSON (matching
#: ``pricing_tiers``), so no join lands on the hot billing path.
_TABLES = ("models", "model_providers")


def upgrade() -> None:
    """Upgrade database schema."""
    for table in _TABLES:
        with op.batch_alter_table(table, schema=None) as batch_op:
            batch_op.add_column(sa.Column("reasoning_cost_per_1m", sa.Float(), nullable=True))
            batch_op.add_column(sa.Column("pricing_time_windows", sa.JSON(), nullable=True))
            batch_op.add_column(sa.Column("service_tier_pricing", sa.JSON(), nullable=True))


def downgrade() -> None:
    """Downgrade database schema."""
    for table in _TABLES:
        with op.batch_alter_table(table, schema=None) as batch_op:
            batch_op.drop_column("service_tier_pricing")
            batch_op.drop_column("pricing_time_windows")
            batch_op.drop_column("reasoning_cost_per_1m")
