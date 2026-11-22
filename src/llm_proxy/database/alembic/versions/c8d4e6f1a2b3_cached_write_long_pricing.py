"""cached write long-lifetime pricing

Revision ID: c8d4e6f1a2b3
Revises: 7a1c5e9d4f20
Create Date: 2026-11-16 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c8d4e6f1a2b3"
down_revision: str | None = "7a1c5e9d4f20"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: Both price lists gain the rate: a model-level default and the per-provider
#: override that wins over it.
_TABLES = ("models", "model_providers")


def upgrade() -> None:
    """Upgrade database schema."""
    for table in _TABLES:
        with op.batch_alter_table(table, schema=None) as batch_op:
            batch_op.add_column(
                sa.Column("cached_write_long_cost_per_1m", sa.Float(), nullable=True)
            )


def downgrade() -> None:
    """Downgrade database schema."""
    for table in _TABLES:
        with op.batch_alter_table(table, schema=None) as batch_op:
            batch_op.drop_column("cached_write_long_cost_per_1m")
