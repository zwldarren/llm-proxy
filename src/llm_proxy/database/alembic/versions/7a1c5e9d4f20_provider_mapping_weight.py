"""provider mapping weight

Revision ID: 7a1c5e9d4f20
Revises: 9b6d2f0c4e71
Create Date: 2026-11-12 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "7a1c5e9d4f20"
down_revision: str | None = "9b6d2f0c4e71"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the traffic weight that splits load inside a priority group.

    Non-null with a server default so every existing mapping reads back as the
    neutral weight 1 (its previous behavior) instead of needing a backfill.
    """
    with op.batch_alter_table("model_providers", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("weight", sa.Float(), server_default=sa.text("1.0"), nullable=False)
        )


def downgrade() -> None:
    """Drop the provider mapping weight."""
    with op.batch_alter_table("model_providers", schema=None) as batch_op:
        batch_op.drop_column("weight")
