"""rerank capability

Revision ID: c4d2a91e7b33
Revises: 78a18792d186
Create Date: 2026-11-02 10:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c4d2a91e7b33"
down_revision: str | None = "78a18792d186"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade database schema."""
    with op.batch_alter_table("models", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("supports_rerank", sa.Boolean(), server_default=sa.false(), nullable=False)
        )


def downgrade() -> None:
    """Downgrade database schema."""
    with op.batch_alter_table("models", schema=None) as batch_op:
        batch_op.drop_column("supports_rerank")
