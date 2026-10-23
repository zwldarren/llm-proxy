"""add fallback count to request logs

A fallback row that succeeded is a plain ``2xx`` with an empty ``error_message``:
until now nothing in SQL could answer "which requests had a provider fail?" — the
count existed only inside ``log_metadata.fallback_attempts``, which is neither
filterable across dialects nor indexed. Storing it as a dedicated column keeps the
same split the token columns use: a recorded fact (0 / >0) is distinct from an
unknown one (NULL, rows written before the column existed).

Revision ID: 41126f5ec832
Revises: c4d2a91e7b33
Create Date: 2026-10-22 23:57:44.121425

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "41126f5ec832"
down_revision: str | None = "c4d2a91e7b33"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade database schema."""
    with op.batch_alter_table("request_logs", schema=None) as batch_op:
        batch_op.add_column(sa.Column("fallback_count", sa.Integer(), nullable=True))


def downgrade() -> None:
    """Downgrade database schema."""
    with op.batch_alter_table("request_logs", schema=None) as batch_op:
        batch_op.drop_column("fallback_count")
