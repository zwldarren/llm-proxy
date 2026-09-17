"""drop never-written request_logs body-compression columns

Revision ID: a1f4c7d92b30
Revises: 1d85a77da088
Create Date: 2026-09-17 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1f4c7d92b30"
down_revision: str | None = "1d85a77da088"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# The request/response body-compression feature was never wired up: no code
# path ever wrote these six columns. Reassembling the streaming log body in its
# non-streaming shape removed the reason to compress at all (see ADR-0015), so
# they are dropped instead of enabled.
_COLUMNS = (
    "request_body_compressed",
    "response_body_compressed",
    "request_body_compression",
    "response_body_compression",
    "request_body_original_size",
    "response_body_original_size",
)


def upgrade() -> None:
    """Drop the body-compression columns from ``request_logs``."""
    with op.batch_alter_table("request_logs", schema=None) as batch_op:
        for column in _COLUMNS:
            batch_op.drop_column(column)


def downgrade() -> None:
    """Restore the columns as empty placeholders (the feature stays unimplemented)."""
    with op.batch_alter_table("request_logs", schema=None) as batch_op:
        batch_op.add_column(sa.Column("request_body_compressed", sa.LargeBinary(), nullable=True))
        batch_op.add_column(sa.Column("response_body_compressed", sa.LargeBinary(), nullable=True))
        batch_op.add_column(
            sa.Column("request_body_compression", sa.String(length=20), nullable=True)
        )
        batch_op.add_column(
            sa.Column("response_body_compression", sa.String(length=20), nullable=True)
        )
        batch_op.add_column(sa.Column("request_body_original_size", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("response_body_original_size", sa.Integer(), nullable=True))
