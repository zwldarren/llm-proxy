"""add trace id to request logs

The proxy mints one trace id per request and hands it to the tracing backends as
their trace id (Langfuse uses it verbatim as ``trace_context``). Until now the id
existed only on the live ``EventContext`` and inside Langfuse's own metadata, so
a stored row could not be joined to the exported trace — the only correlation was
the request id embedded in Langfuse metadata, which is not queryable from SQL.

The column also answers the reverse question ("which rows belong to this trace?")
and is indexed for it: a trace id is a correlation key, not free-form content, so
it is a column rather than a ``log_metadata`` entry. NULL means the row predates
the column (old rows, and the non-endpoint situations that have no trace id).

Revision ID: b7e2c9f40a11
Revises: 41126f5ec832
Create Date: 2026-10-25 09:12:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b7e2c9f40a11"
down_revision: str | None = "41126f5ec832"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade database schema."""
    with op.batch_alter_table("request_logs", schema=None) as batch_op:
        # 255, not 64: the value comes from a caller-supplied header on trusted
        # peers, so it must not be able to overflow the column and cost the
        # request its row (`api.dependencies.extract_trace_id` drops anything
        # longer, and `database.tables.TRACE_ID_MAX_LENGTH` is the same number).
        batch_op.add_column(sa.Column("trace_id", sa.String(length=255), nullable=True))
        batch_op.create_index("ix_request_logs_trace_id", ["trace_id"], unique=False)


def downgrade() -> None:
    """Downgrade database schema."""
    with op.batch_alter_table("request_logs", schema=None) as batch_op:
        batch_op.drop_index("ix_request_logs_trace_id")
        batch_op.drop_column("trace_id")
