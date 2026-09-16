"""dedup cache-read tokens persisted in both dialect columns

Revision ID: 1d85a77da088
Revises: 7f2c9ab41d05
Create Date: 2026-09-16 08:12:21.391408

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "1d85a77da088"
down_revision: str | None = "7f2c9ab41d05"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# The protocol-side streaming transformers (protocols/openai/streaming.py,
# protocols/anthropic/streaming.py) copied the cache-read count into BOTH
# cache_read_input_tokens (canonical flat field) and cached_prompt_tokens
# (OpenAI-dialect duplicate of the same fact) for every streamed Anthropic
# dialect response. Aggregations that sum both columns — e.g. the usage
# page's cache hit rate — therefore counted the same tokens twice and could
# exceed 100%. Per the canonical usage record (ADR-0006) cache-read is one
# fact expressed once: the flat field wins. Rows with both columns > 0 are
# exactly the duplicated rows; the dialect copy is zeroed.
_TABLES = ("usage_records", "request_logs")


def upgrade() -> None:
    """Repair rows carrying the cache-read count in both dialect columns."""
    bind = op.get_bind()
    for table in _TABLES:
        bind.execute(
            sa.text(
                f"UPDATE {table} SET cached_prompt_tokens = 0 "
                "WHERE cache_read_input_tokens > 0 AND cached_prompt_tokens > 0"
            )
        )


def downgrade() -> None:
    """Not reversible: the duplicated dialect copy was discarded, not moved."""
