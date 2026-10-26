"""add account-level rate limit to users

The team view owns member-level quota: model allowlists and account budgets.
A per-key rate limit cannot express that policy because keys are owner-scoped
(an admin cannot edit a member's key). This column carries the member-level
requests-per-minute cap instead, enforced by the request gate across every key
the member owns.

Revision ID: d8f2c1a4b6e9
Revises: c4a1d2e3f4b5
Create Date: 2026-10-26 05:30:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d8f2c1a4b6e9"
down_revision: str | None = "c4a1d2e3f4b5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade database schema."""
    with op.batch_alter_table("users", schema=None) as batch_op:
        batch_op.add_column(sa.Column("rate_limit_rpm", sa.Integer(), nullable=True))


def downgrade() -> None:
    """Downgrade database schema."""
    with op.batch_alter_table("users", schema=None) as batch_op:
        batch_op.drop_column("rate_limit_rpm")
