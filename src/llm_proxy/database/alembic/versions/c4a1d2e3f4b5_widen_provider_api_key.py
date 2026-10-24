"""widen provider api_key for encrypted values

Fernet tokens carry ``enc:`` plus base64 of the version, timestamp, IV,
padded ciphertext, and HMAC — roughly 190 characters more than the plaintext.
A legacy plaintext key that fit the old ``String(500)`` column therefore
cannot be written back as ciphertext; re-encryption at startup failed with a
PostgreSQL ``StringDataRightTruncation`` and aborted the service. ``Text``
removes the artificial bound on a credential field whose real bound belongs to
the provider.

Revision ID: c4a1d2e3f4b5
Revises: b7e2c9f40a11
Create Date: 2026-10-24 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c4a1d2e3f4b5"
down_revision: str | None = "b7e2c9f40a11"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade database schema."""
    with op.batch_alter_table("providers", schema=None) as batch_op:
        batch_op.alter_column(
            "api_key",
            existing_type=sa.String(length=500),
            type_=sa.Text(),
            existing_nullable=False,
        )


def downgrade() -> None:
    """Downgrade database schema."""
    with op.batch_alter_table("providers", schema=None) as batch_op:
        batch_op.alter_column(
            "api_key",
            existing_type=sa.Text(),
            type_=sa.String(length=500),
            existing_nullable=False,
        )
