"""cascade model renames to model_experience

Revision ID: 2833ae93e0e4
Revises: b3f3198d9d83
Create Date: 2026-08-28 18:00:30.859759

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "2833ae93e0e4"
down_revision: str | None = "b3f3198d9d83"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# The FK from model_experience.name -> models.name is unnamed in the initial
# schema; on PostgreSQL the database generated the default name
# <table>_<column>_fkey. SQLite does not store FK constraint names, so batch
# mode needs a copy_from table carrying an explicit name to target the drop.
_FK_NAME = "model_experience_name_fkey"


def _existing_model_experience_table() -> sa.Table:
    """Current model_experience shape, with a named FK for batch rebuilds.

    Matches the initial schema (b3f3198d9d83) column-for-column so the SQLite
    copy-and-move rebuild preserves all data. Alembic cannot drop an *unnamed*
    FK constraint in batch mode, hence the explicit name here.
    """
    return sa.Table(
        "model_experience",
        sa.MetaData(),
        sa.Column("name", sa.String(length=100), primary_key=True, nullable=False),
        sa.Column("samples", sa.Integer(), nullable=False),
        sa.Column("reward_mean", sa.Float(), nullable=False),
        sa.Column("latency", sa.Float(), nullable=False),
        sa.Column("reliability", sa.Float(), nullable=False),
        sa.Column("feedback", sa.Float(), nullable=False),
        sa.Column("cache_affinity", sa.Float(), nullable=False),
        sa.Column("updated_at", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(["name"], ["models.name"], name=_FK_NAME, ondelete="CASCADE"),
    )


def upgrade() -> None:
    """Upgrade database schema."""
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.drop_constraint(_FK_NAME, "model_experience", type_="foreignkey")
        op.create_foreign_key(
            _FK_NAME,
            "model_experience",
            "models",
            ["name"],
            ["name"],
            onupdate="CASCADE",
            ondelete="CASCADE",
        )
    else:
        # SQLite cannot ALTER a foreign key; batch mode rebuilds the table.
        with op.batch_alter_table(
            "model_experience", copy_from=_existing_model_experience_table()
        ) as batch_op:
            batch_op.drop_constraint(_FK_NAME, type_="foreignkey")
            batch_op.create_foreign_key(
                _FK_NAME,
                "models",
                ["name"],
                ["name"],
                onupdate="CASCADE",
                ondelete="CASCADE",
            )


def downgrade() -> None:
    """Downgrade database schema."""
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.drop_constraint(_FK_NAME, "model_experience", type_="foreignkey")
        op.create_foreign_key(
            _FK_NAME,
            "model_experience",
            "models",
            ["name"],
            ["name"],
            ondelete="CASCADE",
        )
    else:
        with op.batch_alter_table(
            "model_experience", copy_from=_existing_model_experience_table()
        ) as batch_op:
            batch_op.drop_constraint(_FK_NAME, type_="foreignkey")
            batch_op.create_foreign_key(
                _FK_NAME,
                "models",
                ["name"],
                ["name"],
                ondelete="CASCADE",
            )
