"""Alembic environment configuration for LLM Proxy.

This module configures Alembic to work with our async SQLAlchemy setup.
It supports both SQLite and PostgreSQL databases.
"""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from llm_proxy.database.base import Base
from llm_proxy.database.db_config import get_db_url

config = context.config

if config.config_file_name is not None:
    import logging

    if not logging.getLogger().hasHandlers():
        fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    Configures the context with a URL instead of an Engine, so no DBAPI is needed.
    """
    url = get_db_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # Compare types to detect column type changes
        compare_type=True,
        # Compare server defaults
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run migrations using the provided connection."""

    # Older databases may have alembic_version.version_num as VARCHAR(32).
    # New revision identifiers can be longer, so widen the column before
    # Alembic updates version_num during upgrade steps.
    if connection.dialect.name == "postgresql":
        column_info = connection.exec_driver_sql(
            """
            SELECT character_maximum_length
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'alembic_version'
              AND column_name = 'version_num'
            """
        ).scalar_one_or_none()

        if column_info is not None:
            try:
                current_length = int(column_info)
                if current_length < 128:
                    connection.exec_driver_sql(
                        "ALTER TABLE alembic_version ALTER COLUMN version_num TYPE VARCHAR(128)"
                    )
            except ValueError, TypeError:
                pass  # Column may have non-integer length or other edge case

    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        # Compare types to detect column type changes
        compare_type=True,
        # Compare server defaults
        compare_server_default=True,
        # Use batch mode only for SQLite (needed for ALTER TABLE support)
        render_as_batch=connection.dialect.name == "sqlite",
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in 'online' mode with async engine."""
    # Create async engine directly with our URL
    connectable = create_async_engine(
        get_db_url(),
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
        await connection.commit()

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    Wrapper that runs the async migration function via asyncio.run().
    """
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
