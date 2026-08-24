"""Database connection management."""

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from llm_proxy.config.settings import get_settings
from llm_proxy.database.db_config import get_db_url, is_sqlite
from llm_proxy.observability.logger import get_logger

logger = get_logger(__name__)

_engine = None
_async_session_factory = None
_db_initialized = False
_migrations_run = False


def _calculate_pool_size() -> int:
    """Calculate optimal database pool size based on CPU cores.

    Uses the formula: pool_size = cpu_cores * 2 + 1
    This provides enough connections for concurrent operations without
    over-allocating resources.

    Returns:
        Recommended pool size.
    """
    try:
        cpu_count = os.cpu_count() or 4
    except Exception:
        logger.debug("Failed to get CPU count, defaulting to 4", exc_info=True)
        cpu_count = 4
    return cpu_count * 2 + 1


def get_engine():
    """Get or create the async engine."""
    global _engine
    if _engine is None:
        db_url = get_db_url()
        engine_kwargs = {
            "echo": False,
            "future": True,
        }
        if not is_sqlite():
            settings = get_settings().db
            # Apply settings with fallbacks to calculated defaults when None
            calculated_pool_size = _calculate_pool_size()
            pool_size = (
                settings.pool_size if settings.pool_size is not None else calculated_pool_size
            )
            max_overflow = settings.max_overflow if settings.max_overflow is not None else pool_size
            # Pydantic already enforces ge constraints on these fields
            pool_recycle = settings.pool_recycle_seconds
            pool_timeout = settings.pool_timeout_seconds

            engine_kwargs = {
                **engine_kwargs,
                "pool_size": pool_size,
                "max_overflow": max_overflow,
                "pool_pre_ping": True,
                "pool_recycle": pool_recycle,
                "pool_timeout": pool_timeout,
            }

            logger.info(
                f"Database connection pool configured: "
                f"pool_size={pool_size}, max_overflow={max_overflow}, "
                f"pool_recycle={pool_recycle}s, pool_timeout={pool_timeout}s"
            )

        _engine = create_async_engine(db_url, **engine_kwargs)
        if is_sqlite():
            from sqlalchemy import event

            @event.listens_for(_engine.sync_engine, "connect")
            def set_sqlite_pragma(dbapi_connection, _connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

    return _engine


def get_session_factory():
    """Get or create the async session factory."""
    global _async_session_factory
    if _async_session_factory is None:
        _async_session_factory = async_sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _async_session_factory


async def _session_generator() -> AsyncIterator[AsyncSession]:
    """Common session generator logic."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        await session.commit()
    except BaseException:
        with suppress(Exception, asyncio.CancelledError):
            await asyncio.shield(session.rollback())
        raise
    finally:
        with suppress(Exception, asyncio.CancelledError):
            await asyncio.shield(session.close())


async def get_async_session() -> AsyncIterator[AsyncSession]:
    """Get an async database session for FastAPI dependency injection.

    This function is designed to be used with FastAPI's Depends() mechanism.
    FastAPI handles the generator lifecycle automatically, ensuring proper
    cleanup even if exceptions occur.

    **When to use**: For FastAPI route handlers and dependencies.

    **When NOT to use**: In background tasks, scheduled jobs, or any code
    outside the FastAPI request lifecycle. Use get_async_session_context() instead.

    Example::

        from fastapi import Depends
        from llm_proxy.database import get_async_session

        @router.get("/items")
        async def get_items(session: AsyncSession = Depends(get_async_session)):
            # session is automatically managed by FastAPI
            result = await session.execute(select(Item))
            return result.scalars().all()

    Yields:
        AsyncSession: An async SQLAlchemy session that auto-commits on success
            or rolls back on exception.
    """
    async for session in _session_generator():
        yield session


@asynccontextmanager
async def get_async_session_context() -> AsyncIterator[AsyncSession]:
    """Get an async database session as a context manager.

    This function provides a session for use outside FastAPI's dependency
    injection system. You MUST use it with `async with` to ensure proper
    cleanup and connection release.

    **When to use**: For background tasks, scheduled jobs, startup/shutdown
    events, or any code outside FastAPI route handlers.

    **When NOT to use**: In FastAPI route handlers where Depends() is available.
    Use get_async_session() with Depends() instead.

    Example::

        from llm_proxy.database import get_async_session_context

        async def background_log_writer():
            \"\"\"Background task that writes logs to database.\"\"\"
            async with get_async_session_context() as session:
                repo = LogRepository(session)
                await repo.save_log(log_entry)
                # Session auto-commits on successful exit

        async def startup_initialization():
            \"\"\"Code that runs during app startup.\"\"\"
            async with get_async_session_context() as session:
                repo = ConfigRepository(session)
                await repo.initialize_defaults()

    Note:
        Failure to use this as a context manager will leak database connections
        and may cause connection pool exhaustion. Always use `async with`.

    Yields:
        AsyncSession: An async SQLAlchemy session that auto-commits on success
            or rolls back on exception.
    """
    async for session in _session_generator():
        yield session


def run_migrations() -> None:
    """Run database migrations synchronously.

    Called via asyncio.to_thread() to avoid blocking the event loop.
    Alembic's env.py internally uses asyncio.run() for async engine support —
    this is safe because to_thread() runs in a separate OS thread with no
    running event loop.
    """
    from pathlib import Path

    from alembic import command
    from alembic.config import Config

    possible_paths = [
        Path.cwd() / "alembic.ini",
        Path(__file__).parent.parent.parent.parent / "alembic.ini",
    ]

    alembic_ini = None
    for path in possible_paths:
        if path.exists():
            alembic_ini = path
            break

    if alembic_ini is None:
        raise FileNotFoundError(
            "Could not find alembic.ini. Database migrations are required for schema management."
        )

    config = Config(str(alembic_ini))
    logger.info("Running database migrations...")
    command.upgrade(config, "head")
    logger.info("Database migrations completed successfully")


async def init_db() -> None:
    """Initialize the database by running Alembic migrations.

    Migrations are the single source of truth for the database schema.
    There is no create_all fallback — if migrations fail, startup fails
    loudly so the problem can be diagnosed and fixed.
    """
    global _db_initialized, _migrations_run
    if _db_initialized:
        return

    import asyncio

    if not _migrations_run:
        await asyncio.to_thread(run_migrations)
        _migrations_run = True

    _db_initialized = True


async def close_db() -> None:
    """Close the database connection."""
    global _engine, _async_session_factory, _db_initialized
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _async_session_factory = None
        _db_initialized = False
