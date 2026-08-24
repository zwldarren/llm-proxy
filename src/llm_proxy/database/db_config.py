"""Database configuration and URL generation."""

from pathlib import Path

import platformdirs
from pydantic import ValidationError

from llm_proxy.observability.logger import get_logger

logger = get_logger(__name__)

# Use platformdirs to get the standard user data directory for the OS
DEFAULT_DB_PATH = Path(platformdirs.user_data_dir("llm-proxy", ensure_exists=True)) / "config.db"


def get_db_url() -> str:
    """Get the database URL from settings or use default.

    Supports two environment variables (managed via pydantic-settings):
    - DATABASE_URL: Full database URL (e.g., postgresql+asyncpg://user:pass@host:port/db)
    - LLM_PROXY_DB_PATH: Path to SQLite database file

    If DATABASE_URL is set, it will be used directly. The URL should include the async driver:
    - PostgreSQL: postgresql+asyncpg://user:password@host:port/database
    - SQLite: sqlite+aiosqlite:///path/to/database.db

    If neither is set, defaults to SQLite at the platform-specific user data directory.

    Raises:
        ValidationError: If database-specific settings are invalid. This will propagate
            to fail-fast rather than silently falling back to SQLite.
    """
    from llm_proxy.config.settings import get_settings

    settings = get_settings().db

    # Check for full DATABASE_URL first (supports PostgreSQL and other databases)
    database_url = settings.database_url
    if database_url:
        # Handle common PostgreSQL URL format without async driver
        if database_url.startswith("postgresql://"):
            database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql+asyncpg://", 1)
        return database_url

    # Fall back to SQLite with LLM_PROXY_DB_PATH or default
    db_path = settings.db_path or str(DEFAULT_DB_PATH)
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite+aiosqlite:///{db_path}"


def is_sqlite() -> bool:
    """Check if the current database is SQLite."""
    try:
        return get_db_url().startswith("sqlite")
    except ValidationError:
        # If database settings fail to validate, we cannot determine the type.
        # Re-raise to fail fast rather than guessing incorrectly.
        raise
    except Exception:
        # Unexpected errors - default to False, caller will handle appropriately
        return False
