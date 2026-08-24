"""SQLAlchemy declarative base for all database models.

Extracted to its own module so that Alembic's env.py can import it
directly through the package system instead of using importlib hacks.
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all database models."""
