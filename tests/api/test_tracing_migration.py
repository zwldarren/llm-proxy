"""Tests for the one-time system→per-user tracing config migration."""

import pytest
from sqlalchemy import StaticPool
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from llm_proxy.api.lifecycle import _migrate_global_tracing_to_admins
from llm_proxy.database import ConfigRepository, UserRepository
from llm_proxy.database.tables import Base
from llm_proxy.security.passwords import hash_password


@pytest.fixture
async def engine():
    eng = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


async def _with_session(engine, fn):
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        return await fn(session)


@pytest.mark.asyncio
async def test_migration_copies_global_config_to_admins_and_deletes_key(engine, monkeypatch):
    """Legacy system-level tracing config is moved into each admin's personal config."""

    # Seed a legacy global tracing config and two admin users (one already has personal).
    async def seed(session):
        await ConfigRepository(session).set_tracing_config(
            {
                "enabled": True,
                "providers": [
                    {
                        "provider": "langfuse",
                        "name": "x",
                        "enabled": True,
                        "settings": {"public_key": "pk", "secret_key": "sk"},
                    }
                ],
            }
        )
        repo = UserRepository(session)
        await repo.create_user("admin1", hash_password("Sup3rSecret!1"), role="admin")
        await repo.create_user("admin2", hash_password("Sup3rSecret!2"), role="admin")
        # admin2 already has a personal config → must NOT be overwritten.
        admin2 = await repo.get_by_username("admin2")
        assert admin2 is not None
        admin2.tracing_config = {"enabled": False, "providers": []}
        await session.commit()

    await _with_session(engine, seed)

    # Patch the session context the migration uses so it operates on our engine.
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    class _Ctx:
        def __aenter__(self):
            self._session = session_factory()
            return self._session.__aenter__()

        async def __aexit__(self, *exc):
            await self._session.__aexit__(*exc)

    monkeypatch.setattr("llm_proxy.database.get_async_session_context", lambda: _Ctx())

    await _migrate_global_tracing_to_admins()

    # The system-level key is gone.
    async def assert_results(session):
        assert await ConfigRepository(session).get_tracing_config() is None
        repo = UserRepository(session)
        admin1 = await repo.get_by_username("admin1")
        admin2 = await repo.get_by_username("admin2")
        assert admin1 is not None and admin2 is not None
        # admin1 inherited the legacy config.
        assert admin1.tracing_config is not None
        assert admin1.tracing_config["enabled"] is True
        # admin2 kept their pre-existing personal config (not overwritten).
        assert admin2.tracing_config == {"enabled": False, "providers": []}

    await _with_session(engine, assert_results)


@pytest.mark.asyncio
async def test_migration_is_noop_when_no_global_config(engine, monkeypatch):
    """With no legacy global config, the migration does nothing and does not error."""

    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    class _Ctx:
        def __aenter__(self):
            self._session = session_factory()
            return self._session.__aenter__()

        async def __aexit__(self, *exc):
            await self._session.__aexit__(*exc)

    monkeypatch.setattr("llm_proxy.database.get_async_session_context", lambda: _Ctx())

    await _migrate_global_tracing_to_admins()  # should not raise

    async def check(session):
        assert await ConfigRepository(session).get_tracing_config() is None

    await _with_session(engine, check)
