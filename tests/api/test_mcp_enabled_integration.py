"""Integration test for MCP server enabled toggle.

This test uses an in-memory SQLite database to verify the actual behavior.
"""

import pytest
from sqlalchemy import StaticPool
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from llm_proxy.database.repositories.config_mcp import McpServerRepository
from llm_proxy.database.tables import Base


class TestMCPServerEnabledIntegration:
    """Integration test for MCP server enabled toggle."""

    @pytest.mark.asyncio
    async def test_update_enabled_field_directly(self):
        """Test that updating only the enabled field works correctly.

        This simulates what happens when user toggles the enabled switch.
        """
        # Create async engine for SQLite
        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            echo=False,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )

        try:
            # Create tables
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            # Create session
            async_session = async_sessionmaker(engine, expire_on_commit=False)

            async with async_session() as session:
                # Create a server
                repo = McpServerRepository(session)
                server = await repo.create_server(
                    name="test-server",
                    type="stdio",
                    command="test-cmd",
                    enabled=True,
                )
                await session.commit()

                # Verify initial state
                assert server.enabled is True

                # Now update only enabled field (simulating toggle off)
                updated = await repo.update_server(
                    "test-server",
                    enabled=False,
                )
                await session.commit()

                # Verify the update
                assert updated is not None
                assert updated.enabled is False, (
                    f"Expected enabled=False, got enabled={updated.enabled}"
                )

                # Fetch from database to confirm persistence
                fetched = await repo.get_server("test-server")
                assert fetched is not None
                assert fetched.enabled is False, (
                    f"After fetch: expected enabled=False, got enabled={fetched.enabled}"
                )
        finally:
            await engine.dispose()

    @pytest.mark.asyncio
    async def test_update_enabled_and_proxy_url(self):
        """Test updating enabled and proxy_url in sequence.

        This simulates what happens in the actual bug scenario where
        stop_server updates proxy_url and then update_mcp_server updates enabled.
        """
        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            echo=False,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )

        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            async_session = async_sessionmaker(engine, expire_on_commit=False)

            async with async_session() as session:
                repo = McpServerRepository(session)

                # Create server with enabled=True and proxy_url set
                await repo.create_server(
                    name="test-server",
                    type="stdio",
                    command="test-cmd",
                    enabled=True,
                )
                # Simulate server being started (proxy_url set)
                await repo.update_server("test-server", proxy_url="/servers/test-server/mcp")
                await session.commit()

                # Verify initial state
                fetched = await repo.get_server("test-server")
                assert fetched is not None
                assert fetched.enabled is True
                assert fetched.proxy_url == "/servers/test-server/mcp"

                # Now simulate what happens when disabling:
                # 1. stop_server updates proxy_url to None
                await repo.update_server("test-server", proxy_url=None)
                # 2. update_mcp_server updates enabled to False
                await repo.update_server("test-server", enabled=False)
                await session.commit()

                # Verify final state
                fetched = await repo.get_server("test-server")
                assert fetched is not None
                assert fetched.enabled is False, (
                    f"Expected enabled=False, got enabled={fetched.enabled}"
                )
                assert fetched.proxy_url is None, (
                    f"Expected proxy_url=None, got proxy_url={fetched.proxy_url}"
                )
        finally:
            await engine.dispose()

    @pytest.mark.asyncio
    async def test_update_enabled_with_kwargs(self):
        """Test that update_server correctly handles enabled in kwargs."""
        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            echo=False,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )

        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            async_session = async_sessionmaker(engine, expire_on_commit=False)

            async with async_session() as session:
                repo = McpServerRepository(session)

                # Create server
                await repo.create_server(
                    name="test-server",
                    type="stdio",
                    command="test-cmd",
                    enabled=True,
                )
                await session.commit()

                # Update with kwargs (like the actual code does)
                update_data = {"enabled": False}
                updated = await repo.update_server("test-server", **update_data)
                await session.commit()

                assert updated is not None
                assert updated.enabled is False, (
                    f"Expected enabled=False, got enabled={updated.enabled}"
                )
        finally:
            await engine.dispose()
