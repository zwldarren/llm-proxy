"""Tests for application lifecycle hooks (api/lifecycle.py).

These exercise the startup/shutdown helpers with their heavy collaborators
(http client, DB, config manager, tracing registry) replaced by mocks, so no
real services are required.
"""

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI

from llm_proxy.api import lifecycle


def _settings_mock() -> MagicMock:
    settings = MagicMock()
    settings.http.max_connections = 100
    settings.http.max_keepalive = 20
    settings.http.disable_http2 = False
    return settings


class TestStartupHttpClient:
    """startup_http_client wires a provider HTTP client onto app.state."""

    async def test_sets_http_client_on_state(self):
        app = FastAPI()
        manager = MagicMock()
        with (
            patch.object(lifecycle, "get_settings", return_value=_settings_mock()),
            patch.object(lifecycle, "ProviderHTTPClientManager", return_value=manager),
        ):
            await lifecycle.startup_http_client(app)

        assert app.state.http_client is manager


class TestStartupDatabase:
    """startup_database triggers the DB migration/init routine."""

    async def test_calls_init_db(self):
        app = FastAPI()
        with patch.object(lifecycle, "init_db", AsyncMock()) as init_db:
            await lifecycle.startup_database(app)

        init_db.assert_awaited_once()


class TestStartupConfig:
    """startup_config loads encryption secrets and the config manager."""

    async def test_initializes_encryption_and_config(self):
        app = FastAPI()
        config_manager = MagicMock()
        config_manager.load = AsyncMock()

        with (
            patch("llm_proxy.config.ensure_secrets", AsyncMock()),
            patch(
                "llm_proxy.config.get_encryption_key",
                MagicMock(return_value="test-encryption-key"),
            ),
            patch.object(lifecycle, "init_encryption", MagicMock()) as init_encryption,
            patch.object(lifecycle, "DatabaseConfigManager", return_value=config_manager),
        ):
            result = await lifecycle.startup_config(app)

        init_encryption.assert_called_once_with("test-encryption-key")
        config_manager.load.assert_awaited_once()
        assert app.state.config_manager is config_manager
        assert result is config_manager


class TestStartupBackgroundServices:
    """startup_background_services starts the background writers."""

    async def test_starts_writers_and_tool_log_service(self):
        app = FastAPI()
        get_tool_log_service = MagicMock()
        with (
            patch.object(lifecycle, "start_background_log_writer", MagicMock()) as start_log,
            patch.object(lifecycle, "start_background_usage_writer", MagicMock()) as start_usage,
            patch.object(
                lifecycle, "get_tool_log_service", return_value=get_tool_log_service
            ) as set_tool_log,
            patch("llm_proxy.observability.service.RequestLogService", MagicMock()),
            patch.object(lifecycle, "resolve_logging_config", MagicMock()),
        ):
            await lifecycle.startup_background_services(app)

        start_log.assert_called_once()
        start_usage.assert_called_once()
        set_tool_log.assert_called_once()


class TestShutdownServices:
    """shutdown_services tears down every active subsystem."""

    async def test_closes_present_subsystems(self):
        app = FastAPI()
        app.state.http_client = AsyncMock()
        app.state.http_client.close = AsyncMock()
        config_manager = MagicMock()
        config = MagicMock()
        config.redis.enabled = True
        config_manager.get_config = AsyncMock(return_value=config)
        app.state.config_manager = config_manager
        app.state.web_search_interceptor = AsyncMock()
        app.state.web_search_interceptor.close = AsyncMock()
        app.state.mcp_manager = MagicMock()
        app.state.mcp_manager.shutdown_all = AsyncMock()

        tracing_registry = MagicMock()
        tracing_registry.shutdown = AsyncMock()

        with (
            patch(
                "llm_proxy.observability.tracing.handlers.registry.get_tracing_registry",
                return_value=tracing_registry,
            ),
            patch.object(lifecycle, "stop_background_log_writer", AsyncMock()) as stop_log,
            patch.object(lifecycle, "stop_background_usage_writer", AsyncMock()) as stop_usage,
            patch.object(lifecycle, "close_db", AsyncMock()) as close_db,
            patch.object(lifecycle, "close_redis_client", AsyncMock()) as close_redis,
        ):
            await lifecycle.shutdown_services(app)

        tracing_registry.shutdown.assert_awaited_once()
        stop_log.assert_awaited_once()
        stop_usage.assert_awaited_once()
        close_db.assert_awaited_once()
        app.state.http_client.close.assert_awaited_once()
        close_redis.assert_awaited_once()
        app.state.mcp_manager.shutdown_all.assert_awaited_once()

    async def test_skips_absent_subsystems(self):
        app = FastAPI()
        # No http_client / config_manager / web_search / mcp on state.

        tracing_registry = MagicMock()
        tracing_registry.shutdown = AsyncMock()

        with (
            patch(
                "llm_proxy.observability.tracing.handlers.registry.get_tracing_registry",
                return_value=tracing_registry,
            ),
            patch.object(lifecycle, "stop_background_log_writer", AsyncMock()) as stop_log,
            patch.object(lifecycle, "stop_background_usage_writer", AsyncMock()) as stop_usage,
            patch.object(lifecycle, "close_db", AsyncMock()) as close_db,
            patch.object(lifecycle, "close_redis_client", AsyncMock()) as close_redis,
        ):
            await lifecycle.shutdown_services(app)

        # The always-on teardown still runs.
        tracing_registry.shutdown.assert_awaited_once()
        stop_log.assert_awaited_once()
        stop_usage.assert_awaited_once()
        close_db.assert_awaited_once()
        # Redis is only closed when a config manager is present and enabled.
        close_redis.assert_not_awaited()

    async def test_closes_redis_only_when_enabled(self):
        app = FastAPI()
        config_manager = MagicMock()
        config = MagicMock()
        config.redis.enabled = False
        config_manager.get_config = AsyncMock(return_value=config)
        app.state.config_manager = config_manager

        tracing_registry = MagicMock()
        tracing_registry.shutdown = AsyncMock()

        with (
            patch(
                "llm_proxy.observability.tracing.handlers.registry.get_tracing_registry",
                return_value=tracing_registry,
            ),
            patch.object(lifecycle, "stop_background_log_writer", AsyncMock()),
            patch.object(lifecycle, "stop_background_usage_writer", AsyncMock()),
            patch.object(lifecycle, "close_db", AsyncMock()),
            patch.object(lifecycle, "close_redis_client", AsyncMock()) as close_redis,
        ):
            await lifecycle.shutdown_services(app)

        close_redis.assert_not_awaited()
