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

    async def test_starts_background_writers(self):
        app = FastAPI()
        logging_config = MagicMock()
        logging_config.retention_days = 45
        with (
            patch.object(lifecycle, "start_background_log_writer", MagicMock()) as start_log,
            patch.object(lifecycle, "start_background_usage_writer", MagicMock()) as start_usage,
            patch.object(
                lifecycle, "resolve_logging_config", MagicMock(return_value=logging_config)
            ),
        ):
            await lifecycle.startup_background_services(app)

        start_log.assert_called_once()
        # Usage records share the UI-managed retention window and get the config
        # manager so later settings changes are picked up per sweep.
        start_usage.assert_called_once_with(45, None)


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


class _AsyncSessionContext:
    """Minimal async context manager standing in for get_async_session_context."""

    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, *exc_info):
        return False


class TestSyncCircuitBreakerConfig:
    """A peer's resilience edit retunes this worker's circuit breaker store."""

    async def test_applies_changed_thresholds(self):
        from llm_proxy.config.types import CircuitBreakerParams
        from llm_proxy.core.circuit_breaker import CircuitBreakerStore

        app = FastAPI()
        store = CircuitBreakerStore()
        app.state.circuit_breaker = store

        config = MagicMock()
        config.server_params.circuit_breaker = CircuitBreakerParams(
            enabled=True, failure_threshold=9, cooldown_seconds=12.5
        )

        await lifecycle._sync_circuit_breaker_config(app, config)

        assert store.config.enabled is True
        assert store.config.failure_threshold == 9
        assert store.config.cooldown_seconds == 12.5

    async def test_is_a_noop_without_a_store(self):
        app = FastAPI()
        config = MagicMock()
        config.server_params.circuit_breaker = MagicMock()

        # Must not raise when circuit breaking is not initialized.
        await lifecycle._sync_circuit_breaker_config(app, config)

    async def test_startup_registers_a_listener_that_retunes_the_store(self):
        from llm_proxy.config.types import CircuitBreakerParams
        from llm_proxy.core.circuit_breaker import CircuitBreakerStore

        app = FastAPI()
        listeners = []
        config_manager = MagicMock()
        config_manager.add_reload_listener = listeners.append
        initial = MagicMock()
        initial.server_params.circuit_breaker = CircuitBreakerParams(
            enabled=True, failure_threshold=5, cooldown_seconds=30.0
        )
        config_manager.get_config = AsyncMock(return_value=initial)
        app.state.config_manager = config_manager

        await lifecycle.startup_circuit_breaker(app)

        assert isinstance(app.state.circuit_breaker, CircuitBreakerStore)
        assert len(listeners) == 1

        # A peer's resilience edit reaches the store through the listener.
        updated = MagicMock()
        updated.server_params.circuit_breaker = CircuitBreakerParams(
            enabled=False, failure_threshold=2, cooldown_seconds=1.0
        )
        await listeners[0](updated)

        assert app.state.circuit_breaker.config.enabled is False
        assert app.state.circuit_breaker.config.failure_threshold == 2


class TestReconcileMcpServers:
    """Peers reconcile MCP child processes against the persisted state."""

    @staticmethod
    def _server(name: str, *, command: str = "npx") -> MagicMock:
        """Build a server record mock carrying a real, fingerprintable definition."""
        from llm_proxy.database.tables import McpServerRecord

        server = MagicMock(spec=McpServerRecord)
        server.name = name
        server.type = "stdio"
        server.command = command
        server.args = ["-y", "some-package"]
        server.env = {}
        server.base_url = None
        return server

    @staticmethod
    def _manager(fingerprints: dict[str, str]) -> MagicMock:
        manager = MagicMock()
        manager.list_server_fingerprints = AsyncMock(return_value=fingerprints)
        manager.start_server = AsyncMock()
        manager.stop_server = AsyncMock()
        manager.restart_server = AsyncMock()
        return manager

    async def test_starts_desired_and_stops_stale_servers(self):
        app = FastAPI()
        manager = self._manager({"stale": "old-fingerprint"})
        app.state.mcp_manager = manager

        mcp_repo = MagicMock()
        mcp_repo.get_all_servers = AsyncMock(return_value=[self._server("fresh")])
        repo = MagicMock()
        repo.mcp_servers = mcp_repo
        session = MagicMock()

        with (
            patch(
                "llm_proxy.database.get_async_session_context",
                return_value=_AsyncSessionContext(session),
            ),
            patch("llm_proxy.database.ConfigRepository", return_value=repo),
        ):
            await lifecycle._reconcile_mcp_servers(app)

        manager.stop_server.assert_awaited_once_with(mcp_repo, "stale")
        manager.start_server.assert_awaited_once_with(mcp_repo, "fresh")
        manager.restart_server.assert_not_awaited()

    async def test_restarts_a_server_whose_definition_changed(self):
        """A config edit keeps the enabled set unchanged, so it is fingerprint-only."""
        from llm_proxy.mcp.manager import MCPProxyManager

        app = FastAPI()
        changed = self._server("changed", command="npx")
        manager = self._manager({"changed": "a-different-definition"})
        app.state.mcp_manager = manager

        mcp_repo = MagicMock()
        mcp_repo.get_all_servers = AsyncMock(return_value=[changed])
        repo = MagicMock()
        repo.mcp_servers = mcp_repo
        session = MagicMock()

        with (
            patch(
                "llm_proxy.database.get_async_session_context",
                return_value=_AsyncSessionContext(session),
            ),
            patch("llm_proxy.database.ConfigRepository", return_value=repo),
        ):
            await lifecycle._reconcile_mcp_servers(app)

        # Sanity-check the fingerprint the reconciler compares against.
        assert MCPProxyManager.server_fingerprint(changed) != "a-different-definition"
        manager.restart_server.assert_awaited_once_with(mcp_repo, "changed")
        manager.start_server.assert_not_awaited()
        manager.stop_server.assert_not_awaited()

    async def test_unchanged_fingerprint_is_left_running(self):
        from llm_proxy.mcp.manager import MCPProxyManager

        app = FastAPI()
        server = self._server("stable")
        manager = self._manager({"stable": MCPProxyManager.server_fingerprint(server)})
        app.state.mcp_manager = manager

        mcp_repo = MagicMock()
        mcp_repo.get_all_servers = AsyncMock(return_value=[server])
        repo = MagicMock()
        repo.mcp_servers = mcp_repo
        session = MagicMock()

        with (
            patch(
                "llm_proxy.database.get_async_session_context",
                return_value=_AsyncSessionContext(session),
            ),
            patch("llm_proxy.database.ConfigRepository", return_value=repo),
        ):
            await lifecycle._reconcile_mcp_servers(app)

        manager.start_server.assert_not_awaited()
        manager.stop_server.assert_not_awaited()
        manager.restart_server.assert_not_awaited()

    async def test_is_a_noop_without_a_manager(self):
        app = FastAPI()

        # Must not even open a session when MCP is not initialized.
        with patch("llm_proxy.database.get_async_session_context") as session_ctx:
            await lifecycle._reconcile_mcp_servers(app)

        session_ctx.assert_not_called()


class TestStartupRedisGenerationChannel:
    """The cross-worker generation channel follows Redis, not the config cache."""

    @staticmethod
    def _config_manager(*, cache_enabled: bool):
        config = MagicMock()
        config.redis.enabled = True
        config.redis.cache.enabled = cache_enabled

        manager = MagicMock()
        manager.get_config = AsyncMock(return_value=config)
        manager.enable_cache = MagicMock()
        manager.set_generation_channel = MagicMock()
        manager.sync_generation = AsyncMock()
        return manager

    @staticmethod
    def _redis_wrapper():
        wrapper = MagicMock()
        wrapper.client = MagicMock()
        return wrapper

    async def test_enables_the_channel_without_config_caching(self):
        app = FastAPI()
        config_manager = self._config_manager(cache_enabled=False)

        with (
            patch.object(
                lifecycle, "get_redis_client", AsyncMock(return_value=self._redis_wrapper())
            ),
            patch(
                "llm_proxy.observability.user_tracing.get_user_tracing_manager",
                return_value=MagicMock(),
            ),
        ):
            await lifecycle.startup_redis(app, config_manager)

        # REDIS_CACHE_ENABLED only governs the provider/model cache; the
        # generation channel must still be on so peers hot-reload.
        config_manager.enable_cache.assert_not_called()
        config_manager.set_generation_channel.assert_called_once()
        config_manager.sync_generation.assert_awaited_once()

    async def test_config_cache_also_carries_the_channel(self):
        app = FastAPI()
        config_manager = self._config_manager(cache_enabled=True)

        with (
            patch.object(
                lifecycle, "get_redis_client", AsyncMock(return_value=self._redis_wrapper())
            ),
            patch(
                "llm_proxy.observability.user_tracing.get_user_tracing_manager",
                return_value=MagicMock(),
            ),
        ):
            await lifecycle.startup_redis(app, config_manager)

        config_manager.enable_cache.assert_called_once()
        config_manager.set_generation_channel.assert_not_called()
        config_manager.sync_generation.assert_awaited_once()


class TestDerivedCacheListeners:
    """Reload listeners drop caches built from tables outside ProxyConfig."""

    async def test_listener_clears_key_role_and_mcp_policy_caches(self):
        from llm_proxy.api.middleware import api_key_cache
        from llm_proxy.api.routers import logs as logs_module
        from llm_proxy.api.routers import mcp as mcp_module

        listeners = []
        config_manager = MagicMock()
        config_manager.add_reload_listener = listeners.append

        with (
            patch.object(api_key_cache, "invalidate_api_key_cache") as invalidate_keys,
            patch.object(logs_module, "clear_user_role_cache") as clear_roles,
            patch.object(mcp_module.mcp_proxy_app, "reset_policy_cache") as reset_policy,
        ):
            await lifecycle.startup_derived_caches(config_manager)
            assert len(listeners) == 2
            for listener in listeners:
                await listener(MagicMock())

        invalidate_keys.assert_called_once()
        clear_roles.assert_called_once()
        reset_policy.assert_called_once()
