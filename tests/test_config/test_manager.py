"""Tests for configuration manager."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_proxy.config.manager import (
    DatabaseConfigManager,
    load_auth_config,
    load_logging_config,
    load_redis_config,
)
from llm_proxy.config.secrets import (
    ENCRYPTION_KEY_STORE,
    JWT_SECRET_KEY,
    ensure_secrets,
    get_encryption_key,
    get_jwt_secret,
    reset_secrets,
)
from llm_proxy.config.settings import reset_settings
from llm_proxy.core.exceptions import ConfigurationError, ProviderNotConfiguredError


@pytest.fixture(autouse=True)
def reset_settings_before_each_test():
    """Reset the settings/secrets singletons so patch.dict(os.environ) takes effect."""
    reset_settings()
    reset_secrets()
    yield
    reset_secrets()


class TestLoadAuthConfig:
    """Tests for load_auth_config function."""

    def test_loads_jwt_secret_from_env(self):
        with patch.dict(os.environ, {"JWT_SECRET": "a" * 32}, clear=False):
            result = load_auth_config()
            assert result.jwt_secret == "a" * 32

    def test_missing_jwt_secret_raises_error(self):
        # No JWT_SECRET set and secrets not initialized: cannot resolve a secret.
        with (
            patch.dict(os.environ, {"JWT_SECRET": ""}, clear=False),
            pytest.raises(ConfigurationError, match="not initialized"),
        ):
            load_auth_config()

    def test_jwt_secret_too_short_raises_error(self):
        with (
            patch.dict(os.environ, {"JWT_SECRET": "short"}, clear=False),
            pytest.raises(ConfigurationError, match="not initialized"),
        ):
            load_auth_config()


class TestLoadLoggingConfig:
    """Tests for load_logging_config function."""

    def test_default_values(self):
        with patch.dict(os.environ, {}, clear=False):
            result = load_logging_config()
            assert result.enable_database_logging is True
            assert result.retention_days == 30
            assert result.mask_sensitive_data is True

    def test_custom_values(self):
        overrides = {
            "log_input_output": "false",
            "log_retention_days": 60,
            "mask_sensitive_data": "false",
            "sampling_rate": 0.5,
        }
        env = {"LOG_LEVEL": "DEBUG"}
        with patch.dict(os.environ, env, clear=False):
            result = load_logging_config(overrides)
            assert result.enable_database_logging is False
            assert result.retention_days == 60
            assert result.mask_sensitive_data is False
            assert result.log_level == "DEBUG"
            assert result.sampling_rate == 0.5

    def test_with_overrides(self):
        overrides = {"log_retention_days": 90}
        result = load_logging_config(overrides)
        assert result.retention_days == 90

    def test_with_log_input_output_override(self):
        overrides = {"log_input_output": False}
        result = load_logging_config(overrides)
        assert result.enable_database_logging is False

    def test_verbose_routing_logs_default_is_false(self):
        result = load_logging_config()
        assert result.verbose_routing_logs is False

    def test_verbose_routing_logs_override_true(self):
        overrides = {"verbose_routing_logs": True}
        result = load_logging_config(overrides)
        assert result.verbose_routing_logs is True

    def test_verbose_routing_logs_override_string_true(self):
        overrides = {"verbose_routing_logs": "true"}
        result = load_logging_config(overrides)
        assert result.verbose_routing_logs is True

    def test_verbose_routing_logs_override_string_false(self):
        overrides = {"verbose_routing_logs": "false"}
        result = load_logging_config(overrides)
        assert result.verbose_routing_logs is False

    def test_with_mask_sensitive_data_override(self):
        result = load_logging_config({"mask_sensitive_data": False})
        assert result.mask_sensitive_data is False

    def test_with_audit_retention_days_override(self):
        result = load_logging_config({"audit_retention_days": 7})
        assert result.audit_retention_days == 7

    def test_with_audit_sampling_rate_override(self):
        result = load_logging_config({"audit_sampling_rate": 0.25})
        assert result.audit_sampling_rate == 0.25
        assert result.get_sampling_rate("audit") == 0.25
        assert result.get_sampling_rate("request") == 1.0

    def test_invalid_retention_days_override_raises_error(self):
        overrides = {"log_retention_days": "abc"}
        with pytest.raises(ConfigurationError, match="Invalid log_retention_days"):
            load_logging_config(overrides)

    def test_invalid_audit_retention_days_override_raises_error(self):
        with pytest.raises(ConfigurationError, match="Invalid audit_retention_days"):
            load_logging_config({"audit_retention_days": "abc"})

    def test_invalid_audit_sampling_rate_override_raises_error(self):
        with pytest.raises(ConfigurationError, match="Invalid audit_sampling_rate"):
            load_logging_config({"audit_sampling_rate": "abc"})

    def test_sampling_rate_invalid_raises_error(self):
        with pytest.raises(ConfigurationError, match="Invalid sampling_rate"):
            load_logging_config({"sampling_rate": "abc"})

    def test_sampling_rate_out_of_range_raises_error(self):
        with pytest.raises(ValueError, match="less than or equal to 1"):
            load_logging_config({"sampling_rate": 2.0})
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            load_logging_config({"sampling_rate": -0.5})

    def test_custom_sensitive_keys(self):
        result = load_logging_config({"sensitive_keys": "custom_key,another_key"})
        assert "custom_key" in result.sensitive_keys
        assert "another_key" in result.sensitive_keys

    def test_custom_sensitive_keys_as_list(self):
        result = load_logging_config({"sensitive_keys": ["Custom_Key", " another_key "]})
        assert "custom_key" in result.sensitive_keys
        assert "another_key" in result.sensitive_keys


class TestLoadRedisConfig:
    """Tests for load_redis_config function."""

    def test_redis_disabled(self):
        with patch.dict(os.environ, {"REDIS_ENABLED": "false"}, clear=False):
            result = load_redis_config()
            assert result.enabled is False

    def test_redis_enabled_basic(self):
        env = {"REDIS_ENABLED": "true", "REDIS_URL": "redis://localhost:6379"}
        with patch.dict(os.environ, env, clear=False):
            result = load_redis_config()
            assert result.enabled is True
            assert result.url == "redis://localhost:6379"

    def test_redis_custom_pool_size(self):
        env = {"REDIS_ENABLED": "true", "REDIS_POOL_SIZE": "20"}
        with patch.dict(os.environ, env, clear=False):
            result = load_redis_config()
            assert result.pool_size == 20

    def test_redis_pool_size_invalid_raises_error(self):
        env = {"REDIS_ENABLED": "true", "REDIS_POOL_SIZE": "not_a_number"}
        with (
            patch.dict(os.environ, env, clear=False),
            pytest.raises(ValueError, match="should be a valid integer"),
        ):
            load_redis_config()

    def test_redis_rate_limit_config(self):
        env = {
            "REDIS_ENABLED": "true",
            "REDIS_RATE_LIMIT_ENABLED": "true",
            "REDIS_RATE_LIMIT_PREFIX": "rl:",
        }
        with patch.dict(os.environ, env, clear=False):
            result = load_redis_config()
            assert result.rate_limit.enabled is True
            assert result.rate_limit.prefix == "rl:"

    def test_redis_cache_config(self):
        env = {
            "REDIS_ENABLED": "true",
            "REDIS_CACHE_ENABLED": "true",
            "REDIS_CACHE_PREFIX": "cache:",
            "REDIS_CACHE_TTL_PROVIDER_CONFIG": "600",
        }
        with patch.dict(os.environ, env, clear=False):
            result = load_redis_config()
            assert result.cache.enabled is True
            assert result.cache.ttl_provider_config == 600

    def test_redis_logging_config(self):
        env = {
            "REDIS_ENABLED": "true",
            "REDIS_LOGGING_ENABLED": "true",
            "REDIS_LOGGING_TTL_DAYS": "7",
        }
        with patch.dict(os.environ, env, clear=False):
            result = load_redis_config()
            assert result.logging.enabled is True
            assert result.logging.ttl_days == 7


class TestEnsureSecrets:
    """Tests for the config.secrets holder (DB-authoritative, env override)."""

    @pytest.mark.asyncio
    async def test_env_override_wins_and_skips_db(self):
        """A valid env var is used as-is; the database is never touched."""
        env = {"JWT_SECRET": "j" * 32, "ENCRYPTION_KEY": "e" * 32}
        with (
            patch.dict(os.environ, env, clear=False),
            patch("llm_proxy.database.connection.get_async_session_context") as mock_session,
        ):
            await ensure_secrets()
            assert get_jwt_secret() == "j" * 32
            assert get_encryption_key() == "e" * 32
            mock_session.assert_not_called()

    @pytest.mark.asyncio
    async def test_generates_and_persists_when_absent(self):
        """No env var and nothing stored: generate, persist, cache in memory."""
        env = {"JWT_SECRET": "", "ENCRYPTION_KEY": ""}
        with patch.dict(os.environ, env, clear=False):
            mock_repo = AsyncMock()
            mock_repo.get_server_config = AsyncMock(return_value=None)
            mock_repo.set_server_config = AsyncMock()

            with patch("llm_proxy.database.connection.get_async_session_context") as mock_session:
                mock_session.return_value.__aenter__ = AsyncMock(return_value=AsyncMock())
                mock_session.return_value.__aexit__ = AsyncMock()

                with patch(
                    "llm_proxy.database.repositories.ServerConfigRepository",
                    return_value=mock_repo,
                ):
                    await ensure_secrets()

                    jwt_secret = get_jwt_secret()
                    encryption_key = get_encryption_key()
                    assert len(jwt_secret) >= 32
                    assert len(encryption_key) >= 32
                    assert jwt_secret != encryption_key

                    # Both secrets persisted under their own keys.
                    persisted_keys = {
                        call.args[0] for call in mock_repo.set_server_config.call_args_list
                    }
                    assert persisted_keys == {JWT_SECRET_KEY, ENCRYPTION_KEY_STORE}

                    # Nothing is written back to the environment.
                    assert os.environ.get("JWT_SECRET") == ""
                    assert os.environ.get("ENCRYPTION_KEY") == ""

    @pytest.mark.asyncio
    async def test_loads_stored_secret(self):
        """No env var but a stored secret: load it, never overwrite it."""
        with patch.dict(os.environ, {"JWT_SECRET": ""}, clear=False):
            stored_secret = "s" * 40
            stored_record = MagicMock()
            stored_record.value = {"key": stored_secret}
            mock_repo = AsyncMock()
            mock_repo.get_server_config = AsyncMock(return_value=stored_record)
            mock_repo.set_server_config = AsyncMock()

            with patch("llm_proxy.database.connection.get_async_session_context") as mock_session:
                mock_session.return_value.__aenter__ = AsyncMock(return_value=AsyncMock())
                mock_session.return_value.__aexit__ = AsyncMock()

                with patch(
                    "llm_proxy.database.repositories.ServerConfigRepository",
                    return_value=mock_repo,
                ):
                    await ensure_secrets()
                    assert get_jwt_secret() == stored_secret
                    mock_repo.set_server_config.assert_not_called()
                    assert os.environ.get("JWT_SECRET") == ""

    @pytest.mark.asyncio
    async def test_stored_secret_too_short_regenerates(self):
        """A stored secret shorter than 32 chars is regenerated."""
        # ENCRYPTION_KEY uses a valid env override so only the JWT path hits the DB.
        env = {"JWT_SECRET": "", "ENCRYPTION_KEY": "e" * 32}
        with patch.dict(os.environ, env, clear=False):
            stored_record = MagicMock()
            stored_record.value = {"key": "short"}
            mock_repo = AsyncMock()
            mock_repo.get_server_config = AsyncMock(return_value=stored_record)
            mock_repo.set_server_config = AsyncMock()

            with patch("llm_proxy.database.connection.get_async_session_context") as mock_session:
                mock_session.return_value.__aenter__ = AsyncMock(return_value=AsyncMock())
                mock_session.return_value.__aexit__ = AsyncMock()

                with patch(
                    "llm_proxy.database.repositories.ServerConfigRepository",
                    return_value=mock_repo,
                ):
                    await ensure_secrets()
                    assert len(get_jwt_secret()) >= 32
                    mock_repo.set_server_config.assert_called_once()

    def test_get_jwt_secret_falls_back_to_env_before_startup(self):
        """Pre-startup callers (tests/CLI) can still use the env override directly."""
        with patch.dict(os.environ, {"JWT_SECRET": "a" * 32}, clear=False):
            assert get_jwt_secret() == "a" * 32


class TestDatabaseConfigManager:
    """Tests for DatabaseConfigManager class."""

    def test_init(self):
        manager = DatabaseConfigManager()
        assert manager._config is None
        assert manager._redis_cache is None
        assert manager._cache_enabled is False

    def test_enable_cache(self):
        manager = DatabaseConfigManager()
        mock_cache = MagicMock()
        manager.enable_cache(mock_cache)
        assert manager._cache_enabled is True
        assert manager._redis_cache == mock_cache


class TestGetProviderConfig:
    """Tests for DatabaseConfigManager.get_provider_config."""

    @pytest.mark.asyncio
    async def test_unknown_provider_raises_provider_not_configured(self):
        """Requesting an unconfigured provider maps to 404."""
        from llm_proxy.config.types import ProxyAuthConfig, ProxyConfig, ServerParams

        server_params = ServerParams(auth=ProxyAuthConfig(jwt_secret="a" * 32))
        manager = DatabaseConfigManager()
        empty_config = ProxyConfig(
            server_params=server_params,
            provider_configs={},
            models={},
        )
        # Patch get_config so no DB access is needed.
        manager.get_config = AsyncMock(return_value=empty_config)

        with pytest.raises(ProviderNotConfiguredError, match="missing-provider"):
            await manager.get_provider_config("missing-provider")
