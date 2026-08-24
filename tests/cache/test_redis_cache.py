"""Tests for redis_cache.py."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from llm_proxy.cache.redis_cache import (
    RedisCache,
    _empty_cache_stats,
    _is_connection_error,
    get_redis_cache,
    invalidate_cache,
)
from llm_proxy.config.types.provider import ProviderConfig
from llm_proxy.config.types.redis import RedisCacheConfig


class TestIsConnectionError:
    """Tests for _is_connection_error helper."""

    def test_connection_error(self):
        exc = ConnectionError("test")
        assert _is_connection_error(exc) is True

    def test_timeout_error(self):
        exc = TimeoutError("test")
        assert _is_connection_error(exc) is True

    def test_redis_connection_error(self):
        exc = RedisConnectionError("test")
        assert _is_connection_error(exc) is True

    def test_redis_timeout_error(self):
        exc = RedisTimeoutError("test")
        assert _is_connection_error(exc) is True

    def test_os_error(self):
        exc = OSError("test")
        assert _is_connection_error(exc) is True

    def test_other_error(self):
        exc = ValueError("test")
        assert _is_connection_error(exc) is False

    def test_nested_connection_error(self):
        inner = ConnectionError("inner")
        outer = ValueError("outer")
        outer.__cause__ = inner
        assert _is_connection_error(outer) is True

    def test_deeply_nested_connection_error(self):
        inner = OSError("inner")
        middle = ValueError("middle")
        outer = TypeError("outer")
        middle.__cause__ = inner
        outer.__cause__ = middle
        assert _is_connection_error(outer) is True


class TestEmptyCacheStats:
    """Tests for _empty_cache_stats helper."""

    def test_returns_correct_structure(self):
        stats = _empty_cache_stats()
        assert stats == {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0,
        }

    def test_returns_new_dict_each_time(self):
        stats1 = _empty_cache_stats()
        stats2 = _empty_cache_stats()
        stats1["hits"] = 10
        assert stats2["hits"] == 0


class TestRedisCache:
    """Tests for RedisCache class."""

    @pytest.fixture
    def mock_redis_client(self):
        client = AsyncMock()
        client.get = AsyncMock()
        client.setex = AsyncMock()
        client.delete = AsyncMock()
        return client

    @pytest.fixture
    def cache(self, mock_redis_client):
        return RedisCache(redis_client=mock_redis_client, config=RedisCacheConfig())

    def test_init_default_config(self):
        cache = RedisCache()
        assert cache.config.prefix == "cache:"
        assert cache._stats == _empty_cache_stats()

    def test_init_custom_config(self):
        config = RedisCacheConfig(prefix="custom:", ttl_provider_config=600)
        cache = RedisCache(config=config)
        assert cache.config.prefix == "custom:"
        assert cache.config.ttl_provider_config == 600

    def test_get_key(self, cache):
        key = cache._get_key("provider", "openai")
        assert key == "cache:provider:openai"

    def test_get_key_custom_prefix(self, mock_redis_client):
        config = RedisCacheConfig(prefix="mycache:")
        cache = RedisCache(redis_client=mock_redis_client, config=config)
        key = cache._get_key("model", "gpt-4")
        assert key == "mycache:model:gpt-4"

    @pytest.mark.asyncio
    async def test_get_provider_config_hit(self, cache, mock_redis_client):
        config_dict = {"name": "openai", "base_url": "https://api.openai.com"}
        mock_redis_client.get = AsyncMock(
            return_value=b'{"name":"openai","base_url":"https://api.openai.com"}'
        )

        with (
            patch(
                "llm_proxy.cache.redis_cache.get_redis_client_async",
                AsyncMock(return_value=mock_redis_client),
            ),
            patch("llm_proxy.cache.redis_cache.orjson") as mock_orjson,
        ):
            mock_orjson.loads.return_value = config_dict
            with patch.object(ProviderConfig, "__init__", return_value=None):
                await cache.get_provider_config("openai")

        assert cache._stats["hits"] == 1

    @pytest.mark.asyncio
    async def test_get_provider_config_miss(self, cache, mock_redis_client):
        mock_redis_client.get = AsyncMock(return_value=None)

        with patch(
            "llm_proxy.cache.redis_cache.get_redis_client_async",
            AsyncMock(return_value=mock_redis_client),
        ):
            result = await cache.get_provider_config("nonexistent")

        assert result is None
        assert cache._stats["misses"] == 1

    @pytest.mark.asyncio
    async def test_get_provider_config_error(self, cache, mock_redis_client):
        mock_redis_client.get = AsyncMock(side_effect=ConnectionError("test"))

        with patch(
            "llm_proxy.cache.redis_cache.get_redis_client_async",
            AsyncMock(return_value=mock_redis_client),
        ):
            result = await cache.get_provider_config("openai")

        assert result is None
        assert cache._stats["errors"] == 1

    @pytest.mark.asyncio
    async def test_set_provider_config(self, cache, mock_redis_client):
        config = MagicMock()
        config.model_dump = MagicMock(return_value={"name": "openai"})

        with (
            patch(
                "llm_proxy.cache.redis_cache.get_redis_client_async",
                AsyncMock(return_value=mock_redis_client),
            ),
            patch("llm_proxy.cache.redis_cache.orjson") as mock_orjson,
        ):
            mock_orjson.dumps.return_value = b'{"name":"openai"}'
            result = await cache.set_provider_config("openai", config)

        assert result is True
        assert cache._stats["sets"] == 1
        mock_redis_client.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_provider_config_error(self, cache, mock_redis_client):
        config = MagicMock()
        config.model_dump = MagicMock(return_value={"name": "openai"})
        mock_redis_client.setex = AsyncMock(side_effect=ConnectionError("test"))

        with (
            patch(
                "llm_proxy.cache.redis_cache.get_redis_client_async",
                AsyncMock(return_value=mock_redis_client),
            ),
            patch("llm_proxy.cache.redis_cache.orjson") as mock_orjson,
        ):
            mock_orjson.dumps.return_value = b'{"name":"openai"}'
            result = await cache.set_provider_config("openai", config)

        assert result is False
        assert cache._stats["errors"] == 1

    @pytest.mark.asyncio
    async def test_invalidate_provider_config(self, cache, mock_redis_client):
        mock_redis_client.delete = AsyncMock(return_value=1)

        with patch(
            "llm_proxy.cache.redis_cache.get_redis_client_async",
            AsyncMock(return_value=mock_redis_client),
        ):
            result = await cache.invalidate_provider_config("openai")

        assert result is True
        assert cache._stats["deletes"] == 1

    @pytest.mark.asyncio
    async def test_invalidate_provider_config_error(self, cache, mock_redis_client):
        mock_redis_client.delete = AsyncMock(side_effect=ConnectionError("test"))

        with patch(
            "llm_proxy.cache.redis_cache.get_redis_client_async",
            AsyncMock(return_value=mock_redis_client),
        ):
            result = await cache.invalidate_provider_config("openai")

        assert result is False
        assert cache._stats["errors"] == 1

    @pytest.mark.asyncio
    async def test_get_model_config_miss(self, cache, mock_redis_client):
        mock_redis_client.get = AsyncMock(return_value=None)

        with patch(
            "llm_proxy.cache.redis_cache.get_redis_client_async",
            AsyncMock(return_value=mock_redis_client),
        ):
            result = await cache.get_model_config("gpt-4")

        assert result is None
        assert cache._stats["misses"] == 1

    @pytest.mark.asyncio
    async def test_set_model_config(self, cache, mock_redis_client):
        config = MagicMock()
        config.model_dump = MagicMock(return_value={"name": "gpt-4"})

        with (
            patch(
                "llm_proxy.cache.redis_cache.get_redis_client_async",
                AsyncMock(return_value=mock_redis_client),
            ),
            patch("llm_proxy.cache.redis_cache.orjson") as mock_orjson,
        ):
            mock_orjson.dumps.return_value = b'{"name":"gpt-4"}'
            result = await cache.set_model_config("gpt-4", config)

        assert result is True
        assert cache._stats["sets"] == 1

    @pytest.mark.asyncio
    async def test_invalidate_model_config(self, cache, mock_redis_client):
        mock_redis_client.delete = AsyncMock(return_value=1)

        with patch(
            "llm_proxy.cache.redis_cache.get_redis_client_async",
            AsyncMock(return_value=mock_redis_client),
        ):
            result = await cache.invalidate_model_config("gpt-4")

        assert result is True
        assert cache._stats["deletes"] == 1

    @pytest.mark.asyncio
    async def test_get_miss(self, cache, mock_redis_client):
        mock_redis_client.get = AsyncMock(return_value=None)

        with patch(
            "llm_proxy.cache.redis_cache.get_redis_client_async",
            AsyncMock(return_value=mock_redis_client),
        ):
            result = await cache.get("test_key")

        assert result is None
        assert cache._stats["misses"] == 1

    @pytest.mark.asyncio
    async def test_get_hit(self, cache, mock_redis_client):
        mock_redis_client.get = AsyncMock(return_value=b'{"data":"test"}')

        with (
            patch(
                "llm_proxy.cache.redis_cache.get_redis_client_async",
                AsyncMock(return_value=mock_redis_client),
            ),
            patch("llm_proxy.cache.redis_cache.orjson") as mock_orjson,
        ):
            mock_orjson.loads.return_value = {"data": "test"}
            result = await cache.get("test_key")

        assert result == {"data": "test"}
        assert cache._stats["hits"] == 1

    @pytest.mark.asyncio
    async def test_set(self, cache, mock_redis_client):
        with (
            patch(
                "llm_proxy.cache.redis_cache.get_redis_client_async",
                AsyncMock(return_value=mock_redis_client),
            ),
            patch("llm_proxy.cache.redis_cache.orjson") as mock_orjson,
        ):
            mock_orjson.dumps.return_value = b'{"data":"test"}'
            result = await cache.set("test_key", {"data": "test"}, ttl=300)

        assert result is True
        assert cache._stats["sets"] == 1
        mock_redis_client.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete(self, cache, mock_redis_client):
        mock_redis_client.delete = AsyncMock(return_value=1)

        with patch(
            "llm_proxy.cache.redis_cache.get_redis_client_async",
            AsyncMock(return_value=mock_redis_client),
        ):
            result = await cache.delete("test_key")

        assert result is True
        assert cache._stats["deletes"] == 1

    @pytest.mark.asyncio
    async def test_delete_not_found(self, cache, mock_redis_client):
        mock_redis_client.delete = AsyncMock(return_value=0)

        with patch(
            "llm_proxy.cache.redis_cache.get_redis_client_async",
            AsyncMock(return_value=mock_redis_client),
        ):
            result = await cache.delete("nonexistent")

        assert result is False
        assert cache._stats["deletes"] == 1

    @pytest.mark.asyncio
    async def test_invalidate_all(self, cache, mock_redis_client):
        with (
            patch(
                "llm_proxy.cache.redis_cache.get_redis_client_async",
                AsyncMock(return_value=mock_redis_client),
            ),
            patch(
                "llm_proxy.cache.redis_cache.scan_keys", AsyncMock(return_value=["key1", "key2"])
            ),
            patch("llm_proxy.cache.redis_cache.delete_keys_pipeline", AsyncMock(return_value=2)),
        ):
            result = await cache.invalidate_all()

        assert result == 2
        assert cache._stats["deletes"] == 2

    @pytest.mark.asyncio
    async def test_invalidate_all_no_keys(self, cache, mock_redis_client):
        with (
            patch(
                "llm_proxy.cache.redis_cache.get_redis_client_async",
                AsyncMock(return_value=mock_redis_client),
            ),
            patch("llm_proxy.cache.redis_cache.scan_keys", AsyncMock(return_value=[])),
        ):
            result = await cache.invalidate_all()

        assert result == 0

    @pytest.mark.asyncio
    async def test_invalidate_all_error(self, cache, mock_redis_client):
        with patch(
            "llm_proxy.cache.redis_cache.get_redis_client_async",
            AsyncMock(side_effect=ConnectionError("test")),
        ):
            result = await cache.invalidate_all()

        assert result == 0
        assert cache._stats["errors"] == 1

    @pytest.mark.asyncio
    async def test_invalidate_cached_client_on_connection_error(self, cache, mock_redis_client):
        mock_redis_client.get = AsyncMock(side_effect=ConnectionError("test"))
        cache._cached_async_client = mock_redis_client

        with patch(
            "llm_proxy.cache.redis_cache.get_redis_client_async",
            AsyncMock(return_value=mock_redis_client),
        ):
            await cache.get_provider_config("openai")

        assert cache._cached_async_client is None


class TestGetRedisCache:
    """Tests for get_redis_cache function."""

    @pytest.mark.asyncio
    async def test_creates_instance(self):
        import llm_proxy.cache.redis_cache as module

        module._redis_cache = None

        with patch("llm_proxy.cache.redis_cache.get_redis_client_async", AsyncMock()):
            cache = await get_redis_cache()
            assert cache is not None

    @pytest.mark.asyncio
    async def test_returns_same_instance(self):
        import llm_proxy.cache.redis_cache as module

        module._redis_cache = None

        with patch("llm_proxy.cache.redis_cache.get_redis_client_async", AsyncMock()):
            cache1 = await get_redis_cache()
            cache2 = await get_redis_cache()
            assert cache1 is cache2


class TestInvalidateCache:
    """Tests for invalidate_cache function."""

    @pytest.mark.asyncio
    async def test_invalidate_provider(self):
        mock_cache = AsyncMock()
        with patch("llm_proxy.cache.redis_cache._redis_cache", mock_cache):
            await invalidate_cache(provider_name="openai")
            mock_cache.invalidate_provider_config.assert_called_once_with("openai")

    @pytest.mark.asyncio
    async def test_invalidate_model(self):
        mock_cache = AsyncMock()
        with patch("llm_proxy.cache.redis_cache._redis_cache", mock_cache):
            await invalidate_cache(model_name="gpt-4")
            mock_cache.invalidate_model_config.assert_called_once_with("gpt-4")

    @pytest.mark.asyncio
    async def test_invalidate_all_when_no_args(self):
        mock_cache = AsyncMock()
        with patch("llm_proxy.cache.redis_cache._redis_cache", mock_cache):
            await invalidate_cache()
            mock_cache.invalidate_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_op_when_no_cache(self):
        with patch("llm_proxy.cache.redis_cache._redis_cache", None):
            await invalidate_cache(provider_name="openai")

    @pytest.mark.asyncio
    async def test_invalidate_both(self):
        mock_cache = AsyncMock()
        with patch("llm_proxy.cache.redis_cache._redis_cache", mock_cache):
            await invalidate_cache(provider_name="openai", model_name="gpt-4")
            mock_cache.invalidate_provider_config.assert_called_once_with("openai")
            mock_cache.invalidate_model_config.assert_called_once_with("gpt-4")
