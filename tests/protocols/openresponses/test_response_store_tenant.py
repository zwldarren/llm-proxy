"""Tests for tenant-scoped ResponseStore."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_proxy.protocols.openresponses.store import ResponseStore


@pytest.fixture
def mock_redis():
    """Mock Redis client with an in-memory store."""
    store: dict[bytes, bytes] = {}

    redis = MagicMock()

    async def _setex(key, ttl, value):
        store[key.encode() if isinstance(key, str) else key] = value

    async def _get(key):
        return store.get(key.encode() if isinstance(key, str) else key)

    async def _delete(key):
        encoded = key.encode() if isinstance(key, str) else key
        return 1 if store.pop(encoded, None) is not None else 0

    redis.setex = AsyncMock(side_effect=_setex)
    redis.get = AsyncMock(side_effect=_get)
    redis.delete = AsyncMock(side_effect=_delete)
    return redis


@pytest.fixture
def response_store(mock_redis):
    """ResponseStore fixture."""
    return ResponseStore(redis_client=mock_redis, ttl=86400)


@pytest.mark.asyncio
async def test_store_scopes_by_api_key_name(response_store, mock_redis) -> None:
    """Responses with the same ID but different API keys do not collide."""
    await response_store.store("key-a", "resp_1", {"output": "a"})
    await response_store.store("key-b", "resp_1", {"output": "b"})

    assert mock_redis.setex.await_count == 2
    calls = [mock_call.args[0] for mock_call in mock_redis.setex.await_args_list]
    assert "openresponses:response:key-a|resp_1" in calls
    assert "openresponses:response:key-b|resp_1" in calls


@pytest.mark.asyncio
async def test_store_with_delimiter_in_api_key(response_store, mock_redis) -> None:
    """API key containing the Redis key delimiter ':' must not cause key collision."""
    await response_store.store("key-a:malicious", "resp_1", {"output": "a"})
    # Verify that the colon in the key doesn't allow access to another tenant's data
    other = await response_store.retrieve("key-a", "malicious:resp_1")
    assert other is None


@pytest.mark.asyncio
async def test_retrieve_only_returns_own_response(response_store) -> None:
    """A tenant cannot retrieve another tenant's stored response."""
    await response_store.store("key-a", "resp_1", {"output": "a"})

    own = await response_store.retrieve("key-a", "resp_1")
    assert own == {"output": "a"}

    other = await response_store.retrieve("key-b", "resp_1")
    assert other is None


@pytest.mark.asyncio
async def test_delete_only_removes_own_response(response_store, mock_redis) -> None:
    """A tenant cannot delete another tenant's stored response."""
    await response_store.store("key-a", "resp_1", {"output": "a"})
    await response_store.store("key-b", "resp_1", {"output": "b"})

    deleted = await response_store.delete("key-a", "resp_1")
    assert deleted is True

    remaining = await response_store.retrieve("key-b", "resp_1")
    assert remaining == {"output": "b"}
