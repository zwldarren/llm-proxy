"""Cross-worker invalidation of per-user tracing registries.

A personal tracing config change is persisted on one worker; peer workers cache
the user's registry in-process, so without a shared signal they keep exporting
to the old backend until restart. These tests cover the Redis version-token
mechanism and its local-only fallback when Redis is disabled.
"""

from unittest.mock import AsyncMock

from llm_proxy.observability.user_tracing import UserTracingManager


class _FakeRedis:
    """Dict-backed stand-in for the raw redis client (reads return bytes)."""

    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}

    async def get(self, key: str) -> bytes | None:
        return self.data.get(key)

    async def set(self, key: str, value: str) -> None:
        self.data[key] = value.encode("utf-8")


async def test_publish_sets_shared_token_and_invalidates_locally():
    manager = UserTracingManager()
    redis = _FakeRedis()
    manager.set_redis_client(redis)
    manager.invalidate = AsyncMock()

    await manager.publish_invalidation(7)

    assert redis.data[manager._version_key(7)]
    manager.invalidate.assert_awaited_once_with(7)


async def test_peer_drops_registry_when_version_changes():
    redis = _FakeRedis()
    manager = UserTracingManager()
    manager.set_redis_client(redis)
    manager._registries[7] = object()
    manager._versions[7] = ""

    redis.data[manager._version_key(7)] = b"peer-token"
    await manager._refresh_if_version_changed(7)

    assert 7 not in manager._registries
    assert manager._versions[7] == "peer-token"


async def test_two_workers_converge_after_a_publish():
    redis = _FakeRedis()
    worker_a = UserTracingManager()
    worker_a.set_redis_client(redis)
    worker_b = UserTracingManager()
    worker_b.set_redis_client(redis)

    # Worker B has already cached the user's old registry.
    worker_b._registries[7] = object()
    worker_b._versions[7] = ""

    await worker_a.publish_invalidation(7)
    await worker_b._refresh_if_version_changed(7)

    assert 7 not in worker_b._registries


async def test_unchanged_version_keeps_registry():
    redis = _FakeRedis()
    manager = UserTracingManager()
    manager.set_redis_client(redis)
    cached = object()
    manager._registries[7] = cached
    manager._versions[7] = ""

    # No token yet: the absent key means "never mutated", not "changed".
    await manager._refresh_if_version_changed(7)

    assert manager._registries[7] is cached


async def test_without_redis_invalidation_is_local_only():
    manager = UserTracingManager()
    manager.invalidate = AsyncMock()

    await manager.publish_invalidation(7)

    manager.invalidate.assert_awaited_once_with(7)
    assert manager._versions == {}


async def test_version_read_failure_keeps_warm_registry():
    class _BrokenRedis:
        async def get(self, key: str) -> bytes:
            raise RuntimeError("redis down")

    manager = UserTracingManager()
    manager.set_redis_client(_BrokenRedis())
    cached = object()
    manager._registries[7] = cached

    await manager._refresh_if_version_changed(7)

    assert manager._registries[7] is cached


async def test_publish_retries_once_on_a_transient_redis_error():
    class _FlakyRedis:
        def __init__(self) -> None:
            self.calls = 0
            self.data: dict[str, bytes] = {}

        async def set(self, key: str, value: str) -> None:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient")
            self.data[key] = value.encode("utf-8")

    redis = _FlakyRedis()
    manager = UserTracingManager()
    manager.set_redis_client(redis)
    manager.invalidate = AsyncMock()

    await manager.publish_invalidation(7)

    assert redis.calls == 2
    assert manager._versions[7] == redis.data[manager._version_key(7)].decode("utf-8")
    manager.invalidate.assert_awaited_once_with(7)


async def test_publish_failure_still_invalidates_locally():
    class _BrokenRedis:
        async def set(self, key: str, value: str) -> None:
            raise RuntimeError("redis down")

    manager = UserTracingManager()
    manager.set_redis_client(_BrokenRedis())
    manager.invalidate = AsyncMock()

    await manager.publish_invalidation(7)

    # Both attempts failed: no shared token recorded, but the local registry is
    # still dropped.
    assert manager._versions == {}
    manager.invalidate.assert_awaited_once_with(7)


class _RecordingHandler:
    """Fake handler that records which teardown verb was called."""

    name = "fake"

    def __init__(self, called: list[str]):
        self._called = called

    async def release(self) -> None:
        self._called.append("release")

    async def shutdown(self) -> None:
        self._called.append("shutdown")


async def test_invalidate_releases_handlers_without_shutting_them_down():
    """Dropping a registry must not tear down in-flight-capable handlers.

    A config change (or a peer worker's change) drops the cached registry while
    requests that already captured it may still be running. Shutting the handler
    down silently lost their traces; ``release`` flushes and keeps it usable.
    """
    called: list[str] = []

    manager = UserTracingManager()
    manager._registries[7] = object()
    manager._user_handlers[7] = [_RecordingHandler(called)]  # type: ignore[list-item]

    await manager.invalidate(7)

    assert called == ["release"]
    assert 7 not in manager._registries
    assert 7 not in manager._user_handlers


async def test_shutdown_all_still_fully_shuts_handlers_down():
    """Process teardown must use ``shutdown`` so buffered data is flushed."""
    called: list[str] = []

    manager = UserTracingManager()
    manager._registries[7] = object()
    manager._user_handlers[7] = [_RecordingHandler(called)]  # type: ignore[list-item]

    await manager.shutdown_all()

    assert called == ["shutdown"]
