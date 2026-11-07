"""The key gate's hash paths: digest lookup and the legacy-hash upgrade.

A migrated key's stored value *is* the digest of the presented string, so the
gate resolves it by index. Only rows written before the migration still hold a
bcrypt hash, and they are moved onto the digest path the first time their
plaintext is verified — because that is the only moment the digest can be
derived.
"""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from llm_proxy.api import gate as gate_module
from llm_proxy.api.gate import verify_api_key
from llm_proxy.api.middleware.api_key_cache import CachedApiKey, get_api_key_cache
from llm_proxy.core.budget import BudgetEnvelope
from llm_proxy.security.passwords import hash_api_key, is_bcrypt_hash

#: Bcrypt-shaped stored hash: only the prefix and length matter here, because
#: the comparison itself is patched.
LEGACY_KEY_HASH = "$2b$12$" + "a" * 53

PLAIN_KEY = "sk-secret"


def _cached_key(**overrides: Any) -> CachedApiKey:
    defaults: dict[str, Any] = {
        "name": "k",
        "key_hash": hash_api_key(PLAIN_KEY),
        "is_active": True,
        "allowed_models": None,
        "allowed_mcp_servers": None,
        "user_id": 1,
        "user_allowed_models": None,
        "user_is_active": True,
        "expires_at": None,
        "budget": BudgetEnvelope(),
        "user_budget": BudgetEnvelope(),
    }
    defaults.update(overrides)
    return CachedApiKey(**defaults)


@pytest.fixture(autouse=True)
def _reset_cache():
    get_api_key_cache().invalidate()
    yield
    get_api_key_cache().invalidate()


class _FakeSession:
    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False


@pytest.mark.asyncio
async def test_digest_key_resolves_without_a_hash_comparison():
    """A migrated key costs one indexed lookup, not a comparison per key."""
    record = _cached_key()

    with (
        patch.object(gate_module, "get_cached_api_keys", new=AsyncMock(return_value=[record])),
        patch.object(gate_module, "key_matches_hash") as compare,
        patch.object(gate_module, "_verify_session_api_key", new=AsyncMock(return_value=None)),
    ):
        auth = await verify_api_key(PLAIN_KEY)

    assert auth is not None
    assert auth.principal_id == "k"
    compare.assert_not_called()


@pytest.mark.asyncio
async def test_stale_digest_entry_is_revalidated_by_index():
    """A lapsed verified entry is trusted again via the digest index."""
    record = _cached_key()
    digest = hash_api_key(PLAIN_KEY)
    cache = get_api_key_cache()
    cache.set_verified_key(digest, "k", None, key_hash=digest)
    cache.get_verified_key(digest).verified_at -= cache.ttl + 1

    with (
        patch.object(gate_module, "get_cached_api_keys", new=AsyncMock(return_value=[record])),
        patch.object(gate_module, "key_matches_hash") as compare,
        patch.object(gate_module, "_verify_session_api_key", new=AsyncMock(return_value=None)),
    ):
        auth = await verify_api_key(PLAIN_KEY)

    assert auth is not None
    compare.assert_not_called()


@pytest.mark.asyncio
async def test_legacy_bcrypt_key_is_upgraded_to_its_digest():
    """A legacy row is verified once by bcrypt, then stored as a digest."""
    record = _cached_key(key_hash=LEGACY_KEY_HASH)
    upgrades: list[tuple[str, str, str]] = []

    class _Repo:
        def __init__(self, session: Any) -> None:
            pass

        async def set_key_hash(self, name: str, key_hash: str, *, expected_hash: str) -> bool:
            upgrades.append((name, key_hash, expected_hash))
            return True

    with (
        patch.object(gate_module, "get_cached_api_keys", new=AsyncMock(return_value=[record])),
        patch.object(gate_module, "key_matches_hash", return_value=True),
        patch.object(gate_module, "_verify_session_api_key", new=AsyncMock(return_value=None)),
        patch.object(gate_module, "get_async_session_context", _FakeSession),
        patch("llm_proxy.database.ApiKeyRepository", _Repo),
    ):
        auth = await verify_api_key(PLAIN_KEY)

    assert auth is not None
    assert upgrades == [("k", hash_api_key(PLAIN_KEY), LEGACY_KEY_HASH)]
    # The snapshot record is updated in place, so later requests hit the index.
    assert record.key_hash == hash_api_key(PLAIN_KEY)
    assert not is_bcrypt_hash(record.key_hash)


@pytest.mark.asyncio
async def test_stale_upgrade_does_not_mutate_the_snapshot():
    """A compare-and-set miss leaves a concurrently rotated row untouched."""
    record = _cached_key(key_hash=LEGACY_KEY_HASH)

    class _Repo:
        def __init__(self, session: Any) -> None:
            pass

        async def set_key_hash(self, name: str, key_hash: str, *, expected_hash: str) -> bool:
            return False

    with (
        patch.object(gate_module, "get_cached_api_keys", new=AsyncMock(return_value=[record])),
        patch.object(gate_module, "key_matches_hash", return_value=True),
        patch.object(gate_module, "_verify_session_api_key", new=AsyncMock(return_value=None)),
        patch.object(gate_module, "get_async_session_context", _FakeSession),
        patch("llm_proxy.database.ApiKeyRepository", _Repo),
    ):
        auth = await verify_api_key(PLAIN_KEY)

    assert auth is not None
    assert record.key_hash == LEGACY_KEY_HASH


@pytest.mark.asyncio
async def test_failed_upgrade_still_authenticates(caplog):
    """A failed digest write must not fail the request it was upgrading."""
    record = _cached_key(key_hash=LEGACY_KEY_HASH)

    def _broken_session() -> Any:
        raise RuntimeError("database is unavailable")

    with (
        caplog.at_level("WARNING"),
        patch.object(gate_module, "get_cached_api_keys", new=AsyncMock(return_value=[record])),
        patch.object(gate_module, "key_matches_hash", return_value=True),
        patch.object(gate_module, "_verify_session_api_key", new=AsyncMock(return_value=None)),
        patch.object(gate_module, "get_async_session_context", _broken_session),
    ):
        auth = await verify_api_key(PLAIN_KEY)

    assert auth is not None
    assert record.key_hash == LEGACY_KEY_HASH
    assert "Could not upgrade the stored hash" in caplog.text
