"""Tests for OpenAIChatGPTOAuthManager caching, refresh rotation, and concurrency."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from llm_proxy.core.exceptions import ProviderError
from llm_proxy.providers.openai_chatgpt.oauth import OAuthTokenPayload
from llm_proxy.providers.openai_chatgpt.token_manager import OpenAIChatGPTOAuthManager


@dataclass
class _FakeCred:
    provider_id: int = 1
    client_id: str = "oaiapp_123"
    client_secret: str | None = None
    access_token: str = "at_initial"
    refresh_token: str | None = "rt_initial"
    scope: str = "openid offline_access chatgpt.tokens.use.direct"
    expires_at: datetime | None = None


class _FakeRedis:
    """Minimal async Redis surface the refresh lock uses."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.deleted: list[str] = []

    async def set(self, key: str, value: str, *, nx: bool = False, ex: int | None = None):
        if nx and key in self.store:
            return None
        self.store[key] = value
        return True

    async def get(self, key: str):
        return self.store.get(key)

    async def delete(self, key: str):
        self.deleted.append(key)
        self.store.pop(key, None)
        return 1


class _FakeRepo:
    def __init__(self, cred: _FakeCred | None):
        self.cred = cred
        self.reads = 0
        self.updates: list[dict] = []

    async def get_by_provider_name(self, _name: str, *, decrypt: bool = True):
        self.reads += 1
        return self.cred

    async def update_tokens(self, _provider_id: int, **kwargs):
        self.updates.append(kwargs)
        if self.cred is not None:
            self.cred.access_token = kwargs["access_token"]
            if kwargs.get("refresh_token") is not None:
                self.cred.refresh_token = kwargs["refresh_token"]
            if kwargs.get("scope") is not None:
                self.cred.scope = kwargs["scope"]
            self.cred.expires_at = kwargs["expires_at"]
        return self.cred


def _patch_db(repo: _FakeRepo):
    session = AsyncMock()
    session.commit = AsyncMock()

    @asynccontextmanager
    async def _ctx():
        yield session

    return (
        patch(
            "llm_proxy.providers.openai_chatgpt.token_manager.get_async_session_context",
            _ctx,
        ),
        patch(
            "llm_proxy.providers.openai_chatgpt.token_manager.ProviderOAuthCredentialRepository",
            return_value=repo,
        ),
    )


class TestOpenAIChatGPTOAuthManager:
    @pytest.mark.asyncio
    async def test_fast_path_cache_hit_avoids_extra_db_reads(self):
        cred = _FakeCred(expires_at=datetime.now(UTC) + timedelta(minutes=45))
        repo = _FakeRepo(cred)
        manager = OpenAIChatGPTOAuthManager()

        ctx_patch, repo_patch = _patch_db(repo)
        with ctx_patch, repo_patch:
            t1 = await manager.get_access_token("my-chatgpt")
            t2 = await manager.get_access_token("my-chatgpt")

        assert t1 == "at_initial"
        assert t2 == "at_initial"
        assert repo.reads == 1

    @pytest.mark.asyncio
    async def test_expiring_token_refreshes_once_under_concurrency(self):
        cred = _FakeCred(expires_at=datetime.now(UTC) + timedelta(seconds=30))
        repo = _FakeRepo(cred)
        manager = OpenAIChatGPTOAuthManager()

        refresh_calls = 0

        async def _fake_refresh(**_kwargs):
            nonlocal refresh_calls
            refresh_calls += 1
            await asyncio.sleep(0.02)
            return OAuthTokenPayload(
                access_token="at_rotated",
                refresh_token="rt_rotated",
                id_token=None,
                token_type="Bearer",
                scope="openid offline_access chatgpt.tokens.use.direct",
                expires_in=3600,
            )

        ctx_patch, repo_patch = _patch_db(repo)
        with (
            ctx_patch,
            repo_patch,
            patch(
                "llm_proxy.providers.openai_chatgpt.token_manager.refresh_access_token",
                side_effect=_fake_refresh,
            ),
        ):
            results = await asyncio.gather(
                manager.get_access_token("my-chatgpt"),
                manager.get_access_token("my-chatgpt"),
                manager.get_access_token("my-chatgpt"),
            )

        assert results == ["at_rotated", "at_rotated", "at_rotated"]
        assert refresh_calls == 1
        assert len(repo.updates) == 1
        assert repo.updates[0]["refresh_token"] == "rt_rotated"

    @pytest.mark.asyncio
    async def test_force_refresh_skips_cache_and_reuses_peer_updated_db_token(self):
        cred = _FakeCred(
            access_token="at_peer_fresh",
            expires_at=datetime.now(UTC) + timedelta(minutes=50),
        )
        repo = _FakeRepo(cred)
        manager = OpenAIChatGPTOAuthManager()

        with _patch_db(repo)[0], _patch_db(repo)[1]:
            token = await manager.get_access_token(
                "my-chatgpt",
                force_refresh=True,
                failed_token="at_old_failed",
            )
        assert token == "at_peer_fresh"

    @pytest.mark.asyncio
    async def test_missing_credentials_and_missing_scope_raise_401(self):
        manager = OpenAIChatGPTOAuthManager()

        missing_repo = _FakeRepo(None)
        ctx_p, repo_p = _patch_db(missing_repo)
        with ctx_p, repo_p, pytest.raises(ProviderError) as exc_info:
            await manager.get_access_token("unlinked")
        assert exc_info.value.status_code == 401
        assert exc_info.value.code == "oauth_not_connected"

        bad_scope_repo = _FakeRepo(_FakeCred(scope="openid profile email"))
        ctx_p2, repo_p2 = _patch_db(bad_scope_repo)
        with ctx_p2, repo_p2, pytest.raises(ProviderError) as exc_info2:
            await manager.get_access_token("bad-scope")
        assert exc_info2.value.status_code == 401
        assert exc_info2.value.code == "oauth_scope_missing"

    @pytest.mark.asyncio
    async def test_installed_redis_client_guards_request_path_refreshes(self):
        """An adapter passes no client; the lifespan-installed one must be used.

        Without it, expiry-driven refreshes (the only kind the request path does)
        are serialized per process, so two workers rotate one refresh token at
        once and the loser gets `invalid_grant`.
        """
        cred = _FakeCred(expires_at=datetime.now(UTC) + timedelta(seconds=30))
        repo = _FakeRepo(cred)
        manager = OpenAIChatGPTOAuthManager()
        fake_redis = _FakeRedis()
        manager.set_redis_client(fake_redis)

        async def _fake_refresh(**_kwargs):
            return OAuthTokenPayload(
                access_token="at_rotated",
                refresh_token="rt_rotated",
                id_token=None,
                token_type="Bearer",
                scope="openid offline_access chatgpt.tokens.use.direct",
                expires_in=3600,
            )

        ctx_p, repo_p = _patch_db(repo)
        with (
            ctx_p,
            repo_p,
            patch(
                "llm_proxy.providers.openai_chatgpt.token_manager.refresh_access_token",
                side_effect=_fake_refresh,
            ),
        ):
            assert await manager.get_access_token("my-chatgpt") == "at_rotated"

        assert "llm_proxy:provider_oauth_refresh:my-chatgpt" in fake_redis.deleted
        assert fake_redis.store == {}
