"""Tests for admin provider OAuth endpoints and model-listing OAuth support."""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qs, urlparse

import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import llm_proxy.providers  # noqa: F401
from llm_proxy.api.dependencies import (
    get_async_session,
    require_admin_role,
    require_authenticated,
)
from llm_proxy.api.middleware.exceptions import register_exception_handlers
from llm_proxy.api.routers.config.config import router as config_router
from llm_proxy.providers.openai_chatgpt.oauth import (
    DYNAMIC_AGENT_CLIENT_ID,
    OAuthTokenPayload,
    get_pending_oauth_store,
)


@dataclass
class _FakeOAuthCred:
    provider_id: int = 1
    client_id: str = "oaiapp_issued_1"
    client_secret: str | None = None
    ext_agent_host_id: str | None = "urn:uuid:host-1"
    access_token: str = "secret_access_token_value"
    refresh_token: str | None = "secret_refresh_token_value"
    token_type: str = "Bearer"
    scope: str = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
    expires_at: datetime | None = field(
        default_factory=lambda: datetime.now(UTC) + timedelta(hours=1)
    )
    account_sub: str | None = "sub_1"
    account_email: str | None = "user@example.com"
    account_name: str | None = "Test User"
    updated_at: datetime | None = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class _FakeProviderRecord:
    id: int = 1
    name: str = "chatgpt-sub"
    type: str = "openai-chatgpt"
    api_key: str = ""
    base_url: str | None = None
    api_version: str | None = None
    timeout: float = 300.0
    rate_limit: int | None = None
    custom_headers: dict = field(default_factory=dict)
    provider_models: list = field(default_factory=list)
    enabled: bool = True
    priority: int = 0
    provider_metadata: dict = field(default_factory=dict)
    icon_url: str | None = None
    model_provider_mappings: list = field(default_factory=list)
    oauth_credential: _FakeOAuthCred | None = None


@pytest.fixture(autouse=True)
def _clear_pending_store():
    get_pending_oauth_store().clear()
    yield
    get_pending_oauth_store().clear()


@pytest.fixture
def fake_state():
    provider = _FakeProviderRecord()
    config_repo = MagicMock()
    config_repo.get_provider = AsyncMock(
        side_effect=lambda name, **_kw: provider if name == provider.name else None
    )
    config_repo.get_all_providers = AsyncMock(return_value=[provider])

    cred_repo = MagicMock()

    async def _get_by_provider_id(_pid: int, **_kw):
        return provider.oauth_credential

    async def _upsert_for_provider(provider_id: int, **kwargs):
        cred = _FakeOAuthCred(
            provider_id=provider_id,
            client_id=kwargs["client_id"],
            client_secret=kwargs.get("client_secret"),
            ext_agent_host_id=kwargs.get("ext_agent_host_id"),
            access_token=kwargs["access_token"],
            refresh_token=kwargs.get("refresh_token"),
            token_type=kwargs.get("token_type", "Bearer"),
            scope=kwargs["scope"],
            expires_at=kwargs.get("expires_at"),
            account_sub=kwargs.get("account_sub"),
            account_email=kwargs.get("account_email"),
            account_name=kwargs.get("account_name"),
        )
        provider.oauth_credential = cred
        return cred

    async def _delete_by_provider_id(_pid: int):
        existed = provider.oauth_credential is not None
        provider.oauth_credential = None
        return existed

    cred_repo.get_by_provider_id = AsyncMock(side_effect=_get_by_provider_id)
    cred_repo.upsert_for_provider = AsyncMock(side_effect=_upsert_for_provider)
    cred_repo.delete_by_provider_id = AsyncMock(side_effect=_delete_by_provider_id)
    return provider, config_repo, cred_repo


@pytest.fixture
def client(fake_state):
    app = FastAPI()
    register_exception_handlers(app)
    mock_config_manager = MagicMock()
    mock_config_manager.reload = AsyncMock()
    app.state.config_manager = mock_config_manager
    app.state.http_client = MagicMock()
    app.state.http_client.client = MagicMock()
    app.dependency_overrides[require_authenticated] = lambda: None
    app.dependency_overrides[require_admin_role] = lambda: None
    mock_session = AsyncMock()
    mock_session.commit = AsyncMock()
    app.dependency_overrides[get_async_session] = lambda: mock_session
    app.include_router(config_router)
    with TestClient(app, raise_server_exceptions=False) as tc:
        yield tc


class TestProviderOAuthRouter:
    def test_start_and_callback_flow_persists_status_and_never_leaks_secrets(
        self, client, fake_state
    ):
        provider, config_repo, cred_repo = fake_state

        with (
            patch(
                "llm_proxy.api.routers.config.provider_oauth.get_config_repository",
                return_value=config_repo,
            ),
            patch(
                "llm_proxy.api.routers.config.provider_oauth.ProviderOAuthCredentialRepository",
                return_value=cred_repo,
            ),
            patch(
                "llm_proxy.api.routers.config.provider_oauth.get_or_create_host_id",
                AsyncMock(return_value="urn:uuid:host-1"),
            ),
        ):
            start_res = client.post(
                f"/api/config/providers/{provider.name}/oauth/start",
                json={"redirect_uri": "http://127.0.0.1:1455/auth/callback"},
            )
            assert start_res.status_code == 200, start_res.text
            start_data = start_res.json()
            q = parse_qs(urlparse(start_data["authorization_url"]).query)
            assert q["client_id"] == [DYNAMIC_AGENT_CLIENT_ID]
            assert q["ext_agent_host_id"] == ["urn:uuid:host-1"]

            tx = get_pending_oauth_store()._memory[start_data["state"]]
            id_token = jwt.encode(
                {
                    "sub": "sub_1",
                    "email": "user@example.com",
                    "name": "Test User",
                    "nonce": tx.nonce,
                },
                "test-hmac-secret-at-least-32-bytes",
                algorithm="HS256",
            )
            token_payload = OAuthTokenPayload(
                access_token="secret_access_token_value",
                refresh_token="secret_refresh_token_value",
                id_token=id_token,
                token_type="Bearer",
                scope=(
                    "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
                ),
                expires_in=3600,
            )

            with patch(
                "llm_proxy.api.routers.config.provider_oauth.exchange_authorization_code",
                AsyncMock(return_value=token_payload),
            ):
                cb_res = client.get(
                    "/api/config/providers/oauth/callback",
                    params={
                        "code": "auth_code_1",
                        "state": start_data["state"],
                        "client_id": "oaiapp_issued_1",
                    },
                )
            assert cb_res.status_code == 200
            assert "llm-proxy-provider-oauth" in cb_res.text
            assert "user@example.com" in cb_res.text
            assert "secret_access_token_value" not in cb_res.text
            assert "secret_refresh_token_value" not in cb_res.text

            # Reusing the same single-use state fails.
            replay_res = client.get(
                "/api/config/providers/oauth/callback",
                params={"code": "auth_code_1", "state": start_data["state"]},
            )
            assert replay_res.status_code == 400

        with patch(
            "llm_proxy.api.routers.config.providers.get_config_repository",
            return_value=config_repo,
        ):
            list_res = client.get("/api/config/providers")
        assert list_res.status_code == 200
        assert "secret_access_token_value" not in list_res.text
        assert "secret_refresh_token_value" not in list_res.text
        row = list_res.json()[0]
        assert row["oauth_status"]["connected"] is True
        assert row["oauth_status"]["has_plan_usage_scope"] is True
        assert row["oauth_status"]["account_email"] == "user@example.com"

    def test_manual_complete_and_disconnect(self, client, fake_state):
        provider, config_repo, cred_repo = fake_state

        with (
            patch(
                "llm_proxy.api.routers.config.provider_oauth.get_config_repository",
                return_value=config_repo,
            ),
            patch(
                "llm_proxy.api.routers.config.provider_oauth.ProviderOAuthCredentialRepository",
                return_value=cred_repo,
            ),
            patch(
                "llm_proxy.api.routers.config.provider_oauth.get_or_create_host_id",
                AsyncMock(return_value="urn:uuid:host-1"),
            ),
        ):
            start_res = client.post(
                f"/api/config/providers/{provider.name}/oauth/start",
                json={"redirect_uri": "http://127.0.0.1:1455/auth/callback"},
            )
            state = start_res.json()["state"]
            token_payload = OAuthTokenPayload(
                access_token="secret_at_manual",
                refresh_token="secret_rt_manual",
                id_token=None,
                token_type="Bearer",
                scope="openid offline_access chatgpt.tokens.use.direct",
                expires_in=3600,
            )
            with patch(
                "llm_proxy.api.routers.config.provider_oauth.exchange_authorization_code",
                AsyncMock(return_value=token_payload),
            ):
                complete_res = client.post(
                    f"/api/config/providers/{provider.name}/oauth/complete",
                    json={
                        "callback_url": (
                            f"http://127.0.0.1:1455/auth/callback?code=c_1&state={state}&client_id=oaiapp_m"
                        )
                    },
                )
            assert complete_res.status_code == 200, complete_res.text
            assert complete_res.json()["connected"] is True
            assert complete_res.json()["client_id"] == "oaiapp_m"

            del_res = client.delete(f"/api/config/providers/{provider.name}/oauth")
            assert del_res.status_code == 200
            assert provider.oauth_credential is None

    def test_dynamic_registration_requires_issued_client_id_on_complete(self, client, fake_state):
        provider, config_repo, cred_repo = fake_state

        with (
            patch(
                "llm_proxy.api.routers.config.provider_oauth.get_config_repository",
                return_value=config_repo,
            ),
            patch(
                "llm_proxy.api.routers.config.provider_oauth.ProviderOAuthCredentialRepository",
                return_value=cred_repo,
            ),
            patch(
                "llm_proxy.api.routers.config.provider_oauth.get_or_create_host_id",
                AsyncMock(return_value="urn:uuid:host-1"),
            ),
        ):
            start_res = client.post(
                f"/api/config/providers/{provider.name}/oauth/start",
                json={"redirect_uri": "http://127.0.0.1:1455/auth/callback"},
            )
            state = start_res.json()["state"]
            complete_res = client.post(
                f"/api/config/providers/{provider.name}/oauth/complete",
                json={"code": "c_1", "state": state},
            )
            assert complete_res.status_code == 400
            assert complete_res.json()["error"]["code"] == "oauth_missing_client_id"

    def test_start_rejects_non_oauth_provider_type(self, client, fake_state):
        provider, config_repo, _ = fake_state
        provider.type = "openai"
        with patch(
            "llm_proxy.api.routers.config.provider_oauth.get_config_repository",
            return_value=config_repo,
        ):
            res = client.post(f"/api/config/providers/{provider.name}/oauth/start", json={})
        assert res.status_code == 400

    def test_provider_models_endpoint_allows_empty_api_key_for_openai_chatgpt(
        self, client, fake_state
    ):
        provider, config_repo, _ = fake_state
        provider.api_key = ""

        fake_adapter = MagicMock()
        fake_adapter.__class__ = type("OpenAIChatGPTAdapter", (), {})
        mock_models = AsyncMock(return_value=[])

        with (
            patch(
                "llm_proxy.api.routers.config.provider_models.get_config_repository",
                return_value=config_repo,
            ),
            patch(
                "llm_proxy.api.routers.config.provider_models.get_adapter",
            ) as mock_get_adapter,
        ):
            from llm_proxy.providers.openai_chatgpt.adapter import OpenAIChatGPTAdapter

            adapter_instance = OpenAIChatGPTAdapter(provider_name=provider.name)
            adapter_instance.list_models = mock_models
            mock_get_adapter.return_value = adapter_instance

            res = client.get(f"/api/config/providers/{provider.name}/models")

        assert res.status_code == 200, res.text
        assert mock_get_adapter.call_args.kwargs["provider_name"] == provider.name
