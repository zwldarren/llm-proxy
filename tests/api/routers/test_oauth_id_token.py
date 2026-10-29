"""OIDC completion and ID-token validation tests."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI
from fastapi.testclient import TestClient
from jwt.algorithms import RSAAlgorithm

from llm_proxy.api.middleware.exceptions import register_exception_handlers
from llm_proxy.api.middleware.rate_limiting import RateLimitManager
from llm_proxy.api.routers.oauth import _verify_id_token, public_router
from llm_proxy.config.types.auth import ProxyAuthConfig
from llm_proxy.config.types.console_auth import OidcProviderSettings
from llm_proxy.core.exceptions import AuthenticationFailedError
from llm_proxy.database import get_async_session
from llm_proxy.security.jwt import JWTManager

TEST_SECRET = "test-secret-at-least-32-characters-long-aaaaaa"
ISSUER = "https://issuer.example"
CLIENT_ID = "client-id"
KID = "key-1"


@pytest.fixture
def signing_key() -> tuple[rsa.RSAPrivateKey, dict]:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = RSAAlgorithm.to_jwk(private_key.public_key(), as_dict=True)
    jwk["kid"] = KID
    return private_key, {"keys": [jwk]}


def _provider() -> OidcProviderSettings:
    return OidcProviderSettings(id="test", name="Test", issuer=ISSUER, client_id=CLIENT_ID)


def _id_token(private_key: rsa.RSAPrivateKey, *, nonce: str, audience: str = CLIENT_ID) -> str:
    now = datetime.now(UTC)
    claims = {
        "iss": ISSUER,
        "aud": audience,
        "sub": "subject-1",
        "email": "user@example.com",
        "email_verified": True,
        "nonce": nonce,
        "iat": now,
        "exp": now + timedelta(minutes=5),
    }
    return jwt.encode(claims, private_key, algorithm="RS256", headers={"kid": KID})


@pytest.mark.asyncio
async def test_valid_id_token_claims_are_returned(signing_key):
    private_key, jwks = signing_key
    token = _id_token(private_key, nonce="nonce-1")

    with patch("llm_proxy.api.routers.oauth._fetch_json", new=AsyncMock(return_value=jwks)):
        claims = await _verify_id_token(
            _provider(),
            {"jwks_uri": f"{ISSUER}/jwks"},
            {"id_token": token},
            nonce="nonce-1",
        )

    assert claims is not None
    assert claims["sub"] == "subject-1"
    assert claims["email"] == "user@example.com"


@pytest.mark.asyncio
async def test_nonce_mismatch_is_rejected(signing_key):
    private_key, jwks = signing_key
    token = _id_token(private_key, nonce="nonce-1")

    with (
        patch("llm_proxy.api.routers.oauth._fetch_json", new=AsyncMock(return_value=jwks)),
        pytest.raises(AuthenticationFailedError) as exc_info,
    ):
        await _verify_id_token(
            _provider(),
            {"jwks_uri": f"{ISSUER}/jwks"},
            {"id_token": token},
            nonce="different",
        )

    assert exc_info.value.code == "oidc_id_token_invalid"


@pytest.mark.asyncio
async def test_audience_mismatch_is_rejected(signing_key):
    private_key, jwks = signing_key
    token = _id_token(private_key, nonce="nonce-1", audience="someone-else")

    with (
        patch("llm_proxy.api.routers.oauth._fetch_json", new=AsyncMock(return_value=jwks)),
        pytest.raises(AuthenticationFailedError) as exc_info,
    ):
        await _verify_id_token(
            _provider(),
            {"jwks_uri": f"{ISSUER}/jwks"},
            {"id_token": token},
            nonce="nonce-1",
        )

    assert exc_info.value.code == "oidc_id_token_invalid"


def test_new_oauth_account_must_choose_a_username():
    auth_config = ProxyAuthConfig(jwt_secret=TEST_SECRET)
    config = MagicMock()
    config.server_params.auth = auth_config
    config_manager = MagicMock()
    config_manager.get_config = AsyncMock(return_value=config)

    user = MagicMock()
    user.id = 1
    user.username = "invited"
    user.is_active = True

    completion = JWTManager(auth_config).create_scoped_token(
        "oauth_complete",
        {"uid": 1, "amr": "oidc", "new_account": True},
        ttl_seconds=60,
    )

    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(public_router)
    app.state.config_manager = config_manager
    app.dependency_overrides[get_async_session] = lambda: AsyncMock()

    user_repo = AsyncMock()
    user_repo.get_by_id = AsyncMock(return_value=user)
    with (
        patch("llm_proxy.api.routers.oauth.UserRepository", return_value=user_repo),
        patch.object(RateLimitManager, "check_rate_limit", new=AsyncMock(return_value=(False, {}))),
        TestClient(app, raise_server_exceptions=False) as client,
    ):
        response = client.post(
            "/api/auth/oauth/complete",
            json={"completion_token": completion},
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "username_required"
