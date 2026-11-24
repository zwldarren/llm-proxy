"""Tests for Sign in with ChatGPT OAuth helpers and pending transaction store."""

import hashlib
import time
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import parse_qs, urlparse

import jwt
import pytest

from llm_proxy.core.exceptions import ValidationError
from llm_proxy.providers.openai_chatgpt.oauth import (
    DEFAULT_AGENT_NAME_HINT,
    DYNAMIC_AGENT_CLIENT_ID,
    OPENAI_API_RESOURCE,
    OPENAI_AUTHORIZE_URL,
    OPENAI_TOKEN_URL,
    PLAN_USAGE_SCOPE,
    PendingOAuthStore,
    PendingOAuthTransaction,
    build_authorization_url,
    exchange_authorization_code,
    extract_id_token_claims,
    generate_pkce_pair,
    parse_callback_url,
    refresh_access_token,
    validate_oauth_redirect_uri,
)
from llm_proxy.security.base64url import b64url_encode


def _mock_http_response(status_code: int, payload: dict):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json = MagicMock(return_value=payload)
    resp.text = ""
    return resp


class TestPkceAndUrls:
    def test_generate_pkce_pair_s256(self):
        verifier, challenge = generate_pkce_pair()
        assert len(verifier) >= 43
        expected = b64url_encode(hashlib.sha256(verifier.encode("ascii")).digest())
        assert challenge == expected

    def test_validate_redirect_uri_allows_https_and_loopback(self):
        assert (
            validate_oauth_redirect_uri(
                "https://proxy.example.com/api/config/providers/oauth/callback"
            )
            == "https://proxy.example.com/api/config/providers/oauth/callback"
        )
        assert (
            validate_oauth_redirect_uri("http://127.0.0.1:1455/auth/callback")
            == "http://127.0.0.1:1455/auth/callback"
        )
        assert (
            validate_oauth_redirect_uri("http://localhost:5173/callback")
            == "http://localhost:5173/callback"
        )

    def test_validate_redirect_uri_rejects_remote_http_and_invalid(self):
        with pytest.raises(ValidationError):
            validate_oauth_redirect_uri("http://proxy.example.com/callback")
        with pytest.raises(ValidationError):
            validate_oauth_redirect_uri("")
        with pytest.raises(ValidationError):
            validate_oauth_redirect_uri("https://example.com/has space")

    def test_build_authorization_url_dynamic_vs_issued_client(self):
        dynamic_url = build_authorization_url(
            client_id=DYNAMIC_AGENT_CLIENT_ID,
            redirect_uri="http://127.0.0.1:1455/auth/callback",
            state="st-1",
            nonce="nc-1",
            code_challenge="ch-1",
            ext_agent_host_id="urn:uuid:1234",
        )
        assert dynamic_url.startswith(OPENAI_AUTHORIZE_URL)
        q = parse_qs(urlparse(dynamic_url).query)
        assert q["client_id"] == [DYNAMIC_AGENT_CLIENT_ID]
        assert q["ext_agent_host_id"] == ["urn:uuid:1234"]
        assert q["agent_name_hint"] == [DEFAULT_AGENT_NAME_HINT]
        assert q["resource"] == [OPENAI_API_RESOURCE]
        assert PLAN_USAGE_SCOPE in q["scope"][0].split()
        assert q["code_challenge_method"] == ["S256"]

        issued_url = build_authorization_url(
            client_id="oaiapp_issued_123",
            redirect_uri="http://127.0.0.1:1455/auth/callback",
            state="st-2",
            nonce="nc-2",
            code_challenge="ch-2",
            ext_agent_host_id="urn:uuid:1234",
        )
        q2 = parse_qs(urlparse(issued_url).query)
        assert q2["client_id"] == ["oaiapp_issued_123"]
        assert "agent_name_hint" not in q2

    def test_parse_callback_url(self):
        parsed = parse_callback_url(
            "http://127.0.0.1:1455/auth/callback?code=ac_1&state=st_1&client_id=oaiapp_x&scope=openid+chatgpt.tokens.use.direct"
        )
        assert parsed["code"] == "ac_1"
        assert parsed["state"] == "st_1"
        assert parsed["client_id"] == "oaiapp_x"
        assert parsed["error"] is None

        err_parsed = parse_callback_url(
            "http://127.0.0.1:1455/auth/callback?error=access_denied&error_description=User+cancelled&state=st_2"
        )
        assert err_parsed["error"] == "access_denied"
        assert err_parsed["error_description"] == "User cancelled"
        assert err_parsed["state"] == "st_2"


class TestTokenExchangeAndClaims:
    @pytest.mark.asyncio
    async def test_exchange_authorization_code_success(self):
        http_client = MagicMock()
        http_client.post = AsyncMock(
            return_value=_mock_http_response(
                200,
                {
                    "access_token": "at_123",
                    "refresh_token": "rt_456",
                    "id_token": "idt_789",
                    "token_type": "Bearer",
                    "scope": (
                        "openid profile email offline_access "
                        "resource.invoke chatgpt.tokens.use.direct"
                    ),
                    "expires_in": 3600,
                },
            )
        )

        payload = await exchange_authorization_code(
            code="code_abc",
            client_id="oaiapp_1",
            code_verifier="verifier_xyz",
            redirect_uri="http://127.0.0.1:1455/auth/callback",
            http_client=http_client,
        )

        assert payload.access_token == "at_123"
        assert payload.refresh_token == "rt_456"
        assert payload.expires_in == 3600
        call_args = http_client.post.call_args
        assert call_args.args[0] == OPENAI_TOKEN_URL
        body = parse_qs(call_args.kwargs["content"].decode("utf-8"))
        assert body["grant_type"] == ["authorization_code"]
        assert body["resource"] == [OPENAI_API_RESOURCE]
        assert body["code_verifier"] == ["verifier_xyz"]

    @pytest.mark.asyncio
    async def test_exchange_rejects_missing_plan_usage_scope(self):
        http_client = MagicMock()
        http_client.post = AsyncMock(
            return_value=_mock_http_response(
                200,
                {
                    "access_token": "at_identity_only",
                    "scope": "openid profile email",
                    "expires_in": 3600,
                },
            )
        )

        with pytest.raises(ValidationError) as exc_info:
            await exchange_authorization_code(
                code="code_abc",
                client_id="oaiapp_1",
                code_verifier="verifier_xyz",
                redirect_uri="http://127.0.0.1:1455/auth/callback",
                http_client=http_client,
            )
        assert exc_info.value.code == "oauth_scope_missing"

    @pytest.mark.asyncio
    async def test_refresh_access_token_sends_resource_and_returns_rotated_refresh(self):
        http_client = MagicMock()
        http_client.post = AsyncMock(
            return_value=_mock_http_response(
                200,
                {
                    "access_token": "at_new",
                    "refresh_token": "rt_rotated",
                    "scope": "openid offline_access chatgpt.tokens.use.direct",
                    "expires_in": 3600,
                },
            )
        )

        payload = await refresh_access_token(
            refresh_token="rt_old",
            client_id="oaiapp_1",
            http_client=http_client,
        )
        assert payload.access_token == "at_new"
        assert payload.refresh_token == "rt_rotated"
        body = parse_qs(http_client.post.call_args.kwargs["content"].decode("utf-8"))
        assert body["grant_type"] == ["refresh_token"]
        assert body["refresh_token"] == ["rt_old"]
        assert body["resource"] == [OPENAI_API_RESOURCE]

    def test_extract_id_token_claims_and_nonce_check(self):
        token = jwt.encode(
            {"sub": "user_1", "email": "op@example.com", "name": "Operator", "nonce": "n1"},
            "test-hmac-secret-at-least-32-bytes",
            algorithm="HS256",
        )
        claims = extract_id_token_claims(token, expected_nonce="n1")
        assert claims == {"sub": "user_1", "email": "op@example.com", "name": "Operator"}

        with pytest.raises(ValidationError) as exc_info:
            extract_id_token_claims(token, expected_nonce="wrong")
        assert exc_info.value.code == "oauth_nonce_mismatch"

        # The claim is required whenever the request carried a nonce: signature,
        # audience and expiry verification are disabled on purpose, so the nonce
        # is the only thing binding this token to the transaction.
        token_without_nonce = jwt.encode(
            {"sub": "user_1"}, "test-hmac-secret-at-least-32-bytes", algorithm="HS256"
        )
        with pytest.raises(ValidationError) as missing_nonce:
            extract_id_token_claims(token_without_nonce, expected_nonce="n1")
        assert missing_nonce.value.code == "oauth_nonce_mismatch"

        assert extract_id_token_claims(None) == {"sub": None, "email": None, "name": None}


class TestPendingOAuthStore:
    @pytest.mark.asyncio
    async def test_single_use_and_ttl_expiry(self):
        store = PendingOAuthStore(ttl_seconds=60)
        tx = PendingOAuthTransaction(
            state="state-1",
            provider_name="chatgpt-pro",
            client_id=DYNAMIC_AGENT_CLIENT_ID,
            client_secret=None,
            ext_agent_host_id="urn:uuid:abc",
            redirect_uri="http://127.0.0.1:1455/auth/callback",
            code_verifier="ver-1",
            nonce="n-1",
            created_at=time.time(),
        )
        await store.put(tx)
        popped = await store.pop("state-1")
        assert popped is not None
        assert popped.provider_name == "chatgpt-pro"
        assert await store.pop("state-1") is None

        expired_tx = PendingOAuthTransaction(
            state="state-expired",
            provider_name="chatgpt-pro",
            client_id=DYNAMIC_AGENT_CLIENT_ID,
            client_secret=None,
            ext_agent_host_id="urn:uuid:abc",
            redirect_uri="http://127.0.0.1:1455/auth/callback",
            code_verifier="ver-2",
            nonce="n-2",
            created_at=time.time() - 120,
        )
        await store.put(expired_tx)
        assert await store.pop("state-expired") is None

    @pytest.mark.asyncio
    async def test_redis_pop_discards_unreadable_entries(self):
        """A corrupt or version-skewed entry reads as an unknown state, not a crash.

        `pop` runs on the unauthenticated callback route outside its try block,
        so a deserialization error there would surface as a raw 500 instead of
        the designed error page.
        """

        class _FakeRedis:
            def __init__(self, value: bytes) -> None:
                self._value = value

            async def getdel(self, _key: str) -> bytes:
                return self._value

        store = PendingOAuthStore()
        assert await store.pop("state-corrupt", redis_client=_FakeRedis(b"not-json")) is None
        assert (
            await store.pop("state-skewed", redis_client=_FakeRedis(b'{"unknown_field": 1}'))
            is None
        )
