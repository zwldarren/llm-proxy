"""OpenAI "Sign in with ChatGPT" OAuth 2.0 / OIDC PKCE helpers.

Implements OpenAI's official Sign in with ChatGPT flow for plan-usage access
(``chatgpt.tokens.use.direct``), including:

* Dynamic Client Registration (``client_id=dynamic_agent_client`` +
  ``ext_agent_host_id`` + ``agent_name_hint``) and static ``oaiapp_`` clients;
* PKCE S256 generation, redirect URI validation, and authorization URL building;
* Authorization-code exchange and refresh-token rotation against
  ``https://auth.openai.com/api/accounts/oauth/token`` with
  ``resource=https://api.openai.com/v1``;
* Hard enforcement that ``chatgpt.tokens.use.direct`` was actually consented to
  and returned in the granted ``scope``;
* Pending single-use OAuth transaction storage (Redis when enabled, bounded
  in-memory fallback otherwise).
"""

import asyncio
import hashlib
import secrets
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx2
import jwt
import orjson
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from llm_proxy.core.exceptions import ValidationError
from llm_proxy.database.repositories.config_server import ServerConfigRepository
from llm_proxy.observability.logger import get_logger
from llm_proxy.security.base64url import b64url_encode
from llm_proxy.security.encryption import decrypt_api_key, encrypt_api_key

logger = get_logger(__name__)

OPENAI_OAUTH_ISSUER = "https://auth.openai.com"
OPENAI_AUTHORIZE_URL = "https://auth.openai.com/api/accounts/authorize"
OPENAI_TOKEN_URL = "https://auth.openai.com/api/accounts/oauth/token"
OPENAI_API_RESOURCE = "https://api.openai.com/v1"

DYNAMIC_AGENT_CLIENT_ID = "dynamic_agent_client"
PLAN_USAGE_SCOPE = "chatgpt.tokens.use.direct"
DEFAULT_OAUTH_SCOPES: tuple[str, ...] = (
    "openid",
    "profile",
    "email",
    "offline_access",
    "resource.invoke",
    PLAN_USAGE_SCOPE,
)
DEFAULT_AGENT_NAME_HINT = "LLM Proxy"
DEFAULT_LOOPBACK_CALLBACK_URI = "http://127.0.0.1:1455/auth/callback"
HOST_ID_CONFIG_KEY = "openai_chatgpt_host_id"
PENDING_OAUTH_TTL_SECONDS = 600

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_PENDING_REDIS_PREFIX = "llm_proxy:provider_oauth_pending:"
_MAX_PENDING_TRANSACTIONS = 256


@dataclass(frozen=True, slots=True)
class OAuthTokenPayload:
    """Normalized response from OpenAI's OAuth token endpoint."""

    access_token: str
    refresh_token: str | None
    id_token: str | None
    token_type: str
    scope: str
    expires_in: int | None


@dataclass(slots=True)
class PendingOAuthTransaction:
    """Single-use PKCE state created when an admin starts Sign in with ChatGPT."""

    state: str
    provider_name: str
    client_id: str
    client_secret: str | None
    ext_agent_host_id: str | None
    redirect_uri: str
    code_verifier: str
    nonce: str
    created_at: float


def generate_pkce_pair() -> tuple[str, str]:
    """Generate a PKCE ``(code_verifier, code_challenge)`` pair (S256)."""
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = b64url_encode(digest)
    return verifier, challenge


def validate_oauth_redirect_uri(uri: str) -> str:
    """Validate an OAuth redirect URI (HTTPS anywhere, or HTTP on loopback only)."""
    cleaned = uri.strip()
    if not cleaned:
        raise ValidationError("Redirect URI cannot be empty.", code="invalid_redirect_uri")
    if any(ch.isspace() for ch in cleaned):
        raise ValidationError(
            "Redirect URI must not contain whitespace.", code="invalid_redirect_uri"
        )
    parsed = urlparse(cleaned)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or not parsed.hostname:
        raise ValidationError(
            "Redirect URI must be an absolute http(s) URL.", code="invalid_redirect_uri"
        )
    if parsed.scheme == "http" and parsed.hostname.lower() not in _LOOPBACK_HOSTS:
        raise ValidationError(
            "Plain HTTP redirect URIs are only allowed for loopback hosts "
            "(localhost, 127.0.0.1, ::1); use HTTPS otherwise.",
            code="invalid_redirect_uri",
        )
    return cleaned


def build_authorization_url(
    *,
    client_id: str,
    redirect_uri: str,
    state: str,
    nonce: str,
    code_challenge: str,
    ext_agent_host_id: str | None = None,
    agent_name_hint: str | None = None,
) -> str:
    """Construct the OpenAI ``/api/accounts/authorize`` URL."""
    params: dict[str, str] = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": " ".join(DEFAULT_OAUTH_SCOPES),
        "resource": OPENAI_API_RESOURCE,
        "state": state,
        "nonce": nonce,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    if ext_agent_host_id:
        params["ext_agent_host_id"] = ext_agent_host_id
    if client_id == DYNAMIC_AGENT_CLIENT_ID:
        params["agent_name_hint"] = agent_name_hint or DEFAULT_AGENT_NAME_HINT
    return f"{OPENAI_AUTHORIZE_URL}?{urlencode(params)}"


def parse_callback_url(callback_url: str) -> dict[str, str | None]:
    """Extract OAuth callback query parameters from a full URL or query string."""
    raw = callback_url.strip()
    if not raw:
        raise ValidationError("Callback URL cannot be empty.", code="invalid_callback_url")
    if "?" in raw:
        query_str = urlparse(raw).query
        if not query_str and raw.startswith("?"):
            query_str = raw[1:]
    elif "code=" in raw or "error=" in raw or "state=" in raw:
        query_str = raw
    else:
        query_str = urlparse(raw).query

    parsed = parse_qs(query_str, keep_blank_values=False)

    def _first(key: str) -> str | None:
        values = parsed.get(key)
        return values[0] if values else None

    return {
        "code": _first("code"),
        "state": _first("state"),
        "client_id": _first("client_id"),
        "scope": _first("scope"),
        "error": _first("error"),
        "error_description": _first("error_description"),
    }


def has_plan_usage_scope(scope: str | None) -> bool:
    """Return True only when ``chatgpt.tokens.use.direct`` is in ``scope``."""
    if not scope:
        return False
    return PLAN_USAGE_SCOPE in set(scope.split())


def extract_id_token_claims(
    id_token: str | None, *, expected_nonce: str | None = None
) -> dict[str, str | None]:
    """Decode display claims (``sub``, ``email``, ``name``) from an OpenAI ID token.

    ``nonce`` is required whenever the caller sent one: with signature
    verification deliberately skipped, it is the only binding of the ID token
    to the transaction that requested it.
    """
    empty: dict[str, str | None] = {"sub": None, "email": None, "name": None}
    if not id_token:
        return empty
    try:
        claims = jwt.decode(
            id_token,
            options={
                "verify_signature": False,
                "verify_aud": False,
                "verify_exp": False,
            },
        )
    except Exception:
        return empty
    if not isinstance(claims, dict):
        return empty
    if expected_nonce is not None:
        # OIDC Core requires the claim to be present and to match when the
        # authentication request carried a nonce; a token that omits it is not
        # proof of anything.
        token_nonce = claims.get("nonce")
        if token_nonce != expected_nonce:
            raise ValidationError("OIDC nonce mismatch in ID token.", code="oauth_nonce_mismatch")
    return {
        "sub": str(claims["sub"]) if claims.get("sub") is not None else None,
        "email": str(claims["email"]) if claims.get("email") is not None else None,
        "name": str(claims["name"]) if claims.get("name") is not None else None,
    }


def _extract_token_error_message(response: Any) -> str:
    try:
        data = response.json()
    except Exception:
        text = getattr(response, "text", "") or ""
        return text[:300] or f"HTTP {response.status_code}"
    if isinstance(data, dict):
        desc = data.get("error_description") or data.get("detail")
        err = data.get("error")
        if isinstance(err, dict):
            return str(err.get("message") or err.get("code") or data)
        if desc and err:
            return f"{err}: {desc}"
        if desc or err:
            return str(desc or err)
    return f"HTTP {response.status_code}"


async def _post_token_form(form: dict[str, str], *, http_client: Any = None) -> OAuthTokenPayload:
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
    }
    encoded = urlencode(form).encode("utf-8")
    if http_client is not None:
        response = await http_client.post(OPENAI_TOKEN_URL, headers=headers, content=encoded)
    else:
        async with httpx2.AsyncClient(timeout=20.0, follow_redirects=False) as client:
            response = await client.post(OPENAI_TOKEN_URL, headers=headers, content=encoded)

    if response.status_code >= 400:
        msg = _extract_token_error_message(response)
        raise ValidationError(
            message=f"OpenAI OAuth token request failed ({response.status_code}): {msg}",
            code="oauth_token_exchange_failed",
        )

    try:
        data = response.json()
    except Exception as exc:
        raise ValidationError(
            message="OpenAI OAuth token endpoint returned invalid JSON.",
            code="oauth_token_invalid_response",
        ) from exc

    if not isinstance(data, dict) or not data.get("access_token"):
        raise ValidationError(
            message="OpenAI OAuth token endpoint response is missing access_token.",
            code="oauth_token_invalid_response",
        )

    raw_expires = data.get("expires_in")
    expires_in: int | None = None
    if isinstance(raw_expires, (int, float)) and not isinstance(raw_expires, bool):
        expires_in = int(raw_expires)

    return OAuthTokenPayload(
        access_token=str(data["access_token"]),
        refresh_token=str(data["refresh_token"]) if data.get("refresh_token") else None,
        id_token=str(data["id_token"]) if data.get("id_token") else None,
        token_type=str(data.get("token_type") or "Bearer"),
        scope=str(data.get("scope") or ""),
        expires_in=expires_in,
    )


async def exchange_authorization_code(
    *,
    code: str,
    client_id: str,
    code_verifier: str,
    redirect_uri: str,
    client_secret: str | None = None,
    http_client: Any = None,
) -> OAuthTokenPayload:
    """Exchange an authorization code for OAuth tokens and enforce plan-usage scope."""
    form: dict[str, str] = {
        "grant_type": "authorization_code",
        "client_id": client_id,
        "code": code,
        "code_verifier": code_verifier,
        "redirect_uri": redirect_uri,
        "resource": OPENAI_API_RESOURCE,
    }
    if client_secret:
        form["client_secret"] = client_secret

    payload = await _post_token_form(form, http_client=http_client)
    if not has_plan_usage_scope(payload.scope):
        raise ValidationError(
            message=(
                "OpenAI did not grant the 'chatgpt.tokens.use.direct' scope. "
                "Verify your ChatGPT subscription supports plan-usage OAuth access "
                "and approve plan usage during sign-in."
            ),
            code="oauth_scope_missing",
        )
    return payload


async def refresh_access_token(
    *,
    refresh_token: str,
    client_id: str,
    client_secret: str | None = None,
    http_client: Any = None,
) -> OAuthTokenPayload:
    """Refresh an access token using a rotated ``refresh_token``."""
    form: dict[str, str] = {
        "grant_type": "refresh_token",
        "client_id": client_id,
        "refresh_token": refresh_token,
        "resource": OPENAI_API_RESOURCE,
    }
    if client_secret:
        form["client_secret"] = client_secret

    payload = await _post_token_form(form, http_client=http_client)
    if payload.scope and not has_plan_usage_scope(payload.scope):
        raise ValidationError(
            message="Refreshed token no longer includes 'chatgpt.tokens.use.direct' scope.",
            code="oauth_scope_missing",
        )
    return payload


class PendingOAuthStore:
    """Single-use store for pending PKCE transactions (Redis + memory fallback)."""

    def __init__(self, ttl_seconds: int = PENDING_OAUTH_TTL_SECONDS) -> None:
        self._ttl_seconds = ttl_seconds
        self._memory: dict[str, PendingOAuthTransaction] = {}
        self._lock: asyncio.Lock | None = None
        self._lock_loop: asyncio.AbstractEventLoop | None = None

    def _get_lock(self) -> asyncio.Lock:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if self._lock is None or self._lock_loop != loop:
            self._lock = asyncio.Lock()
            self._lock_loop = loop
        return self._lock

    def _prune_memory(self, now: float) -> None:
        expired = [
            key for key, tx in self._memory.items() if now - tx.created_at > self._ttl_seconds
        ]
        for key in expired:
            self._memory.pop(key, None)
        while len(self._memory) >= _MAX_PENDING_TRANSACTIONS:
            oldest_key = next(iter(self._memory))
            self._memory.pop(oldest_key, None)

    async def put(self, tx: PendingOAuthTransaction, *, redis_client: Any = None) -> None:
        """Store a pending transaction under ``tx.state``."""
        if redis_client is not None:
            raw = orjson.dumps(asdict(tx)).decode("utf-8")
            # Non-empty input either returns a non-empty string or raises
            # (encryption disabled keeps it verbatim); a fallback to plaintext
            # here would write the PKCE verifier and client secret in the clear.
            encrypted = encrypt_api_key(raw)
            await redis_client.set(
                f"{_PENDING_REDIS_PREFIX}{tx.state}",
                encrypted,
                ex=self._ttl_seconds,
            )
            return

        async with self._get_lock():
            now = time.time()
            self._prune_memory(now)
            self._memory[tx.state] = tx

    async def pop(self, state: str, *, redis_client: Any = None) -> PendingOAuthTransaction | None:
        """Atomically retrieve and consume a pending transaction, or return ``None``."""
        if not state:
            return None
        if redis_client is not None:
            key = f"{_PENDING_REDIS_PREFIX}{state}"
            raw_val: Any = None
            getdel = getattr(redis_client, "getdel", None)
            if callable(getdel):
                raw_val = await getdel(key)
            else:
                raw_val = await redis_client.get(key)
                if raw_val is not None:
                    await redis_client.delete(key)
            if raw_val is None:
                return None
            if isinstance(raw_val, bytes):
                raw_val = raw_val.decode("utf-8")
            try:
                decrypted = decrypt_api_key(str(raw_val))
                if not decrypted:
                    return None
                data = orjson.loads(decrypted)
                tx = PendingOAuthTransaction(**data)
            except Exception as exc:
                # A rotated encryption key, a corrupt value or a deploy that
                # changed the transaction shape must behave like an unknown
                # state, not raise into the unauthenticated callback page.
                logger.warning(f"Discarding unreadable pending OAuth transaction: {exc}")
                return None
            if time.time() - tx.created_at > self._ttl_seconds:
                return None
            return tx

        async with self._get_lock():
            tx = self._memory.pop(state, None)
            if tx is None:
                return None
            if time.time() - tx.created_at > self._ttl_seconds:
                return None
            return tx

    def clear(self) -> None:
        self._memory.clear()


_pending_store = PendingOAuthStore()


def get_pending_oauth_store() -> PendingOAuthStore:
    """Return the process-wide pending OAuth transaction store."""
    return _pending_store


async def get_or_create_host_id(session: AsyncSession) -> str:
    """Return the persisted ``ext_agent_host_id`` (``urn:uuid:...``), creating it once."""
    repo = ServerConfigRepository(session)
    existing = await repo.get_server_config(HOST_ID_CONFIG_KEY)
    if existing is not None and isinstance(existing.value, str) and existing.value.strip():
        return existing.value.strip()

    host_id = f"urn:uuid:{uuid.uuid4()}"
    try:
        async with session.begin_nested():
            await repo.set_server_config(
                key=HOST_ID_CONFIG_KEY,
                value=host_id,
                description=(
                    "Persistent installation identifier for OpenAI ChatGPT OAuth "
                    "dynamic client registration"
                ),
            )
    except IntegrityError:
        # Two first-time sign-ins can race the get-then-insert; the unique key
        # decides the winner and the loser adopts the persisted id.
        existing = await repo.get_server_config(HOST_ID_CONFIG_KEY)
        if existing is not None and isinstance(existing.value, str) and existing.value.strip():
            return existing.value.strip()
        raise
    return host_id


__all__ = [
    "DEFAULT_AGENT_NAME_HINT",
    "DEFAULT_LOOPBACK_CALLBACK_URI",
    "DEFAULT_OAUTH_SCOPES",
    "DYNAMIC_AGENT_CLIENT_ID",
    "HOST_ID_CONFIG_KEY",
    "OPENAI_API_RESOURCE",
    "OPENAI_AUTHORIZE_URL",
    "OPENAI_OAUTH_ISSUER",
    "OPENAI_TOKEN_URL",
    "PENDING_OAUTH_TTL_SECONDS",
    "PLAN_USAGE_SCOPE",
    "OAuthTokenPayload",
    "PendingOAuthStore",
    "PendingOAuthTransaction",
    "build_authorization_url",
    "exchange_authorization_code",
    "extract_id_token_claims",
    "generate_pkce_pair",
    "get_or_create_host_id",
    "get_pending_oauth_store",
    "has_plan_usage_scope",
    "parse_callback_url",
    "refresh_access_token",
    "validate_oauth_redirect_uri",
]
