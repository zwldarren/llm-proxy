"""Dedicated ASGI middleware for the MCP proxy endpoint.

MCP requests are routed through ``/servers/{name}/mcp``. The MCP SDK's
``StreamableHTTPSessionManager`` manages its own anyio task groups and does
not tolerate being wrapped by Starlette's ``BaseHTTPMiddleware`` task groups.
This middleware intercepts MCP requests *before* they enter the main FastAPI
middleware stack, performs API-key authentication inline, and dispatches
directly to the mounted MCP proxy app.
"""

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

import orjson
from starlette.requests import Request
from starlette.types import ASGIApp, Receive, Scope, Send

from llm_proxy.api.error_responses import (
    ErrorResponseBuilder,
    budget_check_unavailable_error_body,
    budget_exceeded_error_body,
    rate_limit_exceeded_error_body,
    user_budget_exceeded_error_body,
)
from llm_proxy.api.middleware.api_key_cache import (
    CachedApiKey,
    VerifiedKeyInfo,
    get_api_key_cache,
    get_budget_spend_cache,
    get_cached_api_keys,
    hash_api_key_for_cache,
)
from llm_proxy.api.middleware.rate_limiting import get_rate_limiter
from llm_proxy.api.middleware.security import (
    build_security_headers,
    get_api_key_lockout_manager,
)
from llm_proxy.core.budget import BudgetEnvelope
from llm_proxy.core.request_utils import get_client_ip
from llm_proxy.database import ApiKeyRepository, get_async_session_context
from llm_proxy.observability.logger import get_logger
from llm_proxy.security.model_acl import intersect_model_lists
from llm_proxy.security.passwords import verify_api_key

if TYPE_CHECKING:
    from llm_proxy.database.repositories.usage_repository import UsageRepository

logger = get_logger(__name__)


async def _add_auth_failure_delay() -> None:
    """Apply a small, jittered delay after an authentication failure."""
    from llm_proxy.api.middleware.security import get_security_params

    delay_ms = get_security_params().auth_failure_delay_ms
    if delay_ms <= 0:
        return
    delay_seconds = delay_ms / 1000.0
    jitter = delay_seconds * 0.1
    await asyncio.sleep(max(0, delay_seconds + random.uniform(-jitter, jitter)))


async def _update_key_last_used(key_name: str) -> None:
    """Update the last_used timestamp for an API key in the background."""
    try:
        async with get_async_session_context() as session:
            repo = ApiKeyRepository(session)
            await repo.update_last_used(key_name)
    except Exception as e:
        logger.warning(f"Failed to update last_used for API key '{key_name}': {e}")


async def _verify_session_api_key(api_key: str) -> dict[str, Any] | None:
    """Verify a session API key (sk-ui- prefix) and return MCP auth info."""
    if not api_key.startswith("sk-ui-"):
        return None
    try:
        async with get_async_session_context() as session:
            from sqlalchemy.sql import select

            from llm_proxy.database.repositories.user_sessions import UserSessionRepository
            from llm_proxy.database.tables import UserRecord

            session_repo = UserSessionRepository(session)
            session_record = await session_repo.get_session_by_token(api_key)
            if session_record is None:
                return None

            stmt = select(UserRecord).where(UserRecord.id == session_record.user_id)
            result = await session.execute(stmt)
            user = result.scalar_one_or_none()
            if user is None or not user.is_active:
                return None

            return {
                "principal_type": "api_key",
                "principal_id": f"session:{session_record.id}",
                # Session keys inherit the owning user's model constraint.
                "allowed_models": user.allowed_models,
                # None means all MCP servers allowed (permissive default).
                "allowed_mcp_servers": None,
                "user_id": user.id,
                # Session keys are subject to the account-level budget too —
                # otherwise it would be trivially bypassable via the UI
                # session key.
                "user_budget": BudgetEnvelope.from_orm_fields(user),
            }
    except Exception as e:
        logger.warning(f"Session API key lookup failed: {e}")
    return None


def _is_key_expired(expires_at: Any) -> bool:
    """Check whether an API key has expired (naive datetimes treated as UTC)."""
    if expires_at is None:
        return False
    now = datetime.now(UTC)
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    return expires_at <= now


def _auth_info_from_key_record(
    key_record: CachedApiKey | VerifiedKeyInfo, effective_models: list[str] | None
) -> dict[str, Any]:
    """Build the auth-info dict for a verified key record."""
    return {
        "principal_type": "api_key",
        "principal_id": key_record.name,
        "allowed_models": effective_models,
        "allowed_mcp_servers": key_record.allowed_mcp_servers,
        "user_id": key_record.user_id,
        "expires_at": key_record.expires_at,
        "budget": key_record.budget,
        "user_budget": key_record.user_budget,
        "rate_limit_rpm": key_record.rate_limit_rpm,
    }


async def verify_api_key_for_mcp(api_key: str) -> dict[str, Any] | None:
    """Verify an API key string and return auth info for MCP access.

    Returns ``None`` when the key is invalid, inactive, or expired. The result
    dict has the same shape as the ``llm_proxy_auth`` scope entry created by
    the main API-key middleware, plus the key's budget configuration so the
    caller can enforce the spending cap.
    """
    cache = get_api_key_cache()
    api_key_sha256 = hash_api_key_for_cache(api_key)
    verified_info = cache.get_verified_key(api_key_sha256)

    if verified_info is not None:
        # Expiry is re-checked on every request: a key may pass its expiry
        # time while its verified-cache entry is still fresh.
        if _is_key_expired(verified_info.expires_at):
            cache.evict_verified_key(api_key_sha256)
            return None
        return _auth_info_from_key_record(verified_info, verified_info.allowed_models)

    cached_keys = await get_cached_api_keys()
    for key_record in cached_keys:
        # Keys owned by a disabled user are rejected alongside inactive keys;
        # expired keys are rejected as well.
        if (
            key_record.is_active
            and key_record.user_is_active
            and not _is_key_expired(key_record.expires_at)
            and verify_api_key(api_key, key_record.key_hash)
        ):
            # The effective model allowlist is the key's list intersected with
            # the owning user's constraint (user constraint always wins).
            effective_models = intersect_model_lists(
                key_record.allowed_models, key_record.user_allowed_models
            )
            cache.set_verified_key(
                api_key_sha256,
                key_record.name,
                effective_models,
                allowed_mcp_servers=key_record.allowed_mcp_servers,
                user_id=key_record.user_id,
                expires_at=key_record.expires_at,
                budget=key_record.budget,
                user_budget=key_record.user_budget,
                rate_limit_rpm=key_record.rate_limit_rpm,
            )
            return _auth_info_from_key_record(key_record, effective_models)

    session_auth = await _verify_session_api_key(api_key)
    if session_auth is not None:
        return session_auth

    return None


class BudgetCheckUnavailableError(Exception):
    """A budgeted principal's current spend could not be confirmed (stats DB error).

    Raised by :func:`is_key_budget_exceeded` and :func:`is_user_budget_exceeded`
    instead of silently allowing the request, so budget enforcement fails
    closed: while the spend cannot be confirmed, the cap cannot be enforced
    and the request must be rejected.
    """

    def __init__(self, subject: str):
        super().__init__(f"Budget check unavailable for {subject}")
        self.subject = subject


async def _spend_exceeds(
    envelope: BudgetEnvelope,
    *,
    cache_key: str,
    subject: str,
    query: Callable[[UsageRepository, float], Awaitable[float]],
) -> bool:
    """Whether the current-period spend under ``envelope`` reached its cap.

    ``cache_key`` is the namespaced spend-cache key ("key:<name>" or
    "user:<id>"); ``query`` fetches the spend from the usage repository for
    the window starting at the given timestamp. Results are cached for a
    short TTL so budget-limited principals cost at most one indexed SUM
    query per TTL window.

    Fails closed: when the spend cannot be confirmed (stats-DB error),
    raises :class:`BudgetCheckUnavailableError` instead of silently allowing
    unbounded spend. Note: usage is written asynchronously after requests
    complete, so this is a best-effort limit — spend may slightly overshoot
    the cap.
    """
    if envelope.budget_usd is None:
        return False
    # The window the enforced spend counts from. Computed before the cache
    # lookup so a spend cached for a *previous* window (a calendar rollover
    # or a manual reset since it was cached) is never enforced against the
    # current window.
    since_ts = envelope.effective_start_ts()
    spend_cache = get_budget_spend_cache()
    spend = spend_cache.get(cache_key, window_start=since_ts)
    if spend is None:
        from llm_proxy.database.repositories.usage_repository import UsageRepository

        try:
            async with get_async_session_context() as session:
                spend = await query(UsageRepository(session), since_ts)
        except Exception as e:
            # Fail closed: a budget exists, and while the spend cannot be
            # confirmed the cap cannot be enforced. Rejecting here turns a
            # stats-DB outage into a 503 for budgeted principals instead of
            # silently unbounded spend; principals without a budget are
            # unaffected.
            logger.error(f"Budget spend lookup failed for {subject}: {e}")
            raise BudgetCheckUnavailableError(subject=subject) from e
        spend_cache.set(cache_key, spend, window_start=since_ts)

    return spend >= envelope.budget_usd


async def is_key_budget_exceeded(auth_info: dict[str, Any]) -> bool:
    """Check whether the verified key has reached its current-period budget.

    Queries the usage table for the key's spend since the start of the
    current budget window (UTC calendar boundary, possibly truncated by a
    manual reset). See :func:`_spend_exceeds` for the shared enforcement.
    """
    key_name = auth_info.get("principal_id")
    # Session keys ("session:<id>") carry no budget configuration.
    if not key_name or str(key_name).startswith("session:"):
        return False
    # Key names are user-controlled strings, so the cache key is namespaced
    # ("key:") — a key named e.g. "user:7" must not alias the account-level
    # spend entry for user 7, or the two checks would read each other's
    # spend (bypassing a cap or producing a false 429).
    return await _spend_exceeds(
        auth_info.get("budget") or BudgetEnvelope(),
        cache_key=f"key:{key_name}",
        subject=f"API key '{key_name}'",
        query=lambda repo, since_ts: repo.get_key_spend_since(str(key_name), since_ts),
    )


async def is_user_budget_exceeded(auth_info: dict[str, Any]) -> bool:
    """Check whether the key owner's account-level budget has been reached.

    Aggregates the owner's spend across all of their keys over the account
    budget window (UTC calendar boundary, possibly truncated by a manual
    reset). Applies to session keys as well as regular API keys. See
    :func:`_spend_exceeds` for the shared enforcement.
    """
    user_id = auth_info.get("user_id")
    if user_id is None:
        return False
    # Distinct namespace from the key-level entries ("key:<name>"): the two
    # checks share one spend cache, so their key spaces must be disjoint.
    return await _spend_exceeds(
        auth_info.get("user_budget") or BudgetEnvelope(),
        cache_key=f"user:{user_id}",
        subject=f"user {user_id}",
        query=lambda repo, since_ts: repo.get_user_spend_since(user_id, since_ts),
    )


class BudgetCheckStatus(Enum):
    """Outcome of a budget check for a verified API key.

    Encodes the states the middlewares act on: under the caps (proceed), the
    key's own cap reached (429), the owner account's cap reached (429), or
    spend unconfirmable (fail closed with 503).
    """

    OK = "ok"
    EXCEEDED = "exceeded"
    USER_EXCEEDED = "user_exceeded"
    UNAVAILABLE = "unavailable"


async def check_key_budget(auth_info: dict[str, Any]) -> BudgetCheckStatus:
    """Check a verified key's budget caps; returns a :class:`BudgetCheckStatus`.

    Checks the key-level cap first, then the owner account's admin-set
    envelope. Shared by the /v1/* and /servers/* middlewares so the
    fail-closed behaviour is defined in one place. When the current spend
    cannot be confirmed (stats-DB error), the check reports ``UNAVAILABLE``
    and the caller must reject with 503 instead of allowing unbounded spend.
    """
    try:
        if await is_key_budget_exceeded(auth_info):
            return BudgetCheckStatus.EXCEEDED
        if await is_user_budget_exceeded(auth_info):
            return BudgetCheckStatus.USER_EXCEEDED
        return BudgetCheckStatus.OK
    except BudgetCheckUnavailableError:
        return BudgetCheckStatus.UNAVAILABLE


@dataclass(frozen=True)
class BudgetRejection:
    """A budget-check rejection: the HTTP status, error body, and log line.

    Returned by :func:`budget_status_rejection` so the /v1/* and /servers/*
    middlewares share one rejection mapping while each owning its own
    transport (JSONResponse vs raw ASGI with security headers).
    """

    status_code: int
    error_body: dict[str, Any]
    log_message: str


def budget_status_rejection(
    status: BudgetCheckStatus,
    *,
    principal_id: str,
    user_id: int | None,
) -> BudgetRejection | None:
    """Map a budget-check outcome to a rejection, or None to proceed.

    Returns a :class:`BudgetRejection` for the states the /v1/* and
    /servers/* middlewares act on: the key's own cap reached (429), the
    owner account's cap reached (429), or spend unconfirmable (fail closed
    with 503). Sharing this mapping keeps both middlewares rejecting
    identically; each caller still owns its own transport (JSONResponse vs
    raw ASGI with security headers).
    """
    if status is BudgetCheckStatus.OK:
        return None
    if status is BudgetCheckStatus.UNAVAILABLE:
        return BudgetRejection(
            status_code=503,
            error_body=budget_check_unavailable_error_body(),
            log_message="budget check unavailable (fail closed)",
        )
    if status is BudgetCheckStatus.EXCEEDED:
        return BudgetRejection(
            status_code=429,
            error_body=budget_exceeded_error_body(principal_id),
            log_message=f"budget exceeded for '{principal_id}'",
        )
    return BudgetRejection(
        status_code=429,
        error_body=user_budget_exceeded_error_body(),
        log_message=f"account budget exceeded for user {user_id}",
    )


async def check_key_rate_limit(auth_info: dict[str, Any]) -> tuple[int, int] | None:
    """Check a verified key's per-minute rate limit.

    Shared by the /v1/* and /servers/* middlewares so the per-key sliding
    window is enforced in one place. Session keys carry no rate-limit
    configuration, so this only applies to regular API keys. Returns
    ``(limit_rpm, retry_after)`` when the key is over its cap, else ``None``.
    """
    rate_limit_rpm = auth_info.get("rate_limit_rpm")
    if rate_limit_rpm is None:
        return None
    is_limited, limit_meta = await get_rate_limiter().check_rate_limit(
        identifier=f"apikey:{auth_info['principal_id']}",
        limit=rate_limit_rpm,
        window_size=60,
    )
    if not is_limited:
        return None
    return rate_limit_rpm, max(1, int(limit_meta["reset_time"] - time.time()))


def _extract_api_key(scope: Scope) -> str | None:
    """Read the API key from Authorization or x-api-key headers."""
    headers = dict(scope.get("headers", []))
    auth_header = None
    x_api_key = None
    for name, value in headers.items():
        decoded_name = name.decode().lower()
        if decoded_name == "authorization":
            auth_header = value.decode()
        elif decoded_name == "x-api-key":
            x_api_key = value.decode()

    if auth_header:
        if not auth_header.startswith("Bearer "):
            return None
        return auth_header[7:]
    if x_api_key:
        return x_api_key
    return None


async def _send_json(
    send: Send,
    status: int,
    body: dict[str, Any],
    extra_headers: list[tuple[bytes, bytes]] | None = None,
) -> None:
    """Send a JSON ASGI response, optionally with extra headers."""
    headers = [[b"content-type", b"application/json"]]
    if extra_headers:
        headers.extend([list(h) for h in extra_headers])
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": headers,
        }
    )
    await send(
        {
            "type": "http.response.body",
            "body": orjson.dumps(body),
        }
    )


class MCPProxyMiddleware:
    """Outermost ASGI middleware that routes MCP requests around FastAPI middleware.

    This is a plain ASGI callable, *not* a ``BaseHTTPMiddleware``, so it does not
    create the anyio task group that conflicts with the MCP SDK's own task-group
    management. It performs API-key authentication inline and then dispatches to
    the mounted MCP proxy app.
    """

    def __init__(
        self,
        app: ASGIApp,
        main_app: ASGIApp,
        mcp_app: ASGIApp,
    ) -> None:
        self.app = app
        self.main_app = main_app
        self.mcp_app = mcp_app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http" or not scope.get("path", "").startswith("/servers/"):
            await self.app(scope, receive, send)
            return

        # OPTIONS preflight must pass through to the inner middleware stack
        # (which includes CORSMiddleware) so CORS preflight succeeds.
        # Browsers do not send Authorization on preflight, so attempting
        # API-key auth here would always 401.
        if scope.get("method") == "OPTIONS":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        client_ip = get_client_ip(request)
        lockout_manager = get_api_key_lockout_manager()

        # This middleware short-circuits before the security-headers middleware,
        # so its own responses must carry the same header set.
        security_headers = [
            (k.encode(), v.encode())
            for k, v in build_security_headers(
                getattr(getattr(self.main_app, "state", None), "config_manager", None)
            ).items()
        ]

        if lockout_manager.is_locked_out(client_ip):
            remaining = lockout_manager.get_lockout_remaining(client_ip)
            logger.warning(
                "MCP request from locked out IP",
                client_ip=client_ip,
                remaining=remaining,
            )
            await _add_auth_failure_delay()
            await _send_json(
                send,
                429,
                {
                    "error": {
                        "message": (
                            f"Too many failed authentication attempts. "
                            f"Try again in {remaining} seconds."
                        ),
                        "type": "rate_limit_error",
                        "code": "too_many_auth_failures",
                    }
                },
                extra_headers=security_headers,
            )
            return

        api_key = _extract_api_key(scope)
        if api_key is None:
            await _add_auth_failure_delay()
            lockout_manager.record_failed_attempt(client_ip)
            await _send_json(
                send,
                401,
                ErrorResponseBuilder.create_openai_error(
                    "Authorization header missing or invalid format",
                    error_type="authentication_error",
                    code="invalid_api_key",
                ),
                extra_headers=security_headers,
            )
            return

        auth_info = await verify_api_key_for_mcp(api_key)
        if auth_info is None:
            await _add_auth_failure_delay()
            lockout_manager.record_failed_attempt(client_ip)
            logger.warning(f"Invalid API key attempt for MCP from {client_ip}")
            await _send_json(
                send,
                401,
                ErrorResponseBuilder.create_openai_error(
                    "Invalid API key",
                    error_type="authentication_error",
                    code="invalid_api_key",
                ),
                extra_headers=security_headers,
            )
            return

        # Per-key rate limit (requests/minute) from the cached key snapshot;
        # session keys carry no rate-limit configuration. Checked before the
        # budget: it is the cheaper rejection. MCP requests never pass
        # through the /v1/* middleware, so the check runs here as well.
        rate_limit = await check_key_rate_limit(auth_info)
        if rate_limit is not None:
            limit_rpm, retry_after = rate_limit
            logger.info(
                f"MCP request rejected: rate limit exceeded for API key "
                f"'{auth_info['principal_id']}'"
            )
            await _send_json(
                send,
                429,
                rate_limit_exceeded_error_body(limit_rpm, retry_after),
                extra_headers=security_headers + [(b"retry-after", str(retry_after).encode())],
            )
            return

        # Budget-limited keys that reached their cap are rejected with 429,
        # as are keys whose owner's account-level budget is exhausted. This
        # is not an auth failure: no lockout penalty, no failure delay. When
        # the budget cannot be checked (stats DB error), enforcement fails
        # closed with 503 instead of silently allowing unbounded spend.
        budget_status = await check_key_budget(auth_info)
        rejection = budget_status_rejection(
            budget_status,
            principal_id=auth_info["principal_id"],
            user_id=auth_info.get("user_id"),
        )
        if rejection is not None:
            logger.info(f"MCP request rejected: {rejection.log_message}")
            await _send_json(
                send,
                rejection.status_code,
                rejection.error_body,
                extra_headers=security_headers,
            )
            return

        lockout_manager.clear_failed_attempts(client_ip)
        asyncio.create_task(_update_key_last_used(auth_info["principal_id"]))

        # Surface the main FastAPI app in the scope so the mounted MCP proxy can
        # reach app.state.mcp_manager.
        scope["app"] = self.main_app
        scope["llm_proxy_auth"] = auth_info

        await self.mcp_app(scope, receive, send)
