"""The API-key gate: one sequence every protected transport passes through.

Key extraction, verification (bcrypt against the cached key snapshot with
cheap revalidation and a session-key fallback), client-IP lockout accounting,
the auth-failure penalty, the per-key rate limit, the key and user budget
(fail-closed), and the throttled last-used claim — one module owns the whole
sequence (:func:`gate_request`). A transport declares a :class:`GatePolicy`
(which checks apply, how strictly a key is extracted) and renders the returned
:class:`GateRejection` in its own wire form; the gate itself logs every
rejection, so "a rejection leaves a row" is an invariant of this module, not a
courtesy of whichever transport remembered it.

The verified principal is a fact of the request, not a return value to carry
around: identity is stamped into the request facts (``RequestIdentity``) here,
and the allowlists travel as facts too. See the "Key gate" entry in
``CONTEXT.md``.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from starlette.requests import Request

from llm_proxy.api.error_responses import (
    auth_error_body,
    budget_check_unavailable_error_body,
    budget_exceeded_error_body,
    lockout_error_body,
    rate_limit_exceeded_error_body,
    user_budget_exceeded_error_body,
)
from llm_proxy.api.middleware.api_key_cache import (
    ApiKeyCache,
    CachedApiKey,
    VerifiedKeyInfo,
    claim_last_used_update,
    get_api_key_cache,
    get_budget_spend_cache,
    get_cached_api_keys,
    get_key_verification_lock,
    hash_api_key_for_cache,
)
from llm_proxy.api.middleware.api_key_cache import (
    persist_key_last_used as _update_key_last_used,
)
from llm_proxy.api.middleware.exceptions import protocol_for_request
from llm_proxy.api.middleware.rate_limiting import get_rate_limiter
from llm_proxy.api.middleware.security import (
    apply_auth_failure_delay,
    get_api_key_lockout_manager,
)
from llm_proxy.api.session_cookies import SESSION_API_KEY_COOKIE
from llm_proxy.core.budget import BudgetEnvelope
from llm_proxy.core.request_facts import (
    RequestIdentity,
    facts_for,
    get_request_identity,
    set_request_identity,
)
from llm_proxy.core.request_utils import get_client_ip
from llm_proxy.database import get_async_session_context
from llm_proxy.observability.log_intake import record_failed_auth, record_rejection
from llm_proxy.observability.logger import get_logger
from llm_proxy.security.model_acl import intersect_model_lists
from llm_proxy.security.passwords import verify_api_key as key_matches_hash

if TYPE_CHECKING:
    from llm_proxy.database.repositories.usage_repository import UsageRepository

logger = get_logger(__name__)


class KeyExtraction(Enum):
    """How strictly a transport reads the API key from the request.

    The three transports legitimately disagree on what a non-Bearer
    ``Authorization`` header means: the /v1/* middleware calls it an invalid
    format (an auth failure), the /servers/* MCP middleware treats it as "no
    key supplied", and the WebSocket transports accept a raw (non-Bearer)
    header because browser clients cannot always set headers.
    """

    #: ``Authorization: Bearer <key>`` or ``x-api-key``; a non-Bearer
    #: Authorization header is an invalid-format auth failure.
    STRICT = "strict"
    #: Same headers, but a non-Bearer Authorization header counts as "no key
    #: supplied" rather than a malformed attempt.
    BEARER_OR_NONE = "bearer_or_none"
    #: WebSocket sources: a Bearer-stripped or raw Authorization header,
    #: ``x-api-key``, the insecure subprotocol prefix, or the ``api_key``
    #: query parameter (browser WebSocket clients cannot set headers).
    WEBSOCKET = "websocket"


@dataclass(frozen=True)
class GatePolicy:
    """Which checks one transport's gate run declares, as data.

    Declared next to the transport that speaks that wire (the provider
    declaration pattern): adding a transport means adding a row here, in the
    transport's own module, not editing the gate. The booleans name checks;
    absence of a check is a declaration, not an accident.
    """

    #: How the request's key is read (see :class:`KeyExtraction`).
    extraction: KeyExtraction
    #: Subprotocol prefix accepted as a key source (WebSocket only).
    ws_subprotocol_prefix: str | None = None
    #: Accept the browser's HttpOnly /v1 session-key cookie when no explicit
    #: Authorization or x-api-key header was supplied. The transport must
    #: validate CSRF before calling the gate for unsafe methods.
    allow_session_cookie: bool = False
    #: Client-IP lockout: check on entry, clear failed attempts on success.
    lockout: bool = True
    #: On a missing/invalid key: record a failed attempt and pay the
    #: auth-failure delay. WebSocket transports do neither (their handshake
    #: has no lockout accounting).
    auth_failure_penalty: bool = True
    #: Per-key requests/minute cap from the cached key snapshot.
    rate_limit: bool = True
    #: Key-level and account-level spending caps, fail-closed.
    budget: bool = True
    #: Claim the throttled ``last_used_at`` persistence slot on success.
    claim_last_used: bool = True


class GateRejectionKind(Enum):
    """What kind of gate outcome the rejection is, for non-HTTP transports.

    The HTTP body already speaks for JSON transports; a WebSocket renderer
    needs the kind to pick its close code and error-event shape.
    """

    AUTH_FAILED = "auth_failed"  #: 401 — missing, malformed, or invalid key
    LOCKED_OUT = "locked_out"  #: 429 — client IP locked out
    RATE_LIMITED = "rate_limited"  #: 429 — per-key requests/minute cap
    BUDGET_EXCEEDED = "budget_exceeded"  #: 429 — key or account cap reached
    BUDGET_UNAVAILABLE = "budget_unavailable"  #: 503 — fail closed


@dataclass(frozen=True)
class GateRejection:
    """A structured gate rejection: everything a transport needs to render it.

    ``body`` is the protocol-shaped HTTP error body (OpenAI envelope, or the
    Anthropic envelope on the Anthropic Messages path). ``message`` is the
    human-readable line for transports that do not speak the HTTP envelope.
    """

    kind: GateRejectionKind
    status_code: int
    body: dict[str, Any]
    message: str
    headers: dict[str, str] = field(default_factory=dict)


# ── Verification ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class VerifiedKeyAuth:
    """The verified key's enforcement facts, as one typed record.

    Returned by :func:`verify_api_key`; the gate's quota checks read it by
    attribute instead of reaching into a string-keyed dict. This is the
    gate's own plumbing, not a principal carrier: the verified principal
    itself travels only as the request facts' identity (the second identity
    channel ADR-0032 rejected).
    """

    #: The key's name, or ``session:<id>`` for a session key.
    principal_id: str
    #: ``None`` = unrestricted, ``[]`` = deny-all.
    allowed_models: list[str] | None = None
    #: ``None`` = all MCP servers allowed (permissive default).
    allowed_mcp_servers: list[str] | None = None
    user_id: int | None = None
    expires_at: datetime | None = None
    budget: BudgetEnvelope | None = None
    user_budget: BudgetEnvelope | None = None
    rate_limit_rpm: int | None = None
    #: Admin-set account-level rpm cap for the key's owner (None = unlimited).
    user_rate_limit_rpm: int | None = None
    #: Session keys (``sk-ui-``) authenticate through the session store: they
    #: carry no key-level budget or rate limit, and their identity includes
    #: the owning user's name.
    is_session: bool = False
    #: The owning user's name, resolved in the same DB pass that verified a
    #: session key — the identity is built from it without a second lookup.
    username: str | None = None


async def _verify_session_api_key(api_key: str) -> VerifiedKeyAuth | None:
    """Verify a session API key (``sk-ui-`` prefix) against the session store.

    One DB pass resolves both the enforcement facts and the owning user's
    name, so the caller builds the full identity without re-querying.
    """
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
            if user is None or not user.is_active or user.must_change_password:
                # A temporary/admin-set password must not grant model API
                # access through the session key before it is changed.
                return None

            return VerifiedKeyAuth(
                principal_id=f"session:{session_record.id}",
                # Session keys inherit the owning user's model constraint.
                allowed_models=user.allowed_models,
                # None means all MCP servers allowed (permissive default).
                allowed_mcp_servers=None,
                user_id=user.id,
                # Session keys are subject to the account-level budget too —
                # otherwise it would be trivially bypassable via the UI
                # session key.
                user_budget=BudgetEnvelope.from_orm_fields(user),
                # The UI session key must not bypass the account-level cap.
                user_rate_limit_rpm=user.rate_limit_rpm,
                is_session=True,
                username=user.username,
            )
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


def _verified_auth_from_key_record(
    key_record: CachedApiKey | VerifiedKeyInfo, effective_models: list[str] | None
) -> VerifiedKeyAuth:
    """Build the verified record for a key resolved from a snapshot or cache."""
    return VerifiedKeyAuth(
        principal_id=key_record.name,
        allowed_models=effective_models,
        allowed_mcp_servers=key_record.allowed_mcp_servers,
        user_id=key_record.user_id,
        expires_at=key_record.expires_at,
        budget=key_record.budget,
        user_budget=key_record.user_budget,
        rate_limit_rpm=key_record.rate_limit_rpm,
        user_rate_limit_rpm=key_record.user_rate_limit_rpm,
    )


def _cache_verified_key(
    cache: ApiKeyCache,
    api_key_sha256: str,
    key_record: CachedApiKey,
    effective_models: list[str] | None,
) -> None:
    """Store a freshly (re)verified key in the verified cache.

    The revalidation and full-verification paths below resolve the same
    record shape, so they share one call site instead of duplicating the
    ten-argument invocation.
    """
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
        user_rate_limit_rpm=key_record.user_rate_limit_rpm,
        key_hash=key_record.key_hash,
        must_change_password=key_record.user_must_change_password,
    )


async def _resolve_verified_key(api_key: str, api_key_sha256: str) -> VerifiedKeyAuth | None:
    """Resolve *api_key* from the verified cache, or verify it against the DB.

    Must be called with the key-verification lock held when the verified
    cache is cold, so a TTL rollover cannot start one bcrypt burst per
    in-flight request.
    """
    cache = get_api_key_cache()
    entry = cache.get_verified_key(api_key_sha256, allow_stale=True)

    if entry is not None:
        # Expiry is re-checked on every request: a key may pass its expiry
        # time while its verified-cache entry is still fresh.
        if _is_key_expired(entry.expires_at):
            cache.evict_verified_key(api_key_sha256)
            return None
        if entry.must_change_password:
            return None
        if time.time() - entry.verified_at <= cache.ttl:
            return _verified_auth_from_key_record(entry, entry.allowed_models)
        # else: the verified TTL lapsed — fall through to cheap revalidation.

    cached_keys = await get_cached_api_keys()

    # Cheap revalidation: an entry whose verified-TTL lapsed can be trusted
    # again without bcrypt if the exact bcrypt hash it matched is still
    # present, active, and unexpired in the refreshed snapshot. The key string
    # cannot have changed (same sha256) and the record is the same one, so the
    # prior verification still holds. Without this, every TTL rollover re-ran
    # bcrypt over all keys and stalled the worker.
    if entry is not None and entry.key_hash is not None:
        match = next((k for k in cached_keys if k.key_hash == entry.key_hash), None)
        if (
            match is not None
            and match.is_active
            and match.user_is_active
            and not match.user_must_change_password
            and not _is_key_expired(match.expires_at)
        ):
            effective_models = intersect_model_lists(
                match.allowed_models, match.user_allowed_models
            )
            _cache_verified_key(cache, api_key_sha256, match, effective_models)
            return _verified_auth_from_key_record(match, effective_models)
        # The recorded hash is gone or now invalid (key rotated/deactivated):
        # fall through to a full verification against the current snapshot.

    def _find_matching_key() -> CachedApiKey | None:
        for key_record in cached_keys:
            # Keys owned by a disabled user are rejected alongside inactive
            # keys; expired keys are rejected as well.
            if (
                key_record.is_active
                and key_record.user_is_active
                and not key_record.user_must_change_password
                and not _is_key_expired(key_record.expires_at)
                and key_matches_hash(api_key, key_record.key_hash)
            ):
                return key_record
        return None

    # bcrypt is CPU-bound and deliberately slow: run the scan off the event
    # loop so an unknown key cannot stall every other request on the worker.
    key_record = await asyncio.to_thread(_find_matching_key)
    if key_record is not None:
        # The effective model allowlist is the key's list intersected with
        # the owning user's constraint (user constraint always wins).
        effective_models = intersect_model_lists(
            key_record.allowed_models, key_record.user_allowed_models
        )
        _cache_verified_key(cache, api_key_sha256, key_record, effective_models)
        return _verified_auth_from_key_record(key_record, effective_models)

    session_auth = await _verify_session_api_key(api_key)
    if session_auth is not None:
        return session_auth

    return None


def _consume_verify_task_exception(task: asyncio.Task[VerifiedKeyAuth | None]) -> None:
    """Retrieve a detached verification task's exception.

    The task is shielded so cancellation of the request cannot release the
    verification lock while a worker thread is still scanning. If the request
    was cancelled, nobody awaits the task; retrieving the exception here keeps
    asyncio from logging an unhandled task error.
    """
    if not task.cancelled():
        task.exception()


async def _verify_locked(api_key: str, api_key_sha256: str) -> VerifiedKeyAuth | None:
    """Resolve a key under the process-wide verification lock.

    The negative cache is re-checked *after* acquiring the lock: concurrent
    requests for the same bogus key all pass the lock-free check before the
    first one records the failure, so without this re-check each waiter would
    still run a full bcrypt scan. The failure is recorded before the lock is
    released, making the cache useful to every subsequent waiter.
    """
    cache = get_api_key_cache()
    async with get_key_verification_lock():
        if cache.is_negative_key(api_key_sha256):
            return None
        result = await _resolve_verified_key(api_key, api_key_sha256)
        if result is None:
            cache.set_negative_key(api_key_sha256)
        return result


async def verify_api_key(api_key: str) -> VerifiedKeyAuth | None:
    """Verify an API key string and return the verified key's record.

    Returns ``None`` when the key is invalid, inactive, expired, or belongs to
    a user who must change their password. The returned
    :class:`VerifiedKeyAuth` carries the key's allowlists, owner, expiry,
    budget envelopes, and rate-limit configuration so the caller can enforce
    them.

    Verification is single-flight: a *fresh* verified-cache entry is served
    without locking; everything else (stale, missing, or invalid) is resolved
    under one lock so a burst of concurrent requests cannot multiply the
    bcrypt + DB cost by the concurrency level.

    The locked work runs in a shielded task. A request cancellation therefore
    cannot release the lock while a bcrypt scan continues in its worker
    thread; later requests still serialize behind that scan instead of
    starting overlapping ones.
    """
    if not api_key:
        return None
    api_key_sha256 = hash_api_key_for_cache(api_key)
    cache = get_api_key_cache()

    # Cheap rejection for a key that just failed: an auth flood that repeats
    # the same string must not pay a full bcrypt scan every time.
    if cache.is_negative_key(api_key_sha256):
        return None

    # Lock-free fast path, but only for an entry that is still fresh and
    # usable. Expired and forced-password-change entries fall through to the
    # locked path, which owns the negative-cache bookkeeping.
    entry = cache.get_verified_key(api_key_sha256, allow_stale=True)
    if entry is not None and time.time() - entry.verified_at <= cache.ttl:
        if _is_key_expired(entry.expires_at):
            cache.evict_verified_key(api_key_sha256)
        elif not entry.must_change_password:
            return _verified_auth_from_key_record(entry, entry.allowed_models)

    task = asyncio.ensure_future(_verify_locked(api_key, api_key_sha256))
    task.add_done_callback(_consume_verify_task_exception)
    return await asyncio.shield(task)


# ── Budget ───────────────────────────────────────────────────────────────


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


async def is_key_budget_exceeded(auth: VerifiedKeyAuth) -> bool:
    """Check whether the verified key has reached its current-period budget.

    Queries the usage table for the key's spend since the start of the
    current budget window (UTC calendar boundary, possibly truncated by a
    manual reset). See :func:`_spend_exceeds` for the shared enforcement.
    """
    # Session keys carry no key-level budget configuration.
    if auth.is_session or not auth.principal_id:
        return False
    key_name = auth.principal_id
    # Key names are user-controlled strings, so the cache key is namespaced
    # ("key:") — a key named e.g. "user:7" must not alias the account-level
    # spend entry for user 7, or the two checks would read each other's
    # spend (bypassing a cap or producing a false 429).
    return await _spend_exceeds(
        auth.budget or BudgetEnvelope(),
        cache_key=f"key:{key_name}",
        subject=f"API key '{key_name}'",
        query=lambda repo, since_ts: repo.get_key_spend_since(
            str(key_name), since_ts, user_id=auth.user_id
        ),
    )


async def is_user_budget_exceeded(auth: VerifiedKeyAuth) -> bool:
    """Check whether the key owner's account-level budget has been reached.

    Aggregates the owner's spend across all of their keys over the account
    budget window (UTC calendar boundary, possibly truncated by a manual
    reset). Applies to session keys as well as regular API keys. See
    :func:`_spend_exceeds` for the shared enforcement.
    """
    user_id = auth.user_id
    if user_id is None:
        return False
    # Distinct namespace from the key-level entries ("key:<name>"): the two
    # checks share one spend cache, so their key spaces must be disjoint.
    return await _spend_exceeds(
        auth.user_budget or BudgetEnvelope(),
        cache_key=f"user:{user_id}",
        subject=f"user {user_id}",
        query=lambda repo, since_ts: repo.get_user_spend_since(user_id, since_ts),
    )


class BudgetCheckStatus(Enum):
    """Outcome of a budget check for a verified API key.

    Encodes the states the gate acts on: under the caps (proceed), the key's
    own cap reached (429), the owner account's cap reached (429), or spend
    unconfirmable (fail closed with 503).
    """

    OK = "ok"
    EXCEEDED = "exceeded"
    USER_EXCEEDED = "user_exceeded"
    UNAVAILABLE = "unavailable"


async def check_key_budget(auth: VerifiedKeyAuth) -> BudgetCheckStatus:
    """Check a verified key's budget caps; returns a :class:`BudgetCheckStatus`.

    Checks the key-level cap first, then the owner account's admin-set
    envelope, so the fail-closed behaviour is defined in one place. When the
    current spend cannot be confirmed (stats-DB error), the check reports
    ``UNAVAILABLE`` and the caller must reject with 503 instead of allowing
    unbounded spend.
    """
    try:
        if await is_key_budget_exceeded(auth):
            return BudgetCheckStatus.EXCEEDED
        if await is_user_budget_exceeded(auth):
            return BudgetCheckStatus.USER_EXCEEDED
        return BudgetCheckStatus.OK
    except BudgetCheckUnavailableError:
        return BudgetCheckStatus.UNAVAILABLE


@dataclass(frozen=True)
class BudgetRejection:
    """A budget-check rejection: the HTTP status, error body, and log line."""

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

    Returns a :class:`BudgetRejection` for the states the gate acts on: the
    key's own cap reached (429), the owner account's cap reached (429), or
    spend unconfirmable (fail closed with 503).
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


# ── Rate limit ───────────────────────────────────────────────────────────


async def check_key_rate_limit(auth: VerifiedKeyAuth) -> tuple[int, int] | None:
    """Check the key-level and account-level per-minute caps.

    The per-key cap is admin-only and settable only on the admin's own keys;
    the account cap is admin-managed in the Team view and spans every key the
    member owns (including the UI session key). Both are sliding 60-second
    windows and either can reject.
    Returns ``(limit_rpm, retry_after)`` for the first cap exceeded.
    """
    checks: list[tuple[str, int]] = []
    if auth.rate_limit_rpm is not None:
        checks.append((f"apikey:{auth.principal_id}", auth.rate_limit_rpm))
    if auth.user_rate_limit_rpm is not None and auth.user_id is not None:
        checks.append((f"user:{auth.user_id}", auth.user_rate_limit_rpm))

    limiter = get_rate_limiter()
    for identifier, limit in checks:
        is_limited, limit_meta = await limiter.check_rate_limit(
            identifier=identifier,
            limit=limit,
            window_size=60,
        )
        if is_limited:
            return limit, max(1, int(limit_meta["reset_time"] - time.time()))
    return None


# ── The gate sequence ────────────────────────────────────────────────────


def _extract_key(request: Request, policy: GatePolicy) -> str | None:
    """Read the API key from the request, per the transport's extraction rule.

    Returns ``None`` when no key is present under the transport's rules; the
    caller turns that into the transport's own "no key" failure message.
    """
    auth_header = request.headers.get("Authorization")
    x_api_key = request.headers.get("x-api-key")

    if policy.extraction is KeyExtraction.WEBSOCKET:
        # A raw (non-Bearer) Authorization header is accepted: browser
        # WebSocket clients cannot always set headers.
        if auth_header:
            key = auth_header.removeprefix("Bearer ").strip() or None
            if key:
                return key
        if x_api_key:
            return x_api_key.strip() or None
        if policy.ws_subprotocol_prefix is not None:
            for subprotocol in request.scope.get("subprotocols") or []:
                if subprotocol.startswith(policy.ws_subprotocol_prefix):
                    key = subprotocol[len(policy.ws_subprotocol_prefix) :]
                    if key:
                        return key
        query_key = request.query_params.get("api_key")
        if query_key:
            return query_key.strip() or None
        return None

    if auth_header:
        if not auth_header.startswith("Bearer "):
            return None
        return auth_header[7:]
    if x_api_key:
        return x_api_key
    if policy.allow_session_cookie:
        return request.cookies.get(SESSION_API_KEY_COOKIE) or None
    return None


def _missing_key_message(request: Request, policy: GatePolicy) -> str:
    """The failure message for a request that yielded no key under the policy."""
    if policy.extraction is KeyExtraction.STRICT:
        if request.headers.get("Authorization"):
            return "Invalid authorization header format"
        return "Authorization header missing"
    if policy.extraction is KeyExtraction.BEARER_OR_NONE:
        return "Authorization header missing or invalid format"
    return "Authentication required."


async def _auth_failure(
    request: Request,
    client_ip: str,
    message: str,
    policy: GatePolicy,
) -> GateRejection:
    """Record the auth-failure penalty, log the audit row, build the rejection.

    The body is shaped per client protocol (OpenAI envelope by default,
    Anthropic envelope on /v1/messages) so SDKs parse it natively, matching
    the exception-handler output for downstream errors.
    """
    if policy.auth_failure_penalty:
        get_api_key_lockout_manager().record_failed_attempt(client_ip)
        await apply_auth_failure_delay()
    _log_auth_failure(request, 401, message, client_ip)
    return GateRejection(
        kind=GateRejectionKind.AUTH_FAILED,
        status_code=401,
        body=auth_error_body(message, protocol_for_request(request)),
        message=message,
    )


async def gate_request(request: Request, policy: GatePolicy) -> GateRejection | None:
    """Run the gate sequence for one request; ``None`` means allowed.

    Any client disconnect while the gate is running is recorded as a 499
    before the cancellation propagates: the endpoint never runs for such a
    request, so this is the only place its abandonment can be logged.
    """
    try:
        return await _gate_request_impl(request, policy)
    except asyncio.CancelledError:
        identity = get_request_identity(request)
        _log_rejection(
            request,
            status_code=499,
            error_message="Client disconnected during authentication",
            error_type="client_error",
            resource_id=identity.api_key_name or "unverified",
        )
        raise


async def _gate_request_impl(request: Request, policy: GatePolicy) -> GateRejection | None:
    """Run the gate sequence for one request; ``None`` means allowed.

    The transport declares its :class:`GatePolicy` and renders the returned
    :class:`GateRejection` in its own wire form. Everything else — lockout
    accounting, verification, identity stamping, quota checks, the rejection
    log rows, the last-used claim — is this module's implementation, so the
    sequence and its observability cannot drift between transports.

    On success the request's facts carry the verified identity and the key's
    allowlists; quota rejections are stamped with the key's identity first, so
    the rows are attributable.
    """
    client_ip = get_client_ip(request)
    lockout_manager = get_api_key_lockout_manager()

    if policy.lockout and lockout_manager.is_locked_out(client_ip):
        remaining = lockout_manager.get_lockout_remaining(client_ip)
        message = f"Too many failed authentication attempts. Try again in {remaining} seconds."
        # Deliberately no auth-failure delay: the lockout is itself the brake
        # (ADR-0032), and Retry-After already states the wait. The delay
        # applies only to the missing/invalid-key paths in ``_auth_failure``.
        _log_auth_failure(
            request, 429, "IP locked out due to too many failed auth attempts", client_ip
        )
        return GateRejection(
            kind=GateRejectionKind.LOCKED_OUT,
            status_code=429,
            body=lockout_error_body(remaining),
            message=message,
            headers={"Retry-After": str(remaining)},
        )

    api_key = _extract_key(request, policy)
    if api_key is None:
        return await _auth_failure(
            request, client_ip, _missing_key_message(request, policy), policy
        )

    auth = await verify_api_key(api_key)
    if auth is None:
        return await _auth_failure(request, client_ip, "Invalid API key", policy)

    # Stamp the identity before the quota checks: a rate-limit/budget
    # rejection must still be attributable to the key in its log row. Session
    # keys were verified in the same DB pass that resolved the owning user's
    # name, so their identity is built from the record — no second lookup.
    set_request_identity(
        request,
        RequestIdentity(
            user=auth.username,
            api_key_name=auth.principal_id,
            auth_method="session_api_key" if auth.is_session else "api_key",
            user_id=auth.user_id,
        ),
    )

    if policy.rate_limit:
        rate_limit = await check_key_rate_limit(auth)
        if rate_limit is not None:
            limit_rpm, retry_after = rate_limit
            message = (
                f"Rate limit exceeded for API key '{auth.principal_id}'"
                f" ({limit_rpm} requests/minute)"
            )
            _log_rejection(
                request,
                status_code=429,
                error_message=message,
                error_type="rate_limit_error",
                resource_id=auth.principal_id,
            )
            return GateRejection(
                kind=GateRejectionKind.RATE_LIMITED,
                status_code=429,
                body=rate_limit_exceeded_error_body(limit_rpm, retry_after),
                message=message,
                headers={"Retry-After": str(retry_after)},
            )

    if policy.budget:
        # Budget-limited keys that reached their cap are rejected with 429,
        # as are keys whose owner's account-level budget is exhausted. This
        # is not an auth failure: no lockout penalty, no failure delay. When
        # the budget cannot be checked (stats DB error), enforcement fails
        # closed with 503 instead of silently allowing unbounded spend.
        budget_status = await check_key_budget(auth)
        rejection = budget_status_rejection(
            budget_status,
            principal_id=auth.principal_id,
            user_id=auth.user_id,
        )
        if rejection is not None:
            if rejection.status_code == 503:
                message = "Budget check unavailable; request rejected (fail closed)."
                kind = GateRejectionKind.BUDGET_UNAVAILABLE
            else:
                message = "Budget limit exceeded for this API key or account."
                kind = GateRejectionKind.BUDGET_EXCEEDED
            _log_rejection(
                request,
                status_code=rejection.status_code,
                error_message=rejection.log_message,
                error_type="rate_limit_error" if rejection.status_code == 429 else "api_error",
                resource_id=auth.principal_id,
            )
            return GateRejection(
                kind=kind,
                status_code=rejection.status_code,
                body=rejection.error_body,
                message=message,
            )

    # Claim the throttle slot synchronously so throttled requests never pay
    # asyncio task-creation cost on the hot path; then clear the failed
    # attempts so a successful request forgives the IP's earlier failures.
    if policy.claim_last_used and claim_last_used_update(auth.principal_id):
        asyncio.create_task(_update_key_last_used(auth.principal_id))
    if policy.lockout:
        lockout_manager.clear_failed_attempts(client_ip)

    # The allowlists are facts of the request, read by the model-restriction
    # middleware, the post-routing re-check, and the mounted MCP app.
    # Overwrite unconditionally: a long-lived WebSocket re-runs this gate, and
    # an administrator may have *removed* a restriction since the previous
    # turn. An empty list is a valid (deny-all) restriction and is kept.
    facts_for(request).allowed_models = auth.allowed_models
    facts_for(request).allowed_mcp_servers = auth.allowed_mcp_servers

    return None


def _log_auth_failure(
    request: Request,
    status_code: int,
    error_message: str,
    client_ip: str,
) -> None:
    """Write an audit row for a failed authentication attempt.

    Deliberately not routed through :func:`record_rejection`'s dedupe: failed
    auth is a security signal, and collapsing repeats would hide a
    credential-stuffing attempt. The client IP stands in as the identity
    because no key was verified.
    """
    record_failed_auth(
        request,
        status_code=status_code,
        error_message=error_message,
        client_ip=client_ip,
    )


def _log_rejection(
    request: Request,
    *,
    status_code: int,
    error_message: str,
    error_type: str,
    resource_id: str,
) -> None:
    """Write the body-less rejection row (the pipeline never runs for one)."""
    record_rejection(
        request,
        status_code=status_code,
        error_message=error_message,
        error_type=error_type,
        resource_id=resource_id,
    )
