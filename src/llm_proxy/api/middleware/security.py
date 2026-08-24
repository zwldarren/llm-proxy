"""Security middleware for rate limiting and security headers.

This module provides:
- Rate limiting to prevent brute force and DoS attacks
- Security headers to protect against common web vulnerabilities
- Account lockout mechanism for failed login attempts
"""

import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from threading import Lock
from typing import TYPE_CHECKING

from fastapi import Request, Response
from starlette.responses import JSONResponse

from llm_proxy.config.types.server import SecurityParams
from llm_proxy.core.constants import LOCKOUT_CLEANUP_INTERVAL_SECONDS
from llm_proxy.observability.logger import get_logger

if TYPE_CHECKING:
    from llm_proxy.config.manager import DatabaseConfigManager

logger = get_logger(__name__)

# Config manager registered at startup so the global lockout managers (which
# are created lazily outside request context) can resolve the current
# UI-managed security parameters. Falls back to code defaults in tests.
_config_manager: DatabaseConfigManager | None = None


def set_security_config_manager(config_manager: DatabaseConfigManager | None) -> None:
    """Register the config manager used to resolve security parameters."""
    global _config_manager
    _config_manager = config_manager


def get_security_params() -> SecurityParams:
    """Resolve the effective security parameters (hot-reloaded from the DB)."""
    from llm_proxy.config.manager import resolve_security_params

    return resolve_security_params(_config_manager)


class BaseLockoutManager:
    """Base class for managing lockout state for failed authentication attempts.

    Thread-safe with background cleanup to avoid per-request cleanup overhead.

    ``max_attempts``/``lockout_duration`` are resolved dynamically when a
    ``params_getter`` is provided (the production singletons use this so
    UI-managed changes apply immediately); otherwise the static constructor
    arguments / class defaults are used.
    """

    default_max_attempts: int = 5
    default_lockout_duration: int = 900
    cleanup_interval: int = LOCKOUT_CLEANUP_INTERVAL_SECONDS
    log_prefix: str = "Lockout"

    def __init__(
        self,
        max_attempts: int | None = None,
        lockout_duration: int | None = None,
        params_getter: Callable[[], tuple[int, int]] | None = None,
    ):
        self._static_max_attempts = max_attempts
        self._static_lockout_duration = lockout_duration
        self._params_getter = params_getter
        self._failed_attempts: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()
        self._last_cleanup = time.time()
        self._cleanup_counter = 0

    @property
    def max_attempts(self) -> int:
        if self._params_getter is not None:
            return self._params_getter()[0]
        if self._static_max_attempts is not None:
            return self._static_max_attempts
        return self.default_max_attempts

    @property
    def lockout_duration(self) -> int:
        if self._params_getter is not None:
            return self._params_getter()[1]
        if self._static_lockout_duration is not None:
            return self._static_lockout_duration
        return self.default_lockout_duration

    def _should_cleanup(self) -> bool:
        """Check if it's time for periodic cleanup."""
        self._cleanup_counter += 1
        return (
            self._cleanup_counter % 100 == 0
            and time.time() - self._last_cleanup > self.cleanup_interval
        )

    def _cleanup_expired(self) -> None:
        """Remove expired entries from all identifiers."""
        current_time = time.time()
        to_remove = []
        for identifier, attempts in list(self._failed_attempts.items()):
            valid_attempts = [t for t in attempts if current_time - t < self.lockout_duration]
            if not valid_attempts:
                to_remove.append(identifier)
            elif len(valid_attempts) != len(attempts):
                self._failed_attempts[identifier] = valid_attempts
        for identifier in to_remove:
            del self._failed_attempts[identifier]
        self._last_cleanup = current_time
        if to_remove:
            prefix = self.log_prefix.lower()
            logger.debug(f"Cleaned up {len(to_remove)} expired {prefix} lockout entries")

    def record_failed_attempt(self, identifier: str) -> None:
        """Record a failed authentication attempt for the given identifier."""
        with self._lock:
            if self._should_cleanup():
                self._cleanup_expired()
            current_time = time.time()
            self._failed_attempts[identifier] = [
                t
                for t in self._failed_attempts[identifier]
                if current_time - t < self.lockout_duration
            ]
            self._failed_attempts[identifier].append(current_time)
            logger.warning(
                f"Failed {self.log_prefix.lower()} attempt for {identifier}. "
                f"Count: {len(self._failed_attempts[identifier])}/{self.max_attempts}"
            )

    def is_locked_out(self, identifier: str) -> bool:
        """Check if the identifier is currently locked out"""
        with self._lock:
            current_time = time.time()
            attempts = self._failed_attempts.get(identifier, [])
            valid_count = sum(1 for t in attempts if current_time - t < self.lockout_duration)
            if valid_count != len(attempts):
                self._failed_attempts[identifier] = [
                    t for t in attempts if current_time - t < self.lockout_duration
                ]
            return valid_count >= self.max_attempts

    def get_lockout_remaining(self, identifier: str) -> int:
        """Get remaining lockout time in seconds."""
        with self._lock:
            if not self._failed_attempts[identifier]:
                return 0
            oldest_attempt = min(self._failed_attempts[identifier])
            return max(0, int(self.lockout_duration - (time.time() - oldest_attempt)))

    def clear_failed_attempts(self, identifier: str) -> None:
        """Clear failed attempts after successful authentication."""
        with self._lock:
            self._failed_attempts.pop(identifier, None)


class AccountLockoutManager(BaseLockoutManager):
    """Manages account lockout state for failed login attempts."""

    def __init__(self):
        super().__init__(
            params_getter=lambda: (
                get_security_params().max_failed_login_attempts,
                get_security_params().lockout_duration_seconds,
            )
        )
        self.log_prefix = "Login"


_lockout_manager: AccountLockoutManager | None = None
_lockout_manager_lock = Lock()


def get_lockout_manager() -> AccountLockoutManager:
    """Get the global account lockout manager instance for login attempts."""
    global _lockout_manager
    if _lockout_manager is None:
        with _lockout_manager_lock:
            if _lockout_manager is None:
                _lockout_manager = AccountLockoutManager()
    assert _lockout_manager is not None
    return _lockout_manager


class APIKeyLockoutManager(BaseLockoutManager):
    """Manages lockout state for failed API key authentication attempts."""

    def __init__(self):
        super().__init__(
            params_getter=lambda: (
                get_security_params().max_failed_api_key_attempts,
                get_security_params().api_key_lockout_duration_seconds,
            )
        )
        self.log_prefix = "API key"


_api_key_lockout_manager: APIKeyLockoutManager | None = None
_api_key_lockout_manager_lock = Lock()


def get_api_key_lockout_manager() -> APIKeyLockoutManager:
    """Get the global API key lockout manager instance."""
    global _api_key_lockout_manager
    if _api_key_lockout_manager is None:
        with _api_key_lockout_manager_lock:
            if _api_key_lockout_manager is None:
                _api_key_lockout_manager = APIKeyLockoutManager()
    assert _api_key_lockout_manager is not None
    return _api_key_lockout_manager


_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
    "Content-Security-Policy": (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "font-src 'self' data:; "
        "connect-src 'self'; "
        "frame-ancestors 'none';"
    ),
}


def build_security_headers(config_manager: DatabaseConfigManager | None = None) -> dict[str, str]:
    """Build the security header set, including HSTS when enabled.

    Shared by the security-headers middleware and by pure-ASGI layers that
    short-circuit before the middleware stack (MCP proxy auth responses) so
    every response carries the same header set.
    """
    headers = dict(_SECURITY_HEADERS)
    from llm_proxy.config.manager import resolve_security_params

    params = resolve_security_params(config_manager)
    if params.hsts_enabled:
        headers["Strict-Transport-Security"] = f"max-age={params.hsts_max_age}; includeSubDomains"
    return headers


async def security_headers_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Add security headers to all responses."""
    response = await call_next(request)

    for header, value in build_security_headers(
        getattr(request.app.state, "config_manager", None)
    ).items():
        response.headers[header] = value

    return response


async def rate_limit_exceeded_handler(
    request: Request,
    exc: Exception,
) -> Response:
    """Handle rate limit exceeded errors with a proper JSON response."""
    retry_after = getattr(exc, "retry_after", None)
    detail = getattr(exc, "detail", "Too many requests")

    return JSONResponse(
        status_code=429,
        content={
            "error": {
                "message": f"Rate limit exceeded: {detail}",
                "type": "rate_limit_error",
                "code": "rate_limit_exceeded",
            }
        },
        headers={"Retry-After": str(retry_after) if retry_after else "60"},
    )
