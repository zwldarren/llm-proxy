"""Rate limiting middleware with support for multiple backends (memory and Redis)."""

import asyncio
import functools
import importlib.util
import inspect
import threading
import time
from collections.abc import Callable
from typing import Any

from fastapi import Request
from slowapi.errors import RateLimitExceeded

from llm_proxy.config.settings import get_settings
from llm_proxy.core.request_utils import get_client_ip
from llm_proxy.database.redis_client import RedisClient, get_redis_client_async
from llm_proxy.observability.logger import get_logger

_signature_cache: dict[int, tuple[list[str], int | None]] = {}


async def _get_cached_signature(func: Callable) -> tuple[list[str], int | None]:
    """Get cached function signature info.

    Returns tuple of (parameter_names, request_param_index).
    """
    func_id = id(func)

    if func_id not in _signature_cache:
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        request_idx: int | None = None
        for i, param_name in enumerate(params):
            if param_name == "request":
                request_idx = i
                break

        _signature_cache[func_id] = (params, request_idx)

    return _signature_cache[func_id]


class CustomRateLimitExceeded(RateLimitExceeded):
    """Custom rate limit exception with additional attributes."""

    def __init__(self, limit_obj: Any, retry_after: int | None = None, detail: str | None = None):
        super().__init__(limit_obj)
        self.retry_after = retry_after
        if detail:
            self.detail = detail


logger = get_logger(__name__)

_PERIOD_MAP: dict[str, int] = {
    "second": 1,
    "minute": 60,
    "hour": 3600,
    "day": 86400,
}

# Named rate-limit buckets with their code defaults. Decorators reference
# buckets by name (e.g. ``@limiter.limit("auth.login")``); the effective value
# is resolved per request so UI-managed overrides (server_config
# ``rate_limits`` key) are hot-reloaded.
DEFAULT_RATE_LIMITS: dict[str, str] = {
    "auth.login": "5/minute",
    "auth.setup": "5/minute",
    "auth.setup_status": "10/minute",
}


def _parse_limit_value(limit_value: str) -> tuple[int, int]:
    """Parse a "N/period" rate-limit spec into (limit, window_size_seconds)."""
    limit_str, period_str = limit_value.split("/", 1)
    limit = int(limit_str)

    window_size = _PERIOD_MAP.get(period_str)
    if window_size is None:
        if period_str.endswith("second"):
            window_size = int(period_str[:-6])
        elif period_str.endswith("minute"):
            window_size = int(period_str[:-6]) * 60
        elif period_str.endswith("hour"):
            window_size = int(period_str[:-4]) * 3600
        else:
            raise ValueError(f"Invalid rate limit period: {period_str}")
    return limit, window_size


def resolve_rate_limit_value(request: Request, bucket: str) -> str:
    """Resolve the effective "N/period" value for a rate-limit bucket.

    Reads UI-managed overrides from the config manager's cached ProxyConfig
    (server_config ``rate_limits`` key); falls back to the bucket's code
    default when no override exists or the override is invalid.
    """
    from llm_proxy.config.types import ProxyConfig

    default = DEFAULT_RATE_LIMITS[bucket]
    config_manager = getattr(request.app.state, "config_manager", None)
    cached = config_manager.get_cached_config() if config_manager is not None else None
    if not isinstance(cached, ProxyConfig):
        return default

    override = cached.server_params.rate_limits.get(bucket)
    if override is None:
        return default
    try:
        _parse_limit_value(override)
    except ValueError, AttributeError:
        logger.warning(
            f"Invalid rate limit override for bucket '{bucket}': {override!r}; "
            f"using default {default}"
        )
        return default
    return override


class RedisRateLimiter:
    """Redis-based sliding window rate limiter.

    Uses Redis sorted sets to implement sliding window rate limiting.

    Algorithm:
    1. For each key (client IP), maintain a sorted set of timestamps
    2. When a request comes in:
        - Remove timestamps older than window_size seconds
        - Count remaining timestamps
        - If count < limit, add current timestamp and allow request
        - Otherwise, deny request
    3. Clean up old data automatically with Redis TTL

    Example Redis commands:
        ZREMRANGEBYSCORE key -inf (current_time - window_size)
        ZCARD key
        ZADD key current_time current_time
        EXPIRE key window_size
    """

    def __init__(
        self,
        redis_client: RedisClient | None = None,
        prefix: str = "rate_limit:",
        window_size: int = 60,
    ):
        self._redis_client = redis_client
        self.prefix = prefix
        self.window_size = window_size

    def _get_key(self, identifier: str) -> str:
        return f"{self.prefix}{identifier}"

    async def is_rate_limited(
        self,
        identifier: str,
        limit: int,
        window_size: int | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        if window_size is None:
            window_size = self.window_size

        redis_client = await get_redis_client_async(self._redis_client)
        key = self._get_key(identifier)
        current_time = time.time()

        async with redis_client.pipeline(transaction=True) as pipe:
            pipe.zremrangebyscore(key, "-inf", current_time - window_size)
            pipe.zcard(key)
            results = await pipe.execute()

        remaining_count = results[1]

        if remaining_count < limit:
            async with redis_client.pipeline(transaction=True) as pipe:
                pipe.zadd(key, {str(current_time): current_time})
                pipe.expire(key, window_size)
                await pipe.execute()

            remaining = limit - remaining_count - 1
            reset_time = current_time + window_size

            return False, {
                "remaining": max(0, remaining),
                "reset_time": reset_time,
                "limit": limit,
            }

        oldest_timestamps = await redis_client.zrange(key, 0, 0, withscores=True)
        if oldest_timestamps:
            oldest_timestamp = float(oldest_timestamps[0][1])
            reset_time = oldest_timestamp + window_size
        else:
            reset_time = current_time + window_size

        return True, {
            "remaining": 0,
            "reset_time": reset_time,
            "limit": limit,
        }


_redis_rate_limiter: RedisRateLimiter | None = None


def get_redis_rate_limiter(
    redis_client: RedisClient | None = None,
    prefix: str | None = None,
    window_size: int | None = None,
) -> RedisRateLimiter:
    global _redis_rate_limiter

    if _redis_rate_limiter is None:
        if prefix is None:
            prefix = "rate_limit:"

        if window_size is None:
            window_size = 60

        _redis_rate_limiter = RedisRateLimiter(
            redis_client=redis_client,
            prefix=prefix,
            window_size=window_size,
        )

    return _redis_rate_limiter


class RateLimitManager:
    """Manages rate limiting with configurable backend (Redis or in-memory)."""

    def __init__(self, use_redis: bool | None = None):
        if use_redis is None:
            settings = get_settings()
            redis_cfg = settings.redis
            use_redis = (
                redis_cfg.enabled
                and redis_cfg.rate_limit_enabled
                and importlib.util.find_spec("redis") is not None
            )
        self._use_redis = use_redis
        self._memory_windows: dict[str, list[float]] = {}
        self._memory_lock_obj: asyncio.Lock | None = None
        self._memory_lock_loop: asyncio.AbstractEventLoop | None = None

    async def check_rate_limit(
        self,
        identifier: str,
        limit: int,
        window_size: int,
    ) -> tuple[bool, dict[str, Any]]:
        """Check (and record) one request against a sliding-window limit.

        Public entry point for middleware-style callers that manage their own
        identifiers (e.g. per-API-key limits), as opposed to the decorator API
        which manages route buckets by client IP.
        """
        from llm_proxy.api.middleware.security import get_security_params

        security = get_security_params()
        if security.rate_limit_disabled:
            return False, {"remaining": limit, "reset_time": 0, "limit": limit}

        if self._use_redis:
            try:
                redis_limiter = get_redis_rate_limiter()
                return await redis_limiter.is_rate_limited(identifier, limit, window_size)
            except Exception as e:
                logger.error(f"Redis rate limiter failed: {e}", exc_info=True)
                if security.redis_rate_limit_fail_closed:
                    logger.warning("Rate limiting failing closed (blocking request)")
                    return True, {
                        "remaining": 0,
                        "reset_time": time.time() + window_size,
                        "limit": limit,
                    }
                else:
                    logger.warning("Rate limiting failing open (allowing request)")
                    return False, {"remaining": limit, "reset_time": 0, "limit": limit}

        # In-memory sliding window rate limiter.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        # Thread-safe lazy initialisation: use a threading lock to prevent
        # two coroutines from creating separate lock objects on different
        # event loops when they both observe _memory_lock_obj is None.
        if self._memory_lock_obj is None or self._memory_lock_loop != loop:
            self._memory_lock_obj = asyncio.Lock()
            self._memory_lock_loop = loop

        current_time = time.time()
        async with self._memory_lock_obj:
            window = self._memory_windows.get(identifier, [])
            # Drop timestamps outside the current window.
            window = [t for t in window if current_time - t < window_size]
            request_count = len(window)

            if request_count < limit:
                window.append(current_time)
                self._memory_windows[identifier] = window
                remaining = limit - request_count - 1
                reset_time = current_time + window_size
                return False, {
                    "remaining": max(0, remaining),
                    "reset_time": reset_time,
                    "limit": limit,
                }

            # Clean up idle identifiers to avoid unbounded memory growth.
            if not window:
                self._memory_windows.pop(identifier, None)

            oldest_timestamp = window[0] if window else current_time
            reset_time = oldest_timestamp + window_size
            return True, {
                "remaining": 0,
                "reset_time": reset_time,
                "limit": limit,
            }

    def create_decorator(
        self,
        limit_value: str,
        key_func: Callable[[Request], str] | None = None,
    ) -> Callable:
        # A value containing "/" is a literal "N/period" spec parsed once at
        # decoration time. Otherwise it names a bucket in DEFAULT_RATE_LIMITS
        # whose effective value is resolved per request (DB overrides via the
        # UI-managed ``rate_limits`` server_config key are hot-reloaded).
        is_bucket = "/" not in limit_value
        if not is_bucket:
            limit, window_size = _parse_limit_value(limit_value)
        elif limit_value not in DEFAULT_RATE_LIMITS:
            raise ValueError(f"Unknown rate limit bucket: {limit_value}")

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                params, request_idx = await _get_cached_signature(func)
                request: Request | None = None

                if "request" in kwargs:
                    request = kwargs["request"]
                elif request_idx is not None and request_idx < len(args):
                    request = args[request_idx]

                if request is None:
                    return await func(*args, **kwargs)

                if is_bucket:
                    eff_value = resolve_rate_limit_value(request, limit_value)
                    eff_limit, eff_window = _parse_limit_value(eff_value)
                else:
                    eff_value, eff_limit, eff_window = limit_value, limit, window_size

                identifier = key_func(request) if key_func else get_client_ip(request)

                is_limited, metadata = await self.check_rate_limit(
                    identifier=identifier,
                    limit=eff_limit,
                    window_size=eff_window,
                )

                if is_limited:
                    limit_obj = type(
                        "Limit",
                        (),
                        {
                            "limit": eff_value,
                            "scope": identifier,
                            "is_exempt": False,
                            "error_message": None,
                        },
                    )()

                    retry_after = int(metadata["reset_time"] - time.time())
                    exc = CustomRateLimitExceeded(
                        limit_obj,
                        retry_after=retry_after,
                        detail=f"Rate limit exceeded. Try again in {retry_after} seconds.",
                    )
                    raise exc

                return await func(*args, **kwargs)

            return wrapper

        return decorator

    def limit(
        self,
        limit_value: str,
        key_func: Callable[[Request], str] | None = None,
    ) -> Callable:
        """Decorator entry point for rate-limited routes."""
        return self.create_decorator(limit_value, key_func)


_rate_limiter: RateLimitManager | None = None
_rate_limiter_lock = threading.Lock()


def get_rate_limiter() -> RateLimitManager:
    """Get global rate limiter instance, creating it if necessary."""
    global _rate_limiter
    if _rate_limiter is None:
        with _rate_limiter_lock:
            if _rate_limiter is None:
                settings = get_settings()
                redis_cfg = settings.redis
                use_redis = redis_cfg.enabled and redis_cfg.rate_limit_enabled
                if use_redis and importlib.util.find_spec("redis") is not None:
                    logger.debug("Initialized Redis rate limiter")
                    _rate_limiter = RateLimitManager(use_redis=True)
                else:
                    logger.debug("Initialized memory rate limiter")
                    _rate_limiter = RateLimitManager(use_redis=False)
    assert _rate_limiter is not None
    return _rate_limiter
