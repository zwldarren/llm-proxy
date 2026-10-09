"""Per-bucket rate-limit grammar and the canonical server-settings model.

The grammar — bucket names, their code defaults, and "N/period" spec parsing —
is the single source shared by the rate-limiting middleware, the admin
server-settings endpoint, and the config manager's runtime read. It lives in
the config package so neither the API layer nor the middleware owns another
module's data (see ADR-0026).
"""

from pydantic import BaseModel, Field

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


def parse_limit_value(limit_value: str) -> tuple[int, int]:
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


def validate_rate_limit_overrides(limits: dict[str, str]) -> None:
    """Validate a bucket-override map, raising ValueError with an operator-facing message.

    Used by the admin server-settings endpoint before persisting overrides;
    the caller translates the error into its wire format (HTTP 422).
    """
    unknown = sorted(set(limits) - set(DEFAULT_RATE_LIMITS))
    if unknown:
        raise ValueError(
            f"Unknown rate limit buckets: {', '.join(unknown)}. "
            f"Known buckets: {', '.join(sorted(DEFAULT_RATE_LIMITS))}"
        )
    for bucket, value in limits.items():
        try:
            parse_limit_value(value)
        except (ValueError, AttributeError) as exc:
            raise ValueError(
                f"Invalid rate limit value for bucket '{bucket}': {value!r} ({exc})"
            ) from exc


class RateLimitsConfig(BaseModel):
    """Per-bucket rate limit overrides (server_config ``rate_limits`` key).

    Keys are bucket names (see DEFAULT_RATE_LIMITS); values are "N/period"
    specs, e.g. "5/minute". Only the overrides are stored; the effective
    limits are the code defaults merged with these.
    """

    limits: dict[str, str] = Field(
        default_factory=dict,
        description="Bucket name → 'N/period' rate limit spec",
    )
