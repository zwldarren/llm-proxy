"""Server configuration API endpoints."""

from typing import Any

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from llm_proxy.api.dependencies import get_async_session_dep, require_admin_role
from llm_proxy.api.routers.config.helpers import (
    commit_and_reload,
    get_config_repository,
    rebuild_web_search_interceptor,
)
from llm_proxy.api.schemas.admin import (
    CorsConfig,
    KeepaliveConfig,
    LoggingConfigUpdate,
    McpSecurityPolicyConfig,
    ProviderSelectionConfigUpdate,
    RateLimitsConfig,
    RequestPolicyConfig,
    ResilienceConfig,
    SecurityConfig,
    SmartRoutingConfigUpdate,
    WebSearchConfigUpdate,
)
from llm_proxy.config.types.provider_selection import ProviderSelectionConfig
from llm_proxy.config.types.smart_routing import SmartRoutingConfig
from llm_proxy.core.circuit_breaker import CircuitBreakerConfig

router = APIRouter(
    prefix="/server", tags=["configuration"], dependencies=[Depends(require_admin_role)]
)


def _build_request_policy_response(value: dict | None) -> dict:
    """Build a request policy response dict with defaults for missing keys."""
    return RequestPolicyConfig(**(value or {})).model_dump()


# Defaults for the UI-managed ``logging`` server_config key. Missing keys in
# stored (legacy) rows are filled from these defaults on read.
_LOGGING_CONFIG_DEFAULTS: dict[str, Any] = {
    "log_input_output": True,
    "log_retention_days": 30,
    "verbose_routing_logs": False,
    "mask_sensitive_data": True,
    "sampling_rate": 1.0,
    "audit_sampling_rate": None,
    "audit_retention_days": None,
    "sensitive_keys": "",
}


@router.get("/logging")
async def get_logging_config(
    request: Request,
    session: AsyncSession = get_async_session_dep,
) -> dict:
    """Get logging configuration.

    Returns the full logging configuration; missing keys in stored (legacy)
    rows are filled with defaults.
    """
    repo = get_config_repository(session)
    log_config = await repo.get_server_config("logging")
    value = dict(_LOGGING_CONFIG_DEFAULTS)
    if log_config and isinstance(log_config.value, dict):
        value.update(log_config.value)
    return value


@router.put("/logging")
async def update_logging_config(
    config_data: LoggingConfigUpdate,
    request: Request,
    session: AsyncSession = get_async_session_dep,
) -> dict:
    """Update logging configuration.

    Only fields the client explicitly sent are updated (partial update);
    omitted fields keep their stored values. Nullable fields such as
    audit_sampling_rate can be cleared by sending them as null.

    Args:
        config_data: Logging configuration data

    Returns:
        Updated logging configuration
    """
    repo = get_config_repository(session)
    existing_config = await repo.get_server_config("logging")

    # Start with defaults + existing values so omitted fields are preserved
    config_value: dict[str, Any] = dict(_LOGGING_CONFIG_DEFAULTS)
    if existing_config and isinstance(existing_config.value, dict):
        config_value.update(existing_config.value)

    # Override with explicitly provided values only
    config_value.update(config_data.model_dump(exclude_unset=True))

    if existing_config:
        existing_config.value = config_value
        existing_config.description = "Logging configuration"
    else:
        existing_config = await repo.set_server_config(
            key="logging",
            value=config_value,
            description="Logging configuration",
        )

    await session.commit()
    from llm_proxy.api.dependencies import get_config_manager

    await get_config_manager(request).reload()

    return existing_config.value


@router.get("/web-search")
async def get_web_search_config(
    request: Request,
    session: AsyncSession = get_async_session_dep,
) -> dict:
    """Get web search configuration.

    Returns the web search configuration including enabled status,
    provider type, and provider-specific settings.
    """
    repo = get_config_repository(session)
    config = await repo.get_web_search_config()
    if config is None:
        # Return default configuration
        return {
            "enabled": False,
            "provider": "searxng",
            "searxng": None,
            "ollama": None,
        }
    return config


@router.put("/web-search")
async def update_web_search_config(
    config_data: WebSearchConfigUpdate,
    request: Request,
    session: AsyncSession = get_async_session_dep,
) -> dict:
    """Update web search configuration.

    Args:
        config_data: Web search configuration including enabled status,
            provider type, and provider-specific settings.

    Returns:
        Updated web search configuration
    """
    repo = get_config_repository(session)

    # Build config dict from request
    config_value: dict[str, Any] = {
        "enabled": config_data.enabled,
        "provider": config_data.provider,
    }

    if config_data.searxng:
        config_value["searxng"] = config_data.searxng.model_dump()

    if config_data.ollama:
        config_value["ollama"] = config_data.ollama.model_dump()

    await repo.set_web_search_config(config_value)

    # Clear configuration cache
    await commit_and_reload(session, request)

    # Rebuild interceptor so new max_results / URL / auth take effect immediately
    await rebuild_web_search_interceptor(request)

    return config_value


@router.delete("/web-search")
async def delete_web_search_config(
    request: Request,
    session: AsyncSession = get_async_session_dep,
) -> dict:
    """Delete web search configuration.

    This disables web search and removes all configuration.
    """
    repo = get_config_repository(session)
    await repo.delete_server_config("web_search_config")

    await commit_and_reload(session, request)

    await rebuild_web_search_interceptor(request)

    return {"message": "Web search configuration deleted"}


@router.get("/smart-routing")
async def get_smart_routing_config(
    request: Request,
    session: AsyncSession = get_async_session_dep,
) -> dict:
    """Get smart routing configuration.

    Returns the smart routing configuration including enabled status
    and mode weights.
    """
    repo = get_config_repository(session)
    row = await repo.get_server_config("smart_routing")
    cfg = SmartRoutingConfig.from_row(row.value if row else {})
    return cfg.model_dump()


@router.put("/smart-routing")
async def update_smart_routing_config(
    config_data: SmartRoutingConfigUpdate,
    request: Request,
    session: AsyncSession = get_async_session_dep,
) -> dict:
    """Update smart routing configuration.

    Args:
        config_data: Smart routing configuration including enabled status
            and mode weights.

    Returns:
        Updated smart routing configuration
    """
    repo = get_config_repository(session)
    row = await repo.get_server_config("smart_routing")
    current = SmartRoutingConfig.from_row(row.value if row else {})

    if config_data.enabled is not None:
        current.enabled = config_data.enabled
    if config_data.mode_weights is not None:
        current.mode_weights = config_data.mode_weights

    await repo.set_server_config(
        "smart_routing",
        current.to_row(),
        description="Smart routing global configuration",
    )
    await commit_and_reload(session, request)
    # No runtime rebuild is needed for smart routing: the config is read
    # fresh from the database on every request via get_smart_routing_config().

    return current.model_dump()


@router.get("/provider-selection")
async def get_provider_selection_config(
    request: Request,
    session: AsyncSession = get_async_session_dep,
) -> dict:
    """Get the global provider-selection strategy.

    Returns the strategy used to pick among same-priority providers of a
    model (random | session_sticky | cost_optimized | balanced).
    """
    repo = get_config_repository(session)
    row = await repo.get_server_config("provider_selection")
    cfg = ProviderSelectionConfig.from_row(dict(row.value) if row else None)
    return cfg.model_dump()


@router.put("/provider-selection")
async def update_provider_selection_config(
    config_data: ProviderSelectionConfigUpdate,
    request: Request,
    session: AsyncSession = get_async_session_dep,
) -> dict:
    """Update the global provider-selection strategy.

    Args:
        config_data: Provider-selection configuration including the
            strategy used for same-priority providers.

    Returns:
        Updated provider-selection configuration
    """
    repo = get_config_repository(session)
    row = await repo.get_server_config("provider_selection")
    current = ProviderSelectionConfig.from_row(dict(row.value) if row else None)

    if config_data.strategy is not None:
        current.strategy = config_data.strategy

    await repo.set_server_config(
        "provider_selection",
        current.to_row(),
        description="Global provider selection configuration",
    )
    await commit_and_reload(session, request)

    return current.model_dump()


@router.get("/request-policy")
async def get_request_policy_config(
    request: Request,
    session: AsyncSession = get_async_session_dep,
) -> dict:
    """Get global request policy configuration.

    Returns the unknown_fields_policy and unsupported_block_policy settings.
    """
    repo = get_config_repository(session)
    row = await repo.get_server_config("request_policy")
    return _build_request_policy_response(dict(row.value) if row else None)


@router.put("/request-policy")
async def update_request_policy_config(
    config_data: RequestPolicyConfig,
    request: Request,
    session: AsyncSession = get_async_session_dep,
) -> dict:
    """Update global request policy configuration.

    Args:
        config_data: Request policy configuration including
            unknown_fields_policy and unsupported_block_policy.

    Returns:
        Updated request policy configuration
    """
    repo = get_config_repository(session)
    config_value = config_data.model_dump()
    await repo.set_server_config(
        "request_policy",
        config_value,
        description="Global request policy configuration",
    )
    await commit_and_reload(session, request)
    # Re-read from DB to return the persisted state, ensuring any
    # post-commit transformations or computed columns are reflected.
    row = await repo.get_server_config("request_policy")
    return _build_request_policy_response(dict(row.value) if row else None)


def _build_resilience_response(value: dict | None) -> dict:
    """Build a resilience config response dict with defaults for missing keys."""
    return ResilienceConfig(**(value or {})).model_dump()


@router.get("/resilience")
async def get_resilience_config(
    request: Request,
    session: AsyncSession = get_async_session_dep,
) -> dict:
    """Get global resilience configuration (retry + fallback + circuit breaker)."""
    repo = get_config_repository(session)
    row = await repo.get_server_config("resilience")
    return _build_resilience_response(dict(row.value) if row else None)


@router.put("/resilience")
async def update_resilience_config(
    config_data: ResilienceConfig,
    request: Request,
    session: AsyncSession = get_async_session_dep,
) -> dict:
    """Update global resilience configuration.

    Args:
        config_data: Resilience configuration including
            max_retries, max_fallback_attempts, and circuit_breaker settings.

    Returns:
        Updated resilience configuration
    """
    repo = get_config_repository(session)
    config_value = config_data.model_dump()
    await repo.set_server_config(
        "resilience",
        config_value,
        description="Global resilience configuration",
    )
    await commit_and_reload(session, request)

    # Also update the circuit breaker store at runtime
    circuit_breaker = getattr(request.app.state, "circuit_breaker", None)
    if circuit_breaker is not None:
        cb = config_data.circuit_breaker
        circuit_breaker.update_config(
            CircuitBreakerConfig(
                enabled=cb.enabled,
                failure_threshold=cb.failure_threshold,
                cooldown_seconds=cb.cooldown_seconds,
            )
        )

    # Re-read from DB to return the persisted state
    row = await repo.get_server_config("resilience")
    return _build_resilience_response(dict(row.value) if row else None)


def _build_security_response(value: dict | None) -> dict:
    """Build a security config response dict with defaults for missing keys."""
    return SecurityConfig(**(value or {})).model_dump()


@router.get("/security")
async def get_security_config(
    request: Request,
    session: AsyncSession = get_async_session_dep,
) -> dict:
    """Get security / rate-limiting configuration (lockout, HSTS, body limit)."""
    repo = get_config_repository(session)
    row = await repo.get_server_config("security")
    return _build_security_response(dict(row.value) if row else None)


@router.put("/security")
async def update_security_config(
    config_data: SecurityConfig,
    request: Request,
    session: AsyncSession = get_async_session_dep,
) -> dict:
    """Update security / rate-limiting configuration.

    Changes are hot-reloaded: middleware reads them from the config manager's
    cached ProxyConfig on every request.

    Args:
        config_data: Security configuration including lockout policy, HSTS,
            request body size limit, and rate-limit switches.

    Returns:
        Updated security configuration
    """
    repo = get_config_repository(session)
    config_value = config_data.model_dump()
    await repo.set_server_config(
        "security",
        config_value,
        description="Security and rate-limiting configuration",
    )
    await commit_and_reload(session, request)

    # Re-read from DB to return the persisted state
    row = await repo.get_server_config("security")
    return _build_security_response(dict(row.value) if row else None)


def _build_keepalive_response(value: dict | None) -> dict:
    """Build a keepalive config response dict with defaults for missing keys."""
    return KeepaliveConfig(**(value or {})).model_dump()


@router.get("/keepalive")
async def get_keepalive_config(
    request: Request,
    session: AsyncSession = get_async_session_dep,
) -> dict:
    """Get non-streaming response keepalive (heartbeat) configuration."""
    repo = get_config_repository(session)
    row = await repo.get_server_config("keepalive")
    return _build_keepalive_response(dict(row.value) if row else None)


@router.put("/keepalive")
async def update_keepalive_config(
    config_data: KeepaliveConfig,
    request: Request,
    session: AsyncSession = get_async_session_dep,
) -> dict:
    """Update non-streaming response keepalive configuration.

    Changes are hot-reloaded: the protocol router reads them from the config
    manager's cached ProxyConfig on every request.

    Args:
        config_data: Keepalive configuration including enabled flag, grace
            period, and heartbeat interval.

    Returns:
        Updated keepalive configuration
    """
    repo = get_config_repository(session)
    config_value = config_data.model_dump()
    await repo.set_server_config(
        "keepalive",
        config_value,
        description="Non-streaming response keepalive configuration",
    )
    await commit_and_reload(session, request)

    # Re-read from DB to return the persisted state
    row = await repo.get_server_config("keepalive")
    return _build_keepalive_response(dict(row.value) if row else None)


def _build_rate_limits_response(overrides: dict | None) -> dict:
    """Build a rate limits response merging stored overrides with code defaults."""
    from llm_proxy.api.middleware.rate_limiting import DEFAULT_RATE_LIMITS

    merged = dict(DEFAULT_RATE_LIMITS)
    if overrides:
        merged.update({k: str(v) for k, v in overrides.items() if k in DEFAULT_RATE_LIMITS})
    return {"limits": merged}


@router.get("/rate-limits")
async def get_rate_limits_config(
    request: Request,
    session: AsyncSession = get_async_session_dep,
) -> dict:
    """Get per-bucket rate limits (code defaults merged with stored overrides)."""
    repo = get_config_repository(session)
    row = await repo.get_server_config("rate_limits")
    overrides = dict(row.value) if row and isinstance(row.value, dict) else None
    return _build_rate_limits_response(overrides)


@router.put("/rate-limits")
async def update_rate_limits_config(
    config_data: RateLimitsConfig,
    request: Request,
    session: AsyncSession = get_async_session_dep,
) -> dict:
    """Replace per-bucket rate limit overrides.

    Only known buckets are accepted; values must be valid "N/period" specs
    (e.g. "5/minute", "100/hour"). Buckets not included keep their code
    defaults. Changes are hot-reloaded.

    Args:
        config_data: Rate limit overrides keyed by bucket name.

    Returns:
        Effective rate limits (defaults merged with the new overrides)
    """
    from fastapi import HTTPException

    from llm_proxy.api.middleware.rate_limiting import DEFAULT_RATE_LIMITS, _parse_limit_value

    unknown = sorted(set(config_data.limits) - set(DEFAULT_RATE_LIMITS))
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown rate limit buckets: {', '.join(unknown)}. "
            f"Known buckets: {', '.join(sorted(DEFAULT_RATE_LIMITS))}",
        )
    for bucket, value in config_data.limits.items():
        try:
            _parse_limit_value(value)
        except (ValueError, AttributeError) as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid rate limit value for bucket '{bucket}': {value!r} ({exc})",
            ) from exc

    repo = get_config_repository(session)
    await repo.set_server_config(
        "rate_limits",
        config_data.limits,
        description="Per-bucket rate limit overrides",
    )
    await commit_and_reload(session, request)
    return _build_rate_limits_response(config_data.limits)


def _build_cors_response(value: list | None) -> dict:
    """Build a CORS config response dict from the stored origins list."""
    return CorsConfig(origins=value or []).model_dump()


@router.get("/cors")
async def get_cors_config(
    request: Request,
    session: AsyncSession = get_async_session_dep,
) -> dict:
    """Get allowed CORS origins (empty list means CORS is disabled)."""
    repo = get_config_repository(session)
    row = await repo.get_server_config("cors_origins")
    return _build_cors_response(list(row.value) if row and isinstance(row.value, list) else None)


@router.put("/cors")
async def update_cors_config(
    config_data: CorsConfig,
    request: Request,
    session: AsyncSession = get_async_session_dep,
) -> dict:
    """Replace the allowed CORS origins.

    Changes are hot-reloaded: the CORS middleware reads the origins from the
    config manager's cached ProxyConfig on every request.

    Args:
        config_data: CORS configuration with the full list of allowed origins.

    Returns:
        Updated CORS configuration
    """
    origins = [o.strip() for o in config_data.origins if o.strip()]
    repo = get_config_repository(session)
    await repo.set_server_config(
        "cors_origins",
        origins,
        description="Allowed CORS origins",
    )
    await commit_and_reload(session, request)
    return _build_cors_response(origins)


@router.get("/mcp-security")
async def get_mcp_security_config(
    request: Request,
    session: AsyncSession = get_async_session_dep,
) -> dict:
    """Get MCP security policy configuration.

    Returns the list-based policy fields (allowed/blocked commands,
    env keys, URL rules) stored in the database.
    """
    repo = get_config_repository(session)
    config = await repo.get_server_config("mcp_security_policy")
    if config is None:
        return McpSecurityPolicyConfig().model_dump()
    # Merge with defaults so partial/legacy DB records don't return
    # responses with missing keys.
    return McpSecurityPolicyConfig(**(config.value or {})).model_dump()


@router.put("/mcp-security")
async def update_mcp_security_config(
    config_data: McpSecurityPolicyConfig,
    request: Request,
    session: AsyncSession = get_async_session_dep,
) -> dict:
    """Update MCP security policy configuration.

    Args:
        config_data: MCP security policy including allowed/blocked commands,
            env keys, and URL rules.

    Returns:
        Updated MCP security policy configuration
    """
    repo = get_config_repository(session)
    config_value = config_data.model_dump()
    await repo.set_server_config(
        "mcp_security_policy",
        config_value,
        description="MCP security policy configuration",
    )
    await commit_and_reload(session, request)
    return config_value
