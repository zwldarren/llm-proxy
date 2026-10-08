"""Server settings registry — one descriptor per UI-managed server_config key.

Each row of ``SERVER_SETTINGS`` declares everything one keyed setting needs:
its ``server_config`` row key, its admin route suffix, its canonical model
(the runtime truth, living in ``llm_proxy.config.types``), how writes merge,
how reads respond, and whether it feeds the runtime config at all. Three
consumers are generated from this single table:

- the admin endpoints — ``api/routers/config/server.py`` registers its routes
  from the registry, so the ~10 copied GET/PUT handler pairs collapse;
- the runtime read — :func:`load_server_settings` builds the config manager's
  typed snapshot from the same rows, so stored values cannot drift from what
  the endpoints persist;
- the contract test — the test suite walks the table key by key.

Adding a UI-managed setting is adding a row here; the endpoint, the runtime
read, and its test coverage come with it.

The write and read modes are deliberately limited to the variance that exists
today — no speculative extension points (ADR-0025).
"""

import logging
from collections.abc import Callable, Mapping
from copy import copy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, create_model

from llm_proxy.config.types.logging_config import (
    DEFAULT_MAX_LOGGED_BODY_BYTES,
    DEFAULT_RETENTION_DAYS,
    LoggingConfig,
    LoggingWireConfig,
)
from llm_proxy.config.types.mcp_security import McpSecurityPolicyConfig
from llm_proxy.config.types.provider_selection import ProviderSelectionConfig
from llm_proxy.config.types.rate_limits import (
    DEFAULT_RATE_LIMITS,
    RateLimitsConfig,
    validate_rate_limit_overrides,
)
from llm_proxy.config.types.server import (
    CircuitBreakerParams,
    CorsConfig,
    KeepaliveParams,
    RequestPolicyConfig,
    ResilienceConfig,
    SecurityParams,
)
from llm_proxy.config.types.smart_routing import SmartRoutingConfig
from llm_proxy.config.types.web_search import WebSearchConfig
from llm_proxy.core.constants import DEFAULT_MAX_FALLBACK_ATTEMPTS, DEFAULT_MAX_RETRIES
from llm_proxy.core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

__all__ = [
    "SERVER_SETTINGS",
    "ReadMode",
    "ServerSetting",
    "ServerSettingsRuntime",
    "WriteMode",
    "apply_write",
    "build_response",
    "load_logging_config",
    "load_server_settings",
    "request_model_for",
]


class WriteMode(Enum):
    """How a PUT stores the validated request body.

    Only the merge semantics the current endpoints actually have; a new shape
    is a new enum member with one implementation in :func:`apply_write`.
    """

    #: Full replace: the stored row is the canonical model's dump. Used by the
    #: keys whose request body is the full model (request_policy, resilience,
    #: security, keepalive, mcp_security_policy).
    REPLACE = "replace"
    #: Replace, dropping unset/None nested blocks: the stored row carries the
    #: always-defaulted top-level fields and only the sub-configs the client
    #: sent (web_search).
    REPLACE_EXCLUDE_NONE = "replace_exclude_none"
    #: Replace with the inner dict of a wrapper model (rate_limits stores the
    #: ``limits`` map, not the wrapper).
    REPLACE_INNER = "replace_inner"
    #: Replace with a stripped bare list (cors_origins stores a list, not a
    #: dict).
    REPLACE_LIST = "replace_list"
    #: Partial merge over the existing row: explicit non-None fields win
    #: (smart_routing, provider_selection). ``update_fields`` narrows which
    #: fields the request may carry.
    MERGE_FIELDS = "merge_fields"
    #: Partial merge over wire defaults plus the existing row, keyed by wire
    #: field names (logging, whose stored dialect differs from the runtime
    #: model).
    MERGE_WIRE_PARTIAL = "merge_wire_partial"


class ReadMode(Enum):
    """How a GET (and the PUT response) renders the stored row."""

    #: Canonical model validation plus dump: ``model(**stored).model_dump()``.
    MODEL_DUMP = "model_dump"
    #: Wire defaults merged over the raw stored dict, echoed unvalidated
    #: (logging, so legacy rows round-trip verbatim).
    WIRE_MERGED = "wire_merged"
    #: The raw stored dict, or a hardcoded default dict when no row exists
    #: (web_search).
    RAW_OR_DEFAULT = "raw_or_default"
    #: Code defaults merged with stored overrides under ``{"limits": ...}``
    #: (rate_limits).
    LIMITS_MERGED = "limits_merged"
    #: The stored bare list wrapped in the canonical model (cors).
    LIST_WRAP = "list_wrap"


@dataclass(frozen=True, slots=True)
class ServerSetting:
    """One UI-managed server_config key, declared once."""

    key: str
    path: str
    description: str
    model: type[BaseModel]
    verbs: tuple[Literal["GET", "PUT", "DELETE"], ...] = ("GET", "PUT")
    write: WriteMode = WriteMode.REPLACE
    read: ReadMode = ReadMode.MODEL_DUMP
    #: MERGE_FIELDS narrowing: which request fields may override the stored
    #: row. ``None`` means every model field. Fields the client sends outside
    #: this list are ignored (the derived request model does not carry them),
    #: matching today's hand-written update schemas.
    update_fields: tuple[str, ...] | None = None
    #: The wire dialect of ``model``; only set when the stored row uses
    #: different field names/types than the runtime model (logging).
    wire_model: type[BaseModel] | None = None
    #: Optional pre-persist validation of the request body; raises ValueError
    #: with an operator-facing message the caller maps to its wire error
    #: (rate_limits: unknown buckets, invalid specs).
    validate: Callable[[BaseModel], None] | None = None
    #: RAW_OR_DEFAULT: the response document when no row exists.
    default_doc: Mapping[str, Any] | None = None
    #: Response document for DELETE.
    delete_message: str | None = None


def _validate_rate_limits(payload: BaseModel) -> None:
    """Pre-persist validation for the rate_limits key (mapped to HTTP 422 by the router)."""
    assert isinstance(payload, RateLimitsConfig)
    validate_rate_limit_overrides(payload.limits)


#: The registry. Order matches the settings UI's section order.
SERVER_SETTINGS: tuple[ServerSetting, ...] = (
    ServerSetting(
        key="logging",
        path="/logging",
        description="Logging configuration",
        model=LoggingConfig,
        wire_model=LoggingWireConfig,
        write=WriteMode.MERGE_WIRE_PARTIAL,
        read=ReadMode.WIRE_MERGED,
    ),
    ServerSetting(
        key="web_search_config",
        path="/web-search",
        description="Web search tool configuration for non-Anthropic providers",
        model=WebSearchConfig,
        verbs=("GET", "PUT", "DELETE"),
        write=WriteMode.REPLACE_EXCLUDE_NONE,
        read=ReadMode.RAW_OR_DEFAULT,
        default_doc={"enabled": False, "provider": "searxng", "searxng": None, "ollama": None},
        delete_message="Web search configuration deleted",
    ),
    ServerSetting(
        key="smart_routing",
        path="/smart-routing",
        description="Smart routing global configuration",
        model=SmartRoutingConfig,
        write=WriteMode.MERGE_FIELDS,
    ),
    ServerSetting(
        key="provider_selection",
        path="/provider-selection",
        description="Global provider selection configuration",
        model=ProviderSelectionConfig,
        write=WriteMode.MERGE_FIELDS,
        update_fields=("strategy",),
    ),
    ServerSetting(
        key="request_policy",
        path="/request-policy",
        description="Global request policy configuration",
        model=RequestPolicyConfig,
    ),
    ServerSetting(
        key="resilience",
        path="/resilience",
        description="Global resilience configuration",
        model=ResilienceConfig,
    ),
    ServerSetting(
        key="security",
        path="/security",
        description="Security and rate-limiting configuration",
        model=SecurityParams,
    ),
    ServerSetting(
        key="keepalive",
        path="/keepalive",
        description="Non-streaming response keepalive configuration",
        model=KeepaliveParams,
    ),
    ServerSetting(
        key="rate_limits",
        path="/rate-limits",
        description="Per-bucket rate limit overrides",
        model=RateLimitsConfig,
        write=WriteMode.REPLACE_INNER,
        read=ReadMode.LIMITS_MERGED,
        validate=_validate_rate_limits,
    ),
    ServerSetting(
        key="cors_origins",
        path="/cors",
        description="Allowed CORS origins",
        model=CorsConfig,
        write=WriteMode.REPLACE_LIST,
        read=ReadMode.LIST_WRAP,
    ),
    ServerSetting(
        key="mcp_security_policy",
        path="/mcp-security",
        description="MCP security policy configuration",
        model=McpSecurityPolicyConfig,
    ),
)


# --------------------------------------------------------------------------
# Request models
# --------------------------------------------------------------------------

_partial_models: dict[tuple[type[BaseModel], tuple[str, ...] | None], type[BaseModel]] = {}


def _derive_partial(
    model: type[BaseModel],
    fields: tuple[str, ...] | None,
) -> type[BaseModel]:
    """Derive an all-optional update model from a canonical model.

    Every field becomes ``<type> | None = None`` so an omitted field is
    distinguishable from an explicit value; merge-style writes overlay only
    the fields the client sent. ``fields`` narrows the model to a subset
    (provider_selection's update schema carries only ``strategy``); request
    bodies carrying other keys are ignored, matching today's behavior.
    """
    names = fields or tuple(model.model_fields)
    definitions: dict[str, Any] = {}
    for name in names:
        field_info = model.model_fields[name]
        annotation = field_info.annotation
        if type(None) not in getattr(annotation, "__args__", ()):  # not already optional
            annotation = annotation | None
        # Copy the canonical FieldInfo so validation constraints (ge/le, ...)
        # and descriptions survive; only the default changes (omitted → None).
        info = copy(field_info)
        info.default = None
        info.default_factory = None
        definitions[name] = (annotation, info)
    return create_model(f"{model.__name__}Update", **definitions)


def request_model_for(setting: ServerSetting) -> type[BaseModel]:
    """The request body model the PUT endpoint accepts for this setting.

    Replace-style writes take the canonical model itself — every field is
    defaulted, matching today's full-replace request bodies. Merge-style
    writes take the derived all-optional model.
    """
    if setting.write in (
        WriteMode.REPLACE,
        WriteMode.REPLACE_EXCLUDE_NONE,
        WriteMode.REPLACE_INNER,
        WriteMode.REPLACE_LIST,
    ):
        return setting.model
    source = setting.wire_model if setting.wire_model is not None else setting.model
    fields = setting.update_fields
    cache_key = (source, fields)
    cached = _partial_models.get(cache_key)
    if cached is None:
        cached = _derive_partial(source, fields)
        _partial_models[cache_key] = cached
    return cached


# --------------------------------------------------------------------------
# Writes
# --------------------------------------------------------------------------


def apply_write(setting: ServerSetting, payload: BaseModel, existing: Any) -> Any:
    """Compute the stored row value for a validated PUT request body."""
    match setting.write:
        case WriteMode.REPLACE:
            return payload.model_dump()
        case WriteMode.REPLACE_EXCLUDE_NONE:
            # Only the top-level presence of sub-configs is conditional (a
            # provided sub-config dumps in full, nulls included — matching the
            # old handler's ``config_data.searxng.model_dump()``); pydantic's
            # ``exclude_none`` would recurse into the sub-config and drop keys.
            dumped = payload.model_dump()
            return {k: v for k, v in dumped.items() if v is not None}
        case WriteMode.REPLACE_INNER:
            assert isinstance(payload, RateLimitsConfig)
            return dict(payload.limits)
        case WriteMode.REPLACE_LIST:
            assert isinstance(payload, CorsConfig)
            return [o.strip() for o in payload.origins if o.strip()]
        case WriteMode.MERGE_FIELDS:
            current = setting.model(**(existing if isinstance(existing, dict) else {}))
            names = setting.update_fields or tuple(setting.model.model_fields)
            for name in names:
                value = getattr(payload, name, None)
                if value is not None:
                    setattr(current, name, value)
            return current.model_dump()
        case WriteMode.MERGE_WIRE_PARTIAL:
            assert setting.wire_model is not None
            stored = setting.wire_model().model_dump()
            if isinstance(existing, dict):
                stored.update(existing)
            stored.update(payload.model_dump(exclude_unset=True))
            return stored
    raise ValueError(f"Unhandled write mode: {setting.write}")  # pragma: no cover


# --------------------------------------------------------------------------
# Reads
# --------------------------------------------------------------------------


def build_response(setting: ServerSetting, stored: Any) -> Any:
    """Render the response document for a stored row (GET and PUT responses)."""
    match setting.read:
        case ReadMode.MODEL_DUMP:
            value = stored if isinstance(stored, dict) else {}
            return setting.model(**value).model_dump()
        case ReadMode.WIRE_MERGED:
            assert setting.wire_model is not None
            merged = setting.wire_model().model_dump()
            if isinstance(stored, dict):
                merged.update(stored)
            return merged
        case ReadMode.RAW_OR_DEFAULT:
            if stored is not None:
                return stored
            return dict(setting.default_doc or {})
        case ReadMode.LIMITS_MERGED:
            merged = dict(DEFAULT_RATE_LIMITS)
            if isinstance(stored, dict):
                merged.update({k: str(v) for k, v in stored.items() if k in DEFAULT_RATE_LIMITS})
            return {"limits": merged}
        case ReadMode.LIST_WRAP:
            origins = [str(o) for o in stored] if isinstance(stored, list) else []
            return setting.model.model_validate({"origins": origins}).model_dump()
    raise ValueError(f"Unhandled read mode: {setting.read}")  # pragma: no cover


# --------------------------------------------------------------------------
# Runtime read
# --------------------------------------------------------------------------


def _parse_bool(value: Any) -> bool:
    """Parse a value as boolean, supporting string and non-string types."""
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


def _parse_optional_float(value: Any, name: str) -> float | None:
    """Parse an optional float config value (None/empty string → None)."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (ValueError, TypeError) as exc:
        raise ConfigurationError(f"Invalid {name}: {exc}") from exc


def _parse_optional_int(value: Any, name: str) -> int | None:
    """Parse an optional int config value (None/empty string → None)."""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (ValueError, TypeError) as exc:
        raise ConfigurationError(f"Invalid {name}: {exc}") from exc


def load_logging_config(overrides: dict[str, Any] | None = None) -> LoggingConfig:
    """Build the logging configuration from database overrides.

    All behavioral fields (masking, sampling, retention, sensitive keys) are
    UI-managed and stored in the ``logging`` server_config key under their
    wire names; this function merges those stored values with code defaults
    into the runtime model. The only remaining env var is ``LOG_LEVEL``
    (startup-time logging pipeline verbosity).
    """
    from llm_proxy.config.settings import get_settings

    settings = get_settings().logging
    overrides = overrides or {}

    log_input_output = _parse_bool(overrides.get("log_input_output", False))
    log_raw_stream = _parse_bool(overrides.get("log_raw_stream", False))

    try:
        retention_days = int(overrides.get("log_retention_days", DEFAULT_RETENTION_DAYS))
    except (ValueError, TypeError) as exc:
        raise ConfigurationError(f"Invalid log_retention_days: {exc}") from exc

    try:
        max_logged_body_bytes = int(
            overrides.get("max_logged_body_bytes", DEFAULT_MAX_LOGGED_BODY_BYTES)
        )
    except (ValueError, TypeError) as exc:
        raise ConfigurationError(f"Invalid max_logged_body_bytes: {exc}") from exc

    verbose_routing_logs = _parse_bool(overrides.get("verbose_routing_logs", False))
    mask_sensitive_data = _parse_bool(overrides.get("mask_sensitive_data", True))

    try:
        sampling_rate = float(overrides.get("sampling_rate", 1.0))
    except (ValueError, TypeError) as exc:
        raise ConfigurationError(f"Invalid sampling_rate: {exc}") from exc

    audit_sampling_rate = _parse_optional_float(
        overrides.get("audit_sampling_rate"), "audit_sampling_rate"
    )
    audit_retention_days = _parse_optional_int(
        overrides.get("audit_retention_days"), "audit_retention_days"
    )

    sensitive_keys = [
        "authorization",
        "api_key",
        "apikey",
        "password",
        "passwd",
        "token",
        "access_token",
        "refresh_token",
        "jwt_secret",
    ]
    extra_keys_raw = overrides.get("sensitive_keys", "")
    if isinstance(extra_keys_raw, str):
        extra_keys = [k.strip().lower() for k in extra_keys_raw.split(",") if k.strip()]
    elif isinstance(extra_keys_raw, list):
        extra_keys = [str(k).strip().lower() for k in extra_keys_raw if str(k).strip()]
    else:
        extra_keys = []
    if extra_keys:
        sensitive_keys = [*dict.fromkeys(sensitive_keys + extra_keys)]

    # Env-only value (startup-time logging pipeline verbosity):
    log_level = settings.level.upper()

    return LoggingConfig(
        log_input_output=log_input_output,
        log_raw_stream=log_raw_stream,
        retention_days=retention_days,
        max_logged_body_bytes=max_logged_body_bytes,
        mask_sensitive_data=mask_sensitive_data,
        log_level=log_level,
        sampling_rate=sampling_rate,
        audit_sampling_rate=audit_sampling_rate,
        audit_retention_days=audit_retention_days,
        sensitive_keys=sensitive_keys,
        verbose_routing_logs=verbose_routing_logs,
    )


@dataclass(slots=True)
class ServerSettingsRuntime:
    """The typed runtime snapshot the config manager assembles ProxyConfig from.

    One field per server_config key the runtime reads; ``mcp_security_policy``
    is deliberately absent — it is read lazily by ``mcp/security/policy.py``
    with its own cache, not through ProxyConfig.
    """

    logging: LoggingConfig
    web_search: WebSearchConfig | None
    request_policy_unknown_fields: Literal["ignore", "passthrough", "error"] | None
    request_policy_unsupported_block: Literal["drop", "degrade", "error"] | None
    resilience: ResilienceConfig
    security: SecurityParams
    keepalive: KeepaliveParams
    rate_limits: dict[str, str]
    cors_origins: list[str]
    smart_routing: SmartRoutingConfig
    provider_selection: ProviderSelectionConfig


def _repair_web_search_row(value: Any) -> tuple[Any, bool]:
    """Clamp out-of-range numeric fields in a stored web_search row.

    The canonical model's bounds were tightened to the API-facing rules
    (``max_results <= 10``, ``1 <= timeout <= 300``); rows written before the
    tightening (or edited directly in the database) would otherwise fail the
    whole-config validate and silently disable web search. Field-level ranges
    are repaired so the rest of the configuration keeps working.
    """
    if not isinstance(value, dict):
        return value, False
    repaired = dict(value)
    changed = False
    for sub in ("searxng", "ollama"):
        block = repaired.get(sub)
        if not isinstance(block, dict):
            continue
        fixed = dict(block)
        max_results = fixed.get("max_results")
        if isinstance(max_results, int) and not isinstance(max_results, bool) and max_results > 10:
            fixed["max_results"] = 10
            changed = True
        timeout = fixed.get("timeout")
        if isinstance(timeout, (int, float)) and not isinstance(timeout, bool):
            if timeout < 1:
                fixed["timeout"] = 1.0
                changed = True
            elif timeout > 300:
                fixed["timeout"] = 300.0
                changed = True
        repaired[sub] = fixed
    return repaired, changed


def load_server_settings(rows: Mapping[str, Any]) -> ServerSettingsRuntime:
    """Parse the server_config rows into the runtime snapshot.

    Tolerant by design: an invalid stored value logs a warning and falls back
    to code defaults (or, for logging, fails loudly — retention and body caps
    must never be silently dropped). This preserves the exact semantics of the
    per-key parsing that used to live inline in ``DatabaseConfigManager.load``.
    """
    logging_overrides = rows.get("logging")
    logging_config = load_logging_config(
        logging_overrides if isinstance(logging_overrides, dict) else None
    )

    web_search_config = None
    if "web_search_config" in rows:
        stored = rows["web_search_config"]
        stored, repaired = _repair_web_search_row(stored)
        if repaired:
            logger.warning(
                "Adjusted out-of-range max_results/timeout in stored web_search_config; "
                "re-save the setting in the admin UI to persist the clamped values"
            )
        try:
            web_search_config = WebSearchConfig.model_validate(stored)
        except Exception as e:
            logger.warning(f"Failed to parse web_search_config: {e}")

    # "default"/absent means "not configured": the per-adapter default
    # applies. An invalid value is treated the same way after a warning, so a
    # typo never silently overrides a provider's own default.
    request_policy = rows.get("request_policy", {})
    if not isinstance(request_policy, dict):
        request_policy = {}

    unknown_fields_policy = request_policy.get("unknown_fields_policy", "default")
    if unknown_fields_policy == "default":
        unknown_fields_policy = None
    elif unknown_fields_policy not in {"ignore", "passthrough", "error"}:
        logger.warning(
            f"Invalid unknown_fields_policy '{unknown_fields_policy}' in server config; "
            "treating it as unset (per-provider default applies)"
        )
        unknown_fields_policy = None

    unsupported_block_policy = request_policy.get("unsupported_block_policy", "default")
    if unsupported_block_policy == "default":
        unsupported_block_policy = None
    elif unsupported_block_policy not in {"drop", "degrade", "error"}:
        logger.warning(
            f"Invalid unsupported_block_policy "
            f"'{unsupported_block_policy}' in server config; "
            "treating it as unset (per-provider default applies)"
        )
        unsupported_block_policy = None

    resilience = rows.get("resilience", {})
    if not isinstance(resilience, dict):
        resilience = {}

    max_fallback_attempts = resilience.get("max_fallback_attempts", DEFAULT_MAX_FALLBACK_ATTEMPTS)
    if not isinstance(max_fallback_attempts, int) or max_fallback_attempts < 0:
        logger.warning("Invalid max_fallback_attempts in server config; falling back to default")
        max_fallback_attempts = DEFAULT_MAX_FALLBACK_ATTEMPTS

    global_max_retries = resilience.get("max_retries", DEFAULT_MAX_RETRIES)
    if not isinstance(global_max_retries, int) or global_max_retries < 0:
        logger.warning("Invalid max_retries in server config; falling back to default")
        global_max_retries = DEFAULT_MAX_RETRIES

    cb = resilience.get("circuit_breaker", {})
    if not isinstance(cb, dict):
        cb = {}
    resilience_config = ResilienceConfig(
        max_retries=global_max_retries,
        max_fallback_attempts=max_fallback_attempts,
        circuit_breaker=CircuitBreakerParams(
            enabled=cb.get("enabled", True),
            failure_threshold=cb.get("failure_threshold", 5),
            cooldown_seconds=cb.get("cooldown_seconds", 60.0),
        ),
    )

    security_value = rows.get("security")
    try:
        security_params = SecurityParams.model_validate(
            security_value if isinstance(security_value, dict) else {}
        )
    except Exception as e:
        logger.warning(f"Failed to parse security config: {e}")
        security_params = SecurityParams()

    keepalive_value = rows.get("keepalive")
    try:
        keepalive_params = KeepaliveParams.model_validate(
            keepalive_value if isinstance(keepalive_value, dict) else {}
        )
    except Exception as e:
        logger.warning(f"Failed to parse keepalive config: {e}")
        keepalive_params = KeepaliveParams()

    rate_limits_value = rows.get("rate_limits")
    rate_limits: dict[str, str] = {}
    if isinstance(rate_limits_value, dict):
        rate_limits = {str(k): str(v) for k, v in rate_limits_value.items()}

    cors_origins_value = rows.get("cors_origins")
    cors_origins: list[str] = []
    if isinstance(cors_origins_value, list):
        cors_origins = [str(o).strip() for o in cors_origins_value if str(o).strip()]

    smart_routing_value = rows.get("smart_routing")
    try:
        smart_routing_config = SmartRoutingConfig.from_row(
            smart_routing_value if isinstance(smart_routing_value, dict) else None
        )
    except Exception as e:
        logger.warning(f"Failed to parse smart_routing config: {e}")
        smart_routing_config = SmartRoutingConfig()

    provider_selection_value = rows.get("provider_selection")
    try:
        provider_selection_config = ProviderSelectionConfig.from_row(
            provider_selection_value if isinstance(provider_selection_value, dict) else None
        )
    except Exception as e:
        logger.warning(f"Failed to parse provider_selection config: {e}")
        provider_selection_config = ProviderSelectionConfig()

    return ServerSettingsRuntime(
        logging=logging_config,
        web_search=web_search_config,
        request_policy_unknown_fields=unknown_fields_policy,
        request_policy_unsupported_block=unsupported_block_policy,
        resilience=resilience_config,
        security=security_params,
        keepalive=keepalive_params,
        rate_limits=rate_limits,
        cors_origins=cors_origins,
        smart_routing=smart_routing_config,
        provider_selection=provider_selection_config,
    )
