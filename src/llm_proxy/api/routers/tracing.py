"""Shared helpers for tracing configuration endpoints.

Tracing is strictly per-user: every user (admin included) manages their own
tracing backends via ``/api/me/tracing`` and their config applies only to their
own requests. There is no system-level tracing config. This module hosts the
request/response helpers (masked-secret preservation, config shaping, SSRF
validation) shared by the self-service router and tests.
"""

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from llm_proxy.api.schemas.tracing import (
    TracingConfigRead,
    TracingConfigWrite,
    TracingProviderRead,
    _mask_settings,
)
from llm_proxy.http.client import validate_server_url, validate_url_format
from llm_proxy.observability.logger import get_logger
from llm_proxy.observability.tracing_config import TracingConfig, TracingProviderConfig

logger = get_logger(__name__)

#: DNS validation is blocking and reachable by any authenticated viewer.
#: Keeping it on a small dedicated pool prevents unique-hostname DNS timeouts
#: from occupying every default-executor thread that also serves API-key
#: verification and other gateway work.
_TRACING_URL_VALIDATION_EXECUTOR = ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="tracing-url-check"
)


def _is_masked_key(value: Any) -> bool:
    """Check if a value is masked (contains **** from backend response)."""
    return isinstance(value, str) and "****" in value


def _restore_masked_dict(
    values: dict[str, Any],
    existing_values: dict[str, Any],
    context: str = "",
) -> dict[str, Any]:
    """Replace masked placeholders with previously stored values.

    Keys whose value is a masked placeholder are replaced with the
    corresponding value from ``existing_values`` when available, or dropped
    when no existing value exists.
    """
    result = {}
    for key, value in values.items():
        if _is_masked_key(value):
            if key in existing_values:
                result[key] = existing_values[key]
            else:
                logger.warning(
                    f"Masked{context} '{key}' has no existing counterpart; key will be dropped"
                )
            continue
        result[key] = value
    return result


def _preserve_masked_values(
    settings: dict[str, Any], existing_settings: dict[str, Any]
) -> dict[str, Any]:
    """Restore values that were masked in the API response.

    Top-level string values and nested header values equal to a masked
    placeholder are replaced with the previously stored value when available.
    If a value is masked and there is no existing value, the key is dropped
    rather than storing the literal placeholder.
    """
    result = _restore_masked_dict(settings, existing_settings)

    headers = result.get("headers")
    existing_headers = existing_settings.get("headers")
    if isinstance(headers, dict):
        if not isinstance(existing_headers, dict):
            existing_headers = {}
        result["headers"] = _restore_masked_dict(headers, existing_headers, context=" header")

    return result


def _build_config_read(config_dict: dict) -> TracingConfigRead:
    """Build TracingConfigRead from config dict."""
    config = TracingConfig.from_dict(config_dict)
    providers = [
        TracingProviderRead(
            provider=p.provider,
            name=p.name,
            enabled=p.enabled,
            settings=p.settings,
            masked_settings=_mask_settings(p.settings),
        )
        for p in config.providers
    ]
    return TracingConfigRead(enabled=config.enabled, providers=providers)


def _validate_tracing_provider_urls(
    providers: list[TracingProviderConfig], *, allow_private_hosts: bool = True
) -> None:
    """Validate tracing provider settings for SSRF (currently Langfuse base_url).

    Admin-managed tracing may point at a private/self-hosted collector, so the
    default stays format-only. Self-service configs (any logged-in user) pass
    ``allow_private_hosts=False``: a viewer must not be able to turn the
    exporter into a blind SSRF primitive. Operators can still permit specific
    internal hosts through ``HTTP_ALLOWED_HOSTS``.
    """
    for provider in providers:
        base_url = provider.settings.get("base_url")
        if base_url and isinstance(base_url, str):
            label = f"tracing provider '{provider.name}' base_url"
            if allow_private_hosts:
                validate_url_format(base_url, label=label)
            else:
                validate_server_url(base_url, label=label, fail_closed=True)


async def _validate_tracing_provider_urls_async(
    providers: list[TracingProviderConfig], *, allow_private_hosts: bool = True
) -> None:
    """Async wrapper for tracing URL validation (bounded blocking DNS)."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        _TRACING_URL_VALIDATION_EXECUTOR,
        functools.partial(
            _validate_tracing_provider_urls,
            providers,
            allow_private_hosts=allow_private_hosts,
        ),
    )


def build_persisted_tracing_dict(
    config_data: TracingConfigWrite,
    existing_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the tracing config dict to persist, preserving masked secrets.

    Values masked in the API response (``****``) are restored from the
    previously stored config so editing other fields does not wipe secrets.
    """
    existing_providers: list[TracingProviderConfig] = []
    if existing_config:
        existing_providers = TracingConfig.from_dict(existing_config).providers

    existing_by_name: dict[str, TracingProviderConfig] = {}
    duplicate_names: set[str] = set()
    for ep in existing_providers:
        if ep.name in existing_by_name:
            duplicate_names.add(ep.name)
        existing_by_name[ep.name] = ep

    incoming_names: list[str] = [p.name for p in config_data.providers]
    if len(incoming_names) != len(set(incoming_names)):
        logger.warning(f"Duplicate provider names detected in incoming config: {incoming_names}")

    def _find_existing_settings(p, idx):
        """Find existing settings for a provider, matching by name or index."""
        if p.name in existing_by_name and p.name not in duplicate_names:
            return existing_by_name[p.name].settings
        if idx < len(existing_providers):
            if p.name in duplicate_names:
                logger.warning(
                    f"Duplicate provider name '{p.name}' detected; falling back to "
                    f"index-based matching for masked value preservation"
                )
            return existing_providers[idx].settings
        return None

    providers: list[dict[str, Any]] = []
    for idx, p in enumerate(config_data.providers):
        settings = dict(p.settings)
        existing = _find_existing_settings(p, idx)
        if existing is not None:
            settings = _preserve_masked_values(settings, existing)

        providers.append(
            {
                "provider": p.provider,
                "name": p.name,
                "enabled": p.enabled,
                "settings": settings,
            }
        )

    return {"enabled": config_data.enabled, "providers": providers}


__all__ = [
    "_build_config_read",
    "_preserve_masked_values",
    "_validate_tracing_provider_urls_async",
    "build_persisted_tracing_dict",
]
