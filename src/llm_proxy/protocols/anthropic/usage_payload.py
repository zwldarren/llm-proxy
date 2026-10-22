"""The single builder for an Anthropic ``usage`` object.

Three hand-built copies of this payload was what let the streaming and
non-streaming bodies drift (ADR-0042 records the same failure for the
OpenResponses snapshot): a fact added to one copy never reached the others.
Every Anthropic usage payload — the non-streaming ``Message``, the streaming
``message_start`` and the terminal ``message_delta`` — comes from here.

Two Anthropic rules this module owns:

* ``input_tokens`` is **exclusive** of the cache counters. The canonical record
  includes them (an Anthropic upstream reports the split itself); emitting the
  inclusive value next to ``cache_read_input_tokens`` makes a client that sums
  the three count the cached prompt twice.
* ``Usage`` and ``MessageDeltaUsage`` declare their fields as *required
  nullable*: the published schema (and the SDKs generated from it) expect the
  key to be present, ``null`` when the provider did not report it. Omitting a
  key a client's type declares required is a parse failure, so unknowns are
  emitted as ``null`` rather than dropped.
"""

from collections.abc import Mapping
from typing import Any

from llm_proxy.models.types import Usage
from llm_proxy.serialization.anthropic import normalize_service_tier_for_anthropic
from llm_proxy.serialization.usage.folds import fold_deepseek_cache_hits

__all__ = ["build_usage_payload", "canonical_usage"]


def _field(container: Any, name: str) -> Any:
    """Read ``name`` from a mapping or a details object (both shapes reach us)."""
    if container is None:
        return None
    if isinstance(container, Mapping):
        return container.get(name)
    return getattr(container, name, None)


def canonical_usage(usage: Usage) -> dict[str, Any]:
    """Express a canonical ``Usage`` record as the dict this builder consumes."""
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_input_tokens": usage.cache_read_input_tokens,
        "cache_creation_input_tokens": usage.cache_creation_input_tokens,
        "reasoning_tokens": usage.reasoning_tokens,
        "prompt_tokens_details": usage.prompt_tokens_details,
        "completion_tokens_details": usage.completion_tokens_details,
        "web_search_requests": usage.web_search_requests,
    }


def build_usage_payload(
    usage: Mapping[str, Any] | None,
    *,
    delta: bool = False,
) -> dict[str, Any]:
    """Assemble an Anthropic ``usage`` payload from canonical facts.

    Args:
        usage: Canonical usage dict (or one merged with ``provider_info``
            extras: ``cache_creation``, ``inference_geo``, ``service_tier``,
            ``server_tool_use``, ``speed``, ``iterations``).
        delta: Build the ``MessageDeltaUsage`` shape instead of the full
            ``Usage`` shape (the delta variant has no ``cache_creation``,
            ``inference_geo`` or ``service_tier``).
    """
    usage = usage or {}
    prompt_details = usage.get("prompt_tokens_details")
    cache_read = _field(usage, "cache_read_input_tokens")
    if cache_read is None:
        # OpenAI-family providers express the fact in the nested dialect;
        # DeepSeek adds a top-level ``prompt_cache_hit_tokens`` alias.
        cache_read = fold_deepseek_cache_hits(
            _field(prompt_details, "cached_tokens"),
            _field(usage, "prompt_cache_hit_tokens"),
        )
    cache_write = _field(usage, "cache_creation_input_tokens")
    if cache_write is None:
        cache_write = _field(prompt_details, "cache_write_tokens")

    details = _field(usage, "output_tokens_details")
    if details is None:
        thinking = _field(usage, "reasoning_tokens")
        if thinking is None:
            thinking = _field(usage.get("completion_tokens_details"), "reasoning_tokens")
        details = {"thinking_tokens": thinking} if thinking is not None else None

    server_tool_use = _field(usage, "server_tool_use")
    if server_tool_use is None:
        web_search = _field(usage, "web_search_requests")
        if web_search is not None:
            server_tool_use = {"web_search_requests": web_search, "web_fetch_requests": 0}

    input_tokens = _field(usage, "input_tokens") or 0
    payload: dict[str, Any] = {
        "input_tokens": max(0, input_tokens - (cache_read or 0) - (cache_write or 0)),
        "output_tokens": _field(usage, "output_tokens") or 0,
        "cache_creation_input_tokens": cache_write,
        "cache_read_input_tokens": cache_read,
        "output_tokens_details": details,
        "server_tool_use": server_tool_use,
    }
    if not delta:
        payload["cache_creation"] = _field(usage, "cache_creation")
        payload["inference_geo"] = _field(usage, "inference_geo")
        # Official response enum only; OpenAI tiers fold or drop.
        tier = _field(usage, "service_tier")
        payload["service_tier"] = (
            normalize_service_tier_for_anthropic(tier) if tier is not None else None
        )
    # Beta counters (fast-mode ``speed``, compaction ``iterations``): top-level
    # token counts exclude compaction iterations, so dropping them would break
    # per-iteration cost accounting.
    for key in ("speed", "iterations"):
        value = _field(usage, key)
        if value is not None:
            payload[key] = value
    return payload
