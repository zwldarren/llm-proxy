"""Unified thinking/reasoning parameter and content conversion.

Provides bidirectional conversion between different protocol and provider
thinking formats:

1. Request parameters (ThinkingConfig): All protocol serializers normalize
   into ThinkingConfig, and all provider serializers convert from
   ThinkingConfig to their native format.

2. Content fields: Defines how each provider's fixed-format reasoning/thinking
   content is named. For OpenAI-compatible providers (openai, deepseek, etc.),
   the reasoning field name (``reasoning_content`` vs ``reasoning``) is
   dynamically detected at runtime — see ``providers/openai/adapter.py``.
   Uses ThinkingBlock as the universal internal representation.

Fixed-format provider reasoning field table:
┌───────────┬──────────────────────┬─────────────────────┬─────────────────────┬───────────┐
│ Provider  │ Message field        │ Stream delta field  │ Content block type  │ Signature │
├───────────┼──────────────────────┼─────────────────────┼─────────────────────┼───────────┤
│ anthropic │ None (content block) │ thinking_delta      │ thinking            │ Yes       │
│ gemini    │ None (content block) │ thought             │ thought             │ Yes       │
│ ollama    │ thinking             │ thinking            │ None (msg field)    │ No        │
└───────────┴──────────────────────┴─────────────────────┴─────────────────────┴───────────┘

OpenAI-compatible providers (openai, deepseek, openrouter, etc.):
  Field names (reasoning_content vs reasoning) are detected dynamically at
  runtime and cached per (base_url, model) — the convention belongs to the
  model, so one gateway serving both conventions keeps each model's
  preference independent. Every response path teaches the model before the
  downstream rename: the parsed non-streaming tier (OpenAIResponseParser),
  the verbatim wire-reuse tier, and streaming chunks
  (adapter._stream_transform_chunk). Never-seen models default to
  ``reasoning_content`` (OpenAI's standard) until their first response.
  See OpenAIRequestBuilder in serialization/openai/components/request_builder.py
  and OpenAICompatibleBase in providers/openai_compatible/_base.py.
"""

from typing import Any

from llm_proxy.models.types import ThinkingConfig

PROVIDER_REASONING_FORMAT: dict[str, dict[str, Any]] = {
    "anthropic": {
        "message_field": None,
        "stream_delta_field": "thinking",
        "content_block_type": "thinking",
        "redacted_block_type": "redacted_thinking",
        "signature_supported": True,
    },
    "gemini": {
        "message_field": None,
        "stream_delta_field": "thought",
        "content_block_type": "thought",
        "signature_supported": True,
    },
    "ollama": {
        "message_field": "thinking",
        "stream_delta_field": "thinking",
        "content_block_type": None,
        "signature_supported": False,
    },
}


def normalize_thinking(data: dict[str, Any]) -> ThinkingConfig | None:
    """Normalize various thinking parameter formats into ThinkingConfig.

    Handles:
    - OpenAI ``reasoning_effort`` (string)
    - OpenAI ``thinking`` dict (with type and budget_tokens)
    - Anthropic ``thinking`` dict (with type and budget_tokens)
    - OpenResponses ``reasoning`` dict (with effort field)

    Args:
        data: Raw request data dict that may contain thinking parameters.

    Returns:
        ThinkingConfig if any thinking parameter is found, else None.
    """
    # Check for OpenAI reasoning_effort
    reasoning_effort = data.get("reasoning_effort")
    if reasoning_effort is not None:
        return thinking_config_from_reasoning_effort(str(reasoning_effort))

    # Check for thinking dict (OpenAI or Anthropic style)
    thinking = data.get("thinking")
    if thinking is True:
        return ThinkingConfig(type="enabled")
    if thinking is False:
        return ThinkingConfig(type="disabled")
    if isinstance(thinking, dict):
        thinking_type = thinking.get("type", "enabled")
        if thinking_type not in ("enabled", "disabled", "adaptive"):
            thinking_type = "enabled"
        effort = thinking.get("effort")
        if effort is None and thinking_type == "adaptive":
            oc = data.get("output_config")
            if isinstance(oc, dict):
                oc_effort = oc.get("effort")
                if oc_effort is not None:
                    effort = str(oc_effort)
        return ThinkingConfig(
            type=thinking_type,
            budget_tokens=thinking.get("budget_tokens"),
            display=thinking.get("display"),
            effort=effort,
        )

    # Check for OpenResponses reasoning object
    reasoning = data.get("reasoning")
    if isinstance(reasoning, dict):
        effort = reasoning.get("effort")
        if effort is not None:
            return thinking_config_from_reasoning_effort(str(effort))

    # Check for Anthropic output_config.effort
    output_config = data.get("output_config")
    if isinstance(output_config, dict):
        effort = output_config.get("effort")
        if effort is not None:
            return thinking_config_from_reasoning_effort(str(effort))

    return None


def thinking_config_from_reasoning_effort(effort: str) -> ThinkingConfig:
    """Convert OpenAI reasoning_effort string to ThinkingConfig.

    Mapping:
    - ``none`` → disabled
    - ``minimal`` / ``low`` → enabled, low budget
    - ``medium`` → enabled, medium budget
    - ``high`` / ``xhigh`` → enabled, high budget
    """
    effort_lower = effort.lower()
    if effort_lower == "none":
        return ThinkingConfig(type="disabled")
    if effort_lower in ("minimal", "low"):
        return ThinkingConfig(type="enabled", budget_tokens=4000, effort=effort_lower)
    if effort_lower == "medium":
        return ThinkingConfig(type="enabled", budget_tokens=16000, effort=effort_lower)
    if effort_lower in ("high", "xhigh"):
        return ThinkingConfig(type="enabled", budget_tokens=32000, effort=effort_lower)
    if effort_lower == "max":
        return ThinkingConfig(type="enabled", effort=effort_lower)
    # Unknown effort - default to enabled with medium budget
    return ThinkingConfig(type="enabled", budget_tokens=16000, effort=effort_lower)


def thinking_config_to_reasoning_effort(config: ThinkingConfig) -> str | None:
    """Convert ThinkingConfig to OpenAI reasoning_effort string.

    Returns the effort field if set, otherwise derives from type/budget_tokens.
    """
    if config.effort is not None:
        return config.effort
    if config.type == "disabled":
        return "none"
    if config.budget_tokens is not None:
        if config.budget_tokens < 8000:
            return "low"
        if config.budget_tokens > 24000:
            return "high"
        return "medium"
    # Default to medium when enabled but no budget specified
    if config.type == "enabled":
        return "medium"
    return None


def resolve_thinking(request: Any) -> ThinkingConfig | None:
    """Resolve the canonical ThinkingConfig that provider builders should use.

    This is the **single source of truth** for the thinking/reasoning effort
    surfaced to non-Anthropic providers (OpenAI Chat, OpenAI Responses, Ollama,
    Gemini). Anthropic is handled separately by its provider serializer because
    it has a native ``output_config`` field — see the note below.

    Precedence (highest first):
    1. ``request.params.thinking`` — set by every protocol parser via
       :func:`normalize_thinking`, which already funnels ``reasoning_effort``,
       the ``thinking`` dict, ``reasoning.effort`` and ``output_config.effort``
       into it. In normal proxy flow this is always set when any effort source is
       present, so the fallbacks below only matter for directly-constructed
       ``InternalRequest`` objects that bypass the parsers.
    2. ``request.params.anthropic.output_config.effort`` — Anthropic effort.
    3. ``request.params.openai.reasoning_effort`` — OpenAI effort.

    Centralizing this here means provider builders never re-derive effort from
    protocol-specific fields; they call ``convert_to_*(resolve_thinking(request))``
    and nothing else. Adding a new provider requires zero thinking-specific
    fallback logic.
    """
    params = request.params
    if params.thinking is not None:
        return params.thinking
    if params.anthropic is not None and params.anthropic.output_config:
        effort = params.anthropic.output_config.get("effort")
        if effort is not None:
            return thinking_config_from_reasoning_effort(str(effort))
    if params.openai is not None and params.openai.reasoning_effort is not None:
        return thinking_config_from_reasoning_effort(str(params.openai.reasoning_effort))
    return None


_VALID_REASONING_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})


def convert_to_openai(config: ThinkingConfig | None) -> dict[str, Any] | None:
    """Convert ThinkingConfig to OpenAI provider format.

    Returns either ``{"reasoning_effort": ...}`` or ``{"thinking": {...}}``.
    Prefers reasoning_effort when a valid effort value can be derived; falls
    back to thinking dict for provider-specific budget_tokens precision.
    """
    if config is None:
        return None

    effort = thinking_config_to_reasoning_effort(config)
    if effort is not None and effort.lower() in _VALID_REASONING_EFFORTS:
        return {"reasoning_effort": effort}

    if config.type == "adaptive":
        return None

    if config.type is not None or config.budget_tokens is not None:
        thinking_dict: dict[str, Any] = {}
        if config.type is not None:
            thinking_dict["type"] = config.type
        if config.budget_tokens is not None:
            thinking_dict["budget_tokens"] = config.budget_tokens
        return {"thinking": thinking_dict}

    return None


def convert_to_anthropic(config: ThinkingConfig | None) -> dict[str, Any] | None:
    """Convert ThinkingConfig to Anthropic provider format.

    When ``effort`` is set, outputs ``output_config`` (the preferred modern
    Anthropic API) and ``thinking`` without ``budget_tokens`` since effort
    already controls the budget.  When ``effort`` is absent, outputs the
    ``thinking`` dict with ``budget_tokens``.

    Effort value mapping (Anthropic supports ``low``, ``medium``,
    ``high``, ``xhigh``, ``max``):
    - ``minimal`` → ``low``
    """
    if config is None or config.type is None:
        return None

    thinking_dict: dict[str, Any] = {"type": config.type}
    if config.budget_tokens is not None and config.effort is None:
        thinking_dict["budget_tokens"] = config.budget_tokens
    if config.display is not None:
        thinking_dict["display"] = config.display

    result: dict[str, Any] = {"thinking": thinking_dict}
    if config.effort is not None:
        effort_lower = config.effort.lower()
        # Map OpenAI-specific effort values to Anthropic-supported values
        mapped_effort = "low" if effort_lower == "minimal" else effort_lower
        result["output_config"] = {"effort": mapped_effort}
    return result


def convert_to_gemini(config: ThinkingConfig | None) -> dict[str, Any] | None:
    """Convert ThinkingConfig to Gemini provider format (Gemini 3+).

    Returns ``{"thinkingConfig": {"thinkingLevel": "LOW" | "MEDIUM" | "HIGH"}}`` suitable for
    merging into ``generationConfig`` for Gemini 3+ models.
    For older Gemini models, use :func:`convert_to_gemini_legacy` instead.
    """
    if config is None:
        return None

    if config.type == "disabled":
        return None  # Gemini does not support explicit disabling

    level = "MEDIUM"
    if config.budget_tokens is not None:
        if config.budget_tokens < 8000:
            level = "LOW"
        elif config.budget_tokens > 24000:
            level = "HIGH"
    elif config.effort is not None:
        effort_lower = config.effort.lower()
        if effort_lower in ("none", "minimal", "low"):
            level = "LOW"
        elif effort_lower == "medium":
            level = "MEDIUM"
        elif effort_lower in ("high", "xhigh", "max"):
            level = "HIGH"

    return {"thinkingConfig": {"thinkingLevel": level}}


def convert_to_gemini_interactions(config: ThinkingConfig | None) -> dict[str, Any] | None:
    """Convert ThinkingConfig to Gemini Interactions generation_config format.

    Returns ``{"thinking_level": "minimal" | "low" | "medium" | "high"}``
    (lowercase enum, unlike the uppercase ``thinkingConfig.thinkingLevel`` of
    the generateContent dialect) suitable for merging into
    ``generation_config`` of an Interactions API request.

    When ``config.display`` is set, ``thinking_summaries`` is emitted too
    (``"auto"`` to enable summaries, ``"none"`` for explicit suppression).
    Omitted when unset — the API default shows the final output only.
    """
    if config is None or config.type == "disabled":
        return None  # Interactions has no explicit disable; omit the field.

    level = "medium"
    if config.budget_tokens is not None:
        if config.budget_tokens < 8000:
            level = "low"
        elif config.budget_tokens > 24000:
            level = "high"
    elif config.effort is not None:
        effort_lower = config.effort.lower()
        if effort_lower == "none":
            level = "minimal"
        elif effort_lower in ("minimal", "low"):
            level = "low"
        elif effort_lower in ("high", "xhigh", "max"):
            level = "high"

    result: dict[str, Any] = {"thinking_level": level}
    if config.display is not None:
        # thinking_summaries is a string enum ("auto" | "none"); any display
        # request other than explicit suppression maps to "auto".
        display = config.display.lower()
        result["thinking_summaries"] = "none" if display in ("none", "off", "hidden") else "auto"
    return result


def convert_to_gemini_legacy(config: ThinkingConfig | None) -> dict[str, Any] | None:
    """Convert ThinkingConfig to Gemini legacy provider format (pre-Gemini 3).

    Returns ``{"thinkingConfig": {"thinkingBudget": n, "includeThoughts": true}}``
    suitable for merging into ``generationConfig``.
    """
    if config is None:
        return None

    if config.type == "disabled":
        return None

    budget = 4096  # default medium
    if config.budget_tokens is not None:
        budget = config.budget_tokens
    elif config.effort is not None:
        effort_lower = config.effort.lower()
        if effort_lower in ("none", "minimal", "low"):
            budget = 1024
        elif effort_lower == "medium":
            budget = 4096
        elif effort_lower in ("high", "xhigh", "max"):
            budget = 8192

    return {"thinkingConfig": {"thinkingBudget": budget, "includeThoughts": True}}


def convert_to_ollama(config: ThinkingConfig | None) -> bool | str | None:
    """Convert ThinkingConfig to Ollama provider think parameter.

    Returns ``bool`` (enable/disable) or ``"low" | "medium" | "high" | "max"``.
    """
    if config is None:
        return None

    if config.type == "disabled":
        return False

    # If effort is set, map directly
    if config.effort is not None:
        effort_lower = config.effort.lower()
        if effort_lower == "none":
            return False
        if effort_lower in ("minimal", "low"):
            return "low"
        if effort_lower == "medium":
            return "medium"
        if effort_lower == "high":
            return "high"
        if effort_lower == "xhigh":
            return "max"
        if effort_lower == "max":
            return "max"

    # Derive from budget_tokens
    if config.budget_tokens is not None:
        if config.budget_tokens < 8000:
            return "low"
        if config.budget_tokens > 24000:
            return "high"
        return "medium"

    # Default: enabled
    return True


def extract_reasoning_from_message(
    message: dict[str, Any], provider: str
) -> tuple[str | None, str | None]:
    """Extract reasoning/thinking content and signature from a provider response message.

    For OpenAI-compatible providers (not in PROVIDER_REASONING_FORMAT), returns (None, None)
    — use the adapter's runtime detection instead.

    Args:
        message: Provider response message dict.
        provider: Provider name key in PROVIDER_REASONING_FORMAT.

    Returns:
        Tuple of (reasoning_text, signature). Both may be None if no reasoning found.
    """
    fmt = PROVIDER_REASONING_FORMAT.get(provider)
    if fmt is None:
        return None, None

    text: str | None = None

    if fmt["message_field"] is not None:
        text = message.get(fmt["message_field"])
        return (text, None) if isinstance(text, str) and text else (None, None)

    return None, None
