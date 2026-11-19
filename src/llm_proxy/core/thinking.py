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

#: Keys inside ``thinking`` dicts that map onto named ``ThinkingConfig`` fields;
#: any remaining non-None keys (e.g. GLM's ``clear_thinking``) ride ``extra``.
_KNOWN_THINKING_KEYS: frozenset[str] = frozenset(
    {
        "type",
        "budget_tokens",
        "token_budget",
        "display",
        "effort",
        "reasoning_effort",
    }
)

_SUPPRESSED_DISPLAY_VALUES: frozenset[str] = frozenset({"none", "off", "hidden", "omitted"})

PROVIDER_REASONING_FORMAT: frozendict[str, frozendict[str, Any]] = frozendict(
    {
        "anthropic": frozendict(
            {
                "message_field": None,
                "stream_delta_field": "thinking",
                "content_block_type": "thinking",
                "redacted_block_type": "redacted_thinking",
                "signature_supported": True,
            }
        ),
        "gemini": frozendict(
            {
                "message_field": None,
                "stream_delta_field": "thought",
                "content_block_type": "thought",
                "signature_supported": True,
            }
        ),
        "ollama": frozendict(
            {
                "message_field": "thinking",
                "stream_delta_field": "thinking",
                "content_block_type": None,
                "signature_supported": False,
            }
        ),
    }
)


def normalize_thinking(data: dict[str, Any]) -> ThinkingConfig | None:
    """Normalize protocol-specific thinking parameters into ThinkingConfig.

    Handles:
    - Anthropic / Cohere / GLM ``thinking`` dict or boolean
    - OpenAI ``reasoning_effort`` string
    - OpenResponses / OpenRouter ``reasoning`` dict
    - Anthropic ``output_config.effort``

    When multiple fields are present, explicit ``thinking`` wins so a provider-
    native disable switch (``thinking: {"type": "disabled"}``) or extension key
    (e.g. GLM ``clear_thinking``) is never overwritten by a sibling effort key.

    Args:
        data: Raw request data dict that may contain thinking parameters.

    Returns:
        ThinkingConfig if any thinking parameter is found, else None.
    """
    thinking = data.get("thinking")
    if thinking is True:
        return ThinkingConfig(type="enabled")
    if thinking is False:
        return ThinkingConfig(type="disabled")
    if isinstance(thinking, dict):
        raw_type = thinking.get("type")
        thinking_type = str(raw_type) if raw_type is not None else "enabled"

        budget_tokens = thinking.get("budget_tokens")
        if budget_tokens is None:
            budget_tokens = thinking.get("token_budget")

        raw_effort = thinking.get("effort")
        if raw_effort is None:
            raw_effort = thinking.get("reasoning_effort")
        effort = str(raw_effort).lower() if raw_effort is not None else None

        if effort is None and thinking_type != "disabled":
            top_effort = data.get("reasoning_effort")
            if top_effort is not None:
                effort = str(top_effort).lower()
            else:
                oc = data.get("output_config")
                if isinstance(oc, dict):
                    oc_effort = oc.get("effort")
                    if oc_effort is not None:
                        effort = str(oc_effort).lower()

        extra_fields = {
            k: v for k, v in thinking.items() if k not in _KNOWN_THINKING_KEYS and v is not None
        }
        return ThinkingConfig(
            type=thinking_type,
            budget_tokens=budget_tokens,
            display=thinking.get("display"),
            effort=effort,
            extra=extra_fields or None,
        )

    reasoning_effort = data.get("reasoning_effort")
    if reasoning_effort is not None:
        return thinking_config_from_reasoning_effort(str(reasoning_effort))

    reasoning = data.get("reasoning")
    if isinstance(reasoning, dict):
        thinking_type: str | None = None
        effort: str | None = None
        budget_tokens: int | None = None
        display: str | None = None

        raw_enabled = reasoning.get("enabled")
        raw_effort = reasoning.get("effort")
        raw_max_tokens = reasoning.get("max_tokens")
        raw_exclude = reasoning.get("exclude")
        raw_summary = reasoning.get("summary")

        if raw_enabled is False:
            thinking_type = "disabled"
        else:
            if raw_effort is not None:
                effort_lower = str(raw_effort).lower()
                if effort_lower == "none":
                    thinking_type = "disabled"
                else:
                    thinking_type = "adaptive"
                    effort = effort_lower
            if isinstance(raw_max_tokens, int) and not isinstance(raw_max_tokens, bool):
                budget_tokens = raw_max_tokens
                if thinking_type is None:
                    thinking_type = "enabled"
            if raw_enabled is True and thinking_type is None:
                thinking_type = "enabled"

        if raw_exclude is True:
            display = "omitted"
        elif isinstance(raw_summary, str) and raw_summary:
            display = raw_summary

        if any(v is not None for v in (thinking_type, budget_tokens, effort, display)):
            return ThinkingConfig(
                type=thinking_type,
                budget_tokens=budget_tokens,
                effort=effort,
                display=display,
            )

    output_config = data.get("output_config")
    if isinstance(output_config, dict):
        effort = output_config.get("effort")
        if effort is not None:
            # Bare effort without a ``thinking`` block leaves ``type=None`` so
            # Anthropic rebuilds do not invent a ``thinking`` key the client omitted.
            return ThinkingConfig(type=None, effort=str(effort).lower())

    return None


def thinking_config_from_reasoning_effort(effort: str) -> ThinkingConfig:
    """Convert an effort string into ThinkingConfig without fabricating budgets."""
    effort_lower = effort.lower()
    if effort_lower == "none":
        return ThinkingConfig(type="disabled")
    return ThinkingConfig(type="adaptive", effort=effort_lower)


def thinking_config_to_reasoning_effort(config: ThinkingConfig) -> str | None:
    """Convert ThinkingConfig to OpenAI reasoning_effort string.

    Returns explicit ``effort``, ``"none"`` for ``disabled``, or bucketed effort
    only when ``budget_tokens`` is set. Bare ``type="enabled"`` returns ``None``
    rather than synthesizing a default effort.
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
    return None


def resolve_thinking(request: Any) -> ThinkingConfig | None:
    """Resolve the canonical ThinkingConfig that provider builders should use.

    This is the **single source of truth** for thinking parameters across
    all provider serializers.

    Precedence (highest first):
    1. ``request.params.thinking`` — populated by every protocol parser via
       :func:`normalize_thinking`.
    2. ``request.params.anthropic.output_config.effort`` — direct fallback for
       synthetically-constructed ``InternalRequest`` objects.
    3. ``request.params.openai.reasoning_effort`` — direct fallback for
       synthetically-constructed ``InternalRequest`` objects.
    """
    params = request.params
    if params.thinking is not None:
        return params.thinking
    if params.anthropic is not None and params.anthropic.output_config:
        effort = params.anthropic.output_config.get("effort")
        if effort is not None:
            return ThinkingConfig(type=None, effort=str(effort).lower())
    if params.openai is not None and params.openai.reasoning_effort is not None:
        return thinking_config_from_reasoning_effort(str(params.openai.reasoning_effort))
    return None


def convert_to_openai(config: ThinkingConfig | None) -> dict[str, Any] | None:
    """Convert ThinkingConfig to OpenAI Chat Completions provider format.

    Emits ``{"reasoning_effort": ...}`` when an effort value is explicit or
    derivable from ``disabled`` / ``budget_tokens`` (preserving ``config.extra``
    inside ``thinking`` if both are present), or ``{"thinking": {...}}`` when
    only non-adaptive thinking fields are present.
    """
    if config is None:
        return None

    effort = thinking_config_to_reasoning_effort(config)
    if effort:
        result: dict[str, Any] = {"reasoning_effort": effort}
        if config.extra and config.type is not None:
            result["thinking"] = {"type": config.type, **config.extra}
        return result

    if config.type in ("adaptive", "between_tools"):
        return None

    if config.type is not None or config.budget_tokens is not None or config.extra:
        thinking_dict: dict[str, Any] = {}
        if config.type is not None:
            thinking_dict["type"] = config.type
        if config.budget_tokens is not None:
            thinking_dict["budget_tokens"] = config.budget_tokens
        if config.extra:
            thinking_dict.update(config.extra)
        return {"thinking": thinking_dict}

    return None


def convert_to_anthropic(config: ThinkingConfig | None) -> dict[str, Any] | None:
    """Convert ThinkingConfig to Anthropic Messages provider format.

    Emits ``thinking`` when ``config.type`` is present (keeping ``budget_tokens``,
    ``display``, and ``extra`` intact) and ``output_config.effort`` whenever
    ``config.effort`` is set, without rewriting or stripping user values.
    """
    if config is None:
        return None

    result: dict[str, Any] = {}
    if config.type is not None:
        thinking_dict: dict[str, Any] = {"type": config.type}
        if config.budget_tokens is not None:
            thinking_dict["budget_tokens"] = config.budget_tokens
        if config.display is not None:
            thinking_dict["display"] = config.display
        if config.extra:
            thinking_dict.update(config.extra)
        result["thinking"] = thinking_dict

    if config.effort is not None:
        result["output_config"] = {"effort": config.effort.lower()}

    return result or None


def _include_thoughts_from_display(display: str | None) -> bool:
    """Whether to request thought summaries based on ``ThinkingConfig.display``."""
    return (display or "").lower() not in _SUPPRESSED_DISPLAY_VALUES


def convert_to_gemini(config: ThinkingConfig | None) -> dict[str, Any] | None:
    """Convert ThinkingConfig to Gemini 3+ ``thinkingConfig`` format.

    Passes explicit effort levels verbatim in uppercase without model-name
    filtering or level rewriting. Bare ``enabled``/``adaptive`` omits
    ``thinkingLevel`` so the upstream model's default applies.
    """
    if config is None:
        return None

    if config.type == "disabled" or (config.effort or "").lower() == "none":
        return {"thinkingConfig": {"thinkingLevel": "MINIMAL"}}

    level: str | None = None
    if config.effort is not None and config.effort.strip():
        level = config.effort.strip().upper()
    elif config.budget_tokens is not None:
        if config.budget_tokens < 8000:
            level = "LOW"
        elif config.budget_tokens > 24000:
            level = "HIGH"
        else:
            level = "MEDIUM"
    elif config.type == "between_tools":
        return None

    thinking_cfg: dict[str, Any] = {
        "includeThoughts": _include_thoughts_from_display(config.display)
    }
    if level is not None:
        thinking_cfg["thinkingLevel"] = level
    return {"thinkingConfig": thinking_cfg}


def convert_to_gemini_interactions(config: ThinkingConfig | None) -> dict[str, Any] | None:
    """Convert ThinkingConfig to Gemini Interactions ``generation_config`` format.

    Emits lowercase ``thinking_level`` verbatim when effort or budget is set,
    and ``thinking_summaries`` when ``config.display`` is present.
    """
    if config is None:
        return None

    if config.type == "disabled" or (config.effort or "").lower() == "none":
        return {"thinking_level": "minimal"}

    level: str | None = None
    if config.effort is not None and config.effort.strip():
        level = config.effort.strip().lower()
    elif config.budget_tokens is not None:
        if config.budget_tokens < 8000:
            level = "low"
        elif config.budget_tokens > 24000:
            level = "high"
        else:
            level = "medium"
    elif config.type == "between_tools":
        level = None

    result: dict[str, Any] = {}
    if level is not None:
        result["thinking_level"] = level
    if config.display is not None:
        display_lower = config.display.lower()
        result["thinking_summaries"] = (
            "none" if display_lower in _SUPPRESSED_DISPLAY_VALUES else "auto"
        )
    return result or None


def convert_to_gemini_legacy(config: ThinkingConfig | None) -> dict[str, Any] | None:
    """Convert ThinkingConfig to Gemini 2.5 ``thinkingBudget`` format."""
    if config is None:
        return None

    if config.type == "disabled" or (config.effort or "").lower() == "none":
        return {"thinkingConfig": {"thinkingBudget": 0}}

    if config.budget_tokens is not None:
        budget = config.budget_tokens
    elif config.effort is not None:
        effort_lower = config.effort.lower()
        if effort_lower in ("minimal", "low"):
            budget = 1024
        elif effort_lower == "medium":
            budget = 4096
        elif effort_lower in ("high", "xhigh", "max"):
            budget = 8192
        else:
            budget = 4096
    elif config.type == "between_tools":
        budget = 1024
    else:
        budget = -1

    return {
        "thinkingConfig": {
            "thinkingBudget": budget,
            "includeThoughts": _include_thoughts_from_display(config.display),
        }
    }


def convert_to_ollama(config: ThinkingConfig | None) -> bool | str | None:
    """Convert ThinkingConfig to Ollama provider ``think`` parameter."""
    if config is None:
        return None

    if config.type == "disabled":
        return False

    if config.effort is not None:
        effort_lower = config.effort.lower()
        if effort_lower == "none":
            return False
        if effort_lower:
            return effort_lower

    if config.type == "between_tools":
        return "low"

    if config.budget_tokens is not None:
        if config.budget_tokens < 8000:
            return "low"
        if config.budget_tokens > 24000:
            return "high"
        return "medium"

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
