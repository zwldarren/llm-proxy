"""Anthropic serialization package.

Provider-dialect knowledge for the Anthropic wire format: the shared
``AnthropicContentMixin`` (content-block conversion shared with the
Anthropic protocol serializer in ``llm_proxy.protocols.anthropic``) and the
registered provider serializer (in ``.serializer``).

The passthrough seam helpers (``normalize_anthropic_messages``,
``parse_usage_and_provider_extras``) are public exports: provider adapters
with declared native Anthropic endpoints (anthropic, openai_compatible
native passthrough) call them directly on raw upstream bodies.
"""

from typing import Any

# Usage keys that are Anthropic-native extensions beyond the official Usage
# shape but are safe to pass through verbatim: SDKs ignore unknown keys.
# ``output_tokens_details``/``service_tier`` belong to the full ``Usage``
# object (message_start); ``speed``/``iterations`` are the beta fast-mode and
# compaction/fallback counters whose top-level token counts exclude the
# compaction iterations, so stripping them would break per-iteration cost
# accounting. Shared by the protocol transformer, the provider serializer,
# and the streaming converter so the set stays single-truth.
#
# Defined before the submodule imports below: ``.serializer`` and
# ``.streaming_converter`` import it back from this package.
ANTHROPIC_USAGE_EXTENSION_KEYS: tuple[str, ...] = (
    "output_tokens_details",
    "service_tier",
    "speed",
    "iterations",
)

#: The official response-side ``Usage.service_tier`` enum (docs/en/api —
#: Usage object): "standard", "priority", or "batch".
ANTHROPIC_SERVICE_TIER_RESPONSE_VALUES: frozenset[str] = frozenset(
    {"standard", "priority", "batch"}
)

#: OpenAI-dialect ``service_tier`` values with a clear Anthropic equivalent.
#: Everything else (e.g. "enterprise") has none and is dropped rather than
#: leaking an illegal enum value to schema-validating clients.
_OPENAI_SERVICE_TIER_RESPONSE_ALIASES: frozendict[str, str] = frozendict(
    {
        "default": "standard",
        "auto": "standard",
        "flex": "standard",
        "scale": "standard",
    }
)


def normalize_service_tier_for_anthropic(value: Any) -> str | None:
    """Map an upstream ``service_tier`` onto the official response enum.

    The request direction maps into ``auto``/``standard_only`` (see the provider
    serializer); responses report the tier that served the request, whose
    official enum is ``standard``/``priority``/``batch``. Native Anthropic
    values pass through unchanged; OpenAI values fold onto the closest
    official tier; unknown values return ``None`` so the field is omitted.
    """
    if not isinstance(value, str) or not value:
        return None
    if value in ANTHROPIC_SERVICE_TIER_RESPONSE_VALUES:
        return value
    return _OPENAI_SERVICE_TIER_RESPONSE_ALIASES.get(value)


from llm_proxy.serialization.anthropic.mixin import AnthropicContentMixin  # noqa: E402
from llm_proxy.serialization.anthropic.serializer import (  # noqa: E402
    normalize_anthropic_messages,
    parse_usage_and_provider_extras,
)

__all__ = [
    "ANTHROPIC_USAGE_EXTENSION_KEYS",
    "AnthropicContentMixin",
    "normalize_anthropic_messages",
    "normalize_service_tier_for_anthropic",
    "parse_usage_and_provider_extras",
]
