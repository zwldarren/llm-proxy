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

from llm_proxy.serialization.anthropic.mixin import AnthropicContentMixin  # noqa: E402
from llm_proxy.serialization.anthropic.serializer import (  # noqa: E402
    normalize_anthropic_messages,
    parse_usage_and_provider_extras,
)

__all__ = [
    "ANTHROPIC_USAGE_EXTENSION_KEYS",
    "AnthropicContentMixin",
    "normalize_anthropic_messages",
    "parse_usage_and_provider_extras",
]
