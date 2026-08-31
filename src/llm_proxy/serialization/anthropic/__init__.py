"""Anthropic serialization package.

Provider-dialect knowledge for the Anthropic wire format: the shared
``AnthropicContentMixin`` (content-block conversion shared with the
Anthropic protocol serializer in ``llm_proxy.protocols.anthropic``) and the
registered provider serializer (in ``.serializer``).
"""

from llm_proxy.serialization.anthropic.mixin import AnthropicContentMixin

# Usage keys that are Anthropic-native extensions beyond the official Usage
# shape but are safe to pass through verbatim: SDKs ignore unknown keys.
# ``output_tokens_details``/``service_tier`` belong to the full ``Usage``
# object (message_start); ``speed``/``iterations`` are the beta fast-mode and
# compaction/fallback counters whose top-level token counts exclude the
# compaction iterations, so stripping them would break per-iteration cost
# accounting. Shared by the protocol transformer, the provider serializer,
# and the streaming converter so the set stays single-truth.
ANTHROPIC_USAGE_EXTENSION_KEYS: tuple[str, ...] = (
    "output_tokens_details",
    "service_tier",
    "speed",
    "iterations",
)

__all__ = [
    "ANTHROPIC_USAGE_EXTENSION_KEYS",
    "AnthropicContentMixin",
]
