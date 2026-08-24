"""Anthropic serialization package.

Provider-dialect knowledge for the Anthropic wire format: the shared
``AnthropicContentMixin`` (content-block conversion shared with the
Anthropic protocol serializer in ``llm_proxy.protocols.anthropic``) and the
registered provider serializer (in ``.serializer``).
"""

from llm_proxy.serialization.anthropic.mixin import AnthropicContentMixin

__all__ = [
    "AnthropicContentMixin",
]
