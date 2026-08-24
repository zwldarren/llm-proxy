"""NanoGPT provider adapter.

This provider uses direct HTTP calls to NanoGPT's OpenAI-compatible API.
NanoGPT provides access to various LLM models with reasoning support.

Reasoning normalization and ThinkingBlock extraction are handled by
NanoGPTProviderSerializer. This adapter handles only streaming normalization
and pricing metadata extraction in the streaming path.
"""

from typing import Any

from llm_proxy.core.adapter import AdapterConfig, register_adapter
from llm_proxy.observability.logger import get_logger
from llm_proxy.providers.nanogpt.pricing import extract_nanogpt_pricing
from llm_proxy.providers.openai_compatible._base import OpenAICompatibleBase
from llm_proxy.serialization.providers import get_provider_serializer

logger = get_logger(__name__)

_serializer = get_provider_serializer("nanogpt")


@register_adapter("nanogpt")
class NanoGPTAdapter(OpenAICompatibleBase):
    """NanoGPT provider using direct HTTP calls to OpenAI-compatible API.

    Non-streaming reasoning normalization and ThinkingBlock extraction
    are handled by NanoGPTProviderSerializer. This adapter handles
    streaming pricing metadata extraction; the reasoning-field
    preference recording and chunk normalization are shared with
    ``OpenAICompatibleBase``.
    """

    _DEFAULT_PROVIDER_NAME = "nanogpt"

    #: Branding for the admin provider catalog (GET /api/config/provider-types).
    DISPLAY_NAME_EN = "NanoGPT"
    DISPLAY_NAME_ZH = "NanoGPT"
    LOBE_ICON_ID = None

    DEFAULT_BASE_URL = "https://nano-gpt.com/api/v1"
    # NanoGPT always expects assistant reasoning as the `reasoning` field.
    _REASONING_FIELD = "reasoning"

    def __init__(
        self,
        *,
        config: AdapterConfig | None = None,
        **kwargs: Any,
    ):
        if config is not None:
            super().__init__(config=config)
        else:
            kwargs.setdefault("provider_name", "nanogpt")
            kwargs.setdefault("base_url", self.DEFAULT_BASE_URL)
            super().__init__(**kwargs)

    def _stream_transform_chunk(
        self, chunk: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any] | None:
        # Shared with the base adapter: records the reasoning-field
        # preference (per routed model + upstream alias) and normalizes
        # ``reasoning`` -> ``reasoning_content`` before the client sees it.
        super()._stream_transform_chunk(chunk, context)

        pricing = extract_nanogpt_pricing(chunk)
        if pricing:
            if "usage" not in chunk or chunk["usage"] is None:
                chunk["usage"] = {}

            usage = chunk["usage"]
            if "input_tokens" in pricing:
                usage["prompt_tokens"] = pricing["input_tokens"]
            if "output_tokens" in pricing:
                usage["completion_tokens"] = pricing["output_tokens"]
            if "prompt_tokens" in usage and "completion_tokens" in usage:
                usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
            # Carry the full pricing envelope and any provider-specific sub-fields
            # so streaming responses expose amount/currency/cache/webSearch/etc.
            for key, value in pricing.items():
                if key in ("input_tokens", "output_tokens"):
                    continue
                usage[key] = value

        return chunk


__all__ = ["NanoGPTAdapter"]
