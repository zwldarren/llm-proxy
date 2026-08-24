"""Gemini provider serializer.

Registered here (ADR-0003); mixin modules live alongside in this package.
The GeminiStreamingTransformer lives in .streaming_converter.
"""

from typing import Any

from llm_proxy.serialization.gemini import (
    GeminiConversationMixin,
    GeminiEmbeddingsMixin,
    GeminiRequestBuilderMixin,
    GeminiResponseParserMixin,
    sanitize_gemini_schema,
)
from llm_proxy.serialization.providers.base import ProviderSerializer
from llm_proxy.serialization.providers.registry import register_provider_serializer


@register_provider_serializer("gemini")
class GeminiProviderSerializer(
    GeminiConversationMixin,
    GeminiEmbeddingsMixin,
    GeminiResponseParserMixin,
    GeminiRequestBuilderMixin,
    ProviderSerializer,
):
    """Gemini Provider serializer.

    Handles Unified -> Gemini conversion for provider requests,
    and Gemini -> Unified conversion for provider responses.
    """

    _DEFAULT_PROVIDER_NAME = "gemini"

    @classmethod
    def _sanitize_gemini_schema(cls, schema: dict[str, Any] | None) -> dict[str, Any]:
        return sanitize_gemini_schema(schema)

    def get_chunk_converter(self, model: str = "", request_id: str = ""):
        """Return a Gemini-specific chunk converter.

        Converts Gemini native streaming chunks to canonical OpenAI
        ``chat.completion.chunk`` dicts.
        """
        from llm_proxy.serialization.gemini.streaming_converter import GeminiStreamingTransformer

        return GeminiStreamingTransformer(model=model, request_id=request_id)
