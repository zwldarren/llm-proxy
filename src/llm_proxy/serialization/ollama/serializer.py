"""Ollama provider serializer.

Registered here (ADR-0003); mixin modules live alongside in this package.
"""

from llm_proxy.serialization.ollama import (
    OllamaConversationMixin,
    OllamaRequestBuilderMixin,
    OllamaResponseParserMixin,
    OllamaStreamingMixin,
)
from llm_proxy.serialization.providers.base import ProviderSerializer
from llm_proxy.serialization.providers.registry import register_provider_serializer


@register_provider_serializer("ollama")
class OllamaProviderSerializer(
    OllamaConversationMixin,
    OllamaStreamingMixin,
    OllamaResponseParserMixin,
    OllamaRequestBuilderMixin,
    ProviderSerializer,
):
    """Ollama provider serializer.

    Handles Unified -> Ollama native format conversion for provider requests,
    and Ollama native -> Unified conversion for provider responses.
    """

    _DEFAULT_PROVIDER_NAME = "ollama"

    def get_chunk_converter(self, model: str = "", request_id: str = ""):
        """Return an Ollama-specific chunk converter.

        Converts Ollama native JSON-line chunks to canonical OpenAI
        ``chat.completion.chunk`` dicts.
        """
        from llm_proxy.serialization.ollama import OllamaChunkConverter

        return OllamaChunkConverter(model=model, request_id=request_id)
