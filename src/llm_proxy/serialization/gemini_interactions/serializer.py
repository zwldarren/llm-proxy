"""Gemini Interactions provider serializer.

Registered as ``"gemini-interactions"`` (the registry resolves
``llm_proxy.serialization.gemini_interactions.serializer``). Implements the
same ``ProviderSerializer`` interface as the legacy
``GeminiProviderSerializer`` — ``build_provider_request`` /
``parse_provider_response`` / ``get_chunk_converter`` — so the Gemini adapter
can switch dialects through its ``api_variant`` configuration without any
changes to the client protocols or internal models.

Mixin modules live alongside in this package. Audio helpers are shared with
the legacy serializer via ``serialization/gemini/speech.py``.
"""

from llm_proxy.serialization.gemini_interactions import (
    GeminiInteractionsConversationMixin,
    GeminiInteractionsRequestBuilderMixin,
    GeminiInteractionsResponseParserMixin,
)
from llm_proxy.serialization.providers.base import ProviderSerializer
from llm_proxy.serialization.providers.registry import register_provider_serializer


@register_provider_serializer("gemini-interactions")
class GeminiInteractionsProviderSerializer(
    GeminiInteractionsConversationMixin,
    GeminiInteractionsResponseParserMixin,
    GeminiInteractionsRequestBuilderMixin,
    ProviderSerializer,
):
    """Gemini Interactions API provider serializer.

    Handles Unified -> Interactions request bodies (stateless Step-array
    ``input``, ``generation_config``, ``response_format``) and Interactions
    -> Unified responses (``steps[]`` timeline, new usage vocabulary).
    """

    _DEFAULT_PROVIDER_NAME = "gemini-interactions"

    def get_chunk_converter(self, model: str = "", request_id: str = ""):
        """Return the Interactions-specific chunk converter.

        Converts Interactions SSE events (``step.start``/``step.delta``/
        ``interaction.completed`` …) to canonical OpenAI
        ``chat.completion.chunk`` dicts.
        """
        from llm_proxy.serialization.gemini_interactions.streaming_converter import (
            InteractionsStreamingTransformer,
        )

        return InteractionsStreamingTransformer(model=model, request_id=request_id)
