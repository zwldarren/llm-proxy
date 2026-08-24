"""Chat Completions provider serializer — shared by openrouter and deepseek."""

from typing import Any

from llm_proxy.models import (
    AudioBlock,
    DocumentBlock,
    FileBlock,
    ImageBlock,
    InternalResponse,
    RedactedThinkingBlock,
    RefusalBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    VideoBlock,
)
from llm_proxy.serialization.openai.components.request_builder import (
    OpenAIRequestBuilder,
)
from llm_proxy.serialization.openai.components.response_parser import (
    OpenAIResponseParser,
)
from llm_proxy.serialization.providers.base import (
    IdentityChunkConverter,
    ProviderSerializer,
)
from llm_proxy.serialization.providers.registry import register_provider_serializer


@register_provider_serializer("openrouter")
@register_provider_serializer("deepseek")
class OpenAIProviderSerializer(ProviderSerializer):
    """OpenAI provider serializer.

    Converts between InternalRequest/InternalResponse and OpenAI Chat Completions API format.
    Uses composition: delegates request building and response parsing to dedicated components.
    """

    _DEFAULT_PROVIDER_NAME = "openai"
    supported_content_blocks = frozenset(
        {
            TextBlock,
            ImageBlock,
            AudioBlock,
            FileBlock,
            DocumentBlock,
            VideoBlock,
            ToolUseBlock,
            ToolResultBlock,
            ThinkingBlock,
            RedactedThinkingBlock,
            RefusalBlock,
        }
    )

    def __init__(self) -> None:
        super().__init__()
        self._request_builder = OpenAIRequestBuilder()
        self._response_parser = OpenAIResponseParser(self._request_builder)

    @property
    def compatible_protocols(self) -> frozenset[str]:
        return frozenset({"openai"})

    def _build_provider_request(self, request: Any, context: Any) -> dict[str, Any]:
        return self._request_builder.build(request, context)

    def parse_provider_response(
        self, response: dict[str, Any], model: str | None = None, **kwargs: Any
    ) -> InternalResponse:
        return self._response_parser.parse(response, model=model, **kwargs)

    def get_chunk_converter(self, model: str = "", request_id: str = "") -> IdentityChunkConverter:
        """Return an identity converter — OpenAI-compatible chunks are already canonical.

        Providers that speak the Chat Completions API (DeepSeek, OpenRouter, etc.)
        already emit chunks in canonical OpenAI ``chat.completion.chunk`` format.
        No provider-native → canonical conversion is needed.
        """
        return IdentityChunkConverter()

    @staticmethod
    def _known_response_fields() -> set[str]:
        return OpenAIResponseParser.known_response_fields()

    def extract_unknown_response_fields(self, response: dict[str, Any]) -> dict[str, Any]:
        from llm_proxy.serialization.providers.field_utils import (
            extract_unknown_response_fields,
        )

        return extract_unknown_response_fields(response, self._known_response_fields())


__all__ = [
    "OpenAIProviderSerializer",
]
