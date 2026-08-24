"""NanoGPT provider serializer.

NanoGPT is OpenAI-compatible for chat completions but uses `reasoning` field
instead of `reasoning_content` in requests and responses. This serializer
handles the normalization and NanoGPT-specific metadata extraction.
"""

from typing import TYPE_CHECKING, Any

from llm_proxy.models import InternalResponse
from llm_proxy.models.types import Usage
from llm_proxy.providers.nanogpt.pricing import extract_nanogpt_pricing
from llm_proxy.serialization.context import BuildContext
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
from llm_proxy.serialization.providers.field_utils import (
    extract_unknown_response_fields,
)
from llm_proxy.serialization.providers.registry import register_provider_serializer

if TYPE_CHECKING:
    from llm_proxy.models import InternalRequest


@register_provider_serializer("nanogpt")
class NanoGPTProviderSerializer(ProviderSerializer):
    """NanoGPT provider serializer.

    Uses composition with OpenAI components, adding NanoGPT-specific:
    - reasoning field normalization (reasoning_content <-> reasoning)
    - NanoGPT pricing metadata extraction
    """

    _DEFAULT_PROVIDER_NAME = "nanogpt"

    def __init__(self) -> None:
        super().__init__()
        self._request_builder = OpenAIRequestBuilder()
        self._response_parser = OpenAIResponseParser(self._request_builder)

    @property
    def compatible_protocols(self) -> frozenset[str]:
        return frozenset({"openai"})

    def _build_provider_request(
        self, request: InternalRequest, context: BuildContext
    ) -> dict[str, Any]:
        """Build the NanoGPT request body."""
        return self._request_builder.build(request, context)

    def parse_provider_response(
        self, response: dict[str, Any], model: str | None = None, **kwargs: Any
    ) -> InternalResponse:
        result = self._response_parser.parse(response, model=model, **kwargs)

        pricing = extract_nanogpt_pricing(response)
        if pricing:
            if result.usage is None:
                result.usage = Usage()
            if "input_tokens" in pricing:
                result.usage.input_tokens = pricing["input_tokens"]
            if "output_tokens" in pricing:
                result.usage.output_tokens = pricing["output_tokens"]
            result.usage.total_tokens = result.usage.input_tokens + result.usage.output_tokens
            # Merge all non-usage pricing fields into provider_info so
            # amount/currency/error/paymentSource/cache tokens/etc. are preserved.
            for key, value in pricing.items():
                if key in ("input_tokens", "output_tokens"):
                    continue
                result.provider_info[key] = value

        unknown_fields = self.extract_unknown_response_fields(response)
        result.provider_info.update(unknown_fields)

        return result

    def get_chunk_converter(self, model: str = "", request_id: str = "") -> IdentityChunkConverter:
        """Return an identity converter — NanoGPT chunks are already canonical."""
        return IdentityChunkConverter()

    @staticmethod
    def _known_response_fields() -> set[str]:
        return OpenAIResponseParser.known_response_fields() | {"x_nanogpt_pricing"}

    def extract_unknown_response_fields(self, response: dict[str, Any]) -> dict[str, Any]:
        return extract_unknown_response_fields(response, self._known_response_fields())
