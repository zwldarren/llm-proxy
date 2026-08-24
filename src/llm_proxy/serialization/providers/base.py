"""ProviderSerializer base class.

Converts between InternalRequest/InternalResponse and provider API format.
Provider adapters use this to build provider-specific request bodies and parse
provider responses.
"""

from abc import ABC, abstractmethod
from typing import Any

from llm_proxy.models import ContentBlock, ConversionTier, InternalRequest, InternalResponse
from llm_proxy.models.embedding import (
    EmbeddingData,
    InternalEmbeddingRequest,
    InternalEmbeddingResponse,
)
from llm_proxy.models.types import Usage
from llm_proxy.serialization.context import BuildContext
from llm_proxy.serialization.providers.field_utils import (
    extract_unknown_response_fields as _extract_unknown_fields,
)
from llm_proxy.streaming.transformer import StreamingTransformer


class IdentityChunkConverter(StreamingTransformer):
    """A no-op chunk converter for providers that already emit OpenAI-format chunks.

    When a provider's streaming response is already in canonical OpenAI
    ``chat.completion.chunk`` dict format (e.g., DeepSeek, OpenRouter, NanoGPT,
    Chutes), no conversion is needed.  This converter implements the
    ``convert_chunk()`` interface as an identity function.

    It extends ``StreamingTransformer`` purely so it satisfies the
    ``ProviderSerializer.get_chunk_converter()`` return type; it has no
    accumulation, no usage tracking, and no SSE formatting.  Adapters that use
    identity converters handle ``[DONE]`` and other stream-level concerns
    themselves, so the protocol-side ``transform()``/``finalize()`` methods are
    unused no-ops.

    Despite the old name, this is NOT the native passthrough seam
    (``llm_proxy.core.conversion``): chunks still round-trip through dict
    form and the protocol-side transformer; nothing is forwarded verbatim.
    """

    def convert_chunk(self, chunk: dict[str, Any]) -> dict[str, Any] | None:
        """Return the chunk unchanged (identity).

        Args:
            chunk: A provider-native chunk that is already in canonical
                   OpenAI ``chat.completion.chunk`` format.

        Returns:
            The same dict, or None if the chunk is empty/null.
        """
        if not chunk:
            return None
        return chunk

    def finalize_chunks(self) -> list[dict[str, Any]]:
        """Return any pending chunks after the stream ends.

        Identity converters have no accumulated state, so this
        always returns an empty list.
        """
        return []

    def transform(self, chunk: str | dict[str, Any]) -> str | None:
        """Protocol-side transform — unused for provider-side identity."""
        return None

    def finalize(self) -> str:
        """Stream end marker — unused; adapters emit ``[DONE]`` directly."""
        return "[DONE]"


__all__ = ["ProviderSerializer", "IdentityChunkConverter"]


class ProviderSerializer(ABC):
    """Convert between Internal* models and a specific provider's API format.

    Each provider (OpenAI, Anthropic, Gemini, Ollama, etc.) has its own
    API format. ProviderSerializer handles the conversion in both directions.

    Subclasses can list compatible wire-format protocols via ``compatible_protocols``.
    """

    _DEFAULT_PROVIDER_NAME: str = ""

    def __init__(self) -> None:
        super().__init__()
        self._registered_provider_name: str = ""

    @property
    def provider_name(self) -> str:
        """Provider name as registered, falling back to the class default."""
        return self._registered_provider_name or self._DEFAULT_PROVIDER_NAME

    # ------------------------------------------------------------------
    # Content block support — subclasses override with the set of block
    # types they can natively serialize.  Used by ``should_degrade_block``
    # for capability-based degradation.
    # ------------------------------------------------------------------

    supported_content_blocks: frozenset[type[ContentBlock]] = frozenset()

    @property
    def compatible_protocols(self) -> frozenset[str]:
        """Protocol names whose wire format this provider can reuse directly.

        Pure capability data read by the conversion seam
        (``llm_proxy.core.conversion.plan_conversion``) via
        ``BuildContext.compatible_protocols``: when the provider natively
        speaks the client protocol's wire format (e.g. OpenAI protocol →
        OpenAI-compatible provider), the seam may prepare the outbound body
        from the stashed raw body (WIRE_REUSE) instead of calling
        ``build_provider_request``.
        """
        return frozenset()

    def filter_extra_for_body(self, extra: dict[str, Any]) -> dict[str, Any]:
        """Return the subset of ``request.extra`` this provider accepts on the body.

        The default accepts every key (the legacy open merge). Dialects that
        reject unknown top-level fields override this with their whitelist so
        the adapter's generic body-merge path (“_merge_extra”) applies the
        same policy the request builder does.
        """
        return extra

    def build_provider_request(
        self,
        request: InternalRequest,
        context: BuildContext | None = None,
    ) -> dict[str, Any]:
        """Build a provider-specific request body from InternalRequest.

        Always a full conversion. The tier decision and all raw-reuse body
        preparation live in the conversion seam
        (``llm_proxy.core.conversion.plan_conversion`` /
        ``prepare_wire_reuse_body``), which runs before the serializer is
        called; this method only rebuilds from the parsed request.
        """
        if context is None:
            context = BuildContext.from_request(
                request,
                provider_name=self.provider_name,
                supported_content_blocks=self.supported_content_blocks,
            )

        body = self._build_provider_request(request, context)
        request.conversion_tier = ConversionTier.FULL_CONVERSION
        return body

    @abstractmethod
    def _build_provider_request(
        self,
        request: InternalRequest,
        context: BuildContext,
    ) -> dict[str, Any]:
        """Build provider-specific request body (subclass implementation)."""
        ...

    @abstractmethod
    def parse_provider_response(
        self, response: dict[str, Any], model: str | None = None, **kwargs: Any
    ) -> InternalResponse:
        """Parse a provider response into InternalResponse.

        Args:
            response: Raw response dict from the provider
            model: Optional model name for the response
            **kwargs: Provider-specific keyword arguments (e.g., request_id, logprobs)

        Returns:
            InternalResponse instance
        """
        ...

    @abstractmethod
    def get_chunk_converter(self, model: str = "", request_id: str = "") -> StreamingTransformer:
        """Get the provider-side chunk converter for streaming responses.

        The converter transforms provider-native streaming chunks into the
        canonical OpenAI ``chat.completion.chunk`` dict format used by the
        protocol-side streaming transformer.

        This is the streaming equivalent of ``parse_provider_response()``:
        just as non-streaming responses go through the serializer to become
        ``InternalResponse``, streaming chunks go through the converter to
        become canonical OpenAI chunk dicts.

        Args:
            model: Model name for this response
            request_id: Request ID for correlation

        Returns:
            StreamingTransformer instance whose ``convert_chunk()`` method
            accepts provider-native chunk dicts and returns canonical OpenAI
            chunk dicts.
        """
        ...

    @staticmethod
    def _known_response_fields() -> set[str]:
        """Return the set of response fields that this serializer handles explicitly.

        Fields not in this set are candidates for provider_info passthrough.
        Override in subclasses to declare explicitly handled response fields.
        """
        return set()

    def extract_unknown_response_fields(self, response: dict[str, Any]) -> dict[str, Any]:
        """Extract fields not in _known_response_fields() for storage in provider_info.

        Override in subclasses to capture provider-specific metadata
        (e.g., OpenRouter's id_provider, cost fields).

        Args:
            response: Raw provider response dict

        Returns:
            Dict of unknown fields (empty by default)
        """
        return _extract_unknown_fields(response, self._known_response_fields())

    def build_provider_embedding_request(self, request: InternalEmbeddingRequest) -> dict[str, Any]:
        """Build a provider-specific embedding request body.

        Default implementation produces OpenAI-compatible format.
        Override for providers with different embedding request formats.

        Args:
            request: The unified embedding request

        Returns:
            Provider-specific request body dict
        """
        body: dict[str, Any] = {
            "model": request.model,
            "input": request.input,
        }
        if request.dimensions is not None:
            body["dimensions"] = request.dimensions
        if request.encoding_format is not None:
            body["encoding_format"] = request.encoding_format
        return body

    def parse_provider_embedding_response(
        self, response: dict[str, Any], model: str = ""
    ) -> InternalEmbeddingResponse:
        """Parse a provider embedding response into InternalEmbeddingResponse.

        Default implementation handles OpenAI-compatible format.
        Override for providers with different embedding response formats.

        Args:
            response: Raw response dict from the provider
            model: Model name for the response

        Returns:
            InternalEmbeddingResponse instance
        """
        data_list: list[EmbeddingData] = []
        for item in response.get("data", []):
            if isinstance(item, dict):
                data_list.append(
                    EmbeddingData(
                        embedding=item.get("embedding", []),
                        index=item.get("index", 0),
                    )
                )

        usage = None
        if "usage" in response:
            usage_data = response["usage"]
            usage = Usage(
                input_tokens=usage_data.get("prompt_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )

        return InternalEmbeddingResponse(
            model=response.get("model", model),
            data=data_list,
            usage=usage,
        )


__all__ = ["ProviderSerializer"]
