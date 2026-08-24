# src/llm_proxy/streaming/transformer.py
"""Base class for streaming transformers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import orjson

from llm_proxy.models.content_blocks import ContentBlock

if TYPE_CHECKING:
    from llm_proxy.models import InternalRequest


@dataclass
class StreamingUsage:
    """Standard usage information for streaming responses.

    Provides a unified interface for token usage data across different
    protocol implementations (OpenAI, Anthropic, OpenResponses).

    All streaming transformers should implement get_usage() to return
    this dataclass with all available fields populated. Cache token
    fields are optional as not all providers support them.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    # Cache tokens (optional - not all providers support these)
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    # Optional detailed token breakdowns
    prompt_tokens_details: dict[str, int] | None = None
    completion_tokens_details: dict[str, int] | None = None
    # Provider-reported cost (e.g., NanoGPT's x_nanogpt_pricing.cost, OpenRouter's cost)
    provider_reported_cost: float | None = None
    # Web search request count (server_tool_use.web_search_requests)
    web_search_requests: int | None = None


class StreamingTransformer(ABC):
    """Base class for streaming response transformers.

    This class is used in **two distinct roles**:

    1. **Protocol-side transformer** (registered on ``ProtocolEndpoint``):
       Converts *canonical* OpenAI ``chat.completion.chunk`` dicts into
       protocol-specific SSE wire format (OpenAI, Anthropic, OpenResponses).
       Defined next to the protocol module (e.g. ``protocols/openai/streaming.py``).

    2. **Provider-side chunk converter** (returned by
       ``ProviderSerializer.get_chunk_converter()``):
       Converts *provider-native* streaming chunks into the canonical
       OpenAI ``chat.completion.chunk`` dict format.  Defined alongside
       each provider's serializer (e.g. ``serialization/anthropic/``).

    The protocol-side role uses ``transform()`` which returns SSE strings.
    The provider-side role uses ``convert_chunk()`` which returns dicts.
    Both roles share ``get_accumulated_output()``, ``get_usage()``, and
    ``finalize()`` for downstream consumers like web search continuation.

    Example (protocol-side):
        transformer = OpenAIStreamingTransformer(response_id="resp_123", model="gpt-4")
        async for chunk in provider_adapter.stream_chat_completion(request):
            sse_chunk = transformer.transform(chunk)
            if sse_chunk:
                yield sse_chunk
        yield transformer.finalize()

    Example (provider-side):
        converter = AnthropicChunkConverter(model="claude-3", request_id="msg_1")
        async for frame in adapter._stream_raw_sse(request):
            data = orjson.loads(extract_data(frame))
            chunk = converter.convert_chunk(data)
            if chunk is not None:
                yield chunk
        yield "[DONE]"
    """

    def __init__(
        self,
        model: str = "",
        request_id: str | None = None,
    ):
        """Initialize the streaming transformer.

        Args:
            model: Model name used for this response
            request_id: Unique identifier for this request/response
        """
        self.response_id = request_id or ""
        self.model = model
        self._accumulated_output: list[ContentBlock] = []

    @abstractmethod
    def transform(self, chunk: str | dict[str, Any]) -> str | None:
        """Transform a raw SSE chunk from the provider.

        This method is called for each raw chunk received from the provider.
        Subclasses can implement passthrough or custom transformation.

        Args:
            chunk: Raw SSE chunk string from the provider, or a parsed chunk dict
                from adapters using a single-serialization streaming path.

        Returns:
            Transformed chunk string, or None if chunk should be filtered
        """
        ...

    @abstractmethod
    def finalize(self) -> str:
        """Generate stream end marker.

        Returns:
            Protocol-specific stream termination chunk (e.g., 'data: [DONE]\\n\\n')
        """
        ...

    def error_frames(self, exc: Exception) -> list[str]:
        """SSE frames to emit when the stream fails mid-flight.

        Error wire shaping is protocol knowledge: each protocol-side
        transformer owns its terminal error format. The default is the OpenAI
        chat-completions shape (generic error frame followed by [DONE]).
        """
        from llm_proxy.core.exceptions import ProviderError
        from llm_proxy.streaming.sse_builder import create_sse_error

        error_type = exc.error_type if isinstance(exc, ProviderError) else "api_error"
        error_dict = {"error": {"message": str(exc), "type": error_type}}
        return [create_sse_error(error_dict, include_done=True)]

    async def finalize_persistence(
        self,
        unified_request: InternalRequest,
        response_store: Any,
        event_context: Any,
    ) -> None:
        """Persist the completed streamed response, if the protocol stores responses.

        Default no-op. The OpenResponses transformer overrides this to persist
        store=true responses so follow-up ``previous_response_id``
        continuations and ``GET /v1/responses/{id}`` work.
        """
        return None

    def convert_chunk(self, chunk: dict[str, Any]) -> dict[str, Any] | None:
        """Convert a provider-native streaming chunk to canonical OpenAI chunk dict.

        This is the provider-side interface (role 2 in the class docstring).
        Subclasses that act as provider-side chunk converters should override
        this method.  The default implementation raises ``NotImplementedError``.

        Args:
            chunk: A provider-native streaming chunk dict.

        Returns:
            A canonical OpenAI ``chat.completion.chunk`` dict, or ``None``
            if the chunk should not produce a client-visible event.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement convert_chunk()")

    def finalize_chunks(self) -> list[dict[str, Any]]:
        """Return any pending chunks after the stream ends.

        Called by the adapter after the SSE stream is exhausted to flush
        any accumulated final chunks (e.g., usage, finish_reason).

        The default implementation returns an empty list.  Subclasses that
        accumulate state during streaming (e.g., ``OpenAIResponsesChunkConverter``)
        should override this method.

        Returns:
            A list of zero or more canonical OpenAI chunk dicts.
        """
        return []

    def _make_chunk(self, data: dict[str, Any]) -> str:
        """Create a SSE chunk string from a dictionary.

        Args:
            data: The data dictionary to format as SSE

        Returns:
            SSE-formatted string (e.g., 'data: {...}\\n\\n')
        """
        return f"data: {orjson.dumps(data).decode()}\n\n"

    def get_accumulated_output(self) -> list[ContentBlock]:
        """Get accumulated output content for final response.

        Returns:
            List of completed ContentBlocks accumulated during streaming.
            This can be used to construct a InternalResponse after streaming
            completes, ensuring all content is captured.
        """
        return self._accumulated_output

    def get_usage(self) -> StreamingUsage | None:
        """Get accumulated usage information from the streaming response.

        Subclasses should override this method to provide protocol-specific
        usage data extraction. The base implementation returns None.

        Returns:
            StreamingUsage object if usage data is available, None otherwise.
        """
        return None
