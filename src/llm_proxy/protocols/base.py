"""ProtocolEndpoint dataclass for protocol configuration.

This module defines the ProtocolEndpoint dataclass that provides
configuration metadata for protocol-specific HTTP endpoints.

ProtocolEndpoint is a pure configuration interface - it defines how to mount
routes (paths), the Pydantic request model (request_model), streaming
transformers, and additional routes/middleware. All serialization logic is
handled by ProtocolSerializer (see ``llm_proxy.protocols.serializer_base``).
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, cast

from pydantic import BaseModel

from llm_proxy.streaming.transformer import StreamingTransformer


@dataclass
class ProtocolEndpoint:
    """HTTP endpoint configuration for a protocol.

    ProtocolEndpoint is a pure configuration dataclass - it defines how to
    mount routes, what Pydantic models to use, and provides protocol-specific
    metadata. Serialization logic (parsing requests and formatting responses)
    is handled by ProtocolSerializer (``llm_proxy.protocols.serializer_base``).

    Fields:
        name: Unique identifier for the protocol
            (e.g., "openai", "anthropic")
        paths: List of HTTP endpoint paths
            (e.g., ["/v1/chat/completions"])
        request_model: Pydantic model for request validation,
            or None for custom parsing
        streaming_transformer: Streaming transformer class,
            or None if not supported
        response_model: Pydantic model for response validation,
            or None for no validation
        parse_http_request: Async function for custom HTTP
            request parsing (multipart, etc.)
        tags: OpenAPI tags for documentation grouping
        description: Human-readable description for documentation
        middleware: List of async middleware functions
        additional_routes: List of additional route tuples
            (path, request_model, response_model, handler)
        on_parse_request: Optional callback invoked before parsing
            a request, receiving the raw request dict. Used for
            protocol-specific side effects like setting context vars.
        on_format_done: Optional callback invoked after response
            formatting completes. Used for protocol-specific cleanup.
        on_provider_selected: Optional callback invoked after the provider
            is selected and the adapter created, receiving the unified
            request, the selected provider name, and the FastAPI request.
            Used for protocol-specific request adjustments that depend on
            the upstream provider.
        on_response_store: Optional callback invoked before a response is
            persisted to the response store, receiving the response body
            dict, the unified request, and the raw request data. Returns the
            body dict, possibly modified (e.g. attaching materialized input).
    """

    name: str
    paths: list[str]
    request_model: type[BaseModel] | None

    streaming_transformer: (
        type[StreamingTransformer] | Callable[[], type[StreamingTransformer]] | None
    ) = None
    response_model: type[BaseModel] | None = None
    parse_http_request: Callable[[Any], Awaitable[Any]] | None = field(default=None, repr=False)
    tags: list[str] = field(default_factory=list)
    description: str = ""
    middleware: list[Callable[[Any, Any], Awaitable[None]]] = field(
        default_factory=list, repr=False
    )
    additional_routes: list[
        tuple[str, type[BaseModel], type[BaseModel] | None, Callable[..., Any]]
    ] = field(default_factory=list, repr=False)
    on_parse_request: Callable[[dict[str, Any]], None] | None = field(default=None, repr=False)
    on_format_done: Callable[[], None] | None = field(default=None, repr=False)
    on_provider_selected: Callable[[Any, str, Any], None] | None = field(default=None, repr=False)
    on_response_store: Callable[[dict[str, Any], Any, dict[str, Any]], dict[str, Any]] | None = (
        field(default=None, repr=False)
    )
    _transformer: type[StreamingTransformer] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.description:
            self.description = f"{self.name.replace('_', ' ').title()} protocol implementation"

    def get_streaming_transformer(self) -> type[StreamingTransformer] | None:
        """Resolve the streaming transformer, evaluating lazily if a factory was given."""
        if self._transformer is None and self.streaming_transformer is not None:
            if isinstance(self.streaming_transformer, type):
                self._transformer = cast(type[StreamingTransformer], self.streaming_transformer)
            else:
                self._transformer = self.streaming_transformer()
        return self._transformer


__all__ = ["ProtocolEndpoint"]
