"""BuildContext for serializer operations."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from llm_proxy.models import InternalRequest

if TYPE_CHECKING:
    from llm_proxy.models import ContentBlock

UnknownFieldsPolicy = Literal["ignore", "passthrough", "error"]
UnsupportedBlockPolicy = Literal["drop", "degrade", "error"]
TargetEndpoint = Literal["chat_completions", "responses"]


@dataclass
class BuildContext:
    """Context object for serializer operations.

    Encapsulates all parameters needed for building provider requests
    and parsing provider responses, avoiding long parameter lists.

    Attributes:
        stream: Whether to enable streaming
        model: Model name override
        request_id: Request ID for correlation
        unknown_fields_policy: How to handle unknown request fields:
            'ignore' (skip), 'passthrough' (keep in body), 'error' (raise ProviderError)
        unsupported_block_policy: How to handle unsupported content blocks:
            'drop' (remove silently), 'degrade' (degrade to text), 'error' (raise ProviderError)
        base_url: Base URL override
        provider_name: Provider name for provider-specific serialization logic
        target_endpoint: Which upstream API the body targets:
            'chat_completions' or 'responses'. Defaults to 'chat_completions'.
        supported_content_blocks: Set of block types this provider can
            natively serialize.  Used by degradation logic.
        compatible_protocols: Protocol names whose wire format the provider
            can reuse directly (the WIRE_REUSE tier). Declared by the
            provider serializer, carried here so the conversion seam
            (``llm_proxy.core.conversion.plan_conversion``) can read it
            without reaching through the adapter.
        response_passthrough: Provider-metadata kill switch
            (``response_passthrough: false``) for the response-side
            WIRE_REUSE tier; when False the response is always fully parsed.
        namespace_map: OpenResponses namespace mapping (flat name ->
            [namespace, original_name]) carried over from the request so
            provider serializers can flatten history tool-call names to match
            the flattened tool definitions sent upstream.
        extra: Additional provider-specific parameters
    """

    stream: bool = False
    model: str | None = None
    request_id: str | None = None
    unknown_fields_policy: UnknownFieldsPolicy = "ignore"
    unsupported_block_policy: UnsupportedBlockPolicy = "drop"
    base_url: str | None = None
    provider_name: str = "openai"
    target_endpoint: TargetEndpoint = "chat_completions"
    supported_content_blocks: frozenset[type[ContentBlock]] = field(default_factory=frozenset)
    compatible_protocols: frozenset[str] = frozenset()
    response_passthrough: bool = True
    namespace_map: dict[str, list[str]] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_request(cls, request: InternalRequest, **kwargs: Any) -> BuildContext:
        """Create context from a InternalRequest.

        Args:
            request: The unified request to extract context from
            **kwargs: Additional parameters to override (base_url,
                unknown_fields_policy, provider_name, supported_content_blocks, etc.)

        Returns:
            BuildContext instance
        """
        base_url = kwargs.pop("base_url", None)
        unknown_fields_policy = kwargs.pop("unknown_fields_policy", "ignore")
        unsupported_block_policy = kwargs.pop("unsupported_block_policy", "drop")
        provider_name = kwargs.pop("provider_name", "openai")
        target_endpoint = kwargs.pop("target_endpoint", "chat_completions")
        supported_content_blocks = kwargs.pop("supported_content_blocks", frozenset())
        compatible_protocols = kwargs.pop("compatible_protocols", frozenset())
        response_passthrough = kwargs.pop("response_passthrough", True)
        return cls(
            stream=request.stream,
            model=request.model,
            request_id=request.metadata.request_id,
            unknown_fields_policy=unknown_fields_policy,
            unsupported_block_policy=unsupported_block_policy,
            base_url=base_url,
            provider_name=provider_name,
            target_endpoint=target_endpoint,
            supported_content_blocks=supported_content_blocks,
            compatible_protocols=compatible_protocols,
            response_passthrough=response_passthrough,
            namespace_map=request._namespace_map,
            extra=kwargs,
        )
