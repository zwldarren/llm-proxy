"""Adapter registry and factory system."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, ClassVar, TypeVar

from llm_proxy.core.exceptions import AdapterNotFoundError, ValidationError
from llm_proxy.core.registry_base import ThreadSafeRegistry
from llm_proxy.models import (
    InternalEmbeddingRequest,
    InternalEmbeddingResponse,
    InternalImageRequest,
    InternalImageResponse,
    InternalRequest,
    InternalResponse,
    InternalSpeechRequest,
    InternalSpeechResponse,
    InternalTranscriptionRequest,
    InternalTranscriptionResponse,
    InternalTranslationRequest,
    InternalTranslationResponse,
)
from llm_proxy.models.image import InternalImageEditRequest
from llm_proxy.observability.logger import get_logger

logger = get_logger(__name__)


class BaseAdapter(ABC):
    """Adapter interface for provider implementations."""

    _provider_name: str | None = None

    #: Protocols whose wire format this provider speaks natively. When the
    #: client protocol matches, requests/responses may be forwarded verbatim
    #: (native passthrough; see llm_proxy.core.conversion). Empty = never.
    #:
    #: Checklist before declaring a protocol native here — the verbatim tier
    #: bypasses every transformation that lives on the canonical
    #: parse/rebuild path, so confirm the provider does not rely on any of:
    #:   * reasoning-content echo repairs (e.g. ``_requires_reasoning_echo``;
    #:     this is why DeepSeek keeps ``openai`` off this set),
    #:   * developer→system role normalization (post-parse mutation, defeated
    #:     by raw reuse),
    #:   * content-block degradation / validation (``supported_content_blocks``
    #:     and ``unsupported_block_policy`` never run),
    #:   * ``unknown_fields_policy`` enforcement (the raw body is not filtered),
    #:   * parameter overrides applied after parse (they re-parse the raw
    #:     body, so they DO reach passthrough — but anything applied only to
    #:     the parsed request does not).
    #: Also keep this set disjoint from the provider serializer's
    #: ``compatible_protocols`` (wire-reuse tier); tests/core/test_conversion_tiers.py
    #: enforces both properties.
    native_protocols: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the adapter.

        Args:
            **kwargs: Subclass-specific configuration parameters.
        """
        super().__init__()

    async def close(self) -> None:  # noqa: B027
        """Clean up resources held by this adapter.

        Default implementation is a no-op. Subclasses that create
        their own HTTP clients should override this to close them.
        """

    def set_retry_recorder(self, recorder: Any) -> None:  # noqa: B027
        """Attach a same-provider retry-attempt recorder.

        Adapters that own a ``RetryPolicy`` override this to forward the
        recorder so retries are surfaced to the request pipeline. The default
        implementation is a no-op so non-HTTP adapters are unaffected.
        """

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, _exc_val, _exc_tb):
        await self.close()
        return False

    def __init_subclass__(cls, *, provider_name: str | None = None, **kwargs: Any):
        super().__init_subclass__(**kwargs)
        if provider_name is not None:
            cls._provider_name = provider_name

    @property
    def provider_name(self) -> str:
        if self._provider_name:
            return self._provider_name
        msg = "provider_name must be implemented by subclass or via provider_name kwarg"
        raise NotImplementedError(msg)

    @abstractmethod
    async def chat_completion(
        self,
        request: InternalRequest,
        **kwargs: Any,
    ) -> InternalResponse:
        """Process a chat completion request."""

    @abstractmethod
    async def stream_chat_completion(
        self,
        request: InternalRequest,
        cancel_token=None,
        **kwargs: Any,
    ) -> AsyncIterator[str | dict[str, Any]]:
        """Process a streaming chat completion request and return chunks.

        Yields either:
        - dict: Parsed chunk for single-serialization path in protocol layer
        - str: Special markers like "[DONE]" for stream termination
        """

    def supports_native_request(
        self, protocol_name: str | None, request: InternalRequest | None = None
    ) -> bool:
        """Whether a request can be forwarded verbatim to the upstream API.

        Data-driven: True when the client protocol is in ``native_protocols``
        and the request-scoped veto ``allows_native_request`` does not fire.
        See llm_proxy.core.conversion for the full decision.
        """
        if protocol_name is None or protocol_name not in self.native_protocols:
            return False
        return request is None or self.allows_native_request(request)

    def allows_native_request(self, request: InternalRequest) -> bool:
        """Request-scoped veto for native request passthrough. Default: allow.

        Vetoes conversations materialized from the proxy's own response store
        (``previous_response_id`` pointing at a proxy-stored response, or
        ``item_reference`` items that reference it): the upstream has no
        knowledge of those proxy-local ids, so the body must be rebuilt, not
        forwarded verbatim.
        """
        return not request.previous_response_materialized

    def native_body_hook(self, body: dict[str, Any]) -> dict[str, Any]:
        """Family-specific repairs applied to the native passthrough body.

        Called once by ``core.conversion.prepare_native_body`` after the
        shared preparation (copy, None-strip, model substitution, stream
        flag). Default: no repairs. Adapters with native protocols override
        this to contribute family knowledge — Anthropic-shaped bodies get
        structural message repairs, Responses-shaped bodies get input-item id
        stripping. The body is already a fresh copy, but nested structures
        (e.g. the messages list) are still shared with the stashed raw body:
        deep-copy before mutating them.
        """
        return body

    def supports_native_streaming(self, protocol_name: str) -> bool:
        """Return True if this adapter can yield protocol-native SSE directly.

        When True, StreamingProcessor skips the canonical dict → transformer
        round-trip and forwards native SSE lines directly to the client.
        """
        return protocol_name in self.native_protocols

    async def stream_chat_completion_native(
        self,
        request: InternalRequest,
        cancel_token=None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream chat completion yielding protocol-native SSE lines.

        Only called when supports_native_streaming() returns True for the
        current protocol. Adapters override this to skip the canonical
        OpenAI dict intermediate format.
        """
        raise NotImplementedError(f"{self.provider_name} does not support native streaming")

    async def embeddings(
        self,
        request: InternalEmbeddingRequest,
        **kwargs: Any,
    ) -> InternalEmbeddingResponse:
        """Generate embeddings for the input text."""
        raise NotImplementedError(f"{self.provider_name} does not support embeddings.")

    def from_image_provider_format(self, response: dict[str, Any]) -> InternalImageResponse:
        """Convert image generation response from provider's native format to unified format.

        Default implementation for OpenAI-compatible image generation responses.
        Providers with different formats should override this method.

        Args:
            response: The raw image generation response from the provider

        Returns:
            A InternalImageResponse instance
        """
        from llm_proxy.models.image import ImageData
        from llm_proxy.models.types import Usage

        data_list: list[ImageData] = []
        for item in response.get("data", []):
            if isinstance(item, dict):
                data_list.append(
                    ImageData(
                        url=item.get("url"),
                        b64_json=item.get("b64_json"),
                        revised_prompt=item.get("revised_prompt"),
                    )
                )

        created = response.get("created")
        if not isinstance(created, int) or isinstance(created, bool):
            raise ValidationError(
                message="Image provider response is missing required integer field 'created'",
                code="api_error",
                status_code=502,
            )

        # Parse usage if present (GPT image models)
        usage = None
        raw_usage = response.get("usage")
        if raw_usage and isinstance(raw_usage, dict):
            # gpt-image: input_tokens_details.{text_tokens,image_tokens}
            prompt_details = None
            input_details = raw_usage.get("input_tokens_details")
            if isinstance(input_details, dict):
                from llm_proxy.models.types import PromptTokensDetails

                prompt_details = PromptTokensDetails(
                    text_tokens=input_details.get("text_tokens"),
                    image_tokens=input_details.get("image_tokens"),
                )
            usage = Usage(
                input_tokens=raw_usage.get("input_tokens", 0),
                output_tokens=raw_usage.get("output_tokens", 0),
                total_tokens=raw_usage.get("total_tokens"),
                prompt_tokens_details=prompt_details,
            )

        return InternalImageResponse(
            created=created,
            data=data_list,
            background=response.get("background"),
            output_format=response.get("output_format"),
            quality=response.get("quality"),
            size=response.get("size"),
            usage=usage,
        )

    def from_image_edit_provider_format(self, response: dict[str, Any]) -> InternalImageResponse:
        """Convert image edit response from provider's native format to unified format.

        Shares the same response format as image generation.
        """
        return self.from_image_provider_format(response)

    async def image_generation(
        self,
        request: InternalImageRequest,
        **kwargs: Any,
    ) -> InternalImageResponse:
        """Generate images from a text prompt."""
        raise NotImplementedError(f"{self.provider_name} does not support image generation.")

    async def stream_image_generation(
        self,
        request: InternalImageRequest,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream image generation from a text prompt.

        Yields SSE event strings (e.g., 'data: {...}\\n\\n').
        """
        raise NotImplementedError(
            f"{self.provider_name} does not support streaming image generation."
        )

    async def stream_image_edit(
        self,
        request: InternalImageEditRequest,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream image edit progress events from a prompt and reference images.

        Yields SSE event strings (e.g., 'data: {...}\\n\\n').
        """
        raise NotImplementedError(f"{self.provider_name} does not support streaming image editing.")

    async def image_edit(
        self,
        request: InternalImageEditRequest,
        **kwargs: Any,
    ) -> InternalImageResponse:
        """Edit an existing image based on a prompt."""
        raise NotImplementedError(
            f"{self.provider_name} does not support image editing. "
            "Configure a separate image editing provider."
        )

    async def speech(
        self,
        request: InternalSpeechRequest,
        **kwargs: Any,
    ) -> InternalSpeechResponse:
        """Generate audio from text."""
        raise NotImplementedError(f"{self.provider_name} does not support speech generation.")

    async def stream_speech(
        self,
        request: InternalSpeechRequest,
        **kwargs: Any,
    ) -> AsyncIterator[bytes]:
        """Stream audio generation from text."""
        raise NotImplementedError(
            f"{self.provider_name} does not support streaming speech generation."
        )

    def speech_stream_media_type(self, request: InternalSpeechRequest) -> str | None:
        """Media type of the bytes produced by :meth:`stream_speech`.

        Providers whose output format differs from the requested
        ``response_format`` (e.g. Gemini TTS always outputs PCM, wrapped as
        WAV) override this so the HTTP response advertises an honest
        Content-Type. Returning None falls back to deriving the media type
        from ``request.response_format``.
        """
        return None

    async def transcription(
        self,
        request: InternalTranscriptionRequest,
        **kwargs: Any,
    ) -> InternalTranscriptionResponse:
        """Transcribe audio to text."""
        raise NotImplementedError(f"{self.provider_name} does not support transcription.")

    async def stream_transcription(
        self,
        request: InternalTranscriptionRequest,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream audio transcription."""
        raise NotImplementedError(f"{self.provider_name} does not support streaming transcription.")

    async def translation(
        self,
        request: InternalTranslationRequest,
        **kwargs: Any,
    ) -> InternalTranslationResponse:
        """Translate audio to English text."""
        raise NotImplementedError(f"{self.provider_name} does not support translation.")


T = TypeVar("T", bound=BaseAdapter)

# Thread-safe registry for adapter classes
_ADAPTER_REGISTRY = ThreadSafeRegistry[type[BaseAdapter]]()


@dataclass
class AdapterConfig:
    """Configuration for creating an adapter instance.

    This dataclass consolidates all adapter configuration parameters,
    making the factory function cleaner and more maintainable.
    """

    provider_name: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    connect_timeout: float = 10.0
    read_timeout: float = 600.0
    max_retries: int = 3
    custom_headers: dict[str, str] | None = None
    http_client: Any | None = None
    http_client_manager: Any | None = None
    disable_http2: bool = True
    max_connections: int = 200
    max_keepalive: int = 200
    # Additional provider-specific config stored here
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> AdapterConfig:
        """Create AdapterConfig from keyword arguments.

        Extracts known parameters and stores the rest in 'extra'.
        """
        known_params = {
            "provider_name",
            "api_key",
            "base_url",
            "connect_timeout",
            "read_timeout",
            "max_retries",
            "custom_headers",
            "http_client",
            "http_client_manager",
            "disable_http2",
            "max_connections",
            "max_keepalive",
        }

        config_kwargs = {}
        extra = {}

        for key, value in kwargs.items():
            if key in known_params:
                config_kwargs[key] = value
            else:
                extra[key] = value

        return cls(**config_kwargs, extra=extra)


def register_adapter(
    provider_name: str,
):
    """Decorator to register an adapter class for a provider."""

    def decorator(cls: type[BaseAdapter]):
        cls._provider_name = provider_name
        _ADAPTER_REGISTRY.register(provider_name, cls)
        return cls

    return decorator


def get_adapter(provider: str, **kwargs: Any) -> BaseAdapter:
    """Get or create an adapter for the specified provider.

    Resolves the provider against the adapter registry.

    Args:
        provider: The provider name (e.g., "openai", "anthropic", "ollama")
        **kwargs: Additional configuration parameters

    Returns:
        A configured adapter instance

    Raises:
        ValueError: If provider is None or empty string
        AdapterNotFoundError: If no adapter is available for the provider
    """
    if not provider or not provider.strip():
        raise ValidationError(
            message="Provider name cannot be None or empty. "
            "Please ensure provider_config.type is configured correctly."
        )

    normalized_provider = provider.strip()

    # Build AdapterConfig from kwargs, injecting provider_name if not present
    if "provider_name" not in kwargs:
        kwargs["provider_name"] = normalized_provider
    config = AdapterConfig.from_kwargs(**kwargs)

    adapter_class = _ADAPTER_REGISTRY.get(provider)
    if adapter_class is not None:
        return adapter_class(config=config)

    provider_lower = provider.lower()
    adapter_class_lower = _ADAPTER_REGISTRY.get(provider_lower)
    if adapter_class_lower is not None:
        return adapter_class_lower(config=config)

    raise AdapterNotFoundError(
        f"No adapter registered for provider '{provider}'. "
        f"Registered providers: {list_providers()}. "
        "Ensure providers are imported before calling get_adapter()."
    )


@dataclass
class ProviderTypeInfo:
    """Branding metadata for a registered provider type (adapter).

    Consumed by the admin UI provider catalog (``GET /api/config/provider-types``)
    so the frontend can render provider types without a per-provider static
    list. Adapters declare the values as classvars; adapters without metadata
    fall back to the type name and no icon.
    """

    type: str
    name_en: str
    name_zh: str
    icon_id: str | None = None
    icon_variant: str = "color"


def list_providers() -> list[str]:
    """List all registered providers."""
    return _ADAPTER_REGISTRY.list_all()


def list_provider_types() -> list[ProviderTypeInfo]:
    """Return all registered provider types with their branding metadata.

    Metadata is read from optional classvars on the adapter class
    (``DISPLAY_NAME_EN``, ``DISPLAY_NAME_ZH``, ``LOBE_ICON_ID``,
    ``LOBE_ICON_VARIANT``). Missing names fall back to the type name and
    missing icons to None, so unannotated adapters degrade gracefully.
    Results are sorted by display name for a stable UI list.
    """
    types: list[ProviderTypeInfo] = []
    for name, cls in _ADAPTER_REGISTRY.get_all().items():
        icon_id = getattr(cls, "LOBE_ICON_ID", None) or None
        types.append(
            ProviderTypeInfo(
                type=name,
                name_en=getattr(cls, "DISPLAY_NAME_EN", "") or name,
                name_zh=getattr(cls, "DISPLAY_NAME_ZH", "") or name,
                icon_id=icon_id,
                # Variant is only meaningful alongside an icon; inherit
                # pitfalls (e.g. a family base declaring a mono icon while a
                # subclass has none) are normalized to the default.
                icon_variant=(getattr(cls, "LOBE_ICON_VARIANT", "") or "color")
                if icon_id
                else "color",
            )
        )
    return sorted(types, key=lambda info: (info.name_en.casefold(), info.type))


__all__ = [
    "BaseAdapter",
    "AdapterConfig",
    "ProviderTypeInfo",
    "register_adapter",
    "get_adapter",
    "list_providers",
    "list_provider_types",
]
