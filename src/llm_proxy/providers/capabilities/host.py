"""Structural host contract for the capability mixins.

The mixins in this package (chat/embedding/image/audio) call back into the
hosting adapter for HTTP transport, retries, endpoint resolution, and body
finalization. ``CapabilityHost`` declares that contract so the mixins
type-check without importing :class:`BaseHttpProvider` concretely (which
would be circular — ``providers/base.py``'s ``__init_subclass__`` inspects
this package's classes by module name).

Each mixin annotates its methods' ``self`` with a per-mixin ``*Self``
protocol below: the shared host contract plus the mixin's own hook surface
(members the mixin defines and calls through ``self``, with defaults a host
may override). ``BaseHttpProvider`` satisfies ``CapabilityHost``
structurally — see the conformance check at the bottom, which the type
checker validates.
"""

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

if TYPE_CHECKING:
    from llm_proxy.core.exceptions import ProviderError
    from llm_proxy.http.client import AsyncSession
    from llm_proxy.models import (
        InternalEmbeddingRequest,
        InternalRequest,
        InternalSpeechRequest,
        InternalTranscriptionRequest,
        InternalTranscriptionResponse,
        InternalTranslationRequest,
        InternalTranslationResponse,
    )
    from llm_proxy.providers.base import BaseHttpProvider, OutboundBody
    from llm_proxy.serialization.providers.base import ProviderSerializer

T = TypeVar("T")


class CapabilityHost(Protocol):
    """Members a capability mixin expects its hosting adapter to provide."""

    @property
    def provider_name(self) -> str: ...

    def _get_serializer(self) -> ProviderSerializer: ...

    def _resolve_endpoint_url(
        self, endpoint_type: str, default_path: str, model: str | None = None
    ) -> str: ...

    def _build_headers(
        self, auth_header: str | None = None, auth_prefix: str | None = None
    ) -> dict[str, str]: ...

    def _get_stream_timeout(self) -> tuple[float, float]: ...

    async def _get_client(self) -> AsyncSession: ...

    def _streaming_post(
        self,
        client: AsyncSession,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        files: Any = None,
        timeout: tuple[float, float],
    ) -> AbstractAsyncContextManager[Any]: ...

    async def _raise_for_stream_status(self, response: Any) -> None: ...

    async def _check_response_status(self, response: Any) -> None: ...

    def _iter_stream_lines(self, response: Any, cancel_token: Any = None) -> AsyncIterator[str]: ...

    async def _handle_http_error(self, error: Exception) -> ProviderError: ...

    async def _with_retry(
        self,
        operation: Callable[[], Awaitable[T]],
        retryable_errors: set[str] | None = None,
    ) -> T: ...

    def _with_retry_generator(
        self,
        generator_factory: Callable[[], AsyncIterator[T]],
        retryable_errors: set[str] | None = None,
        cancel_token: Any = None,
    ) -> AsyncIterator[T]: ...

    async def _post_json_response_with_retry(
        self, url: str, headers: dict[str, str], body: dict[str, Any]
    ) -> Any: ...

    def _build_outbound_body(
        self,
        request: Any,
        *,
        request_type: str,
        exempt_keys: set[str] | None = None,
        stream: bool = False,
    ) -> OutboundBody: ...


class ChatSelf(CapabilityHost, Protocol):
    """Self-type for ``ChatCapabilityMixin``: host contract + chat hooks."""

    CHAT_ENDPOINT: str

    def _stream_url(self, request: InternalRequest) -> str: ...

    def _stream_headers(self) -> dict[str, str]: ...

    def _stream_body(self, request: InternalRequest) -> dict[str, Any]: ...

    def _stream_filter_line(self, line_str: str) -> str | None: ...

    def _stream_transform_chunk(
        self, chunk: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any] | str | None: ...

    def _stream_finalize(self, context: dict[str, Any]) -> dict[str, Any] | str | None: ...


class EmbeddingSelf(CapabilityHost, Protocol):
    """Self-type for ``EmbeddingCapabilityMixin``."""

    EMBEDDINGS_ENDPOINT: str

    def _embeddings_url(self, request: InternalEmbeddingRequest) -> str: ...

    def _embeddings_headers(self) -> dict[str, str]: ...


class ImageSelf(CapabilityHost, Protocol):
    """Self-type for ``ImageCapabilityMixin``."""

    IMAGES_ENDPOINT: str
    IMAGES_EDITS_ENDPOINT: str

    def _image_generation_url(self, model: str | None = None) -> str: ...

    def _image_edit_url(self, model: str | None = None) -> str: ...

    def _build_image_request_body(self, request: Any) -> dict[str, Any]: ...

    def _stream_image_request(
        self,
        request: Any,
        *,
        url: str,
        request_type: str,
    ) -> AsyncIterator[str]: ...


class AudioSelf(CapabilityHost, Protocol):
    """Self-type for ``AudioCapabilityMixin``."""

    SPEECH_ENDPOINT: str
    TRANSCRIPTION_ENDPOINT: str
    TRANSLATION_ENDPOINT: str

    def _speech_url(self, request: InternalSpeechRequest | None = None) -> str: ...

    def _transcription_url(self, request: InternalTranscriptionRequest | None = None) -> str: ...

    def _translation_url(self, request: InternalTranslationRequest | None = None) -> str: ...

    def _build_transcription_data(
        self, request: InternalTranscriptionRequest, stream: bool = False
    ) -> tuple[dict[str, Any], dict[str, Any]]: ...

    def _build_translation_data(
        self, request: InternalTranslationRequest
    ) -> tuple[dict[str, Any], dict[str, Any]]: ...

    def _parse_usage(self, raw_usage: Any) -> Any: ...

    def _parse_transcription_response(
        self, data: dict[str, Any], format_type: str
    ) -> InternalTranscriptionResponse: ...

    def _parse_translation_response(
        self, data: dict[str, Any], format_type: str
    ) -> InternalTranslationResponse: ...

    async def _post_audio_form(
        self,
        url: str,
        request: Any,
        *,
        request_type: str,
    ) -> tuple[dict[str, Any], dict[str, str]]: ...


if TYPE_CHECKING:

    def _check_base_http_provider_satisfies_host(
        provider: BaseHttpProvider,
    ) -> CapabilityHost:
        """Static conformance check — never called at runtime.

        The annotated return type forces the type checker to verify that
        ``BaseHttpProvider`` structurally satisfies ``CapabilityHost``; a
        missing or mistyped host member fails ``ty check`` here.
        """
        return provider


__all__ = [
    "AudioSelf",
    "CapabilityHost",
    "ChatSelf",
    "EmbeddingSelf",
    "ImageSelf",
]
