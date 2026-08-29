"""Base class for native provider implementations."""

from abc import ABC
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar, cast

import orjson
from orjson import JSONDecodeError

from llm_proxy.config.types.provider import ProviderConfig
from llm_proxy.core.adapter import AdapterConfig, BaseAdapter
from llm_proxy.core.conversion import (
    plan_conversion,
    prepare_native_body,
    prepare_wire_reuse_body,
)
from llm_proxy.core.errors.utils import extract_retry_after
from llm_proxy.core.exceptions import ProviderError
from llm_proxy.core.request_type import RequestType
from llm_proxy.core.utils import quiet_aclose
from llm_proxy.http.client import AsyncSession
from llm_proxy.models import (
    ConversionTier,
    InternalResponse,
    Usage,
)
from llm_proxy.observability.logger import get_logger
from llm_proxy.providers.components import ErrorTranslator, HttpTransport, RetryPolicy
from llm_proxy.serialization.openai.serializer import parse_usage_from_response

if TYPE_CHECKING:
    from llm_proxy.models import (
        InternalRequest,
    )
    from llm_proxy.models.provider import ProviderModelInfo

logger = get_logger(__name__)

#: Module prefix of the capability mixins. ``__init_subclass__`` uses it to
#: detect misordered bases without importing the mixin classes (capabilities/*
#: already import this module — a reverse import would be circular).
_CAPABILITY_MODULE_PREFIX = "llm_proxy.providers.capabilities."


# Informational end-to-end response headers worth forwarding to the caller
# (upstream response headers pass through minus hop-by-hop headers).
_PASSTHROUGH_RESPONSE_HEADERS = frozenset(
    {
        "retry-after",
        "x-request-id",
        # Anthropic returns ``request-id`` (no x- prefix); Claude Code reads it
        # for diagnostics.
        "request-id",
        "openai-version",
        "openai-processing-ms",
    }
)


def extract_rate_limit_headers(headers) -> dict[str, str]:
    """Capture upstream rate-limit and informational headers for passthrough.

    Preserves ``x-ratelimit-*``, ``RateLimit-*``, ``anthropic-ratelimit-*``,
    ``Retry-After`` and a small set of end-to-end informational headers
    (``x-request-id``, ``request-id``, ``openai-version``,
    ``openai-processing-ms``) so the API layer can forward them to the caller.
    """
    if not headers:
        return {}
    captured: dict[str, str] = {}
    for key, value in headers.items():
        lower = key.lower()
        if (
            lower in _PASSTHROUGH_RESPONSE_HEADERS
            or lower.startswith("x-ratelimit-")
            or lower.startswith("ratelimit-")
            or lower.startswith("anthropic-ratelimit-")
        ):
            captured[key] = value
    return captured


T = TypeVar("T")


@dataclass
class OutboundBody:
    """Holds the final outbound body for a provider request.

    Exactly one of json_body or form_data should be set, depending on
    the endpoint type. files holds multipart file uploads when present.
    """

    json_body: dict[str, Any] | None = None
    form_data: dict[str, Any] | None = None
    files: Any = None

    def __post_init__(self) -> None:
        if (self.json_body is None) == (self.form_data is None):
            raise ValueError("Exactly one of json_body or form_data must be set")


# Stream timeouts: read_timeout resets on each data chunk (idle timeout)
STREAM_READ_TIMEOUT = 600.0  # 10 min idle


class BaseHttpProvider(BaseAdapter, ABC):
    """Base class for native provider implementations.

    Streaming uses the template method pattern via ``stream_chat_completion``
    with hooks: _stream_url, _stream_headers, _stream_body,
    _stream_filter_line, _stream_transform_chunk, _stream_finalize.

    Adapters using the template method: OpenAI, OpenRouter, Anthropic.
    Adapters with custom streaming: Gemini, Ollama, NanoGPT, OpenAI Responses.
    """

    _DEFAULT_PROVIDER_NAME: str = ""
    DEFAULT_BASE_URL: str = ""

    _adapter_type: str = ""

    AUTH_HEADER: str = "Authorization"
    AUTH_PREFIX: str = "Bearer"
    EXTRA_HEADERS: dict[str, str] = {}

    _api_key: str | None = None

    #: Native embedding parameters that must survive the unknown-fields policy
    #: (e.g. Ollama's keep_alive/truncate/options). Adapters with native
    #: embedding params override this; empty by default.
    _EMBEDDING_EXEMPT_EXTRA_KEYS: frozenset[str] = frozenset()

    def __init__(self, config: AdapterConfig | None = None, **kwargs: Any):
        if config is None:
            config = AdapterConfig.from_kwargs(**kwargs)

        provider_name = config.provider_name
        api_key = config.get_api_key() if isinstance(config, ProviderConfig) else config.api_key
        base_url = config.base_url
        connect_timeout = config.connect_timeout
        read_timeout = config.read_timeout
        max_retries = config.max_retries
        custom_headers = config.custom_headers
        http_client = config.http_client
        extra = config.extra

        # Preserve the class-level _provider_name (set by @register_adapter)
        # before the instance attribute overrides it.  This is the adapter type
        # name used for serializer lookup.
        class_level_name = type(self)._provider_name
        if class_level_name:
            self._adapter_type = class_level_name
        elif self._DEFAULT_PROVIDER_NAME:
            self._adapter_type = self._DEFAULT_PROVIDER_NAME

        self._provider_name = provider_name
        self._api_key = api_key
        resolved_base_url = base_url or self.DEFAULT_BASE_URL
        if resolved_base_url:
            normalized_base_url = resolved_base_url.rstrip("/")
            # Tolerate provider configs that double-write /v1 ("{base}/v1"
            # combined with endpoint paths that already carry /v1) by
            # collapsing the trailing "/v1/v1" loop.
            while normalized_base_url.endswith("/v1/v1"):
                normalized_base_url = normalized_base_url[: -len("/v1")]
            self._base_url = normalized_base_url
        else:
            self._base_url = None
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout
        self._max_retries = max_retries
        self._custom_headers = custom_headers or {}
        self._endpoint_base_urls = extra.get("endpoint_base_urls", {})
        self._extra_config = extra
        # Informational/rate-limit headers from the most recent streaming
        # upstream response, stashed so the API layer can forward them to
        # the client once the StreamingResponse is created (upstream response
        # headers pass through minus hop-by-hop headers).
        self._last_stream_response_headers: dict[str, str] = {}

        self._transport = HttpTransport(
            provider_name=self._provider_name,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            http_client=http_client,
            disable_http2=config.disable_http2,
            max_connections=config.max_connections,
            max_keepalive=config.max_keepalive,
        )
        self._error_translator = ErrorTranslator(
            provider_name=self._provider_name,
        )
        self._retry = RetryPolicy(
            max_retries=max_retries,
            provider_name=self._provider_name or "",
            error_translator=self._error_translator,
        )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Fail fast when a capability mixin is placed after the base class.

        Adapter correctness depends on base-class order
        ``(*CapabilityMixins, BaseHttpProvider)``: a mixin placed after
        ``BaseHttpProvider`` is silently shadowed by ``BaseAdapter``'s
        ``NotImplementedError`` interface stubs. Detect the misordering at
        class-creation time instead of at first call.
        """
        super().__init_subclass__(**kwargs)
        mro = cls.__mro__
        provider_idx = mro.index(BaseHttpProvider)
        misplaced = [
            base.__name__
            for base in mro[provider_idx + 1 :]
            if base.__module__.startswith(_CAPABILITY_MODULE_PREFIX)
        ]
        if misplaced:
            raise TypeError(
                f"{cls.__name__}: capability mixins ({', '.join(misplaced)}) "
                "must precede BaseHttpProvider in the base-class list — "
                "otherwise BaseAdapter's NotImplementedError stubs shadow "
                "their implementations. "
                "Use class ...(ChatCapabilityMixin, ..., BaseHttpProvider)."
            )

    @property
    def provider_name(self) -> str:
        return self._provider_name or self._DEFAULT_PROVIDER_NAME

    def _wrap_parse_error(self, exception: Exception, raw_response: Any) -> ProviderError:
        return ProviderError(
            message=f"Failed to parse provider response: {exception}",
            error_type="api_error",
            status_code=500,
            provider_name=self.provider_name,
            original_error={"raw_response": raw_response, "parse_error": str(exception)},
        )

    async def _get_client(self) -> AsyncSession:
        client = getattr(self, "_http_client", None)
        if client is not None:
            return client
        return await self._transport.get_client()

    async def close(self) -> None:
        await self._transport.close()

    def set_retry_recorder(self, recorder: Any) -> None:
        """Forward a retry-attempt recorder to this adapter's RetryPolicy."""
        self._retry.set_recorder(recorder)

    def _build_headers(
        self,
        auth_header: str | None = None,
        auth_prefix: str | None = None,
    ) -> dict[str, str]:
        if auth_header is None:
            auth_header = self.AUTH_HEADER
        if auth_prefix is None:
            auth_prefix = self.AUTH_PREFIX
        headers: dict[str, str] = {"Content-Type": "application/json"}
        headers.update(self.EXTRA_HEADERS)
        if self._custom_headers:
            headers.update(self._custom_headers)
        if self._api_key and auth_header:
            headers[auth_header] = f"{auth_prefix} {self._api_key}".strip()
        return headers

    async def _with_retry(
        self,
        operation: Callable[[], Awaitable[T]],
        retryable_errors: set[str] | None = None,
    ) -> T:
        return await self._retry.execute(operation, retryable_errors)

    def _with_retry_generator(
        self,
        generator_factory: Callable[[], AsyncIterator[T]],
        retryable_errors: set[str] | None = None,
        cancel_token=None,
    ) -> AsyncIterator[T]:
        return self._retry.execute_generator(
            generator_factory, retryable_errors=retryable_errors, cancel_token=cancel_token
        )

    async def _handle_http_error(self, error: Exception) -> ProviderError:
        return await self._error_translator.translate_error(error)

    # ------------------------------------------------------------------
    # Shared HTTP helpers (Items 1, 3, 6)
    # ------------------------------------------------------------------

    async def _post_json(
        self, url: str, headers: dict[str, str], body: dict[str, Any]
    ) -> dict[str, Any]:
        client = await self._get_client()
        response = await client.post(url, headers=headers, json=body)
        await self._check_response_status(response)
        return response.json()

    async def _post_json_with_retry(
        self, url: str, headers: dict[str, str], body: dict[str, Any]
    ) -> dict[str, Any]:
        return await self._with_retry(lambda: self._post_json(url, headers, body))

    async def _post_json_response_with_retry(
        self, url: str, headers: dict[str, str], body: dict[str, Any]
    ) -> Any:
        """Post JSON and return the raw HTTP response so callers can read headers."""

        async def _make_request():
            client = await self._get_client()
            response = await client.post(url, headers=headers, json=body)
            await self._check_response_status(response)
            return response

        return await self._with_retry(_make_request)

    async def _post_raw(self, url: str, headers: dict[str, str], body: dict[str, Any]) -> bytes:
        client = await self._get_client()
        response = await client.post(url, headers=headers, json=body)
        await self._check_response_status(response)
        content = response.content
        if content is None:
            raise ProviderError(
                message="Empty response body",
                error_type="api_error",
                status_code=response.status_code,
                provider_name=self.provider_name,
            )
        return content

    async def _post_raw_with_retry(
        self, url: str, headers: dict[str, str], body: dict[str, Any]
    ) -> bytes:
        return await self._with_retry(lambda: self._post_raw(url, headers, body))

    async def _raise_for_stream_status(self, response) -> None:
        if response.status_code is None or response.status_code < 400:
            return
        await response.aread()
        raw_text = response.text
        try:
            error_body = orjson.loads(raw_text)
        except JSONDecodeError:
            msg = (
                raw_text[:500]
                if raw_text
                else f"HTTP {response.status_code}: Empty error response body"
            )
            error_body = {"error": {"message": msg}}
        error = self._parse_error_response(response.status_code, error_body)
        # Preserve upstream Retry-After so the retry policy can honor it.
        retry_after = extract_retry_after(response.headers)
        if retry_after is not None:
            if error.original_error is None or not isinstance(error.original_error, dict):
                error.original_error = {}
            error.original_error["retry_after"] = retry_after
        raise error

    async def _iter_stream_lines(self, response, cancel_token=None) -> AsyncIterator[str]:
        lines = cast(AsyncIterator[bytes], response.iter_lines())
        try:
            async for line in lines:
                if cancel_token and cancel_token.is_set():
                    logger.debug(f"{self.provider_name} stream cancelled")
                    break
                if not line:
                    continue
                yield line.decode("utf-8") if isinstance(line, bytes) else line
        finally:
            # Close the inner iterator explicitly instead of abandoning it to
            # asyncio's async-gen GC finalizer (see quiet_aclose).
            await quiet_aclose(lines)

    def _stash_stream_response_headers(self, response) -> None:
        # Stash upstream response headers so the API layer can forward them
        # (request-id, ratelimit-*, ...) once the client StreamingResponse
        # is created.
        self._last_stream_response_headers = extract_rate_limit_headers(
            getattr(response, "headers", None)
        )

    def _stream_raw_sse(
        self,
        url: str,
        body: dict[str, Any],
        cancel_token=None,
    ) -> AsyncIterator[str]:
        """Forward upstream SSE blocks verbatim (one complete block per yield).

        Uses ``response.iter_lines()`` directly to preserve the empty-line
        delimiter that terminates each SSE block (the shared framing loop for
        native passthrough streams; see llm_proxy.core.conversion).
        """
        headers = self._build_headers()
        stream_timeout = self._get_stream_timeout()

        async def _generator():
            client = await self._get_client()
            try:
                async with self._streaming_post(
                    client,
                    url,
                    headers=headers,
                    json=body,
                    timeout=stream_timeout,
                ) as response:
                    await self._raise_for_stream_status(response)
                    self._stash_stream_response_headers(response)

                    buf: list[str] = []
                    async for raw_line in response.iter_lines():
                        if cancel_token and cancel_token.is_set():
                            logger.debug(f"{self.provider_name} stream cancelled")
                            break
                        line_str = (
                            raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                        )
                        for part in line_str.split("\n"):
                            if part == "":
                                if buf:
                                    yield "\n".join(buf) + "\n\n"
                                    buf = []
                            else:
                                buf.append(part)
                    if buf and not (cancel_token and cancel_token.is_set()):
                        yield "\n".join(buf) + "\n\n"

            except ProviderError:
                raise
            except Exception as e:
                error = await self._handle_http_error(e)
                raise error from e

        return _generator()

    # ------------------------------------------------------------------
    # Shared response parsing helper (Item 2)
    # ------------------------------------------------------------------

    def _parse_response(
        self,
        serializer,
        response: dict[str, Any],
        *,
        model: str,
        request_id: str | None = None,
        request: InternalRequest | None = None,
        **kwargs: Any,
    ) -> InternalResponse:
        # Stamp the response-side tier for observability (the mirror of
        # request.conversion_tier); the passthrough chokepoint stamps
        # NATIVE_PASSTHROUGH instead.
        if request is not None:
            request.response_tier = ConversionTier.FULL_CONVERSION
        try:
            result = serializer.parse_provider_response(response, model=model, **kwargs)
        except Exception as e:
            raise self._wrap_parse_error(e, response) from e
        if request_id:
            result.request_id = request_id
        return result

    # ------------------------------------------------------------------
    # Chat context builder (minimal – refined in Task 5)
    # ------------------------------------------------------------------

    def _get_serializer(self):
        """Return the serializer for this provider.

        Uses ``_adapter_type`` (the @register_adapter name, e.g. "ollama")
        instead of ``provider_name`` which may be a display label like
        "Ollama Cloud".  Subclasses may override for provider-specific
        serializer selection (e.g. OpenAICompatibleBase returns the
        openrouter serializer).
        """
        from llm_proxy.serialization.providers import get_provider_serializer

        return get_provider_serializer(self._adapter_type)

    def _build_chat_context(self, request: Any) -> Any:
        """Build a BuildContext with policies for provider request construction."""
        from llm_proxy.serialization.context import BuildContext

        serializer = self._get_serializer()
        supported = serializer.supported_content_blocks
        return BuildContext.from_request(
            request,
            base_url=self._base_url,
            provider_name=self.provider_name,
            target_endpoint=self._target_endpoint(),
            unknown_fields_policy=self._resolve_field_policy(),
            unsupported_block_policy=self._resolve_block_policy(),
            supported_content_blocks=supported,
            compatible_protocols=serializer.compatible_protocols,
            # Provider-metadata kill switch for the response-side WIRE_REUSE
            # tier (same pattern as the native_passthrough kill switch).
            response_passthrough=bool(self._extra_config.get("response_passthrough", True)),
        )

    # ------------------------------------------------------------------
    # Raw builder hooks (overridable)
    # ------------------------------------------------------------------
    # Default implementations raise so the dispatch below fails loudly with a
    # clear message when an adapter reaches an endpoint without the matching
    # capability mixin. Mixins (ChatCapabilityMixin etc.) provide the real
    # implementations and must precede this class in the adapter's base list
    # (enforced by __init_subclass__).

    def _build_chat_raw(self, request: Any, context: Any) -> dict[str, Any]:
        """Build the raw chat body. Provided by ``ChatCapabilityMixin``."""
        raise NotImplementedError(
            f"{type(self).__name__}: include ChatCapabilityMixin or override _build_chat_raw"
        )

    def _build_embedding_raw(self, request: Any) -> dict[str, Any]:
        """Build the raw embedding body. Provided by ``EmbeddingCapabilityMixin``."""
        raise NotImplementedError(
            f"{type(self).__name__}: include EmbeddingCapabilityMixin "
            "or override _build_embedding_raw"
        )

    def _build_speech_raw(self, request: Any) -> dict[str, Any]:
        """Build the raw speech body. Provided by ``AudioCapabilityMixin``."""
        raise NotImplementedError(
            f"{type(self).__name__}: include AudioCapabilityMixin or override _build_speech_raw"
        )

    def _build_image_raw(self, request: Any) -> dict[str, Any]:
        """Build the raw image-generation body. Provided by ``ImageCapabilityMixin``."""
        raise NotImplementedError(
            f"{type(self).__name__}: include ImageCapabilityMixin or override _build_image_raw"
        )

    def _build_image_edit_raw(self, request: Any) -> tuple[dict[str, Any], Any]:
        """Build the raw image-edit body (+ files). Provided by ``ImageCapabilityMixin``."""
        raise NotImplementedError(
            f"{type(self).__name__}: include ImageCapabilityMixin or override _build_image_edit_raw"
        )

    def _build_transcription_raw(
        self, request: Any, stream: bool = False
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build raw transcription data + files. Provided by ``AudioCapabilityMixin``."""
        raise NotImplementedError(
            f"{type(self).__name__}: include AudioCapabilityMixin "
            "or override _build_transcription_raw"
        )

    def _build_translation_raw(self, request: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build raw translation data + files. Provided by ``AudioCapabilityMixin``."""
        raise NotImplementedError(
            f"{type(self).__name__}: include AudioCapabilityMixin "
            "or override _build_translation_raw"
        )

    # ------------------------------------------------------------------
    # Outbound body dispatch chokepoint
    # ------------------------------------------------------------------

    def _build_outbound_body(
        self,
        request: Any,
        *,
        request_type: str,
        exempt_keys: set[str] | None = None,
        stream: bool = False,
    ) -> OutboundBody:
        """Single chokepoint: build raw body -> merge extra -> apply field policy.

        Every endpoint should route through this method, calling the appropriate
        raw builder hook then running _finalize_body to merge extra keys and
        enforce the configured field policy.

        Chat uses merge_extra=False because chat serializers already merge
        request.extra.  All other branches merge_extra=True.
        """

        def _finalize(body: dict[str, Any], merge_extra: bool = True) -> dict[str, Any]:
            return self._finalize_body(
                body, request, exempt_keys=exempt_keys, merge_extra=merge_extra
            )

        rt = RequestType(request_type)

        if rt == RequestType.CHAT:
            ctx = self._build_chat_context(request)
            # The conversion seam decides the tier. Native passthrough
            # forwards the stashed raw body verbatim: it already carries
            # parameter overrides (ParameterOverrideStage re-parses and
            # stores it) and must not be filtered by the field policy (its
            # fields are all protocol-valid). Wire reuse prepares a detached
            # copy of the stash in the seam, then gets the same field-policy
            # treatment as a rebuilt body. Anything else is a full
            # conversion via the provider serializer.
            plan = plan_conversion(self, request, context=ctx)
            if plan.request_tier == ConversionTier.NATIVE_PASSTHROUGH:
                return OutboundBody(json_body=prepare_native_body(self, request))
            if plan.request_tier == ConversionTier.WIRE_REUSE:
                return OutboundBody(
                    json_body=_finalize(prepare_wire_reuse_body(request, ctx), merge_extra=False)
                )
            return OutboundBody(
                json_body=_finalize(self._build_chat_raw(request, ctx), merge_extra=False)
            )

        if rt == RequestType.EMBEDDING:
            # Adapters may declare native embedding parameters (e.g. Ollama's
            # keep_alive/truncate/options) that must survive the
            # unknown-fields policy after _merge_extra injects them into the
            # body. Declared at the chokepoint so every call path honors it.
            if self._EMBEDDING_EXEMPT_EXTRA_KEYS:
                exempt_keys = (exempt_keys or set()) | set(self._EMBEDDING_EXEMPT_EXTRA_KEYS)
            return OutboundBody(json_body=_finalize(self._build_embedding_raw(request)))
        if rt == RequestType.SPEECH:
            return OutboundBody(json_body=_finalize(self._build_speech_raw(request)))
        if rt == RequestType.IMAGE_GENERATION:
            return OutboundBody(json_body=_finalize(self._build_image_raw(request)))
        if rt == RequestType.IMAGE_EDIT:
            body, files = self._build_image_edit_raw(request)
            finalized = _finalize(body)
            if files:
                return OutboundBody(form_data=finalized, files=files)
            return OutboundBody(json_body=finalized)
        if rt == RequestType.TRANSCRIPTION:
            data, files = self._build_transcription_raw(request, stream=stream)
            return OutboundBody(form_data=_finalize(data), files=files)
        if rt == RequestType.TRANSLATION:
            data, files = self._build_translation_raw(request)
            return OutboundBody(form_data=_finalize(data), files=files)

        raise ValueError(f"Unsupported request_type for outbound body: {request_type}")

    # ------------------------------------------------------------------
    # Native passthrough response helpers
    # ------------------------------------------------------------------

    def _build_passthrough_response(
        self,
        body: Any,
        request: InternalRequest,
        *,
        tier: ConversionTier = ConversionTier.NATIVE_PASSTHROUGH,
    ) -> InternalResponse:
        """Build a minimal InternalResponse carrying the raw upstream body.

        The body is stashed in ``provider_info["_raw_response_body"]`` so the
        protocol formatter emits it verbatim instead of re-serializing parsed
        blocks. Only usage is extracted (for billing); content blocks are not
        parsed at all.

        The upstream body's ``model`` field is rewritten to the
        client-requested name (``user_facing_model``) so the verbatim
        passthrough response echoes the same alias as the parsed and
        streaming paths — set even when the upstream omits the field,
        mirroring the streaming transformer's unconditional aliasing. The
        resolved provider model name remains visible in
        ``event_context.internal_model`` / ``provider_model_name``.
        """
        usage = None
        extras: dict[str, Any] = {}
        response_id = ""
        request.response_tier = tier
        if isinstance(body, dict):
            response_id = body.get("id", "")
            usage, extras = self._parse_passthrough_usage(body, request)
            if request.user_facing_model:
                body["model"] = request.user_facing_model
        return InternalResponse(
            id=response_id,
            model=request.model,
            output=[],
            usage=usage,
            provider_info={"provider": self.provider_name, "_raw_response_body": body, **extras},
        )

    def _parse_passthrough_usage(
        self, body: dict[str, Any], request: InternalRequest
    ) -> tuple[Usage | None, dict[str, Any]]:
        """Extract usage and billing extras from a raw passthrough body.

        Default: OpenAI-style usage parsing. Anthropic-family providers
        override to add cache-token folding and billing extras.
        """
        return parse_usage_from_response(body), {}

    # ------------------------------------------------------------------
    # Policy resolution helpers
    # ------------------------------------------------------------------

    def _resolve_field_policy(self) -> str:
        return self._extra_config.get("unknown_fields_policy", "ignore")

    def _resolve_block_policy(self) -> str:
        return self._extra_config.get("unsupported_block_policy", "drop")

    def _target_endpoint(self) -> str:
        """Return the target upstream endpoint type for chat requests.

        Adapters that talk to the OpenAI Responses API should override this
        to 'responses'. The default 'chat_completions' is used by all Chat
        Completions API providers (openai-compatible, openrouter, deepseek, etc.).
        """
        return "chat_completions"

    # ------------------------------------------------------------------
    # Body merge & policy helpers
    # ------------------------------------------------------------------

    def _merge_extra(self, body: dict[str, Any], extra: dict[str, Any] | None) -> dict[str, Any]:
        if not extra:
            return body
        for k, v in extra.items():
            if v is not None:
                body[k] = v
        return body

    def _apply_field_policy(
        self,
        body: dict[str, Any],
        extra: dict[str, Any] | None,
        exempt_keys: set[str] | None = None,
    ) -> dict[str, Any]:
        if not extra:
            return body
        policy = self._resolve_field_policy()
        relevant = {k: v for k, v in extra.items() if k not in (exempt_keys or set())}
        if not relevant:
            return body
        match policy:
            case "passthrough":
                return body
            case "ignore":
                return {k: v for k, v in body.items() if k not in relevant}
            case "error":
                raise ProviderError(
                    message=(
                        f"Provider '{self.provider_name}' received unknown request "
                        f"fields that are not supported: "
                        f"{', '.join(sorted(relevant.keys()))}. "
                        "Remove these fields or set unknown_fields_policy to "
                        "'ignore' or 'passthrough'."
                    ),
                    error_type="invalid_request_error",
                    provider_name=self.provider_name,
                )
            case _:
                logger.warning(
                    f"Unrecognized unknown_fields_policy '{policy}' for provider "
                    f"'{self.provider_name}'; defaulting to passthrough. Valid values: "
                    f"ignore, passthrough, error."
                )
                return body

    def _finalize_body(
        self,
        body: dict[str, Any],
        request: Any,
        *,
        exempt_keys: set[str] | None = None,
        merge_extra: bool = True,
    ) -> dict[str, Any]:
        extra = getattr(request, "extra", None)
        if merge_extra:
            body = self._merge_extra(body, extra)
        # Automatically exempt keys injected by parameter overrides so they
        # are not silently stripped (or rejected) by unknown_fields_policy.
        # getattr is deliberate: request is Any-typed here (chat/audio/image
        # models), and non-chat models do not declare this field.
        override_keys = getattr(request, "_override_injected_keys", None)
        if override_keys:
            exempt_keys = (exempt_keys or set()) | override_keys
        return self._apply_field_policy(body, extra, exempt_keys=exempt_keys)

    def _parse_error_response(self, status_code: int, error_body: dict[str, Any]) -> ProviderError:
        """Parse provider error response body. Override for provider-specific formats."""
        return self._error_translator.parse_error_response(status_code, error_body)

    async def _check_response_status(self, response) -> None:
        """Check response status and raise ProviderError for error responses."""
        await self._transport.check_response_status(
            response, self.provider_name, self._parse_error_response
        )

    @asynccontextmanager
    async def _streaming_post(
        self,
        client: AsyncSession,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        files: Any = None,
        timeout: tuple[float, float],
    ) -> AsyncIterator[Any]:
        async with self._transport.streaming_post(
            client,
            url,
            headers=headers,
            json=json,
            data=data,
            files=files,
            timeout=timeout,
        ) as response:
            yield response

    def _get_stream_timeout(self) -> tuple[float, float]:
        return (self._connect_timeout, STREAM_READ_TIMEOUT)

    _models_endpoint: str = "/models"
    _models_data_key: str = "data"

    def _models_url(self) -> str:
        return f"{self._base_url}{self._models_endpoint}"

    def _models_headers(self) -> dict[str, str]:
        return self._build_headers()

    def _parse_model(self, raw: dict[str, Any]) -> ProviderModelInfo:
        from llm_proxy.models.provider import ProviderModelInfo

        return ProviderModelInfo(
            id=raw.get("id", ""),
            name=raw.get("id", ""),
            description=raw.get("description"),
            owned_by=raw.get("owned_by"),
        )

    async def list_models(self, client: AsyncSession | None = None) -> list[ProviderModelInfo]:
        from llm_proxy.http.client import fetch_json

        http_client = client or await self._get_client()
        url = self._models_url()
        headers = self._models_headers()

        data = await fetch_json(http_client, url, headers=headers)
        models = data.get(self._models_data_key, [])

        return sorted(
            [self._parse_model(m) for m in models],
            key=lambda m: m.id,
        )

    CHAT_ENDPOINT: str = ""

    def _resolve_endpoint_url(
        self, endpoint_type: str, default_path: str, model: str | None = None
    ) -> str:
        if self._endpoint_base_urls and endpoint_type in self._endpoint_base_urls:
            url = self._endpoint_base_urls[endpoint_type].rstrip("/")
        else:
            url = f"{self._base_url}{default_path}"
        if model and "{model}" in url:
            url = url.replace("{model}", model)
        return url


__all__ = [
    "BaseHttpProvider",
    "STREAM_READ_TIMEOUT",
]
