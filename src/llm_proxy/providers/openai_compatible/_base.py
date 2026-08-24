"""Chat Completions base adapter — shared base for DeepSeek and openai-compatible."""

from collections.abc import AsyncIterator
from typing import Any

from llm_proxy.core.adapter import AdapterConfig, register_adapter
from llm_proxy.core.conversion import plan_conversion
from llm_proxy.core.reasoning_cache import (
    try_cache_reasoning_from_chat_completion_body,
    try_cache_reasoning_from_response,
)
from llm_proxy.http.client import AsyncSession  # noqa: F401 — imported for test mocking
from llm_proxy.models import (
    ConversionTier,
    InternalImageEditRequest,
    InternalImageRequest,
    InternalImageResponse,
    InternalRequest,
    InternalResponse,
)
from llm_proxy.observability.logger import get_logger
from llm_proxy.providers.base import BaseHttpProvider, _extract_rate_limit_headers
from llm_proxy.providers.capabilities import (
    AudioCapabilityMixin,
    ChatCapabilityMixin,
    EmbeddingCapabilityMixin,
    ImageCapabilityMixin,
)
from llm_proxy.providers.reasoning import (
    REASONING_ECHO_MODEL_MARKERS,
    detect_reasoning_field_in_response_body,
    detect_reasoning_field_in_stream_chunk,
    ensure_reasoning_echo,
    normalize_reasoning_in_response_body,
    normalize_reasoning_in_stream_chunk,
)
from llm_proxy.serialization.context import BuildContext
from llm_proxy.serialization.openai.components.request_builder import OpenAIRequestBuilder
from llm_proxy.serialization.openai.components.response_parser import OpenAIResponseParser
from llm_proxy.serialization.providers import get_provider_serializer

logger = get_logger(__name__)


@register_adapter("openai-compatible")
class OpenAICompatibleBase(
    ChatCapabilityMixin,
    EmbeddingCapabilityMixin,
    ImageCapabilityMixin,
    AudioCapabilityMixin,
    BaseHttpProvider,
):
    """Base adapter for Chat Completions API format providers.

    Registered as the generic ``openai-compatible`` provider type.
    Also used as base for DeepSeekAdapter, OpenRouterAdapter, etc.

    Inherits the OpenAI-shaped wire format for every capability (chat,
    embeddings, images, audio) — the endpoint constants below select the
    default paths; adapters override the ones their upstream does not
    actually serve.
    """

    _DEFAULT_PROVIDER_NAME = "openai-compatible"

    #: Branding for the admin provider catalog (GET /api/config/provider-types).
    DISPLAY_NAME_EN = "OpenAI Compatible"
    DISPLAY_NAME_ZH = "OpenAI 兼容"
    LOBE_ICON_ID = "openai"
    LOBE_ICON_VARIANT = "mono"

    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    _REASONING_FIELD: str | None = None
    # Model-name markers that require DeepSeek-style reasoning echo on
    # tool-call turns (HTTP 400 otherwise). Adapters may override.
    REASONING_ECHO_MODEL_MARKERS: tuple[str, ...] = REASONING_ECHO_MODEL_MARKERS
    CHAT_ENDPOINT = "/chat/completions"
    EMBEDDINGS_ENDPOINT = "/embeddings"
    IMAGES_ENDPOINT = "/images/generations"
    IMAGES_EDITS_ENDPOINT = "/images/edits"
    SPEECH_ENDPOINT = "/audio/speech"
    TRANSCRIPTION_ENDPOINT = "/audio/transcriptions"
    TRANSLATION_ENDPOINT = "/audio/translations"

    def _get_serializer(self):
        return get_provider_serializer("openrouter")

    def __init__(
        self,
        *,
        config: AdapterConfig | None = None,
        **kwargs: Any,
    ):
        if config is not None:
            super().__init__(config=config)
        else:
            kwargs.setdefault("provider_name", self._DEFAULT_PROVIDER_NAME)
            kwargs.setdefault("base_url", self.DEFAULT_BASE_URL)
            super().__init__(**kwargs)

    def _build_chat_raw(self, request: InternalRequest, context: BuildContext) -> dict[str, Any]:
        return self._get_serializer().build_provider_request(request, context)

    def _build_request_body(self, request: InternalRequest) -> dict[str, Any]:
        outbound = self._build_outbound_body(request, request_type="chat")
        if outbound.json_body is None:
            raise ValueError("Expected JSON body for chat request")
        body = outbound.json_body
        builder = self._get_request_builder()
        preferred = self._reasoning_field_preference()
        model = body.get("model")
        body = builder.normalize_reasoning_for_request(body, self._base_url, preferred, model=model)
        return self._enforce_reasoning_echo(body, request, preferred, model=model)

    def _enforce_reasoning_echo(
        self,
        body: dict[str, Any],
        request: InternalRequest,
        preferred: str | None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Apply the DeepSeek-style reasoning echo guarantee for this request.

        Runs after ``normalize_reasoning_for_request`` so the placeholder is
        injected into the field the upstream model expects (``reasoning``
        for OpenRouter/NanoGPT, ``reasoning_content`` otherwise). Skipped for
        models that do not require the echo and when thinking mode is
        explicitly disabled.
        """
        if not self._requires_reasoning_echo(request):
            return body
        builder = self._get_request_builder()
        preferred_field = preferred or builder.get_reasoning_field_preference(
            self._base_url, model=model
        )
        field = "reasoning" if preferred_field == "reasoning" else "reasoning_content"
        return ensure_reasoning_echo(body, field, request)

    def _requires_reasoning_echo(self, request: InternalRequest) -> bool:
        """Whether this request's model needs the reasoning echo guarantee.

        Default: any model whose name contains a known marker (``deepseek``,
        ``kimi``). Adapters for platforms that always enforce the rule (e.g.
        the dedicated DeepSeek adapter) can override to return True.
        """
        model = (request.model or "").lower()
        return any(marker in model for marker in self.REASONING_ECHO_MODEL_MARKERS)

    def _reasoning_field_preference(self) -> str | None:
        """Return the provider's preferred assistant reasoning field name.

        ``None`` means "use detected/cached preference, defaulting to
        ``reasoning_content``". Adapters for providers that always expect the
        ``reasoning`` field (e.g. OpenRouter, NanoGPT) override this.
        """
        return self._REASONING_FIELD

    def _get_request_builder(self) -> OpenAIRequestBuilder:
        """Return the OpenAIRequestBuilder used by this adapter's serializer.

        Adapters that use a custom serializer (e.g. NanoGPT) can override this.
        """
        serializer = self._get_serializer()
        # Most OpenAI-compatible serializers compose OpenAIRequestBuilder.
        if hasattr(serializer, "_request_builder"):
            return serializer._request_builder
        logger.warning(
            "Serializer %s has no _request_builder; falling back to default "
            "OpenAIRequestBuilder. Reasoning normalization may be inconsistent.",
            serializer_name=type(serializer).__name__,
        )
        return OpenAIRequestBuilder()

    def _stream_body(self, request: InternalRequest) -> dict[str, Any]:
        return self._build_request_body(request)

    def _record_reasoning_field_preference(
        self,
        detected: str | None,
        *,
        model: str | None = None,
        response_model: Any = None,
    ) -> None:
        """Teach the request side which reasoning field the upstream model uses.

        Runs on the response side — streaming chunks and the wire-reuse
        body — *before* the unconditional ``reasoning`` ->
        ``reasoning_content`` rename hides the provider's original field
        name. Recording it lets ``normalize_reasoning_for_request`` convert
        the client's ``reasoning_content`` echo back to the model's field on
        subsequent turns. This is the streaming/wire-reuse counterpart of the
        non-stream parsed-path detection in ``OpenAIResponseParser``.

        The preference is cached per model: ``model`` is the routed model id
        (the key future requests look up), and ``response_model`` covers
        model aliasing when the upstream reports a different id.
        """
        if detected is None:
            return
        builder = self._get_request_builder()
        builder.record_reasoning_field_preference(
            self._base_url,
            detected,
            model=model,
            response_model=response_model,
        )

    async def chat_completion(self, request: InternalRequest, **kwargs: Any) -> InternalResponse:
        url = self._resolve_endpoint_url("chat_completion", self.CHAT_ENDPOINT, model=request.model)
        headers = self._build_headers()
        body = self._build_request_body(request)

        response = await self._post_json_response_with_retry(url, headers, body)
        response_data = response.json()
        plan = plan_conversion(self, request, context=self._build_chat_context(request))
        if plan.response_mode == ConversionTier.WIRE_REUSE:
            # Response wire-reuse: the provider answered in the client's own
            # protocol, so the raw body rides verbatim (the strategy layer
            # emits provider_info["_raw_response_body"] as-is). Load-bearing
            # transforms still run, mirroring the streaming path: the
            # reasoning-field rename (stream renames reasoning ->
            # reasoning_content in chunks), model aliasing and usage
            # extraction (inside _build_passthrough_response).
            self._record_reasoning_field_preference(
                detect_reasoning_field_in_response_body(response_data),
                model=request.model,
                response_model=response_data.get("model"),
            )
            normalize_reasoning_in_response_body(response_data)
            result = self._build_passthrough_response(
                response_data, request, tier=ConversionTier.WIRE_REUSE
            )
            # Verbatim bodies never become parsed blocks, so write the
            # reasoning cache straight from the wire shape — the non-stream
            # counterpart of the streaming path's cache write. Never fatal.
            try_cache_reasoning_from_chat_completion_body(response_data)
        else:
            result = self._parse_response(
                self._get_serializer(),
                response_data,
                model=request.model,
                request_id=request.request_id,
                request=request,
                base_url=self._base_url,
            )
            try_cache_reasoning_from_response(result)
        # Runs on both tiers: provider metadata extraction (e.g. OpenRouter
        # cost fields for billing) only needs the raw body + provider_info.
        result = self._post_process_chat_response(response_data, result)
        result.provider_info["_rate_limit_headers"] = _extract_rate_limit_headers(
            getattr(response, "headers", None)
        )
        return result

    def _parse_passthrough_usage(
        self, body: dict[str, Any], request: InternalRequest
    ) -> tuple[Any, dict[str, Any]]:
        """Chat-shaped usage extraction for the wire-reuse response tier.

        Reuses the chat response parser's usage routine so billing is
        identical to the parsed tier by construction (DeepSeek cache-hit
        folding and server_tool_use web-search counts included). The base
        default (``parse_usage_from_response``) is Responses-API-shaped and
        would read zeros from a Chat Completions usage object.
        """
        return OpenAIResponseParser.parse_usage(body), {}

    def _post_process_chat_response(
        self, response: dict[str, Any], result: InternalResponse
    ) -> InternalResponse:
        return result

    def _stream_transform_chunk(
        self, chunk: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any] | None:
        self._record_reasoning_field_preference(
            detect_reasoning_field_in_stream_chunk(chunk),
            model=context.get("model"),
            response_model=chunk.get("model"),
        )
        return normalize_reasoning_in_stream_chunk(chunk)

    async def stream_chat_completion(
        self,
        request: InternalRequest,
        cancel_token: Any | None = None,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str | dict[str, Any]]:
        if context is None:
            context = {"model": request.model}
        else:
            context["model"] = request.model
        return await ChatCapabilityMixin.stream_chat_completion(
            self, request, cancel_token=cancel_token, context=context, **kwargs
        )

    async def image_generation(
        self, request: InternalImageRequest, **kwargs: Any
    ) -> InternalImageResponse:
        url = self._image_generation_url(model=request.model)
        headers = self._build_headers()
        outbound = self._build_outbound_body(request, request_type="image_generation")
        if outbound.json_body is None:
            raise ValueError("Expected JSON body for chat request")
        response = await self._post_json_with_retry(url, headers, outbound.json_body)
        return self.from_image_provider_format(response)

    async def image_edit(
        self, request: InternalImageEditRequest, **kwargs: Any
    ) -> InternalImageResponse:
        url = self._image_edit_url(model=request.model)
        headers = self._build_headers()
        outbound = self._build_outbound_body(request, request_type="image_edit")
        if outbound.form_data is not None:
            headers.pop("Content-Type", None)

            async def _make_form_request():
                client = await self._get_client()
                response = await client.post(
                    url,
                    headers=headers,
                    data=outbound.form_data,
                    files=outbound.files,
                )
                await self._check_response_status(response)
                return response.json()

            response = await self._with_retry(_make_form_request)
        else:
            if outbound.json_body is None:
                raise ValueError("Expected JSON or multipart body for image edit request")
            response = await self._post_json_with_retry(url, headers, outbound.json_body)
        return self.from_image_edit_provider_format(response)


__all__ = ["OpenAICompatibleBase"]
