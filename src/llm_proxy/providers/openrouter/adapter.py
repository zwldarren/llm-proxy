"""OpenRouter provider adapter.

This provider uses direct HTTP calls to OpenRouter's OpenAI-compatible API.
OpenRouter provides a unified interface to access 100+ LLM models from various providers.

OpenRouter specific features:
- Supports model routing with "provider/model" format (e.g., "anthropic/claude-3-opus")
- Supports custom headers for analytics (HTTP-Referer, X-Title)
- Returns `reasoning` field instead of `reasoning_content` for thinking models
- Expects `reasoning` field in assistant messages (not `reasoning_content`)
- Sends SSE comments (e.g., ": OPENROUTER PROCESSING") during streaming
"""

import base64
from typing import Any

from llm_proxy.core.adapter import register_adapter
from llm_proxy.core.exceptions import ValidationError
from llm_proxy.core.request_type import RequestType
from llm_proxy.models import InternalRequest, InternalResponse
from llm_proxy.observability.logger import get_logger
from llm_proxy.providers.anthropic.client_headers import (
    get_client_headers as get_anthropic_client_headers,
)
from llm_proxy.providers.base import extract_rate_limit_headers
from llm_proxy.providers.headers import merge_passthrough_headers
from llm_proxy.providers.openai.client_headers import get_openrouter_client_headers
from llm_proxy.providers.openai_compatible._native import NativePassthroughChatBase
from llm_proxy.streaming.sse_parse import parse_sse_data_line

logger = get_logger(__name__)

#: Default app-attribution values, applied when the operator has not configured
#: their own: requests made through the proxy are credited to the LLM Proxy
#: project on OpenRouter's rankings and analytics. Overridden per provider by
#: the ``app_attribution`` config field or by ``HTTP-Referer`` /
#: ``X-OpenRouter-Title`` entries in ``custom_headers``.
DEFAULT_ATTRIBUTION_URL = "https://github.com/zwldarren/llm-proxy"
DEFAULT_ATTRIBUTION_TITLE = "LLM Proxy"


@register_adapter("openrouter")
class OpenRouterAdapter(NativePassthroughChatBase):
    """OpenRouter provider using direct HTTP calls to OpenAI-compatible API.

    OpenRouter provides unified access to multiple LLM providers through a
    single API. Extends ``NativePassthroughChatBase`` with OpenRouter-specific
    features: a native Responses endpoint, the unified ``reasoning`` request
    object, per-request control-header forwarding, and app attribution.
    """

    _DEFAULT_PROVIDER_NAME = "openrouter"

    #: Branding for the admin provider catalog (GET /api/config/provider-types).
    DISPLAY_NAME_EN = "OpenRouter"
    DISPLAY_NAME_ZH = "OpenRouter"
    LOBE_ICON_ID = "openrouter"
    LOBE_ICON_VARIANT = "mono"

    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
    _REASONING_FIELD = "reasoning"
    # OpenRouter's dedicated Image API is at /images (not /images/generations).
    IMAGES_ENDPOINT = "/images"

    # OpenRouter exposes native OpenAI Responses and Anthropic Messages
    # endpoints alongside Chat Completions, so ``/v1/responses`` and
    # ``/v1/messages`` clients are forwarded verbatim: reasoning mode/context,
    # provider routing, plugins, transforms and usage accounting all survive
    # instead of being dropped by the Chat Completions translation (ADR-0002).
    # Anthropic clients otherwise pay a double translation — Messages ->
    # Chat Completions -> Messages — on every request.
    native_protocols = frozenset({"anthropic", "openresponses"})
    RESPONSES_PATH = "/responses"
    ANTHROPIC_MESSAGES_PATH = "/messages"

    # The Chat Completions wire accepts the unified ``reasoning`` object
    # (``effort`` plus the Responses-only ``mode``/``context``/``summary``),
    # so the request builder emits that instead of a bare ``reasoning_effort``.
    REASONING_OBJECT = True

    # A router fans out to many upstreams: the client's Codex/Claude Code
    # fingerprint headers describe its original target, not the routed
    # upstream, so only OpenRouter's own control headers are forwarded.
    FORWARD_CLIENT_HEADERS = False

    #: App-attribution title headers, in precedence order. ``X-OpenRouter-Title``
    #: is the current name; ``X-Title`` is kept for backwards compatibility.
    _ATTRIBUTION_TITLE_HEADERS = ("X-OpenRouter-Title", "X-Title")

    #: Attribution headers a client must never control (``X-Title`` mirrors
    #: ``_ATTRIBUTION_TITLE_HEADERS``; the rest complete the app identity).
    #: They come from the operator's ``app_attribution``/``custom_headers``
    #: only, so a client cannot spoof the app credited on OpenRouter.
    _ATTRIBUTION_CONTROL_HEADERS = frozenset(
        {
            "x-openrouter-title",
            "x-title",
            "x-openrouter-categories",
            "x-openrouter-app-visibility",
        }
    )

    #: Protocol-level Anthropic headers forwarded from an Anthropic-protocol
    #: client. ``anthropic-version`` pins the API revision and
    #: ``anthropic-beta`` opts into beta features; both describe the request,
    #: unlike the app fingerprints (``x-app``, ``user-agent``) captured
    #: alongside them, which describe the client's original target and stay
    #: behind for a router.
    _ANTHROPIC_PROTOCOL_HEADERS = frozenset({"anthropic-version", "anthropic-beta"})

    #: OpenRouter request fields documented as supported on each endpoint but
    #: absent from the OpenAI schema, so the default
    #: ``unknown_fields_policy="ignore"`` would strip them as if the upstream
    #: did not understand them. Declaring them here keeps a client's routing,
    #: fallback, plugin, caching and accounting options intact without forcing
    #: operators to switch the whole provider to ``passthrough``.
    EXEMPT_EXTRA_KEYS = {
        RequestType.CHAT: frozenset(
            {
                # Provider routing and model fallbacks. ``route`` is
                # deprecated upstream in favour of providers.sort.partition.
                "provider",
                "models",
                "route",
                # Request/response extensions. ``transforms`` (middle-out),
                # ``usage`` (include) and ``include_reasoning`` are legacy:
                # current docs replace them with the context-compression
                # plugin, stream_options.include_usage and reasoning.exclude.
                # Kept so clients that still send them are not silently
                # stripped.
                "plugins",
                "transforms",
                "usage",
                "session_id",
                "trace",
                "preset",
                "include_reasoning",
                "structured_outputs",
                # Prompt caching: a top-level breakpoint covers the last
                # cacheable block. Per-block markers need no exemption — they
                # ride inside message content, which the wire-reuse tier
                # forwards verbatim; only a rebuilt body drops them.
                "cache_control",
                # Sampling parameters OpenRouter documents but OpenAI's schema
                # does not have.
                "min_p",
                "top_k",
                "top_a",
                "repetition_penalty",
            }
        ),
        # /api/v1/embeddings documents input_type, session_id and trace
        # alongside the shared provider routing block.
        RequestType.EMBEDDING: frozenset({"input_type", "session_id", "trace", "provider", "user"}),
        # /api/v1/images documents the normalized resolution/aspect_ratio
        # knobs, seed, reference images and the routing block.
        RequestType.IMAGE_GENERATION: frozenset(
            {"resolution", "aspect_ratio", "seed", "input_references", "provider"}
        ),
        # /api/v1/audio/speech takes voice-cloning references and provider
        # options.
        RequestType.SPEECH: frozenset({"input_references", "provider"}),
        # /api/v1/audio/transcriptions takes provider options; routing
        # preferences (order/only/ignore) are explicitly not applied upstream.
        RequestType.TRANSCRIPTION: frozenset({"provider"}),
    }

    def _models_params(self) -> dict[str, Any]:
        """List every modality, not just text.

        ``/api/v1/models`` defaults to ``output_modalities=text``, which hides
        the image, video, audio and embedding models. This adapter also serves
        OpenRouter's image, speech and transcription endpoints, so the operator
        picking a model here has to be able to see them.
        """
        return {"output_modalities": "all"}

    def _build_headers(
        self,
        auth_header: str | None = None,
        auth_prefix: str | None = None,
    ) -> dict[str, str]:
        """Build upstream headers: provider headers + attribution + client opt-ins.

        Order matters. ``super()`` applies ``EXTRA_HEADERS`` and the operator's
        ``custom_headers`` first; the client's forwarded OpenRouter control
        headers (router metadata, response caching, ``x-session-id``,
        ``x-anthropic-beta``) come next; the attribution defaults fill in last.
        Both merges are non-destructive, so an operator
        who sets ``HTTP-Referer``/``X-OpenRouter-Title`` in ``custom_headers``
        (or through the typed attribution fields) always wins over the default
        project link. Attribution headers are filtered out of the client set
        first, so a client cannot spoof the app identity.
        """
        headers = super()._build_headers(auth_header, auth_prefix)
        client_headers = {
            key: value
            for key, value in get_openrouter_client_headers().items()
            if key.lower() not in self._ATTRIBUTION_CONTROL_HEADERS
        }
        client_headers.update(
            {
                key: value
                for key, value in get_anthropic_client_headers().items()
                if key.lower() in self._ANTHROPIC_PROTOCOL_HEADERS
            }
        )
        merge_passthrough_headers(headers, client_headers)
        self._apply_attribution_headers(headers)
        return headers

    def _apply_attribution_headers(self, headers: dict[str, str]) -> None:
        """Fill in app-attribution headers for any the operator has not set."""
        attribution = self._extra_config.get("app_attribution") or {}
        if not isinstance(attribution, dict):
            attribution = {}

        existing = {key.lower() for key in headers}

        referer = attribution.get("url") or DEFAULT_ATTRIBUTION_URL
        if referer and "http-referer" not in existing:
            headers["HTTP-Referer"] = referer

        title = attribution.get("title") or DEFAULT_ATTRIBUTION_TITLE
        if title and not any(h.lower() in existing for h in self._ATTRIBUTION_TITLE_HEADERS):
            headers["X-OpenRouter-Title"] = title

        categories = attribution.get("categories")
        if categories and "x-openrouter-categories" not in existing:
            if isinstance(categories, list):
                categories = ",".join(str(c) for c in categories)
            headers["X-OpenRouter-Categories"] = str(categories)

        visibility = attribution.get("visibility")
        if visibility == "hidden" and "x-openrouter-app-visibility" not in existing:
            # Only meaningful alongside HTTP-Referer — and that header is always
            # present by the time we get here (default or operator-supplied).
            headers["X-OpenRouter-App-Visibility"] = "hidden"

    async def _native_completion(self, request: InternalRequest) -> InternalResponse:
        """Native Responses completion with OpenRouter cost extraction.

        The base implementation carries the upstream body verbatim and parses
        only usage; OpenRouter also reports the actual billed cost in
        ``usage.cost``, which the billing pipeline reads from
        ``provider_info["openrouter_cost"]``. Reusing the chat path's
        post-processing keeps the two tiers' metadata identical.
        """
        result = await super()._native_completion(request)
        body = result.provider_info.get("_raw_response_body")
        if isinstance(body, dict):
            result = self._post_process_chat_response(body, result)
        return result

    def _stream_filter_line(self, line_str: str) -> str | None:
        if line_str.startswith(":"):
            return None
        return parse_sse_data_line(line_str)

    def _post_process_chat_response(
        self, response: dict[str, Any], result: InternalResponse
    ) -> InternalResponse:
        """Extract provider metadata from response."""
        if "choices" in response:
            # The chat serializer's unknown-field dump only makes sense for a
            # Chat Completions body. A Responses or Messages body (both take the
            # native tier) has no ``choices`` and would otherwise spill its whole
            # envelope — ``output``, ``content``, ``stop_reason``, ... — into
            # provider_info.
            unknown_fields = self._get_serializer().extract_unknown_response_fields(response)
            if unknown_fields:
                result.provider_info.update(unknown_fields)

        # OpenRouter reports cost in usage.cost (non-streaming), on every
        # endpoint including the native Messages one.
        usage = response.get("usage", {})
        if isinstance(usage, dict):
            cost = _reported_cost(usage)
            if cost is not None:
                result.provider_info["openrouter_cost"] = cost

            cost_details = usage.get("cost_details")
            if isinstance(cost_details, dict):
                result.provider_info["openrouter_cost_details"] = cost_details

            is_byok = usage.get("is_byok")
            if isinstance(is_byok, bool):
                result.provider_info["openrouter_is_byok"] = is_byok

        return result

    def _build_image_raw(self, request: Any) -> dict[str, Any]:
        """Build the Image API body: ``output_format``, not ``response_format``.

        ``/api/v1/images`` documents ``output_format`` (png/jpeg/webp/svg) and
        always answers with base64 data, so the OpenAI-shaped
        ``response_format`` (url|b64_json) and ``style`` are not part of its
        schema and are dropped instead of being forwarded as unknown fields.
        The same schema documents ``background`` and ``output_compression`` for
        every image model, which the shared builder only emits for models whose
        slug starts with ``gpt-image-`` — never the case here, since OpenRouter
        slugs are namespaced (``openai/gpt-image-1``).
        """
        body = super()._build_image_raw(request)
        body.pop("response_format", None)
        body.pop("style", None)
        for key in ("output_format", "background", "output_compression"):
            value = getattr(request, key, None)
            if value is not None:
                body[key] = value
        return body

    def _build_speech_raw(self, request: Any) -> dict[str, Any]:
        """Build the TTS body after checking the format OpenRouter accepts."""
        _require_response_format(request.response_format, SPEECH_RESPONSE_FORMATS, "audio/speech")
        return super()._build_speech_raw(request)

    async def image_generation(self, request: Any, **kwargs: Any) -> Any:
        """Generate images via OpenRouter's dedicated Image API.

        OpenRouter's Image API is at ``/api/v1/images`` (not
        ``/api/v1/images/generations``) and reports the actual cost in
        ``usage.cost``.  This override:

        1. Uses the correct endpoint (``/images``).
        2. Extracts ``usage.cost`` from the raw response and stores it in
           ``provider_info["openrouter_cost"]`` so the billing pipeline
           can use the provider-reported cost.
        """
        url = self._image_generation_url(model=request.model)
        headers = self._build_headers()
        outbound = self._build_outbound_body(request, request_type="image_generation")
        if outbound.json_body is None:
            raise ValueError("Expected JSON body for image generation request")

        response = await self._post_json_with_retry(url, headers, outbound.json_body)

        # Extract OpenRouter's reported cost before parsing.
        openrouter_cost = _reported_cost(response.get("usage"))

        result = self.from_image_provider_format(response)
        if openrouter_cost is not None:
            result.provider_info["openrouter_cost"] = openrouter_cost

        return result

    # ------------------------------------------------------------------
    # Speech-to-Text — OpenRouter uses JSON with input_audio instead of
    # multipart form data (OpenAI-compatible format difference).
    # ------------------------------------------------------------------

    async def transcription(self, request: Any, **kwargs: Any) -> Any:
        """Transcribe audio using OpenRouter's JSON-based STT API.

        OpenRouter expects JSON with ``input_audio`` (base64 + format)
        rather than OpenAI's multipart form data with a file upload.
        """
        if not request.file:
            raise ValueError("Audio file data is required for transcription")

        url, headers, body = self._build_transcription_request(request)
        body = self._finalize_body(
            body,
            request,
            exempt_keys=self._exempt_keys_for(RequestType.TRANSCRIPTION),
            merge_extra=True,
        )

        async def _make_request():
            client = await self._get_client()
            response = await client.post(url, headers=headers, json=body)
            await self._check_response_status(response)
            return response

        response = await self._with_retry(_make_request)
        content_type = response.headers.get("content-type", "")

        if "application/json" in content_type:
            response_data = response.json()
        else:
            response_data = {"text": response.text}

        result = self._parse_transcription_response(response_data, request.response_format)
        # STT is billed per second upstream; ``usage.cost`` is the authoritative
        # figure and takes precedence over the local audio-duration estimate.
        openrouter_cost = _reported_cost(response_data.get("usage"))
        if openrouter_cost is not None:
            result.provider_info["openrouter_cost"] = openrouter_cost
        # Parity with AudioCapabilityMixin.transcription, which this override
        # replaces: carries Retry-After / X-Generation-Id to the client.
        result.provider_info["_rate_limit_headers"] = extract_rate_limit_headers(
            getattr(response, "headers", None)
        )
        return result

    async def stream_transcription(self, request: Any, **kwargs: Any) -> Any:
        """Stream transcription using OpenRouter's JSON-based STT API."""
        if not request.file:
            raise ValueError("Audio file data is required for transcription")

        url, headers, body = self._build_transcription_request(request, stream=True)
        body = self._finalize_body(
            body,
            request,
            exempt_keys=self._exempt_keys_for(RequestType.TRANSCRIPTION),
            merge_extra=True,
        )

        stream_timeout = self._get_stream_timeout()

        async def _generator():
            client = await self._get_client()
            async with self._streaming_post(
                client,
                url,
                headers=headers,
                json=body,
                timeout=stream_timeout,
            ) as response:
                await self._check_response_status(response)
                async for line in response.iter_lines():
                    if not line:
                        continue
                    line_str = line.decode("utf-8") if isinstance(line, bytes) else line
                    yield f"{line_str}\n"

        return self._with_retry_generator(_generator)

    async def translation(self, request: Any, **_kwargs: Any) -> Any:
        """Reject audio translation: OpenRouter has no such endpoint.

        The inherited ``AudioCapabilityMixin.translation`` posts multipart to
        ``/audio/translations``; OpenRouter documents speech and transcriptions
        only, so that request can only come back as an opaque upstream 404.
        Failing here names the limitation instead.
        """
        raise ValidationError(
            message=(
                "OpenRouter does not provide an audio translation endpoint; "
                "use /v1/audio/transcriptions with a translation prompt"
            ),
            code="invalid_request_error",
            status_code=400,
        )

    def _build_transcription_request(
        self, request: Any, *, stream: bool = False
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        """Build URL, headers, and body for an OpenRouter transcription request.

        Shared between transcription() and stream_transcription() to avoid
        code duplication.
        """
        url = self._transcription_url(request)
        headers = self._build_headers()
        headers["Content-Type"] = "application/json"

        _require_response_format(
            request.response_format, TRANSCRIPTION_RESPONSE_FORMATS, "audio/transcriptions"
        )

        audio_format = _infer_audio_format(request.filename)
        base64_data = base64.b64encode(request.file).decode("utf-8")

        body: dict[str, Any] = {
            "model": request.model,
            "input_audio": {
                "data": base64_data,
                "format": audio_format,
            },
        }
        if stream:
            body["stream"] = True
        if request.language is not None:
            body["language"] = request.language
        if request.prompt is not None:
            body["prompt"] = request.prompt
        if request.temperature != 0.0:
            body["temperature"] = request.temperature
        if request.response_format != "json":
            body["response_format"] = request.response_format
        if request.timestamp_granularities:
            body["timestamp_granularities"] = list(request.timestamp_granularities)

        return url, headers, body


#: ``response_format`` values OpenRouter's audio endpoints accept. TTS takes
#: ``mp3`` or ``pcm`` (the upstream default is ``pcm``; the model layer's
#: ``mp3`` is in range). The TTS guide's prose mentions ``wav``, but the
#: OpenAPI schema enums only ``mp3``/``pcm``, so validation follows the
#: schema. STT takes ``json`` or ``verbose_json`` — ``text``, ``srt`` and
#: ``vtt`` are rejected upstream with a 400.
SPEECH_RESPONSE_FORMATS = frozenset({"mp3", "pcm"})
TRANSCRIPTION_RESPONSE_FORMATS = frozenset({"json", "verbose_json"})


def _require_response_format(value: str | None, supported: frozenset[str], endpoint: str) -> None:
    """Reject an audio ``response_format`` OpenRouter does not accept.

    Validating at the proxy boundary turns an opaque upstream 400 into a
    request error naming the endpoint and the accepted values.
    """
    if value is None or value in supported:
        return
    raise ValidationError(
        message=(
            f"OpenRouter /{endpoint} does not support response_format "
            f"'{value}'; supported: {', '.join(sorted(supported))}"
        ),
        code="invalid_request_error",
        status_code=400,
    )


def _reported_cost(usage: Any) -> float | None:
    """Return the billed cost reported in an OpenRouter ``usage`` object.

    OpenRouter prices every endpoint itself and reports the charge in
    ``usage.cost``; the billing pipeline reads it from
    ``provider_info["openrouter_cost"]`` in preference to a local estimate.
    """
    if not isinstance(usage, dict):
        return None
    cost = usage.get("cost")
    if isinstance(cost, int | float) and cost > 0:
        return float(cost)
    return None


def _infer_audio_format(filename: str) -> str:
    """Infer audio format from filename extension."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    format_map = {
        "wav": "wav",
        "mp3": "mp3",
        "flac": "flac",
        "m4a": "m4a",
        "ogg": "ogg",
        "webm": "webm",
        "aac": "aac",
        "aiff": "aiff",
        "pcm": "pcm16",
    }
    if ext not in format_map:
        raise ValueError(
            f"Unsupported audio format extension: '{ext}'. "
            f"Supported: {', '.join(sorted(format_map.keys()))}"
        )
    return format_map[ext]


__all__ = ["OpenRouterAdapter"]
