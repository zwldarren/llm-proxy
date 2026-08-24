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

from llm_proxy.core.adapter import AdapterConfig, register_adapter
from llm_proxy.models import InternalResponse
from llm_proxy.observability.logger import get_logger
from llm_proxy.providers.openai_compatible._base import OpenAICompatibleBase

logger = get_logger(__name__)


@register_adapter("openrouter")
class OpenRouterAdapter(OpenAICompatibleBase):
    """OpenRouter provider using direct HTTP calls to OpenAI-compatible API.

    OpenRouter provides unified access to multiple LLM providers through a single API.
    This adapter extends OpenAICompatibleBase with OpenRouter-specific features.
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

    def __init__(
        self,
        *,
        config: AdapterConfig | None = None,
        **kwargs: Any,
    ):
        if config is not None:
            super().__init__(config=config)
        else:
            kwargs.setdefault("provider_name", "openrouter")
            kwargs.setdefault("base_url", self.DEFAULT_BASE_URL)
            super().__init__(**kwargs)

    def _stream_filter_line(self, line_str: str) -> str | None:
        if line_str.startswith(":"):
            return None
        if line_str.startswith("data: "):
            return line_str[6:].strip()
        return None

    def _post_process_chat_response(
        self, response: dict[str, Any], result: InternalResponse
    ) -> InternalResponse:
        """Extract provider metadata from response."""
        unknown_fields = self._get_serializer().extract_unknown_response_fields(response)
        if unknown_fields:
            result.provider_info.update(unknown_fields)

        # OpenRouter reports cost in usage.cost (non-streaming)
        usage = response.get("usage", {})
        if isinstance(usage, dict):
            cost = usage.get("cost")
            if isinstance(cost, int | float) and cost > 0:
                result.provider_info["openrouter_cost"] = cost

            cost_details = usage.get("cost_details")
            if isinstance(cost_details, dict):
                result.provider_info["openrouter_cost_details"] = cost_details

            is_byok = usage.get("is_byok")
            if isinstance(is_byok, bool):
                result.provider_info["openrouter_is_byok"] = is_byok

        return result

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
        raw_usage = response.get("usage", {})
        openrouter_cost: float | None = None
        if isinstance(raw_usage, dict):
            cost = raw_usage.get("cost")
            if isinstance(cost, int | float) and cost > 0:
                openrouter_cost = cost

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
        body = self._finalize_body(body, request, merge_extra=True)

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

        return self._parse_transcription_response(response_data, request.response_format)

    async def stream_transcription(self, request: Any, **kwargs: Any) -> Any:
        """Stream transcription using OpenRouter's JSON-based STT API."""
        if not request.file:
            raise ValueError("Audio file data is required for transcription")

        url, headers, body = self._build_transcription_request(request, stream=True)
        body = self._finalize_body(body, request, merge_extra=True)

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

        return url, headers, body


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
