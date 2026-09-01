"""Gemini provider implementation using native HTTP calls."""

import asyncio
import base64
import math
import time
from collections import OrderedDict
from collections.abc import AsyncIterator
from threading import Lock
from typing import TYPE_CHECKING, Any, cast

import orjson

from llm_proxy.core.adapter import AdapterConfig, register_adapter
from llm_proxy.core.errors import get_error_type_for_status
from llm_proxy.core.exceptions import ProviderError
from llm_proxy.http.client import AsyncSession, download_image_as_base64
from llm_proxy.models import (
    CustomToolUseBlock,
    InternalEmbeddingRequest,
    InternalEmbeddingResponse,
    InternalRequest,
    InternalResponse,
    InternalSpeechRequest,
    InternalSpeechResponse,
    ToolUseBlock,
)
from llm_proxy.models.image import (
    ImageData,
    ImageSize,
    InternalImageEditRequest,
    InternalImageRequest,
    InternalImageResponse,
)
from llm_proxy.observability.logger import get_logger
from llm_proxy.providers.base import BaseHttpProvider
from llm_proxy.providers.capabilities import (
    AudioCapabilityMixin,
    ChatCapabilityMixin,
    EmbeddingCapabilityMixin,
)
from llm_proxy.providers.capabilities.image import normalize_image_stream_chunk
from llm_proxy.serialization.gemini.speech import (
    build_speech_config,
    parse_audio_mime,
    pcm_to_wav,
    resolve_voice,
    wav_header,
)
from llm_proxy.serialization.providers import get_provider_serializer
from llm_proxy.streaming.sse_parse import strip_sse_data_prefix

if TYPE_CHECKING:
    from llm_proxy.models.provider import ProviderModelInfo

logger = get_logger(__name__)

# Legacy generateContent serializer (embeddings, models list, and the default
# chat dialect). The Interactions dialect is selected per-instance when the
# provider's metadata.api_variant is "interactions".
_serializer = get_provider_serializer("gemini")

#: Provider metadata key selecting the upstream API dialect.
#: "generate_content" (default) | "interactions"
_API_VARIANT_EXTRA_KEY = "api_variant"

#: Path suffix appended to the base URL for the Interactions dialect.
#: The docs migration guide shows v1beta2/interactions while the API
#: reference serves the beta under v1beta/interactions; the configured
#: base URL already carries the version segment, so only the resource path
#: is appended here.
_INTERACTIONS_PATH = "/interactions"

# Gemini ImageConfig.aspectRatio supports a fixed set of ratios. OpenAI sizes
# like 1792x1024 (7:4) are not in it, so we map to the nearest supported one.
_SUPPORTED_ASPECT_RATIOS: dict[str, float] = {
    "1:1": 1.0,
    "1:4": 0.25,
    "4:1": 4.0,
    "1:8": 0.125,
    "8:1": 8.0,
    "2:3": 2 / 3,
    "3:2": 1.5,
    "3:4": 0.75,
    "4:3": 4 / 3,
    "4:5": 0.8,
    "5:4": 1.25,
    "9:16": 9 / 16,
    "16:9": 16 / 9,
    "21:9": 21 / 9,
}


#: Guidance appended to Gemini's "User location is not supported" error.
#: Google geolocates API requests by their source IP and the Gemini API is
#: not available in every country; this usually means the proxy's egress IP
#: resolved to an unsupported region (or Google misclassified a datacenter
#: IP). Operators fix it by changing the egress path, not the request.
_LOCATION_BLOCK_HINT = (
    "Google geolocates API requests by their source IP, and the Gemini API is "
    "not available in every country. This usually means the proxy's egress IP "
    "resolved to an unsupported region (or Google misclassified a datacenter "
    "IP). Route this provider's outbound traffic through a supported region "
    "(e.g. set HTTPS_PROXY for the process or run the proxy on a server in a "
    "supported country), or check that the API key was created in a supported "
    "region."
)


# OpenAI image parameters that Gemini's generateContent has no equivalent for.
# They are ignored by the adapter; a warning is logged when a client sets them.
_UNSUPPORTED_IMAGE_PARAMS = (
    "quality",
    "style",
    "background",
    "moderation",
    "output_format",
    "output_compression",
    "partial_images",
    "input_fidelity",
)


def _nearest_supported_aspect_ratio(width: int, height: int) -> str:
    """Return the closest Gemini-supported aspect ratio for a pixel size."""
    target = width / height
    return min(
        _SUPPORTED_ASPECT_RATIOS,
        key=lambda ratio: abs(_SUPPORTED_ASPECT_RATIOS[ratio] - target),
    )


@register_adapter("gemini")
class GeminiAdapter(
    ChatCapabilityMixin, EmbeddingCapabilityMixin, AudioCapabilityMixin, BaseHttpProvider
):
    """Gemini provider using direct HTTP calls."""

    _DEFAULT_PROVIDER_NAME = "gemini"

    #: Branding for the admin provider catalog (GET /api/config/provider-types).
    DISPLAY_NAME_EN = "Gemini"
    DISPLAY_NAME_ZH = "Gemini"
    LOBE_ICON_ID = "gemini"
    LOBE_ICON_VARIANT = "color"

    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    _MAX_THOUGHT_SIGNATURE_CACHE = 1000

    # Class-level cache: thought_signatures from Gemini responses keyed by tool_call_id.
    # Shared across all adapter instances since get_adapter() creates a new instance
    # per request. Gemini requires thoughtSignature in functionCall parts when continuing
    # a conversation with tool calls. Since clients may drop this non-standard field,
    # we cache it at the class level and re-attach when building subsequent requests.
    # Uses OrderedDict for LRU eviction: on overflow, only the oldest entry is removed.
    _thought_signature_cache: OrderedDict[str, str] = OrderedDict()
    _thought_signature_cache_lock: Lock = Lock()

    def __init__(
        self,
        *,
        config: AdapterConfig | None = None,
        **kwargs: Any,
    ):
        if config is not None:
            super().__init__(config=config)
        else:
            kwargs.setdefault("provider_name", "gemini")
            kwargs.setdefault("base_url", self.DEFAULT_BASE_URL)
            super().__init__(**kwargs)

        # Upstream API dialect switch (metadata.api_variant). Defaults to the
        # legacy generateContent dialect; "interactions" selects Google's GA
        # Interactions API (see docs/adr/0010-gemini-interactions-variant.md).
        # The choice is per provider instance, so operators can grayscale per
        # provider and roll back instantly.
        self._api_variant = str(self._extra_config.get(_API_VARIANT_EXTRA_KEY, "generate_content"))
        # Acquired through the registry, never by direct import (CONTEXT.md):
        # the registry applies its post-create hook (_registered_provider_name)
        # and is the single source of truth for provider serializer keys.
        self._serializer = (
            get_provider_serializer("gemini-interactions")
            if self._api_variant == "interactions"
            else _serializer
        )

    @property
    def api_variant(self) -> str:
        """The upstream API dialect selected for this adapter instance."""
        return self._api_variant

    def _is_interactions_variant(self) -> bool:
        return self._api_variant == "interactions"

    def _finalize_body(
        self,
        body: dict[str, Any],
        request: Any,
        *,
        exempt_keys: set[str] | None = None,
        merge_extra: bool = True,
    ) -> dict[str, Any]:
        """Enforce the Interactions extra whitelist on every body path.

        The chat serializer already filters ``request.extra`` itself
        (``EXTRA_ALLOWED_KEYS``) and consumes the whitelisted keys into the
        body; speech/image bodies go through the generic ``_finalize_body``
        merge instead. The serializer owns the whitelist policy
        (``filter_extra_for_body``): under the Interactions variant only
        whitelisted keys may reach the body — the API rejects unknown
        top-level fields, unlike generateContent. Whitelisted keys are also
        exempt from the unknown-fields policy (they are supported fields),
        so the default "ignore" policy cannot strip them from the body and
        the "error" policy cannot reject them.
        """
        if not self._is_interactions_variant():
            return super()._finalize_body(
                body, request, exempt_keys=exempt_keys, merge_extra=merge_extra
            )

        extra = getattr(request, "extra", None) or {}
        filtered = self._serializer.filter_extra_for_body(extra)
        if merge_extra and filtered:
            body = self._merge_extra(body, filtered)
        exempt = (exempt_keys or set()) | set(filtered)
        override_keys = getattr(request, "_override_injected_keys", None)
        if override_keys:
            exempt = exempt | override_keys
        return self._apply_field_policy(body, extra, exempt_keys=exempt)

    def _build_headers(
        self,
        auth_header: str | None = None,
        auth_prefix: str | None = None,
    ) -> dict[str, str]:
        # Gemini uses x-goog-api-key header (not URL query param)
        return super()._build_headers(auth_header="x-goog-api-key", auth_prefix="")

    def _build_url(self, model: str, stream: bool = False) -> str:
        if self._is_interactions_variant():
            # Single endpoint for both stream and non-stream; streaming is
            # selected via the body's "stream": true flag.
            return f"{self._base_url}{_INTERACTIONS_PATH}"
        method = "streamGenerateContent" if stream else "generateContent"
        model_name = model.removeprefix("models/")
        suffix = "?alt=sse" if stream else ""
        return f"{self._base_url}/models/{model_name}:{method}{suffix}"

    async def _download_images_in_gemini_contents(
        self, contents: list[dict[str, Any]], client: AsyncSession
    ) -> list[dict[str, Any]]:
        """Download HTTP(S) image/audio URLs in contents and replace with base64 inline_data.

        Walks the serializer-produced contents structure, finds any file_data parts
        with HTTP(S) file_uri, downloads them, and replaces with inline_data.
        Video URLs are passed through as-is since Gemini fetches them server-side.
        """
        download_tasks: list[tuple[int, int, str, str]] = []

        for msg_idx, msg in enumerate(contents):
            parts = msg.get("parts", [])
            for part_idx, part in enumerate(parts):
                file_data = part.get("file_data")
                if isinstance(file_data, dict):
                    file_uri = file_data.get("file_uri", "")
                    mime_type = file_data.get("mime_type", "")
                    if isinstance(file_uri, str) and file_uri.startswith(("http://", "https://")):
                        # Skip video URLs - Gemini fetches them server-side
                        if mime_type.startswith("video/"):
                            continue
                        download_tasks.append((msg_idx, part_idx, file_uri, mime_type))

        if not download_tasks:
            return contents

        async def download_one(url: str) -> str | None:
            try:
                result = await download_image_as_base64(client, url)
                return result[0] if result else None
            except Exception:
                return None

        results = await asyncio.gather(*[download_one(url) for _, _, url, _ in download_tasks])
        url_to_data: dict[str, str | None] = {
            url: data_url for (_, _, url, _), data_url in zip(download_tasks, results, strict=True)
        }

        new_contents = []
        for msg in contents:
            new_parts = []
            needs_copy = False
            for part in msg.get("parts", []):
                file_data = part.get("file_data")
                if isinstance(file_data, dict):
                    file_uri = file_data.get("file_uri", "")
                    if (
                        isinstance(file_uri, str)
                        and file_uri in url_to_data
                        and url_to_data[file_uri] is not None
                    ):
                        needs_copy = True
                        data_url: str = url_to_data[file_uri] or ""
                        mime_type, data = self._parse_data_url(data_url)
                        new_parts.append({"inline_data": {"mime_type": mime_type, "data": data}})
                        continue
                new_parts.append(part)
            if needs_copy:
                new_msg = dict(msg, parts=new_parts)
                new_contents.append(new_msg)
            else:
                new_contents.append(msg)

        return new_contents

    async def _download_images_in_interactions_input(
        self, input_items: list[dict[str, Any]], client: AsyncSession
    ) -> list[dict[str, Any]]:
        """Download HTTP(S) image/audio URLs in an Interactions input Step array.

        Steps carry ``content`` arrays of typed Content items; URL-sourced
        images/audio are emitted as ``{"type": …, "uri": …}`` items and get
        replaced with inline ``data`` items here. Video/document URIs pass
        through (server-side fetch).
        """
        download_tasks: list[tuple[int, int, str, str]] = []

        for step_idx, step in enumerate(input_items):
            content = step.get("content", [])
            if not isinstance(content, list):
                continue
            for item_idx, item in enumerate(content):
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type not in ("image", "audio"):
                    continue
                uri = item.get("uri")
                if isinstance(uri, str) and uri.startswith(("http://", "https://")):
                    download_tasks.append((step_idx, item_idx, uri, item.get("mime_type", "")))

        if not download_tasks:
            return input_items

        async def download_one(url: str) -> str | None:
            try:
                result = await download_image_as_base64(client, url)
                return result[0] if result else None
            except Exception:
                return None

        results = await asyncio.gather(*[download_one(url) for _, _, url, _ in download_tasks])
        url_to_data: dict[str, str | None] = {
            url: data_url for (_, _, url, _), data_url in zip(download_tasks, results, strict=True)
        }

        new_input = []
        for step in input_items:
            content = step.get("content", [])
            if not isinstance(content, list):
                new_input.append(step)
                continue
            new_content = []
            needs_copy = False
            for item in content:
                if not isinstance(item, dict):
                    new_content.append(item)
                    continue
                uri = item.get("uri")
                if isinstance(uri, str) and uri in url_to_data and url_to_data[uri] is not None:
                    needs_copy = True
                    data_url: str = url_to_data[uri] or ""
                    mime_type, data = self._parse_data_url(data_url)
                    new_item = {
                        "type": item.get("type", "image"),
                        "data": data,
                        "mime_type": item.get("mime_type") or mime_type,
                    }
                    new_content.append(new_item)
                    continue
                new_content.append(item)
            if needs_copy:
                new_input.append(dict(step, content=new_content))
            else:
                new_input.append(step)

        return new_input

    async def _prepare_input_images(self, body: dict[str, Any], client: AsyncSession) -> None:
        """Pre-download HTTP(S) image/audio URLs in the request body, in place.

        Dispatches on the active API variant: the legacy dialect walks
        ``contents[].parts[].file_data`` while the Interactions dialect walks
        the Step-array ``input`` content items.
        """
        if self._is_interactions_variant():
            body["input"] = await self._download_images_in_interactions_input(
                body.get("input", []), client
            )
        else:
            body["contents"] = await self._download_images_in_gemini_contents(
                body.get("contents", []), client
            )

    @staticmethod
    def _parse_data_url(data_url: str) -> tuple[str, str]:
        if not data_url.startswith("data:"):
            return "image/png", data_url

        try:
            header, data = data_url.split(",", 1)
            mime_part = header.split(":")[1].split(";")[0]
            return mime_part, data
        except ValueError, IndexError:
            return "image/png", data_url.split(",", 1)[-1] if "," in data_url else ""

    def _enrich_conversation_with_thought_signatures(self, conversation) -> None:
        """Re-attach cached thought_signatures to tool-use blocks in the conversation.

        Handles both ToolUseBlock and CustomToolUseBlock: the Interactions
        stateless replay requires the signature on every function_call step,
        and custom tools (e.g. codex's ``exec``) parse as CustomToolUseBlock.
        """
        with self._thought_signature_cache_lock:
            if not self._thought_signature_cache:
                return
            for msg in conversation.messages:
                for block in msg.content:
                    if isinstance(block, (ToolUseBlock, CustomToolUseBlock)) and (
                        not block.extra.get("thought_signature")
                        and block.id in self._thought_signature_cache
                    ):
                        block.extra["thought_signature"] = self._thought_signature_cache[block.id]

    def _cache_thought_signatures(self, blocks: list[ToolUseBlock]) -> None:
        """Cache thought_signatures from tool use blocks for follow-up requests.

        Uses LRU eviction via OrderedDict: on overflow, only the oldest entry
        is removed instead of clearing the entire cache.
        """
        with self._thought_signature_cache_lock:
            cache = self._thought_signature_cache
            max_cache = self._MAX_THOUGHT_SIGNATURE_CACHE
            for block in blocks:
                if isinstance(block, (ToolUseBlock, CustomToolUseBlock)) and (
                    ts := block.extra.get("thought_signature")
                ):
                    if block.id in cache:
                        cache.move_to_end(block.id)
                    else:
                        if len(cache) >= max_cache:
                            cache.popitem(last=False)
                    cache[block.id] = ts

    def _parse_error_response(self, status_code: int, error_body: dict[str, Any]):
        error_data = error_body.get("error", {})
        if isinstance(error_data, dict):
            message = error_data.get("message", str(error_body))
            status = error_data.get("status", "")
            error_type = _map_gemini_error_status(status, status_code)
        else:
            message = str(error_data) if error_data else str(error_body)
            error_type = get_error_type_for_status(status_code)

        if "user location is not supported" in message.lower():
            message = f"{message} {_LOCATION_BLOCK_HINT}"

        return ProviderError(
            message=message,
            error_type=error_type,
            status_code=status_code,
            provider_name=self.provider_name,
            original_error=error_body,
        )

    def _build_chat_raw(self, request, context):
        self._enrich_conversation_with_thought_signatures(request.conversation)
        return self._serializer.build_provider_request(request, context)

    async def chat_completion(self, request: InternalRequest, **_kwargs: Any) -> InternalResponse:
        client = await self._get_client()

        url = self._build_url(request.model, stream=False)
        headers = self._build_headers()
        outbound = self._build_outbound_body(request, request_type="chat")
        if outbound.json_body is None:
            raise ValueError("_build_outbound_body returned no json_body for chat")
        await self._prepare_input_images(outbound.json_body, client)
        body = outbound.json_body

        async def _make_request():
            new_client = await self._get_client()
            response = await new_client.post(url, headers=headers, json=body)
            await self._check_response_status(response)
            return response.json()

        response = await self._with_retry(_make_request)
        result = self._parse_response(
            self._serializer, response, model=request.model, request=request
        )
        self._cache_thought_signatures([b for b in result.output if isinstance(b, ToolUseBlock)])
        return result

    async def stream_chat_completion(
        self,
        request: InternalRequest,
        cancel_token=None,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str | dict[str, Any]]:
        client = await self._get_client()

        url = self._build_url(request.model, stream=True)
        headers = self._build_headers()
        outbound = self._build_outbound_body(request, request_type="chat")
        if outbound.json_body is None:
            raise ValueError("_build_outbound_body returned no json_body for chat")
        await self._prepare_input_images(outbound.json_body, client)
        body = outbound.json_body
        if self._is_interactions_variant():
            # Interactions selects streaming via a body flag, not a URL suffix.
            body["stream"] = True
        stream_timeout = self._get_stream_timeout()

        async def _stream_generator():
            new_client = await self._get_client()
            converter = self._serializer.get_chunk_converter(
                model=request.model, request_id=request.request_id or ""
            )

            try:
                async with self._streaming_post(
                    new_client,
                    url,
                    headers=headers,
                    json=body,
                    timeout=stream_timeout,
                ) as response:
                    assert response.status_code is not None
                    if response.status_code >= 400:
                        await response.aread()
                        try:
                            error_body = response.json()
                        except Exception:
                            error_text = response.text
                            raise ProviderError(
                                message=error_text or f"HTTP {response.status_code}",
                                error_type=get_error_type_for_status(response.status_code),
                                status_code=response.status_code,
                                provider_name=self.provider_name,
                            ) from None
                        raise self._parse_error_response(response.status_code, error_body)

                    async for line in cast(
                        AsyncIterator[bytes],
                        response.iter_lines(),
                    ):
                        if cancel_token and cancel_token.is_set():
                            break

                        if not line:
                            continue

                        line_str = strip_sse_data_prefix(line.decode("utf-8"))
                        if not line_str or line_str == "[DONE]":
                            continue

                        try:
                            data = orjson.loads(line_str)
                            openai_chunk = converter.convert_chunk(data)
                            if openai_chunk:
                                yield openai_chunk
                        except orjson.JSONDecodeError:
                            continue

                self._cache_thought_signatures(getattr(converter, "_accumulated_output", []))

                yield "[DONE]"

            except ProviderError:
                raise
            except Exception as e:
                error = await self._handle_http_error(e)
                raise error from e

        return self._with_retry_generator(_stream_generator, cancel_token=cancel_token)

    # ------------------------------------------------------------------
    # Speech (TTS)
    # https://ai.google.dev/gemini-api/docs/speech-generation
    # ------------------------------------------------------------------

    # Gemini TTS returns raw PCM (audio/L16). We can serve it as-is ("pcm")
    # or wrap it in a WAV container; every other requested format falls back
    # to WAV (never mislabeled as mp3/aac/etc.).
    _SPEECH_PCM_CONTENT_TYPE = "audio/L16"
    _SPEECH_WAV_CONTENT_TYPE = "audio/wav"

    @staticmethod
    def _wants_pcm(format_name: str | None) -> bool:
        return (format_name or "").lower() == "pcm"

    def _speech_url(
        self, request: InternalSpeechRequest | None = None, *, stream: bool | None = None
    ) -> str:
        if request is None or not request.model:
            raise ValueError("Gemini speech requests require a model")
        if stream is None:
            stream = bool(request.stream)
        return self._build_url(request.model, stream=stream)

    def speech_stream_media_type(self, request: InternalSpeechRequest) -> str | None:
        if self._wants_pcm(request.response_format):
            return self._SPEECH_PCM_CONTENT_TYPE
        return self._SPEECH_WAV_CONTENT_TYPE

    def _build_speech_raw(self, request: InternalSpeechRequest) -> dict[str, Any]:
        """Build a Gemini body for TTS, per the active API variant.

        Provider-consumed extension keys (``speech_config``,
        ``language_code``) are popped from ``request.extra`` so the
        dispatch's extra merge cannot leak them as invalid top-level
        Gemini fields.
        """
        extra_speech_config = request.extra.pop("speech_config", None)
        language_code = request.extra.pop("language_code", None)

        if self._is_interactions_variant():
            return self._build_interactions_speech_raw(
                request,
                extra_speech_config=extra_speech_config,
                language_code=language_code if isinstance(language_code, str) else None,
            )

        # Explicit SpeechConfig wins: params.gemini.speech_config (typed
        # channel) first, then the extra escape hatch; otherwise derive one
        # from the requested voice.
        if request.params.gemini and request.params.gemini.speech_config:
            speech_config = request.params.gemini.speech_config
        elif isinstance(extra_speech_config, dict):
            speech_config = extra_speech_config
        else:
            speech_config = build_speech_config(
                request.voice,
                language_code=language_code if isinstance(language_code, str) else None,
            )

        # Gemini TTS is prompt-steerable: OpenAI's ``instructions`` map
        # naturally onto a directorial prefix.
        text = request.input
        if request.instructions:
            text = f"{request.instructions}\n\n{text}"

        return {
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": speech_config,
            },
        }

    def _build_interactions_speech_raw(
        self,
        request: InternalSpeechRequest,
        *,
        extra_speech_config: Any,
        language_code: str | None,
    ) -> dict[str, Any]:
        """Build an Interactions body for TTS.

        The Interactions dialect takes ``response_format`` (audio) plus
        ``generation_config.speech_config`` as an ARRAY of
        ``{language, speaker, voice}`` objects.
        """
        text = request.input
        if request.instructions:
            text = f"{request.instructions}\n\n{text}"

        speech_config: list[dict[str, Any]]
        if isinstance(extra_speech_config, list):
            # The escape hatch is already in the Interactions array shape.
            speech_config = [s for s in extra_speech_config if isinstance(s, dict)]
        else:
            voice = None
            if request.params.gemini and request.params.gemini.speech_config:
                # Legacy nested voiceConfig form -> Interactions array form.
                legacy = request.params.gemini.speech_config
                vc = legacy.get("voiceConfig", {}) if isinstance(legacy, dict) else {}
                pvc = vc.get("prebuiltVoiceConfig", {}) if isinstance(vc, dict) else {}
                voice = pvc.get("voiceName") if isinstance(pvc, dict) else None
            if not voice:
                voice = resolve_voice(request.voice)
            item: dict[str, Any] = {"voice": voice}
            if language_code:
                item["language"] = language_code
            speech_config = [item]

        return {
            "model": request.model,
            "input": [{"type": "text", "text": text}],
            "response_format": {"type": "audio"},
            "generation_config": {"speech_config": speech_config},
            "store": False,
        }

    @staticmethod
    def _extract_speech_audio(
        payload: dict[str, Any],
    ) -> tuple[list[bytes], int | None]:
        """Extract (audio chunks, sample rate) from a Gemini payload.

        Works for complete responses and streaming chunks of BOTH dialects:
        legacy ``candidates[].content.parts[].inlineData`` and Interactions
        ``steps[].content[].audio`` items / ``step.delta`` audio events.
        """
        chunks: list[bytes] = []
        sample_rate: int | None = None

        # Interactions dialect: audio lives in model_output step content or
        # step.delta events (the adapter also feeds parsed events here).
        audio_payloads: list[tuple[Any, Any]] = []
        for step in payload.get("steps", []) or []:
            if not isinstance(step, dict):
                continue
            for item in step.get("content", []) or []:
                if isinstance(item, dict) and item.get("type") == "audio":
                    audio_payloads.append((item.get("data"), item.get("mime_type") or ""))
        delta = payload.get("delta")
        if isinstance(delta, dict) and delta.get("type") == "audio":
            audio_payloads.append((delta.get("data"), delta.get("mime_type") or ""))
        for data, mime_type in audio_payloads:
            _, rate = parse_audio_mime(str(mime_type))
            sample_rate = rate
            if isinstance(data, str) and data:
                try:
                    chunks.append(base64.b64decode(data))
                except ValueError:
                    logger.warning("Skipping malformed base64 audio chunk from Gemini TTS")
        if chunks:
            return chunks, sample_rate

        for candidate in payload.get("candidates", []):
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content", {})
            for part in content.get("parts", []):
                if not isinstance(part, dict):
                    continue
                inline = part.get("inlineData") or part.get("inline_data")
                if not isinstance(inline, dict):
                    continue
                mime_type = inline.get("mimeType") or inline.get("mime_type") or ""
                if not str(mime_type).startswith("audio/"):
                    continue
                _, rate = parse_audio_mime(str(mime_type))
                sample_rate = rate
                data = inline.get("data")
                if isinstance(data, str) and data:
                    try:
                        chunks.append(base64.b64decode(data))
                    except ValueError:
                        # binascii.Error subclasses ValueError; skip malformed
                        # chunks instead of crashing the whole request.
                        logger.warning("Skipping malformed base64 audio chunk from Gemini TTS")
        return chunks, sample_rate

    def _encode_speech_output(
        self, pcm: bytes, sample_rate: int, response_format: str | None
    ) -> tuple[bytes, str]:
        """Encode PCM into the negotiated wire format: (content, content_type)."""
        if self._wants_pcm(response_format):
            return pcm, self._SPEECH_PCM_CONTENT_TYPE
        if response_format and response_format.lower() not in ("wav", "pcm"):
            logger.info(
                "Gemini TTS outputs raw PCM; returning WAV instead of requested "
                f"response_format={response_format!r}."
            )
        return pcm_to_wav(pcm, sample_rate=sample_rate), self._SPEECH_WAV_CONTENT_TYPE

    async def speech(
        self, request: InternalSpeechRequest, **_kwargs: Any
    ) -> InternalSpeechResponse:
        """Generate speech via Gemini's generateContent endpoint."""
        url = self._speech_url(request)
        headers = self._build_headers()
        outbound = self._build_outbound_body(request, request_type="speech")
        if outbound.json_body is None:
            raise ValueError("Expected json_body for speech request, got None")
        body = outbound.json_body

        async def _make_request():
            client = await self._get_client()
            response = await client.post(url, headers=headers, json=body)
            await self._check_response_status(response)
            return response.json()

        response = await self._with_retry(_make_request)

        chunks, sample_rate = self._extract_speech_audio(response)
        if not chunks:
            raise ProviderError(
                message=(
                    "Gemini TTS returned no audio in the response. "
                    "Check that the model supports speech generation "
                    "(e.g. gemini-3.1-flash-tts-preview)."
                ),
                error_type="api_error",
                status_code=500,
                provider_name=self.provider_name,
                original_error=response,
            )

        content, content_type = self._encode_speech_output(
            b"".join(chunks), sample_rate or 24000, request.response_format
        )
        return InternalSpeechResponse(
            content=content,
            content_type=content_type,
            request_id=request.request_id,
        )

    async def stream_speech(
        self, request: InternalSpeechRequest, **_kwargs: Any
    ) -> AsyncIterator[bytes]:
        """Stream speech via Gemini's streamGenerateContent endpoint.

        Gemini streams JSON SSE chunks with base64 PCM inlineData; this
        decodes them into a raw audio byte stream (with a WAV header first
        unless raw PCM was requested).
        """
        url = self._speech_url(request, stream=True)
        headers = self._build_headers()
        outbound = self._build_outbound_body(request, request_type="speech")
        if outbound.json_body is None:
            raise ValueError("Expected json_body for speech request, got None")
        body = outbound.json_body
        if self._is_interactions_variant():
            body["stream"] = True
        stream_timeout = self._get_stream_timeout()
        wants_pcm = self._wants_pcm(request.response_format)

        async def _generator():
            client = await self._get_client()
            header_sent = False
            try:
                async with self._streaming_post(
                    client, url, headers=headers, json=body, timeout=stream_timeout
                ) as response:
                    assert response.status_code is not None
                    if response.status_code >= 400:
                        await response.aread()
                        try:
                            error_body = response.json()
                        except Exception:
                            error_text = response.text
                            raise ProviderError(
                                message=error_text or f"HTTP {response.status_code}",
                                error_type=get_error_type_for_status(response.status_code),
                                status_code=response.status_code,
                                provider_name=self.provider_name,
                            ) from None
                        raise self._parse_error_response(response.status_code, error_body)

                    async for line in cast(AsyncIterator[bytes], response.iter_lines()):
                        if not line:
                            continue
                        line_str = strip_sse_data_prefix(line.decode("utf-8"))
                        if not line_str or line_str == "[DONE]":
                            continue
                        try:
                            data = orjson.loads(line_str)
                        except orjson.JSONDecodeError:
                            continue
                        chunks, sample_rate = self._extract_speech_audio(data)
                        if not chunks:
                            continue
                        if not header_sent:
                            header_sent = True
                            if not wants_pcm:
                                # Streaming WAV: header with unknown length,
                                # then raw PCM frames.
                                yield wav_header(None, sample_rate=sample_rate or 24000)
                        for chunk in chunks:
                            yield chunk
            except ProviderError:
                raise
            except Exception as e:
                error = await self._handle_http_error(e)
                raise error from e

        return cast(AsyncIterator[bytes], self._with_retry_generator(_generator))

    async def _embed_single(
        self, model: str, url: str, headers: dict[str, str], body: dict[str, Any]
    ) -> dict[str, Any]:
        async def _make_request():
            client = await self._get_client()
            response = await client.post(url, headers=headers, json=body)
            await self._check_response_status(response)
            return response.json()

        return await self._with_retry(_make_request)

    async def _embed_batch(
        self,
        model: str,
        headers: dict[str, str],
        inputs: list[str],
        dimensions: int | None = None,
    ) -> dict[str, Any]:
        url = f"{self._base_url}/models/{model}:batchEmbedContents"
        batch_requests = []
        for text in inputs:
            req_item: dict[str, Any] = {
                "model": f"models/{model}",
                "content": {"parts": [{"text": text}]},
            }
            if dimensions is not None:
                req_item["embedContentConfig"] = {"outputDimensionality": dimensions}
            batch_requests.append(req_item)

        async def _make_request():
            client = await self._get_client()
            response = await client.post(url, headers=headers, json={"requests": batch_requests})
            await self._check_response_status(response)
            return response.json()

        return await self._with_retry(_make_request)

    async def embeddings(
        self, request: InternalEmbeddingRequest, **_kwargs: Any
    ) -> InternalEmbeddingResponse:
        serializer = _serializer

        model = request.model.removeprefix("models/")
        headers = self._build_headers()

        if not isinstance(request.input, list):
            url = f"{self._base_url}/models/{model}:embedContent"
            body = serializer.build_provider_embedding_request(request)
            response = await self._embed_single(model, url, headers, body)
            return serializer.parse_provider_embedding_response(response, model=request.model)

        inputs: list[str] = request.input
        if len(inputs) == 1:
            url = f"{self._base_url}/models/{model}:embedContent"
            single_request = InternalEmbeddingRequest(
                model=request.model, input=inputs[0], dimensions=request.dimensions
            )
            body = serializer.build_provider_embedding_request(single_request)
            response = await self._embed_single(model, url, headers, body)
            return serializer.parse_provider_embedding_response(response, model=request.model)

        # Multiple inputs: try batchEmbedContents first
        _FALLBACK_STATUSES = (400, 401, 403, 404, 405)

        try:
            response = await self._embed_batch(model, headers, inputs, request.dimensions)
            return serializer.parse_provider_embedding_response(response, model=request.model)
        except ProviderError as e:
            if e.status_code not in _FALLBACK_STATUSES:
                raise
            logger.warning(
                f"batchEmbedContents failed (HTTP {e.status_code}), "
                f"falling back to individual embedContent calls: {e.message}",
            )

        # Fallback: individual embedContent calls
        from llm_proxy.models.embedding import EmbeddingData
        from llm_proxy.models.types import Usage

        url = f"{self._base_url}/models/{model}:embedContent"
        data_list: list[EmbeddingData] = []
        total_input = 0
        for idx, text in enumerate(inputs):
            single_request = InternalEmbeddingRequest(
                model=request.model, input=text, dimensions=request.dimensions
            )
            body = serializer.build_provider_embedding_request(single_request)
            response = await self._embed_single(model, url, headers, body)
            parsed = serializer.parse_provider_embedding_response(response, model=request.model)
            for d in parsed.data:
                data_list.append(EmbeddingData(embedding=d.embedding, index=idx))
            if parsed.usage is not None:
                total_input += parsed.usage.input_tokens

        usage = None
        if total_input:
            usage = Usage(input_tokens=total_input)

        return InternalEmbeddingResponse(model=request.model, data=data_list, usage=usage)

    @staticmethod
    def _parse_image_size_to_gemini(size: ImageSize | None) -> dict[str, Any] | None:
        """Convert ImageSize to Gemini aspect_ratio and image_size."""
        if size is None:
            return None
        w, h = size.width, size.height
        if w <= 0 or h <= 0:
            return None

        # Compute reduced aspect ratio; map ratios Gemini does not support
        # (e.g. 7:4 from 1792x1024) to the nearest supported one.
        gcd = math.gcd(w, h)
        aspect_ratio = f"{w // gcd}:{h // gcd}"
        if aspect_ratio not in _SUPPORTED_ASPECT_RATIOS:
            mapped = _nearest_supported_aspect_ratio(w, h)
            logger.warning(
                f"Gemini image generation does not support aspect ratio {aspect_ratio}; "
                f"using {mapped} instead"
            )
            aspect_ratio = mapped

        # Map max dimension to Gemini imageSize using the supported 512/1K/2K/4K
        # shorthand. See Gemini ImageConfig docs for valid values.
        max_dim = max(w, h)
        if max_dim <= 512:
            image_size = "512"
        elif max_dim <= 1024:
            image_size = "1K"
        elif max_dim <= 2048:
            image_size = "2K"
        else:
            image_size = "4K"

        return {"aspect_ratio": aspect_ratio, "image_size": image_size}

    @staticmethod
    def _warn_unsupported_image_params(request: Any, *, edit: bool) -> None:
        """Log a warning for OpenAI image parameters Gemini cannot honor."""
        unsupported = [
            field
            for field in _UNSUPPORTED_IMAGE_PARAMS
            if getattr(request, field, None) is not None
        ]
        if unsupported:
            logger.warning(
                f"Gemini image {'editing' if edit else 'generation'} does not support "
                f"parameter(s): {', '.join(unsupported)}; they will be ignored"
            )

    @staticmethod
    def _warn_unsupported_image_count(request: Any, *, edit: bool) -> None:
        """Log a warning when a client requests more than one image."""
        n = getattr(request, "n", 1) or 1
        if n > 1:
            logger.warning(
                f"Gemini image {'editing' if edit else 'generation'} does not support "
                f"n>1 (requested n={n}); generating a single image"
            )

    def _build_gemini_image_request(
        self,
        prompt: str,
        model: str,
        size: ImageSize | None = None,
    ) -> dict[str, Any]:
        """Build a Gemini body for image generation, per the active variant.

        Does NOT merge request.extra — the dispatch's _finalize_body handles that.
        """
        gemini_size = self._parse_image_size_to_gemini(size)

        if self._is_interactions_variant():
            response_format: dict[str, Any] = {"type": "image"}
            if gemini_size is not None:
                response_format["aspect_ratio"] = gemini_size["aspect_ratio"]
                response_format["image_size"] = gemini_size["image_size"]
            return {
                "model": model,
                "input": [{"type": "text", "text": prompt}],
                "response_format": response_format,
                "store": False,
            }

        generation_config: dict[str, Any] = {
            "responseModalities": ["IMAGE"],
        }
        if gemini_size is not None:
            generation_config["imageConfig"] = {
                "aspectRatio": gemini_size["aspect_ratio"],
                "imageSize": gemini_size["image_size"],
            }

        return {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": generation_config,
        }

    def _build_image_raw(self, request: Any) -> dict[str, Any]:
        """Build raw Gemini image body without extra merge or field policy.

        Overrides BaseProvider._build_image_raw so the dispatch can route
        image_generation/stream_image_generation through the chokepoint.
        """
        self._warn_unsupported_image_params(request, edit=False)
        self._warn_unsupported_image_count(request, edit=False)
        return self._build_gemini_image_request(
            prompt=request.prompt,
            model=request.model,
            size=request.size,
        )

    async def _gemini_image_response_to_internal(
        self,
        response: dict[str, Any],
        model: str,
    ) -> InternalImageResponse:
        """Parse a Gemini response into InternalImageResponse (both dialects)."""
        images: list[ImageData] = []
        revised_prompt: str | None = None
        usage = None

        if self._is_interactions_variant():
            # Interactions: images/text live in model_output step content.
            for step in response.get("steps", []) or []:
                if not isinstance(step, dict) or step.get("type") != "model_output":
                    continue
                for item in step.get("content", []) or []:
                    if not isinstance(item, dict):
                        continue
                    item_type = item.get("type")
                    if item_type == "image" and item.get("data"):
                        images.append(ImageData(b64_json=item["data"]))
                    elif item_type == "text" and item.get("text") and revised_prompt is None:
                        revised_prompt = item["text"]
            usage = self._parse_image_usage(response)
        else:
            candidates = response.get("candidates", [])
            if candidates:
                candidate = candidates[0]
                content = candidate.get("content", {})
                parts = content.get("parts", [])

                for part in parts:
                    match part:
                        case {"text": str(text)} if text and revised_prompt is None:
                            revised_prompt = text
                        case {"inlineData": {"data": str(b64_data)}}:
                            images.append(ImageData(b64_json=b64_data))
                        case _:
                            pass

            usage = self._parse_image_usage(response)

        if not images:
            raise ProviderError(
                message=(
                    "Gemini image generation returned no images in the response. "
                    "Check that the model supports image generation."
                ),
                error_type="api_error",
                status_code=500,
                provider_name=self.provider_name,
                original_error=response,
            )

        if revised_prompt and images:
            images[0].revised_prompt = revised_prompt

        return InternalImageResponse(
            created=int(time.time()),
            data=images,
            model=model,
            usage=usage,
        )

    def _parse_image_usage(self, response: dict[str, Any]) -> Any:
        """Build the canonical Usage record from an image response.

        Dispatches on the active dialect: the Interactions vocabulary
        (``total_input_tokens`` / ``total_output_tokens``) or the legacy
        ``usageMetadata`` dict. Returns None when the response carries no
        usage.
        """
        from llm_proxy.models.types import Usage

        if self._is_interactions_variant():
            raw_usage = response.get("usage")
            if not isinstance(raw_usage, dict):
                return None
            from llm_proxy.serialization.gemini_interactions.usage import (
                interactions_billable_token_counts,
            )

            input_tokens, output_tokens = interactions_billable_token_counts(
                raw_usage, has_search_grounding=False
            )
            return Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=raw_usage.get("total_tokens"),
            )

        if "usageMetadata" not in response:
            return None
        meta = response["usageMetadata"]
        return Usage(
            input_tokens=meta.get("promptTokenCount", 0),
            output_tokens=meta.get("candidatesTokenCount", 0),
            total_tokens=meta.get("totalTokenCount", 0),
        )

    async def image_generation(
        self,
        request: InternalImageRequest,
        **kwargs: Any,
    ) -> InternalImageResponse:
        """Generate images using Gemini's generateContent endpoint."""
        url = self._build_url(request.model, stream=False)
        headers = self._build_headers()

        outbound = self._build_outbound_body(request, request_type="image_generation")
        if outbound.json_body is None:
            raise ValueError("_build_outbound_body returned no json_body for image_generation")
        body = outbound.json_body

        async def _make_request():
            client = await self._get_client()
            response = await client.post(url, headers=headers, json=body)
            await self._check_response_status(response)
            return response.json()

        response = await self._with_retry(_make_request)
        return await self._gemini_image_response_to_internal(response, model=request.model)

    async def stream_image_generation(
        self,
        request: InternalImageRequest,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream image generation using Gemini's stream endpoint.

        The legacy dialect streams ``streamGenerateContent`` chunks while the
        Interactions dialect streams typed SSE events on the single
        ``/interactions`` endpoint (``step.delta`` image events +
        ``interaction.completed`` usage). Both are normalized to the same
        OpenAI Images event shape.
        """
        url = self._build_url(request.model, stream=True)
        headers = self._build_headers()

        outbound = self._build_outbound_body(request, request_type="image_generation")
        if outbound.json_body is None:
            raise ValueError("_build_outbound_body returned no json_body for image_generation")
        body = outbound.json_body
        if self._is_interactions_variant():
            body["stream"] = True

        stream_timeout = self._get_stream_timeout()

        async def _generator():
            created_at = int(time.time())
            partial_image_index = 0
            completed = False
            last_b64_json: str | None = None
            client = await self._get_client()
            try:
                async with self._streaming_post(
                    client, url, headers=headers, json=body, timeout=stream_timeout
                ) as response:
                    assert response.status_code is not None
                    if response.status_code >= 400:
                        await response.aread()
                        try:
                            error_body = response.json()
                        except Exception:
                            error_text = response.text
                            raise ProviderError(
                                message=error_text or f"HTTP {response.status_code}",
                                error_type=get_error_type_for_status(response.status_code),
                                status_code=response.status_code,
                                provider_name=self.provider_name,
                            ) from None
                        raise self._parse_error_response(response.status_code, error_body)

                    async for line in cast(AsyncIterator[bytes], response.iter_lines()):
                        if not line:
                            continue

                        line_str = strip_sse_data_prefix(line.decode("utf-8"))
                        if not line_str or line_str == "[DONE]":
                            continue

                        try:
                            data = orjson.loads(line_str)
                            if self._is_interactions_variant():
                                normalized = self._interactions_image_stream_events(
                                    data,
                                    created_at=created_at,
                                )
                            else:
                                normalized = normalize_image_stream_chunk(
                                    f"data: {line_str}\n\n",
                                    created_at=created_at,
                                    partial_image_index=partial_image_index,
                                )
                            partial_image_index += sum(
                                item.get("type", "").endswith(".partial_image")
                                for item in normalized
                            )
                            for event in normalized:
                                if event.get("type") == "image_generation.completed":
                                    if completed:
                                        continue
                                    completed = True
                                if event.get("type", "").endswith(".partial_image"):
                                    last_b64_json = event.get("b64_json")
                                event_type = event["type"]
                                yield (
                                    f"event: {event_type}\n"
                                    f"data: {orjson.dumps(event).decode('utf-8')}\n\n"
                                )
                        except orjson.JSONDecodeError:
                            continue

                if not completed:
                    event: dict[str, Any] = {
                        "type": "image_generation.completed",
                        "created_at": created_at,
                    }
                    if last_b64_json is not None:
                        event["b64_json"] = last_b64_json
                    yield (
                        "event: image_generation.completed\n"
                        f"data: {orjson.dumps(event).decode('utf-8')}\n\n"
                    )
                yield "data: [DONE]\n\n"

            except ProviderError:
                raise
            except Exception as e:
                error = await self._handle_http_error(e)
                raise error from e

        return self._with_retry_generator(_generator)

    @staticmethod
    def _interactions_image_stream_events(
        data: dict[str, Any],
        *,
        created_at: int,
    ) -> list[dict[str, Any]]:
        """Normalize Interactions SSE events into OpenAI Images event dicts.

        Mirrors ``normalize_image_stream_chunk`` for the Interactions dialect:
        ``step.delta`` image events become ``image_generation.partial_image``
        and ``interaction.completed`` becomes ``image_generation.completed``
        with mapped usage.
        """
        normalized: list[dict[str, Any]] = []
        event_type = data.get("type") or data.get("event_type")

        if event_type == "step.delta":
            delta = data.get("delta")
            if isinstance(delta, dict) and delta.get("type") == "image" and delta.get("data"):
                normalized.append(
                    {
                        "type": "image_generation.partial_image",
                        "b64_json": delta["data"],
                        "created_at": created_at,
                    }
                )
            return normalized

        if event_type == "interaction.completed":
            interaction = data.get("interaction") or {}
            usage = interaction.get("usage")
            completed: dict[str, Any] = {
                "type": "image_generation.completed",
                "created_at": created_at,
            }
            if isinstance(usage, dict):
                from llm_proxy.serialization.gemini_interactions.usage import (
                    interactions_billable_token_counts,
                    interactions_input_tokens_by_modality,
                    interactions_web_search_requests,
                )

                has_search = interactions_web_search_requests(usage) > 0
                input_tokens, output_tokens = interactions_billable_token_counts(
                    usage, has_search_grounding=has_search
                )
                total = usage.get("total_tokens")
                by_modality = interactions_input_tokens_by_modality(usage)
                completed["usage"] = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total if total is not None else input_tokens + output_tokens,
                    "input_tokens_details": {
                        "text_tokens": by_modality.get("text", input_tokens),
                        "image_tokens": by_modality.get("image", 0),
                    },
                }
            normalized.append(completed)
            return normalized

        return normalized

    def _build_image_edit_raw(self, request: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build raw Gemini image edit body without extra merge or field policy.

        Overrides BaseProvider._build_image_edit_raw so the dispatch routes
        image_edit through the chokepoint.  Downloads are handled separately
        in image_edit() after the chokepoint (mirroring the chat pattern).
        """
        self._warn_unsupported_image_params(request, edit=True)
        self._warn_unsupported_image_count(request, edit=True)

        if self._is_interactions_variant():
            return self._build_interactions_image_edit_raw(request), {}

        parts: list[dict[str, Any]] = [{"text": request.prompt}]

        for img_ref in request.images:
            if img_ref.file is not None:
                parts.append(
                    {
                        "inline_data": {
                            "mime_type": img_ref.content_type or "image/png",
                            "data": base64.b64encode(bytes(img_ref.file)).decode("ascii"),
                        }
                    }
                )
            elif img_ref.file_id:
                parts.append({"file_data": {"file_uri": img_ref.file_id, "mime_type": "image/png"}})
            elif img_ref.image_url:
                if img_ref.image_url.startswith("data:"):
                    mime_type, data = self._parse_data_url(img_ref.image_url)
                    parts.append({"inline_data": {"mime_type": mime_type, "data": data}})
                else:
                    parts.append(
                        {
                            "file_data": {
                                "file_uri": img_ref.image_url,
                                "mime_type": "image/png",
                            }
                        }
                    )

        if request.mask:
            if request.mask.file is not None:
                parts.append(
                    {
                        "inline_data": {
                            "mime_type": request.mask.content_type or "image/png",
                            "data": base64.b64encode(bytes(request.mask.file)).decode("ascii"),
                        }
                    }
                )
            elif request.mask.file_id:
                parts.append(
                    {"file_data": {"file_uri": request.mask.file_id, "mime_type": "image/png"}}
                )
            elif request.mask.image_url:
                if request.mask.image_url.startswith("data:"):
                    mime_type, data = self._parse_data_url(request.mask.image_url)
                    parts.append({"inline_data": {"mime_type": mime_type, "data": data}})
                else:
                    parts.append(
                        {
                            "file_data": {
                                "file_uri": request.mask.image_url,
                                "mime_type": "image/png",
                            }
                        }
                    )

        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"responseModalities": ["IMAGE"]},
        }

        return body, {}

    @staticmethod
    def _image_ref_to_input_item(img_ref: Any, mime_type: str = "image/png") -> dict[str, Any]:
        """Convert an image reference to an Interactions Content item."""
        if img_ref.file is not None:
            return {
                "type": "image",
                "data": base64.b64encode(bytes(img_ref.file)).decode("ascii"),
                "mime_type": img_ref.content_type or mime_type,
            }
        if img_ref.file_id:
            return {"type": "image", "uri": img_ref.file_id, "mime_type": mime_type}
        if img_ref.image_url:
            if img_ref.image_url.startswith("data:"):
                mime, data = GeminiAdapter._parse_data_url(img_ref.image_url)
                return {"type": "image", "data": data, "mime_type": mime}
            return {"type": "image", "uri": img_ref.image_url, "mime_type": mime_type}
        return {"type": "image", "uri": "", "mime_type": mime_type}

    def _build_interactions_image_edit_raw(self, request: Any) -> dict[str, Any]:
        """Build an Interactions image-edit body (input items + image format)."""
        input_items: list[dict[str, Any]] = [{"type": "text", "text": request.prompt}]
        for img_ref in request.images:
            input_items.append(self._image_ref_to_input_item(img_ref))
        if request.mask:
            input_items.append(self._image_ref_to_input_item(request.mask))
        return {
            "model": request.model,
            "input": input_items,
            "response_format": {"type": "image"},
            "store": False,
        }

    async def image_edit(
        self,
        request: InternalImageEditRequest,
        **kwargs: Any,
    ) -> InternalImageResponse:
        """Edit images using Gemini's generateContent endpoint."""
        client = await self._get_client()
        url = self._build_url(request.model, stream=False)
        headers = self._build_headers()

        outbound = self._build_outbound_body(request, request_type="image_edit")
        if outbound.json_body is None:
            raise ValueError("_build_outbound_body returned no json_body for image_edit")
        body = outbound.json_body
        await self._prepare_input_images(body, client)

        async def _make_request():
            new_client = await self._get_client()
            response = await new_client.post(url, headers=headers, json=body)
            await self._check_response_status(response)
            return response.json()

        response = await self._with_retry(_make_request)
        return await self._gemini_image_response_to_internal(response, model=request.model)

    async def stream_image_edit(
        self,
        request: InternalImageEditRequest,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Gemini does not support streaming image edits.

        Raising a clean ProviderError (instead of NotImplementedError) keeps
        the failure a 400 invalid_request_error rather than a 500.
        """
        raise ProviderError(
            message=(
                f"{self.provider_name} does not support streaming image editing; "
                "retry with stream=false"
            ),
            error_type="invalid_request_error",
            status_code=400,
            provider_name=self.provider_name,
        )

    _models_data_key = "models"

    def _models_url(self) -> str:
        return f"{self._base_url}/models"

    def _models_headers(self) -> dict[str, str]:
        return self._build_headers()

    def _parse_model(self, raw: dict[str, Any]) -> ProviderModelInfo:
        from llm_proxy.models.provider import ProviderModelInfo

        model_id = raw.get("name", "").removeprefix("models/")
        return ProviderModelInfo(
            id=model_id,
            name=raw.get("displayName", model_id),
            description=raw.get("description"),
            owned_by=None,
        )


def _map_gemini_error_status(status: str, _http_code: int) -> str:
    mapping = {
        "UNAUTHENTICATED": "authentication_error",
        "PERMISSION_DENIED": "permission_error",
        "RESOURCE_EXHAUSTED": "rate_limit_error",
        "INVALID_ARGUMENT": "invalid_request_error",
        "NOT_FOUND": "not_found_error",
    }
    return mapping.get(status, "api_error")


__all__ = ["GeminiAdapter"]
