"""Image generation and editing capability mixin."""

import time
from collections.abc import AsyncIterator
from typing import Any

import orjson

from llm_proxy.core.exceptions import ProviderError, ValidationError
from llm_proxy.models import InternalImageResponse
from llm_proxy.providers.capabilities.host import ImageSelf


def _image_sse_payloads(chunk: str) -> tuple[list[dict[str, Any]], bool]:
    """Decode JSON payloads and the terminal marker from one SSE chunk."""
    payloads: list[dict[str, Any]] = []
    done = False
    for line in chunk.splitlines():
        line = line.strip()
        if not line.startswith("data: "):
            continue
        raw = line[6:]
        if raw == "[DONE]":
            done = True
            continue
        try:
            value = orjson.loads(raw)
        except orjson.JSONDecodeError:
            continue
        if isinstance(value, dict):
            payloads.append(value)
    return payloads, done


def normalize_image_stream_chunk(
    chunk: str,
    *,
    created_at: int | None = None,
    partial_image_index: int = 0,
) -> list[dict[str, Any]]:
    """Translate provider image SSE payloads into OpenAI Images events.

    OpenAI-shaped events (``image_generation.*`` and ``image_edit.*``) are
    retained, while Gemini's ``candidates`` and ``usageMetadata`` are
    converted to ``partial_image``/``completed`` events so provider-specific
    fields never cross the protocol boundary.
    """
    if not isinstance(chunk, str):
        return []
    payloads, _done = _image_sse_payloads(chunk)
    normalized: list[dict[str, Any]] = []
    timestamp = created_at or int(time.time())
    for payload in payloads:
        event_type = payload.get("type")
        if isinstance(event_type, str) and (
            event_type.startswith("image_generation.") or event_type.startswith("image_edit.")
        ):
            event = dict(payload)
            event["type"] = event_type
            event.setdefault("created_at", timestamp)
            usage = event.get("usage")
            if event_type.endswith(".completed") and isinstance(usage, dict):
                details = usage.get("input_tokens_details")
                if isinstance(details, dict):
                    details = dict(details)
                    details.setdefault("text_tokens", 0)
                    details.setdefault("image_tokens", 0)
                    usage = dict(usage)
                    usage["input_tokens_details"] = details
                    event["usage"] = usage
            normalized.append(event)
            continue

        usage_metadata = payload.get("usageMetadata")
        candidates = payload.get("candidates")
        if not isinstance(candidates, list):
            candidates = []
        last_b64_json: str | None = None
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content")
            parts = content.get("parts", []) if isinstance(content, dict) else []
            if not isinstance(parts, list):
                continue
            for part in parts:
                if not isinstance(part, dict):
                    continue
                inline = part.get("inlineData")
                if not isinstance(inline, dict) or not str(inline.get("mimeType", "")).startswith(
                    "image/"
                ):
                    continue
                last_b64_json = inline.get("data", "")
                normalized.append(
                    {
                        "type": "image_generation.partial_image",
                        "b64_json": last_b64_json,
                        "partial_image_index": partial_image_index,
                        "created_at": timestamp,
                    }
                )
                partial_image_index += 1

        if isinstance(usage_metadata, dict):
            prompt_tokens = usage_metadata.get("promptTokenCount", 0) or 0
            output_tokens = usage_metadata.get("candidatesTokenCount", 0) or 0
            total_tokens = usage_metadata.get("totalTokenCount")
            completed: dict[str, Any] = {
                "type": "image_generation.completed",
                "created_at": timestamp,
                "usage": {
                    "input_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens
                    if total_tokens is not None
                    else prompt_tokens + output_tokens,
                    "input_tokens_details": {
                        "text_tokens": prompt_tokens,
                        "image_tokens": 0,
                    },
                },
            }
            if last_b64_json is not None:
                completed["b64_json"] = last_b64_json
            normalized.append(completed)
    return normalized


class ImageCapabilityMixin:
    """Mixin for provider adapters that support image generation and editing.

    Adapters include this mixin when they speak the OpenAI images wire format
    (OpenAI and the openai-compatible family — Chutes, DeepSeek, NanoGPT,
    OpenRouter — via ``OpenAICompatibleBase``).
    """

    IMAGES_ENDPOINT: str = ""
    IMAGES_EDITS_ENDPOINT: str = ""

    async def image_generation(
        self: ImageSelf,
        request,
        **kwargs: Any,
    ) -> InternalImageResponse:
        raise NotImplementedError(
            f"{self.provider_name} does not have a native image generation API. "
            "Configure a separate image generation provider."
        )

    async def stream_image_generation(
        self: ImageSelf,
        request,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        return self._stream_image_request(
            request,
            url=self._image_generation_url(model=getattr(request, "model", None)),
            request_type="image_generation",
        )

    def _stream_image_request(
        self: ImageSelf,
        request,
        *,
        url: str,
        request_type: str,
    ) -> AsyncIterator[str]:
        """Shared streaming flow for image generation/edit progress events."""
        event_prefix = "image_edit" if request_type == "image_edit" else "image_generation"
        headers = self._build_headers()
        outbound = self._build_outbound_body(request, request_type=request_type)
        body = outbound.json_body
        data = outbound.form_data
        files = outbound.files
        if data is not None:
            headers.pop("Content-Type", None)
        stream_timeout = self._get_stream_timeout()
        created_at = int(time.time())
        completed = False
        partial_image_index = 0
        last_b64_json: str | None = None

        async def _generator():
            nonlocal completed, partial_image_index, last_b64_json
            client = await self._get_client()
            try:
                async with self._streaming_post(
                    client,
                    url,
                    headers=headers,
                    json=body,
                    data=data,
                    files=files,
                    timeout=stream_timeout,
                ) as response:
                    await self._raise_for_stream_status(response)
                    async for line_str in self._iter_stream_lines(response):
                        normalized = normalize_image_stream_chunk(
                            line_str,
                            created_at=created_at,
                            partial_image_index=partial_image_index,
                        )
                        partial_image_index += sum(
                            item.get("type", "").endswith(".partial_image") for item in normalized
                        )
                        for event in normalized:
                            if event.get("type") == f"{event_prefix}.completed":
                                completed = True
                            if event.get("type", "").endswith(".partial_image"):
                                last_b64_json = event.get("b64_json")
                            event_type = event["type"]
                            yield (
                                f"event: {event_type}\n"
                                f"data: {orjson.dumps(event).decode('utf-8')}\n\n"
                            )
                        if "data: [DONE]" in line_str:
                            if not completed:
                                completed = True
                                event: dict[str, Any] = {
                                    "type": f"{event_prefix}.completed",
                                    "created_at": created_at,
                                }
                                if last_b64_json is not None:
                                    event["b64_json"] = last_b64_json
                                yield (
                                    f"event: {event_prefix}.completed\n"
                                    f"data: {orjson.dumps(event).decode('utf-8')}\n\n"
                                )
                            yield "data: [DONE]\n\n"
            except ProviderError:
                raise
            except Exception as e:
                error = await self._handle_http_error(e)
                raise error from e

        return self._with_retry_generator(_generator)

    async def image_edit(
        self: ImageSelf,
        request,
        **kwargs: Any,
    ) -> InternalImageResponse:
        raise NotImplementedError(
            f"{self.provider_name} does not support image editing. "
            "Configure a separate image editing provider."
        )

    async def stream_image_edit(
        self: ImageSelf,
        request,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream image edit progress events from the edits endpoint.

        Uses the image edit URL and request body so streaming edit requests
        are not misrouted to the image generation endpoint.
        """
        return self._stream_image_request(
            request,
            url=self._image_edit_url(model=getattr(request, "model", None)),
            request_type="image_edit",
        )

    def _image_generation_url(self: ImageSelf, model: str | None = None) -> str:
        return self._resolve_endpoint_url("image_generation", self.IMAGES_ENDPOINT, model=model)

    def _image_edit_url(self: ImageSelf, model: str | None = None) -> str:
        return self._resolve_endpoint_url("image_edit", self.IMAGES_EDITS_ENDPOINT, model=model)

    def _build_image_request_body(self: ImageSelf, request: Any) -> dict[str, Any]:
        """Build image generation body without extra merge or field policy.

        The dispatch's _finalize_body handles merging request.extra and
        applying the configured field policy.
        """
        body: dict[str, Any] = {
            "prompt": request.prompt,
        }

        model = request.model
        is_gpt_image_model = model is not None and model.startswith("gpt-image-")

        if model:
            body["model"] = model

        if request.n is not None:
            body["n"] = request.n

        if request.quality:
            body["quality"] = request.quality

        response_format = getattr(request, "response_format", None)
        if response_format and not is_gpt_image_model:
            body["response_format"] = response_format

        if request.size:
            body["size"] = f"{request.size.width}x{request.size.height}"
        elif getattr(request, "size_auto", False):
            body["size"] = "auto"

        if request.style and not is_gpt_image_model:
            body["style"] = request.style

        if request.user:
            body["user"] = request.user

        if is_gpt_image_model:
            if request.background:
                body["background"] = request.background
            if request.moderation:
                body["moderation"] = request.moderation
            if request.output_compression is not None:
                body["output_compression"] = request.output_compression
            if request.output_format:
                body["output_format"] = request.output_format
            if request.partial_images is not None:
                body["partial_images"] = request.partial_images

        if request.stream:
            body["stream"] = request.stream

        return body

    def _build_image_raw(self: ImageSelf, request: Any) -> dict[str, Any]:
        """Build raw image generation body."""
        return self._build_image_request_body(request)

    def _build_image_edit_raw(self: ImageSelf, request: Any) -> tuple[dict[str, Any], Any]:
        """Build raw image edit JSON body without extra merge or field policy.

        The dispatch's _finalize_body handles merging request.extra and
        applying the configured field policy.
        """
        body: dict[str, Any] = {"prompt": request.prompt}
        files: list[tuple[str, tuple[str, bytes | bytearray, str]]] = []
        uploaded_images = [src for src in request.images if src.file is not None]
        if uploaded_images:
            if len(uploaded_images) != len(request.images):
                raise ValidationError(
                    message="Multipart image edits require every image to be an uploaded file",
                    code="invalid_request_error",
                    status_code=400,
                )
            for src in uploaded_images:
                files.append(
                    (
                        "image[]",
                        (
                            src.filename or "image.png",
                            src.file,
                            src.content_type or "image/png",
                        ),
                    )
                )
        else:
            body["images"] = [
                {
                    k: v
                    for k, v in {"file_id": src.file_id, "image_url": src.image_url}.items()
                    if v is not None
                }
                for src in request.images
            ]
        if request.model:
            body["model"] = request.model
        if request.background:
            body["background"] = request.background
        if request.input_fidelity:
            body["input_fidelity"] = request.input_fidelity
        if request.moderation:
            body["moderation"] = request.moderation
        if request.n != 1:
            body["n"] = request.n
        if getattr(request, "response_format", None):
            body["response_format"] = request.response_format
        if request.mask:
            if request.mask.file is not None:
                files.append(
                    (
                        "mask",
                        (
                            request.mask.filename or "mask.png",
                            request.mask.file,
                            request.mask.content_type or "image/png",
                        ),
                    )
                )
            else:
                mask_fields = {
                    "file_id": request.mask.file_id,
                    "image_url": request.mask.image_url,
                }
                body["mask"] = {k: v for k, v in mask_fields.items() if v is not None}
        if request.output_compression is not None:
            body["output_compression"] = request.output_compression
        if request.output_format:
            body["output_format"] = request.output_format
        if request.partial_images is not None:
            body["partial_images"] = request.partial_images
        if request.quality:
            body["quality"] = request.quality
        if request.size:
            body["size"] = f"{request.size.width}x{request.size.height}"
        elif getattr(request, "size_auto", False):
            body["size"] = "auto"
        if request.user:
            body["user"] = request.user
        if request.stream:
            body["stream"] = request.stream
        return body, files


__all__ = ["ImageCapabilityMixin", "normalize_image_stream_chunk"]
