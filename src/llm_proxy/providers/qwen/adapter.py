"""Qwen provider adapter — Alibaba Cloud Model Studio (DashScope), China.

Alibaba Cloud Model Studio (help.aliyun.com/zh/model-studio/base-url) serves
Qwen models from regional DashScope domains with three wire protocols:

- Chat Completions: ``{base_url}/chat/completions`` — OpenAI-compatible
  (``https://dashscope.aliyuncs.com/compatible-mode/v1``, Beijing)
- Anthropic Messages: ``{host}/apps/anthropic/v1/messages`` — the Anthropic
  endpoint lives on the site root (``https://dashscope.aliyuncs.com/apps/anthropic``),
  documented with ``ANTHROPIC_AUTH_TOKEN`` (Bearer) auth
- OpenAI Responses: ``{base_url}/responses`` — the Responses endpoint rides
  the OpenAI-compatible base itself
  (``https://dashscope.aliyuncs.com/compatible-mode/v1/responses``; the older
  ``/api/v2/apps/protocols/compatible-mode/v1/responses`` path is deprecated,
  see help.aliyun.com/zh/model-studio/compatibility-with-openai-responses-api)

The Anthropic URL derives from the site root (the ``/compatible-mode/v1``
compatibility alias is stripped), so business-space dedicated domains
(``{workspace}.cn-beijing.maas.aliyuncs.com``) and other regions work through
a ``base_url`` override. International (Singapore) keys should use the
``qwen-intl`` provider type instead.

Text embeddings (``text-embedding-v1..v4``, ``qwen3.7-text-embedding``) ride
the OpenAI-compatible ``{base_url}/embeddings`` endpoint
(help.aliyun.com/zh/model-studio/text-embedding-synchronous-api).

Image generation/editing (wanx, e.g. ``wan2.7-image-pro``) is **not** served
through the OpenAI compatibility mode — the official docs state image models
are DashScope-native only
(help.aliyun.com/zh/model-studio/wan-image-generation-and-editing-api-reference).
This adapter implements the native async task flow: submit
``POST {root}/api/v1/services/aigc/image-generation/generation`` with the
``X-DashScope-Async: enable`` header, then poll
``GET {root}/api/v1/tasks/{task_id}`` until the task reaches a terminal
status. The task data and generated image URLs expire after 24 hours.
"""

import asyncio
import base64
import time
from collections.abc import AsyncIterator
from typing import Any

import orjson

from llm_proxy.core.adapter import register_adapter
from llm_proxy.core.exceptions import ProviderError, ValidationError
from llm_proxy.models import (
    InternalImageEditRequest,
    InternalImageRequest,
    InternalImageResponse,
)
from llm_proxy.models.image import ImageData
from llm_proxy.models.types import Usage
from llm_proxy.providers.openai_compatible._native import NativePassthroughChatBase

#: DashScope async task statuses that never transition again.
_TERMINAL_TASK_STATUSES: frozenset[str] = frozenset({"SUCCEEDED", "FAILED", "CANCELED", "UNKNOWN"})


def _task_output(body: dict[str, Any]) -> dict[str, Any]:
    """The wanx task ``output`` block, normalized to a dict.

    The block is absent or None until the task completes, so callers can
    always ``.get()`` on the result.
    """
    return body.get("output") or {}


@register_adapter("qwen")
class QwenAdapter(NativePassthroughChatBase):
    _DEFAULT_PROVIDER_NAME = "qwen"

    #: Branding for the admin provider catalog (GET /api/config/provider-types).
    DISPLAY_NAME_EN = "Qwen (Model Studio)"
    DISPLAY_NAME_ZH = "通义千问 (百炼)"
    LOBE_ICON_ID = "qwen"
    LOBE_ICON_VARIANT = "color"

    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    native_protocols = frozenset({"anthropic", "openresponses"})

    #: Anthropic Messages path appended to the site root (see
    #: ``_native_root_base_url``): ``{host}/apps/anthropic/v1/messages``.
    ANTHROPIC_MESSAGES_PATH = "/apps/anthropic/v1/messages"

    #: Wanx image task endpoints, hung off the site root (the compatible-mode
    #: alias carries no image service). ``{root}/api/v1/services/...``.
    IMAGE_GENERATION_PATH = "/api/v1/services/aigc/image-generation/generation"
    IMAGE_TASKS_PATH = "/api/v1/tasks"
    #: Seconds between task-status polls.
    IMAGE_TASK_POLL_INTERVAL: float = 1.0
    #: Poll cap before the task is considered hung (~10 min at 1 s).
    IMAGE_TASK_MAX_POLLS: int = 600

    #: Square pixel sizes map to the recommended DashScope 1K/2K/4K specs.
    #: Only wanx models accept these specs; qwen-image models require the
    #: explicit ``{width}*{height}`` pixel form (see ``_dashscope_image_size``).
    _SQUARE_SIZE_SPECS: dict[int, str] = {1024: "1K", 2048: "2K", 4096: "4K"}

    def _native_root_base_url(self) -> str:
        """Site root hosting the Anthropic endpoint.

        The configured base_url carries the ``/compatible-mode/v1``
        compatibility alias (or the native ``/api/v1``); the Anthropic
        endpoint hangs off the bare host, so the alias is stripped.
        """
        base = self._base_url or self.DEFAULT_BASE_URL
        for suffix in ("/compatible-mode/v1", "/api/v1"):
            if base.endswith(suffix):
                return base[: -len(suffix)]
        return base

    def _responses_url(self, model: str | None = None) -> str:
        """Responses endpoint on the OpenAI-compatible base (``{base_url}/responses``).

        Unlike the Anthropic endpoint (site root), the Responses endpoint
        keeps the ``/compatible-mode/v1`` alias (the older
        ``/api/v2/apps/protocols/compatible-mode/v1/responses`` path is
        deprecated per the docs), so the base URL is used as-is instead of
        going through ``_native_root_base_url``.
        """
        if "responses" in self._endpoint_base_urls:
            return self._resolve_endpoint_url("responses", "", model=model)
        return f"{self._base_url or self.DEFAULT_BASE_URL}/responses"

    # ------------------------------------------------------------------
    # Image generation / editing (native wanx async task flow)
    # ------------------------------------------------------------------

    def _image_generation_url(self, model: str | None = None) -> str:
        """Submit URL for the native wanx image task endpoint."""
        if "image_generation" in self._endpoint_base_urls:
            return self._resolve_endpoint_url("image_generation", "", model=model)
        return f"{self._native_root_base_url()}{self.IMAGE_GENERATION_PATH}"

    def _image_task_url(self, task_id: str) -> str:
        """Poll URL for a submitted wanx image task."""
        if "image_task" in self._endpoint_base_urls:
            url = self._endpoint_base_urls["image_task"].rstrip("/")
            return url.replace("{task_id}", task_id)
        return f"{self._native_root_base_url()}{self.IMAGE_TASKS_PATH}/{task_id}"

    def _dashscope_image_size(
        self, request: InternalImageRequest | InternalImageEditRequest
    ) -> str | None:
        """Map the OpenAI ``{w}x{h}`` size onto a DashScope size spec.

        Square 1K/2K/4K requests map to the recommended ``1K|2K|4K`` specs
        for wanx models only; qwen-image models accept only the explicit
        ``{width}*{height}`` pixel form (the ``1K`` spec is rejected with
        ``InvalidParameter: Expected format: '<width>*<height>'``). Any other
        size is passed as ``{width}*{height}`` pixels. ``auto`` / no size
        lets the provider default (2K) apply.
        """
        if request.size_auto or request.size is None:
            return None
        size = request.size
        if size.width == size.height and request.model.startswith("wan"):
            spec = self._SQUARE_SIZE_SPECS.get(size.width)
            if spec is not None:
                return spec
        return f"{size.width}*{size.height}"

    def _build_dashscope_image_body(
        self,
        model: str,
        prompt: str,
        *,
        images: list[str] | None = None,
        request: InternalImageRequest | InternalImageEditRequest,
    ) -> dict[str, Any]:
        """Build the wanx generation body (single-turn messages shape).

        ``request.extra`` keys (``seed``, ``watermark``, ``thinking_mode``,
        ``color_palette``, ``bbox_list``, ``enable_sequential``, ...) are
        merged into ``parameters`` — the wanx parameter namespace.
        """
        content: list[dict[str, Any]] = []
        if images:
            content.extend({"image": url} for url in images)
        content.append({"text": prompt})
        parameters: dict[str, Any] = {}
        n = getattr(request, "n", 1)
        if n and n > 1:
            parameters["n"] = n
        size = self._dashscope_image_size(request)
        if size:
            parameters["size"] = size
        parameters.update(request.extra)
        body: dict[str, Any] = {
            "model": model,
            "input": {"messages": [{"role": "user", "content": content}]},
        }
        if parameters:
            body["parameters"] = parameters
        return body

    async def _submit_image_task(self, body: dict[str, Any]) -> str:
        """Submit an async wanx task; return the ``task_id``."""
        headers = self._build_headers()
        headers["X-DashScope-Async"] = "enable"
        response = await self._post_json_response_with_retry(
            self._image_generation_url(model=body.get("model")), headers, body
        )
        output = _task_output(response.json())
        task_id = output.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ProviderError(
                message=f"{self.provider_name} image task submission returned no task_id",
                error_type="api_error",
                status_code=502,
                provider_name=self.provider_name,
            )
        return task_id

    async def _get_json(self, url: str, headers: dict[str, str]) -> dict[str, Any]:
        """GET a JSON resource with the standard retry/error handling."""

        async def _make_request():
            client = await self._get_client()
            response = await client.get(url, headers=headers)
            await self._check_response_status(response)
            return response.json()

        return await self._with_retry(_make_request)

    async def _poll_image_task(self, task_id: str) -> dict[str, Any]:
        """Poll until the task succeeds; raise ProviderError on failure/timeout."""
        url = self._image_task_url(task_id)
        headers = self._build_headers()
        for _ in range(self.IMAGE_TASK_MAX_POLLS):
            await asyncio.sleep(self.IMAGE_TASK_POLL_INTERVAL)
            body = await self._get_json(url, headers)
            status = _task_output(body).get("task_status")
            if status == "SUCCEEDED":
                return body
            if status in _TERMINAL_TASK_STATUSES:
                raise ProviderError(
                    message=(
                        f"{self.provider_name} image task {status.lower()}: "
                        f"{body.get('message') or 'see upstream error'}"
                    ),
                    code=body.get("code"),
                    error_type="api_error",
                    status_code=502,
                    provider_name=self.provider_name,
                )
        raise ProviderError(
            message=f"{self.provider_name} image task {task_id} did not finish in time",
            error_type="api_error",
            status_code=504,
            provider_name=self.provider_name,
        )

    def _parse_image_task_result(self, body: dict[str, Any], model: str) -> InternalImageResponse:
        """Parse the succeeded task body into the unified image response."""
        output = _task_output(body)
        images: list[ImageData] = []
        for choice in output.get("choices") or []:
            content = (choice.get("message") or {}).get("content") or []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("image"), str):
                    images.append(ImageData(url=item["image"]))
        if not images:
            raise ProviderError(
                message=f"{self.provider_name} image task succeeded without image results",
                error_type="api_error",
                status_code=502,
                provider_name=self.provider_name,
            )
        usage = None
        raw_usage = body.get("usage")
        if isinstance(raw_usage, dict):
            usage = Usage(
                input_tokens=raw_usage.get("input_tokens", 0),
                output_tokens=raw_usage.get("output_tokens", 0),
                total_tokens=raw_usage.get("total_tokens"),
            )
        result = InternalImageResponse(
            created=int(time.time()),
            data=images,
            model=model,
            usage=usage,
            request_id=body.get("request_id"),
        )
        # Stash the raw task body for debugging. Deliberately NOT under
        # ``_raw_response_body``: that key is the native-passthrough signal
        # that makes the protocol formatter emit the body verbatim, which
        # would bypass the OpenAI Images serialization of this response.
        result.provider_info["_dashscope_task_body"] = body
        return result

    async def image_generation(
        self, request: InternalImageRequest, **kwargs: Any
    ) -> InternalImageResponse:
        body = self._build_dashscope_image_body(request.model, request.prompt, request=request)
        task_id = await self._submit_image_task(body)
        return self._parse_image_task_result(await self._poll_image_task(task_id), request.model)

    def _dashscope_edit_images(self, request: InternalImageEditRequest) -> list[str]:
        """Map reference images to wanx ``image`` content entries.

        URLs pass through; uploaded files become ``data:`` URLs. File-API ids
        are provider-local and cannot be referenced from DashScope — reject
        them with a clear error.
        """
        if request.mask is not None and (
            request.mask.file is not None
            or request.mask.image_url is not None
            or request.mask.file_id is not None
        ):
            raise ValidationError(
                message=(
                    "Qwen (wanx) image editing has no mask concept — pass "
                    "bbox_list coordinates via extra parameters instead"
                ),
                code="invalid_request_error",
                status_code=400,
            )
        sources: list[str] = []
        for src in request.images:
            if src.image_url:
                sources.append(src.image_url)
            elif src.file is not None:
                mime = src.content_type or "image/png"
                encoded = base64.b64encode(bytes(src.file)).decode("ascii")
                sources.append(f"data:{mime};base64,{encoded}")
            elif src.file_id:
                raise ValidationError(
                    message=(
                        "Qwen (wanx) image editing needs a public image URL or "
                        "an uploaded file, not a File API id"
                    ),
                    code="invalid_request_error",
                    status_code=400,
                )
            else:
                raise ValidationError(
                    message="Image edit references contain no image data",
                    code="invalid_request_error",
                    status_code=400,
                )
        if not sources:
            raise ValidationError(
                message="Image editing requires at least one reference image",
                code="invalid_request_error",
                status_code=400,
            )
        return sources

    async def image_edit(
        self, request: InternalImageEditRequest, **kwargs: Any
    ) -> InternalImageResponse:
        images = self._dashscope_edit_images(request)
        body = self._build_dashscope_image_body(
            request.model, request.prompt, images=images, request=request
        )
        task_id = await self._submit_image_task(body)
        return self._parse_image_task_result(await self._poll_image_task(task_id), request.model)

    def _image_task_stream_events(
        self, result: InternalImageResponse, *, event_prefix: str
    ) -> AsyncIterator[str]:
        """Emit one ``{prefix}.completed`` SSE event per generated image."""

        async def _generator():
            event_type = f"{event_prefix}.completed"
            for img in result.data:
                event: dict[str, Any] = {
                    "type": event_type,
                    "created_at": result.created,
                }
                if img.url is not None:
                    event["url"] = img.url
                if img.b64_json is not None:
                    event["b64_json"] = img.b64_json
                if result.usage is not None:
                    event["usage"] = {
                        "input_tokens": result.usage.input_tokens,
                        "output_tokens": result.usage.output_tokens,
                        "total_tokens": result.usage.total_tokens,
                    }
                yield f"event: {event_type}\n" + (
                    f"data: {orjson.dumps(event).decode('utf-8')}\n\n"
                )
            yield "data: [DONE]\n\n"

        return _generator()

    async def stream_image_generation(
        self, request: InternalImageRequest, **kwargs: Any
    ) -> AsyncIterator[str]:
        result = await self.image_generation(request, **kwargs)
        return self._image_task_stream_events(result, event_prefix="image_generation")

    async def stream_image_edit(
        self, request: InternalImageEditRequest, **kwargs: Any
    ) -> AsyncIterator[str]:
        result = await self.image_edit(request, **kwargs)
        return self._image_task_stream_events(result, event_prefix="image_edit")


__all__ = ["QwenAdapter"]
