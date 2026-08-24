"""Chutes provider adapter.

Chutes provides OpenAI-compatible chat completions and model-specific embeddings
and image generation endpoints. This adapter routes:
- Chat completions: https://llm.chutes.ai/v1/chat/completions
- Embeddings (model-specific):
  - Qwen/Qwen3-Embedding-8B -> https://chutes-qwen-qwen3-embedding-8b.chutes.ai/v1/embeddings
  - Qwen/Qwen3-Embedding-0.6B -> https://chutes-qwen-qwen3-embedding-0-6b.chutes.ai/v1/embeddings
- Image generation (model-specific):
  - z-image-turbo: model_in_url pattern -> https://chutes-z-image-turbo.chutes.ai/generate
  - Qwen-Image-2512: model_in_body pattern -> https://image.chutes.ai/generate
"""

# pyright: reportCallIssue=false, reportAttributeAccessIssue=false

import base64
import time
from typing import Any

from llm_proxy.core.adapter import AdapterConfig, register_adapter
from llm_proxy.core.exceptions import ProviderError
from llm_proxy.models import (
    InternalEmbeddingRequest,
    InternalImageRequest,
    InternalImageResponse,
)
from llm_proxy.models.image import ImageData
from llm_proxy.observability.logger import get_logger
from llm_proxy.providers.openai_compatible._base import OpenAICompatibleBase

logger = get_logger(__name__)

_EMBEDDING_ENDPOINTS = {
    "Qwen/Qwen3-Embedding-8B": "https://chutes-qwen-qwen3-embedding-8b.chutes.ai/v1/embeddings",
    "Qwen/Qwen3-Embedding-0.6B": "https://chutes-qwen-qwen3-embedding-0-6b.chutes.ai/v1/embeddings",
}

_IMAGE_MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "z-image-turbo": {
        "url_pattern": "model_in_url",
    },
    "Qwen-Image-2512": {
        "url_pattern": "model_in_body",
        "default_params": {
            "guidance_scale": 7.5,
            "num_inference_steps": 50,
        },
    },
}


@register_adapter("chutes")
class ChutesAdapter(OpenAICompatibleBase):
    """Chutes provider adapter extending OpenAIAdapter.

    This adapter handles Chutes-specific features:
    - Chat completions at https://llm.chutes.ai/v1
    - Model-specific embedding endpoints
    - Model name normalization (strips chutes/ and openai/ prefixes)
    - Base64-encoded embedding response decoding
    """

    _DEFAULT_PROVIDER_NAME = "chutes"

    #: Branding for the admin provider catalog (GET /api/config/provider-types).
    DISPLAY_NAME_EN = "Chutes"
    DISPLAY_NAME_ZH = "Chutes"
    LOBE_ICON_ID = None

    DEFAULT_BASE_URL = "https://llm.chutes.ai/v1"

    def __init__(
        self,
        *,
        config: AdapterConfig | None = None,
        **kwargs: Any,
    ):
        if config is not None:
            super().__init__(config=config)
        else:
            kwargs.setdefault("provider_name", "chutes")
            kwargs.setdefault("base_url", self.DEFAULT_BASE_URL)
            super().__init__(**kwargs)

    def _normalize_model_name(self, model: str) -> str:
        if "/" in model:
            prefix, remainder = model.split("/", 1)
            if prefix.strip().lower() in {"chutes", "openai"}:
                return remainder
        return model

    def _build_chat_raw(self, request: Any, context: Any) -> dict[str, Any]:
        """Normalize model name before building the chat body."""
        body = super()._build_chat_raw(request, context)
        if isinstance(body, dict) and "model" in body:
            body["model"] = self._normalize_model_name(body["model"])
        return body

    def _get_image_model_config(self, model: str) -> dict[str, Any]:
        normalized = self._normalize_model_name(model)
        # Return known config or default to model_in_body pattern
        return _IMAGE_MODEL_CONFIGS.get(normalized, {"url_pattern": "model_in_body"})

    def _resolve_image_url(self, model: str, config: dict[str, Any]) -> str:
        if self._endpoint_base_urls and "image_generation" in self._endpoint_base_urls:
            return self._resolve_endpoint_url("image_generation", "", model=model)

        url_pattern = config.get("url_pattern", "model_in_body")

        if url_pattern == "model_in_url":
            normalized = self._normalize_model_name(model)
            return f"https://chutes-{normalized}.chutes.ai/generate"
        else:
            # model_in_body pattern
            return "https://image.chutes.ai/generate"

    def _build_chutes_image_request_body(
        self, request: InternalImageRequest, config: dict[str, Any]
    ) -> dict[str, Any]:
        """Build raw chutes image body without extra merge (for testing/legacy).

        The dispatch's _finalize_body handles merging request.extra and
        applying the configured field policy.
        """
        body: dict[str, Any] = {
            "prompt": request.prompt,
        }

        url_pattern = config.get("url_pattern", "model_in_body")

        # Only include model for model_in_body pattern
        if url_pattern == "model_in_body" and request.model:
            body["model"] = self._normalize_model_name(request.model)

        # Convert size to width/height
        if request.size:
            body["width"] = request.size.width
            body["height"] = request.size.height
        else:
            body["width"] = 1024
            body["height"] = 1024

        # Apply default params from config
        default_params = config.get("default_params", {})
        for key, value in default_params.items():
            if key not in body:
                body[key] = value

        return body

    def _build_image_raw(self, request: InternalImageRequest) -> dict[str, Any]:
        """Dispatch hook: build raw chutes image body via config-driven builder."""
        model = request.model or ""
        config = self._get_image_model_config(model)
        return self._build_chutes_image_request_body(request, config)

    def _parse_chutes_image_response(self, image_data: bytes, model: str) -> InternalImageResponse:
        b64_image = base64.b64encode(image_data).decode("utf-8")

        return InternalImageResponse(
            created=int(time.time()),
            data=[ImageData(b64_json=b64_image)],
        )

    def _resolve_embeddings_base_url(self, model: str) -> str:
        normalized = self._normalize_model_name(model)
        endpoint = _EMBEDDING_ENDPOINTS.get(normalized)
        if not endpoint:
            raise ProviderError(
                message=(
                    "Chutes embeddings only support models: "
                    f"{', '.join(sorted(_EMBEDDING_ENDPOINTS.keys()))}"
                ),
                error_type="invalid_request_error",
                status_code=400,
                provider_name=self.provider_name,
            )
        if endpoint.endswith("/embeddings"):
            return endpoint.rsplit("/embeddings", 1)[0]
        return endpoint

    def _embeddings_url(self, request: InternalEmbeddingRequest) -> str:
        if self._endpoint_base_urls and "embeddings" in self._endpoint_base_urls:
            return self._resolve_endpoint_url("embeddings", "", model=request.model)

        normalized_model = self._normalize_model_name(request.model)
        base_url = self._resolve_embeddings_base_url(normalized_model)
        return f"{base_url}{self.EMBEDDINGS_ENDPOINT}"

    async def image_generation(
        self, request: InternalImageRequest, **kwargs: Any
    ) -> InternalImageResponse:
        model = request.model or ""
        config = self._get_image_model_config(model)
        url = self._resolve_image_url(model, config)
        headers = self._build_headers()
        outbound = self._build_outbound_body(request, request_type="image_generation")
        if outbound.json_body is None:
            raise ValueError("image_generation request must produce a JSON body")

        image_data = await self._post_raw_with_retry(url, headers, outbound.json_body)
        return self._parse_chutes_image_response(image_data, model)


__all__ = ["ChutesAdapter"]
