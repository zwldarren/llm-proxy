"""Embedding capability mixin."""

from typing import Any

from llm_proxy.models import InternalEmbeddingRequest, InternalEmbeddingResponse
from llm_proxy.providers.base import extract_rate_limit_headers
from llm_proxy.providers.capabilities.host import EmbeddingSelf


class EmbeddingCapabilityMixin:
    """Mixin for provider adapters that support embeddings.

    Adapters using this mixin: OpenAI, Gemini, Ollama, Chutes.
    """

    EMBEDDINGS_ENDPOINT: str = ""

    def _build_embedding_raw(self: EmbeddingSelf, request: Any) -> dict[str, Any]:
        """Build raw embedding body."""
        return self._get_serializer().build_provider_embedding_request(request)

    def _embeddings_url(self: EmbeddingSelf, request: InternalEmbeddingRequest) -> str:
        if self.EMBEDDINGS_ENDPOINT:
            return self._resolve_endpoint_url(
                "embeddings", self.EMBEDDINGS_ENDPOINT, model=request.model
            )
        raise NotImplementedError(
            "Subclasses must define EMBEDDINGS_ENDPOINT or override _embeddings_url"
        )

    def _embeddings_headers(self: EmbeddingSelf) -> dict[str, str]:
        return self._build_headers()

    async def embeddings(
        self: EmbeddingSelf,
        request: InternalEmbeddingRequest,
        **kwargs: Any,
    ) -> InternalEmbeddingResponse:
        url = self._embeddings_url(request)
        headers = self._embeddings_headers()
        outbound = self._build_outbound_body(request, request_type="embedding")
        if outbound.json_body is None:
            raise ValueError("Expected json_body for embedding request, got None")
        response = await self._post_json_response_with_retry(url, headers, outbound.json_body)
        result = self._get_serializer().parse_provider_embedding_response(
            response.json(), model=request.model
        )
        result.provider_info["_rate_limit_headers"] = extract_rate_limit_headers(
            getattr(response, "headers", None)
        )
        return result


__all__ = ["EmbeddingCapabilityMixin"]
