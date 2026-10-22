"""Rerank capability mixin."""

from typing import Any

from llm_proxy.models.rerank import InternalRerankRequest, InternalRerankResponse
from llm_proxy.providers.base import extract_rate_limit_headers
from llm_proxy.providers.capabilities.host import RerankSelf


class RerankCapabilityMixin:
    """Mixin for provider adapters that support rerank requests.

    The wire shape is Cohere's ``/v1/rerank``, which OpenRouter, Jina and the
    vLLM/SGLang ``/rerank`` route also serve. Adapters using this mixin:
    the ``openai-compatible`` family (including OpenRouter).
    """

    RERANK_ENDPOINT: str = ""

    def _build_rerank_raw(self: RerankSelf, request: Any) -> dict[str, Any]:
        """Build raw rerank body."""
        return self._get_serializer().build_provider_rerank_request(request)

    def _rerank_url(self: RerankSelf, request: InternalRerankRequest) -> str:
        if self.RERANK_ENDPOINT:
            return self._resolve_endpoint_url("rerank", self.RERANK_ENDPOINT, model=request.model)
        raise NotImplementedError("Subclasses must define RERANK_ENDPOINT or override _rerank_url")

    def _rerank_headers(self: RerankSelf) -> dict[str, str]:
        return self._build_headers()

    async def rerank(
        self: RerankSelf,
        request: InternalRerankRequest,
        **kwargs: Any,
    ) -> InternalRerankResponse:
        url = self._rerank_url(request)
        headers = self._rerank_headers()
        outbound = self._build_outbound_body(request, request_type="rerank")
        if outbound.json_body is None:
            raise ValueError("Expected json_body for rerank request, got None")
        response = await self._post_json_response_with_retry(url, headers, outbound.json_body)
        result = self._get_serializer().parse_provider_rerank_response(
            response.json(), model=request.model
        )
        if request.return_documents is False:
            # A response projection, not an upstream field: the client asked for
            # ranked indexes and scores only, so the echoed documents are
            # dropped before the protocol formatter sees them.
            for ranked in result.results:
                ranked.document = None
        result.provider_info["_rate_limit_headers"] = extract_rate_limit_headers(
            getattr(response, "headers", None)
        )
        return result


__all__ = ["RerankCapabilityMixin"]
