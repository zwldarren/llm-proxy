"""System One capability mixin."""

from typing import Any

from llm_proxy.models.systemone import (
    InternalSystemOneRequest,
    InternalSystemOneResponse,
    SystemOneImageChannel,
)
from llm_proxy.providers.base import extract_rate_limit_headers
from llm_proxy.providers.capabilities.host import SystemOneSelf


class SystemOneCapabilityMixin:
    """Mixin for provider adapters that support System One evaluation.

    Adapters using this mixin: TypeSafe (direct), OpenRouter and Ollama
    (local models, v0.35+). All upstreams share the same ``/systemone`` wire
    format, so the endpoint URL, auth headers and body/response conversion are
    identical — only the base URL, the image channel, the optional-field
    exemptions and the extra response fields differ, and each adapter supplies
    those through its own config and serializer. An upstream that also serves
    the format on a Decisions-named route points ``_systemone_decisions_url`` at
    it for the bridged ``/v1/decisions`` direction.
    """

    SYSTEMONE_ENDPOINT: str = "/systemone"

    #: How this upstream carries images (see
    #: :data:`~llm_proxy.models.systemone.SystemOneImageChannel`). The wire
    #: format has no image field of its own, so a bridged Decisions request's
    #: pictures reach the upstream through the channel it documents — ``images``
    #: for Ollama, ``state_parts`` for OpenRouter, ``assets`` for TypeSafe — and
    #: an adapter that documents none leaves it unset, which sends the text
    #: alone.
    SYSTEMONE_IMAGE_CHANNEL: SystemOneImageChannel | None = None

    def _build_systemone_raw(self: SystemOneSelf, request: Any) -> dict[str, Any]:
        """Build the raw System One body via the provider serializer."""
        return self._get_serializer().build_provider_systemone_request(request)

    def _systemone_url(self: SystemOneSelf, request: InternalSystemOneRequest) -> str:
        return self._resolve_endpoint_url("systemone", self.SYSTEMONE_ENDPOINT, model=request.model)

    def _systemone_decisions_url(self: SystemOneSelf, request: InternalSystemOneRequest) -> str:
        """URL a bridged ``/v1/decisions`` request reaches.

        The same route as System One by default — one upstream, one route, and
        the only place a provider has to change to reach a second one. An
        upstream that serves the envelope on a Decisions-named route overrides
        this: OpenRouter points it at ``/api/alpha/decisions``, resolved through
        its own ``endpoint_base_urls.decisions`` key so that override does not
        move the System One route. Providers without a second route inherit
        this default, so their ``/v1/decisions`` follows the
        ``endpoint_base_urls.systemone`` override like System One does.
        """
        return self._systemone_url(request)

    def _systemone_headers(
        self: SystemOneSelf, request: InternalSystemOneRequest
    ) -> dict[str, str]:
        """Upstream headers for one System One request.

        Takes the request so an adapter can add a header the body makes necessary
        (TypeSafe's danger-zone opt-in for images and box questions) without
        rebuilding the body to find out.
        """
        return self._build_headers()

    async def _post_systemone(
        self: SystemOneSelf,
        url: str,
        request: InternalSystemOneRequest,
        **kwargs: Any,
    ) -> InternalSystemOneResponse:
        """POST an already-built System One body to ``url`` and parse the answers.

        Separate from ``systemone()`` so a direction that reaches the same
        envelope by another route (``DecisionsOverSystemOneMixin``) reuses the retry
        policy, usage echo and rate-limit capture instead of restating them.
        """
        headers = self._systemone_headers(request)
        outbound = self._build_outbound_body(request, request_type="systemone")
        if outbound.json_body is None:
            raise ValueError("Expected json_body for System One request, got None")

        response = await self._post_json_response_with_retry(url, headers, outbound.json_body)
        result = self._get_serializer().parse_provider_systemone_response(
            response.json(), model=request.model
        )
        result.provider_info["_rate_limit_headers"] = extract_rate_limit_headers(
            getattr(response, "headers", None)
        )
        return result

    async def systemone(
        self: SystemOneSelf,
        request: InternalSystemOneRequest,
        **kwargs: Any,
    ) -> InternalSystemOneResponse:
        return await self._post_systemone(self._systemone_url(request), request, **kwargs)


__all__ = ["SystemOneCapabilityMixin"]
