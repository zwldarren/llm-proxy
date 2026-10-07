"""Decisions capability mixins.

Three mixins, because the Decisions endpoint and the System One endpoint are two
envelopes over the same evaluation primitives (:mod:`llm_proxy.models.decisions_bridge`)
and an adapter speaks exactly one of them natively:

``DecisionsCapabilityMixin``
    The adapter's upstream serves ``POST /v1/decisions`` (the ``openai``
    adapter). Body building and response parsing go through the provider
    serializer, like every other capability.

``DecisionsOverSystemOneMixin``
    The adapter's upstream serves ``POST /v1/systemone`` instead (``typesafe``,
    ``openrouter``, ``ollama``). ``/v1/decisions`` traffic is converted into a
    System One request and served by the adapter's own System One transport — so
    it inherits that path's retries, usage echo and rate-limit capture — and the
    answers are converted back. A provider whose upstream serves that envelope
    under a Decisions route (OpenRouter's ``/api/alpha/decisions``) overrides
    ``_systemone_decisions_url`` there; otherwise the two endpoints share one
    route.

``SystemOneOverDecisionsMixin``
    The adapter's upstream serves Decisions while a client asked
    ``/v1/systemone`` (the ``openai`` adapter). The mirror of the previous one:
    convert, delegate to ``decisions()``, convert back. Requires
    ``DecisionsCapabilityMixin`` for ``decisions()``, ``DECISIONS_ENDPOINT``
    and ``_decisions_headers()``.

An adapter inherits one of the two bridge mixins, never both: each defines
``decisions()`` (or ``systemone()``) and the last one in the MRO would silently
win.
"""

from typing import Any

from llm_proxy.models.decisions import (
    InternalDecisionRequest,
    InternalDecisionResponse,
)
from llm_proxy.models.decisions_bridge import (
    decisions_to_systemone_request,
    decisions_to_systemone_response,
    systemone_to_decisions_request,
    systemone_to_decisions_response,
)
from llm_proxy.models.systemone import (
    InternalSystemOneRequest,
    InternalSystemOneResponse,
)
from llm_proxy.providers.base import extract_rate_limit_headers
from llm_proxy.providers.capabilities.host import DecisionsSelf, SystemOneSelf


class DecisionsCapabilityMixin:
    """Mixin for provider adapters whose upstream serves Decisions natively."""

    DECISIONS_ENDPOINT: str = "/decisions"

    def _build_decisions_raw(self: DecisionsSelf, request: Any) -> dict[str, Any]:
        """Build the raw Decisions body via the provider serializer."""
        return self._get_serializer().build_provider_decisions_request(request)

    def _decisions_url(self: DecisionsSelf, request: InternalDecisionRequest) -> str:
        return self._resolve_endpoint_url("decisions", self.DECISIONS_ENDPOINT, model=request.model)

    def _decisions_headers(self: DecisionsSelf) -> dict[str, str]:
        return self._build_headers()

    async def decisions(
        self: DecisionsSelf,
        request: InternalDecisionRequest,
        **kwargs: Any,
    ) -> InternalDecisionResponse:
        url = self._decisions_url(request)
        headers = self._decisions_headers()
        outbound = self._build_outbound_body(request, request_type="decisions")
        if outbound.json_body is None:
            raise ValueError("Expected json_body for Decisions request, got None")

        response = await self._post_json_response_with_retry(url, headers, outbound.json_body)
        result = self._get_serializer().parse_provider_decisions_response(
            response.json(), model=request.model
        )
        result.provider_info["_rate_limit_headers"] = extract_rate_limit_headers(
            getattr(response, "headers", None)
        )
        return result


class DecisionsOverSystemOneMixin:
    """Serve ``/v1/decisions`` on an adapter that speaks System One.

    Pair with ``SystemOneCapabilityMixin``: the bridge reuses its transport (body
    building, retries, usage echo, rate-limit capture) and asks it for the
    ``decisions`` route, so a provider serving that envelope under a
    Decisions-named route (OpenRouter's ``/api/alpha/decisions``) declares the route
    on the System One mixin rather than restating the call here.
    """

    async def decisions(
        self: SystemOneSelf,
        request: InternalDecisionRequest,
        **kwargs: Any,
    ) -> InternalDecisionResponse:
        systemone_request = decisions_to_systemone_request(request)
        url = self._systemone_decisions_url(systemone_request)
        response = await self._post_systemone(url, systemone_request, **kwargs)
        return systemone_to_decisions_response(response, request)


class SystemOneOverDecisionsMixin:
    """Serve ``/v1/systemone`` on an adapter that speaks Decisions.

    Pair with ``DecisionsCapabilityMixin``, which supplies ``decisions()``,
    ``DECISIONS_ENDPOINT`` and ``_decisions_headers()``.
    """

    def _systemone_url(self: DecisionsSelf, request: InternalSystemOneRequest) -> str:
        return self._resolve_endpoint_url("decisions", self.DECISIONS_ENDPOINT, model=request.model)

    def _systemone_headers(self: DecisionsSelf) -> dict[str, str]:
        return self._decisions_headers()

    async def systemone(
        self: DecisionsSelf,
        request: InternalSystemOneRequest,
        **kwargs: Any,
    ) -> InternalSystemOneResponse:
        decisions_request = systemone_to_decisions_request(request)
        response = await self.decisions(decisions_request, **kwargs)
        return decisions_to_systemone_response(response, request)


__all__ = [
    "DecisionsCapabilityMixin",
    "DecisionsOverSystemOneMixin",
    "SystemOneOverDecisionsMixin",
]
