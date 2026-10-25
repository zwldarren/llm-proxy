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
    ``openrouter``, ``ollama``, ``opencode``). ``/v1/decisions`` traffic is
    converted into a System One request and served by the adapter's own System
    One transport — so it inherits that path's retries, usage echo and
    rate-limit capture — and the answers are converted back, with images routed
    through the adapter's ``SYSTEMONE_IMAGE_CHANNEL``. A provider whose upstream
    serves that envelope under a Decisions route (OpenRouter's
    ``/api/alpha/decisions``) overrides ``_systemone_decisions_url`` there;
    otherwise the two endpoints share one route.

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

import dataclasses
from collections import Counter
from typing import Any, disjoint_base

from llm_proxy.core.exceptions import ValidationError
from llm_proxy.http.media import RequestDownloads
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


@disjoint_base
class DecisionsOverSystemOneMixin:
    """Mixin for adapters whose upstream serves System One, not Decisions.

    Declared a disjoint base (PEP 800): an adapter inherits exactly one of the
    two bridge mixins, and combining this one with
    :class:`SystemOneOverDecisionsMixin` would silently shadow ``decisions()``
    / ``systemone()`` by MRO order. The type checker now rejects that shape
    instead of leaving it to this docstring.
    """

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
        request = await _inline_hosted_images(self, request)
        systemone_request = decisions_to_systemone_request(
            request, image_channel=self.SYSTEMONE_IMAGE_CHANNEL
        )
        url = self._systemone_decisions_url(systemone_request)
        response = await self._post_systemone(url, systemone_request, **kwargs)
        return systemone_to_decisions_response(response, request)


@disjoint_base
class SystemOneOverDecisionsMixin:
    """Mixin for adapters that serve System One traffic over a Decisions upstream.

    Disjoint with :class:`DecisionsOverSystemOneMixin` for the reason given
    there: the two bridges define the same methods in opposite directions.
    """

    """Serve ``/v1/systemone`` on an adapter that speaks Decisions.

    Pair with ``DecisionsCapabilityMixin``, which supplies ``decisions()``,
    ``DECISIONS_ENDPOINT`` and ``_decisions_headers()``.
    """

    def _systemone_url(self: DecisionsSelf, request: InternalSystemOneRequest) -> str:
        return self._resolve_endpoint_url("decisions", self.DECISIONS_ENDPOINT, model=request.model)

    def _systemone_headers(
        self: DecisionsSelf, request: InternalSystemOneRequest
    ) -> dict[str, str]:
        return self._decisions_headers()

    async def systemone(
        self: DecisionsSelf,
        request: InternalSystemOneRequest,
        **kwargs: Any,
    ) -> InternalSystemOneResponse:
        decisions_request = systemone_to_decisions_request(request)
        response = await self.decisions(decisions_request, **kwargs)
        return decisions_to_systemone_response(response, request)


def _hosted_image_urls(value: Any) -> list[str]:
    """Every distinct public HTTP(S) image URL a request payload refers to."""
    return list(dict.fromkeys(_hosted_image_occurrences(value)))


def _hosted_image_occurrences(value: Any) -> list[str]:
    """Every public HTTP(S) image URL, one entry per reference.

    The budget bounds what the request *retains*, and a URL named twice is
    inlined twice, so the caller needs the count as well as the distinct set.
    """
    urls: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, dict):
            if node.get("type") == "input_json":
                # JSON evidence is text the model reads, not an image part: an
                # ``input_image``-shaped object inside it is data, not a
                # reference the proxy resolves or rewrites.
                return
            url = node.get("image_url")
            if (
                node.get("type") == "input_image"
                and isinstance(url, str)
                and url.startswith(("http://", "https://"))
            ):
                urls.append(url)
            for item in node.values():
                walk(item)

    walk(value)
    return urls


def _replace_image_urls(value: Any, replacements: dict[str, str]) -> Any:
    """Return ``value`` with each hosted image URL replaced by its data URL."""
    if isinstance(value, list):
        return [_replace_image_urls(item, replacements) for item in value]
    if not isinstance(value, dict):
        return value
    if value.get("type") == "input_json":
        # JSON evidence travels verbatim: only an ``input_image`` part names an
        # image the request carries.
        return value
    rewritten = {key: _replace_image_urls(item, replacements) for key, item in value.items()}
    if rewritten.get("type") == "input_image" and rewritten.get("image_url") in replacements:
        rewritten["image_url"] = replacements[rewritten["image_url"]]
    return rewritten


async def _inline_hosted_images(
    host: SystemOneSelf, request: InternalDecisionRequest
) -> InternalDecisionRequest:
    """Download a request's hosted image URLs so the bridge can carry them.

    A Decisions client may point at a public HTTP(S) image and let the upstream
    fetch it, which is what a native Decisions endpoint does. No System One
    upstream fetches: OpenRouter documents that remote URLs are not fetched,
    Ollama takes raw base64 and TypeSafe takes base64 assets. The proxy therefore
    downloads each image once and inlines it, so the same request keeps its
    pictures on every provider instead of silently losing them.
    """
    urls = [*_hosted_image_urls(request.input), *_hosted_image_urls(request.questions)]
    if not urls or host.SYSTEMONE_IMAGE_CHANNEL is None:
        # Nothing to inline when the upstream documents no image channel: the
        # images would be dropped a moment later, so neither download them nor
        # fail the request over one that cannot be carried.
        return request
    client = await host._get_client()
    downloads = RequestDownloads()
    outcomes = await downloads.gather(client, urls)

    replacements: dict[str, str] = {}
    for url, outcome in outcomes.items():
        if isinstance(outcome, BaseException):
            # The upstreams below cannot fetch, and dropping the image would
            # answer a question about evidence the model never saw.
            raise ValidationError(
                message=f"Could not download the image at '{url}': {outcome}",
                code="invalid_request_error",
                status_code=400,
            ) from outcome
        if outcome is None:
            raise ValidationError(
                message=f"Could not download the image at '{url}': the response was empty.",
                code="invalid_request_error",
                status_code=400,
            )
        replacements[url] = outcome[0]

    # The budget bounds what the request retains: a URL named twice is inlined
    # twice, so the extra copies are charged here (``gather`` charged the
    # download itself).
    occurrences = Counter(
        [*_hosted_image_occurrences(request.input), *_hosted_image_occurrences(request.questions)]
    )
    for url, data_url in replacements.items():
        extra = occurrences[url] - 1
        if extra > 0:
            downloads.retain(extra * len(data_url), url=url)

    return dataclasses.replace(
        request,
        input=_replace_image_urls(request.input, replacements),
        questions=[_replace_image_urls(question, replacements) for question in request.questions],
    )


__all__ = [
    "DecisionsCapabilityMixin",
    "DecisionsOverSystemOneMixin",
    "SystemOneOverDecisionsMixin",
]
