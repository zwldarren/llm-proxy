"""TypeSafe (Jev) provider implementation.

TypeSafe serves the System One evaluation API at
``https://api.typesafe.ai/v1/systemone``. It is not a chat provider, so the
adapter implements the mandatory chat methods with a clear error and exposes
:class:`SystemOneCapabilityMixin` for ``/v1/systemone`` traffic.
"""

from collections.abc import AsyncIterator
from typing import Any

from llm_proxy.core.adapter import register_adapter
from llm_proxy.core.exceptions import ValidationError
from llm_proxy.core.request_type import RequestType
from llm_proxy.models import InternalRequest, InternalResponse
from llm_proxy.models.systemone import InternalSystemOneRequest
from llm_proxy.observability.logger import get_logger
from llm_proxy.providers.base import BaseHttpProvider
from llm_proxy.providers.capabilities import (
    DecisionsOverSystemOneMixin,
    SystemOneCapabilityMixin,
)

logger = get_logger(__name__)

_UNSUPPORTED_MESSAGE = (
    "TypeSafe serves System One evaluation only (the /v1/systemone endpoint); "
    "it does not provide chat completions."
)


def _needs_image_danger_zone(request: InternalSystemOneRequest) -> bool:
    """Whether a request carries one of TypeSafe's gated image features.

    ``assets`` is the image channel (the bridge fills it with a Decisions
    request's inline images) and ``box`` is the question type that locates
    something in one. TypeSafe documents both as early features behind a
    ``TypeSafe-Danger-Zone-Images`` opt-in header.
    """
    if request.extra.get("assets"):
        return True
    return any(
        isinstance(question, dict) and question.get("type") == "box"
        for question in request.questions.values()
    )


@register_adapter("typesafe")
class TypeSafeAdapter(DecisionsOverSystemOneMixin, SystemOneCapabilityMixin, BaseHttpProvider):
    """Provider adapter for TypeSafe's System One API (the Jev model).

    TypeSafe documents ``model``/``state``/``questions`` plus ``assets``, its
    image channel; the optional OpenRouter-only fields
    (``provider``/``session_id``/``trace``/``user``) stay under the default
    ``ignore`` unknown-fields policy and are stripped before the request leaves
    the proxy. Images arrive as ``assets`` — a placeholder map the upstream
    substitutes into ``state`` and instruction text — which is the channel
    :attr:`SYSTEMONE_IMAGE_CHANNEL` declares, so a bridged Decisions request
    keeps its pictures.

    ``/v1/decisions`` is served by bridging onto that same System One endpoint
    (``DecisionsOverSystemOneMixin``), so a Decisions request reaches Jev
    through the one upstream TypeSafe documents.
    """

    _DEFAULT_PROVIDER_NAME = "typesafe"

    #: Branding for the admin provider catalog (GET /api/config/provider-types).
    DISPLAY_NAME_EN = "TypeSafe"
    DISPLAY_NAME_ZH = "TypeSafe"

    DEFAULT_BASE_URL = "https://api.typesafe.ai/v1"

    #: TypeSafe takes images as an ``assets`` map keyed by placeholder, so the
    #: bridge renders each image as one entry and leaves the placeholder where
    #: the image belongs in the text.
    SYSTEMONE_IMAGE_CHANNEL = "assets"

    #: ``assets`` is documented on this request type; the OpenRouter-only fields
    #: are not, and stay under the default policy.
    EXEMPT_EXTRA_KEYS = {
        RequestType.SYSTEMONE: frozenset({"assets"}),
    }

    #: Opt-in header TypeSafe requires on a request that carries an image asset
    #: or a box question.
    DANGER_ZONE_HEADER = "TypeSafe-Danger-Zone-Images"

    def _systemone_headers(self, request: InternalSystemOneRequest) -> dict[str, str]:
        """Add TypeSafe's image opt-in when the request needs it.

        The header is sent only for the requests that carry an image feature, so a
        text-only evaluation keeps the default (ungated) route.
        """
        headers = super()._systemone_headers(request)
        if _needs_image_danger_zone(request):
            headers[self.DANGER_ZONE_HEADER] = "true"
        return headers

    async def chat_completion(self, request: InternalRequest, **_kwargs: Any) -> InternalResponse:
        raise ValidationError(
            message=_UNSUPPORTED_MESSAGE,
            code="invalid_request_error",
            status_code=400,
        )

    async def stream_chat_completion(
        self,
        request: InternalRequest,
        cancel_token: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[str | dict[str, Any]]:
        raise ValidationError(
            message=_UNSUPPORTED_MESSAGE,
            code="invalid_request_error",
            status_code=400,
        )


__all__ = ["TypeSafeAdapter"]
