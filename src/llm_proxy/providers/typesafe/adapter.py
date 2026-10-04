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
from llm_proxy.models import InternalRequest, InternalResponse
from llm_proxy.observability.logger import get_logger
from llm_proxy.providers.base import BaseHttpProvider
from llm_proxy.providers.capabilities import SystemOneCapabilityMixin

logger = get_logger(__name__)

_UNSUPPORTED_MESSAGE = (
    "TypeSafe serves System One evaluation only (the /v1/systemone endpoint); "
    "it does not provide chat completions."
)


@register_adapter("typesafe")
class TypeSafeAdapter(SystemOneCapabilityMixin, BaseHttpProvider):
    """Provider adapter for TypeSafe's System One API (the Jev model).

    TypeSafe documents only ``model``/``state``/``questions``, so the optional
    OpenRouter-only fields (``provider``/``session_id``/``trace``/``user``) stay
    under the default ``ignore`` unknown-fields policy and are stripped before
    the request leaves the proxy: ``EXEMPT_EXTRA_KEYS`` is deliberately left at
    its base-class default, and the ``/systemone`` path at the mixin default.
    """

    _DEFAULT_PROVIDER_NAME = "typesafe"

    #: Branding for the admin provider catalog (GET /api/config/provider-types).
    DISPLAY_NAME_EN = "TypeSafe"
    DISPLAY_NAME_ZH = "TypeSafe"

    DEFAULT_BASE_URL = "https://api.typesafe.ai/v1"

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
