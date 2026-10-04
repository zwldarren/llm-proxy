"""TypeSafe (Jev) provider serializer."""

from typing import Any

from llm_proxy.models import InternalRequest, InternalResponse
from llm_proxy.models.systemone import InternalSystemOneRequest
from llm_proxy.serialization.context import BuildContext
from llm_proxy.serialization.providers.base import ProviderSerializer
from llm_proxy.serialization.providers.registry import register_provider_serializer
from llm_proxy.streaming.transformer import StreamingTransformer

#: OpenRouter maps bare Jev model ids onto the ``typesafe/`` namespace (and
#: returns namespaced ids). A client echoing such an id against TypeSafe direct
#: would upstream an invalid model, so the prefix is stripped here; bare ids
#: such as ``jev-latest`` pass through unchanged.
_TYPESAFE_NAMESPACE = "typesafe/"

_UNSUPPORTED = (
    "TypeSafe serves System One evaluation only; it has no chat provider "
    "serializer. Use the shared System One builder/parser instead."
)


@register_provider_serializer("typesafe")
class TypeSafeProviderSerializer(ProviderSerializer):
    """Provider serializer for TypeSafe's System One API.

    TypeSafe serves only the System One evaluation endpoint, so the shared
    System One request-builder/response-parser defaults in
    :class:`ProviderSerializer` apply unchanged. The chat methods are required
    by the ABC but never reached: the adapter rejects chat before any body is
    built.
    """

    _DEFAULT_PROVIDER_NAME = "typesafe"

    def build_provider_systemone_request(self, request: InternalSystemOneRequest) -> dict[str, Any]:
        """Build the TypeSafe System One body, stripping the ``typesafe/`` prefix.

        The shared default is the whole body; TypeSafe itself only understands
        bare Jev ids, so an OpenRouter-namespaced model id is unnamespaced
        before it leaves the proxy.
        """
        body = super().build_provider_systemone_request(request)
        model = body.get("model")
        if isinstance(model, str) and model.startswith(_TYPESAFE_NAMESPACE):
            body["model"] = model[len(_TYPESAFE_NAMESPACE) :]
        return body

    def _build_provider_request(
        self,
        request: InternalRequest,
        context: BuildContext,
    ) -> dict[str, Any]:
        raise NotImplementedError(_UNSUPPORTED)

    def parse_provider_response(
        self, response: dict[str, Any], model: str | None = None, **kwargs: Any
    ) -> InternalResponse:
        raise NotImplementedError(_UNSUPPORTED)

    def get_chunk_converter(self, model: str = "", request_id: str = "") -> StreamingTransformer:
        raise NotImplementedError(_UNSUPPORTED)


__all__ = ["TypeSafeProviderSerializer"]
