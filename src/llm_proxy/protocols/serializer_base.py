"""ProtocolSerializer base class.

Converts between wire protocol format and InternalRequest/InternalResponse.
Protocol serializers live next to their protocol module
(``protocols/<name>/serializer.py``) and use this class to parse incoming
requests and format outgoing responses.

This is the client-facing serializer; the upstream-facing counterpart is
``ProviderSerializer`` in ``llm_proxy.serialization.providers.base``.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from llm_proxy.models import InternalResponse

if TYPE_CHECKING:
    from llm_proxy.serialization.format_context import FormatContext


class ProtocolSerializer(ABC):
    """Convert between wire protocol format and Internal* models.

    Each protocol (OpenAI Chat Completions, Anthropic Messages, OpenAI Responses)
    has its own wire format. ProtocolSerializer handles the conversion in both
    directions.
    """

    @property
    @abstractmethod
    def protocol_name(self) -> str:
        """Protocol name: openai, anthropic, openai_responses."""
        ...

    @abstractmethod
    def parse_request(self, data: dict[str, Any]) -> Any:
        """Parse protocol-format request into an internal request model.

        Args:
            data: Raw request data dict from the client

        Returns:
            An internal request instance (InternalRequest, InternalImageRequest, etc.)

        Raises:
            ValidationError: If required fields are missing or invalid
        """
        ...

    @abstractmethod
    def format_response(
        self, response: InternalResponse, context: FormatContext | None = None
    ) -> dict[str, Any]:
        """Format InternalResponse into protocol wire format.

        Args:
            response: Unified response from the provider
            context: Optional FormatContext with request fields for response construction

        Returns:
            Response dict in protocol wire format
        """
        ...

    @staticmethod
    def _known_request_fields() -> set[str]:
        """Return the set of field names that this serializer handles explicitly.

        Fields not in this set are treated as extra/passthrough and
        stored in InternalRequest.extra.
        """
        return set()


__all__ = ["ProtocolSerializer"]
