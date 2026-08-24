"""Error protocol types and formatting interface.

Core-layer definitions that the api layer implements.
This breaks the reverse dependency where core/errors/handler.py
and protocols/base.py imported from api/error_responses.py.
"""

from abc import ABC, abstractmethod
from typing import Any, Literal

ErrorProtocol = Literal["openai", "anthropic"]


def protocol_for_name(protocol_name: str | None) -> ErrorProtocol:
    """Map a protocol endpoint name to an ErrorProtocol.

    Only the Anthropic Messages protocol formats errors in the Anthropic
    shape; OpenResponses errors use the OpenAI envelope (plus spec-code
    mapping applied by the middleware layer), so everything else maps to
    ``openai``.
    """
    return "anthropic" if protocol_name == "anthropic" else "openai"


class ErrorFormatter(ABC):
    """Interface for formatting errors into protocol-specific responses.

    The core layer depends on this interface. The api layer provides
    the concrete implementation (ErrorResponseBuilder).
    """

    @abstractmethod
    def from_provider_error(
        self,
        error: Any,
        protocol: ErrorProtocol = "openai",
    ) -> dict[str, Any]:
        """Format a provider error as a protocol-specific dict."""
        ...
