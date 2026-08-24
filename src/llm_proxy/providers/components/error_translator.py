"""Map transport/HTTP errors to ProviderError."""

from typing import Any

from llm_proxy.core.errors import get_error_handler, get_error_type_for_status
from llm_proxy.core.exceptions import ProviderError


class ErrorTranslator:
    """Translate transport and HTTP errors into ProviderError.

    Consolidates error mapping logic from:
    - BaseProvider._parse_error_response (provider-specific error body parsing)
    - BaseProvider._handle_http_error (delegates to ErrorHandler)
    - Error mapping in _with_retry and _with_retry_generator except blocks
    """

    def __init__(
        self,
        provider_name: str | None = None,
        error_handler=None,
    ):
        self._provider_name = provider_name
        self._error_handler = error_handler or get_error_handler()

    def parse_error_response(self, status_code: int, error_body: dict[str, Any]) -> ProviderError:
        """Parse provider error response body. Override for provider-specific formats."""
        error_data = error_body.get("error", {})
        if isinstance(error_data, dict):
            message = error_data.get("message", str(error_body))
            error_type = error_data.get("type", get_error_type_for_status(status_code))
            code = error_data.get("code")
        else:
            message = str(error_data) if error_data else str(error_body)
            error_type = get_error_type_for_status(status_code)
            code = None

        return ProviderError(
            message=message,
            error_type=error_type,
            code=code,
            status_code=status_code,
            provider_name=self._provider_name,
            original_error=error_body,
        )

    async def translate_error(self, error: Exception) -> ProviderError:
        """Convert transport/HTTP errors to ProviderError.

        Handles asyncio.TimeoutError and delegates httpx2 errors to ErrorHandler.
        """
        import asyncio

        if isinstance(error, (asyncio.TimeoutError, TimeoutError)):
            return ProviderError(
                message=f"Request timeout: {error}",
                error_type="timeout_error",
                provider_name=self._provider_name,
                original_error={"type": type(error).__name__, "message": str(error)},
            )

        return await self._error_handler.handle_http_error(
            error=error,
            provider_name=self._provider_name or "unknown",
        )


__all__ = ["ErrorTranslator"]
