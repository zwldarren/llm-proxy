"""Unified error handling service."""

from collections.abc import Callable
from typing import Any

import httpx2
from fastapi.responses import Response

from llm_proxy.core.errors.classification import (
    error_type_from_stream_finish_reason,
)
from llm_proxy.core.errors.protocols import ErrorFormatter, ErrorProtocol
from llm_proxy.core.errors.utils import extract_retry_after, get_error_type_for_status
from llm_proxy.core.exceptions import ProviderError
from llm_proxy.observability.logger import get_logger

logger = get_logger(__name__)


_formatter_factory: Callable[[], ErrorFormatter] | None = None


def register_formatter_factory(factory: Callable[[], ErrorFormatter]) -> None:
    """Register a factory that creates the ErrorFormatter implementation.

    Called once at app startup by the api layer to wire up the concrete
    ErrorResponseBuilder without core importing from api.
    """
    global _formatter_factory
    _formatter_factory = factory


class ErrorHandler:
    """Unified error handling service.

    Consolidates error classification, formatting, and response creation
    into a single service, eliminating scattered error handling logic.
    """

    def __init__(self, formatter: ErrorFormatter | None = None):
        self._formatter = formatter

    @property
    def formatter(self) -> ErrorFormatter:
        if self._formatter is None:
            if _formatter_factory is not None:
                self._formatter = _formatter_factory()
            else:
                import importlib

                mod = importlib.import_module("llm_proxy.api.error_responses")
                self._formatter = mod.ErrorResponseBuilder()
        return self._formatter

    def format_response(
        self,
        error: ProviderError,
        protocol: ErrorProtocol = "openai",
    ) -> Response:
        """Format an error as a protocol-specific response.

        When the error carries an upstream response body that is already
        protocol-shaped (``original_error`` with an ``error`` object, as
        captured by ``handle_http_error``), the upstream body is passed
        through verbatim — preserving the provider's own ``error.type`` and
        ``request_id`` instead of re-encoding them. Otherwise the error is
        rebuilt in the protocol shape. An upstream ``Retry-After`` (stashed
        into ``original_error`` by ``handle_http_error``) is surfaced as a
        response header so clients like Claude Code can pace their retries.
        """
        import orjson

        original = error.original_error
        if isinstance(original, dict) and isinstance(original.get("error"), dict):
            error_dict = original
        else:
            error_dict = self.formatter.from_provider_error(error, protocol=protocol)
        headers = {}
        retry_after = original.get("retry_after") if isinstance(original, dict) else None
        if retry_after is not None:
            headers["Retry-After"] = str(retry_after)
        return Response(
            content=orjson.dumps(error_dict),
            status_code=error.status_code or 500,
            media_type="application/json",
            headers=headers,
        )

    def create_provider_error(
        self,
        message: str,
        error_type: str = "api_error",
        status_code: int | None = None,
        provider_name: str | None = None,
        original_error: dict[str, Any] | None = None,
    ) -> ProviderError:
        return ProviderError(
            message=message,
            error_type=error_type,
            status_code=status_code,
            provider_name=provider_name,
            original_error=original_error,
        )

    def create_context_length_error(
        self,
        provider_name: str,
        finish_reason: str,
    ) -> ProviderError:
        return self.create_provider_error(
            message=(
                f"Provider {provider_name} returned finish_reason "
                f"'{finish_reason}' indicating context window exceeded"
            ),
            error_type="context_length_error",
            provider_name=provider_name,
            status_code=400,
        )

    def create_retryable_stream_error(
        self,
        provider_name: str,
        finish_reason: str,
    ) -> ProviderError:
        error_type = error_type_from_stream_finish_reason(finish_reason)
        return self.create_provider_error(
            message=(
                f"Provider {provider_name} returned finish_reason "
                f"'{finish_reason}' before stream produced content"
            ),
            error_type=error_type,
            provider_name=provider_name,
            status_code=502,
        )

    def create_empty_stream_error(self, provider_name: str) -> ProviderError:
        return self.create_provider_error(
            message=f"Provider {provider_name} returned empty stream",
            error_type="api_error",
            provider_name=provider_name,
            status_code=502,
        )

    async def handle_http_error(
        self,
        error: Exception,
        provider_name: str,
    ) -> ProviderError:
        """Convert HTTP/transport errors to ProviderError."""

        if isinstance(error, httpx2.TimeoutException):
            return self.create_provider_error(
                message=f"{provider_name} request timed out: {error}",
                error_type="timeout_error",
                status_code=504,
                provider_name=provider_name,
            )

        if isinstance(error, httpx2.NetworkError):
            return self.create_provider_error(
                message=f"{provider_name} network error: {error}",
                error_type="network_error",
                status_code=503,
                provider_name=provider_name,
            )

        if isinstance(error, httpx2.HTTPStatusError) and error.response is not None:
            response = error.response
            status_code = response.status_code
            retry_after = extract_retry_after(response.headers)
            try:
                error_body = response.json()
                error_type = get_error_type_for_status(status_code) if status_code else "api_error"
                original_error = dict(error_body)
                if retry_after is not None:
                    original_error["retry_after"] = retry_after
                return self.create_provider_error(
                    message=self._extract_message(error_body),
                    error_type=error_type,
                    status_code=status_code,
                    provider_name=provider_name,
                    original_error=original_error,
                )
            except Exception:
                error_type = get_error_type_for_status(status_code) if status_code else "api_error"
                message = str(error)
                try:
                    response_text = response.text
                    if response_text:
                        message = response_text
                except Exception:
                    logger.debug("Failed to read response text for error message", exc_info=True)
                original_error: dict[str, Any] = {"response_text": message}
                if retry_after is not None:
                    original_error["retry_after"] = retry_after
                return self.create_provider_error(
                    message=message,
                    error_type=error_type,
                    status_code=status_code,
                    provider_name=provider_name,
                    original_error=original_error,
                )

        if isinstance(error, httpx2.RemoteProtocolError):
            return self.create_provider_error(
                message=f"{provider_name} request failed: connection closed mid-stream",
                error_type="network_error",
                status_code=502,
                provider_name=provider_name,
            )

        return self.create_provider_error(
            message=f"{provider_name} request failed: {error}",
            error_type="api_error",
            provider_name=provider_name,
        )

    def _extract_message(self, error_body: dict[str, Any]) -> str:
        """Extract error message from provider response."""
        error_data = error_body.get("error", {})
        if isinstance(error_data, dict):
            return error_data.get("message", str(error_body))
        return str(error_data) if error_data else str(error_body)


_global_handler: ErrorHandler | None = None


def get_error_handler() -> ErrorHandler:
    """Get the global ErrorHandler instance."""
    global _global_handler
    if _global_handler is None:
        _global_handler = ErrorHandler()
    return _global_handler
