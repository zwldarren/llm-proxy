"""Custom exceptions for LLM Proxy."""


class LLMProxyError(Exception):
    """Base exception for all LLM Proxy errors."""

    def __init__(
        self,
        message: str = "An error occurred",
        code: str | None = None,
        status_code: int | None = None,
    ):
        self.message = message
        self.code = code or "internal_error"
        self.status_code = status_code
        super().__init__(message)


class AdapterNotFoundError(LLMProxyError):
    """Raised when no adapter is found for a provider."""


class ClientDisconnectedError(LLMProxyError):
    """Raised (or recorded) when the client goes away before the response.

    Behind CDNs such as Cloudflare this is how a 524 surfaces server-side: the
    CDN abandons the request after its time-to-first-byte budget expires and
    closes the connection while the origin keeps generating. Status 499 follows
    the nginx "client closed request" convention so it lands in the 4xx error
    bucket of the logs UI instead of looking like a success.
    """

    def __init__(self, message: str | None = None):
        super().__init__(
            message
            or "Client disconnected before the response completed "
            "(likely a client/CDN timeout, e.g. Cloudflare 524)",
            code="client_disconnected",
            status_code=499,
        )


class ConfigurationError(LLMProxyError):
    """Raised when there's a configuration error."""


class ModelNotFoundError(ConfigurationError):
    """Raised when a requested model is not configured."""

    def __init__(self, model: str):
        super().__init__(
            message=f"Model '{model}' not found in configuration",
            code="model_not_found",
            status_code=404,
        )
        self.error_type = "not_found_error"


class ProviderNotConfiguredError(ConfigurationError):
    """Raised when a requested provider is not configured."""

    def __init__(self, provider: str):
        super().__init__(
            message=f"Provider '{provider}' not configured",
            code="provider_not_configured",
            status_code=404,
        )
        self.error_type = "not_found_error"


class EncryptionError(LLMProxyError):
    """Raised when encryption/decryption fails."""


class RequestError(LLMProxyError):
    """Base exception for request-related errors."""


class MCPTimeoutError(RequestError):
    """Raised when a request times out."""


class ProviderError(RequestError):
    """Raised when a provider returns an error."""

    def __init__(
        self,
        message: str,
        error_type: str | None = None,
        param: str | None = None,
        code: str | None = None,
        status_code: int | None = None,
        provider_name: str | None = None,
        original_error: dict | None = None,
    ):
        super().__init__(message, code=code, status_code=status_code)
        self.error_type = error_type
        self.param = param
        self.provider_name = provider_name
        self.original_error = original_error


class ValidationError(LLMProxyError, ValueError):
    """Raised when request/response validation fails."""


class NotFoundError(LLMProxyError):
    """Raised when a resource is not found."""

    def __init__(
        self,
        message: str = "Resource not found",
        code: str | None = None,
        status_code: int | None = None,
    ):
        super().__init__(message, code or "not_found", status_code=status_code)


class AuthenticationFailedError(LLMProxyError):
    """Raised when authentication fails."""

    def __init__(
        self,
        message: str = "Authentication failed",
        code: str | None = None,
        status_code: int | None = None,
    ):
        super().__init__(message, code or "authentication_failed", status_code=status_code)


class ForbiddenError(LLMProxyError):
    """Raised when the user is authenticated but lacks permission."""

    def __init__(
        self,
        message: str = "Forbidden",
        code: str | None = None,
        status_code: int | None = None,
    ):
        super().__init__(message, code or "forbidden", status_code=status_code or 403)


class ConflictError(LLMProxyError):
    """Raised when there's a conflict (e.g., duplicate resource)."""

    def __init__(self, message: str = "Resource conflict", code: str | None = None):
        super().__init__(message, code or "conflict")


class WebSearchError(LLMProxyError):
    """Error during web search execution."""

    def __init__(
        self,
        message: str,
        error_code: str = "unavailable",
        provider_name: str | None = None,
    ):
        super().__init__(message=message, code=error_code)
        self.provider_name = provider_name


class MCPError(LLMProxyError):
    """Base exception for MCP-related errors."""


class MCPConnectionError(MCPError):
    """Raised when not connected to an MCP backend."""

    def __init__(
        self, message: str = "Not connected to backend", server_name: str | None = None
    ) -> None:
        super().__init__(message)
        self.server_name = server_name
        self.error_type = "mcp_connection_error"


class MCPServerNotFoundError(MCPError):
    """Raised when an MCP server is not found or cannot be located."""

    def __init__(self, server_name: str, details: str | None = None) -> None:
        super().__init__(f"MCP server not found: {server_name}")
        self.server_name = server_name
        self.details = details


class MCPStartupError(MCPError):
    """Raised when an MCP server fails to start or initialize."""

    def __init__(
        self,
        server_name: str,
        reason: str,
        original_error: Exception | None = None,
        *,
        error_type: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(f"MCP server startup failed: {server_name} - {reason}")
        self.server_name = server_name
        self.reason = reason
        self.original_error = original_error
        self.error_type = error_type or "mcp_startup_error"
        self.status_code = status_code


class MCPSecurityError(MCPError):
    """Raised when an MCP server configuration violates the security policy."""

    def __init__(self, message: str, server_name: str | None = None) -> None:
        super().__init__(message)
        self.server_name = server_name
        self.error_type = "mcp_security_error"


__all__ = [
    "AdapterNotFoundError",
    "AuthenticationFailedError",
    "ClientDisconnectedError",
    "ConfigurationError",
    "ConflictError",
    "ForbiddenError",
    "EncryptionError",
    "LLMProxyError",
    "MCPConnectionError",
    "MCPError",
    "MCPSecurityError",
    "MCPServerNotFoundError",
    "MCPStartupError",
    "MCPTimeoutError",
    "ModelNotFoundError",
    "NotFoundError",
    "ProviderError",
    "ProviderNotConfiguredError",
    "RequestError",
    "ValidationError",
    "WebSearchError",
]
