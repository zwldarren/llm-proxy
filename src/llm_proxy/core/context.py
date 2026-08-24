"""Request-scoped context variables for accessing dependencies.

This module provides context variables for accessing request-scoped dependencies
without passing them through the entire call chain. This reduces coupling and
makes the code more maintainable.
"""

from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_proxy.config.manager import DatabaseConfigManager


@dataclass
class RequestUserContext:
    """Stores user identity information for the current request."""

    user_id: int | None = None
    user_identity: str | None = None
    api_key_name: str | None = None
    auth_method: str | None = None


_config_manager_var: ContextVar[DatabaseConfigManager | None] = ContextVar(
    "config_manager", default=None
)

_request_user_context_var: ContextVar[RequestUserContext | None] = ContextVar(
    "request_user_context", default=None
)


def set_config_manager(config_manager: DatabaseConfigManager) -> None:
    """Set the config manager for the current request context."""
    _config_manager_var.set(config_manager)


def get_config_manager() -> DatabaseConfigManager | None:
    """Get the config manager for the current request context."""
    return _config_manager_var.get()


def set_request_user_context(context: RequestUserContext) -> None:
    """Set the user identity context for the current request."""
    _request_user_context_var.set(context)


def get_request_user_context() -> RequestUserContext | None:
    """Get the user identity context for the current request."""
    return _request_user_context_var.get()


def reset_context() -> None:
    """Reset all context variables for the current request.

    This should be called at the end of request processing to ensure
    clean state for the next request (especially important for testing).
    """
    _config_manager_var.set(None)
    _request_user_context_var.set(None)


__all__ = [
    "get_config_manager",
    "get_request_user_context",
    "reset_context",
    "set_config_manager",
    "set_request_user_context",
    "RequestUserContext",
]
