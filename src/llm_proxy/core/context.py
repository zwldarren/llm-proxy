"""Request-scoped context variables for accessing dependencies.

This module provides context variables for accessing request-scoped dependencies
without passing them through the entire call chain. This reduces coupling and
makes the code more maintainable.

NOTE: a ContextVar only carries a value from one asyncio task to the tasks it
*spawns*. A value written inside the pipeline task — which
``keepalive.await_with_*`` wraps in ``ensure_future`` — is invisible to the
response task that streams the body back to the client, and vice versa (see
``_RequestState`` in the Langfuse handler for the bug this already caused).
Anything that must cross that streaming task boundary must travel by explicit
call argument instead, which is why identity data lives on
:class:`RequestUserContext` and is passed through call chains (e.g.
``RequestUserContext.from_event_context``), not stored here.
"""

from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llm_proxy.config.manager import DatabaseConfigManager


_config_manager_var: ContextVar[DatabaseConfigManager | None] = ContextVar(
    "config_manager", default=None
)


def set_config_manager(config_manager: DatabaseConfigManager) -> None:
    """Set the config manager for the current request context."""
    _config_manager_var.set(config_manager)


def get_config_manager() -> DatabaseConfigManager | None:
    """Get the config manager for the current request context."""
    return _config_manager_var.get()


@dataclass
class RequestUserContext:
    """Stores user identity information for the current request.

    Deliberately not bound to a ContextVar: the streaming request lifecycle
    spans several asyncio tasks, so identity must travel by call argument.
    """

    user_id: int | None = None
    user_identity: str | None = None
    api_key_name: str | None = None
    auth_method: str | None = None

    @classmethod
    def from_event_context(cls, event_context: Any) -> RequestUserContext:
        """Build this identity bundle from an observability ``EventContext``.

        Accepting the event object instead of reading a module-level ContextVar
        keeps the bundle correct across the streaming lifecycle's task boundary
        (see the module note above).
        """
        return cls(
            user_id=getattr(event_context, "auth_user_id", None),
            user_identity=getattr(event_context, "user_id", None),
            api_key_name=getattr(event_context, "api_key_name", None),
            auth_method=getattr(event_context, "auth_method", None),
        )


def reset_context() -> None:
    """Reset all context variables for the current request.

    This should be called at the end of request processing to ensure
    clean state for the next request (especially important for testing).
    """
    _config_manager_var.set(None)


__all__ = [
    "RequestUserContext",
    "get_config_manager",
    "reset_context",
    "set_config_manager",
]
