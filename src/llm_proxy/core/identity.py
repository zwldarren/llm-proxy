"""Typed identity container for request authentication context.

Instead of scattering auth attributes across request.state with untyped
getattr/setattr calls, we centralize them into a single dataclass that
is set once by the auth middleware and read everywhere else through
get_request_identity().

This module lives in the core layer so that processing code can access
request identity without importing from the api layer.
"""

from dataclasses import dataclass

from fastapi import Request


@dataclass
class RequestIdentity:
    """Authenticated identity extracted from the request.

    Set once by auth middleware, read by logging, audit, and dependency code.
    """

    user: str | None = None
    api_key_name: str | None = None
    auth_method: str | None = None
    user_id: int | None = None

    @property
    def is_authenticated(self) -> bool:
        return self.user is not None or self.api_key_name is not None

    @property
    def display_name(self) -> str | None:
        return self.user or self.api_key_name


_IDENTITY_ATTR = "identity"


def set_request_identity(request: Request, identity: RequestIdentity) -> None:
    """Store identity on request.state (called only by auth middleware)."""
    setattr(request.state, _IDENTITY_ATTR, identity)


def get_request_identity(request: Request) -> RequestIdentity:
    """Retrieve identity from request.state.

    Returns a default (unauthenticated) RequestIdentity when the auth
    middleware hasn't set one yet (e.g. health-check endpoints).
    """
    identity = getattr(request.state, _IDENTITY_ATTR, None)
    if isinstance(identity, RequestIdentity):
        return identity
    return RequestIdentity()


__all__ = ["RequestIdentity", "get_request_identity", "set_request_identity"]
