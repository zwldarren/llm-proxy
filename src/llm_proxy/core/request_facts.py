"""The typed record of what the proxy learns about one in-flight request.

A request accumulates ambient facts as it travels: who sent it (the auth
gates), what the client asked for and where it was routed (the pipeline), what
the logger captured for it (the audit middleware), and two latches — "the
client is gone", "an audit row already exists". Before this module those facts
lived as untyped `request.state` attributes, written by 21 `setattr` sites and
read by `getattr(request.state, "name", None)` probes across 14 modules: a typo
was a permanent `None` (a missing request id, a missing allowlist, a silently
disabled latch), a rename compiled nothing, and no reader could tell "the
writer never ran" from "the fact is legitimately absent".

:class:`RequestFacts` is the record; one slot in ``request.state`` holds it.
Readers call :func:`facts_for` (or :func:`facts_from_scope` from a pure-ASGI
middleware) and use typed fields, and nothing else in the codebase spells one
of these names.

**Who writes what.** Ownership is per group, and it is the ordering that makes
each group readable:

- ``request_id`` — minted when the record is created, i.e. by the first toucher.
  For HTTP that is the logging middleware, which sits outside every auth gate,
  so a rejected request still has its id for the rejection row. Websocket
  handshakes mint their own (``ws_`` prefix) in
  :func:`llm_proxy.core.ws_common.build_ws_request`.
- ``identity``, ``allowed_models`` — the auth gates only.
- ``model``, ``provider``, ``session_id``, ``parsed_request_body`` — the
  protocol handlers and the processing pipeline.
- ``request_headers``/``request_body``/``response_headers``/``response_body`` —
  the audit middleware (``HttpLoggingMiddleware``), for the intake module's
  row assembly.
- ``client_disconnected`` (writer: ``api/keepalive.py``) and
  ``audit_log_written`` (writer: the intake module's verbs) — latches, each
  with exactly one writer.

Tests construct the record instead of poking strings: ``facts_for(request)``
works on any object with a ``state`` attribute, including a bare
``MagicMock``/``SimpleNamespace`` double.
"""

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

#: The one ``request.state`` slot this module owns.
_FACTS_ATTR = "request_facts"


@dataclass
class RequestIdentity:
    """Authenticated identity extracted from the request.

    Set once by an auth gate, read by logging, audit, routing and dependency
    code. Reached through :func:`get_request_identity`, which is the identity
    part of the request's :class:`RequestFacts`.
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


def mint_request_id(prefix: str = "") -> str:
    """Mint the opaque id that correlates one request's log rows.

    The format is owned here so a request id looks the same wherever it is
    minted; ``prefix`` distinguishes a websocket connection's id from a
    request's.
    """
    return f"{prefix}{uuid4().hex}"


@dataclass
class RequestFacts:
    """What the proxy has learned about one in-flight request.

    Fields are documented by writer in the module docstring; every one of them
    is optional in the sense that a given request may never learn it, which is
    why the absent value is a typed default (``None``/``False``/``{}``) rather
    than a missing attribute.
    """

    request_id: str = field(default_factory=mint_request_id)
    identity: RequestIdentity = field(default_factory=RequestIdentity)
    #: The API key's model allowlist (``None`` = unrestricted, ``[]`` = deny-all).
    allowed_models: list[str] | None = None
    #: The user-facing model, until provider selection replaces it with the
    #: concrete model being served — the value error rows and logs report.
    model: str | None = None
    provider: str | None = None
    session_id: str | None = None
    #: The protocol handler's parsed request body, for early-failure logging.
    parsed_request_body: Any = None
    client_disconnected: bool = False
    audit_log_written: bool = False
    request_headers: dict[str, Any] = field(default_factory=dict)
    request_body: Any = None
    response_headers: dict[str, Any] = field(default_factory=dict)
    response_body: Any = None


def facts_from_scope(scope: Any) -> RequestFacts:
    """Return the record for a raw ASGI scope, creating it on first use.

    The pure-ASGI middlewares hold a scope, not a ``Request``; both reach the
    same carrier (``scope["state"]`` is the dict ``request.state`` reads).
    """
    state = scope.setdefault("state", {})
    facts = state.get(_FACTS_ATTR) if hasattr(state, "get") else None
    if not isinstance(facts, RequestFacts):
        facts = RequestFacts()
        state[_FACTS_ATTR] = facts
    return facts


def facts_for(request: Any) -> RequestFacts:
    """Return the record for a request, creating it on first use.

    Works on a ``Request``, a websocket-derived fake request, or a test double
    whose ``state`` accepts attribute writes.
    """
    state = request.state
    facts = getattr(state, _FACTS_ATTR, None)
    if not isinstance(facts, RequestFacts):
        facts = RequestFacts()
        setattr(state, _FACTS_ATTR, facts)
    return facts


def set_request_identity(request: Any, identity: RequestIdentity) -> None:
    """Store the authenticated identity (called only by an auth gate)."""
    facts_for(request).identity = identity


def get_request_identity(request: Any) -> RequestIdentity:
    """Retrieve identity from the request's facts.

    Returns a default (unauthenticated) ``RequestIdentity`` when no auth gate
    has run (e.g. health checks, a websocket handshake before verification).
    """
    return facts_for(request).identity


__all__ = [
    "RequestFacts",
    "RequestIdentity",
    "facts_for",
    "facts_from_scope",
    "get_request_identity",
    "mint_request_id",
    "set_request_identity",
]
