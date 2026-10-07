"""Usage observation and per-turn request logging for Realtime sessions.

The Realtime API reports usage in the ``response.done`` event
(``response.usage``). The observer watches the upstream→client stream and
writes one background request log per completed model turn so the dashboard
shows per-call tokens and cost. Observation is best-effort: failures are
logged and never break the relay.
"""

from dataclasses import dataclass
from typing import Any, Literal

import orjson

from llm_proxy.billing.cost import calculate_cost
from llm_proxy.observability.log_intake import record_realtime_turn
from llm_proxy.observability.logger import get_logger
from llm_proxy.realtime.relay import KIND_TEXT

logger = get_logger(__name__)

# The upstream event that carries per-turn usage.
_RESPONSE_DONE = "response.done"

# The upstream event that announces the provider-issued session id.
_SESSION_CREATED = "session.created"


@dataclass
class RealtimeSessionContext:
    """Identity and log-context fields shared by every turn of one session.

    Built once per WebSocket connection by the router and passed to the
    usage observer; every per-turn request log entry is stamped from it.
    ``session_id`` starts as the proxy connection id and is replaced by the
    provider-issued session id once ``session.created`` announces it.
    """

    model: str
    provider: str
    api_key_name: str
    request_id: str
    client_ip: str | None = None
    user_agent: str | None = None
    session_id: str | None = None
    user_id: int | None = None


class RealtimeUsageObserver:
    """Extract usage from ``response.done`` events and log each turn.

    Also captures the provider-issued session id from ``session.created``
    for log correlation. The session id falls back to the proxy connection
    id until the upstream session is announced.

    Args:
        context: Session identity/log-context (model, provider, key name,
            connection ids, client metadata)
        config_manager: Config manager for pricing lookup (cost stays None
            when omitted)
    """

    def __init__(self, *, context: RealtimeSessionContext, config_manager: Any = None):
        self._context = context
        self._config_manager = config_manager
        self._turns = 0

    @property
    def turns(self) -> int:
        """Number of ``response.done`` turns recorded for this session."""
        return self._turns

    @property
    def session_id(self) -> str | None:
        """Session id for log correlation (provider-issued once known)."""
        return self._context.session_id

    async def on_upstream_message(self, kind: Literal["text", "binary"], data: str | bytes) -> None:
        """Relay hook: record a log entry for each ``response.done`` event."""
        if kind != KIND_TEXT or not isinstance(data, str):
            return
        try:
            event = orjson.loads(data)
        except orjson.JSONDecodeError:
            return
        if not isinstance(event, dict):
            return
        event_type = event.get("type")
        if event_type == _SESSION_CREATED:
            # Correlate the connection with the upstream session: the
            # provider-issued session id (not the proxy connection id) is
            # the useful key when cross-checking with OpenAI dashboards.
            session = event.get("session")
            if isinstance(session, dict) and session.get("id"):
                self._context.session_id = session["id"]
            return
        if event_type != _RESPONSE_DONE:
            return
        response = event.get("response")
        if not isinstance(response, dict):
            return
        usage = response.get("usage")
        if not isinstance(usage, dict):
            usage = None
        await self._record_turn(response, usage)

    async def _record_turn(self, response: dict[str, Any], usage: dict[str, Any] | None) -> None:
        """Calculate cost and write the background request log for one turn.

        A ``response.done`` without usage (e.g. failed/cancelled turns that
        never produced tokens) still yields a zero-token log entry so the
        dashboard shows every turn; ``usage_missing`` marks it.
        """
        response_id = response.get("id") or ""
        self._turns += 1
        breakdown = await calculate_cost(
            usage,
            self._context.model,
            config_manager=self._config_manager,
            provider_name=self._context.provider,
        )
        record_realtime_turn(
            self._context,
            response_id=response_id,
            response_status=response.get("status"),
            breakdown=breakdown,
            usage_missing=usage is None,
        )


__all__ = ["RealtimeSessionContext", "RealtimeUsageObserver"]
