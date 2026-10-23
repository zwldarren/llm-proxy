"""The process-lifetime database fan-out: ``request_logs`` + ``usage_records``.

The two stores used to be written by one intake function (``_dispatch``) that
received an already-assembled row and a usage record. They are two sinks of one
observation now; this module owns their composition and lifetime so the intake
verbs do not (ADR-0044).

The bus is a process singleton rather than a service on the request state: its
sinks hold no per-request state, and the intake verbs run from places that have
no request object (middleware shutdown, background judges, MCP calls). Its
lifecycle is explicit all the same — ``api/lifecycle.shutdown_services`` closes
it, which is what reaches ``ObservationSink.close`` for a sink added later.

See ADR-0044.
"""

from threading import RLock

from llm_proxy.observability.observation.bus import ObservationBus
from llm_proxy.observability.observation.policy import SinkPolicy
from llm_proxy.observability.sinks.database_log import DatabaseLogSink
from llm_proxy.observability.sinks.database_usage import DatabaseUsageSink

_bus: ObservationBus | None = None
_lock = RLock()


def database_bus() -> ObservationBus:
    """The bus carrying the request-log and usage sinks (built on first use)."""
    global _bus
    with _lock:
        if _bus is None:
            bus = ObservationBus()
            # The log store asks for the request's own content mode: the bus
            # narrows it, so "full" here means "as much as the deployment allows",
            # not an escalation. The store applies the deployment's body cap
            # itself, from the live config (see ``sinks/database_log.py``).
            bus.register(DatabaseLogSink(), SinkPolicy(content="full"))
            # Accounting is never sampled and never needs content.
            bus.register(DatabaseUsageSink(), SinkPolicy(record=True, content="none"))
            _bus = bus
        return _bus


__all__ = ["database_bus"]
