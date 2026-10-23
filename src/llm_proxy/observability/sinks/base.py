"""The sink interface: collect once, project by policy, emit once.

A sink does not observe the request lifecycle. It receives an already finished,
already policy-filtered :class:`RequestObservation` and writes it somewhere. That
is the whole interface: the kinds it subscribes to, and ``emit`` / ``flush`` /
``close``.

There is deliberately no capability descriptor beside that. What a sink may
*consume* is the policy it registered with (``SinkPolicy``); a second set of
content/sampling flags could only drift from it. A backend that genuinely needs
live events — a trace view that updates as tokens arrive — is not a sink on this
bus: it keeps the request lifecycle it needs and reads the same observation at
its boundaries. The Langfuse exporter (``sinks/langfuse.py``) is that case, and
its handler is the lifecycle adapter.

See ADR-0044.
"""

from typing import Protocol, runtime_checkable

from llm_proxy.observability.observation.model import ObservationKind, RequestObservation


@runtime_checkable
class ObservationSink(Protocol):
    """One output of the observation pipeline."""

    name: str
    kinds: frozenset[ObservationKind]

    async def emit(self, observation: RequestObservation) -> None:
        """Write one already-projected observation."""
        ...

    async def flush(self) -> None:
        """Flush anything buffered (called on config reload and shutdown)."""
        ...

    async def close(self) -> None:
        """Release resources at process teardown."""
        ...


__all__ = ["ObservationSink"]
