"""Fan one observation out to the sinks that asked for it.

The bus owns the only cross-sink rules: which kinds a sink subscribes to, which
policy it registered with, that a failing sink never fails a request, and that
flush/close reach every sink. Everything else — what the facts are, what a sink
does with them — lives on either side of it.

See ADR-0044.
"""

import asyncio
from dataclasses import dataclass
from threading import RLock

from llm_proxy.observability.logger import get_logger
from llm_proxy.observability.observation.model import ObservationKind, RequestObservation
from llm_proxy.observability.observation.policy import FACTS_ONLY, SinkPolicy
from llm_proxy.observability.observation.projection import apply_sink_policy
from llm_proxy.observability.sinks.base import ObservationSink

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class _Registration:
    sink: ObservationSink
    policy: SinkPolicy = FACTS_ONLY


class ObservationBus:
    """Ordered, failure-isolated fan-out of observations to sinks."""

    def __init__(self) -> None:
        self._registrations: list[_Registration] = []
        self._lock = RLock()

    def register(self, sink: ObservationSink, policy: SinkPolicy = FACTS_ONLY) -> None:
        """Subscribe a sink, with its policy, at the end of the fan-out order."""
        with self._lock:
            if any(registration.sink is sink for registration in self._registrations):
                return
            self._registrations.append(_Registration(sink=sink, policy=policy))

    def unregister(self, sink: ObservationSink) -> None:
        """Remove a sink (a config change dropped it)."""
        with self._lock:
            self._registrations = [r for r in self._registrations if r.sink is not sink]

    @property
    def sinks(self) -> list[ObservationSink]:
        """A copy of the registered sinks."""
        with self._lock:
            return [registration.sink for registration in self._registrations]

    def _for_kind(self, kind: ObservationKind) -> list[_Registration]:
        with self._lock:
            return [r for r in self._registrations if kind in r.sink.kinds]

    async def emit(self, observation: RequestObservation) -> int:
        """Deliver an observation; returns how many sinks accepted it.

        Every sink is isolated: one backend being down or misconfigured must not
        cost the request its audit row, and must not stop the other sinks.
        Delivery runs concurrently because a slow network backend must not delay
        the rest of the fan-out.
        """
        deliveries: list[tuple[_Registration, RequestObservation]] = []
        for registration in self._for_kind(observation.kind):
            narrowed = apply_sink_policy(observation, registration.policy)
            if narrowed is not None:
                deliveries.append((registration, narrowed))
        if not deliveries:
            return 0
        results = await asyncio.gather(
            *(registration.sink.emit(narrowed) for registration, narrowed in deliveries),
            return_exceptions=True,
        )
        delivered = 0
        for (registration, _), result in zip(deliveries, results, strict=True):
            if isinstance(result, BaseException):
                logger.warning(
                    f"Observation sink {registration.sink.name!r} failed to accept "
                    f"{observation.kind} observation {observation.core.request_id}",
                    exc_info=result,
                )
                continue
            delivered += 1
        return delivered

    async def flush(self) -> None:
        """Flush every sink, isolating failures."""
        await self._fanout("flush")

    async def close(self) -> None:
        """Close every sink, isolating failures."""
        await self._fanout("close")

    async def _fanout(self, method: str) -> None:
        with self._lock:
            sinks = [registration.sink for registration in self._registrations]
        calls = [getattr(sink, method)() for sink in sinks]
        if not calls:
            return
        results = await asyncio.gather(*calls, return_exceptions=True)
        for sink, result in zip(sinks, results, strict=True):
            if isinstance(result, BaseException):
                logger.warning(
                    f"Observation sink {sink.name!r} {method} failed",
                    exc_info=result,
                )


__all__ = ["ObservationBus"]
