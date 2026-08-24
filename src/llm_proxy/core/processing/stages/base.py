"""Base types for pipeline stages."""

from abc import ABC, abstractmethod
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from llm_proxy.core.processing.base import RequestContext
from llm_proxy.observability.event_context import EventContext


@dataclass
class PipelineState:
    """Mutable working state shared across pipeline stages — the "what has
    happened so far" object.

    Owns the per-request working set that stages read AND write: the parsed
    request (``unified_request``), live attempt data (``adapter``,
    ``selection``), the raw body (``raw_data`` / ``original_raw_data``), and
    the terminal outcome (``response`` or ``error``). One instance is created
    per request in UnifiedProcessor and threaded through the stage pipeline.

    Do NOT put stable services here (use :class:`RequestContext`) and do NOT
    put observability capture here (use :class:`EventContext`,
    ``state.event_context``).
    """

    raw_data: dict[str, Any]
    unified_request: Any
    req: Any
    strategy: Any
    trace_id: str
    event_context: EventContext
    exit_stack: AsyncExitStack = field(default_factory=AsyncExitStack)
    selection: Any = None
    adapter: Any = None
    response: Any = None
    error: Any = None
    # The pristine client request body, captured before ParameterOverrideStage
    # rebinds ``raw_data`` to the override-modified copy. Provider fallback
    # re-applies each provider's own overrides from THIS body so overrides
    # from a failed provider never leak into the next attempt.
    original_raw_data: dict[str, Any] | None = None

    @property
    def fallback_raw_data(self) -> dict[str, Any]:
        """The body to re-run from on provider fallback: the pristine client
        body when captured, else the current raw data."""
        return self.original_raw_data if self.original_raw_data is not None else self.raw_data


class PipelineStage(ABC):
    """Base class for pipeline stages."""

    @abstractmethod
    async def process(self, state: PipelineState, context: RequestContext) -> None:
        """Process the pipeline state. Set state.response or state.error when done."""
        ...
