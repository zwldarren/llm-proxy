"""The one normalized fact set every observability sink reads.

An :class:`EventContext` is the collector's working state — mutable, with
streaming buffers and provider-specific fields. A :class:`RequestObservation` is
the finished, immutable snapshot taken at the end of a request: it is neither a
database row nor a Langfuse payload, but the upstream fact both are projected
from. Adding a backend must not require understanding the request lifecycle;
adding a field must not require touching every sink.

Sections are grouped by the question they answer (who / where / what / how much /
how long / how it ended), so a sink declares which sections it consumes. Audit
classification (``event_type``, ``outcome`` …) and the hash chain are *not* here:
they are the request-log sink's projection of these facts, not part of the
generic observation — an OTLP or Phoenix backend must not have to understand
them.

``content`` is captured only when the request paid to capture it; whether a sink
is allowed to *see* it is a separate decision, made by
:mod:`llm_proxy.observability.observation.policy`. The collector records what
exists; the policy narrows it per sink.

See ADR-0044.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from llm_proxy.models.types import Usage
from llm_proxy.observability.observation.policy import ContentPolicy


class ObservationKind(StrEnum):
    """What kind of operation an observation describes.

    The endpoint lifecycle is ``REQUEST``; the standalone situations (tool calls,
    proxy-executed web searches, judge consultations, realtime turns) each have
    their own kind so a sink can subscribe to the ones it understands.

    Only ``REQUEST`` is produced today: the remaining situations still build their
    rows in the intake and adopt the bus one at a time (ADR-0044, migration step
    6). The other kinds are declared so that adoption is a producer change, not a
    protocol change.
    """

    REQUEST = "request"
    MCP = "mcp"
    WEB_SEARCH = "web_search"
    JUDGE = "judge"
    REALTIME = "realtime"


@dataclass(frozen=True, slots=True)
class CoreFacts:
    """Identity and classification of the operation itself."""

    request_id: str
    trace_id: str | None
    #: The pipeline's request type, or ``None`` when it never set one. Kept
    #: distinct from the "chat" default a log row shows, because a backend
    #: profile must not guess a type the pipeline did not declare.
    request_type: str | None
    log_type: str
    endpoint: str
    method: str
    #: Unix timestamp of the moment the request started.
    started_at: float
    is_api_endpoint: bool
    #: The audit resource this request names, derived once at capture from the
    #: raw body. A sink may not see that body, so it cannot derive this itself.
    resource_id: str | None = None


@dataclass(frozen=True, slots=True)
class IdentityFacts:
    """Who made the request."""

    user: str | None = None
    user_id: int | None = None
    api_key_name: str | None = None
    auth_method: str | None = None
    session_id: str | None = None
    client_ip: str | None = None
    user_agent: str | None = None


@dataclass(frozen=True, slots=True)
class RoutingFacts:
    """Which model/provider served the request, and what it took to get there."""

    model: str | None = None
    internal_model: str | None = None
    provider: str | None = None
    provider_model_name: str | None = None
    fallbacks: tuple[Mapping[str, Any], ...] = ()
    retries: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class UsageFacts:
    """Every billable quantity of the turn.

    ``usage`` is the canonical record (ADR-0025), copied in by the collector so
    the observation cannot be mutated through the provider's object. The flat
    dimensions beside it are the ones the proxy itself produced or derived.
    """

    usage: Usage | None = None
    provider_reported_cost: float | None = None
    web_search_requests: int | None = None
    images_generated: int | None = None
    tts_characters: int | None = None
    cache_savings_usd: float | None = None
    cost_usd: float | None = None


@dataclass(frozen=True, slots=True)
class TimingFacts:
    """How long the request took."""

    latency_ms: float | None = None
    ttft_ms: float | None = None
    #: Wall-clock time of the first streamed chunk (Langfuse's
    #: ``completion_start_time``); not a duration.
    first_chunk_time: datetime | None = None


@dataclass(frozen=True, slots=True)
class OutcomeFacts:
    """How the request ended."""

    #: Whether the operation failed. A sink needs this to choose the failure
    #: projection; it must not infer it from the status code (a successful
    #: fallback row can still carry an error message).
    failed: bool = False
    status_code: int | None = None
    error_message: str | None = None
    error_stack_trace: str | None = None
    error_details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ContentFacts:
    """What was actually on the wire.

    ``captured`` records the sampling decision taken when the request started:
    ``False`` means the payloads were deliberately not kept, which is a different
    fact from "the payload was empty". ``raw_stream`` holds the buffered SSE text
    only when raw capture was enabled; ``assembled_response_body`` is the
    non-streaming body reassembled from the stream otherwise.
    """

    request_headers: Mapping[str, Any] = field(default_factory=dict)
    request_body: Any = None
    response_headers: Mapping[str, Any] = field(default_factory=dict)
    response_body: Any = None
    assembled_response_body: Any = None
    #: The buffered raw SSE text, verbatim. This is the one content field a
    #: projection does not rewrite: it exists for byte-exact replay, so masking
    #: it would defeat its purpose. A policy can only cap it (see
    #: ``projection``); turning raw capture on is the operator's decision to store
    #: the frames as they arrived.
    raw_stream: bytes | None = None
    #: Whether raw SSE capture was requested (``None`` raw frames then mean
    #: "nothing arrived", not "raw capture was off").
    raw_stream_requested: bool = False
    raw_stream_truncated: bool = False
    #: The transformer's accumulated blocks for a finished stream, for a backend
    #: that renders richer output than the reassembled body (tool calls, …). It
    #: carries tool arguments, so it is content: a projection masks it.
    stream_output_blocks: Any = None
    streaming: bool = False
    #: Whether the capture actually kept the payloads (the sampling decision).
    captured: bool = False
    #: Set when a sink's policy withheld the payloads from this projection.
    #: ``captured`` still says whether they existed, so a sink can explain the
    #: absence ("bodies disabled" vs "sampled out") instead of guessing.
    withheld: bool = False


@dataclass(frozen=True, slots=True)
class OperationFacts:
    """What a non-request operation (tool call, search, judge, turn) did."""

    operation: str
    resource_type: str | None = None
    resource_name: str | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RequestObservation:
    """One finished operation, normalized for every sink.

    ``attributes`` carries the residual, non-content metadata a sink may publish
    as-is (endpoint, request type, routing details, fallback entries). It is a
    plain mapping because sinks render it differently: the log store persists it
    as ``log_metadata``, Langfuse merges it into ``metadata``.
    """

    kind: ObservationKind
    core: CoreFacts
    identity: IdentityFacts = field(default_factory=IdentityFacts)
    routing: RoutingFacts = field(default_factory=RoutingFacts)
    usage: UsageFacts = field(default_factory=UsageFacts)
    timing: TimingFacts = field(default_factory=TimingFacts)
    outcome: OutcomeFacts = field(default_factory=OutcomeFacts)
    content: ContentFacts | None = None
    operation: OperationFacts | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)
    #: The deployment's content decision for this request. A sink narrows it
    #: (never widens it) through :class:`~llm_proxy.observability.observation.policy.SinkPolicy`.
    policy: ContentPolicy = field(default_factory=ContentPolicy)


__all__ = [
    "ContentFacts",
    "CoreFacts",
    "IdentityFacts",
    "ObservationKind",
    "OperationFacts",
    "OutcomeFacts",
    "RequestObservation",
    "RoutingFacts",
    "TimingFacts",
    "UsageFacts",
]
