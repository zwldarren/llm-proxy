"""Unified event context for data capture across logging and tracing.

This module provides a single context object that is created at the start
of request processing and passed through the entire lifecycle to all handlers.
It consolidates data capture that was previously duplicated between
logging middleware and tracing handlers.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from llm_proxy.core.request_type import RequestType
from llm_proxy.models.types import Usage
from llm_proxy.serialization.usage import copy_usage, merge_latest


def _is_count(value: Any) -> bool:
    """Whether a value is a count: an ``int``, and not a ``bool`` (which is one)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _monotonic(current: Any, incoming: Any) -> int | None:
    """The larger of two counts, with a non-count meaning "not reported".

    Usage within one turn is cumulative, so an observation can add to a counter
    but never take away from it (``PendingTerminalState.fold_usage`` states the
    same rule for the converted tier). Details arrive as untyped provider dicts,
    so a value that is not a count is dropped rather than compared: the accounting
    path must never raise on a malformed one, and billing ignored such a value
    anyway.
    """
    if not _is_count(incoming):
        return current if _is_count(current) else None
    if not _is_count(current):
        return incoming
    return max(current, incoming)


@dataclass
class EventContext:
    """Unified context for event capture across logging and tracing.

    Created at the start of request processing and passed through
    the entire lifecycle to all handlers. This eliminates duplicate
    data capture between logging middleware and tracing handlers.

    Lifecycle:
        1. Created in UnifiedProcessor with request metadata and sampling decision
        2. Populated during request processing (provider selection, usage, etc.)
        3. Passed to all TracingHandler methods via *_with_context methods
        4. Used by AuditLogHandler to write logs to database

    Ownership: this is the **observability capture** object — stages, adapters
    and transformers WRITE what should be recorded (usage, fallback attempts,
    errors, response body). It is never read for control-flow decisions; if
    you need to influence routing/processing, write :class:`PipelineState`
    (working state) or :class:`RequestContext` (services/hooks) instead.
    """

    # === Request metadata (immutable after creation) ===
    request_id: str
    trace_id: str
    model: str | None
    internal_model: str | None = None
    provider: str | None = None
    session_id: str | None = None
    # user_id: the user-facing identifier (e.g. username or display name) used in
    #          audit logs, request logs, and usage records for human-readable attribution.
    # auth_user_id: the internal numeric primary key from the users table, used for
    #               programmatic lookups and data isolation (e.g. filtering logs by user).
    # These are distinct: user_id is a string for display, auth_user_id is an int for FK.
    user_id: str | None = None
    auth_user_id: int | None = None
    request_type: RequestType = RequestType.CHAT
    log_type: str = "endpoint"  # LogType enum value
    is_api_endpoint: bool = False

    # === Sampling decision (made once at creation) ===
    should_capture_full_body: bool = True
    should_log_input_output: bool = True
    # The deployment's masking switch and the key names it masks. Captured here
    # so every sink (the log store, the tracing exporters) resolves the same
    # :class:`ContentPolicy` instead of re-reading configuration on its own.
    mask_sensitive_data: bool = True
    sensitive_keys: frozenset[str] = frozenset()
    # Raw SSE capture is opt-in (``logging.log_raw_stream``); ``x-log-full``
    # forces it for a single request. When off, the stream lifecycle assembles
    # ``assembled_response_body`` instead of buffering every chunk.
    should_capture_raw_stream: bool = False

    # === Timing data ===
    start_time: float = field(default_factory=time.perf_counter)
    start_timestamp: float = field(default_factory=time.time)
    first_chunk_time: datetime | None = None  # For TTFT calculation

    # === Accumulated usage (populated during processing) ===
    # One record for every fact a provider reported about the turn (ADR-0025):
    # counters, cache read/write, reasoning, the audio and image details, the
    # provider's own cost, the search calls it announced. ``update_usage`` is its
    # only writer, so each fact keeps the intake seam's merge rule instead of a
    # second, hand-maintained copy of the same number.
    usage: Usage | None = None
    # Two billable dimensions reach billing through more than one route, so they
    # stay one field here rather than moving into the record wholesale:
    # ``provider_reported_cost`` also arrives as a provider-info extra
    # (nanogpt_cost / openrouter_cost), and ``web_search_requests`` also counts the
    # searches the proxy executed itself (the web-search continuation).
    provider_reported_cost: float | None = None
    web_search_requests: int | None = None
    # === Proxy-computed billable dimensions ===
    # What the proxy itself did or derived — not provider facts, so not the
    # record's business; billing prices them beside it.
    images_generated: int | None = None
    tts_characters: int | None = None
    cache_savings_usd: float | None = None
    cost_usd: float | None = None

    # === HTTP layer data (for audit logging) ===
    request_headers: dict[str, Any] = field(default_factory=dict)
    request_body: Any = None
    response_headers: dict[str, Any] = field(default_factory=dict)
    response_body: Any = None
    # Non-streaming wire shape reassembled from the streaming transformer's
    # accumulated blocks. Set by the stream lifecycle when raw capture is off;
    # the audit handler stores it in place of the raw SSE text.
    assembled_response_body: dict[str, Any] | None = None
    response_status_code: int | None = None

    # === Error information ===
    error_message: str | None = None
    error_stack_trace: str | None = None
    error_details: dict[str, Any] = field(default_factory=dict)

    # === Client/Authentication metadata ===
    client_ip: str | None = None
    user_agent: str | None = None
    api_key_name: str | None = None
    auth_method: str | None = None

    # === Streaming state ===
    is_streaming: bool = False
    streaming_chunks_captured: int = 0
    streaming_body_bytes: bytearray = field(default_factory=bytearray)
    streaming_truncated: bool = False
    streaming_chunk_limit: int = 0
    max_response_body_bytes: int = 1024 * 1024  # 1MB default

    # === Transformer reference (for streaming usage extraction) ===
    transformer: Any = None

    # === Additional metadata ===
    metadata: dict[str, Any] = field(default_factory=dict)

    # === Provider model name (actual model sent to provider API) ===
    provider_model_name: str | None = None

    # === Fallback tracking ===
    fallback_attempts: list[dict[str, Any]] = field(default_factory=list)

    # Same-provider retry tracking
    retry_attempts: list[dict[str, Any]] = field(default_factory=list)

    def update_usage(self, usage: Usage | None) -> None:
        """Fold one observation of a turn's usage into the record.

        Usage arrives cumulatively: the native passthrough tier captures one
        streaming frame at a time, so a frame that reports only some facts must
        not withdraw the ones already seen, and a provider that repeats its
        terminal frame with fewer fields must not shrink the turn. The rule is
        :func:`~llm_proxy.serialization.usage.merge_latest` — the same one
        ``PendingTerminalState.fold_usage`` applies on the converted tier — so a
        later zero never clobbers an earlier non-zero, and a dialect with no wire
        total keeps the total its merged counters imply.

        The record is copied in: the caller may still hold (and log) the one it
        passed, and both merge rules write into the nested details blocks. Every
        observation is copied, not just the first — ``merge_latest`` adopts the
        source's nested details block when the record has none, so folding the
        caller's object in directly would leave the record sharing it.

        The two billable dimensions that have a second route (the provider-info
        cost extra, the proxy-executed search count) are mirrored onto their
        fields, which stay billing's single home for them.

        Args:
            usage: Usage object from InternalResponse or streaming transformer
        """
        if not isinstance(usage, Usage):
            # ``None``, and anything that is not a record at all: the accounting
            # path never raises on a value it cannot read, and a caller that hands
            # it something else is a type error the checker catches, not a reason to
            # fail a request.
            return

        if self.usage is None:
            self.usage = copy_usage(usage)
        else:
            merge_latest(self.usage, copy_usage(usage))

        cost = usage.provider_reported_cost
        if cost is not None and isinstance(cost, int | float) and cost > 0:
            self.provider_reported_cost = cost
        self.web_search_requests = _monotonic(self.web_search_requests, usage.web_search_requests)

    def update_provider_info(self, provider_info: dict[str, Any] | None) -> None:
        """Update provider-reported metadata from provider_info.

        Args:
            provider_info: Provider info dict from InternalResponse
        """
        if not provider_info:
            return

        # Extract provider-reported cost (e.g., NanoGPT's x_nanogpt_pricing.cost)
        provider_cost = provider_info.get("nanogpt_cost")
        if not (isinstance(provider_cost, int | float) and provider_cost > 0):
            provider_cost = provider_info.get("openrouter_cost")
        if (
            provider_cost is not None
            and isinstance(provider_cost, int | float)
            and provider_cost > 0
        ):
            self.provider_reported_cost = provider_cost

        # Extract web search request count from server_tool_use (Anthropic).
        # Same fact as the record's ``web_search_requests`` (which may also carry
        # the searches the proxy executed, added on top): a later observation can
        # only raise it, so a provider-info block that repeats a smaller count
        # cannot withdraw what the turn already reported.
        server_tool_use = provider_info.get("server_tool_use")
        if isinstance(server_tool_use, dict):
            ws_count = server_tool_use.get("web_search_requests")
            if isinstance(ws_count, int) and ws_count is not None:
                self.web_search_requests = _monotonic(self.web_search_requests, ws_count)

    @property
    def latency_ms(self) -> float:
        """Calculate latency from start time.

        Returns:
            Latency in milliseconds
        """
        return (time.perf_counter() - self.start_time) * 1000

    @property
    def ttft_ms(self) -> float | None:
        """Calculate Time To First Token if available.

        Returns:
            TTFT in milliseconds, or None if not available
        """
        if self.first_chunk_time is None:
            return None
        # Calculate from start_timestamp (unix) to first_chunk_time (datetime)
        return (self.first_chunk_time.timestamp() - self.start_timestamp) * 1000

    @property
    def request_type_display(self) -> str:
        """Get the request type as a string, defaulting to "chat" if not set.

        Returns:
            String representation of the request type
        """
        return str(self.request_type) if self.request_type is not None else "chat"

    def capture_streaming_chunk(self, chunk: str | bytes) -> bool:
        """Capture a streaming chunk for logging.

        Args:
            chunk: The chunk data to capture

        Returns:
            True if captured, False if truncated/skipped
        """
        if not self.should_capture_raw_stream:
            return False

        chunk_bytes = chunk.encode("utf-8") if isinstance(chunk, str) else chunk

        # Check if we've hit the limit
        if len(self.streaming_body_bytes) + len(chunk_bytes) > self.max_response_body_bytes:
            self.streaming_truncated = True
            return False

        # Check chunk limit
        if (
            self.streaming_chunk_limit > 0
            and self.streaming_chunks_captured >= self.streaming_chunk_limit
        ):
            self.streaming_truncated = True
            return False

        self.streaming_body_bytes.extend(chunk_bytes)
        self.streaming_chunks_captured += 1
        return True

    def get_streaming_body(self) -> bytes:
        """Get the captured streaming body.

        Returns:
            Captured bytes from streaming
        """
        return bytes(self.streaming_body_bytes)

    def has_token_data(self) -> bool:
        """Check if we have any token data.

        True once any observation reached the record, matching what the flat
        mirrors meant: an adapter reports the counters it saw, zeros included.

        Returns:
            True if any token counts are available
        """
        return self.usage is not None

    def has_billable_data(self) -> bool:
        """Check if any billable usage dimension has data.

        Covers token-based billing and unit-based billing (images generated,
        TTS characters, web search requests). A record's presence is enough on
        its own — it carries the audio duration too — so the record and the
        proxy-produced dimensions are the two things to test.

        Returns:
            True if any billable quantity is available
        """
        return self.has_token_data() or any(
            v is not None
            for v in [
                self.images_generated,
                self.tts_characters,
                self.web_search_requests,
            ]
        )
