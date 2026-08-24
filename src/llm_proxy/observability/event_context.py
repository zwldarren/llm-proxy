"""Unified event context for data capture across logging and tracing.

This module provides a single context object that is created at the start
of request processing and passed through the entire lifecycle to all handlers.
It consolidates data capture that was previously duplicated between
logging middleware and tracing handlers.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llm_proxy.models.types import Usage

from llm_proxy.core.request_type import RequestType


def _get_attr_or_item(obj: Any, key: str) -> Any:
    """Get attribute or dict key from an object that may be either."""
    if hasattr(obj, key):
        return getattr(obj, key)
    if isinstance(obj, dict):
        return obj.get(key)
    return None


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

    # === Timing data ===
    start_time: float = field(default_factory=time.perf_counter)
    start_timestamp: float = field(default_factory=time.time)
    first_chunk_time: datetime | None = None  # For TTFT calculation

    # === Accumulated token/cost data (populated during processing) ===
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    cached_prompt_tokens: int | None = None
    # OpenAI-dialect cache-write tokens (prompt_tokens_details.cache_write_tokens),
    # carried to the billing seam as-is — not a DB column. The flat-vs-nested
    # tolerance lives in extract_tokens_from_usage, not here (ADR-0006).
    cache_write_tokens: int | None = None
    cache_savings_usd: float | None = None
    cost_usd: float | None = None
    # Cost reported by provider (e.g., NanoGPT x_nanogpt_pricing.cost)
    provider_reported_cost: float | None = None
    audio_input_tokens: int | None = None
    audio_output_tokens: int | None = None
    reasoning_tokens: int | None = None
    image_input_tokens: int | None = None
    # === Non-token billable dimensions ===
    images_generated: int | None = None
    audio_duration_seconds: float | None = None
    tts_characters: int | None = None
    web_search_requests: int | None = None

    # === HTTP layer data (for audit logging) ===
    request_headers: dict[str, Any] = field(default_factory=dict)
    request_body: Any = None
    response_headers: dict[str, Any] = field(default_factory=dict)
    response_body: Any = None
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
        """Update token counts from Usage object.

        Args:
            usage: Usage object from InternalResponse or streaming transformer
        """
        if usage is None:
            return

        self.prompt_tokens = usage.input_tokens
        self.completion_tokens = usage.output_tokens
        self.total_tokens = usage.total_tokens

        # Anthropic-style cache tokens (flat on Usage)
        if usage.cache_read_input_tokens is not None:
            self.cache_read_input_tokens = usage.cache_read_input_tokens
        self.cache_creation_input_tokens = usage.cache_creation_input_tokens

        if hasattr(usage, "reasoning_tokens"):
            self.reasoning_tokens = usage.reasoning_tokens

        # Handle prompt tokens details (supports both dataclass and dict formats)
        ptd = getattr(usage, "prompt_tokens_details", None)
        if ptd:
            cached = _get_attr_or_item(ptd, "cached_tokens")
            audio_in = _get_attr_or_item(ptd, "audio_tokens")
            image_in = _get_attr_or_item(ptd, "image_tokens")
            cache_write = _get_attr_or_item(ptd, "cache_write_tokens")
            if cached is not None:
                self.cached_prompt_tokens = cached
            if audio_in is not None:
                self.audio_input_tokens = audio_in
            if image_in is not None:
                self.image_input_tokens = image_in
            # Kept in the dialect field; mapped at the billing seam (ADR-0006).
            if cache_write is not None:
                self.cache_write_tokens = cache_write

        # Handle gpt-image style input_tokens_details (image tokens in prompt)
        # Only fall back to input_tokens_details if prompt_tokens_details
        # didn't provide image tokens
        if self.image_input_tokens is None:
            itd = getattr(usage, "input_tokens_details", None)
            if itd:
                image_in = _get_attr_or_item(itd, "image_tokens")
                if image_in is not None:
                    self.image_input_tokens = image_in

        # Non-token billable dimensions reported by providers (e.g. STT duration)
        duration = getattr(usage, "audio_duration_seconds", None)
        if duration is not None:
            self.audio_duration_seconds = duration

        # Web search request count (from StreamingUsage or Usage)
        ws = getattr(usage, "web_search_requests", None)
        if ws is not None:
            self.web_search_requests = ws

        # Handle completion tokens details (supports both dataclass and dict formats)
        ctd = getattr(usage, "completion_tokens_details", None)
        if ctd:
            audio_out = _get_attr_or_item(ctd, "audio_tokens")
            if audio_out is not None:
                self.audio_output_tokens = audio_out

        # Handle provider-reported cost (from StreamingUsage)
        cost = getattr(usage, "provider_reported_cost", None)
        if cost is not None and isinstance(cost, int | float) and cost > 0:
            self.provider_reported_cost = cost

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

        # Extract web search request count from server_tool_use (Anthropic)
        server_tool_use = provider_info.get("server_tool_use")
        if isinstance(server_tool_use, dict):
            ws_count = server_tool_use.get("web_search_requests")
            if isinstance(ws_count, int) and ws_count is not None:
                self.web_search_requests = ws_count

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
        if not self.should_capture_full_body:
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

        Returns:
            True if any token counts are available
        """
        return any(
            v is not None
            for v in [
                self.prompt_tokens,
                self.completion_tokens,
                self.total_tokens,
            ]
        )

    def has_billable_data(self) -> bool:
        """Check if any billable usage dimension has data.

        Covers token-based billing and unit-based billing (images generated,
        audio duration, TTS characters, web search requests).

        Returns:
            True if any billable quantity is available
        """
        return self.has_token_data() or any(
            v is not None
            for v in [
                self.images_generated,
                self.audio_duration_seconds,
                self.tts_characters,
                self.web_search_requests,
            ]
        )

    def to_usage_dict(self) -> dict[str, Any]:
        """Convert to a dictionary suitable for cost calculation.

        Produces the nested format required by extract_tokens_from_usage():
        - cache_creation_input_tokens / cache_read_input_tokens (flat, Anthropic)
        - prompt_tokens_details.cached_tokens (OpenAI cached tokens)
        - prompt_tokens_details.audio_tokens / completion_tokens_details.audio_tokens

        Returns:
            Dictionary with token counts
        """
        result: dict[str, Any] = {}

        if self.prompt_tokens is not None:
            result["prompt_tokens"] = self.prompt_tokens
        if self.completion_tokens is not None:
            result["completion_tokens"] = self.completion_tokens
        if self.total_tokens is not None:
            result["total_tokens"] = self.total_tokens

        # Anthropic-style cache fields (flat keys)
        if self.cache_creation_input_tokens is not None:
            result["cache_creation_input_tokens"] = self.cache_creation_input_tokens
        if self.cache_read_input_tokens is not None:
            result["cache_read_input_tokens"] = self.cache_read_input_tokens

        # OpenAI-style details (nested for extract_tokens_from_usage compatibility)
        prompt_details: dict[str, Any] = {}
        if self.cached_prompt_tokens is not None:
            prompt_details["cached_tokens"] = self.cached_prompt_tokens
        if self.cache_write_tokens is not None:
            prompt_details["cache_write_tokens"] = self.cache_write_tokens
        if self.audio_input_tokens is not None:
            prompt_details["audio_tokens"] = self.audio_input_tokens
        if self.image_input_tokens is not None:
            prompt_details["image_tokens"] = self.image_input_tokens
        if prompt_details:
            result["prompt_tokens_details"] = prompt_details

        completion_details: dict[str, Any] = {}
        if self.audio_output_tokens is not None:
            completion_details["audio_tokens"] = self.audio_output_tokens
        if completion_details:
            result["completion_tokens_details"] = completion_details

        # Non-token billable dimensions (flat keys)
        if self.images_generated is not None:
            result["images_generated"] = self.images_generated
        if self.audio_duration_seconds is not None:
            result["audio_duration_seconds"] = self.audio_duration_seconds
        if self.tts_characters is not None:
            result["tts_characters"] = self.tts_characters
        if self.web_search_requests is not None:
            result["web_search_requests"] = self.web_search_requests

        return result
