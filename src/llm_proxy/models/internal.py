# src/llm_proxy/models/internal.py
"""Internal request/response models for LLM Proxy."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from llm_proxy.models.content_blocks import AudioBlock, ContentBlock
from llm_proxy.models.content_blocks.extended import RefusalBlock, ThinkingBlock
from llm_proxy.models.conversation import ConversationContext
from llm_proxy.models.params import GenerationParams
from llm_proxy.models.tools import ToolChoiceSpec, ToolDefinition
from llm_proxy.models.types import (
    ChoiceLogprobs,
    ChoiceMetadata,
    ResponseStatus,
    StreamOptions,
    Usage,
)

if TYPE_CHECKING:
    pass


@dataclass
class RequestMetadata:
    """Request metadata for tracking and provider-specific extensions."""

    request_id: str | None = None
    user: str | None = None
    protocol_version: str | None = None
    protocol_name: str | None = None


class ConversionTier(StrEnum):
    """How a chat request's wire body was produced for the upstream.

    Stamped on ``InternalRequest.conversion_tier`` at the moment the outbound
    body is built (per attempt — retries/fallbacks re-stamp), and mirrored
    into ``EventContext.metadata`` for logs/audit. Purely observational; the
    verdict comes from ``llm_proxy.core.conversion.plan_conversion``.
    """

    #: Verbatim forwarding of the stashed raw protocol body (and native SSE
    #: stream) — see llm_proxy.core.conversion.
    NATIVE_PASSTHROUGH = "native_passthrough"
    #: Wire-compatible reuse: stashed raw body reused with model/stream
    #: rewritten and top-level None fields stripped (see prepare_wire_reuse_body).
    WIRE_REUSE = "wire_reuse"
    #: Full parse → InternalRequest → provider body rebuild.
    FULL_CONVERSION = "full_conversion"


@dataclass
class InternalRequest:
    """Unified internal request representation.

    This is the protocol-agnostic request format used internally by LLM Proxy.
    All protocol handlers parse their specific request formats into InternalRequest,
    and all provider adapters receive InternalRequest for execution.

    Attributes:
        request_type: The type of request - "chat", "embedding", or "image".
            Automatically set based on which unified model class is instantiated.
            Used by UnifiedProcessor to route to the correct handler.
        model: The model identifier to use (e.g., "gpt-5")
        conversation: The conversation context including messages and system prompt
        tools: Optional list of tool definitions available for the model
        tool_choice: Optional specification for how tools should be used
        params: Generation parameters (temperature, max_tokens, etc.)
        stream: Whether to stream the response
        stream_options: Optional streaming configuration
        request_id: Optional request identifier for tracking
        user: Optional user identifier
        metadata: Additional request metadata
        extra: Catch-all dict for passthrough data that has no explicit field.
            Usage patterns:
            - Unknown/unhandled protocol fields from parse_request
            - Provider-specific passthrough data (e.g., _system_blocks)
            - Multi-turn state (e.g., previous_response_id)
            - Fields forwarded to the provider request body
            Use sparingly — prefer adding explicit fields for well-known data.
            The _apply_field_policy() controls how extra fields are
            handled at the provider level.
    """

    request_type: str = field(default="chat", init=False)
    model: str
    conversation: ConversationContext
    tools: list[ToolDefinition] | None = None
    tool_choice: ToolChoiceSpec | None = None
    params: GenerationParams = field(default_factory=GenerationParams)

    stream: bool = False
    stream_options: StreamOptions | None = None

    n: int | None = None

    metadata: RequestMetadata = field(default_factory=RequestMetadata)

    extra: dict[str, Any] = field(default_factory=dict)

    # Set of top-level keys injected by parameter overrides.
    # These are automatically exempted from unknown_fields_policy stripping.
    _override_injected_keys: set[str] = field(default_factory=set, repr=False)
    _raw_protocol_data: dict[str, Any] | None = field(default=None, repr=False)
    # Unresolvable ``item_reference`` items from an OpenResponses request,
    # recorded as (message_index, ref_id) by the protocol serializer and
    # consumed by PreviousResponseResolutionStage after the stored previous
    # response has been materialized into the conversation.
    _unresolved_item_references: list[tuple[int, str]] | None = field(default=None, repr=False)
    # Namespace mapping from the OpenResponses request (flat name ->
    # [namespace, original_name]), set by the protocol serializer. Provider
    # serializers use it to flatten history tool-call names so they match the
    # flattened tool definitions sent upstream (models echo the history name).
    _namespace_map: dict[str, list[str]] | None = field(default=None, repr=False)

    # The model name the client actually requested (pre-override / pre-routing
    # alias, e.g. "fast"). Written by ProviderSelectionStage; read via the
    # ``echo_model`` property so the client always sees the name it asked for.
    user_facing_model: str | None = field(default=None, repr=False)

    # Raw-reuse lifecycle flags (see llm_proxy.core.conversion).
    # Written by pipeline stages; read by plan_conversion's verdicts.
    # native_request_disabled: a stage mutated the parsed request post-parse
    # (proxy-side web search interception, developer->system role
    # normalization), so the stashed raw body no longer reflects it and every
    # raw-reuse tier (native passthrough, wire reuse) must rebuild instead.
    native_request_disabled: bool = field(default=False, repr=False)
    # previous_response_materialized: the conversation was materialized from
    # the proxy's response store, so the body must be rebuilt (the upstream
    # cannot resolve proxy-local response/item ids).
    previous_response_materialized: bool = field(default=False, repr=False)

    # Which conversion tier produced the outbound body for the live attempt
    # (observability only; None until the body is built).
    conversion_tier: ConversionTier | None = field(default=None, repr=False)

    # Which conversion tier produced the upstream response for the live
    # attempt (observability only; None until the response is handled).
    # Stamped by the two response chokepoints: _build_passthrough_response
    # (NATIVE_PASSTHROUGH or WIRE_REUSE) and _parse_response (FULL_CONVERSION).
    response_tier: ConversionTier | None = field(default=None, repr=False)

    @property
    def request_id(self) -> str | None:
        """Request identifier delegating to metadata."""
        return self.metadata.request_id

    @request_id.setter
    def request_id(self, value: str | None) -> None:
        self.metadata.request_id = value

    @property
    def protocol_name(self) -> str | None:
        """Client protocol name, delegating to metadata."""
        return self.metadata.protocol_name

    @property
    def user(self) -> str | None:
        """User identifier delegating to metadata."""
        return self.metadata.user

    @user.setter
    def user(self, value: str | None) -> None:
        self.metadata.user = value

    @property
    def echo_model(self) -> str:
        """The model name echoed to the client.

        The client-requested alias (``user_facing_model``) when one was
        recorded by ProviderSelectionStage, otherwise the resolved provider
        model name. Every response echo point (streaming transformer, native
        passthrough injection, non-stream formatter) reads this single
        decision instead of re-implementing the fallback.
        """
        return self.user_facing_model or self.model


@dataclass
class InternalResponse:
    """Unified internal response representation.

    This is the protocol-agnostic response format used internally by LLM Proxy.
    All provider adapters convert their responses to InternalResponse,
    and all protocol handlers format InternalResponse into their specific format.

    Attributes:
        id: Response identifier (e.g., "resp_123", "chatcmpl-xxx")
        model: The model that generated the response
        output: List of content blocks in the response
        usage: Optional token usage information
        status: Response status (completed, incomplete, error)
        finish_reason: Optional reason for completion (e.g., "stop", "length")
        response_time_ms: Optional response time in milliseconds
        request_id: Optional request identifier for correlation
        provider_info: Optional provider-specific metadata
    """

    id: str
    model: str
    output: list[ContentBlock]

    usage: Usage | None = None

    status: ResponseStatus = "completed"
    finish_reason: str | None = None

    response_time_ms: float | None = None
    request_id: str | None = None
    provider_info: dict[str, Any] = field(default_factory=dict)

    logprobs: ChoiceLogprobs | None = None

    # Provider output items that have no internal ContentBlock equivalent
    # (e.g. Responses API local_shell_call / agent_message / compaction
    # items). Each entry is ``(position, item)`` where ``position`` is the
    # number of converted blocks that precede the raw item, so protocol
    # formatters can re-insert it at the correct spot and the round-trip
    # stays lossless for native upstreams.
    raw_output: list[tuple[int, dict[str, Any]]] | None = None

    # Multiple choices output (for n > 1). Each inner list is a choice's ContentBlock list.
    choices_outputs: list[list[ContentBlock]] = field(default_factory=list)
    # Per-choice metadata for n > 1 (finish_reason, logprobs, annotations).
    # Index 0 corresponds to the first additional choice (beyond self.output).
    choices_metadata: list[ChoiceMetadata] = field(default_factory=list)

    # Generic metadata usable by any protocol
    created_at: int | None = None  # Unix timestamp of response creation

    def get_thinking_content(self) -> str | None:
        """Extract thinking content from output blocks."""
        for block in self.output:
            if isinstance(block, ThinkingBlock):
                return block.thinking
        return None

    def get_refusal(self) -> str | None:
        """Extract refusal content from output blocks."""
        for block in self.output:
            if isinstance(block, RefusalBlock):
                return block.refusal
        return None

    def get_audio(self) -> dict[str, Any] | None:
        """Extract audio data from output blocks."""
        from dataclasses import asdict

        for block in self.output:
            if isinstance(block, AudioBlock):
                return asdict(block.source) if block.source else None
        return None
