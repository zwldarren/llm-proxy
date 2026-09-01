"""Conversion seam — the single home of the conversion-tier decision and of
raw-reuse body preparation.

A chat request reaches the upstream through exactly one of three tiers:

- ``NATIVE_PASSTHROUGH`` — the stashed raw protocol body (and the SSE stream)
  is forwarded verbatim, because client protocol and provider API are
  wire-identical;
- ``WIRE_REUSE`` — the wire-compatible rebuild shortcut: the stashed raw body
  is reused after a detached copy, with ``model``/``stream`` rewritten and
  top-level ``None`` fields stripped;
- ``FULL_CONVERSION`` — the canonical parse → InternalRequest → rebuild path.

The decision is made once, by :func:`plan_conversion`, which returns a
:class:`ConversionPlan` with three independent fields. The fields
legitimately disagree: a request whose conversation was materialized from the
proxy's response store must have its body rebuilt (the upstream cannot
resolve proxy-local ids), yet the upstream still speaks the same wire
protocol, so the *stream* may still be native. One verdict cannot express
"rebuilt request + native stream" — three fields can (ADR-0002, ADR-0011).

Inputs to the decision, all read here:

- adapter capability, declared as data: ``BaseAdapter.native_protocols``
  plus the request-scoped veto ``BaseAdapter.allows_native_request``;
- serializer wire compatibility, declared as data:
  ``ProviderSerializer.compatible_protocols``, carried into the decision by
  ``BuildContext.compatible_protocols``;
- request flags on ``InternalRequest``: ``native_request_disabled`` (a
  pipeline stage needs the parsed/rebuilt path, e.g. proxy-side web search
  interception) and ``previous_response_materialized`` (read by the veto);
- material availability: ``request._raw_protocol_data`` must be present for
  any raw-reuse tier — the verdict folds this in, so callers never re-check.

To disable raw reuse for a new feature, set ``request.native_request_disabled``
in a pipeline stage — the plan honors it for every tier on every side; no
other site needs editing.

Body preparation for both raw-reuse tiers lives here too
(:func:`prepare_native_body`, :func:`prepare_wire_reuse_body`), next to the
decision. Both return bodies fully detached from the stashed raw body:
downstream mutators (reasoning-field normalization, reasoning-echo
injection) write into nested message dicts in place, and the stash is the
same object as ``PipelineState.original_raw_data`` when no parameter
overrides ran — the fallback chain re-parses from it as the pristine client
body (ADR-0008). The stash is never handed out for in-place mutation
(ADR-0005, extended to the wire-reuse tier by ADR-0011).

The provider serializers' ``build_provider_request`` no longer decides
anything: it always performs a full conversion. Tier knowledge is not
duplicated there.

The :class:`NativePassthroughHandler` holds the bookkeeping that native
streams still require (usage capture for billing, response snapshots for
persistence, model-name injection) — passthrough-lifecycle knowledge, not
wire-shape conversion.
"""

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import orjson
from orjson import JSONDecodeError

from llm_proxy.core.adapter import BaseAdapter
from llm_proxy.core.reasoning_cache import try_cache_reasoning_from_responses_output
from llm_proxy.models import ConversionTier, InternalRequest
from llm_proxy.observability.event_context import EventContext
from llm_proxy.observability.logger import get_logger
from llm_proxy.streaming.sse_parse import iter_sse_data_events

if TYPE_CHECKING:
    from llm_proxy.serialization.context import BuildContext

logger = get_logger(__name__)


@dataclass(frozen=True)
class ConversionPlan:
    """The conversion seam's verdict for one request, per side.

    Fields:
        request_tier: How the outbound body is produced. ``None`` when the
            plan was computed without a ``BuildContext`` (stream/response-side
            callers): the request tier needs the serializer's declared
            ``compatible_protocols`` and the block policy, both carried by the
            context. Body-building callers always pass a context.
        stream_mode: ``NATIVE_PASSTHROUGH`` when the adapter yields
            protocol-native SSE for the client protocol, else
            ``FULL_CONVERSION`` (canonical dict → transformer round-trip).
            Never ``WIRE_REUSE``: streams have no rebuild shortcut.
        response_mode: ``NATIVE_PASSTHROUGH`` when the non-streaming response
            body is carried verbatim (``provider_info["_raw_response_body"]``),
            else ``FULL_CONVERSION`` (parsed into ``InternalResponse``).
    """

    request_tier: ConversionTier | None
    stream_mode: ConversionTier
    response_mode: ConversionTier


def plan_conversion(
    adapter: BaseAdapter,
    request: InternalRequest,
    context: BuildContext | None = None,
) -> ConversionPlan:
    """The single source of the conversion-tier decision, for all three sides.

    Reads adapter capability declarations, serializer wire compatibility (via
    ``context``), the request-scoped veto, the pipeline disable flag, and
    stash availability — everything that decides a tier lives in this
    function.

    ``response_mode`` follows the native request verdict (capability + veto +
    stash) with a wire-reuse tier below it: when the request body cannot be
    forwarded verbatim but the provider speaks the client's protocol, the raw
    response body still rides verbatim. This folds in the
    ``_raw_protocol_data`` presence check the adapters used to repeat at each
    call site; a stash-less request (e.g. rebuilt by the process_request
    middleware) now takes the wire-reuse/parsed response paths, which are
    the well-trodden defaults.
    """
    disabled = request.native_request_disabled
    protocol = request.protocol_name

    native_request = (
        not disabled
        and adapter.supports_native_request(protocol, request)
        and request._raw_protocol_data is not None
    )

    # Stream side: native streaming bypasses the canonical chunk transformer,
    # so it is incompatible with proxy-side features that rewrite content
    # mid-stream (web-search interception sets native_request_disabled). Note
    # the request-side veto (allows_native_request) does NOT apply here — a
    # locally materialized conversation forces a rebuilt request body, but
    # the upstream's response stream is still protocol-native.
    stream_mode = (
        ConversionTier.NATIVE_PASSTHROUGH
        if not disabled and adapter.supports_native_streaming(protocol or "")
        else ConversionTier.FULL_CONVERSION
    )
    if native_request:
        response_mode = ConversionTier.NATIVE_PASSTHROUGH
    elif (
        context is not None
        and not disabled
        and protocol is not None
        and protocol in context.compatible_protocols
        and context.response_passthrough
    ):
        # Wire-compatible response: the provider answers in the client's own
        # protocol, so the raw body rides verbatim (model aliasing and the
        # reasoning-field rename still run — the response-side mirror of
        # request WIRE_REUSE). ``native_request_disabled`` vetoes it: the
        # post-parse mutations that set the flag (web search, role
        # normalization) come with consumers that need a parsed
        # InternalResponse (e.g. the non-streaming web-search continuation).
        # The provider-metadata kill switch ``response_passthrough: false``
        # arrives via the context.
        response_mode = ConversionTier.WIRE_REUSE
    else:
        response_mode = ConversionTier.FULL_CONVERSION

    request_tier: ConversionTier | None = None
    if context is not None:
        if native_request:
            request_tier = ConversionTier.NATIVE_PASSTHROUGH
        elif (
            not disabled
            and request._raw_protocol_data is not None
            and protocol is not None
            and protocol in context.compatible_protocols
            # Only reuse the wire when the block policy is 'drop' (the
            # default): 'error'/'degrade' must run block validation and
            # degradation, which live on the rebuild path.
            and context.unsupported_block_policy == "drop"
        ):
            request_tier = ConversionTier.WIRE_REUSE
        else:
            request_tier = ConversionTier.FULL_CONVERSION

    return ConversionPlan(
        request_tier=request_tier,
        stream_mode=stream_mode,
        response_mode=response_mode,
    )


def prepare_native_body(
    adapter: BaseAdapter,
    request: InternalRequest,
    *,
    stream: bool | None = None,
) -> dict[str, Any]:
    """Build the native passthrough request body from the stashed raw protocol body.

    The single home of native body *preparation* (the decision lives in
    ``plan_conversion``). Shared preparation, in order:

    1. shallow-copy the stashed raw body, dropping top-level ``None`` values
       (a parameter override set to None means "delete the field", and strict
       upstreams reject explicit nulls — parity with the wire-compatible
       rebuild shortcut);
    2. substitute the routed upstream model id (``InternalRequest.model``) —
       the raw body still carries the client's model alias;
    3. set the ``stream`` flag when given;
    4. delegate to ``adapter.native_body_hook`` for family-specific repairs
       (Anthropic message normalization, OpenAI input-item id stripping).

    The stashed raw body is never mutated: the returned dict is a fresh copy,
    and hooks deep-copy nested structures before repairing them.

    Raises:
        ValueError: when ``request._raw_protocol_data`` is missing. The plan
            folds stash presence into its verdict, so reaching here without a
            stash means the caller bypassed the plan — fail loudly.
    """
    raw = request._raw_protocol_data
    if raw is None:
        raise ValueError(
            "prepare_native_body requires request._raw_protocol_data; "
            "native passthrough cannot forward a body that was never stashed."
        )
    # Stamp the conversion tier for observability: every native body flows
    # through this seam (ADR-0005), and callers only reach it after
    # plan_conversion returned NATIVE_PASSTHROUGH for the request side.
    request.conversion_tier = ConversionTier.NATIVE_PASSTHROUGH
    body = {k: v for k, v in raw.items() if v is not None}
    body["model"] = request.model
    if stream is not None:
        body["stream"] = stream
    return adapter.native_body_hook(body)


def prepare_wire_reuse_body(request: InternalRequest, context: BuildContext) -> dict[str, Any]:
    """Build the wire-compatible rebuild-shortcut body from the stashed raw body.

    The single home of wire-reuse body *preparation* (the decision lives in
    ``plan_conversion``). Preparation, in order:

    1. deep-copy the stashed raw body, dropping top-level ``None`` values —
       the copy is fully detached: downstream mutators (reasoning-field
       normalization, reasoning-echo injection) write into nested message
       dicts in place, and a shallow copy would let those edits reach the
       stash, which is the same object as ``PipelineState.original_raw_data``
       when no parameter overrides ran (the fallback chain re-parses from it
       as the pristine client body, ADR-0008);
    2. substitute the routed model id (``context.model``), falling back to
       the raw body's own ``model``;
    3. set the ``stream`` flag from the context.

    Field-policy enforcement (``unknown_fields_policy``) is NOT done here: the
    adapter's outbound chokepoint applies it to the returned body, exactly as
    for a rebuilt body. Family repairs (``native_body_hook``) do not apply
    either — they are native-tier knowledge.

    Raises:
        ValueError: when ``request._raw_protocol_data`` is missing. The plan
            only returns WIRE_REUSE when a stash is present, so reaching here
            without one means the caller bypassed the plan — fail loudly.
    """
    raw = request._raw_protocol_data
    if raw is None:
        raise ValueError(
            "prepare_wire_reuse_body requires request._raw_protocol_data; "
            "the wire-compatible rebuild shortcut reuses a body that was never stashed."
        )
    request.conversion_tier = ConversionTier.WIRE_REUSE
    body = copy.deepcopy({k: v for k, v in raw.items() if v is not None})
    body["model"] = context.model or body.get("model")
    body["stream"] = context.stream
    return body


class NativePassthroughHandler:
    """Bookkeeping required while forwarding protocol-native SSE to the client.

    When native streaming bypasses the canonical dict → transformer
    round-trip, the transformer cannot aggregate usage or capture the final
    response snapshot. This handler parses the native frames just enough to
    keep billing (token usage) and persistence (store=true snapshots) working,
    plus protocol-specific adjustments: the client-requested alias
    (``InternalRequest.user_facing_model``) is injected into Anthropic
    ``message_start`` frames and rewritten into OpenResponses terminal
    snapshots so native streams echo the same model name as the transformer
    path.
    """

    @staticmethod
    def inject_model_into_anthropic_message_start(frame: str, model: str) -> str:
        """Replace the model name inside an Anthropic message_start SSE frame.

        Only touches the ``message.model`` field inside the ``data:`` line,
        leaving every other native field intact.
        """
        for prefix in ("data: ", "data:"):
            idx = frame.find(prefix)
            if idx != -1:
                start = idx + len(prefix)
                end = frame.find("\n", start)
                data_str = frame[start:].strip() if end == -1 else frame[start:end].strip()
                if not data_str:
                    continue
                try:
                    payload = orjson.loads(data_str)
                except JSONDecodeError:
                    continue
                msg = payload.get("message")
                if isinstance(msg, dict):
                    msg["model"] = model
                    new_data = orjson.dumps(payload).decode()
                    return frame[:start] + new_data + frame[end:]
        return frame

    @staticmethod
    def maybe_capture_native_openresponses(
        chunk: str | dict[str, Any],
        transformer: Any,
        event_context: EventContext | None,
        model: str | None = None,
    ) -> str | dict[str, Any] | None:
        """Capture the final snapshot and usage from native Responses SSE frames.

        Native passthrough bypasses the streaming transformer, so the
        ``response.completed`` / ``response.incomplete`` / ``response.failed``
        snapshot is captured here into ``transformer.state.final_response_payload``
        (reused by the store=true persistence path) and its ``usage`` is
        written into the EventContext so cost calculation still works.

        Both ``event:``-annotated frames and bare ``data:`` frames are
        accepted: spec-compliant upstreams send the ``event:`` line, but some
        compatible Responses providers omit it, in which case the event type
        is read from the payload's own ``type`` field.

        When ``model`` is given, the terminal snapshot's ``model`` field is
        rewritten to it (set even when the upstream omits the field) — both
        in the stashed ``final_response_payload`` and in the emitted frame —
        so the native stream echoes the client-requested alias exactly like
        the transformer path does.

        Returns the (possibly rewritten) chunk, or None when the chunk was
            not modified.
        """
        if not isinstance(chunk, str):
            return None

        for event_type, parsed in iter_sse_data_events(chunk):
            effective_type = event_type or (
                parsed.get("type") if isinstance(parsed, dict) else None
            )
            if effective_type in (
                "response.completed",
                "response.incomplete",
                "response.failed",
            ):
                payload = parsed.get("response")
                if isinstance(payload, dict):
                    state = getattr(transformer, "state", None)
                    if state is not None and hasattr(state, "final_response_payload"):
                        state.final_response_payload = payload
                    NativePassthroughHandler._apply_openresponses_usage(
                        payload.get("usage"), event_context
                    )
                    # Reasoning cache: native streams bypass the
                    # transformer (whose accumulation feeds the cache on
                    # converted streams), so write it from the terminal
                    # snapshot instead. Never fatal.
                    try_cache_reasoning_from_responses_output(
                        payload.get("output"),
                        payload.get("id", "") or "",
                        logger_prefix="NativePassthrough",
                    )
                    if model:
                        payload["model"] = model
                        return NativePassthroughHandler._rewrite_openresponses_frame(chunk, parsed)
        return None

    @staticmethod
    def _rewrite_openresponses_frame(frame: str, payload: dict[str, Any]) -> str:
        """Replace the data payload inside an OpenResponses SSE frame.

        Only touches the ``data:`` line, leaving every other line (event
        annotations, trailing blank lines) intact. Returns the frame
        unchanged when no data line can be located.
        """
        new_data = orjson.dumps(payload).decode()
        for prefix in ("data: ", "data:"):
            idx = frame.find(prefix)
            if idx != -1:
                start = idx + len(prefix)
                end = frame.find("\n", start)
                if end == -1:
                    end = len(frame)
                return frame[:start] + new_data + frame[end:]
        return frame

    @staticmethod
    def _apply_openresponses_usage(usage: Any, event_context: EventContext | None) -> None:
        """Write a Responses API usage dict into the EventContext."""
        if event_context is None or not isinstance(usage, dict):
            return
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        if input_tokens is not None:
            event_context.prompt_tokens = input_tokens
        if output_tokens is not None:
            event_context.completion_tokens = output_tokens
        total = usage.get("total_tokens")
        if total is not None:
            event_context.total_tokens = total
        elif input_tokens is not None and output_tokens is not None:
            event_context.total_tokens = input_tokens + output_tokens
        input_details = usage.get("input_tokens_details")
        if isinstance(input_details, dict) and input_details.get("cached_tokens") is not None:
            event_context.cache_read_input_tokens = input_details.get("cached_tokens")
        output_details = usage.get("output_tokens_details")
        if isinstance(output_details, dict) and output_details.get("reasoning_tokens") is not None:
            event_context.reasoning_tokens = output_details.get("reasoning_tokens")

    @staticmethod
    def process_anthropic_usage_event(
        event_type: str, data: dict[str, Any], event_context: EventContext
    ) -> None:
        """Update EventContext token counts from a single Anthropic usage event."""
        usage: dict[str, Any] | None = None
        if event_type == "message_start":
            message = data.get("message", {})
            usage = message.get("usage")
        elif event_type == "message_delta":
            usage = data.get("usage")

        if not usage:
            return

        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        cache_read = usage.get("cache_read_input_tokens")
        cache_create = usage.get("cache_creation_input_tokens")

        if input_tokens is not None:
            event_context.prompt_tokens = input_tokens + (cache_read or 0) + (cache_create or 0)
        if output_tokens is not None:
            event_context.completion_tokens = output_tokens
        if cache_read is not None:
            event_context.cache_read_input_tokens = cache_read
        if cache_create is not None:
            event_context.cache_creation_input_tokens = cache_create

        # Web search request count for billing (server_tool_use.web_search_requests)
        server_tool_use = usage.get("server_tool_use")
        if isinstance(server_tool_use, dict):
            ws_count = server_tool_use.get("web_search_requests")
            if isinstance(ws_count, int) and ws_count > 0:
                event_context.web_search_requests = ws_count

        # Keep total_tokens in sync only when both values are known.
        if event_context.prompt_tokens is not None and event_context.completion_tokens is not None:
            event_context.total_tokens = (
                event_context.prompt_tokens + event_context.completion_tokens
            )

    @staticmethod
    def maybe_capture_native_streaming_usage(
        chunk: str | dict[str, Any], event_context: EventContext | None
    ) -> None:
        """Extract usage from native Anthropic SSE frames and update EventContext.

        Anthropic native streaming emits usage inside ``message_start`` and
        ``message_delta`` events. Because native passthrough bypasses the
        streaming transformer, the transformer cannot aggregate usage for us.
        We parse those frames directly so cost calculation still works.

        A single chunk may contain multiple SSE events; each complete event is
        processed immediately so that usage from earlier frames is not lost.
        """
        if event_context is None:
            return

        if not isinstance(chunk, str) or "event:" not in chunk:
            return

        for event_type, parsed in iter_sse_data_events(chunk):
            if event_type in ("message_start", "message_delta"):
                NativePassthroughHandler.process_anthropic_usage_event(
                    event_type, parsed, event_context
                )


__all__ = [
    "ConversionPlan",
    "NativePassthroughHandler",
    "plan_conversion",
    "prepare_native_body",
    "prepare_wire_reuse_body",
]
