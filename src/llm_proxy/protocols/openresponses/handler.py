"""OpenResponses protocol endpoint.

This module provides the OpenResponses protocol endpoint configuration
for OpenAI Responses API format.
"""

import contextvars
from typing import Any

from llm_proxy.observability.logger import get_logger
from llm_proxy.protocols.base import ProtocolEndpoint
from llm_proxy.protocols.openresponses.schemas import ResponsesRequest
from llm_proxy.protocols.openresponses.serializer import conversation_to_input_items
from llm_proxy.serialization.format_context import FormatContext

logger = get_logger(__name__)

_request_context: contextvars.ContextVar[FormatContext | None] = contextvars.ContextVar(
    "openresponses_format_context", default=None
)

_FORMAT_CONTEXT_FIELDS: tuple[str, ...] = (
    "instructions",
    "previous_response_id",
    "store",
    "metadata",
    "temperature",
    "top_p",
    "presence_penalty",
    "frequency_penalty",
    "truncation",
    "parallel_tool_calls",
    "max_output_tokens",
    "max_tool_calls",
    "reasoning",
    "service_tier",
    "text",
    "top_logprobs",
    "background",
    "safety_identifier",
    "prompt_cache_key",
    "include",
    "tool_choice",
)


def _collect_raw_tools(data: dict) -> list[dict]:
    """Collect raw tool dicts from the request for the FormatContext.

    Tools may arrive both in the top-level ``tools`` array and inside
    ``additional_tools`` input items (Codex sends all tools this way).
    Both sources are needed so downstream consumers (e.g. custom tool
    detection for ``custom_tool_call`` emission) see the complete set.
    """
    tools: list[dict] = [t for t in data.get("tools") or [] if isinstance(t, dict)]
    input_items = data.get("input")
    if isinstance(input_items, list):
        seen = {t.get("name") for t in tools}
        for item in input_items:
            if not isinstance(item, dict) or item.get("type") != "additional_tools":
                continue
            for tool in item.get("tools") or []:
                if isinstance(tool, dict) and tool.get("name") not in seen:
                    tools.append(tool)
                    seen.add(tool.get("name"))
    return tools


def set_format_context(data: dict) -> None:
    """Set the FormatContext for the current request from raw request data.

    Called by UnifiedProcessor before parsing to store context needed
    for response formatting.
    """
    kwargs: dict[str, Any] = {name: data.get(name) for name in _FORMAT_CONTEXT_FIELDS}
    # OpenAI parity: ``store`` defaults to true when omitted — the response is
    # persisted server-side unless the client explicitly opts out with
    # store=false. Normalizing here (rather than at each consumer) makes the
    # response echo and the persistence decision — which reads the echoed
    # body — both see the effective value.
    if kwargs.get("store") is None:
        kwargs["store"] = True
    kwargs["tools"] = _collect_raw_tools(data)
    _request_context.set(FormatContext(**kwargs))


def get_format_context() -> FormatContext:
    """Get the FormatContext for the current request."""
    return _request_context.get() or FormatContext()


def clear_format_context() -> None:
    """Clear the FormatContext after response formatting."""
    _request_context.set(None)
    from llm_proxy.providers.openai.client_headers import clear_client_headers

    clear_client_headers()


async def _capture_client_headers(request: Any, fastapi_request: Any) -> None:
    """Protocol middleware: capture Codex client headers for upstream passthrough.

    Runs before UnifiedProcessor; the OpenAI adapter merges the captured
    headers when building native Responses upstream requests (OAuth-style
    upstreams rely on these fingerprint headers to identify the client).
    """
    from llm_proxy.providers.openai.client_headers import capture_client_headers

    headers = getattr(fastapi_request, "headers", None)
    if headers is not None:
        capture_client_headers(headers)


def update_format_context(**kwargs: object) -> None:
    """Merge kwargs into the current FormatContext.

    Used after parsing to add fields (e.g. namespace_map) that are computed
    from the InternalRequest and not available in the raw request dict.
    """
    ctx = get_format_context()
    for k, v in kwargs.items():
        if v is not None:
            setattr(ctx, k, v)
    _request_context.set(ctx)


def _get_streaming_transformer():
    from llm_proxy.protocols.openresponses.streaming import (
        OpenResponsesStreamingTransformer,
    )

    return OpenResponsesStreamingTransformer


def _materialize_stored_input(
    response_data: dict[str, Any],
    unified_request: Any,
    raw_request_data: dict[str, Any],
) -> dict[str, Any]:
    """Attach the materialized conversation to a stored response body.

    The formatted response body does not include the request input, but stored
    responses must carry it so follow-up ``previous_response_id``
    continuations (and ``/v1/responses/compact``) can replay the full
    conversation. The conversation on the unified request is already
    materialized (previous input + previous output + new input) by the time
    this hook runs, so converting it back to items is faithful for multi-turn
    chains; the raw request input is the fallback.

    ``instructions`` is the response's echoed instructions value: the
    matching system message is excluded from the serialized input because
    continuations restore instructions from the response's own
    ``instructions`` field.

    Returns the body dict, attaching the input when there is something to
    attach.
    """
    instructions = response_data.get("instructions")
    conversation = getattr(unified_request, "conversation", None)
    if conversation is not None and conversation.messages:
        try:
            items = conversation_to_input_items(conversation, exclude_system_text=instructions)
            if items:
                response_data["input"] = items
                return response_data
        except Exception:
            logger.debug("Failed to materialize conversation input items", exc_info=True)
    if isinstance(raw_request_data, dict) and raw_request_data.get("input") is not None:
        response_data["input"] = raw_request_data["input"]
    return response_data


openresponses_protocol = ProtocolEndpoint(
    name="openresponses",
    # Path aliases: clients whose base_url is missing /v1 or double-writes
    # it ("{base}/v1" + "/v1/responses") still reach the endpoint.
    # create_protocol_router registers the handler for every path.
    paths=["/v1/responses", "/responses", "/v1/v1/responses"],
    request_model=ResponsesRequest,
    streaming_transformer=_get_streaming_transformer,
    tags=["responses"],
    middleware=[_capture_client_headers],
    on_parse_request=set_format_context,
    on_format_done=clear_format_context,
    on_response_store=_materialize_stored_input,
)


__all__ = [
    "openresponses_protocol",
    "set_format_context",
    "get_format_context",
    "clear_format_context",
    "update_format_context",
]
