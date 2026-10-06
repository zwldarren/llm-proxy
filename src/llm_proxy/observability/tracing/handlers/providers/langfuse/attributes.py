"""Langfuse SDK-specific data builders.

These helpers convert internal request/response/context data into native Python
dicts that the Langfuse Python SDK accepts for ``input``, ``output``,
``usage_details``, ``cost_details``, and ``metadata``.
"""

import uuid
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import orjson

if TYPE_CHECKING:
    from llm_proxy.models import InternalRequest, InternalResponse
    from llm_proxy.models.params import GenerationParams
    from llm_proxy.observability.event_context import EventContext

from llm_proxy.observability.logger import get_logger


def _endpoint_display_name(endpoint: str | None) -> str | None:
    """Convert an HTTP endpoint path to a human-readable display name.

    Examples:
        "/v1/chat/completions" -> "chat completions"
        "/v1/messages" -> "messages"
        "/v1/responses" -> "responses"
        "/v1/embeddings" -> "embeddings"
    """
    if not endpoint:
        return None
    path = endpoint.strip("/")
    if path.startswith("v1/"):
        path = path[3:]
    return path.replace("/", " ").replace("_", " ") or None


def _serialize_value(value: Any, depth: int = 3) -> Any:
    """Recursively serialize a value into something JSON-safe."""
    if depth <= 0:
        return str(value) if not isinstance(value, str | int | float | bool | type(None)) else value
    if value is None:
        return None
    if isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list | tuple):
        return [_serialize_value(item, depth - 1) for item in value]
    if isinstance(value, dict):
        return {str(k): _serialize_value(v, depth - 1) for k, v in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        return _serialize_value(asdict(value), depth - 1)  # type: ignore[arg-type]
    return str(value)


def _message_to_dict(msg: Any) -> dict[str, Any]:
    """Convert an internal message-like object to a JSON-safe dict."""
    role = getattr(msg, "role", None)
    result: dict[str, Any] = {
        "role": role,
        "content": _serialize_value(getattr(msg, "content", None)),
        "name": getattr(msg, "name", None),
    }

    if role == "assistant":
        content = getattr(msg, "content", None)
        if content:
            tool_calls: list[dict[str, Any]] = []
            for block in content:
                block_type = getattr(block, "type", None)
                if block_type == "tool_use":
                    tool_input = getattr(block, "input", {})
                    if isinstance(tool_input, dict):
                        arguments = orjson.dumps(tool_input).decode()
                    else:
                        arguments = str(tool_input)
                    tool_calls.append(
                        {
                            "id": getattr(block, "id", None),
                            "type": "function",
                            "function": {
                                "name": getattr(block, "name", ""),
                                "arguments": arguments,
                            },
                        }
                    )
            if tool_calls:
                result["tool_calls"] = tool_calls

    return result


def _tool_to_dict(tool: Any) -> dict[str, Any] | None:
    """Convert a ToolDefinition-like object to an OpenAI-style tool dict.

    Produces ``{"type": "function", "function": {"name", "description", "parameters"}}``
    which is the shape the Langfuse UI renders as a tool definition.
    Returns ``None`` if the tool has no name.
    """
    name = getattr(tool, "name", None) or ""
    if not name:
        get_logger(__name__).warning("Tool definition with missing name skipped for Langfuse")
        return None
    description = getattr(tool, "description", None)
    parameters = getattr(tool, "parameters", None)
    if not parameters:
        # CustomTool / other variants may not expose a JSON schema; provide a
        # minimal valid schema so the entry is valid and the Langfuse UI
        # does not break.
        parameters = {"type": "object"}
    function: dict[str, Any] = {"name": name, "parameters": parameters}
    if description is not None:
        function["description"] = description
    return {"type": "function", "function": function}


def build_request_input_data(
    request: InternalRequest,
) -> list[dict[str, Any]] | dict[str, Any] | None:
    """Build the native input payload for a Langfuse generation.

    Matches the Langfuse OpenAI integration: when tools are present, returns a
    dict ``{"tools": [...], "messages": [...]}`` so the Langfuse UI can render
    the available tool definitions alongside the conversation. Without tools,
    returns just the conversation messages as a list. Returns ``None`` when no
    messages exist.
    """
    conversation = getattr(request, "conversation", None)
    if conversation is None:
        return None

    messages: list[dict[str, Any]] = []
    for sys_msg in getattr(conversation, "system_messages", []) or []:
        messages.append(_message_to_dict(sys_msg))
    for msg in getattr(conversation, "messages", []) or []:
        messages.append(_message_to_dict(msg))

    tools = getattr(request, "tools", None)
    if tools:
        tool_defs = [d for tool in tools if (d := _tool_to_dict(tool)) is not None]
        if not messages:
            # Tools without messages: still surface the tool definitions.
            return {"tools": tool_defs, "messages": []}
        return {"tools": tool_defs, "messages": messages}

    return messages if messages else None


def extract_tool_uses(output: list[Any] | None) -> list[dict[str, Any]]:
    """Extract tool-call invocations from response output blocks.

    Returns a list of ``{"id", "name", "input"}`` dicts for each ``tool_use``
    block so the handler can emit one Langfuse ``tool`` observation per call.
    """
    if not output:
        return []
    tool_uses: list[dict[str, Any]] = []
    for block in output:
        block_type = getattr(block, "type", None)
        is_tool_use_block = block_type == "tool_use" or (
            block_type is None and type(block).__name__ == "ToolUseBlock"
        )
        if is_tool_use_block:
            tool_uses.append(
                {
                    "id": getattr(block, "id", None) or str(uuid.uuid4()),
                    "name": getattr(block, "name", None) or "",
                    "input": getattr(block, "input", None) or {},
                }
            )
    return tool_uses


def _format_output_for_langfuse(output_blocks: list[Any]) -> dict[str, Any]:
    """Format output blocks for Langfuse, separating tool_calls from content.

    Langfuse expects tool_calls as a separate field in output, not embedded in
    content. This matches the OpenAI API response format that Langfuse's UI
    parses to display tool calls in the trace detail view.
    """
    content_parts: list[Any] = []
    tool_calls: list[dict[str, Any]] = []

    for block in output_blocks:
        block_type = getattr(block, "type", None) or type(block).__name__

        if block_type == "tool_use" or "ToolUseBlock" in str(type(block)):
            tool_input = getattr(block, "input", {})
            if isinstance(tool_input, dict):
                arguments = orjson.dumps(tool_input).decode()
            else:
                arguments = str(tool_input)

            tool_calls.append(
                {
                    "id": getattr(block, "id", None),
                    "type": "function",
                    "function": {
                        "name": getattr(block, "name", ""),
                        "arguments": arguments,
                    },
                }
            )
        elif block_type == "text" or "TextBlock" in str(type(block)):
            text = getattr(block, "text", "")
            if text:
                content_parts.append(text)
        elif block_type == "thinking" or "ThinkingBlock" in str(type(block)):
            thinking = getattr(block, "thinking", "")
            if thinking:
                content_parts.append({"type": "thinking", "thinking": thinking})
        else:
            content_parts.append(_serialize_value(block))

    result: dict[str, Any] = {"role": "assistant"}
    if content_parts:
        if len(content_parts) == 1 and isinstance(content_parts[0], str):
            result["content"] = content_parts[0]
        else:
            result["content"] = content_parts
    else:
        result["content"] = None

    if tool_calls:
        result["tool_calls"] = tool_calls

    return result


def build_response_output_data(response: InternalResponse) -> dict[str, Any] | None:
    """Build the native output payload for a Langfuse generation.

    Returns an OpenAI-compatible assistant message dict, or ``None`` when there
    is no output to record.
    """
    output = getattr(response, "output", None)
    if not output:
        return None

    formatted = _format_output_for_langfuse(output)
    finish_reason = getattr(response, "finish_reason", None)
    if finish_reason:
        formatted["finish_reason"] = finish_reason

    model = getattr(response, "model", None)
    if model:
        formatted["model"] = model

    return formatted


def build_usage_details(context: EventContext) -> dict[str, int] | None:
    """Build Langfuse ``usage_details`` as mutually exclusive buckets.

    Langfuse treats every key as a non-overlapping bucket: ``input`` must exclude
    ``input_*`` values and ``output`` must exclude ``output_*`` values, or the
    tokens are counted (and priced) twice. The proxy's own token counts are
    inclusive — ``prompt_tokens`` already contains cache reads/writes and audio
    tokens, ``completion_tokens`` already contains reasoning and audio tokens —
    so each detail is subtracted from ``input``/``output`` before being emitted
    under its own bucket (see the Langfuse token & cost tracking contract).

    Image tokens stay inside ``input``: Langfuse's image usage key varies by
    model definition, so carving them out could silently price them at zero.
    """
    if not context.has_token_data():
        return None

    # Canonical fields only (CONTEXT.md "Canonical usage record"): dialect
    # aliases are normalized into these at capture time.
    cache_read = context.cache_read_input_tokens
    cache_write = context.cache_creation_input_tokens
    audio_in = context.audio_input_tokens
    audio_out = context.audio_output_tokens
    reasoning = context.reasoning_tokens

    details: dict[str, int] = {}
    if context.prompt_tokens is not None:
        exclusive_input = context.prompt_tokens - (
            (cache_read or 0) + (cache_write or 0) + (audio_in or 0)
        )
        details["input"] = max(exclusive_input, 0)
    if context.completion_tokens is not None:
        exclusive_output = context.completion_tokens - ((audio_out or 0) + (reasoning or 0))
        details["output"] = max(exclusive_output, 0)
    if context.total_tokens is not None:
        details["total"] = context.total_tokens

    if cache_read:
        details["cache_read_input_tokens"] = cache_read
    if cache_write:
        details["cache_creation_input_tokens"] = cache_write
    if audio_in:
        details["input_audio_tokens"] = audio_in
    if audio_out:
        details["output_audio_tokens"] = audio_out
    if reasoning:
        details["output_reasoning_tokens"] = reasoning

    return details if details else None


def build_cost_details(context: EventContext) -> dict[str, float] | None:
    """Build Langfuse ``cost_details`` from cost data in the event context.

    Only ``total`` is emitted: the proxy computes a single USD total and does not
    split it per usage bucket. The provider-reported cost is informational (it is
    the same total when it wins cost precedence) and is carried in metadata
    instead of as a phantom cost bucket.
    """
    if context.cost_usd is None:
        return None
    return {"total": context.cost_usd}


def build_metadata(context: EventContext) -> dict[str, Any]:
    """Build Langfuse observation metadata from the event context."""
    metadata: dict[str, Any] = {
        "request_id": context.request_id,
        "trace_id": context.trace_id,
    }
    if context.provider:
        metadata["provider"] = context.provider
    if context.model:
        metadata["model"] = context.model
    if context.request_type is not None:
        metadata["request_type"] = context.request_type.value

    endpoint = context.metadata.get("endpoint")
    if endpoint:
        metadata["endpoint"] = endpoint
        endpoint_name = _endpoint_display_name(endpoint)
        if endpoint_name:
            metadata["endpoint_name"] = endpoint_name

    if context.is_streaming:
        metadata["stream"] = True
    if context.ttft_ms is not None:
        metadata["ttft_ms"] = context.ttft_ms
    if context.latency_ms is not None:
        metadata["latency_ms"] = round(context.latency_ms, 3)
    if context.error_message:
        metadata["error_message"] = context.error_message

    # Token/usage facts that are not emitted as ``usage_details`` buckets (see
    # ``build_usage_details``) stay visible here.
    if context.image_input_tokens is not None:
        metadata["image_input_tokens"] = context.image_input_tokens
    if context.cache_savings_usd is not None:
        metadata["cache_savings_usd"] = context.cache_savings_usd
    if context.provider_reported_cost is not None:
        metadata["provider_reported_cost"] = context.provider_reported_cost
    if context.provider_model_name:
        metadata["provider_model_name"] = context.provider_model_name

    return metadata


def build_model_parameters(params: GenerationParams | None) -> dict[str, Any] | None:
    """Build model parameters dict for Langfuse generation.

    Extracts common generation parameters (temperature, max_tokens, top_p,
    stop sequences, etc.) from the GenerationParams object.
    """
    if params is None:
        return None

    common = params.common
    if common is None:
        return None

    result: dict[str, Any] = {}
    for key in (
        "temperature",
        "top_p",
        "max_tokens",
        "stop",
        "frequency_penalty",
        "presence_penalty",
        "seed",
    ):
        val = getattr(common, key, None)
        if val is not None:
            result[key] = val

    return result if result else None
