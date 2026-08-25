"""OpenResponses protocol serializer.

Converts between OpenAI Responses API wire format and InternalRequest/InternalResponse.
Uses the same conversion logic that was previously in
protocols/openresponses/converters.py, now properly registered as a ProtocolSerializer.
"""

import logging
import secrets
import time
from dataclasses import asdict
from typing import Any

import orjson

from llm_proxy.core.reasoning_cache import try_cache_reasoning_from_responses_output
from llm_proxy.core.thinking import thinking_config_from_reasoning_effort
from llm_proxy.core.utils import generate_response_id
from llm_proxy.models import (
    ContentBlock,
    ConversationContext,
    FunctionTool,
    InternalRequest,
    InternalResponse,
    Message,
    SystemMessage,
    TextBlock,
    ToolChoice,
    ToolChoiceFunction,
    ToolChoiceSpec,
    ToolDefinition,
)
from llm_proxy.models.content_blocks import (
    CustomToolUseBlock,
    DocumentBlock,
    FileBlock,
    ImageBlock,
    RedactedThinkingBlock,
    RefusalBlock,
    ServerToolUseBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from llm_proxy.models.content_blocks.anthropic_builtin import (
    BashCodeExecutionToolResultBlock,
    CodeExecutionToolResultBlock,
    TextEditorCodeExecutionToolResultBlock,
    ToolSearchToolResultBlock,
    WebFetchToolResultBlock,
    WebSearchToolResultBlock,
)
from llm_proxy.models.params import GenerationParams, OpenAISpecificParams
from llm_proxy.models.tools import is_web_search_tool_name
from llm_proxy.models.types import ResponseFormat
from llm_proxy.protocols.openresponses.schemas import ResponsesRequest
from llm_proxy.protocols.registry import register_protocol_serializer
from llm_proxy.protocols.serializer_base import ProtocolSerializer
from llm_proxy.routing.message_extract import function_call_output_to_text
from llm_proxy.serialization.format_context import FormatContext
from llm_proxy.serialization.responses_toolkit import (
    NamespaceMapping,
    _extract_reasoning_text,
    _extract_summary_text,
    generate_item_id,
    restore_tool_name,
)

logger = logging.getLogger(__name__)

# Audio format to media type mapping for input_audio content blocks
_AUDIO_FORMAT_TO_MEDIA_TYPE: dict[str, str] = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "ogg": "audio/ogg",
    "flac": "audio/flac",
    "webm": "audio/webm",
    "mp4": "audio/mp4",
}


def _convert_input_content(content: Any) -> list[ContentBlock]:
    if content is None:
        return [TextBlock(text="")]
    if isinstance(content, str):
        return [TextBlock(text=content)]
    if not isinstance(content, list):
        return [TextBlock(text=str(content))]

    result: list[ContentBlock] = []
    for part in content:
        if hasattr(part, "type"):
            part_dict = part.model_dump() if hasattr(part, "model_dump") else dict(part)
        elif isinstance(part, dict):
            part_dict = part
        else:
            continue

        part_type = part_dict.get("type")
        if part_type == "input_text":
            result.append(TextBlock(text=part_dict.get("text", "")))
        elif part_type == "output_text":
            # Assistant message content round-tripped from a prior response.
            # Must be recognized so assistant messages keep their text instead
            # of degrading to an empty TextBlock (which yields an assistant message
            # with no content/tool_calls that providers reject).
            result.append(TextBlock(text=part_dict.get("text", "")))
        elif part_type == "reasoning_text":
            # Reasoning content round-tripped from a prior reasoning item. Keep
            # it as a ThinkingBlock so it serializes back to reasoning_content.
            result.append(ThinkingBlock(thinking=part_dict.get("text", "")))
        elif part_type == "refusal":
            from llm_proxy.models import RefusalBlock

            result.append(RefusalBlock(refusal=part_dict.get("refusal", "")))
        elif part_type == "input_image":
            from llm_proxy.models.content_blocks import ImageBlock
            from llm_proxy.models.types import ImageSource

            image_url = part_dict.get("image_url", "")
            if image_url.startswith("data:"):
                data_part = image_url.split(";base64,", 1)
                media_type = data_part[0].replace("data:", "") if len(data_part) == 2 else None
                data = data_part[1] if len(data_part) == 2 else image_url
                result.append(
                    ImageBlock(
                        source=ImageSource(type="base64", data=data, media_type=media_type),
                        detail=part_dict.get("detail"),
                    )
                )
            else:
                result.append(
                    ImageBlock(
                        source=ImageSource(type="url", data=image_url, media_type=None),
                        detail=part_dict.get("detail"),
                    )
                )
        elif part_type == "input_file":
            from llm_proxy.models.content_blocks import FileBlock

            file_data = part_dict.get("file_data")
            file_id = part_dict.get("file_id")
            file_url = part_dict.get("file_url")
            if file_url and not file_data and not file_id:
                file_data = file_url
            result.append(
                FileBlock(
                    file_data=file_data,
                    file_id=file_id,
                    filename=part_dict.get("filename"),
                )
            )
        elif part_type == "input_video":
            result.append(TextBlock(text=part_dict.get("video_url", "")))
        elif part_type == "input_audio":
            from llm_proxy.models.content_blocks import AudioBlock
            from llm_proxy.models.types import AudioSource

            audio_data = part_dict.get("audio_data", "")
            audio_url = part_dict.get("audio_url", "")
            audio_format = part_dict.get("format", "wav")
            media_type = _AUDIO_FORMAT_TO_MEDIA_TYPE.get(audio_format, f"audio/{audio_format}")

            if audio_data:
                result.append(
                    AudioBlock(
                        source=AudioSource(type="base64", data=audio_data, media_type=media_type)
                    )
                )
            elif audio_url.startswith("data:"):
                data_part = audio_url.split(";base64,", 1)
                mime_type = data_part[0].replace("data:", "") if len(data_part) == 2 else None
                data = data_part[1] if len(data_part) == 2 else audio_url
                result.append(
                    AudioBlock(
                        source=AudioSource(
                            type="base64",
                            data=data,
                            media_type=mime_type or media_type,
                        )
                    )
                )
            else:
                result.append(
                    AudioBlock(
                        source=AudioSource(type="url", data=audio_url, media_type=media_type)
                    )
                )
    return result if result else [TextBlock(text="")]


def conversation_to_input_items(
    conversation: ConversationContext,
    *,
    exclude_system_text: str | None = None,
) -> list[dict[str, Any]]:
    """Convert a materialized conversation back into OpenResponses input items.

    Inverse of the input-item processing in ``_dispatch_input_item`` (and of
    ``replay_stored_response``), so the result can be stored on a response
    (``response.input``) and replayed faithfully by a follow-up request that
    references it via ``previous_response_id``.

    System/developer messages are serialized first (their natural position),
    then reasoning blocks become ``reasoning`` items, tool calls become
    ``function_call`` / ``custom_tool_call`` items, tool results become
    ``function_call_output`` items, and text/image/refusal content becomes a
    ``message`` item (keeping the assistant ``phase``). Content that has no
    round-trippable item shape (audio/video/file) is skipped.

    ``exclude_system_text``: skip the first system message whose text matches
    this value. Used by storage call sites to avoid serializing the
    request-level ``instructions`` — those are already carried by the
    response's own ``instructions`` field and restored from there on
    continuation, so emitting them as an input item too would duplicate them.
    """
    from llm_proxy.models import RefusalBlock
    from llm_proxy.models.content_blocks import ImageBlock, RedactedThinkingBlock

    items: list[dict[str, Any]] = []

    skipped_excluded = False
    for sys_msg in getattr(conversation, "system_messages", None) or []:
        text = sys_msg.text_content
        if not text:
            continue
        if exclude_system_text is not None and not skipped_excluded and text == exclude_system_text:
            skipped_excluded = True
            continue
        sys_item: dict[str, Any] = {
            "type": "message",
            "role": getattr(sys_msg, "role", "system") or "system",
            "content": text,
        }
        if getattr(sys_msg, "name", None):
            sys_item["name"] = sys_msg.name
        items.append(sys_item)

    for msg in conversation.messages:
        reasoning_items: list[dict[str, Any]] = []
        content_parts: list[dict[str, Any]] = []
        tool_call_items: list[dict[str, Any]] = []
        tool_output_items: list[dict[str, Any]] = []

        for block in msg.content:
            if isinstance(block, ThinkingBlock):
                reasoning_item: dict[str, Any] = {"type": "reasoning", "summary": []}
                if block.thinking:
                    reasoning_item["summary"] = [{"type": "summary_text", "text": block.thinking}]
                if block.encrypted_content:
                    reasoning_item["encrypted_content"] = block.encrypted_content
                if reasoning_item["summary"] or reasoning_item.get("encrypted_content"):
                    reasoning_items.append(reasoning_item)
            elif isinstance(block, RedactedThinkingBlock) and block.data:
                reasoning_items.append(
                    {
                        "type": "reasoning",
                        "summary": [{"type": "summary_text", "text": block.data}],
                    }
                )
            elif isinstance(block, ToolUseBlock):
                tool_call_items.append(
                    {
                        "type": "function_call",
                        "call_id": block.id,
                        "name": block.name,
                        "arguments": orjson.dumps(block.input).decode(),
                    }
                )
            elif isinstance(block, CustomToolUseBlock):
                tool_call_items.append(
                    {
                        "type": "custom_tool_call",
                        "call_id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )
            elif isinstance(block, ServerToolUseBlock):
                # Map server-side tools back to their native item types so the
                # round-trip keeps the original semantics (a custom_tool_call
                # would make the follow-up turn treat web search as an opaque
                # custom tool instead of a completed server-side search).
                if is_web_search_tool_name(block.name):
                    upstream_action = block.extra.get("responses_action") if block.extra else None
                    if not isinstance(upstream_action, dict):
                        query = block.input.get("query", "")
                        upstream_action = {
                            "type": "search",
                            "query": query,
                            "queries": [query] if query else [],
                        }
                    item = {
                        "type": "web_search_call",
                        "id": block.id,
                        "status": "completed",
                        "action": dict(upstream_action),
                    }
                elif block.name == "tool_search":
                    arguments = block.input.get("arguments", block.input)
                    item = {
                        "type": "tool_search_call",
                        "id": block.id,
                        "arguments": arguments if isinstance(arguments, dict) else {},
                    }
                else:
                    item = {
                        "type": "custom_tool_call",
                        "call_id": block.id,
                        "name": block.name,
                        "input": (
                            orjson.dumps(block.input).decode()
                            if isinstance(block.input, (dict, list))
                            else block.input
                        ),
                    }
                tool_call_items.append(item)
            elif isinstance(block, ToolResultBlock):
                output: Any = block.content
                if isinstance(output, list):
                    output = "\n".join(
                        part.text for part in output if isinstance(part, TextBlock) and part.text
                    )
                if output:
                    tool_output_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": block.tool_use_id,
                            "output": output,
                        }
                    )
            elif isinstance(block, TextBlock):
                content_parts.append({"type": "input_text", "text": block.text})
            elif isinstance(block, ImageBlock):
                source = block.source
                if source.type == "base64":
                    media = source.media_type or "image/png"
                    content_parts.append(
                        {
                            "type": "input_image",
                            "image_url": f"data:{media};base64,{source.data}",
                        }
                    )
                elif source.type == "url":
                    content_parts.append({"type": "input_image", "image_url": source.data})
            elif isinstance(block, RefusalBlock):
                content_parts.append({"type": "refusal", "refusal": block.refusal})

        items.extend(reasoning_items)
        if content_parts and msg.role in ("user", "assistant", "system", "developer"):
            message_item: dict[str, Any] = {
                "type": "message",
                "role": msg.role,
                "content": content_parts,
            }
            if msg.role == "assistant" and msg.phase in ("commentary", "final_answer"):
                message_item["phase"] = msg.phase
            items.append(message_item)
        items.extend(tool_call_items)
        items.extend(tool_output_items)
    return items


def _parse_tool_arguments(args: Any) -> dict[str, Any]:
    """Parse a Responses API ``function_call`` arguments value into a dict.

    ``ToolUseBlock.input`` is typed as ``dict[str, Any]``. Storing the raw
    JSON string would cause the OpenAI Chat Completions converter to double-
    encode it via ``orjson.dumps(block.input)`` (a JSON string literal instead
    of a JSON object), so the upstream model receives garbage tool arguments.

    Mirrors the Chat Completions request parser and the streaming transformer:
    parse to a dict, falling back to ``{}`` on invalid JSON and wrapping
    non-object JSON values under a ``"value"`` key.
    """
    if not args:
        return {}
    if isinstance(args, dict):
        return args
    if not isinstance(args, str):
        return {"value": args}
    try:
        parsed = orjson.loads(args)
    except orjson.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _parse_tool(tool: dict[str, Any]) -> ToolDefinition:
    src = tool.get("function", tool)
    return FunctionTool(
        name=src.get("name", ""),
        description=src.get("description"),
        parameters=src.get("parameters", {"type": "object"}),
        strict=tool.get("strict", False),
    )


def _parse_tool_choice(choice: Any) -> ToolChoiceSpec | None:
    if choice is None:
        return None
    if isinstance(choice, str):
        return ToolChoice(mode=choice)
    # The ResponsesRequest schema parses ``tool_choice`` into pydantic models
    # (FunctionToolChoice / AllowedToolChoice). Normalize to a plain dict so the
    # dict handling below applies uniformly to both pydantic objects and raw
    # dicts coming from other call sites.
    if not isinstance(choice, dict):
        model_dump = getattr(choice, "model_dump", None)
        if callable(model_dump):
            choice = model_dump()
        else:
            return None
    if isinstance(choice, dict):
        tc_type = choice.get("type")
        if tc_type == "function":
            # Accept both the Responses shape ({type, name}) and the chat
            # completions shape ({type, function: {name}}).
            name = choice.get("name") or choice.get("function", {}).get("name", "")
            return ToolChoiceFunction(name=name)
        if tc_type == "custom":
            name = choice.get("name", "")
            if name:
                from llm_proxy.models.tools import ToolChoiceCustom

                return ToolChoiceCustom(name=name)
        if tc_type == "allowed_tools":
            mode = choice.get("mode", "auto")
            tools = choice.get("tools", []) or []
            if mode in ("auto", "none", "required") and not tools:
                return ToolChoice(mode=mode)
            if tools and len(tools) == 1:
                single = tools[0]
                single_name = (
                    single.get("name", "")
                    if isinstance(single, dict)
                    else getattr(single, "name", "")
                )
                return ToolChoiceFunction(name=single_name)
            from llm_proxy.models.tools import AllowedToolsConfig, ToolChoiceAllowedTools

            return ToolChoiceAllowedTools(
                allowed_tools=AllowedToolsConfig(mode=mode or "auto", tools=tools)
            )
    return None


def _tool_call_like_item_to_use_block(
    item_dict: dict[str, Any],
) -> ToolUseBlock | CustomToolUseBlock | None:
    """Map a tool-call-like Responses input item to a ToolUseBlock.

    Covers custom_tool_call, local_shell_call and tool_search_call. function_call
    is handled inline by callers to preserve thought_signature passthrough.
    Returns None for unrelated types.
    """
    item_type = item_dict.get("type")
    call_id = item_dict.get("call_id") or generate_item_id()
    if item_type == "custom_tool_call":
        # Preserve the raw freeform input (e.g. JavaScript source for Codex's
        # exec tool) instead of parsing it as JSON arguments. Provider
        # serializers re-wrap it into the {"content": ...} envelope used by
        # the custom-tool function bridge.
        raw_input = item_dict.get("input", "")
        if not isinstance(raw_input, str):
            raw_input = orjson.dumps(raw_input).decode()
        return CustomToolUseBlock(
            id=call_id,
            name=item_dict.get("name", ""),
            input=raw_input,
        )
    if item_type == "local_shell_call":
        action = item_dict.get("action")
        return ToolUseBlock(
            id=call_id,
            name="local_shell",
            input=action if isinstance(action, dict) else {"type": "exec"},
        )
    if item_type == "tool_search_call":
        args = item_dict.get("arguments")
        return ToolUseBlock(
            id=call_id,
            name="tool_search",
            input=args if isinstance(args, dict) else {},
        )
    return None


def _tool_output_like_item_to_result_block(item_dict: dict[str, Any]) -> ToolResultBlock | None:
    """Map a tool-output-like Responses input item to a ToolResultBlock.

    Covers custom_tool_call_output, local_shell_call_output, and
    tool_search_output. function_call_output is handled inline by callers.
    Returns None for unrelated types.
    """
    item_type = item_dict.get("type")
    if item_type in ("custom_tool_call_output", "local_shell_call_output"):
        return ToolResultBlock(
            tool_use_id=item_dict.get("call_id", ""),
            content=function_call_output_to_text(item_dict.get("output", "")),
        )
    if item_type == "tool_search_output":
        tools = item_dict.get("tools")
        content = orjson.dumps(tools).decode() if tools else ""
        return ToolResultBlock(
            tool_use_id=item_dict.get("call_id") or generate_item_id(),
            content=content,
        )
    return None


def _agent_message_content_to_text(content: Any) -> str:
    """Extract readable text from an agent_message content array.

    Joins input_text items with newlines; encrypted_content items are skipped.
    Mirrors Codex plaintext_agent_message_content.
    """
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "input_text":
            text = part.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)


def _flush_pending_assistant(
    pending_assistant_blocks: list[ContentBlock],
    pending_web_search_calls: list[tuple[str, str]],
    messages: list[Message],
    phase: str | None = None,
) -> None:
    """Flush pending assistant blocks and web search placeholders into messages."""
    if pending_assistant_blocks:
        messages.append(
            Message(role="assistant", content=list(pending_assistant_blocks), phase=phase)
        )
        pending_assistant_blocks.clear()
    for ws_call_id, ws_query in pending_web_search_calls:
        placeholder = (
            (
                f'[Web search performed for "{ws_query}"; '
                "results not retained in conversation history.]"
            )
            if ws_query
            else ("[Web search performed; results not retained in conversation history.]")
        )
        messages.append(
            Message(
                role="tool",
                content=[ToolResultBlock(tool_use_id=ws_call_id, content=placeholder)],
            )
        )
    pending_web_search_calls.clear()


def _flush_pending_turn(
    pending_assistant_blocks: list[ContentBlock],
    pending_web_search_calls: list[tuple[str, str]],
    messages: list[Message],
    pending_phase: list[str | None],
) -> None:
    """Flush the pending assistant turn into messages and reset its phase."""
    _flush_pending_assistant(
        pending_assistant_blocks, pending_web_search_calls, messages, pending_phase[0]
    )
    pending_phase[0] = None


def _process_message_item(
    item_dict: dict[str, Any],
    messages: list[Message],
    system_messages: list[SystemMessage],
    pending_assistant_blocks: list[ContentBlock],
    pending_web_search_calls: list[tuple[str, str]],
    pending_phase: list[str | None],
) -> None:
    """Process a ``message`` input item."""
    role = item_dict.get("role", "")
    content = item_dict.get("content")
    if role == "system":
        _flush_pending_turn(
            pending_assistant_blocks, pending_web_search_calls, messages, pending_phase
        )
        if isinstance(content, str):
            system_content = content
        elif isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    parts.append(part.get("text", ""))
                elif hasattr(part, "text"):
                    parts.append(part.text)
            system_content = "\n".join(parts)
        else:
            system_content = str(content) if content else ""
        system_messages.append(SystemMessage.from_text(role="system", text=system_content))
        return
    content_blocks = _convert_input_content(content)
    if role == "assistant":
        phase = item_dict.get("phase")
        if phase in ("commentary", "final_answer"):
            # A phase change within one assistant turn means the turn is split
            # into distinct items (e.g. commentary then final_answer); flush the
            # pending blocks first so the phase boundary is preserved instead of
            # the items being merged with the last phase winning.
            if pending_phase[0] is not None and pending_phase[0] != phase:
                _flush_pending_turn(
                    pending_assistant_blocks, pending_web_search_calls, messages, pending_phase
                )
            pending_phase[0] = phase
        pending_assistant_blocks.extend(content_blocks)
    else:
        _flush_pending_turn(
            pending_assistant_blocks, pending_web_search_calls, messages, pending_phase
        )
        messages.append(Message(role=role, content=content_blocks))


def _process_function_call_item(
    item_dict: dict[str, Any],
    pending_assistant_blocks: list[ContentBlock],
) -> None:
    """Process a ``function_call`` input item."""
    pending_assistant_blocks.append(
        ToolUseBlock(
            id=item_dict.get("call_id", generate_item_id()),
            name=item_dict.get("name", ""),
            input=_parse_tool_arguments(item_dict.get("arguments", "{}")),
            extra={"thought_signature": item_dict.get("thought_signature")}
            if item_dict.get("thought_signature")
            else {},
        )
    )


def _process_tool_call_like_item(
    item_dict: dict[str, Any],
    pending_assistant_blocks: list[ContentBlock],
) -> None:
    """Process a custom_tool_call, local_shell_call, or tool_search_call input item."""
    use_block = _tool_call_like_item_to_use_block(item_dict)
    if use_block is not None:
        pending_assistant_blocks.append(use_block)


def _process_function_call_output_item(
    item_dict: dict[str, Any],
    messages: list[Message],
) -> None:
    """Process a ``function_call_output`` input item."""
    output = function_call_output_to_text(item_dict.get("output", ""))
    messages.append(
        Message(
            role="tool",
            content=[
                ToolResultBlock(
                    tool_use_id=item_dict.get("call_id", ""),
                    content=output,
                )
            ],
        )
    )


def _process_tool_output_like_item(
    item_dict: dict[str, Any],
    messages: list[Message],
) -> None:
    """Process a custom_tool_call_output or tool_search_output input item."""
    result_block = _tool_output_like_item_to_result_block(item_dict)
    if result_block is not None:
        messages.append(Message(role="tool", content=[result_block]))


def _process_reasoning_item(
    item_dict: dict[str, Any],
    pending_assistant_blocks: list[ContentBlock],
) -> None:
    """Process a ``reasoning`` input item."""
    thinking_text = _extract_reasoning_text(item_dict.get("content", []))
    if not thinking_text:
        thinking_text = _extract_summary_text(item_dict.get("summary", []))
    encrypted = item_dict.get("encrypted_content")
    if thinking_text or encrypted:
        pending_assistant_blocks.append(
            ThinkingBlock(thinking=thinking_text, encrypted_content=encrypted)
        )


def _process_compaction_item(
    item_dict: dict[str, Any],
    pending_assistant_blocks: list[ContentBlock],
) -> None:
    """Process a compaction, compaction_summary, or context_compaction input item."""
    encrypted = item_dict.get("encrypted_content")
    if encrypted:
        pending_assistant_blocks.append(ThinkingBlock(thinking="", encrypted_content=encrypted))


def _process_agent_message_item(
    item_dict: dict[str, Any],
    messages: list[Message],
) -> None:
    """Process an ``agent_message`` input item."""
    text = _agent_message_content_to_text(item_dict.get("content", []))
    if text:
        messages.append(Message(role="user", content=[TextBlock(text=text)]))


def _process_web_search_call_item(
    item_dict: dict[str, Any],
    pending_assistant_blocks: list[ContentBlock],
    pending_web_search_calls: list[tuple[str, str]],
) -> None:
    """Process a ``web_search_call`` input item."""
    action = item_dict.get("action") or {}
    ws_query = ""
    if isinstance(action, dict):
        ws_query = action.get("query") or ""
        if not ws_query:
            queries = action.get("queries")
            if isinstance(queries, list) and queries:
                first = queries[0]
                ws_query = str(first) if first else ""
    ws_call_id = item_dict.get("id") or generate_item_id()
    pending_assistant_blocks.append(
        ToolUseBlock(id=ws_call_id, name="web_search", input={"query": ws_query})
    )
    pending_web_search_calls.append((ws_call_id, ws_query))


def _process_additional_tools_item(
    item_dict: dict[str, Any],
    request: Any,
) -> None:
    """Process an ``additional_tools`` input item."""
    additional = item_dict.get("tools") or []
    if not additional:
        return
    if request.tools is None:
        request.tools = []
    existing_names = {t.get("name") for t in request.tools if isinstance(t, dict) and t.get("name")}
    new_tools: list[dict[str, Any]] = []
    for tool in additional:
        if isinstance(tool, dict) and tool.get("name") not in existing_names:
            new_tools.append(tool)
            existing_names.add(tool["name"])
    if new_tools:
        request.tools = [*request.tools, *new_tools]


def _dispatch_input_item(
    item_dict: dict[str, Any],
    messages: list[Message],
    system_messages: list[SystemMessage],
    pending_assistant_blocks: list[ContentBlock],
    pending_web_search_calls: list[tuple[str, str]],
    pending_phase: list[str | None],
    request: Any,
    seen_items: dict[str, dict[str, Any]],
    unresolved_refs: list[tuple[int, str]] | None = None,
) -> None:
    """Dispatch a single input item dict to its processor.

    Shared by the main input loop and compaction-blob rehydration so both
    paths handle every item type identically.

    ``unresolved_refs``: when provided, unresolvable ``item_reference`` items
    are recorded as ``(message_index, ref_id)`` pairs (the message boundary
    where the reference appeared) instead of being dropped, so the pipeline's
    previous-response resolution can splice the referenced items in after the
    stored previous response is materialized.
    """
    item_type = item_dict.get("type")

    if item_type == "item_reference":
        # Resolve against an item that appeared earlier in this input. When
        # unresolved_refs is given, unknown references are recorded for the
        # pipeline stage (they may point at items stored with a previous
        # response); otherwise they are skipped with a warning (the proxy has
        # no item-level store to hydrate them from).
        ref_id = item_dict.get("id") or ""
        referenced = seen_items.get(ref_id)
        if referenced is None:
            logger.warning("item_reference '%s' could not be resolved; skipping", ref_id)
            if unresolved_refs is not None:
                _flush_pending_turn(
                    pending_assistant_blocks,
                    pending_web_search_calls,
                    messages,
                    pending_phase,
                )
                unresolved_refs.append((len(messages), ref_id))
            return
        item_dict = referenced
        item_type = item_dict.get("type")

    if item_type == "message":
        _process_message_item(
            item_dict,
            messages,
            system_messages,
            pending_assistant_blocks,
            pending_web_search_calls,
            pending_phase,
        )
    elif item_type == "function_call":
        _process_function_call_item(item_dict, pending_assistant_blocks)
    elif item_type in ("custom_tool_call", "local_shell_call", "tool_search_call"):
        _process_tool_call_like_item(item_dict, pending_assistant_blocks)
    elif item_type == "function_call_output":
        _flush_pending_turn(
            pending_assistant_blocks, pending_web_search_calls, messages, pending_phase
        )
        _process_function_call_output_item(item_dict, messages)
    elif item_type in (
        "custom_tool_call_output",
        "local_shell_call_output",
        "tool_search_output",
    ):
        _flush_pending_turn(
            pending_assistant_blocks, pending_web_search_calls, messages, pending_phase
        )
        _process_tool_output_like_item(item_dict, messages)
    elif item_type == "reasoning":
        _process_reasoning_item(item_dict, pending_assistant_blocks)
    elif item_type in ("compaction", "compaction_summary", "context_compaction"):
        # Rehydrate proxy-produced compaction blobs (from /v1/responses/compact)
        # into their original conversation items; foreign blobs (e.g. Codex)
        # stay opaque encrypted content.
        from llm_proxy.protocols.openresponses.compaction import decode_compaction_blob

        rehydrated = decode_compaction_blob(item_dict.get("encrypted_content") or "")
        if rehydrated is not None:
            for sub_item in rehydrated:
                _dispatch_input_item(
                    sub_item,
                    messages,
                    system_messages,
                    pending_assistant_blocks,
                    pending_web_search_calls,
                    pending_phase,
                    request,
                    seen_items,
                    unresolved_refs,
                )
        else:
            _process_compaction_item(item_dict, pending_assistant_blocks)
    elif item_type == "agent_message":
        _flush_pending_turn(
            pending_assistant_blocks, pending_web_search_calls, messages, pending_phase
        )
        _process_agent_message_item(item_dict, messages)
    elif item_type == "web_search_call":
        _process_web_search_call_item(item_dict, pending_assistant_blocks, pending_web_search_calls)
    elif item_type == "additional_tools":
        _flush_pending_turn(
            pending_assistant_blocks, pending_web_search_calls, messages, pending_phase
        )
        _process_additional_tools_item(item_dict, request)

    # Index the item for later item_reference resolution.
    item_id = item_dict.get("id")
    if isinstance(item_id, str) and item_id:
        seen_items[item_id] = item_dict


def _convert_request_to_unified(
    request: Any, raw_text: dict[str, Any] | None = None
) -> InternalRequest:
    """Convert an OpenResponses request model to InternalRequest.

    ``raw_text`` is the original ``text`` dict from the wire request. It is
    passed separately because the Responses API ``text.format`` nested shape
    (``text: {format: {type, ...}}``) is coerced into the flat ``TextFormatParam``
    union by the Pydantic schema and loses the ``format`` key. Keeping the raw
    value lets us read both the real nested API shape and the legacy flat shape
    without losing the JSON schema. Used internally by ``parse_request``.
    """
    messages: list[Message] = []
    system_messages: list[SystemMessage] = []
    # ``item_reference`` entries that could not be resolved within this input,
    # recorded as (message_index, ref_id) so the pipeline can resolve them
    # against a stored previous response.
    unresolved_item_refs: list[tuple[int, str]] = []

    if request.instructions:
        system_messages.append(SystemMessage.from_text(role="system", text=request.instructions))

    if isinstance(request.input, str):
        messages.append(Message(role="user", content=[TextBlock(text=request.input)]))
    else:
        # Accumulate blocks from consecutive assistant-turn items (reasoning,
        # assistant message, function_call) and merge them into a single
        # assistant Message. Chat Completions represents one assistant turn as a
        # single message carrying reasoning_content + content + tool_calls
        # together, while the Responses API splits the same turn into separate
        # reasoning / message / function_call items. Merging here reconstructs the
        # canonical Chat Completions shape and avoids emitting assistant messages
        # that have neither content nor tool_calls — which providers reject and
        # surface as "Upstream request failed" on multi-turn tool-calling requests.
        pending_assistant_blocks: list[ContentBlock] = []
        pending_web_search_calls: list[tuple[str, str]] = []
        # Mutable holder for the phase of the assistant turn being accumulated.
        pending_phase: list[str | None] = [None]
        # Items already seen in this input, keyed by id, so ``item_reference``
        # entries can be resolved against them.
        seen_items: dict[str, dict[str, Any]] = {}

        for item in request.input:
            item_dict = item.model_dump() if hasattr(item, "model_dump") else dict(item)
            _dispatch_input_item(
                item_dict,
                messages,
                system_messages,
                pending_assistant_blocks,
                pending_web_search_calls,
                pending_phase,
                request,
                seen_items,
                unresolved_item_refs,
            )

        _flush_pending_assistant(
            pending_assistant_blocks, pending_web_search_calls, messages, pending_phase[0]
        )

    tools: list[ToolDefinition] | None = None
    namespace_mapping = None
    preserved_tools: list[dict[str, Any]] = []
    if request.tools:
        from llm_proxy.protocols.openresponses.tool_converter import (
            convert_responses_tools,
        )

        raw_tools = [t if isinstance(t, dict) else t.model_dump() for t in request.tools]
        tools, namespace_mapping, preserved_tools = convert_responses_tools(raw_tools)
        if not tools:
            tools = None

    tool_choice = _parse_tool_choice(request.tool_choice)

    openai_params = OpenAISpecificParams()
    has_openai_params = False

    for attr in (
        "store",
        "service_tier",
        "safety_identifier",
        "prompt_cache_key",
        "parallel_tool_calls",
    ):
        val = getattr(request, attr, None)
        if val is not None:
            setattr(openai_params, attr, val)
            has_openai_params = True
    if request.top_logprobs is not None:
        openai_params.top_logprobs = request.top_logprobs
        openai_params.logprobs = True
        has_openai_params = True
    if request.reasoning is not None and request.reasoning.effort is not None:
        openai_params.reasoning_effort = request.reasoning.effort
        has_openai_params = True
    if request.metadata:
        openai_params.metadata = request.metadata
        has_openai_params = True

    response_format = None
    # Prefer the raw ``text`` dict because the Responses API real shape nests
    # the format under ``text.format`` (``text: {format: {type, ...}}``), while
    # the Pydantic ``TextFormatParam`` union flattens to ``text: {type, ...}``
    # and drops the nested ``format`` key. Supporting both shapes keeps the
    # JSON schema intact and fixes the alias bug where ``JsonSchemaResponseFormat``
    # serializes its field as ``schema_`` unless ``by_alias=True`` is used.
    text_source = raw_text
    if text_source is None and request.text is not None:
        text_source = (
            request.text.model_dump(by_alias=True)
            if hasattr(request.text, "model_dump")
            else request.text
        )
    if isinstance(text_source, dict):
        # Real API: text.format; legacy flat: text itself.
        fmt_dict = text_source.get("format") or text_source
        if isinstance(fmt_dict, dict):
            fmt_type = fmt_dict.get("type", "text")
            json_schema: dict[str, Any] | None = None
            if fmt_type == "json_schema":
                # Internal convention (matching the Chat Completions parser)
                # stores the full wrapper {name, description, schema, strict}
                # in ResponseFormat.json_schema.
                wrapper: dict[str, Any] = {
                    k: v for k, v in fmt_dict.items() if k != "type" and v is not None
                }
                # Guard against model_dump() with alias disabled producing schema_.
                if "schema_" in wrapper and "schema" not in wrapper:
                    wrapper["schema"] = wrapper.pop("schema_")
                json_schema = wrapper or None
            response_format = ResponseFormat(type=fmt_type, json_schema=json_schema)

    thinking = None
    if request.reasoning is not None and request.reasoning.effort is not None:
        thinking = thinking_config_from_reasoning_effort(request.reasoning.effort)

    params = GenerationParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_output_tokens,
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
        response_format=response_format,
        openai=openai_params if has_openai_params else None,
        thinking=thinking,
    )

    extra_fields: dict[str, Any] = {}
    if request.previous_response_id is not None:
        extra_fields["previous_response_id"] = request.previous_response_id
    # Stateful / template fields the proxy cannot resolve locally are forwarded
    # verbatim to native Responses providers (new-api parity); the Chat
    # Completions request builder strips them via _RESPONSES_ONLY_EXTRA_KEYS.
    if request.conversation is not None:
        extra_fields["conversation"] = request.conversation
    if request.prompt is not None:
        extra_fields["prompt"] = request.prompt
    if request.background is not None:
        extra_fields["background"] = request.background
    if request.max_tool_calls is not None:
        extra_fields["max_tool_calls"] = request.max_tool_calls
    if request.truncation is not None:
        extra_fields["truncation"] = request.truncation
    if request.include is not None:
        extra_fields["include"] = request.include
    # Forward ``stream_options.include_obfuscation`` to providers that accept
    # it (OpenAI Responses). ``include_usage`` is deliberately NOT forwarded:
    # the Responses API rejects it with 400 "Unknown parameter" (usage is
    # always included in response.completed events).
    if (
        request.stream_options is not None
        and request.stream_options.include_obfuscation is not None
    ):
        extra_fields["stream_options"] = {
            "include_obfuscation": request.stream_options.include_obfuscation
        }
    if request.reasoning is not None:
        reasoning_dict = request.reasoning.model_dump(exclude_none=True)
        if reasoning_dict:
            extra_fields["reasoning"] = reasoning_dict

    # Forward the spec-shaped ``text`` config (``{format, verbosity}``) to
    # providers that accept it (OpenAI Responses). The flat legacy shape
    # (``text: {type, ...}``) is handled via ``response_format`` above and is
    # not forwarded.
    if isinstance(raw_text, dict) and ("format" in raw_text or "verbosity" in raw_text):
        text_extra = {k: v for k, v in raw_text.items() if v is not None}
        if text_extra:
            extra_fields["text"] = text_extra

    # Preserve Responses-only tools (file_search, code_interpreter, computer_use,
    # mcp, future built-ins) verbatim so that a native Responses provider can
    # forward them. Non-OpenAI providers reading InternalRequest.tools will not
    # see them and must log their own drop warning.
    if preserved_tools:
        extra_fields["responses_tools"] = preserved_tools

    # Preserve raw request fields the schema does not model (extra="allow"
    # accepts them without validation): context_management,
    # prompt_cache_options, prompt_cache_retention, moderation, and future
    # Responses API fields. A native Responses provider must receive them
    # verbatim (the rebuild path would otherwise silently drop them); the
    # Chat/Anthropic/Gemini/Ollama request builders strip the carrier key via
    # their Responses-only denylists.
    model_extra = getattr(request, "model_extra", None)
    if isinstance(model_extra, dict) and model_extra:
        extra_fields["responses_raw_fields"] = dict(model_extra)

    # Propagate namespace_map to FormatContext for response formatting
    # (stored here, not in extra_fields — FormatContext is the canonical place).
    if namespace_mapping is not None:
        from llm_proxy.protocols.openresponses.handler import update_format_context

        update_format_context(namespace_map=namespace_mapping.to_dict())

    req = InternalRequest(
        model=request.model,
        conversation=ConversationContext(system_messages=system_messages, messages=messages),
        tools=tools if tools else None,
        tool_choice=tool_choice,
        params=params,
        stream=request.stream,
        extra=extra_fields,
    )
    if namespace_mapping is not None:
        # Provider serializers flatten history tool-call names with this map so
        # they match the flattened tool definitions sent upstream.
        req._namespace_map = namespace_mapping.to_dict()
    if unresolved_item_refs:
        # Consumed by PreviousResponseResolutionStage once the referenced
        # previous response has been materialized into the conversation.
        req._unresolved_item_references = unresolved_item_refs
    if request.metadata and request.metadata.get("user"):
        req.user = request.metadata.get("user")
    return req


def _convert_usage(usage: dict[str, Any] | None) -> dict[str, Any]:
    if not usage:
        usage = {}

    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", input_tokens + output_tokens)

    # Spec: all five fields are required on ResponseResource.usage, including
    # both details objects (with their own required inner fields), so always
    # emit them with zero-value defaults when the provider did not supply them.
    input_details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details")
    result: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "input_tokens_details": {
            "cached_tokens": 0,
        },
    }
    if isinstance(input_details, dict):
        filtered = {k: v for k, v in input_details.items() if v is not None}
        result["input_tokens_details"].update(filtered)

    output_details = usage.get("completion_tokens_details") or usage.get("output_tokens_details")
    result["output_tokens_details"] = {
        "reasoning_tokens": 0,
    }
    if isinstance(output_details, dict):
        filtered = {k: v for k, v in output_details.items() if v is not None}
        result["output_tokens_details"].update(filtered)

    return result


def _content_blocks_to_text(blocks: list[Any]) -> str:
    """Convert a list of ContentBlocks to a plain text string."""
    parts: list[str] = []
    for b in blocks:
        if hasattr(b, "text"):
            parts.append(b.text)
        elif hasattr(b, "thinking"):
            parts.append(f"[thinking: {b.thinking}]")
        elif isinstance(b, dict):
            parts.append(b.get("text", b.get("title", str(b))))
        else:
            parts.append(str(b))
    return "\n".join(parts)


def web_search_declared_as_function(raw_tools: list[dict[str, Any]] | None) -> bool:
    """Whether the client declared ``web_search`` as a client-executed tool.

    Clients with a client-side search implementation (e.g. Hermes Agent)
    declare ``{"type": "function", "name": "web_search"}`` and expect the
    model's call back as a ``function_call`` item they execute themselves.
    Clients that declare the builtin ``{"type": "web_search"}`` expect the
    search to be executed server-side and reported as ``web_search_call``
    items.
    """
    if not raw_tools:
        return False
    for tool in raw_tools:
        if not isinstance(tool, dict):
            continue
        tool_type = tool.get("type")
        if tool_type in ("function", "custom") and tool.get("name") == "web_search":
            return True
        if tool_type == "namespace":
            for child in tool.get("tools") or []:
                if isinstance(child, dict) and child.get("name") == "web_search":
                    return True
    return False


def extract_custom_tool_names(raw_tools: list[dict[str, Any]]) -> set[str]:
    """Extract names of custom tools from raw request tool dicts.

    Used by ``format_response`` and the streaming transformer to detect when a
    ``function_call`` result should be emitted as a ``custom_tool_call`` item.

    Custom tools nested inside ``namespace`` tools are flattened with the same
    naming scheme used by ``convert_responses_tools`` so that provider-side
    function calls can be recognized and restored correctly.
    """
    names: set[str] = set()
    namespace_mapping: NamespaceMapping | None = None
    for tool in raw_tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "custom":
            name = tool.get("name", "")
            if name:
                names.add(name)
        elif tool.get("type") == "namespace":
            ns_name = tool.get("name", "")
            if not ns_name:
                continue
            if namespace_mapping is None:
                namespace_mapping = NamespaceMapping()
            for child in tool.get("tools") or []:
                if isinstance(child, dict) and child.get("type") == "custom":
                    child_name = child.get("name", "")
                    if child_name:
                        names.add(namespace_mapping.flatten(ns_name, child_name))
    return names


def match_custom_tool_name(name: str, custom_names: set[str]) -> str | None:
    """Match a tool name against custom tool names, tolerating short names.

    Providers may return the flattened name (``functions__exec``) or the
    original short name (``exec``) depending on what the model emitted —
    models often echo the short name from the conversation history. Returns
    the matched flattened name (for namespace restoration) or None.

    Suffix candidates are matched in sorted order so the result is
    deterministic when several namespaces declare the same short name
    (``a__exec`` / ``b__exec``); an ambiguous match is logged.
    """
    if name in custom_names:
        return name
    matches = sorted(flat for flat in custom_names if flat.endswith("__" + name))
    if len(matches) > 1:
        logger.warning(
            "Short tool name '%s' matches multiple custom tools %s; using '%s'",
            name,
            matches,
            matches[0],
        )
    return matches[0] if matches else None


def unwrap_custom_tool_arguments(arguments: str) -> str:
    """Extract the raw input string from JSON-wrapped function-tool arguments.

    The Anthropic bridge converts custom tools to function tools, so the model
    returns arguments in one of these shapes:

    - a plain JSON string (string input_schema, e.g. Codex ``exec`` raw JS)
    - ``{"input": "..."}`` (the object wrapper the bridge schema declares)
    - ``{"content": "..."}`` (legacy content-wrapped schema)
    - ``{"value": "..."}`` (legacy raw-input wrapper from history blocks)

    This extracts the raw string in each case. Any other shape (e.g. a native
    passthrough input like ``{"patch": "..."}`` that the client expects to
    receive verbatim) is returned unchanged.
    """
    if not arguments:
        return ""
    try:
        parsed = orjson.loads(arguments)
        if isinstance(parsed, str):
            return parsed
        if isinstance(parsed, dict):
            for key in ("input", "content", "value"):
                if key in parsed:
                    return str(parsed[key])
    except orjson.JSONDecodeError, TypeError, ValueError:
        pass
    return arguments


def _resolve_format_context(context: FormatContext | None) -> FormatContext:
    """Resolve the format context, falling back to the handler's context."""
    if context is None:
        try:
            from llm_proxy.protocols.openresponses.handler import get_format_context

            ctx = get_format_context()
            if ctx is not None:
                context = ctx
        except ImportError:
            logger.debug(
                "Could not import get_format_context from openresponses handler; "
                "proceeding without format context."
            )
    if context is None:
        context = FormatContext()
    return context


def _collect_web_search_results(output: list[Any]) -> dict[str, dict[str, Any]]:
    """First pass: collect web search results keyed by tool_use_id."""
    web_search_results: dict[str, dict[str, Any]] = {}
    for block in output:
        if isinstance(block, WebSearchToolResultBlock):
            sources: list[dict[str, str]] = []
            if isinstance(block.content, list):
                for item in block.content:
                    if isinstance(item, dict):
                        sources.append(
                            {
                                "url": str(item.get("url", "")),
                                "title": str(item.get("title", "")),
                            }
                        )
            web_search_results[block.tool_use_id] = {
                "sources": sources,
                "is_error": block.is_error,
            }
    return web_search_results


def _format_text_block(
    block: TextBlock, include: list[str] | None, response: InternalResponse
) -> dict[str, Any]:
    """Format a text block as a completed assistant message item."""
    annotations: list[dict[str, Any]] = []
    if block.citations:
        for citation in block.citations:
            if isinstance(citation, dict) and citation.get("type") == "url_citation":
                annotations.append(
                    {
                        "type": "url_citation",
                        "url": citation.get("url", ""),
                        "start_index": citation.get("start_index", 0),
                        "end_index": citation.get("end_index", 0),
                        "title": citation.get("title", ""),
                    }
                )
    text_part: dict[str, Any] = {
        "type": "output_text",
        "text": block.text,
        "annotations": annotations,
        "logprobs": [],
    }
    if include and "message.output_text.logprobs" in include:
        logprobs_list: list[dict[str, Any]] = []
        if response.logprobs and response.logprobs.content:
            logprobs_list = [
                {
                    "token": t.token,
                    "logprob": t.logprob,
                    "bytes": t.bytes,
                    "top_logprobs": [
                        {"token": x.token, "logprob": x.logprob, "bytes": x.bytes}
                        for x in (t.top_logprobs or [])
                    ],
                }
                for t in response.logprobs.content
            ]
        text_part["logprobs"] = logprobs_list
    return {
        "type": "message",
        "id": generate_item_id(),
        "status": "completed",
        "role": "assistant",
        "phase": "final_answer",
        "content": [text_part],
    }


def _format_thinking_block(
    block: ThinkingBlock | RedactedThinkingBlock, include: list[str] | None
) -> dict[str, Any]:
    """Format a thinking block as a reasoning item."""
    text = "[redacted]" if isinstance(block, RedactedThinkingBlock) else block.thinking
    reasoning_item: dict[str, Any] = {
        "type": "reasoning",
        "id": generate_item_id(),
        "status": "completed",
        "content": [],
        "summary": [{"type": "summary_text", "text": text}],
    }
    if include and "reasoning.encrypted_content" in include:
        encrypted = getattr(block, "encrypted_content", None)
        if encrypted:
            reasoning_item["encrypted_content"] = encrypted
    return reasoning_item


def _format_tool_use_block(
    block: ToolUseBlock | ServerToolUseBlock | CustomToolUseBlock,
    include: list[str] | None,
    web_search_results: dict[str, dict[str, Any]],
    context: FormatContext,
) -> dict[str, Any]:
    """Format a tool-use block as a function_call / web_search_call / tool_search_call item."""
    name = block.name
    if is_web_search_tool_name(name):
        args = block.input
        query = args.get("query", "") if isinstance(args, dict) else ""
        result_data = web_search_results.get(block.id)
        if not result_data and web_search_declared_as_function(context.tools):
            # The client declared web_search as a client-executed
            # function tool (e.g. Hermes Agent): emit a function_call
            # so it can run the search and return results. A
            # web_search_call would make such clients believe the
            # search already ran server-side and silently end the
            # turn.
            if isinstance(args, dict):
                args_str = orjson.dumps(args).decode()
            elif isinstance(args, str):
                args_str = args
            else:
                args_str = "{}"
            return {
                "type": "function_call",
                "id": generate_item_id(),
                "call_id": block.id,
                "name": name,
                "arguments": args_str,
                "status": "completed",
            }

        # The proxy interceptor executed the search (result_data) or
        # the client declared the builtin web_search tool: report a
        # completed web_search_call. When the upstream itself
        # executed the search, its full action (query/queries and,
        # when requested via include, sources) is re-emitted
        # verbatim so non-streaming native passthrough keeps the
        # sources.
        ws_item: dict[str, Any] = {
            "type": "web_search_call",
            "id": f"ws_{secrets.token_hex(12)}",
            "status": "completed",
        }
        upstream_action = None
        if isinstance(block, (ToolUseBlock, ServerToolUseBlock)) and block.extra:
            upstream_action = block.extra.get("responses_action")
        if isinstance(upstream_action, dict):
            ws_item["action"] = dict(upstream_action)
        else:
            ws_item["action"] = {
                "type": "search",
                "query": query,
                "queries": [query] if query else [],
            }
        if result_data:
            if include and "web_search_call.action.sources" in include:
                ws_item["action"]["sources"] = result_data["sources"]
            if include and "web_search_call.results" in include:
                ws_item["results"] = result_data["sources"]
        return ws_item

    if name == "tool_search":
        ts_args = block.input
        if isinstance(ts_args, str):
            try:
                ts_args = orjson.loads(ts_args)
            except orjson.JSONDecodeError:
                logger.warning(
                    "Failed to parse tool_search arguments as JSON: %s",
                    ts_args[:200],
                )
                ts_args = {}
        return {
            "type": "tool_search_call",
            "id": generate_item_id(),
            "call_id": block.id,
            "status": "completed",
            "execution": "client",
            "arguments": ts_args if isinstance(ts_args, dict) else {},
        }

    display_name, namespace = restore_tool_name(context.namespace_map, block.name)

    args = block.input
    if isinstance(args, dict):
        args = orjson.dumps(args).decode()

    # Preserve custom tool type when the original request defined
    # this name as a custom tool. Otherwise emit a normal function_call.
    # Tolerant matching: models often echo the short history name
    # (``exec``) rather than the flattened tool-definition name
    # (``functions__exec``) carried by the custom-name set.
    item_type = "function_call"
    if isinstance(block, CustomToolUseBlock) or (
        context.tools
        and match_custom_tool_name(name, extract_custom_tool_names(context.tools)) is not None
    ):
        item_type = "custom_tool_call"

    func_call_item: dict[str, Any] = {
        "type": item_type,
        "id": generate_item_id(),
        "call_id": block.id,
        "name": display_name,
        "status": "completed",
    }
    if namespace:
        func_call_item["namespace"] = namespace
    if item_type == "custom_tool_call":
        # Unwrap the JSON {"content": "..."} wrapper that the
        # function-tool conversion added, leaving only the raw text
        # that Codex expects as ``input``.
        func_call_item["input"] = unwrap_custom_tool_arguments(args)
    else:
        func_call_item["arguments"] = args
    if isinstance(block, ToolUseBlock) and block.extra.get("thought_signature"):
        func_call_item["thought_signature"] = block.extra["thought_signature"]
    return func_call_item


def _format_refusal_block(
    block: RefusalBlock, output: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Format a refusal block, extending the trailing message when possible."""
    msg_content: list[dict[str, Any]] = [{"type": "refusal", "refusal": block.refusal}]
    if output and output[-1]["type"] == "message":
        output[-1]["content"].extend(msg_content)
        return None
    return {
        "type": "message",
        "id": generate_item_id(),
        "status": "completed",
        "role": "assistant",
        "phase": "final_answer",
        "content": msg_content,
    }


def _format_media_block(block: ImageBlock | DocumentBlock | FileBlock) -> dict[str, Any]:
    """Format image/document/file blocks as placeholder assistant messages."""
    if isinstance(block, ImageBlock):
        text = f"[Image: {block.source.media_type or 'unknown'}]"
        if block.source.type == "file_id":
            text = f"[Image: file_id={block.source.data}]"
    elif isinstance(block, DocumentBlock):
        title = f" '{block.title}'" if block.title else ""
        text = f"[Document{title}: {block.source.media_type or 'unknown'}]"
    else:
        file_name = block.filename or block.file_id or "unknown"
        text = f"[File: {file_name}]"
    return {
        "type": "message",
        "id": generate_item_id(),
        "status": "completed",
        "role": "assistant",
        "phase": "final_answer",
        "content": [{"type": "output_text", "text": text}],
    }


def _format_tool_result_block(block: Any) -> dict[str, Any] | None:
    """Format tool result blocks as function_call_output / tool_search_output items."""
    # Skip web search results — they are already represented
    # as web_search_call items above
    if isinstance(block, WebSearchToolResultBlock):
        return None
    if isinstance(block, ToolSearchToolResultBlock):
        tools = block.content
        if isinstance(tools, str):
            try:
                tools = orjson.loads(tools)
            except orjson.JSONDecodeError:
                logger.warning(
                    "Failed to parse tool_search result as JSON: %s",
                    tools[:200],
                )
                tools = []
        return {
            "type": "tool_search_output",
            "id": generate_item_id(),
            "call_id": block.tool_use_id,
            "status": "completed",
            "execution": "client",
            "tools": tools,
        }
    content = block.content
    if isinstance(content, list):
        content = _content_blocks_to_text(content)
    elif not isinstance(content, str):
        content = str(content)
    return {
        "type": "function_call_output",
        "id": generate_item_id(),
        "call_id": block.tool_use_id,
        "output": content,
        "status": "completed",
    }


def _format_output_item(
    block: Any,
    output: list[dict[str, Any]],
    include: list[str] | None,
    response: InternalResponse,
    web_search_results: dict[str, dict[str, Any]],
    context: FormatContext,
) -> dict[str, Any] | None:
    """Convert one output ContentBlock to a Responses output item.

    Returns None when the block is already represented by an earlier item
    (web search results) or was merged into the trailing message (refusals).
    """
    if isinstance(block, TextBlock):
        return _format_text_block(block, include, response)
    if isinstance(block, (ThinkingBlock, RedactedThinkingBlock)):
        return _format_thinking_block(block, include)
    if isinstance(block, (ToolUseBlock, ServerToolUseBlock, CustomToolUseBlock)):
        return _format_tool_use_block(block, include, web_search_results, context)
    if isinstance(block, RefusalBlock):
        return _format_refusal_block(block, output)
    if isinstance(block, (ImageBlock, DocumentBlock, FileBlock)):
        return _format_media_block(block)
    if isinstance(
        block,
        (
            ToolResultBlock,
            WebSearchToolResultBlock,
            WebFetchToolResultBlock,
            CodeExecutionToolResultBlock,
            BashCodeExecutionToolResultBlock,
            TextEditorCodeExecutionToolResultBlock,
            ToolSearchToolResultBlock,
        ),
    ):
        return _format_tool_result_block(block)
    return None


def _build_status_and_error(
    response: InternalResponse,
) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None]:
    """Derive response status, error and incomplete_details from finish_reason."""
    status = "completed"
    error: dict[str, Any] | None = None
    incomplete_details: dict[str, Any] | None = None
    finish_reason = response.finish_reason
    if finish_reason == "length":
        status = "incomplete"
        # Preserve the upstream incomplete reason (max_output_tokens /
        # content_filter) when the provider supplied one; otherwise keep
        # the generic "length" reason.
        incomplete_reason = (response.provider_info or {}).get("incomplete_reason")
        incomplete_details = {"reason": incomplete_reason or "length"}
    elif finish_reason == "error":
        status = "failed"
        # Terminal-status validation: an HTTP 2xx response object whose
        # status is failed/cancelled must surface the upstream error
        # instead of masquerading as a completion. The upstream error
        # payload is carried through provider_info by the OpenAI provider
        # serializer.
        upstream_error = (response.provider_info or {}).get("upstream_error")
        if isinstance(upstream_error, dict) and (
            upstream_error.get("message") or upstream_error.get("code")
        ):
            error = {
                "code": upstream_error.get("code") or "provider_error",
                "message": upstream_error.get("message")
                or "Provider returned an error finish reason.",
                "type": upstream_error.get("type") or "provider_error",
                "param": upstream_error.get("param"),
            }
        else:
            error = {
                "code": "provider_error",
                "message": "Provider returned an error finish reason.",
                "type": "provider_error",
                "param": None,
            }
    return status, error, incomplete_details


def _build_usage(response: InternalResponse) -> dict[str, Any]:
    """Convert InternalResponse usage to the Responses usage payload."""
    if not response.usage:
        return _convert_usage(None)
    usage_payload: dict[str, Any] = {
        "prompt_tokens": response.usage.input_tokens,
        "completion_tokens": response.usage.output_tokens,
        "total_tokens": (
            response.usage.total_tokens
            if response.usage.total_tokens is not None
            else response.usage.input_tokens + response.usage.output_tokens
        ),
    }
    if response.usage.prompt_tokens_details:
        usage_payload["prompt_tokens_details"] = asdict(response.usage.prompt_tokens_details)
    if response.usage.completion_tokens_details:
        usage_payload["completion_tokens_details"] = asdict(
            response.usage.completion_tokens_details
        )
    return _convert_usage(usage_payload)


def _build_response_resource(
    context: FormatContext,
    response_id: str,
    response_model: str,
    created_at: int,
    completed_at: int | None,
    status: str,
    output: list[dict[str, Any]],
    error: dict[str, Any] | None,
    usage: dict[str, Any],
    incomplete_details: dict[str, Any] | None,
) -> dict[str, Any]:
    """Assemble the final Responses ResponseResource payload."""
    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "completed_at": completed_at,
        "status": status,
        "model": response_model,
        "previous_response_id": context.previous_response_id,
        "instructions": context.instructions,
        "output": output,
        "error": error,
        # Echo the request's tool declarations (raw Responses-shaped dicts
        # collected by the protocol layer), as the spec's ResponseResource
        # carries the effective tool set.
        "tools": [t for t in (context.tools or []) if isinstance(t, dict)],
        "tool_choice": context.tool_choice if context.tool_choice is not None else "auto",
        "truncation": context.truncation or "disabled",
        "parallel_tool_calls": (
            True if context.parallel_tool_calls is None else context.parallel_tool_calls
        ),
        "text": context.text or {"format": {"type": "text"}},
        # Spec: top_p/temperature are required numbers on ResponseResource;
        # default to the OpenAI defaults (1.0) when not provided.
        "top_p": context.top_p if context.top_p is not None else 1.0,
        "presence_penalty": (
            context.presence_penalty if context.presence_penalty is not None else 0.0
        ),
        "frequency_penalty": (
            context.frequency_penalty if context.frequency_penalty is not None else 0.0
        ),
        "top_logprobs": context.top_logprobs or 0,
        "temperature": context.temperature if context.temperature is not None else 1.0,
        "reasoning": context.reasoning,
        "usage": usage,
        "max_output_tokens": context.max_output_tokens,
        "max_tool_calls": context.max_tool_calls,
        "store": context.store or False,
        "background": context.background or False,
        "service_tier": context.service_tier or "default",
        "metadata": context.metadata or {},
        "safety_identifier": context.safety_identifier,
        "prompt_cache_key": context.prompt_cache_key,
        "incomplete_details": incomplete_details,
    }


@register_protocol_serializer("openresponses")
class OpenResponsesProtocolSerializer(ProtocolSerializer):
    """Protocol serializer for OpenAI Responses API format.

    Converts between /v1/responses wire format and InternalRequest/InternalResponse.
    """

    @property
    def protocol_name(self) -> str:
        return "openresponses"

    def parse_request(self, data: dict[str, Any]) -> InternalRequest:
        """Parse OpenResponses-format request dict to InternalRequest.

        For use when the request is available as a raw dict (e.g., after
        parameter overrides). Passes the raw ``text`` dict separately so the
        nested ``text.format`` shape is not lost by the Pydantic schema union.
        """
        request = ResponsesRequest.model_validate(data)
        return _convert_request_to_unified(request, raw_text=data.get("text"))

    def format_response(
        self, response: InternalResponse, context: FormatContext | None = None
    ) -> dict[str, Any]:
        """Format InternalResponse to OpenResponses wire format.

        Unlike the old llm_response_to_openresponses_format which accessed
        provider raw_response directly, this works from InternalResponse.
        """
        if not isinstance(response, InternalResponse):
            logger.error("Invalid response type: %s", type(response).__name__)
            return {"error": "Invalid response type"}

        context = _resolve_format_context(context)
        response_id = response.id or generate_response_id()
        response_model = response.model or "unknown"
        created_at = response.created_at or int(time.time())

        # First pass: collect web search results keyed by tool_use_id
        web_search_results = _collect_web_search_results(response.output)

        # Second pass: build output items
        output: list[dict[str, Any]] = []
        for block in response.output:
            item = _format_output_item(
                block, output, context.include, response, web_search_results, context
            )
            if item is not None:
                output.append(item)

        if not output:
            output.append(
                {
                    "type": "message",
                    "id": generate_item_id(),
                    "status": "completed",
                    "role": "assistant",
                    "phase": "final_answer",
                    "content": [{"type": "output_text", "text": ""}],
                }
            )

        # Re-insert raw upstream items that had no internal ContentBlock
        # equivalent (local_shell_call, agent_message, compaction, ...),
        # keeping the native upstream round-trip lossless. ``position`` is the
        # number of converted blocks that preceded the item, which maps 1:1 to
        # output items for typical responses. Each insertion shifts the list,
        # so earlier inserts are accumulated into the position of later ones.
        raw_output = getattr(response, "raw_output", None)
        if raw_output:
            for inserted, (position, raw_item) in enumerate(
                sorted(raw_output, key=lambda pair: pair[0])
            ):
                output.insert(min(position + inserted, len(output)), dict(raw_item))

        status, error, incomplete_details = _build_status_and_error(response)
        if status == "incomplete" and output:
            # Spec: an item that ends in a terminal incomplete state MUST be the
            # last item emitted, and the containing response MUST be incomplete.
            output[-1]["status"] = "incomplete"

        completed_at = int(time.time()) if status != "in_progress" else None

        usage = _build_usage(response)

        # Cache reasoning keyed by function call_id for next-turn restoration.
        # Best-effort: a cache write must never fail the client response.
        try_cache_reasoning_from_responses_output(output, response_id, logger_prefix="Serializer")

        return _build_response_resource(
            context,
            response_id=response_id,
            response_model=response_model,
            created_at=created_at,
            completed_at=completed_at,
            status=status,
            output=output,
            error=error,
            usage=usage,
            incomplete_details=incomplete_details,
        )


__all__ = [
    "OpenResponsesProtocolSerializer",
    "_convert_input_content",
    "conversation_to_input_items",
    "_parse_tool",
    "_parse_tool_arguments",
    "extract_custom_tool_names",
    "unwrap_custom_tool_arguments",
]
