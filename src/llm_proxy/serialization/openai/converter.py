# src/llm_proxy/core/serialization/openai/converter.py
"""OpenAI message and content conversion functions.

Pure functions for converting between Unified models and OpenAI message format.
These functions have no dependency on a serializer class.
"""

from typing import Any

import orjson

from llm_proxy.models import (
    AudioBlock,
    ConversationContext,
    CustomToolUseBlock,
    DocumentBlock,
    FileBlock,
    ImageBlock,
    Message,
    RefusalBlock,
    ServerToolUseBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    VideoBlock,
)
from llm_proxy.models.content_blocks.anthropic_builtin import (
    BashCodeExecutionToolResultBlock,
    CodeExecutionToolResultBlock,
    ContainerUploadBlock,
    MidConversationSystemBlock,
    SearchResultBlock,
    TextEditorCodeExecutionToolResultBlock,
    ToolReferenceBlock,
    ToolSearchToolResultBlock,
    WebFetchToolResultBlock,
    WebSearchResultContentBlock,
    WebSearchToolResultBlock,
)
from llm_proxy.models.content_blocks.extended import RawBlock, RedactedThinkingBlock
from llm_proxy.observability.logger import get_logger
from llm_proxy.serialization._shared_conversion import try_convert_block
from llm_proxy.serialization._shared_degradation import (
    degrade_block_to_text,
    should_degrade_block,
)
from llm_proxy.serialization.context import BuildContext
from llm_proxy.serialization.responses_toolkit.namespace import (
    flatten_history_tool_name,
)

logger = get_logger(__name__)

# Tuples of tool result block types used in isinstance checks across multiple functions
_TOOL_RESULT_BLOCKS = (
    ToolResultBlock,
    WebSearchToolResultBlock,
    WebFetchToolResultBlock,
    CodeExecutionToolResultBlock,
    BashCodeExecutionToolResultBlock,
    TextEditorCodeExecutionToolResultBlock,
    ToolSearchToolResultBlock,
)

_DEGRADABLE_TOOL_RESULT_BLOCKS = (
    WebFetchToolResultBlock,
    CodeExecutionToolResultBlock,
    BashCodeExecutionToolResultBlock,
    TextEditorCodeExecutionToolResultBlock,
    ToolSearchToolResultBlock,
)


def _effective_role_for_provider(role: str, context: BuildContext | None = None) -> str:
    """Return the role to send to the upstream provider.

    The OpenAI Responses API supports the ``developer`` role. OpenAI's own
    Chat Completions endpoint also accepts it, but many other OpenAI-compatible
    Chat Completions providers reject it. Degrade ``developer`` to ``system``
    only when the target endpoint is Chat Completions. The decision is based on
    the target endpoint, not the provider display name.
    """
    if role != "developer":
        return role
    if context is None:
        return "developer"
    return "system" if context.target_endpoint == "chat_completions" else "developer"


def format_conversation(
    conv: ConversationContext, context: BuildContext | None = None
) -> list[dict[str, Any]]:
    """Convert ConversationContext to OpenAI messages format.

    Note: In OpenAI format, tool_result blocks must be separate messages
    with role="tool", not embedded in user messages. This function handles
    the conversion from unified format (where tool_result can be in user messages)
    to OpenAI format.
    """
    result: list[dict[str, Any]] = []

    for sys_msg in conv.system_messages:
        msg_dict: dict[str, Any] = {
            "role": _effective_role_for_provider(sys_msg.role, context),
            "content": sys_msg.text_content,
        }
        if sys_msg.name is not None:
            msg_dict["name"] = sys_msg.name
        result.append(msg_dict)

    for msg in conv.messages:
        converted = _message_to_openai(msg, context)
        messages = converted if isinstance(converted, list) else [converted]
        for m in messages:
            if not _is_empty_assistant_message(m):
                result.append(m)

    return result


def _is_empty_assistant_message(message: dict[str, Any]) -> bool:
    """Return True for an assistant message that providers cannot consume.

    Chat-completions providers reject an assistant turn that says nothing and
    calls nothing (surfaced as "Upstream request failed"). This arises when a
    Responses-API turn carried only encrypted reasoning — ``encrypted_content``
    is OpenAI-Responses-specific and is dropped for chat-completions targets, so
    the message serializes to at most a blank ``reasoning_content`` with no
    ``content``/``tool_calls``. Dropping such messages here keeps the Chat
    Completions body valid; the reasoning is still preserved on the internal
    ``ThinkingBlock`` so a Responses-API provider can round-trip it.

    A message carrying a non-empty ``reasoning_content``/``reasoning`` is kept,
    since reasoning-capable chat-completions providers (e.g. DeepSeek) accept and
    use it.
    """
    if message.get("role") != "assistant":
        return False
    if message.get("content") or message.get("tool_calls"):
        return False
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    if reasoning.strip() == "tool call":
        return False
    return not reasoning.strip()


def _message_to_openai(
    msg: Message, context: BuildContext | None = None
) -> dict[str, Any] | list[dict[str, Any]]:
    """Convert a single Message to OpenAI format.

    Returns:
        A single message dict, or a list of message dicts if the message
        contains ToolResultBlock(s) that need to be separated into tool role messages.
    """
    if msg.role == "tool":
        return _tool_message_to_openai(msg)
    if msg.role == "developer":
        return {
            "role": _effective_role_for_provider(msg.role, context),
            "content": content_to_openai_parts(msg.content, context),
        }
    if msg.role == "user":
        return _user_message_to_openai(msg, context)
    if msg.role == "assistant":
        return _assistant_message_to_openai(msg, context)
    if msg.role == "system":
        # Anthropic's mid-conversation system messages (mid_conv_system) are not
        # supported by most of OpenAI-format providers. Degrade to user message so the
        # instruction text is still sent without causing provider errors.
        # Wrap the content in XML tags to preserve semantic intent.
        sys_text = _content_to_string(msg.content)
        if sys_text:
            wrapped = f"<system-prompt>\n{sys_text}\n</system-prompt>"
        else:
            wrapped = "<system-prompt></system-prompt>"
        return {"role": "user", "content": wrapped}

    return {"role": msg.role, "content": content_to_openai_parts(msg.content, context)}


def _tool_message_to_openai(msg: Message) -> dict[str, Any]:
    """Convert a tool message to OpenAI format.

    Tool result content may be a plain string or a list of content blocks
    (text + images). All text blocks are concatenated and any non-text blocks
    are degraded to placeholders so nothing is silently dropped.
    """
    tool_msg: dict[str, Any] = {"role": msg.role}
    content_text = ""
    tool_call_id = ""
    for block in msg.content:
        if isinstance(block, ToolResultBlock):
            content_text = _tool_result_content_to_output_text(block.content)
            tool_call_id = block.tool_use_id
            break
        if isinstance(block, ToolUseBlock):
            tool_call_id = block.id
            break
    tool_msg["content"] = content_text
    tool_msg["tool_call_id"] = tool_call_id
    if msg.name is not None:
        tool_msg["name"] = msg.name
    return tool_msg


def _tool_result_content_to_output_text(content: Any) -> str:
    """Convert tool result content to a single output string.

    Text blocks are concatenated. Non-text blocks (e.g. images) cannot be
    carried in the string-only ``function_call_output.output`` field, so they
    are degraded to placeholders. This prevents tool result content from being
    silently lost when the upstream API only supports string output.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, TextBlock):
                if block.text:
                    parts.append(block.text)
            else:
                degraded = degrade_block_to_text(block)
                if degraded:
                    parts.append(degraded)
        return " ".join(parts)

    return str(content) if content is not None else ""


def _user_message_to_openai(
    msg: Message, context: BuildContext | None = None
) -> dict[str, Any] | list[dict[str, Any]]:
    """Convert a user message to OpenAI format.

    User messages support multimodal content (text + images).
    If the user message contains ToolResultBlock(s), they must be converted
    to separate tool role messages per OpenAI format requirements.

    In Anthropic format, tool_result blocks are inside user messages.
    In OpenAI format, tool results must be separate messages with role="tool".

    Returns:
        A single user message dict, or a list of [user_message, tool_message, ...]
        if tool_result blocks are present.
    """
    tool_results: list[Any] = []
    other_blocks: list[Any] = []

    for block in msg.content:
        if isinstance(
            block,
            _TOOL_RESULT_BLOCKS,
        ):
            tool_results.append(block)
        else:
            other_blocks.append(block)

    if not tool_results:
        content = content_to_openai_parts(msg.content, context)
        result: dict[str, Any] = {"role": msg.role, "content": content}
        if msg.name is not None:
            result["name"] = msg.name
        return result

    result_messages: list[dict[str, Any]] = []

    if other_blocks:
        user_content = content_to_openai_parts(other_blocks, context)
        user_msg: dict[str, Any] = {"role": "user", "content": user_content}
        if msg.name is not None:
            user_msg["name"] = msg.name
        result_messages.append(user_msg)

    for tr in tool_results:
        tool_msg = _tool_result_to_openai_tool_message(tr, msg.name)
        result_messages.append(tool_msg)

    return result_messages


def _tool_result_to_openai_tool_message(block: Any, name: str | None = None) -> dict[str, Any]:
    """Convert a ToolResultBlock (or compatible result block) to OpenAI tool role message format.

    OpenAI format requires:
    - role: "tool"
    - tool_call_id: the ID of the tool call this result is for
    - content: the result content (must be string)
    """
    content = getattr(block, "content", "")
    if isinstance(content, list):
        content = _content_to_string(content)
    elif not isinstance(content, str):
        content = str(content) if content is not None else ""

    tool_msg: dict[str, Any] = {
        "role": "tool",
        "tool_call_id": getattr(block, "tool_use_id", ""),
        "content": content,
    }
    if name is not None:
        tool_msg["name"] = name
    if getattr(block, "is_error", False):
        tool_msg["content"] = f"Error: {content}"
    caller = getattr(block, "caller", None)
    if caller:
        tool_msg["caller"] = {"type": caller.type}
        if caller.tool_id:
            tool_msg["caller"]["tool_id"] = caller.tool_id
    return tool_msg


def _assistant_message_to_openai(
    msg: Message, context: BuildContext | None = None
) -> dict[str, Any] | list[dict[str, Any]]:
    """Convert an assistant message to OpenAI format.

    Assistant messages can contain text and tool calls.
    ThinkingBlock is converted to `reasoning_content` field.

    If the message contains ToolResultBlock or WebSearchToolResultBlock
    (from Anthropic server_tool_use results), they are converted to
    separate tool role messages.

    Returns:
        A single assistant message dict, or a list of
        [assistant_message, tool_message, ...] if tool result blocks are present.
    """
    tool_result_blocks: list[Any] = []

    reasoning_parts: list[str] = []
    reasoning_signatures: list[str] = []
    reasoning_is_redacted = False
    content_parts: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []

    for block in msg.content:
        if isinstance(block, TextBlock):
            content_parts.append({"type": "text", "text": block.text})
        elif isinstance(block, ThinkingBlock):
            if block.thinking:
                reasoning_parts.append(block.thinking)
            if block.signature:
                reasoning_signatures.append(block.signature)
        elif isinstance(block, RedactedThinkingBlock):
            reasoning_parts.append(block.data)
            reasoning_is_redacted = True
        elif isinstance(block, (ToolUseBlock, ServerToolUseBlock)):
            # Flatten history call names for Chat Completions targets so they
            # match the flattened tool definitions sent upstream (models echo
            # the history name). Native Responses targets keep original names
            # because their tool definitions are not flattened.
            tc_name = block.name
            if context is not None and context.target_endpoint != "responses":
                tc_name = flatten_history_tool_name(context.namespace_map, tc_name)
            tc: dict[str, Any] = {
                "id": block.id,
                "type": "function",
                "function": {
                    "name": tc_name,
                    "arguments": orjson.dumps(block.input).decode(),
                },
            }
            if isinstance(block, ToolUseBlock) and block.extra.get("thought_signature"):
                tc["thought_signature"] = block.extra["thought_signature"]
            tool_calls.append(tc)
        elif isinstance(block, CustomToolUseBlock):
            if context is not None and context.target_endpoint == "responses":
                # The Responses API supports custom tool calls natively.
                tool_calls.append(
                    {
                        "id": block.id,
                        "type": "custom",
                        "custom": {
                            "name": block.name,
                            "input": block.input,
                        },
                    }
                )
            else:
                # Chat Completions only accepts ``type: "function"`` tool calls;
                # strict OpenAI-compatible providers reject ``type: "custom"``
                # ("unknown variant `custom`, expected `function`"). Re-wrap the
                # freeform input into the ``{"content": ...}`` envelope used by
                # the custom-tool function bridge (see
                # OpenAIToolsHandler._custom_tool_to_function) so the history
                # matches the converted tool definitions the provider received.
                tool_calls.append(
                    {
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": flatten_history_tool_name(
                                context.namespace_map if context else None, block.name
                            ),
                            "arguments": orjson.dumps({"content": block.input}).decode(),
                        },
                    }
                )
        elif isinstance(block, AudioBlock) and block.source.type == "base64":
            content_parts.append(
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": (f"data:{block.source.media_type};base64,{block.source.data}")
                    },
                }
            )
        elif isinstance(block, RefusalBlock):
            content_parts.append({"type": "refusal", "refusal": block.refusal})
        elif isinstance(
            block,
            _TOOL_RESULT_BLOCKS,
        ):
            tool_result_blocks.append(block)

    use_structured = tool_calls or any(p.get("type") != "text" for p in content_parts)

    result: dict[str, Any] = {"role": msg.role}

    if use_structured:
        if content_parts:
            result["content"] = content_parts
        if tool_calls:
            result["tool_calls"] = tool_calls
    else:
        text_content = " ".join(p.get("text", "") for p in content_parts if p.get("type") == "text")
        if text_content:
            result["content"] = text_content

    if reasoning_parts:
        result["reasoning_content"] = " ".join(reasoning_parts)
    elif tool_calls:
        restored = _restore_reasoning_from_cache(tool_calls)
        if restored:
            result["reasoning_content"] = restored
            logger.info(
                f"Converter: restored reasoning from cache for {len(tool_calls)} tool call(s)"
            )
    if reasoning_signatures:
        result["reasoning_signature"] = "".join(reasoning_signatures)
    if reasoning_is_redacted:
        result["reasoning_is_redacted"] = True

    if msg.name is not None:
        result["name"] = msg.name

    if not tool_result_blocks:
        return result

    result_messages: list[dict[str, Any]] = [result]

    for tr in tool_result_blocks:
        tool_msg = _tool_result_to_openai_tool_message(tr, msg.name)
        result_messages.append(tool_msg)

    return result_messages


def _restore_reasoning_from_cache(tool_calls: list[dict[str, Any]]) -> str | None:
    """Try to restore reasoning from cache for tool calls without explicit reasoning."""
    from llm_proxy.core.reasoning_cache import get as _cache_get

    for tc in tool_calls:
        cid = tc.get("id")
        if cid:
            reasoning = _cache_get(cid)
            if reasoning:
                return reasoning
    return None


def _content_to_string(content: list[Any]) -> str:
    """Convert content blocks to a plain string.

    Extracts text from TextBlocks and ToolResultBlocks.
    """
    result: list[str] = []
    for block in content:
        if isinstance(block, TextBlock):
            result.append(block.text)
        elif isinstance(block, ToolResultBlock):
            if isinstance(block.content, str):
                result.append(block.content)
            elif isinstance(block.content, list):
                result.append(_content_to_string(block.content))
        elif isinstance(block, WebSearchResultContentBlock):
            result.append(f"[{block.title}]({block.url})")
    return "".join(result)


def _make_media_url(source: Any, default_media_type: str) -> str:
    """Build a data URL or passthrough URL from a media source.

    For base64 sources, constructs a ``data:`` URL with the given default
    media type if the source does not specify one. For url and file_id
    sources, returns the raw data unchanged.
    """
    if source.type == "base64":
        media_type = source.media_type or default_media_type
        return f"data:{media_type};base64,{source.data}"
    return source.data


def _block_to_openai_part(block: Any, provider_name: str) -> dict[str, Any] | None:
    """Convert a single content block to an OpenAI part dict.

    Returns the part dict for recognized block types, or ``None`` if the block
    type is not directly convertible (caller should handle degradation).
    """
    converted = try_convert_block(block)
    if converted is not None:
        block = converted

    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}

    if isinstance(block, ImageBlock):
        url = _make_media_url(block.source, "image/png")
        return {
            "type": "image_url",
            "image_url": {"url": url, "detail": block.detail},
        }

    if isinstance(block, AudioBlock):
        if provider_name == "openrouter":
            if block.source.type == "base64":
                media_type = block.source.media_type or "audio/wav"
                audio_format = media_type.split("/")[-1] if "/" in media_type else "wav"
                if audio_format == "mpeg":
                    audio_format = "mp3"
                return {
                    "type": "input_audio",
                    "input_audio": {
                        "data": block.source.data,
                        "format": audio_format,
                    },
                }
            logger.warning(
                "OpenRouter does not support audio source, degrading to text",
                extra={"source_type": block.source.type},
            )
            degraded = degrade_block_to_text(block)
            return {"type": "text", "text": degraded} if degraded else None
        if block.source.type in ("url", "base64", "file_id"):
            url = _make_media_url(block.source, "audio/wav")
            return {"type": "audio_url", "audio_url": {"url": url}}
        return None

    if isinstance(block, FileBlock):
        file_dict: dict[str, Any] = {}
        if block.file_data:
            file_dict["file_data"] = block.file_data
        if block.file_id:
            file_dict["file_id"] = block.file_id
        if block.filename:
            file_dict["filename"] = block.filename
        if provider_name == "deepseek":
            # DeepSeek's Chat Completions endpoint carries file blocks with
            # top-level keys (file_id / file_data / filename) instead of the
            # OpenAI nested ``file`` object (see DeepSeek vision docs).
            if file_dict:
                return {"type": "file", **file_dict}
        elif provider_name in ("openai", "openrouter"):
            return {"type": "file", "file": file_dict}
        degraded = degrade_block_to_text(block)
        return {"type": "text", "text": degraded} if degraded else None

    if isinstance(block, DocumentBlock):
        if block.source.type in ("text", "content"):
            doc_data = block.source.data
            doc_text = doc_data if isinstance(doc_data, str) else str(doc_data)
            return {"type": "text", "text": doc_text}
        if provider_name in ("openai", "openrouter"):
            if block.source.type == "base64":
                media_type = block.source.media_type or "application/pdf"
                file_data = f"data:{media_type};base64,{block.source.data}"
                file_dict = {"file_data": file_data}
            elif block.source.type == "url":
                file_dict = {"file_data": block.source.data}
            elif block.source.type == "file_id":
                file_dict = {"file_id": block.source.data}
            else:
                file_dict = None
            if file_dict:
                if block.title:
                    file_dict["filename"] = block.title
                return {"type": "file", "file": file_dict}
        degraded = degrade_block_to_text(block)
        return {"type": "text", "text": degraded} if degraded else None

    if isinstance(block, VideoBlock):
        if block.source.type in ("url", "base64"):
            url = _make_media_url(block.source, "video/mp4")
            return {"type": "video_url", "video_url": {"url": url}}
        if block.source.type == "file_id":
            logger.warning("VideoBlock with file_id source is not supported, degrading to text")
            degraded = degrade_block_to_text(block)
            return {"type": "text", "text": degraded} if degraded else None
        return None

    if isinstance(block, WebSearchResultContentBlock):
        return {"type": "text", "text": f"[{block.title}]({block.url})"}

    if isinstance(block, MidConversationSystemBlock):
        sys_text = _content_to_string(block.content)
        if sys_text:
            return {"type": "text", "text": sys_text}
        return None

    if isinstance(block, _DEGRADABLE_TOOL_RESULT_BLOCKS):
        result_text = (
            _content_to_string(block.content)
            if isinstance(block.content, list)
            else str(block.content or "")
        )
        if result_text:
            return {"type": "text", "text": result_text}
        return None

    if isinstance(block, SearchResultBlock):
        search_text = (
            _content_to_string(block.content)
            if isinstance(block.content, list)
            else str(block.content or "")
        )
        if search_text:
            return {"type": "text", "text": search_text}
        return None

    if isinstance(block, ContainerUploadBlock):
        upload_text = block.filename or block.file_id or "[Container upload]"
        return {"type": "text", "text": upload_text}

    if isinstance(block, ToolReferenceBlock):
        ref_text = block.tool_name or block.tool_id or "[Tool reference]"
        return {"type": "text", "text": ref_text}

    if isinstance(block, RawBlock):
        if block.provider_type.startswith("openai:"):
            return block.data
        return {"type": "text", "text": f"[Raw block: {block.provider_type}]"}

    return None


def content_to_openai_parts(
    content: list[Any], context: BuildContext | None = None
) -> list[dict[str, Any]] | str:
    """Convert content blocks to OpenAI content parts format.

    Returns:
        String if only text content, otherwise list of content part dicts.
    """
    parts: list[dict[str, Any]] = []
    policy = context.unsupported_block_policy if context else "drop"
    provider_name = context.provider_name if context else "openai"
    supported_blocks = context.supported_content_blocks if context else frozenset()

    for block in content:
        part = _block_to_openai_part(block, provider_name)
        if part is not None:
            parts.append(part)
        else:
            # Handle unsupported block types via shared degradation logic.
            is_supported = supported_blocks is not None and type(block) in supported_blocks
            if not is_supported and not should_degrade_block(
                policy, block, provider_name, supported_blocks=supported_blocks
            ):
                continue
            degraded = degrade_block_to_text(block)
            if degraded:
                parts.append({"type": "text", "text": degraded})

    if len(parts) == 1 and parts[0].get("type") == "text":
        return parts[0].get("text", "")
    return parts
