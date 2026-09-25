# src/llm_proxy/core/serialization/openai/converter.py
"""OpenAI message and content conversion functions.

Pure functions for converting between Unified models and OpenAI message format.
These functions have no dependency on a serializer class.
"""

from dataclasses import dataclass
from typing import Any, Literal

from llm_proxy.core.utils import as_http_url, normalize_media_type
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
from llm_proxy.serialization._canonical_json import (
    canonical_json_string,
    canonical_json_string_if_parseable,
)
from llm_proxy.serialization._shared_conversion import try_convert_block
from llm_proxy.serialization._shared_degradation import (
    degrade_block_to_text,
    should_degrade_block,
)
from llm_proxy.serialization.content_parsers import (
    parse_image_block_anthropic,
    parse_text_block,
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

    if _chat_target_forbids_file_with_media(_context_provider_type(context)):
        result = _split_mixed_file_and_media_messages(result)

    return result


def _context_provider_type(context: BuildContext | None) -> str:
    """Resolve the provider *type* (e.g. ``openrouter``) from a build context.

    ``BuildContext.provider_name`` is the operator-chosen provider label and may
    differ from the adapter type; ``provider_type`` carries the registered
    adapter/serializer type that provider-specific wire decisions key on. Direct
    callers and tests that set only ``provider_name`` still resolve correctly.
    """
    if context is None:
        return "openai"
    return context.provider_type or context.provider_name


_CHAT_FILE_PART_TYPES = frozenset({"file"})
_CHAT_VISUAL_PART_TYPES = frozenset({"image_url", "video_url"})


def _chat_part_media_class(part: Any) -> str | None:
    """Classify a content part for the file/media co-occurrence rule."""
    if not isinstance(part, dict):
        return None
    part_type = part.get("type")
    if part_type in _CHAT_FILE_PART_TYPES:
        return "file"
    if part_type in _CHAT_VISUAL_PART_TYPES:
        return "visual"
    return None


def _split_mixed_file_and_media_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Split a message that mixes a ``file`` part with image/video parts.

    Zhipu / Z.AI GLM reject a ``content`` array carrying a ``file`` part together
    with ``image_url`` / ``video_url`` ("Not support passing both the ``file``
    and ``image_url`` or ``video_url`` parameters at the same time"). Text parts
    are neutral, so they stay with whichever neighbour they were next to and the
    message is split only at a file↔media boundary, preserving order.
    """
    split: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, list):
            split.append(message)
            continue
        present = {part.get("type") for part in content if isinstance(part, dict)}
        if not (present & _CHAT_FILE_PART_TYPES and present & _CHAT_VISUAL_PART_TYPES):
            split.append(message)
            continue

        current: list[Any] = []
        current_class: str | None = None
        for part in content:
            media_class = _chat_part_media_class(part)
            if (
                current
                and current_class is not None
                and media_class is not None
                and media_class != current_class
            ):
                split.append({**message, "content": current})
                current = []
                current_class = None
            current.append(part)
            if media_class is not None:
                current_class = media_class
        if current:
            split.append({**message, "content": current})
    return split


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
    if message.get("reasoning_details"):
        # A details-only turn: OpenRouter returns encrypted/summary reasoning
        # with no plaintext, and a model that produced it requires the array
        # back on the next turn.
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
    reasoning_details: list[dict[str, Any]] = []
    content_parts: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []

    for block in msg.content:
        if isinstance(block, TextBlock):
            content_parts.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageBlock):
            # An assistant turn carrying an image has no faithful wire shape on
            # either target (Chat Completions assistant content is text-only;
            # Responses assistant input content is ``output_text``/``refusal``).
            # Keep a placeholder instead of dropping the block silently.
            degraded = degrade_block_to_text(block)
            if degraded:
                content_parts.append({"type": "text", "text": degraded})
        elif isinstance(block, ThinkingBlock):
            if block.thinking:
                reasoning_parts.append(block.thinking)
            if block.signature:
                reasoning_signatures.append(block.signature)
            details = block.extra.get("reasoning_details")
            if isinstance(details, list):
                reasoning_details.extend(details)
        elif isinstance(block, RedactedThinkingBlock):
            reasoning_parts.append(block.data)
            reasoning_is_redacted = True
            details = block.extra.get("reasoning_details")
            if isinstance(details, list):
                reasoning_details.extend(details)
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
                    # Canonical (key-sorted, compact) arguments: the same
                    # logical call re-encoded by the client with a different
                    # key order must stay byte-identical across turns so
                    # upstream prefix/prompt caches keep hitting.
                    "arguments": canonical_json_string(block.input),
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
                inner = block.input
                if isinstance(inner, str):
                    # Freeform custom-tool input usually carries JSON; normalize
                    # parseable payloads so history calls stay byte-identical
                    # across turns (same rationale as ``arguments``).
                    inner = canonical_json_string_if_parseable(inner)
                tool_calls.append(
                    {
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": flatten_history_tool_name(
                                context.namespace_map if context else None, block.name
                            ),
                            "arguments": canonical_json_string({"content": inner}),
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
    if reasoning_details and (context is None or context.target_endpoint != "responses"):
        # OpenRouter's structured reasoning array, echoed back verbatim. Like
        # ``reasoning_content`` it is replayed whatever the destination: the
        # client sent it, and the Chat Completions message schema tolerates the
        # extra field. Not for Responses targets, whose items have no such key.
        result["reasoning_details"] = reasoning_details

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


# ---------------------------------------------------------------------------
# Per-provider document/file mapping on the Chat Completions wire
# ---------------------------------------------------------------------------
# Chat Completions has no single ``file`` shape: OpenAI, OpenRouter and Zhipu
# (Z.AI) nest it under a ``file`` object, DeepSeek uses top-level keys, Mistral
# spells documents ``document_url``, and only some providers accept an external
# URL at all. ``_CHAT_FILE_TARGETS`` is the single source of truth for that
# matrix. A provider absent from the table has no document content part on its
# Chat Completions endpoint, so documents degrade per
# ``unsupported_block_policy``.
#
# Evidence (official docs):
# - OpenAI: ``file`` = {file_data (base64), file_id, filename}, PDF only, no URL
#   (https://developers.openai.com/api/docs/guides/file-inputs).
# - OpenRouter: ``file.file_data`` is documented as "base64 data URL or URL"
#   (https://openrouter.ai/docs/guides/overview/multimodal/pdfs).
# - Zhipu / Z.AI GLM: ``file`` accepts file_url / file_data / file_id
#   (https://docs.z.ai/api-reference/llm/chat-completion).
# - Mistral: ``{"type": "document_url", "document_url": <url-or-data-uri>}``
#   (https://docs.mistral.ai/studio/document-processing/document_qna).
# - DeepSeek: top-level ``file`` part for images and Files-API ids, no URL
#   (https://api-docs.deepseek.com/guides/vision).
# - xAI / Qwen / Moonshot / MiniMax: no document content part on
#   /chat/completions (documents only via Responses / file-extract APIs), so
#   they stay out of the table and degrade.
#
# GLM additionally forbids a ``file`` part in the same message as an image or
# video part; see ``no_file_with_media`` and
# ``_split_mixed_file_and_media_messages``.


@dataclass(frozen=True)
class _ChatFileTarget:
    """How one provider carries a document/file on the Chat Completions wire."""

    style: Literal["nested_file", "top_level_file", "document_url"]
    #: Field that may hold an external URL (``file_data`` / ``file_url`` /
    #: ``document_url``), or ``None`` when the provider rejects URL sources.
    url_field: str | None
    #: Whether a Files-API ``file_id`` reference is accepted.
    file_id: bool
    accepts_pdf: bool
    #: Non-PDF documents (docx, xlsx, text, …).
    accepts_other_documents: bool
    #: Images carried through the ``file`` part (normally ``image_url``).
    accepts_images: bool
    #: Whether a ``file`` part may not share a message with ``image_url`` /
    #: ``video_url`` (Zhipu / Z.AI GLM), requiring the message to be split.
    no_file_with_media: bool = False


_CHAT_FILE_TARGETS: dict[str, _ChatFileTarget] = {
    # OpenAI Chat Completions: PDF only, base64 or Files-API id, never a URL.
    "openai": _ChatFileTarget(
        style="nested_file",
        url_field=None,
        file_id=True,
        accepts_pdf=True,
        accepts_other_documents=False,
        accepts_images=False,
    ),
    # The generic OpenAI-compatible adapter speaks the same wire shape as OpenAI
    # (its base URL is operator-supplied), so it inherits OpenAI's file rules.
    "openai-compatible": _ChatFileTarget(
        style="nested_file",
        url_field=None,
        file_id=True,
        accepts_pdf=True,
        accepts_other_documents=False,
        accepts_images=False,
    ),
    # OpenRouter's file parser accepts URLs in ``file_data`` and converts a
    # broad set of document types server-side.
    "openrouter": _ChatFileTarget(
        style="nested_file",
        url_field="file_data",
        file_id=True,
        accepts_pdf=True,
        accepts_other_documents=True,
        accepts_images=True,
    ),
    # Zhipu / Z.AI GLM: nested ``file`` object with a dedicated ``file_url``, and
    # a ``file`` part may not be mixed with an image/video part in one message.
    "zai": _ChatFileTarget(
        style="nested_file",
        url_field="file_url",
        file_id=True,
        accepts_pdf=True,
        accepts_other_documents=True,
        accepts_images=False,
        no_file_with_media=True,
    ),
    "zai-coding": _ChatFileTarget(
        style="nested_file",
        url_field="file_url",
        file_id=True,
        accepts_pdf=True,
        accepts_other_documents=True,
        accepts_images=False,
        no_file_with_media=True,
    ),
    "zhipu": _ChatFileTarget(
        style="nested_file",
        url_field="file_url",
        file_id=True,
        accepts_pdf=True,
        accepts_other_documents=True,
        accepts_images=False,
        no_file_with_media=True,
    ),
    "zhipu-coding": _ChatFileTarget(
        style="nested_file",
        url_field="file_url",
        file_id=True,
        accepts_pdf=True,
        accepts_other_documents=True,
        accepts_images=False,
        no_file_with_media=True,
    ),
    # Mistral: a flat ``document_url`` string (URL or base64 data URI).
    "mistral": _ChatFileTarget(
        style="document_url",
        url_field="document_url",
        file_id=False,
        accepts_pdf=True,
        accepts_other_documents=True,
        accepts_images=False,
    ),
    # DeepSeek: top-level file part, images and Files-API ids only.
    "deepseek": _ChatFileTarget(
        style="top_level_file",
        url_field=None,
        file_id=True,
        accepts_pdf=False,
        accepts_other_documents=False,
        accepts_images=True,
    ),
}


def _chat_target_forbids_file_with_media(provider_name: str) -> bool:
    """Whether the provider rejects ``file`` alongside image/video parts."""
    target = _CHAT_FILE_TARGETS.get(provider_name)
    return target is not None and target.no_file_with_media


def chat_document_url_needs_download(provider_name: str) -> bool:
    """Whether URL document sources must be downloaded and inlined.

    True when the provider has no URL field on its ``file`` part but does accept
    the inlined form — i.e. Chat Completions OpenAI / the generic
    ``openai-compatible`` adapter, whose ``file`` part carries base64 data only.
    Providers that take a URL directly keep it; providers with no document part
    at all cannot be helped by inlining.
    """
    target = _CHAT_FILE_TARGETS.get(provider_name)
    if target is None or target.url_field is not None:
        return False
    return target.accepts_pdf or target.accepts_other_documents


def chat_file_media_type_accepted(provider_name: str, media_type: str) -> bool:
    """Whether *media_type* may be sent inlined to *provider_name*'s file part."""
    target = _CHAT_FILE_TARGETS.get(provider_name)
    return target is not None and _accepts_chat_file_media_type(target, media_type)


def _accepts_chat_file_media_type(target: _ChatFileTarget, media_type: str) -> bool:
    """Whether *target* accepts a file of *media_type* through its file part."""
    normalized = normalize_media_type(media_type)
    if normalized.startswith("image/"):
        return target.accepts_images
    if normalized == "application/pdf":
        return target.accepts_pdf
    return target.accepts_other_documents


def _data_uri_media_type(value: str) -> str | None:
    """Return the media type of a ``data:`` URI, or None when it is not one."""
    if not value.startswith("data:"):
        return None
    header = value[5:].split(",", 1)[0]
    media_type = normalize_media_type(header)
    return media_type or None


def _render_chat_file_part(
    target: _ChatFileTarget,
    *,
    data_uri: str | None = None,
    url: str | None = None,
    file_id: str | None = None,
    filename: str | None = None,
) -> dict[str, Any] | None:
    """Render a document/file part for *target*, or None when unrepresentable.

    A URL source is refused when the provider has no URL field, and a
    ``file_id`` when the provider has no Files API — the caller then degrades
    the block instead of forwarding a field the upstream rejects.
    """
    if target.style == "document_url":
        value = url or data_uri
        if value is None:
            return None
        return {"type": "document_url", "document_url": value}

    file_dict: dict[str, Any] = {}
    has_content = False
    if url is not None:
        if target.url_field is None:
            return None
        file_dict[target.url_field] = url
        has_content = True
    elif data_uri is not None:
        file_dict["file_data"] = data_uri
        has_content = True
    if file_id is not None:
        if not target.file_id:
            return None
        file_dict["file_id"] = file_id
        has_content = True
    if not has_content:
        # ``filename`` alone is not a file input; degrade instead of emitting a
        # ``file`` part the upstream cannot resolve.
        return None
    if filename:
        file_dict["filename"] = filename
    if target.style == "top_level_file":
        return {"type": "file", **file_dict}
    return {"type": "file", "file": file_dict}


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
        image_url: dict[str, Any] = {"url": url}
        # ``detail`` is an optional enum (auto/low/high); an explicit null is
        # off-schema for strict OpenAI-compatible providers, so omit it.
        if block.detail is not None:
            image_url["detail"] = block.detail
        return {"type": "image_url", "image_url": image_url}

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
        target = _CHAT_FILE_TARGETS.get(provider_name)
        if target is not None:
            file_data = block.file_data
            file_url = block.file_url
            # Chat Completions callers (and the pre-split Responses mapping)
            # put an external URL in ``file_data``; treat it as a URL source so
            # each provider gets it in the field it actually accepts.
            if as_http_url(file_data):
                file_url = file_url or file_data
                file_data = None
            if file_data is not None:
                media_type = _data_uri_media_type(file_data)
                if media_type is not None and not _accepts_chat_file_media_type(target, media_type):
                    file_data = None
            part = _render_chat_file_part(
                target,
                data_uri=file_data,
                url=file_url,
                file_id=block.file_id,
                filename=block.filename,
            )
            if part is not None:
                return part
        degraded = degrade_block_to_text(block)
        return {"type": "text", "text": degraded} if degraded else None

    if isinstance(block, DocumentBlock):
        if block.source.type == "text":
            doc_data = block.source.data
            doc_text = doc_data if isinstance(doc_data, str) else str(doc_data)
            return {"type": "text", "text": doc_text}
        target = _CHAT_FILE_TARGETS.get(provider_name)
        # A provider whose file part carries no documents at all (DeepSeek is
        # images-only) must degrade a document block whatever its source type.
        if target is not None and (target.accepts_pdf or target.accepts_other_documents):
            if block.source.type == "base64":
                media_type = block.source.media_type or "application/pdf"
                if _accepts_chat_file_media_type(target, media_type):
                    part = _render_chat_file_part(
                        target,
                        data_uri=f"data:{media_type};base64,{block.source.data}",
                        filename=block.title,
                    )
                    if part is not None:
                        return part
            elif block.source.type == "url":
                # Anthropic URL document sources are PDFs. Providers without a
                # URL field refuse rather than mislabel the URL as base64.
                part = _render_chat_file_part(target, url=block.source.data, filename=block.title)
                if part is not None:
                    return part
            elif block.source.type == "file_id":
                part = _render_chat_file_part(
                    target, file_id=block.source.data, filename=block.title
                )
                if part is not None:
                    return part
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


def _block_to_responses_part(block: Any) -> dict[str, Any] | None:
    """Build an OpenAI Responses content part for a block whose Chat
    Completions shape cannot carry the information faithfully.

    Chat Completions has no ``file_id`` image source, no ``file_url`` field and
    no per-file ``detail``; the generic converter flattens those into a single
    ``image_url``/``file_data`` string. Returns ``None`` for every block the
    Chat Completions shape represents faithfully (the provider serializer
    rewrites those part types afterwards).
    """
    if isinstance(block, ImageBlock) and block.source.type == "file_id":
        part: dict[str, Any] = {"type": "input_image", "file_id": block.source.data}
        if block.detail is not None:
            part["detail"] = block.detail
        return part

    if isinstance(block, FileBlock):
        part = {"type": "input_file"}
        for key, value in (
            ("file_url", block.file_url),
            ("file_data", block.file_data),
            ("file_id", block.file_id),
            ("filename", block.filename),
            ("detail", block.detail),
        ):
            if value is not None:
                part[key] = value
        return part

    if isinstance(block, DocumentBlock) and block.source.type in ("base64", "url", "file_id"):
        part = {"type": "input_file"}
        if block.source.type == "base64":
            media_type = block.source.media_type or "application/pdf"
            part["file_data"] = f"data:{media_type};base64,{block.source.data}"
        elif block.source.type == "url":
            part["file_url"] = block.source.data
        else:
            part["file_id"] = block.source.data
        if block.title:
            part["filename"] = block.title
        return part

    return None


def _document_content_to_parts(
    block: DocumentBlock, context: BuildContext | None
) -> list[dict[str, Any]]:
    """Convert an Anthropic ``document`` ``content`` source into OpenAI parts.

    ``DocumentSource.data`` for a ``content`` source holds the raw Anthropic
    payload: a plain string or a list of ``text``/``image`` chunks. The generic
    ``_block_to_openai_part`` path applied ``str()`` to the list, emitting a
    Python repr as text and silently dropping any nested image. Parse the
    chunks into blocks and convert them through the normal path instead.
    """
    data = block.source.data
    if isinstance(data, str):
        return [{"type": "text", "text": data}] if data else []
    if not isinstance(data, list):
        return [{"type": "text", "text": str(data)}] if data else []

    nested: list[Any] = []
    for chunk in data:
        if isinstance(chunk, str):
            if chunk:
                nested.append(TextBlock(text=chunk))
            continue
        if not isinstance(chunk, dict):
            continue
        text_block = parse_text_block(chunk)
        if text_block is not None:
            nested.append(text_block)
            continue
        image_block = parse_image_block_anthropic(chunk)
        if image_block is not None:
            nested.append(image_block)
            continue
        # Unknown chunk type: keep a placeholder so nothing is lost silently.
        nested.append(TextBlock(text=f"[{chunk.get('type', 'content')}]"))

    if not nested:
        return []
    converted = content_to_openai_parts(nested, context)
    if isinstance(converted, str):
        return [{"type": "text", "text": converted}] if converted else []
    return converted


def content_to_openai_parts(
    content: list[Any], context: BuildContext | None = None
) -> list[dict[str, Any]] | str:
    """Convert content blocks to OpenAI content parts format.

    Returns:
        String if only text content, otherwise list of content part dicts.
    """
    parts: list[dict[str, Any]] = []
    policy = context.unsupported_block_policy if context else "drop"
    provider_name = _context_provider_type(context)
    supported_blocks = context.supported_content_blocks if context else frozenset()
    responses_target = context is not None and context.target_endpoint == "responses"

    for block in content:
        if responses_target:
            # Blocks whose Chat Completions shape loses information (file_id
            # images, file URLs, file detail) are built directly in Responses
            # form while the block is still available.
            responses_part = _block_to_responses_part(block)
            if responses_part is not None:
                parts.append(responses_part)
                continue

        if isinstance(block, DocumentBlock) and block.source.type == "content":
            # ``content`` sources may carry a list of nested text/image chunks;
            # expand them into parts instead of str()-ing the list.
            parts.extend(_document_content_to_parts(block, context))
            continue

        if isinstance(block, ImageBlock) and block.source.type == "file_id":
            # Chat Completions has no file_id image source. Degrade instead of
            # emitting the raw id as an ``image_url`` (a bad URL upstream).
            degraded = degrade_block_to_text(block)
            if degraded:
                parts.append({"type": "text", "text": degraded})
            continue

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
