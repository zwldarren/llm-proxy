"""Anthropic provider serializer.

Handles conversion between InternalRequest/InternalResponse and Anthropic API format.
"""

import logging
from typing import Any

import orjson

from llm_proxy.core.thinking import convert_to_anthropic
from llm_proxy.core.utils import generate_response_id
from llm_proxy.models import (
    AudioBlock,
    CustomToolUseBlock,
    DocumentBlock,
    FileBlock,
    ImageBlock,
    InternalRequest,
    InternalResponse,
    RedactedThinkingBlock,
    RefusalBlock,
    ServerToolUseBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
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
from llm_proxy.models.finish_reasons import map_finish_reason
from llm_proxy.models.tools import (
    BashTool,
    CodeExecutionTool,
    CustomTool,
    FunctionTool,
    MemoryTool,
    OpenAIToolSearchTool,
    TextEditorTool,
    ToolChoice,
    ToolChoiceAllowedTools,
    ToolChoiceFunction,
    ToolChoiceNamed,
    ToolSearchTool,
    UserLocation,
    WebFetchTool,
    WebSearchTool,
)
from llm_proxy.models.types import Usage
from llm_proxy.serialization.anthropic.mixin import AnthropicContentMixin
from llm_proxy.serialization.anthropic.streaming_converter import AnthropicChunkConverter
from llm_proxy.serialization.context import BuildContext
from llm_proxy.serialization.providers.base import ProviderSerializer
from llm_proxy.serialization.providers.registry import register_provider_serializer

_TOOL_CHOICE_MODE_TO_ANTHROPIC = {
    "auto": "auto",
    "none": "none",
    "required": "any",
    "any": "any",
}

logger = logging.getLogger(__name__)

# OpenAI ``service_tier`` values mapped to Anthropic ``service_tier``.
# Anthropic Messages API only accepts ``"auto"`` or ``"standard_only"``.
_OPENAI_SERVICE_TIER_TO_ANTHROPIC: dict[str, str] = {
    "auto": "auto",
    "scale": "auto",
    "default": "standard_only",
    "flex": "standard_only",
    "priority": "auto",
}

# Extra keys that must NOT leak into the Anthropic request body. These are
# either internal bookkeeping keys (``_system_blocks``) consumed elsewhere, or
# cross-protocol fields from OpenAI/Responses that have no Anthropic top-level
# meaning and are handled explicitly (metadata, service_tier, parallel tool
# use) or deliberately dropped (reasoning config, previous_response_id, ...).
_NON_ANTHROPIC_EXTRA_KEYS: frozenset[str] = frozenset(
    {
        "_system_blocks",
        "reasoning",
        "previous_response_id",
        "truncation",
        "include",
        "background",
        "max_tool_calls",
        "disable_parallel_tool_use",
        "parallel_tool_calls",
        "thought_signature",
        "metadata",
        "service_tier",
        "user",
        "responses_tools",
        "text",
        "responses_raw_fields",
    }
)


def _map_openai_service_tier(value: str) -> str | None:
    """Map an OpenAI ``service_tier`` value to an Anthropic ``service_tier`` value.

    Returns ``None`` (and emits a warning) for values with no Anthropic equivalent.
    """
    mapped = _OPENAI_SERVICE_TIER_TO_ANTHROPIC.get(value)
    if mapped is None:
        logger.warning(
            "Dropping unsupported OpenAI service_tier value %r for Anthropic provider",
            value,
        )
    return mapped


def _normalize_input_schema(parameters: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize a function tool's parameters JSON Schema for Anthropic.

    Anthropic requires ``input_schema.type == "object"``. OpenAI Responses
    function tools may carry ``parameters: null`` or ``{"type": null}``, which
    Anthropic-compatible endpoints reject with a 400. Mirrors cc-switch's
    ``normalize_function_parameters``.
    """
    schema = (
        dict(parameters) if isinstance(parameters, dict) else {"type": "object", "properties": {}}
    )
    if schema.get("type") != "object":
        schema["type"] = "object"
    return schema


def _set_temperature_and_top_p(body: dict[str, Any], params: Any) -> None:
    """Emit ``temperature``/``top_p`` into the body when set.

    Claude models reject both while thinking is enabled, so for Claude models
    they are only emitted for non-thinking requests (and restored when a
    forced tool_choice disables thinking). Non-Claude models behind a custom
    base_url accept them alongside thinking and always get them.
    """
    if params.temperature is not None:
        body["temperature"] = params.temperature
    if params.top_p is not None:
        body["top_p"] = params.top_p


def _custom_tool_definition_json(tool: CustomTool) -> str:
    """Serialize a CustomTool back to its Responses-style definition dict.

    Used to embed the original definition (including format/grammar metadata)
    in the bridged function-tool description so the model can see the tool is
    freeform and how its input must be formatted. Keys are sorted for stable
    output across requests.
    """
    definition: dict[str, Any] = {"type": "custom", "name": tool.name}
    if tool.description:
        definition["description"] = tool.description
    if tool.format_type:
        definition["format"] = {"type": tool.format_type}
        if tool.grammar_definition:
            definition["format"]["definition"] = tool.grammar_definition
        if tool.grammar_syntax:
            definition["format"]["syntax"] = tool.grammar_syntax
    return orjson.dumps(definition, option=orjson.OPT_SORT_KEYS).decode()


def _custom_tool_description(tool: CustomTool) -> str:
    """Build the bridged function-tool description for a custom tool.

    Embeds the original definition (including any format/grammar metadata) in
    the description so the model knows the tool is freeform and how its input
    must be formatted — mirrors cc-switch's "Original tool definition:" trick.
    """
    parts = [tool.description] if tool.description else []
    parts.append(
        "Original tool definition:\n```json\n" + _custom_tool_definition_json(tool) + "\n```"
    )
    return "\n\n".join(parts)


def _block_ids(message: dict[str, Any], block_type: str, id_field: str) -> list[str]:
    """Collect ``id_field`` values of all ``block_type`` blocks in a message."""
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [
        str(block.get(id_field, ""))
        for block in content
        if isinstance(block, dict) and block.get("type") == block_type
    ]


def _drop_tool_result_blocks(message: dict[str, Any]) -> None:
    """Remove all ``tool_result`` blocks from a user message in place."""
    content = message.get("content")
    if isinstance(content, list):
        message["content"] = [
            block
            for block in content
            if not (isinstance(block, dict) and block.get("type") == "tool_result")
        ]


def _message_has_content(message: dict[str, Any]) -> bool:
    """Whether a message carries any content blocks."""
    content = message.get("content")
    return bool(content) if isinstance(content, list) else True


def _drop_incomplete_tool_turns(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop tool turns that no longer form a complete adjacent pair.

    Anthropic requires every ``tool_use`` block in an assistant turn to be
    answered by a ``tool_result`` in the immediately following user turn.
    Compacted/resumed sessions (e.g. Codex via /v1/responses) may replay an
    assistant turn whose ``tool_result`` pair was truncated; replaying it
    verbatim would 400. Mirrors cc-switch's ``drop_incomplete_tool_turns``.
    """
    sanitized: list[dict[str, Any]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        is_assistant = message.get("role") == "assistant"
        tool_use_ids = _block_ids(message, "tool_use", "id") if is_assistant else []
        if tool_use_ids:
            paired_user = (
                messages[index + 1]
                if index + 1 < len(messages) and messages[index + 1].get("role") == "user"
                else None
            )
            tool_result_ids = (
                _block_ids(paired_user, "tool_result", "tool_use_id") if paired_user else []
            )
            complete = (
                all(tool_use_ids)
                and all(tool_result_ids)
                and len(set(tool_use_ids)) == len(tool_use_ids)
                and len(set(tool_result_ids)) == len(tool_result_ids)
                and set(tool_use_ids) == set(tool_result_ids)
            )
            if complete and paired_user is not None:
                # complete can only hold with a paired user turn: the
                # set-equality check cannot pass against an empty id list.
                sanitized.append(message)
                sanitized.append(paired_user)
            elif paired_user is not None:
                # The pair is broken: keep the user turn but strip its
                # orphaned tool_result blocks.
                user = dict(paired_user)
                _drop_tool_result_blocks(user)
                if _message_has_content(user):
                    sanitized.append(user)
            index += 2 if paired_user is not None else 1
            continue
        if message.get("role") == "user":
            # A user message not consumed as the adjacent half of a complete
            # tool pair cannot legally retain tool_result blocks.
            _drop_tool_result_blocks(message)
        if _message_has_content(message):
            sanitized.append(message)
        index += 1
    return sanitized


def _trim_trailing_assistant_text(messages: list[dict[str, Any]]) -> None:
    """Trim trailing whitespace from the last assistant text block in place.

    Anthropic 400s on a text block whose text is empty or whitespace-only;
    such blocks arise when a prior Responses turn recorded an empty
    ``output_text`` alongside a ``tool_use``. Mirrors cc-switch's
    ``trim_trailing_assistant_text``.
    """
    if not messages or messages[-1].get("role") != "assistant":
        return
    content = messages[-1].get("content")
    if not isinstance(content, list) or not content:
        return
    last_block = content[-1]
    if not isinstance(last_block, dict) or last_block.get("type") != "text":
        return
    text = last_block.get("text")
    if not isinstance(text, str):
        return
    trimmed = text.rstrip()
    if not trimmed:
        content.pop()
        return
    last_block["text"] = trimmed


def _drop_empty_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove messages whose content list is empty."""
    return [m for m in messages if _message_has_content(m)]


def _ensure_leading_user_message(messages: list[dict[str, Any]]) -> None:
    """Guarantee the first message has role ``user``.

    Anthropic requires the first message to be a user message; compacted or
    resumed sessions (e.g. Codex) may start with an assistant turn, and
    normalization may leave the list empty (e.g. an orphaned ``tool_result``
    that Anthropic would reject). A placeholder user message is inserted in
    both cases. Mirrors cc-switch's ``ensure_leading_user_message``.
    """
    if not messages or messages[0].get("role") != "user":
        messages.insert(
            0,
            {
                "role": "user",
                "content": [{"type": "text", "text": "(continuing the conversation)"}],
            },
        )


def _normalize_anthropic_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize the outbound message list for Anthropic's API constraints.

    Anthropic requires non-empty messages, the first message to be ``user``,
    and every ``tool_use`` answered by a ``tool_result`` in the immediately
    following user turn. Compacted/resumed sessions (e.g. Codex via
    /v1/responses) can violate all three; normalize before sending. Mirrors
    cc-switch's message sanitization in ``transform_codex_anthropic``.
    """
    messages = _drop_incomplete_tool_turns(messages)
    _trim_trailing_assistant_text(messages)
    messages = _drop_empty_messages(messages)
    _ensure_leading_user_message(messages)
    return messages


def parse_usage_and_provider_extras(
    response: dict[str, Any],
) -> tuple[Usage | None, dict[str, Any]]:
    """Parse usage (with cache folding) and billing extras from a raw Anthropic response.

    Anthropic's ``input_tokens`` only counts tokens after the last cache
    breakpoint; ``cache_read_input_tokens``/``cache_creation_input_tokens``
    are separate. Total input = input_tokens + cache_read + cache_creation.
    The extras dict collects usage-level keys (``cache_creation``,
    ``inference_geo``, ``server_tool_use``) that downstream billing reads
    from ``provider_info`` (e.g. web search request counts).
    """
    usage = None
    extras: dict[str, Any] = {}
    usage_data = response.get("usage")
    if isinstance(usage_data, dict):
        prompt_tokens = (
            usage_data.get("input_tokens", 0)
            + (usage_data.get("cache_read_input_tokens") or 0)
            + (usage_data.get("cache_creation_input_tokens") or 0)
        )
        usage = Usage(
            input_tokens=prompt_tokens,
            output_tokens=usage_data.get("output_tokens", 0),
            total_tokens=prompt_tokens + (usage_data.get("output_tokens", 0)),
            cache_read_input_tokens=usage_data.get("cache_read_input_tokens"),
            cache_creation_input_tokens=usage_data.get("cache_creation_input_tokens"),
        )

        # Beta extensions (fast-mode "speed", compaction/fallback
        # "iterations") ride extras verbatim so anthropic clients keep the
        # full Usage shape on converted paths.
        for key in (
            "cache_creation",
            "inference_geo",
            "server_tool_use",
            "output_tokens_details",
            "service_tier",
            "speed",
            "iterations",
        ):
            if key in usage_data and usage_data[key] is not None:
                extras[key] = usage_data[key]
    return usage, extras


@register_provider_serializer("anthropic")
class AnthropicProviderSerializer(AnthropicContentMixin, ProviderSerializer):
    """Anthropic provider serializer.

    Handles conversion between InternalRequest/InternalResponse and Anthropic API format.
    """

    _DEFAULT_PROVIDER_NAME = "anthropic"

    supported_content_blocks = frozenset(
        {
            TextBlock,
            ImageBlock,
            ToolUseBlock,
            ServerToolUseBlock,
            CustomToolUseBlock,
            ThinkingBlock,
            RedactedThinkingBlock,
            RefusalBlock,
            DocumentBlock,
            AudioBlock,
            FileBlock,
            SearchResultBlock,
            ContainerUploadBlock,
            ToolReferenceBlock,
            MidConversationSystemBlock,
            WebSearchToolResultBlock,
            WebFetchToolResultBlock,
            WebSearchResultContentBlock,
            CodeExecutionToolResultBlock,
            BashCodeExecutionToolResultBlock,
            TextEditorCodeExecutionToolResultBlock,
            ToolSearchToolResultBlock,
            ToolResultBlock,
        }
    )

    def _build_provider_request(
        self, request: InternalRequest, context: BuildContext
    ) -> dict[str, Any]:
        system = None
        messages: list[dict[str, Any]] = []

        if request.conversation.system_messages:
            if request.extra and "_system_blocks" in request.extra:
                system = request.extra["_system_blocks"]
            else:
                system_texts = []
                for sm in request.conversation.system_messages:
                    if sm.text_content:
                        system_texts.append(sm.text_content)
                if system_texts:
                    system = [{"type": "text", "text": "\n\n".join(system_texts)}]

        # Anthropic has no ``developer`` role; developer messages (from the
        # OpenAI Responses protocol) carry system-level instructions and are
        # merged into the top-level ``system`` parameter (mirrors cc-switch's
        # normalize_developer_roles). Accumulated here and appended after the
        # message loop so the loop stays a single pass.
        developer_texts: list[str] = []

        for msg in request.conversation.messages:
            # Deprecated OpenAI ``function`` role is not a valid Anthropic
            # message role. Degrade it the same way as ``role: tool`` results:
            # a user message containing a ``tool_result`` block.
            if msg.role == "function":
                result_text = " ".join(
                    block.text for block in msg.content if isinstance(block, TextBlock)
                )
                tool_use_id = msg.name or "function_call"
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": result_text,
                            }
                        ],
                    }
                )
                continue

            if msg.role == "developer":
                if msg.text_content:
                    developer_texts.append(msg.text_content)
                continue

            content = self.format_content_blocks(msg.content, context=context)
            role = "user" if msg.role == "tool" else msg.role
            messages.append(
                {
                    "role": role,
                    "content": content if isinstance(content, list) else [content],
                }
            )

        if developer_texts:
            developer_block = {"type": "text", "text": "\n\n".join(developer_texts)}
            if system is None:
                system = [developer_block]
            elif isinstance(system, list):
                system.append(developer_block)
            else:
                system = [{"type": "text", "text": system}, developer_block]

        messages = _normalize_anthropic_messages(messages)

        body: dict[str, Any] = {
            "model": context.model or request.model,
            # ``max_tokens: 0`` is a valid Anthropic value (cache pre-warm without
            # generation) — only an absent value defaults to 16384.
            "max_tokens": (
                16384 if request.params.max_tokens is None else request.params.max_tokens
            ),
            "messages": messages,
        }

        if system:
            body["system"] = system
        if context.stream:
            body["stream"] = True

        # Resolve thinking first: Claude models reject temperature/top_p and
        # forced tool_choice while thinking is enabled, so the decision must
        # be made before those fields are emitted. The Anthropic provider may
        # also target non-Claude models through a custom base_url (Kimi,
        # DeepSeek, OpenRouter's Anthropic-compatible endpoint, ...), which
        # accept these parameters alongside thinking — the Claude-specific
        # constraints only apply to Claude models.
        thinking_result = convert_to_anthropic(request.params.thinking)
        thinking_enabled = bool(
            thinking_result
            and thinking_result.get("thinking", {}).get("type") in ("enabled", "adaptive")
        )
        is_claude_model = "claude" in (context.model or request.model or "").lower()

        if not thinking_enabled or not is_claude_model:
            _set_temperature_and_top_p(body, request.params)
        if request.params.stop is not None:
            stop = request.params.stop
            body["stop_sequences"] = [stop] if isinstance(stop, str) else stop

        tools = list(request.tools) if request.tools else []
        if (
            request.params.openai
            and request.params.openai.web_search_options
            and not any(isinstance(t, WebSearchTool) for t in tools)
        ):
            # Map OpenAI web_search_options to Anthropic WebSearchTool fields.
            # search_context_size is OpenAI-specific and has no Anthropic equivalent.
            wso = request.params.openai.web_search_options
            user_location = self._extract_user_location(wso)
            tools.append(
                WebSearchTool(
                    name="web_search",
                    type="web_search_20250305",
                    user_location=user_location,
                )
            )

        # Enforce the allowed_tools hard constraint: Anthropic has no
        # allowed_tools concept, so the tool set is filtered to the allowed
        # subset and the mode is mapped onto tool_choice.
        if isinstance(request.tool_choice, ToolChoiceAllowedTools):
            allowed_names = {
                t.get("name", "")
                for t in request.tool_choice.allowed_tools.tools
                if isinstance(t, dict)
            }
            if allowed_names:
                tools = [t for t in tools if getattr(t, "name", None) in allowed_names]

        if tools:
            body["tools"] = self._build_tools(tools)
        if request.tool_choice:
            body["tool_choice"] = self._build_tool_choice(request.tool_choice)

        # OpenAI ``parallel_tool_calls: false`` and the Anthropic protocol's
        # ``disable_parallel_tool_use`` flag must both end up inside
        # ``tool_choice.disable_parallel_tool_use`` for the Anthropic API.
        # We only emit a tool_choice when tools are actually present; without
        # tools the flag is dropped because it has no meaning.
        disable_parallel = (
            request.extra.get("disable_parallel_tool_use") is True
            or request.extra.get("parallel_tool_calls") is False
            or (
                request.params.openai is not None
                and request.params.openai.parallel_tool_calls is False
            )
        )
        if disable_parallel and tools:
            tool_choice = body.get("tool_choice")
            if isinstance(tool_choice, dict):
                tool_choice["disable_parallel_tool_use"] = True
            else:
                body["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}

        if thinking_result:
            body.update(thinking_result)

        if request.params.anthropic:
            anthropic_params = request.params.anthropic

            if anthropic_params.top_k is not None:
                body["top_k"] = anthropic_params.top_k

            if anthropic_params.cache_control is not None:
                body["cache_control"] = anthropic_params.cache_control

            if anthropic_params.container is not None:
                body["container"] = anthropic_params.container

            if anthropic_params.inference_geo is not None:
                body["inference_geo"] = anthropic_params.inference_geo

            if anthropic_params.output_config is not None:
                body["output_config"] = anthropic_params.output_config

            if anthropic_params.service_tier is not None:
                body["service_tier"] = anthropic_params.service_tier

            if anthropic_params.context_management is not None:
                body["context_management"] = anthropic_params.context_management

            if anthropic_params.disable_parallel_tool_use is not None:
                body["disable_parallel_tool_use"] = anthropic_params.disable_parallel_tool_use

        # Claude models reject a forced tool_choice (``any``/``tool``) while
        # thinking is enabled. Preserve the caller's explicit tool constraint
        # and disable thinking for this request instead of failing — mirrors
        # cc-switch's forced-tool_choice handling. output_config (effort) is
        # invalid once thinking is disabled, so it is dropped too, and
        # temperature/top_p (rejected while thinking was on) are restored.
        # Non-Claude models accept forced tool_choice alongside thinking, so
        # the fallback is skipped for them.
        tool_choice = body.get("tool_choice")
        if (
            thinking_enabled
            and is_claude_model
            and isinstance(tool_choice, dict)
            and tool_choice.get("type") in ("any", "tool")
        ):
            body["thinking"] = {"type": "disabled"}
            body.pop("output_config", None)
            _set_temperature_and_top_p(body, request.params)

        # Merge metadata into Anthropic ``metadata``.
        metadata: dict[str, Any] = {}
        if request.params.anthropic and request.params.anthropic.metadata:
            metadata.update(request.params.anthropic.metadata)
        if request.params.openai and request.params.openai.metadata:
            openai_user_id = request.params.openai.metadata.get("user_id")
            if openai_user_id is not None and "user_id" not in metadata:
                metadata["user_id"] = openai_user_id
        if request.user and not metadata.get("user_id"):
            metadata["user_id"] = request.user
        if metadata:
            body["metadata"] = metadata

        # Map OpenAI ``service_tier`` to Anthropic values when no Anthropic-native
        # value was already provided.
        if (
            "service_tier" not in body
            and request.params.openai
            and request.params.openai.service_tier
        ):
            mapped_tier = _map_openai_service_tier(request.params.openai.service_tier)
            if mapped_tier is not None:
                body["service_tier"] = mapped_tier

        if request.extra:
            if request.extra.get("responses_tools") is not None:
                logger.warning(
                    "responses_tools is not supported by Anthropic provider and will be ignored."
                )
            for key, value in request.extra.items():
                if value is None or key in _NON_ANTHROPIC_EXTRA_KEYS:
                    continue
                body[key] = value

        return body

    def parse_provider_response(
        self, response: dict[str, Any], model: str | None = None, **kwargs: Any
    ) -> InternalResponse:
        request_id: str | None = kwargs.get("request_id")
        content_blocks = response.get("content", [])
        output = self.parse_content_blocks(content_blocks)

        usage, usage_extras = parse_usage_and_provider_extras(response)
        provider_info: dict[str, Any] = {"provider": "anthropic", **usage_extras}

        stop_reason = response.get("stop_reason")
        finish_reason = map_finish_reason(stop_reason, "anthropic", "openai")

        stop_sequence = response.get("stop_sequence")
        container = response.get("container")
        service_tier = response.get("service_tier")
        stop_details = response.get("stop_details")
        if stop_sequence:
            provider_info["stop_sequence"] = stop_sequence
        if container:
            provider_info["container"] = container
        if service_tier:
            provider_info.setdefault("service_tier", service_tier)
        if stop_details:
            provider_info["stop_details"] = stop_details
        # Cache-diagnostics beta (cache-diagnosis-2026-04-07): the response's
        # message-level ``diagnostics`` object. An explicit ``null`` is a
        # meaningful state (no divergence), so presence is keyed on the raw
        # field, not on the value.
        if "diagnostics" in response:
            provider_info["diagnostics"] = response["diagnostics"]

        return InternalResponse(
            id=response.get("id") or generate_response_id(),
            model=model or "unknown",
            output=output,
            usage=usage,
            finish_reason=finish_reason,
            request_id=request_id,
            provider_info=provider_info,
        )

    def get_chunk_converter(self, model: str = "", request_id: str = ""):
        """Return an Anthropic-specific chunk converter.

        Converts Anthropic SSE streaming events (message_start,
        content_block_start, content_block_delta, etc.) into canonical
        OpenAI ``chat.completion.chunk`` dicts.
        """
        return AnthropicChunkConverter(model=model, request_id=request_id)

    def _build_tools(self, tools: list) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for tool in tools:
            if isinstance(tool, FunctionTool):
                tool_def = {
                    "name": tool.name,
                    "input_schema": _normalize_input_schema(tool.parameters),
                }
                if tool.description:
                    tool_def["description"] = tool.description
                if tool.strict:
                    tool_def["strict"] = True
                if tool.eager_input_streaming is not None:
                    tool_def["eager_input_streaming"] = tool.eager_input_streaming
                if tool.input_examples:
                    tool_def["input_examples"] = tool.input_examples
                self._add_common_tool_fields(tool_def, tool)
                result.append(tool_def)
            elif isinstance(tool, CustomTool):
                # Responses API custom tools (e.g. the Codex CLI ``exec`` tool)
                # have no Anthropic native equivalent. Expose them as plain
                # function tools so the model can see and invoke them. Custom
                # tools without a grammar/format carry raw text input (e.g.
                # the Codex ``exec`` tool takes raw JavaScript source), but
                # Anthropic-compatible endpoints require an object schema, so
                # declare a single ``input`` property holding the raw text.
                # The response side unwraps it back to the plain string the
                # client expects (see unwrap_custom_tool_arguments).
                tool_def: dict[str, Any] = {
                    "name": tool.name,
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "input": {
                                "type": "string",
                                "description": (
                                    "Raw string input for the original custom tool. "
                                    "Preserve formatting exactly and follow the "
                                    "original tool definition embedded in the "
                                    "description."
                                ),
                            }
                        },
                        "required": ["input"],
                    },
                }
                tool_def["description"] = _custom_tool_description(tool)
                result.append(tool_def)
            elif isinstance(tool, WebSearchTool):
                tool_def = {
                    "type": tool.type,
                    "name": tool.name,
                }
                if tool.allowed_domains:
                    tool_def["allowed_domains"] = tool.allowed_domains
                if tool.blocked_domains:
                    tool_def["blocked_domains"] = tool.blocked_domains
                if tool.max_uses is not None:
                    tool_def["max_uses"] = tool.max_uses
                if tool.response_inclusion is not None:
                    # web_search_20260318+: "full" | "excluded".
                    tool_def["response_inclusion"] = tool.response_inclusion
                if tool.user_location:
                    tool_def["user_location"] = {
                        "type": tool.user_location.type,
                    }
                    if tool.user_location.city:
                        tool_def["user_location"]["city"] = tool.user_location.city
                    if tool.user_location.country:
                        tool_def["user_location"]["country"] = tool.user_location.country
                    if tool.user_location.region:
                        tool_def["user_location"]["region"] = tool.user_location.region
                    if tool.user_location.timezone:
                        tool_def["user_location"]["timezone"] = tool.user_location.timezone
                self._add_common_tool_fields(tool_def, tool)
                result.append(tool_def)
            elif isinstance(tool, WebFetchTool):
                tool_def = {
                    "type": tool.type,
                    "name": tool.name,
                }
                if tool.allowed_domains:
                    tool_def["allowed_domains"] = tool.allowed_domains
                if tool.blocked_domains:
                    tool_def["blocked_domains"] = tool.blocked_domains
                if tool.citations:
                    tool_def["citations"] = tool.citations
                if tool.max_content_tokens is not None:
                    tool_def["max_content_tokens"] = tool.max_content_tokens
                if tool.max_uses is not None:
                    tool_def["max_uses"] = tool.max_uses
                if tool.response_inclusion is not None:
                    # web_fetch_20260318+: "full" | "excluded"; mirrors web_search.
                    tool_def["response_inclusion"] = tool.response_inclusion
                self._add_common_tool_fields(tool_def, tool)
                result.append(tool_def)
            elif isinstance(tool, OpenAIToolSearchTool):
                # Convert OpenAI Responses tool_search to a standard function
                # tool so Anthropic models can recognize and invoke it.
                tool_def = {
                    "name": "tool_search",
                    "description": (
                        "Search for tools available to the assistant. "
                        "Use this when the user asks for a specific tool "
                        "or capability that you don't currently have."
                    ),
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to find relevant tools",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of tools to return (default: 10)",
                                "default": 10,
                            },
                        },
                        "required": ["query"],
                    },
                }
                result.append(tool_def)
            elif isinstance(
                tool,
                (BashTool, CodeExecutionTool, MemoryTool, TextEditorTool, ToolSearchTool),
            ):
                tool_def = {
                    "type": tool.type,
                    "name": tool.name,
                }
                if isinstance(tool, (BashTool, MemoryTool, TextEditorTool)) and tool.input_examples:
                    tool_def["input_examples"] = tool.input_examples
                if isinstance(tool, TextEditorTool) and tool.max_characters is not None:
                    tool_def["max_characters"] = tool.max_characters
                self._add_common_tool_fields(tool_def, tool)
                result.append(tool_def)
            elif isinstance(tool, dict):
                result.append(tool)
        return result

    def _add_common_tool_fields(self, tool_def: dict[str, Any], tool: Any) -> None:
        """Add common tool fields that are shared across multiple tool types.

        Fields: allowed_callers, defer_loading, strict, cache_control
        """
        if tool.allowed_callers:
            tool_def["allowed_callers"] = tool.allowed_callers
        if tool.defer_loading is not None:
            tool_def["defer_loading"] = tool.defer_loading
        if tool.strict:
            tool_def["strict"] = True
        cache_control = getattr(tool, "cache_control", None)
        if cache_control:
            tool_def["cache_control"] = cache_control

    @staticmethod
    def _extract_user_location(options: dict[str, Any]) -> UserLocation | None:
        """Extract Anthropic UserLocation from OpenAI web_search_options dict.

        OpenAI format nests location under ``user_location.approximate``;
        Anthropic expects the fields flat under ``user_location``.
        """
        ul = options.get("user_location")
        if not isinstance(ul, dict):
            return None
        if ul.get("type") != "approximate":
            return None
        approx = ul.get("approximate", {})
        if not isinstance(approx, dict):
            return None
        return UserLocation(
            type="approximate",
            city=approx.get("city"),
            region=approx.get("region"),
            country=approx.get("country"),
            timezone=approx.get("timezone"),
        )

    def _build_tool_choice(self, tool_choice: Any) -> Any:
        if tool_choice is None:
            return None

        if isinstance(tool_choice, ToolChoice):
            result = {"type": _TOOL_CHOICE_MODE_TO_ANTHROPIC.get(tool_choice.mode, "auto")}
            if tool_choice.disable_parallel_tool_use is not None:
                result["disable_parallel_tool_use"] = tool_choice.disable_parallel_tool_use
            return result

        if isinstance(tool_choice, ToolChoiceNamed):
            result = {"type": "tool"}
            if tool_choice.name:
                result["name"] = tool_choice.name
            if tool_choice.disable_parallel_tool_use is not None:
                result["disable_parallel_tool_use"] = tool_choice.disable_parallel_tool_use
            return result

        if isinstance(tool_choice, ToolChoiceFunction):
            return {"type": "tool", "name": tool_choice.name}

        if isinstance(tool_choice, ToolChoiceAllowedTools):
            # The tool set was already filtered to the allowed subset by the
            # request builder; map the selection mode onto Anthropic's
            # tool_choice ("any" = required).
            return {
                "type": _TOOL_CHOICE_MODE_TO_ANTHROPIC.get(tool_choice.allowed_tools.mode, "auto")
            }

        if isinstance(tool_choice, str):
            return {"type": _TOOL_CHOICE_MODE_TO_ANTHROPIC.get(tool_choice, "auto")}

        if isinstance(tool_choice, dict):
            if tool_choice.get("type") == "function" and "function" in tool_choice:
                return {
                    "type": "tool",
                    "name": tool_choice["function"].get("name"),
                }
            if tool_choice.get("type") in ("tool", "any", "auto", "none"):
                return tool_choice

        return None
