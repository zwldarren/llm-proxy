# src/llm_proxy/protocols/openai/parsing.py
"""OpenAI request parsing mixin.

Handles parsing of OpenAI-format requests into internal Unified models.
Used by OpenAIProtocolSerializer (protocols/openai/serializer.py); the
provider side reuses the shared content parsers from
``llm_proxy.serialization.content_parsers``.
"""

from typing import Any

import orjson

from llm_proxy.core.thinking import normalize_thinking
from llm_proxy.models import (
    AudioBlock,
    ContentBlock,
    ConversationContext,
    CustomToolUseBlock,
    GenerationParams,
    InternalRequest,
    Message,
    OpenAISpecificParams,
    RefusalBlock,
    RequestMetadata,
    ResponseFormat,
    StreamOptions,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from llm_proxy.models.tools import (
    FunctionTool,
    ToolChoice,
    ToolChoiceFunction,
    ToolDefinition,
)
from llm_proxy.models.tools.openai_builtin import WebSearchTool
from llm_proxy.models.types import AudioSource
from llm_proxy.serialization.content_parsers import (
    parse_audio_block_openai,
    parse_file_block_openai,
    parse_image_block_openai,
    parse_reasoning_content,
    parse_text_block,
    parse_video_block_openai,
)


class OpenAIParsingMixin:
    """Mixin for OpenAI request parsing methods.

    Provides parse_request, parse_conversation, parse_content_blocks,
    and related parameter parsing methods.
    """

    @staticmethod
    def _known_request_fields() -> set[str]:
        """Fields that are handled explicitly and not stored in extra."""
        return {
            "model",
            "messages",
            "temperature",
            "top_p",
            "max_tokens",
            "max_completion_tokens",
            "stop",
            "presence_penalty",
            "frequency_penalty",
            "n",
            "stream",
            "stream_options",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "user",
            "request_id",
            "response_format",
            "seed",
            "logprobs",
            "top_logprobs",
            "service_tier",
            "verbosity",
            "store",
            "metadata",
            "prompt_cache_key",
            "prompt_cache_retention",
            "safety_identifier",
            "audio",
            "modalities",
            "reasoning_effort",
            "prediction",
            "web_search_options",
            "thinking",
            "logit_bias",
        }

    def parse_request(self, data: dict[str, Any]) -> InternalRequest:
        """Parse OpenAI request format to InternalRequest."""
        from llm_proxy.core.exceptions import ValidationError

        try:
            model = data["model"]
        except KeyError as e:
            raise ValidationError(f"Missing required field: {e}") from e

        messages = data.get("messages", [])
        conversation = self.parse_conversation(messages)

        params = self._parse_params(data)

        metadata = RequestMetadata(
            request_id=data.get("request_id"),
            user=data.get("user"),
        )

        # Parse stream options. include_usage defaults to False, matching the
        # OpenAI API's own default: the proxy does not request the terminal
        # usage chunk on the client's behalf. When usage is absent, billing
        # falls back to token estimation (see billing.cost.calculate_cost).
        stream_opts = None
        stream_options = data.get("stream_options")
        if stream_options:
            stream_opts = StreamOptions(
                include_usage=stream_options.get("include_usage", False),
                include_obfuscation=stream_options.get("include_obfuscation"),
            )

        extra = {
            k: v for k, v in data.items() if k not in self._known_request_fields() and v is not None
        }
        if data.get("parallel_tool_calls") is False:
            extra["disable_parallel_tool_use"] = True

        return InternalRequest(
            model=model,
            conversation=conversation,
            params=params,
            tools=self._parse_tools(data.get("tools")),
            tool_choice=self._parse_tool_choice(data.get("tool_choice")),
            n=data.get("n"),
            stream=data.get("stream", False),
            stream_options=stream_opts,
            metadata=metadata,
            extra=extra,
        )

    def parse_conversation(self, messages: list[dict]) -> ConversationContext:
        """Parse OpenAI messages to ConversationContext.

        Handles all OpenAI message types:
        - system: System messages
        - developer: Developer messages (treated as system)
        - user: User messages
        - assistant: Assistant messages (with optional tool_calls)
        - tool: Tool result messages
        """
        system_messages: list[SystemMessage] = []
        conv_messages: list[Message] = []
        # Map tool_call_id → function_name for populating ToolResultBlock.name
        tool_call_id_to_name: dict[str, str] = {}
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    if isinstance(tc, dict) and tc.get("type") == "function":
                        func = tc.get("function") or {}
                        tc_id = tc.get("id", "")
                        func_name = func.get("name", "")
                        if tc_id and func_name:
                            tool_call_id_to_name[tc_id] = func_name

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")
            name = msg.get("name")

            # Handle system and developer messages
            if role in ("system", "developer"):
                text_content = self._extract_text_content(content)
                system_messages.append(
                    SystemMessage.from_text(role=role, text=text_content, name=name)
                )
                continue

            # Handle tool results
            if role == "tool":
                tool_call_id = msg.get("tool_call_id", "")
                content_blocks = self.parse_content_blocks(content)
                content_str = " ".join(b.text for b in content_blocks if isinstance(b, TextBlock))
                func_name = tool_call_id_to_name.get(tool_call_id, name)
                conv_messages.append(
                    Message(
                        role="tool",
                        content=[
                            ToolResultBlock(
                                tool_use_id=tool_call_id,
                                content=content_str,
                                name=func_name,
                            ),
                        ],
                        name=name,
                    )
                )
                continue

            # Handle user and assistant messages
            content_blocks = self.parse_content_blocks(content)

            # For assistant messages, also parse tool_calls if present
            if role == "assistant":
                tool_calls = msg.get("tool_calls")

                if tool_calls:
                    for tc in tool_calls:
                        if not isinstance(tc, dict):
                            continue
                        tc_type = tc.get("type")
                        if tc_type == "function":
                            func = tc.get("function") or {}
                            args_str = func.get("arguments", "{}")
                            try:
                                args = (
                                    orjson.loads(args_str)
                                    if isinstance(args_str, str)
                                    else args_str
                                )
                            except orjson.JSONDecodeError:
                                args = {}
                            content_blocks.append(
                                ToolUseBlock(
                                    id=tc.get("id", ""),
                                    name=func.get("name", ""),
                                    input=args,
                                    extra={"thought_signature": tc.get("thought_signature")}
                                    if tc.get("thought_signature")
                                    else {},
                                )
                            )
                        elif tc_type == "custom":
                            custom = tc.get("custom") or {}
                            content_blocks.append(
                                CustomToolUseBlock(
                                    id=tc.get("id", ""),
                                    name=custom.get("name", ""),
                                    input=custom.get("input", ""),
                                )
                            )

                # Parse reasoning_content into ThinkingBlock. Reasoning always
                # precedes the answer text, so insert at the front of the block
                # list — appending at the end would emit model_output before
                # thought on the Gemini Interactions variant, which the API
                # rejects ("Model turns with thought summaries must start with
                # a thought block in thinking models").
                reasoning_block = parse_reasoning_content(msg)
                if reasoning_block is not None:
                    content_blocks.insert(0, reasoning_block)

                # Parse refusal
                refusal = msg.get("refusal")
                if refusal and isinstance(refusal, str):
                    content_blocks.append(RefusalBlock(refusal=refusal))

                # Parse audio for multi-turn
                audio_data = msg.get("audio")
                if audio_data and isinstance(audio_data, dict):
                    content_blocks.append(
                        AudioBlock(
                            source=AudioSource(
                                type="base64",
                                data=audio_data.get("data", "") or "",
                                media_type="audio/wav",
                                id=audio_data.get("id"),
                            )
                        )
                    )

                # Note: OpenAI sets content=None when there are tool calls,
                # which parse_content_blocks converts to [TextBlock(text="")].
                # We keep this empty TextBlock to match the original handler behavior.

            conv_messages.append(Message(role=role, content=content_blocks, name=name))

        return ConversationContext(system_messages=system_messages, messages=conv_messages)

    def _extract_text_content(self, content: Any) -> str:
        """Extract text from content (string or list of content parts)."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        return str(content) if content else ""

    def parse_content_blocks(self, content: Any) -> list[ContentBlock]:
        """Parse OpenAI content format to ContentBlock list.

        Supports:
        - text: plain text content
        - image_url: image from URL or base64 data URI
        - input_audio: audio input (for GPT-4o audio)
        - file: file input
        - refusal: refusal content
        """
        if content is None:
            return [TextBlock(text="")]

        if isinstance(content, str):
            return [TextBlock(text=content)]

        if isinstance(content, list):
            blocks: list[ContentBlock] = []
            for part in content:
                if isinstance(part, str):
                    blocks.append(TextBlock(text=part))
                elif isinstance(part, dict):
                    block = parse_text_block(part)
                    if block:
                        blocks.append(block)
                        continue
                    block = parse_image_block_openai(part)
                    if block:
                        blocks.append(block)
                        continue
                    block = parse_audio_block_openai(part)
                    if block:
                        blocks.append(block)
                        continue
                    block = parse_file_block_openai(part)
                    if block:
                        blocks.append(block)
                        continue
                    block = parse_video_block_openai(part)
                    if block:
                        blocks.append(block)
                        continue
                    if part.get("type") == "refusal":
                        blocks.append(RefusalBlock(refusal=part.get("refusal", "")))
                        continue
                    # Catch-all: degrade unrecognized content types to text.
                    # This handles Anthropic-native blocks when routing
                    # Anthropic protocol requests through the OpenAI serializer.
                    text = self._degrade_anthropic_block(part)
                    if text:
                        blocks.append(TextBlock(text=text))
            return blocks if blocks else [TextBlock(text="")]

        return [TextBlock(text=str(content))]

    def _parse_params(self, data: dict[str, Any]) -> GenerationParams:
        """Parse generation parameters from request.

        Handles both standard parameters and protocol-specific parameters.
        """
        # Check if max_completion_tokens was explicitly provided (for o-series models)
        max_completion_tokens = data.get("max_completion_tokens")
        # max_completion_tokens is OpenAI's preferred alias for max_tokens. Provider
        # builders (Anthropic/Gemini/Ollama/OpenAI Responses) only read the common
        # ``max_tokens`` field, so when the client sends ``max_completion_tokens``
        # without ``max_tokens`` we mirror it into the common field. An explicit
        # ``max_tokens`` always wins.
        max_tokens = data.get("max_tokens")
        if max_tokens is None and max_completion_tokens is not None:
            max_tokens = max_completion_tokens

        openai_specific = OpenAISpecificParams(
            max_completion_tokens=max_completion_tokens,
            logprobs=data.get("logprobs"),
            top_logprobs=data.get("top_logprobs"),
            service_tier=data.get("service_tier"),
            verbosity=data.get("verbosity"),
            store=data.get("store"),
            metadata=data.get("metadata"),
            prompt_cache_key=data.get("prompt_cache_key"),
            prompt_cache_retention=data.get("prompt_cache_retention"),
            safety_identifier=data.get("safety_identifier"),
            audio=data.get("audio"),
            modalities=data.get("modalities"),
            reasoning_effort=data.get("reasoning_effort"),
            prediction=data.get("prediction"),
            web_search_options=data.get("web_search_options"),
            parallel_tool_calls=data.get("parallel_tool_calls"),
            logit_bias=data.get("logit_bias"),
        )
        if not any(v is not None for v in vars(openai_specific).values()):
            openai_specific = None

        # Parse response_format
        response_format = None
        rf = data.get("response_format")
        if rf:
            rf_type = rf.get("type", "text")
            if rf_type == "json_schema":
                response_format = ResponseFormat(
                    type=rf_type,
                    json_schema=rf.get("json_schema"),
                )
            else:
                response_format = ResponseFormat(type=rf_type, json_schema=None)

        # Parse stop sequences
        stop = data.get("stop")
        if isinstance(stop, str):
            stop = [stop]

        return GenerationParams(
            max_tokens=max_tokens,
            temperature=data.get("temperature"),
            top_p=data.get("top_p"),
            stop=stop,
            frequency_penalty=data.get("frequency_penalty"),
            presence_penalty=data.get("presence_penalty"),
            seed=data.get("seed"),
            response_format=response_format,
            openai=openai_specific,
            thinking=normalize_thinking(data),
        )

    def _parse_tools(self, tools: list[dict] | None) -> list[ToolDefinition] | None:
        """Parse tools from OpenAI format to ToolDefinition list.

        Supports:
        - function tools
        - custom tools
        """
        if not tools:
            return None

        result: list[ToolDefinition] = []
        for tool in tools:
            tool_type = tool.get("type", "function")
            if tool_type == "function":
                func = tool.get("function", {})
                result.append(
                    FunctionTool(
                        name=func.get("name", ""),
                        description=func.get("description"),
                        parameters=func.get("parameters", {"type": "object"}),
                        strict=func.get("strict", False),
                    )
                )
            elif tool_type == "custom":
                from llm_proxy.models import CustomTool

                custom = tool.get("custom", {}) or {}
                format_info = custom.get("format", {}) or {}
                grammar_info = format_info.get("grammar") or {}
                result.append(
                    CustomTool(
                        name=custom.get("name", ""),
                        description=custom.get("description"),
                        format_type=format_info.get("type"),
                        grammar_definition=grammar_info.get("definition")
                        if format_info.get("type") == "grammar"
                        else None,
                        grammar_syntax=grammar_info.get("syntax")
                        if format_info.get("type") == "grammar"
                        else None,
                    )
                )
            elif tool_type in ("web_search", "web_search_preview"):
                # LiteLLM pattern: web_search / web_search_preview built-in
                # tools in the Chat Completions tools array.
                loc = tool.get("user_location")
                allowed_domains = tool.get("allowed_domains")
                blocked_domains = tool.get("blocked_domains")
                filters = tool.get("filters")
                if isinstance(filters, dict):
                    if allowed_domains is None:
                        allowed_domains = filters.get("allowed_domains")
                    if blocked_domains is None:
                        blocked_domains = filters.get("blocked_domains")
                result.append(
                    WebSearchTool(
                        name="web_search",
                        type=tool_type,
                        search_context_size=tool.get("search_context_size"),
                        allowed_domains=allowed_domains,
                        blocked_domains=blocked_domains,
                        user_location=(
                            {
                                "type": loc.get("type", "approximate"),
                                "city": loc.get("city"),
                                "country": loc.get("country"),
                                "region": loc.get("region"),
                                "timezone": loc.get("timezone"),
                            }
                            if isinstance(loc, dict)
                            else None
                        ),
                        external_web_access=tool.get("external_web_access"),
                        return_token_budget=tool.get("return_token_budget"),
                        search_content_types=tool.get("search_content_types"),
                        image_settings=tool.get("image_settings"),
                        max_uses=tool.get("max_uses"),
                    )
                )
        return result if result else None

    def _parse_tool_choice(self, tool_choice: Any | None) -> Any:
        """Parse tool_choice from request."""
        match tool_choice:
            case None:
                return None
            case str() as mode:
                return ToolChoice(mode=mode)
            case {"type": "function", "function": {"name": str(name)}}:
                return ToolChoiceFunction(name=name)
            case _:
                return tool_choice

    @staticmethod
    def _degrade_anthropic_block(part: dict[str, Any]) -> str | None:
        """Degrade an Anthropic-native content block to text.

        Used as a catch-all in parse_content_blocks when routing
        Anthropic-format content through the OpenAI protocol serializer.
        Extracts meaningful text content from various block structures.
        """
        block_type = part.get("type", "")
        if not block_type or block_type in (
            "text",
            "image_url",
            "input_audio",
            "file",
            "refusal",
            "video_url",
        ):
            return None

        match block_type:
            case "tool_use" | "server_tool_use":
                name = part.get("name", "")
                inp = part.get("input", {})
                return f"[{block_type}: {name}({inp})]"
            case "thinking":
                thinking = part.get("thinking", "")
                return f"[thinking: {thinking}]"
            case "redacted_thinking":
                return "[redacted_thinking]"
            case "document":
                source = part.get("source", {})
                data = source.get("data", "") if isinstance(source, dict) else ""
                media_type = source.get("media_type", "") if isinstance(source, dict) else ""
                title = part.get("title", "")
                parts = []
                if title:
                    parts.append(f"title={title}")
                if media_type:
                    parts.append(f"media_type={media_type}")
                if data:
                    text = data[:200]
                    parts.append(f"data={text}")
                return f"[document: {', '.join(parts)}]" if parts else "[document]"
            case _:
                content = part.get("content")
                if content is not None:
                    if isinstance(content, str):
                        return content
                    if isinstance(content, list):
                        texts = []
                        for item in content:
                            if isinstance(item, dict):
                                texts.append(
                                    item.get("text", "") or item.get("data", "") or str(item)
                                )
                            elif isinstance(item, str):
                                texts.append(item)
                        return " ".join(texts) if texts else f"[{block_type}]"

                url = part.get("url", "")
                title = part.get("title", "")
                if url or title:
                    parts = []
                    if title:
                        parts.append(f"title={title}")
                    if url:
                        parts.append(f"url={url}")
                    return f"[{block_type}: {', '.join(parts)}]"

                return f"[{block_type}]"
