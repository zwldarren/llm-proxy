# src/llm_proxy/protocols/openai/formatting.py
"""OpenAI response formatting mixin.

Handles formatting of internal Unified models to OpenAI response format,
including response building, tool formatting, and message formatting.
Used by OpenAIProtocolSerializer (protocols/openai/serializer.py).
"""

import copy
import logging
from typing import TYPE_CHECKING, Any

import orjson

from llm_proxy.models import (
    AudioBlock,
    ContentBlock,
    FileBlock,
    ImageBlock,
    InternalResponse,
    RedactedThinkingBlock,
    RefusalBlock,
    TextBlock,
    ThinkingBlock,
)
from llm_proxy.models.types import ChoiceLogprobs

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from llm_proxy.serialization.format_context import FormatContext


def _citation_to_annotation(citation) -> dict[str, Any]:
    """Convert a Citation object to an OpenAI annotation dict."""
    if isinstance(citation, dict):
        return citation
    ann: dict[str, Any] = {"type": getattr(citation, "type", "char_location")}
    for attr in [
        "cited_text",
        "document_index",
        "document_title",
        "start_char_index",
        "end_char_index",
        "start_page_number",
        "end_page_number",
        "start_block_index",
        "end_block_index",
        "encrypted_index",
        "title",
        "url",
        "search_result_index",
        "source",
    ]:
        val = getattr(citation, attr, None)
        if val is not None:
            ann[attr] = val
    return ann


class OpenAIFormattingMixin:
    """Mixin for OpenAI response formatting methods.

    Provides format_response, format_content_blocks, and _format_output_to_message.
    """

    def format_response(
        self, response: InternalResponse, context: FormatContext | None = None
    ) -> dict[str, Any]:
        """Format InternalResponse to OpenAI response format."""
        import time

        choices: list[dict[str, Any]] = []
        num_choices = 1
        if context is not None and context.n is not None and context.n > 0:
            num_choices = context.n

        # Build outputs for each choice from provider data when available.
        # Guard against empty primary output with non-empty choices_outputs.
        all_outputs: list[list[ContentBlock]] = []
        if response.output and response.choices_outputs:
            all_outputs = [response.output, *response.choices_outputs]
        elif response.choices_outputs:
            all_outputs = list(response.choices_outputs)
        else:
            all_outputs = [response.output]

        # Pre-format all available outputs so deepcopy works on plain dicts, not live objects
        formatted_messages: list[dict[str, Any]] = [
            self._format_output_to_message(out, response) for out in all_outputs
        ]

        # Build per-choice metadata list aligned with all_outputs indices
        choice_meta = response.choices_metadata

        for i in range(num_choices):
            if i < len(formatted_messages):
                message = copy.deepcopy(formatted_messages[i])
            else:
                # Fallback: reuse the last available choice output (deep-copied).
                # The provider returned fewer distinct outputs than requested n;
                # remaining choices are duplicates of the last real output.
                logger.warning(
                    "Provider returned %d distinct outputs but n=%d was requested; "
                    "duplicating last output for remaining choices",
                    len(formatted_messages),
                    num_choices,
                )
                message = copy.deepcopy(formatted_messages[-1])

            # Use per-choice metadata when available, fall back to response-level defaults
            if i == 0:
                finish_reason = response.finish_reason or "stop"
            elif i - 1 < len(choice_meta) and choice_meta[i - 1].finish_reason is not None:
                finish_reason = choice_meta[i - 1].finish_reason
            else:
                finish_reason = response.finish_reason or "stop"

            choice_entry: dict[str, Any] = {
                "index": i,
                "message": message,
                "finish_reason": finish_reason,
            }
            choices.append(choice_entry)

        result: dict[str, Any] = {
            "id": response.id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": response.model,
            "choices": choices,
        }

        if response.usage:
            usage_dict: dict[str, Any] = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            if response.usage.completion_tokens_details:
                details = response.usage.completion_tokens_details
                usage_dict["completion_tokens_details"] = {
                    k: v
                    for k, v in [
                        ("accepted_prediction_tokens", details.accepted_prediction_tokens),
                        ("audio_tokens", details.audio_tokens),
                        ("reasoning_tokens", details.reasoning_tokens),
                        ("rejected_prediction_tokens", details.rejected_prediction_tokens),
                    ]
                    if v is not None
                }
            if response.usage.prompt_tokens_details:
                details = response.usage.prompt_tokens_details
                usage_dict["prompt_tokens_details"] = {
                    k: v
                    for k, v in [
                        ("audio_tokens", details.audio_tokens),
                        ("cached_tokens", details.cached_tokens),
                    ]
                    if v is not None
                }
            result["usage"] = usage_dict

        if response.provider_info.get("system_fingerprint"):
            result["system_fingerprint"] = response.provider_info["system_fingerprint"]

        if response.provider_info.get("service_tier"):
            result["service_tier"] = response.provider_info["service_tier"]

        if response.logprobs or choice_meta:
            # Add logprobs to each choice using per-choice metadata when available
            for i, choice in enumerate(choices):
                logprobs_for_choice: ChoiceLogprobs | None = None
                if i == 0:
                    logprobs_for_choice = response.logprobs
                elif i - 1 < len(choice_meta):
                    logprobs_for_choice = choice_meta[i - 1].logprobs

                if logprobs_for_choice:
                    logprobs_dict: dict[str, Any] = {}
                    if logprobs_for_choice.content:
                        logprobs_dict["content"] = [
                            {
                                "token": t.token,
                                "logprob": t.logprob,
                                **({"bytes": t.bytes} if t.bytes is not None else {}),
                            }
                            for t in logprobs_for_choice.content
                        ]
                    if logprobs_for_choice.refusal:
                        logprobs_dict["refusal"] = [
                            {
                                "token": t.token,
                                "logprob": t.logprob,
                                **({"bytes": t.bytes} if t.bytes is not None else {}),
                            }
                            for t in logprobs_for_choice.refusal
                        ]
                    if logprobs_dict:
                        choice["logprobs"] = logprobs_dict

        return result

    def format_content_blocks(self, blocks: list[ContentBlock]) -> Any:
        """Format ContentBlock list to OpenAI content format."""
        from llm_proxy.serialization.openai.converter import content_to_openai_parts

        if not blocks:
            return None
        return content_to_openai_parts(blocks)

    def _format_output_to_message(
        self, output: list[ContentBlock], response: InternalResponse | None = None
    ) -> dict[str, Any]:
        """Format output blocks to OpenAI message format.

        Block types without a direct OpenAI Chat equivalent are serialized as
        structured text so no information is silently dropped during cross-protocol
        routing (e.g. Anthropic provider → OpenAI protocol).
        """
        from llm_proxy.models import (
            CustomToolUseBlock,
            DocumentBlock,
            ServerToolUseBlock,
            ToolResultBlock,
            ToolUseBlock,
        )
        from llm_proxy.models.content_blocks.anthropic_builtin import (
            BashCodeExecutionToolResultBlock,
            CodeExecutionToolResultBlock,
            ContainerUploadBlock,
            SearchResultBlock,
            TextEditorCodeExecutionToolResultBlock,
            ToolReferenceBlock,
            ToolSearchToolResultBlock,
            WebFetchToolResultBlock,
            WebSearchResultContentBlock,
            WebSearchToolResultBlock,
        )

        message: dict[str, Any] = {"role": "assistant"}

        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        collected_annotations: list[dict[str, Any]] = []

        for block in output:
            match block:
                case TextBlock():
                    text_parts.append(block.text)
                    if block.citations:
                        collected_annotations.extend(
                            _citation_to_annotation(c) for c in block.citations
                        )
                case ToolUseBlock():
                    tc: dict[str, Any] = {
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": orjson.dumps(block.input).decode(),
                        },
                    }
                    if block.extra.get("thought_signature"):
                        tc["thought_signature"] = block.extra["thought_signature"]
                    tool_calls.append(tc)
                case CustomToolUseBlock():
                    # Chat Completions clients only understand ``type: "function"``
                    # tool calls. Re-wrap the freeform custom-tool input in the
                    # ``{"content": ...}`` envelope used by the custom-tool bridge
                    # (mirror of OpenAIToolsHandler._custom_tool_to_function).
                    tool_calls.append(
                        {
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": orjson.dumps({"content": block.input}).decode(),
                            },
                        }
                    )
                case ThinkingBlock():
                    message["reasoning_content"] = block.thinking
                    if block.signature:
                        message["reasoning_signature"] = block.signature
                case RedactedThinkingBlock():
                    message["reasoning_content"] = "[redacted]"
                    message["reasoning_is_redacted"] = True
                case RefusalBlock():
                    message["refusal"] = block.refusal
                case AudioBlock():
                    if block.source:
                        message["audio"] = {
                            "id": block.source.id if hasattr(block.source, "id") else None,
                            "data": block.source.data if block.source.type == "base64" else None,
                            "expires_at": (
                                block.source.expires_at
                                if hasattr(block.source, "expires_at")
                                else None
                            ),
                            "transcript": (
                                block.source.transcript
                                if hasattr(block.source, "transcript")
                                else None
                            ),
                        }
                case ServerToolUseBlock():
                    text_parts.append(
                        orjson.dumps(
                            {
                                "type": "server_tool_use",
                                "id": block.id,
                                "name": block.name,
                                "input": block.input,
                            }
                        ).decode()
                    )
                case WebSearchToolResultBlock():
                    text_parts.append(self._format_anthropic_block_as_text("web_search", block))
                case WebFetchToolResultBlock():
                    text_parts.append(self._format_anthropic_block_as_text("web_fetch", block))
                case WebSearchResultContentBlock():
                    text_parts.append(
                        orjson.dumps({"url": block.url, "title": block.title}).decode()
                    )
                case (
                    CodeExecutionToolResultBlock()
                    | BashCodeExecutionToolResultBlock()
                    | TextEditorCodeExecutionToolResultBlock()
                ):
                    text_parts.append(self._format_code_execution_block(block))
                case SearchResultBlock():
                    texts = []
                    if block.content:
                        for cb in block.content:
                            if isinstance(cb, TextBlock):
                                texts.append(cb.text)
                    text_parts.append(
                        orjson.dumps(
                            {
                                "title": block.title,
                                "content": texts,
                                "file_id": block.file_id,
                            }
                        ).decode()
                    )
                case ContainerUploadBlock():
                    text_parts.append(f"[container_upload: {block.filename or block.file_id}]")
                case ToolReferenceBlock():
                    text_parts.append(f"[tool_reference: {block.tool_name or block.tool_id}]")
                case ToolSearchToolResultBlock():
                    text_parts.append(self._format_anthropic_block_as_text("tool_search", block))
                case ImageBlock():
                    image_url = self._format_image_block(block)
                    if image_url:
                        text_parts.append(f"![image]({image_url})")
                case DocumentBlock():
                    media_type = block.source.media_type or "unknown"
                    title = f" '{block.title}'" if block.title else ""
                    text_parts.append(f"[Document{title}: {media_type}]")
                case FileBlock():
                    name = block.filename or block.file_id or "unknown"
                    text_parts.append(f"[File: {name}]")
                case ToolResultBlock():
                    if isinstance(block.content, str):
                        text_parts.append(block.content)
                    else:
                        texts: list[str] = []
                        for cb in block.content:
                            if isinstance(cb, TextBlock):
                                if cb.text:
                                    texts.append(cb.text)
                            elif isinstance(cb, RefusalBlock):
                                if cb.refusal:
                                    texts.append(f"[Refusal: {cb.refusal}]")
                            else:
                                texts.append(str(cb))
                        text_parts.append(" ".join(texts))

        has_refusal = "refusal" in message

        if text_parts:
            message["content"] = " ".join(text_parts)
        else:
            message["content"] = None

        # OpenAI API requires content to be None when refusal is present
        if has_refusal:
            message["content"] = None

        if tool_calls:
            message["tool_calls"] = tool_calls

        if response is not None:
            if "reasoning_content" not in message:
                reasoning = response.get_thinking_content()
                if reasoning:
                    message["reasoning_content"] = reasoning
            if "refusal" not in message:
                refusal = response.get_refusal()
                if refusal:
                    message["refusal"] = refusal
                    has_refusal = True
                    message["content"] = None
            if "audio" not in message:
                audio = response.get_audio()
                if audio:
                    message["audio"] = audio
            extra_annotations = response.provider_info.get("annotations", [])
            if extra_annotations:
                collected_annotations = extra_annotations + collected_annotations

        if collected_annotations:
            message["annotations"] = collected_annotations

        return message

    @staticmethod
    def _format_anthropic_block_as_text(block_type: str, block) -> str:
        """Serialize an Anthropic-specific tool result block as structured JSON text.

        Used as a fallback when routing Anthropic provider output through
        OpenAI protocol, which has no native equivalent for these block types.
        """
        content = getattr(block, "content", "")
        if isinstance(content, list):
            content = str(content)
        return orjson.dumps(
            {
                "type": f"{block_type}_result",
                "tool_use_id": getattr(block, "tool_use_id", ""),
                "content": content,
            }
        ).decode()

    @staticmethod
    def _format_code_execution_block(block) -> str:
        """Serialize a code execution result block as markdown code text.

        Anthropic code execution blocks have no OpenAI Chat equivalent.
        """
        content = getattr(block, "content", "")
        return f"```\n{content}\n```"

    def _format_image_block(self, block) -> str | None:
        """Format an ImageBlock to an OpenAI image_url string.

        Returns a data URL for base64 images, or the URL directly for url type.
        Returns None for file_id type (not supported in chat completions response).
        """
        source = block.source
        match source.type:
            case "base64":
                media_type = source.media_type or "image/png"
                return f"data:{media_type};base64,{source.data}"
            case "url":
                return source.data
            case "file_id":
                return None
