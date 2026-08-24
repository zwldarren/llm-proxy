"""Shared content parsing and formatting mixin for Anthropic format.

Used by both AnthropicProtocolSerializer and AnthropicProviderSerializer
to avoid duplicating content block conversion logic.
"""

from typing import TYPE_CHECKING, Any

from llm_proxy.models import (
    ContentBlock,
    RawBlock,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from llm_proxy.models.content_blocks.anthropic_builtin import (
    CacheControl,
    Caller,
    MidConversationSystemBlock,
)
from llm_proxy.models.finish_reasons import map_finish_reason
from llm_proxy.serialization.content_parsers import (
    parse_audio_block_anthropic,
    parse_file_block_anthropic,
    parse_image_block_anthropic,
)
from llm_proxy.serialization.responses_toolkit.namespace import (
    flatten_history_tool_name,
)

if TYPE_CHECKING:
    from llm_proxy.serialization.context import BuildContext


class AnthropicContentMixin:
    """Shared content parsing and formatting for Anthropic format."""

    provider_name: str = "anthropic"

    @staticmethod
    def _parse_cache_control(part: dict[str, Any]) -> CacheControl | None:
        """Extract CacheControl from a content part dict, if present."""
        raw = part.get("cache_control")
        if not raw:
            return None
        return CacheControl(
            type=raw.get("type", "ephemeral"),
            ttl=raw.get("ttl"),
        )

    def parse_content_blocks(self, content: Any) -> list[ContentBlock]:
        """Parse Anthropic content format to ContentBlock list."""
        from llm_proxy.models import (
            DocumentBlock,
            DocumentSource,
            ServerToolUseBlock,
        )
        from llm_proxy.models.content_blocks.anthropic_builtin import (
            WebSearchResultContentBlock,
            WebSearchToolResultBlock,
        )

        if content is None:
            return []

        if isinstance(content, str):
            return [TextBlock(text=content)]

        if isinstance(content, list):
            blocks: list[ContentBlock] = []
            for part in content:
                if isinstance(part, str):
                    blocks.append(TextBlock(text=part))
                elif isinstance(part, dict):
                    part_type = part.get("type", "text")

                    if part_type == "text":
                        text = part.get("text", "")
                        citations = part.get("citations")
                        cache_control = self._parse_cache_control(part)
                        blocks.append(
                            TextBlock(
                                text=text,
                                citations=citations,
                                cache_control=cache_control,
                            )
                        )
                        continue

                    block = parse_image_block_anthropic(part)
                    if block:
                        block.cache_control = self._parse_cache_control(part)
                        blocks.append(block)
                        continue
                    block = parse_audio_block_anthropic(part)
                    if block:
                        block.cache_control = self._parse_cache_control(part)
                        blocks.append(block)
                        continue
                    block = parse_file_block_anthropic(part)
                    if block:
                        block.cache_control = self._parse_cache_control(part)
                        blocks.append(block)
                        continue

                    if part_type == "document":
                        source = part.get("source", {})
                        source_type = source.get("type", "base64")
                        doc_data = source.get("data", source.get("url", source.get("file_id", "")))
                        if source_type == "content":
                            doc_data = source.get("content", "")
                        elif source_type in ("file_id", "file"):
                            source_type = "file_id"
                            doc_data = source.get("file_id", "")
                        cache_control = self._parse_cache_control(part)
                        blocks.append(
                            DocumentBlock(
                                source=DocumentSource(
                                    type=source_type,
                                    data=doc_data,
                                    media_type=source.get("media_type"),
                                ),
                                title=part.get("title"),
                                cache_control=cache_control,
                            )
                        )
                        continue

                    if part_type == "tool_use":
                        cache_control = self._parse_cache_control(part)
                        blocks.append(
                            ToolUseBlock(
                                id=part.get("id", ""),
                                name=part.get("name", ""),
                                input=part.get("input", {}),
                                cache_control=cache_control,
                            )
                        )
                        continue

                    if part_type == "server_tool_use":
                        cache_control = self._parse_cache_control(part)
                        blocks.append(
                            ServerToolUseBlock(
                                id=part.get("id", ""),
                                name=part.get("name", ""),
                                input=part.get("input", {}),
                                cache_control=cache_control,
                            )
                        )
                        continue

                    if part_type == "tool_result":
                        result_content = part.get("content", "")
                        if isinstance(result_content, list):
                            result_content = self.parse_content_blocks(result_content)
                        cache_control = self._parse_cache_control(part)
                        blocks.append(
                            ToolResultBlock(
                                tool_use_id=part.get("tool_use_id", ""),
                                content=result_content,
                                is_error=part.get("is_error", False),
                                cache_control=cache_control,
                            )
                        )
                        continue

                    if part_type == "thinking":
                        cache_control = self._parse_cache_control(part)
                        blocks.append(
                            ThinkingBlock(
                                thinking=part.get("thinking", ""),
                                signature=part.get("signature"),
                                cache_control=cache_control,
                            )
                        )
                        continue

                    if part_type == "redacted_thinking":
                        cache_control = self._parse_cache_control(part)
                        blocks.append(
                            RedactedThinkingBlock(
                                data=part.get("data", ""),
                                cache_control=cache_control,
                            )
                        )
                        continue

                    if part_type == "search_result":
                        from llm_proxy.models.content_blocks.anthropic_builtin import (
                            SearchResultBlock,
                        )

                        cache_control = self._parse_cache_control(part)
                        content_blocks = self.parse_content_blocks(part.get("content", []))
                        blocks.append(
                            SearchResultBlock(
                                source=part.get("source", ""),
                                title=part.get("title", ""),
                                content=content_blocks,
                                metadata=part.get("metadata"),
                                cache_control=cache_control,
                            )
                        )
                        continue

                    if part_type == "container_upload":
                        from llm_proxy.models.content_blocks.anthropic_builtin import (
                            ContainerUploadBlock,
                        )

                        blocks.append(
                            ContainerUploadBlock(
                                file_id=part.get("file_id"),
                                filename=part.get("filename"),
                                content=part.get("content"),
                                media_type=part.get("media_type"),
                            )
                        )
                        continue

                    if part_type == "tool_reference":
                        from llm_proxy.models.content_blocks.anthropic_builtin import (
                            ToolReferenceBlock,
                        )

                        blocks.append(
                            ToolReferenceBlock(
                                tool_id=part.get("tool_id", ""),
                                tool_name=part.get("tool_name"),
                                tool_type=part.get("tool_type"),
                            )
                        )
                        continue

                    if part_type == "web_search_tool_result":
                        result_content = part.get("content", "")
                        if isinstance(result_content, list):
                            result_content = self.parse_content_blocks(result_content)
                        caller = None
                        if part.get("caller"):
                            c = part["caller"]
                            caller = Caller(type=c.get("type", "direct"), tool_id=c.get("tool_id"))
                        blocks.append(
                            WebSearchToolResultBlock(
                                tool_use_id=part.get("tool_use_id", ""),
                                content=result_content,
                                is_error=part.get("is_error", False),
                                caller=caller,
                            )
                        )
                        continue

                    if part_type == "web_search_result":
                        blocks.append(
                            WebSearchResultContentBlock(
                                url=part.get("url", ""),
                                title=part.get("title", ""),
                                encoded_content=part.get("encoded_content", ""),
                                page_age=part.get("page_age"),
                            )
                        )
                        continue

                    if part_type == "mid_conv_system":
                        content_blocks = self.parse_content_blocks(part.get("content", []))
                        cache_control = self._parse_cache_control(part)
                        blocks.append(
                            MidConversationSystemBlock(
                                content=content_blocks,
                                cache_control=cache_control,
                            )
                        )
                        continue
            return blocks

        return [TextBlock(text=str(content))]

    def format_content_blocks(
        self,
        blocks: list[ContentBlock],
        context: BuildContext | None = None,
    ) -> Any:
        """Format ContentBlock list to Anthropic content format."""
        from llm_proxy.models import (
            AudioBlock,
            CustomToolUseBlock,
            DocumentBlock,
            FileBlock,
            ImageBlock,
            RefusalBlock,
            ServerToolUseBlock,
            ToolResultBlock,
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

        result: list[dict[str, Any]] = []
        for block in blocks:
            if isinstance(block, TextBlock):
                text_block: dict[str, Any] = {"type": "text", "text": block.text}
                if block.citations:
                    text_block["citations"] = self._format_citations(block.citations)
                if block.cache_control:
                    text_block["cache_control"] = self._format_cache_control(block.cache_control)
                result.append(text_block)
            elif isinstance(block, ToolUseBlock):
                tool_block: dict[str, Any] = {
                    "type": "tool_use",
                    "id": block.id,
                    "name": flatten_history_tool_name(
                        context.namespace_map if context else None, block.name
                    ),
                    "input": block.input,
                }
                if block.cache_control:
                    tool_block["cache_control"] = self._format_cache_control(block.cache_control)
                result.append(tool_block)
            elif isinstance(block, ServerToolUseBlock):
                server_tool_block: dict[str, Any] = {
                    "type": "server_tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                }
                if block.cache_control:
                    server_tool_block["cache_control"] = self._format_cache_control(
                        block.cache_control
                    )
                result.append(server_tool_block)
            elif isinstance(block, CustomToolUseBlock):
                import orjson

                tool_input: dict[str, Any] = {}
                try:
                    parsed = orjson.loads(block.input)
                    if isinstance(parsed, dict):
                        tool_input = parsed
                except orjson.JSONDecodeError, TypeError:
                    # Raw freeform input (e.g. Codex ``exec`` JavaScript source)
                    # is wrapped under the same ``input`` key the bridged
                    # function-tool schema declares, so the model sees a
                    # consistent format in history and echoes it back.
                    tool_input = {"input": block.input}
                custom_tool_block: dict[str, Any] = {
                    "type": "tool_use",
                    "id": block.id,
                    "name": flatten_history_tool_name(
                        context.namespace_map if context else None, block.name
                    ),
                    "input": tool_input,
                }
                result.append(custom_tool_block)
            elif isinstance(block, ThinkingBlock):
                if not block.thinking:
                    continue
                thinking_block: dict[str, Any] = {
                    "type": "thinking",
                    "thinking": block.thinking,
                }
                if block.signature:
                    thinking_block["signature"] = block.signature
                result.append(thinking_block)
            elif isinstance(block, RedactedThinkingBlock):
                result.append(
                    {
                        "type": "redacted_thinking",
                        "data": block.data,
                    }
                )
            elif isinstance(block, RefusalBlock):
                result.append(
                    {
                        "type": "refusal",
                        "refusal": block.refusal,
                    }
                )
            elif isinstance(block, ImageBlock):
                if block.source.type == "base64":
                    result.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": block.source.media_type,
                                "data": block.source.data,
                            },
                        }
                    )
                elif block.source.type == "url":
                    result.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": block.source.data,
                            },
                        }
                    )
                elif block.source.type == "file_id":
                    result.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "file",
                                "file_id": block.source.data,
                            },
                        }
                    )
            elif isinstance(block, DocumentBlock):
                if block.source.type == "base64":
                    doc_block: dict[str, Any] = {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": block.source.media_type,
                            "data": block.source.data,
                        },
                    }
                    if block.title:
                        doc_block["title"] = block.title
                    if block.citations:
                        doc_block["citations"] = block.citations
                    if block.context:
                        doc_block["context"] = block.context
                    result.append(doc_block)
                elif block.source.type == "url":
                    doc_block: dict[str, Any] = {
                        "type": "document",
                        "source": {
                            "type": "url",
                            "url": block.source.data,
                        },
                    }
                    if block.title:
                        doc_block["title"] = block.title
                    if block.citations:
                        doc_block["citations"] = block.citations
                    if block.context:
                        doc_block["context"] = block.context
                    result.append(doc_block)
                elif block.source.type == "file_id":
                    doc_block = {
                        "type": "document",
                        "source": {
                            "type": "file",
                            "file_id": block.source.data,
                        },
                    }
                    if block.title:
                        doc_block["title"] = block.title
                    if block.citations:
                        doc_block["citations"] = block.citations
                    if block.context:
                        doc_block["context"] = block.context
                    result.append(doc_block)
                elif block.source.type == "text":
                    doc_block: dict[str, Any] = {
                        "type": "document",
                        "source": {
                            "type": "text",
                            "media_type": block.source.media_type or "text/plain",
                            "data": block.source.data,
                        },
                    }
                    if block.title:
                        doc_block["title"] = block.title
                    if block.citations:
                        doc_block["citations"] = block.citations
                    if block.context:
                        doc_block["context"] = block.context
                    result.append(doc_block)
                elif block.source.type == "content":
                    doc_block = {
                        "type": "document",
                        "source": {
                            "type": "content",
                            "content": block.source.data,
                        },
                    }
                    if block.source.media_type:
                        doc_block["source"]["media_type"] = block.source.media_type
                    if block.title:
                        doc_block["title"] = block.title
                    if block.citations:
                        doc_block["citations"] = block.citations
                    if block.context:
                        doc_block["context"] = block.context
                    result.append(doc_block)
            elif isinstance(block, AudioBlock):
                if block.source.type == "base64":
                    result.append(
                        {
                            "type": "audio",
                            "source": {
                                "type": "base64",
                                "media_type": block.source.media_type,
                                "data": block.source.data,
                            },
                        }
                    )
                elif block.source.type == "url":
                    result.append(
                        {
                            "type": "audio",
                            "source": {
                                "type": "url",
                                "url": block.source.data,
                            },
                        }
                    )
                elif block.source.type == "file_id":
                    # Anthropic API does not support audio content blocks via file_id.
                    # Degrade to text placeholder to avoid crashing when routing from
                    # other protocols that support audio file_id (e.g. OpenAI).
                    result.append(
                        {
                            "type": "text",
                            "text": f"[Audio: file_id={block.source.data}]",
                        }
                    )
            elif isinstance(block, FileBlock):
                file_block: dict[str, Any] = {"type": "file"}
                if block.file_data:
                    file_block["file_data"] = block.file_data
                if block.file_id:
                    file_block["file_id"] = block.file_id
                if block.filename:
                    file_block["filename"] = block.filename
                result.append(file_block)
            elif isinstance(block, SearchResultBlock):
                search_block: dict[str, Any] = {"type": "search_result"}
                if block.source:
                    search_block["source"] = block.source
                if block.file_id:
                    search_block["file_id"] = block.file_id
                if block.title:
                    search_block["title"] = block.title
                if block.content:
                    search_block["content"] = self.format_content_blocks(
                        block.content, context=context
                    )
                if block.metadata:
                    search_block["metadata"] = block.metadata
                if block.cache_control:
                    search_block["cache_control"] = self._format_cache_control(block.cache_control)
                result.append(search_block)
            elif isinstance(block, ContainerUploadBlock):
                container_block: dict[str, Any] = {"type": "container_upload"}
                if block.file_id:
                    container_block["file_id"] = block.file_id
                if block.filename:
                    container_block["filename"] = block.filename
                if block.content:
                    container_block["content"] = block.content
                if block.media_type:
                    container_block["media_type"] = block.media_type
                result.append(container_block)
            elif isinstance(block, ToolReferenceBlock):
                ref_block: dict[str, Any] = {
                    "type": "tool_reference",
                    "tool_id": block.tool_id,
                }
                if block.tool_name:
                    ref_block["tool_name"] = block.tool_name
                if block.tool_type:
                    ref_block["tool_type"] = block.tool_type
                result.append(ref_block)
            elif isinstance(block, MidConversationSystemBlock):
                sys_block: dict[str, Any] = {
                    "type": "mid_conv_system",
                    "content": self.format_content_blocks(block.content, context=context),
                }
                if block.cache_control:
                    sys_block["cache_control"] = self._format_cache_control(block.cache_control)
                result.append(sys_block)
            elif isinstance(block, WebSearchToolResultBlock):
                result.append(
                    self._format_tool_result_block("web_search_tool_result", block, context=context)
                )
            elif isinstance(block, WebFetchToolResultBlock):
                result.append(
                    self._format_tool_result_block("web_fetch_tool_result", block, context=context)
                )
            elif isinstance(block, WebSearchResultContentBlock):
                ws_block: dict[str, Any] = {
                    "type": "web_search_result",
                    "url": block.url,
                    "title": block.title,
                    "encoded_content": block.encoded_content,
                }
                if block.page_age:
                    ws_block["page_age"] = block.page_age
                result.append(ws_block)
            elif isinstance(block, CodeExecutionToolResultBlock):
                result.append(
                    self._format_tool_result_block(
                        "code_execution_tool_result", block, context=context
                    )
                )
            elif isinstance(block, BashCodeExecutionToolResultBlock):
                result.append(
                    self._format_tool_result_block(
                        "bash_code_execution_tool_result", block, context=context
                    )
                )
            elif isinstance(block, TextEditorCodeExecutionToolResultBlock):
                result.append(
                    self._format_tool_result_block(
                        "text_editor_code_execution_tool_result", block, context=context
                    )
                )
            elif isinstance(block, ToolSearchToolResultBlock):
                result.append(
                    self._format_tool_result_block(
                        "tool_search_tool_result", block, context=context
                    )
                )
            elif isinstance(block, ToolResultBlock):
                content = block.content
                if isinstance(content, list):
                    content = self.format_content_blocks(content, context=context)
                tr_block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": block.tool_use_id,
                    "content": content,
                }
                if block.is_error:
                    tr_block["is_error"] = True
                if block.cache_control:
                    tr_block["cache_control"] = self._format_cache_control(block.cache_control)
                result.append(tr_block)
            elif isinstance(block, RawBlock):
                # Passthrough for provider-specific blocks
                if block.provider_type.startswith("anthropic:"):
                    result.append(block.data)
                else:
                    # Degrade foreign RawBlock to text placeholder
                    result.append({"type": "text", "text": f"[Raw block: {block.provider_type}]"})
            else:
                from llm_proxy.serialization._shared_degradation import (
                    degrade_block_to_text,
                    should_degrade_block,
                )

                policy = getattr(context, "unsupported_block_policy", "drop") if context else "drop"
                supported = (
                    getattr(context, "supported_content_blocks", frozenset())
                    if context
                    else frozenset()
                )
                if not should_degrade_block(
                    policy, block, self.provider_name, supported_blocks=supported
                ):
                    continue
                degraded = degrade_block_to_text(block)
                if degraded is not None:
                    result.append({"type": "text", "text": degraded})

        # Strip empty text blocks when there are other meaningful blocks.
        # This prevents round-trip noise from OpenAI-style null content
        # being converted into an empty Anthropic text block.
        if len(result) > 1:
            result = [
                r
                for r in result
                if not (isinstance(r, dict) and r.get("type") == "text" and r.get("text") == "")
            ]
        if not result:
            result = [{"type": "text", "text": ""}]
        return result

    def _format_citations(self, citations: list) -> list[dict[str, Any]]:
        formatted = []
        for c in citations:
            if isinstance(c, dict):
                formatted.append(c)
                continue
            cit: dict[str, Any] = {"type": getattr(c, "type", "char_location")}
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
                if hasattr(c, attr):
                    val = getattr(c, attr)
                    if val is not None:
                        cit[attr] = val
            formatted.append(cit)
        return formatted

    def _format_cache_control(self, cache_control) -> dict[str, Any]:
        """Format CacheControl to Anthropic wire format."""
        cc: dict[str, Any] = {"type": cache_control.type}
        if cache_control.ttl is not None:
            cc["ttl"] = cache_control.ttl
        return cc

    def _format_tool_result_block(
        self, block_type: str, block, context: BuildContext | None = None
    ) -> dict[str, Any]:
        if isinstance(block.content, list):
            if block.content and isinstance(block.content[0], dict):
                content = block.content
            else:
                content = self.format_content_blocks(block.content, context=context)
        else:
            content = block.content if isinstance(block.content, str) else str(block.content)

        result: dict[str, Any] = {
            "type": block_type,
            "tool_use_id": block.tool_use_id,
            "content": content,
        }
        if block.is_error:
            result["is_error"] = True
        if hasattr(block, "caller") and block.caller:
            caller_dict: dict[str, Any] = {"type": block.caller.type}
            if block.caller.tool_id:
                caller_dict["tool_id"] = block.caller.tool_id
            result["caller"] = caller_dict
        return result

    def _map_finish_reason(self, finish_reason: str) -> str:
        result = map_finish_reason(finish_reason, "openai", "anthropic")
        return result if result else finish_reason
