"""Shared content parsing and formatting mixin for Anthropic format.

Used by both AnthropicProtocolSerializer and AnthropicProviderSerializer
to avoid duplicating content block conversion logic.
"""

from typing import TYPE_CHECKING, Any

import orjson

from llm_proxy.models import (
    AudioBlock,
    AudioSource,
    ContentBlock,
    CustomToolUseBlock,
    DocumentBlock,
    DocumentSource,
    FileBlock,
    ImageBlock,
    ImageSource,
    RawBlock,
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
    CacheControl,
    Caller,
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
from llm_proxy.serialization._shared_degradation import (
    degrade_block_to_text,
    should_degrade_block,
)
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
        result: list[dict[str, Any]] = []
        for block in blocks:
            formatted = self._format_content_block(block, context=context)
            if formatted is not None:
                result.append(formatted)
        return self._finalize_content_blocks(result)

    def _format_content_block(
        self, block: ContentBlock, context: BuildContext | None
    ) -> dict[str, Any] | None:
        """Format a single ContentBlock to Anthropic wire format, or None to skip."""
        if isinstance(block, TextBlock):
            return self._format_text_block(block)
        if isinstance(block, ToolUseBlock):
            return self._format_tool_use_block(block, context=context)
        if isinstance(block, ServerToolUseBlock):
            return self._format_server_tool_use_block(block, context=context)
        if isinstance(block, CustomToolUseBlock):
            return self._format_custom_tool_use_block(block, context=context)
        if isinstance(block, ThinkingBlock):
            return self._format_thinking_block(block)
        if isinstance(block, RedactedThinkingBlock):
            return self._format_redacted_thinking_block(block)
        if isinstance(block, RefusalBlock):
            return self._format_refusal_block(block)
        if isinstance(block, ImageBlock):
            return self._format_image_block(block)
        if isinstance(block, DocumentBlock):
            return self._format_document_block(block)
        if isinstance(block, AudioBlock):
            return self._format_audio_block(block)
        if isinstance(block, FileBlock):
            return self._format_file_block(block)
        if isinstance(block, SearchResultBlock):
            return self._format_search_result_block(block, context=context)
        if isinstance(block, ContainerUploadBlock):
            return self._format_container_upload_block(block)
        if isinstance(block, ToolReferenceBlock):
            return self._format_tool_reference_block(block)
        if isinstance(block, MidConversationSystemBlock):
            return self._format_mid_conv_system_block(block, context=context)
        if isinstance(block, WebSearchToolResultBlock):
            return self._format_tool_result_block("web_search_tool_result", block, context=context)
        if isinstance(block, WebFetchToolResultBlock):
            return self._format_tool_result_block("web_fetch_tool_result", block, context=context)
        if isinstance(block, WebSearchResultContentBlock):
            return self._format_web_search_result_block(block)
        if isinstance(block, CodeExecutionToolResultBlock):
            return self._format_tool_result_block(
                "code_execution_tool_result", block, context=context
            )
        if isinstance(block, BashCodeExecutionToolResultBlock):
            return self._format_tool_result_block(
                "bash_code_execution_tool_result", block, context=context
            )
        if isinstance(block, TextEditorCodeExecutionToolResultBlock):
            return self._format_tool_result_block(
                "text_editor_code_execution_tool_result", block, context=context
            )
        if isinstance(block, ToolSearchToolResultBlock):
            return self._format_tool_result_block("tool_search_tool_result", block, context=context)
        if isinstance(block, ToolResultBlock):
            return self._format_tool_result(block, context=context)
        if isinstance(block, RawBlock):
            return self._format_raw_block(block)
        return self._format_unsupported_block(block, context=context)

    def _format_text_block(self, block: TextBlock) -> dict[str, Any]:
        """Format a TextBlock to Anthropic wire format."""
        text_block: dict[str, Any] = {"type": "text", "text": block.text}
        if block.citations:
            text_block["citations"] = self._format_citations(block.citations)
        if block.cache_control:
            text_block["cache_control"] = self._format_cache_control(block.cache_control)
        return text_block

    def _format_tool_use_block(
        self, block: ToolUseBlock, context: BuildContext | None
    ) -> dict[str, Any]:
        """Format a ToolUseBlock to Anthropic wire format."""
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
        return tool_block

    def _format_server_tool_use_block(
        self, block: ServerToolUseBlock, context: BuildContext | None
    ) -> dict[str, Any]:
        """Format a ServerToolUseBlock to Anthropic wire format."""
        server_tool_block: dict[str, Any] = {
            "type": "server_tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }
        if block.cache_control:
            server_tool_block["cache_control"] = self._format_cache_control(block.cache_control)
        return server_tool_block

    def _format_custom_tool_use_block(
        self, block: CustomToolUseBlock, context: BuildContext | None
    ) -> dict[str, Any]:
        """Format a CustomToolUseBlock to Anthropic wire format."""
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
        return custom_tool_block

    def _format_thinking_block(self, block: ThinkingBlock) -> dict[str, Any] | None:
        """Format a ThinkingBlock, or None to skip empty thinking."""
        if not block.thinking:
            return None
        thinking_block: dict[str, Any] = {
            "type": "thinking",
            "thinking": block.thinking,
        }
        if block.signature:
            thinking_block["signature"] = block.signature
        return thinking_block

    def _format_redacted_thinking_block(self, block: RedactedThinkingBlock) -> dict[str, Any]:
        """Format a RedactedThinkingBlock to Anthropic wire format."""
        return {
            "type": "redacted_thinking",
            "data": block.data,
        }

    def _format_refusal_block(self, block: RefusalBlock) -> dict[str, Any]:
        """Format a RefusalBlock to Anthropic wire format."""
        return {
            "type": "refusal",
            "refusal": block.refusal,
        }

    @staticmethod
    def _format_file_source(
        source: ImageSource | AudioSource | DocumentSource,
    ) -> dict[str, Any] | None:
        """Map a base64/url/file_id source to Anthropic wire format.

        Returns None for source types callers handle specially (e.g. document
        ``text``/``content`` sources, or audio ``file_id`` degradation).
        """
        if source.type == "base64":
            return {
                "type": "base64",
                "media_type": source.media_type,
                "data": source.data,
            }
        if source.type == "url":
            return {"type": "url", "url": source.data}
        if source.type == "file_id":
            return {"type": "file", "file_id": source.data}
        return None

    def _format_image_block(self, block: ImageBlock) -> dict[str, Any] | None:
        """Format an ImageBlock, or None for unsupported source types."""
        source = self._format_file_source(block.source)
        if source is None:
            return None
        return {"type": "image", "source": source}

    def _format_document_block(self, block: DocumentBlock) -> dict[str, Any] | None:
        """Format a DocumentBlock, or None for unsupported source types."""
        source = block.source
        source_dict = self._format_file_source(source)
        if source_dict is not None:
            doc_block: dict[str, Any] = {"type": "document", "source": source_dict}
        elif source.type == "text":
            doc_block = {
                "type": "document",
                "source": {
                    "type": "text",
                    "media_type": source.media_type or "text/plain",
                    "data": source.data,
                },
            }
        elif source.type == "content":
            doc_block = {
                "type": "document",
                "source": {
                    "type": "content",
                    "content": source.data,
                },
            }
            if source.media_type:
                doc_block["source"]["media_type"] = source.media_type
        else:
            return None
        if block.title:
            doc_block["title"] = block.title
        if block.citations:
            doc_block["citations"] = block.citations
        if block.context:
            doc_block["context"] = block.context
        return doc_block

    def _format_audio_block(self, block: AudioBlock) -> dict[str, Any] | None:
        """Format an AudioBlock, or None for unsupported source types."""
        if block.source.type == "file_id":
            # Anthropic API does not support audio content blocks via file_id.
            # Degrade to text placeholder to avoid crashing when routing from
            # other protocols that support audio file_id (e.g. OpenAI).
            return {
                "type": "text",
                "text": f"[Audio: file_id={block.source.data}]",
            }
        source = self._format_file_source(block.source)
        if source is None:
            return None
        return {"type": "audio", "source": source}

    def _format_file_block(self, block: FileBlock) -> dict[str, Any]:
        """Format a FileBlock to Anthropic wire format."""
        file_block: dict[str, Any] = {"type": "file"}
        if block.file_data:
            file_block["file_data"] = block.file_data
        if block.file_id:
            file_block["file_id"] = block.file_id
        if block.filename:
            file_block["filename"] = block.filename
        return file_block

    def _format_search_result_block(
        self, block: SearchResultBlock, context: BuildContext | None
    ) -> dict[str, Any]:
        """Format a SearchResultBlock to Anthropic wire format."""
        search_block: dict[str, Any] = {"type": "search_result"}
        if block.source:
            search_block["source"] = block.source
        if block.file_id:
            search_block["file_id"] = block.file_id
        if block.title:
            search_block["title"] = block.title
        if block.content:
            search_block["content"] = self.format_content_blocks(block.content, context=context)
        if block.metadata:
            search_block["metadata"] = block.metadata
        if block.cache_control:
            search_block["cache_control"] = self._format_cache_control(block.cache_control)
        return search_block

    def _format_container_upload_block(self, block: ContainerUploadBlock) -> dict[str, Any]:
        """Format a ContainerUploadBlock to Anthropic wire format."""
        container_block: dict[str, Any] = {"type": "container_upload"}
        if block.file_id:
            container_block["file_id"] = block.file_id
        if block.filename:
            container_block["filename"] = block.filename
        if block.content:
            container_block["content"] = block.content
        if block.media_type:
            container_block["media_type"] = block.media_type
        return container_block

    def _format_tool_reference_block(self, block: ToolReferenceBlock) -> dict[str, Any]:
        """Format a ToolReferenceBlock to Anthropic wire format."""
        ref_block: dict[str, Any] = {
            "type": "tool_reference",
            "tool_id": block.tool_id,
        }
        if block.tool_name:
            ref_block["tool_name"] = block.tool_name
        if block.tool_type:
            ref_block["tool_type"] = block.tool_type
        return ref_block

    def _format_mid_conv_system_block(
        self, block: MidConversationSystemBlock, context: BuildContext | None
    ) -> dict[str, Any]:
        """Format a MidConversationSystemBlock to Anthropic wire format."""
        sys_block: dict[str, Any] = {
            "type": "mid_conv_system",
            "content": self.format_content_blocks(block.content, context=context),
        }
        if block.cache_control:
            sys_block["cache_control"] = self._format_cache_control(block.cache_control)
        return sys_block

    def _format_web_search_result_block(self, block: WebSearchResultContentBlock) -> dict[str, Any]:
        """Format a WebSearchResultContentBlock to Anthropic wire format."""
        ws_block: dict[str, Any] = {
            "type": "web_search_result",
            "url": block.url,
            "title": block.title,
            "encoded_content": block.encoded_content,
        }
        if block.page_age:
            ws_block["page_age"] = block.page_age
        return ws_block

    def _format_tool_result(
        self, block: ToolResultBlock, context: BuildContext | None
    ) -> dict[str, Any]:
        """Format a plain ToolResultBlock to Anthropic wire format."""
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
        return tr_block

    def _format_raw_block(self, block: RawBlock) -> dict[str, Any]:
        """Format a RawBlock: passthrough for Anthropic blocks, text placeholder otherwise."""
        if block.provider_type.startswith("anthropic:"):
            return block.data
        # Degrade foreign RawBlock to text placeholder
        return {"type": "text", "text": f"[Raw block: {block.provider_type}]"}

    def _format_unsupported_block(
        self, block: ContentBlock, context: BuildContext | None
    ) -> dict[str, Any] | None:
        """Apply the unsupported-block policy, or None to drop the block."""
        policy = getattr(context, "unsupported_block_policy", "drop") if context else "drop"
        supported = (
            getattr(context, "supported_content_blocks", frozenset()) if context else frozenset()
        )
        if not should_degrade_block(policy, block, self.provider_name, supported_blocks=supported):
            return None
        degraded = degrade_block_to_text(block)
        if degraded is not None:
            return {"type": "text", "text": degraded}
        return None

    def _finalize_content_blocks(self, result: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Strip empty text blocks and guarantee a non-empty result."""
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
