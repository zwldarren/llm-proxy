"""Ollama conversation conversion mixin."""

import re
from typing import Any

import orjson

from llm_proxy.core.exceptions import ValidationError
from llm_proxy.models import (
    ContentBlock,
    ConversationContext,
    CustomToolUseBlock,
    DocumentBlock,
    ImageBlock,
    TextBlock,
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


class OllamaConversationMixin:
    """Convert ConversationContext to Ollama native message format."""

    provider_name: str = "ollama"
    supported_content_blocks: frozenset[type[ContentBlock]] = frozenset(
        {
            TextBlock,
            ImageBlock,
            ToolUseBlock,
            CustomToolUseBlock,
            ToolResultBlock,
            ThinkingBlock,
            WebSearchToolResultBlock,
            WebFetchToolResultBlock,
            CodeExecutionToolResultBlock,
            BashCodeExecutionToolResultBlock,
            TextEditorCodeExecutionToolResultBlock,
            ToolSearchToolResultBlock,
        }
    )

    def _convert_conversation_to_ollama(
        self, conversation: ConversationContext, context: BuildContext | None = None
    ) -> list[dict[str, Any]]:
        from llm_proxy.models import (
            AudioBlock,
            FileBlock,
        )

        prepared: list[dict[str, Any]] = []
        tool_id_to_name: dict[str, str] = {}
        policy = context.unsupported_block_policy if context else "drop"
        namespace_map = context.namespace_map if context else None

        for msg in conversation.messages:
            if msg.role == "assistant":
                for block in msg.content:
                    if (
                        isinstance(block, (ToolUseBlock, CustomToolUseBlock))
                        and block.id
                        and block.name
                    ):
                        # Flatten history call names so they match the
                        # flattened tool definitions sent upstream (models
                        # echo the history name).
                        tool_id_to_name[block.id] = flatten_history_tool_name(
                            namespace_map, block.name
                        )

        for sys_msg in conversation.system_messages:
            text = sys_msg.text_content
            if text:
                # Ollama native chat only supports system/user/assistant/tool.
                role = "system" if sys_msg.role == "developer" else sys_msg.role
                prepared.append({"role": role, "content": text})

        for msg in conversation.messages:
            if msg.role == "system":
                sys_text = msg.text_content
                wrapped = (
                    f"<system-prompt>\n{sys_text}\n</system-prompt>"
                    if sys_text
                    else "<system-prompt></system-prompt>"
                )
                prepared.append({"role": "user", "content": wrapped})
                continue
            # Ollama native chat only supports system/user/assistant/tool.
            role = "system" if msg.role == "developer" else msg.role
            out: dict[str, Any] = {"role": role}
            texts: list[str] = []
            images: list[str] = []
            tool_calls_out: list[dict[str, Any]] = []
            tool_result_msgs: list[dict[str, Any]] = []

            for block in msg.content:
                # Try lossless conversion before applying policy.
                converted = try_convert_block(block)
                if converted is not None:
                    block = converted

                if isinstance(block, TextBlock):
                    if block.text:
                        texts.append(block.text)
                elif isinstance(block, ImageBlock):
                    if block.source.type == "base64" and block.source.data:
                        images.append(block.source.data)
                    elif block.source.type == "url" and block.source.data:
                        base64_data = self._extract_base64_from_image_url(block.source.data)
                        if base64_data:
                            images.append(base64_data)
                        else:
                            texts.append(f"[Image URL: {block.source.data}]")
                            logger.warning(
                                "Failed to extract base64 from image URL for Ollama",
                                extra={"url": block.source.data[:200]},
                            )
                elif isinstance(block, DocumentBlock):
                    # Ollama chat has no document part. Extract plain-text content
                    # when available (text/content source, or base64 text media);
                    # otherwise degrade to a placeholder so the block is not
                    # silently lost.
                    doc_text = self._extract_document_text(block)
                    if doc_text:
                        texts.append(doc_text)
                    else:
                        degraded = degrade_block_to_text(block)
                        if degraded:
                            texts.append(degraded)
                elif isinstance(block, (AudioBlock, FileBlock)):
                    pass
                elif isinstance(block, ThinkingBlock):
                    if block.thinking:
                        out["thinking"] = block.thinking
                elif isinstance(block, (CustomToolUseBlock, ToolUseBlock)):
                    if isinstance(block, CustomToolUseBlock):
                        arguments = {"content": block.input}
                    else:
                        arguments = self._try_parse_tool_arguments(block.input)
                    tool_calls_out.append(
                        {
                            "function": {
                                "name": flatten_history_tool_name(namespace_map, block.name),
                                "arguments": arguments,
                            },
                        }
                    )
                elif isinstance(
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
                    tool_msg = self._tool_result_to_ollama_tool_message(block, tool_id_to_name)
                    tool_result_msgs.append(tool_msg)
                else:
                    # Handle unsupported block types via shared degradation logic.
                    # If a block is declared as supported but fell through unhandled,
                    # degrade it anyway as a safe fallback rather than silently dropping.
                    is_supported = (
                        self.supported_content_blocks
                        and type(block) in self.supported_content_blocks
                    )
                    if not is_supported and not should_degrade_block(
                        policy,
                        block,
                        self.provider_name,
                        supported_blocks=self.supported_content_blocks,
                    ):
                        continue
                    degraded = degrade_block_to_text(block)
                    if degraded:
                        texts.append(degraded)

            if (not texts and not images and not tool_calls_out) and "thinking" not in out:
                prepared.extend(tool_result_msgs)
                continue

            if texts and images:
                out["content"] = " ".join(texts)
                out["images"] = images
            elif texts:
                out["content"] = " ".join(texts)
            elif images:
                out["content"] = ""
                out["images"] = images
            else:
                out["content"] = ""

            if tool_calls_out:
                out["tool_calls"] = tool_calls_out

            prepared.append(out)
            prepared.extend(tool_result_msgs)

        return prepared

    def _tool_result_to_ollama_tool_message(
        self, block: Any, tool_id_to_name: dict[str, str]
    ) -> dict[str, Any]:
        """Convert a tool result block to Ollama tool message format."""
        content_str = ""
        content_raw = getattr(block, "content", "")
        if isinstance(content_raw, str):
            content_str = content_raw
        elif isinstance(content_raw, list):
            content_str = " ".join(
                b.text for b in content_raw if isinstance(b, TextBlock) and b.text
            )
        else:
            content_str = str(content_raw) if content_raw is not None else ""

        tool_use_id = getattr(block, "tool_use_id", "")
        tool_name = getattr(block, "name", None) or tool_id_to_name.get(tool_use_id, "")
        tool_msg: dict[str, Any] = {
            "role": "tool",
            "content": content_str,
        }
        if tool_name:
            tool_msg["tool_name"] = tool_name
        return tool_msg

    def _extract_document_text(self, block: DocumentBlock) -> str | None:
        """Extract plain-text content from a DocumentBlock for Ollama.

        Ollama chat has no document part, so documents must be surfaced as
        text. Plain-text sources (``text``/``content``) and base64-encoded
        text media (``text/*``) are decoded into text; binary documents (PDFs,
        etc.) return None so the caller can degrade them to a placeholder
        instead of silently dropping them.
        """
        import base64

        source = block.source
        source_type = source.type
        data = source.data

        if source_type in ("text", "content"):
            if isinstance(data, str):
                return data if data else None
            if isinstance(data, list):
                parts: list[str] = []
                for chunk in data:
                    if isinstance(chunk, dict):
                        if chunk.get("type") == "text":
                            text = chunk.get("text", "")
                            if text:
                                parts.append(text)
                        else:
                            # Non-text chunk inside document content: surface
                            # a placeholder so it is not silently lost.
                            parts.append(f"[{chunk.get('type', 'content')}]")
                    elif isinstance(chunk, str) and chunk:
                        parts.append(chunk)
                return " ".join(parts) if parts else None
            return str(data) if data else None

        if (
            source_type == "base64"
            and isinstance(source.media_type, str)
            and source.media_type.startswith("text/")
        ):
            if not isinstance(data, str) or not data:
                return None
            try:
                decoded = base64.b64decode(data).decode("utf-8", errors="replace")
            except Exception:
                logger.debug("Failed to decode base64 text document", exc_info=True)
                return None
            return decoded or None

        return None

    def _extract_base64_from_image_url(self, url: str) -> str | None:
        if not url:
            return None

        if url.startswith("data:"):
            match = re.match(r"data:([^;]+);base64,(.+)", url)
            if not match:
                raise ValidationError("Malformed data URI for image")
            import base64

            try:
                base64.b64decode(match.group(2), validate=True)
            except Exception as exc:
                raise ValidationError("Invalid base64 image data in data URI") from exc
            return match.group(2)

        # HTTP(S) and file URLs are not inline base64 data.
        if url.startswith(("http://", "https://", "file://")):
            return None

        import base64

        try:
            base64.b64decode(url, validate=True)
            return url
        except Exception as exc:
            raise ValidationError("Invalid base64 image data") from exc

    def _try_parse_tool_arguments(self, raw_args: Any) -> dict[str, Any]:
        if isinstance(raw_args, dict):
            return raw_args

        if raw_args is None:
            return {}

        if isinstance(raw_args, (bytes, bytearray)):
            raw_args = raw_args.decode(errors="ignore")

        if isinstance(raw_args, str):
            if not raw_args.strip():
                return {}

            try:
                parsed = orjson.loads(raw_args)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

            import json

            try:
                decoder = json.JSONDecoder()
                obj, _ = decoder.raw_decode(raw_args)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                logger.debug(
                    "Failed to parse tool arguments",
                    exc_info=True,
                )

            return {}

        return {}
