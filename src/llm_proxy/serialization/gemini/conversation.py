"""Gemini conversation conversion mixin."""

import base64
import re
from typing import Any

import orjson

from llm_proxy.core.exceptions import ValidationError
from llm_proxy.models import (
    AudioBlock,
    ContentBlock,
    ConversationContext,
    CustomToolUseBlock,
    DocumentBlock,
    ImageBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    VideoBlock,
)
from llm_proxy.serialization._shared_conversion import try_convert_block
from llm_proxy.serialization._shared_degradation import (
    degrade_block_to_text,
    should_degrade_block,
)
from llm_proxy.serialization.context import BuildContext
from llm_proxy.serialization.gemini.request_builder import GeminiRequestBuilderMixin
from llm_proxy.serialization.responses_toolkit.namespace import (
    flatten_history_tool_name,
)

# Google's recommended dummy thought signature for conversation history from
# models that do not produce thought signatures (base64 of
# "skip_thought_signature_validator"). It passes Gemini's validation while
# real signatures are preserved verbatim.
_DUMMY_THOUGHT_SIGNATURE = base64.b64encode(b"skip_thought_signature_validator").decode()


class GeminiConversationMixin:
    """Convert ConversationContext to Gemini format directly."""

    provider_name: str = "gemini"
    supported_content_blocks: frozenset[type[ContentBlock]] = frozenset(
        {
            TextBlock,
            ImageBlock,
            AudioBlock,
            VideoBlock,
            DocumentBlock,
            ToolUseBlock,
            CustomToolUseBlock,
            ToolResultBlock,
            ThinkingBlock,
        }
    )

    @staticmethod
    def _make_file_part(
        file_data: str | None,
        file_id: str | None,
    ) -> dict[str, Any] | None:
        """Build a Gemini part from a FileBlock.

        ``file_data`` may be either a ``data:<mime>;base64,<payload>`` URI or a
        plain URL. The data URI is decoded into a Gemini ``inline_data`` part
        (with the data: prefix stripped and the real mime_type extracted); a
        plain URL is emitted as a ``file_data`` part. ``file_id`` references a
        previously uploaded Gemini File API resource and is emitted as
        ``file_data`` as well.
        """
        if file_data:
            decoded = GeminiConversationMixin._decode_data_uri(file_data)
            if decoded is not None:
                mime_type, b64 = decoded
                return {"inline_data": {"mime_type": mime_type, "data": b64}}
            return {"file_data": {"file_uri": file_data}}
        if file_id:
            return {"file_data": {"file_uri": file_id}}
        return None

    @staticmethod
    def _make_media_part(
        source_type: str,
        data: str,
        media_type: str | None,
        default_media_type: str,
    ) -> dict[str, Any] | None:
        """Build a Gemini inline_data or file_data part from a media block source."""
        if not data:
            return None
        mime = media_type or default_media_type
        if source_type == "base64":
            return {"inline_data": {"mime_type": mime, "data": data}}
        if source_type in ("file_id", "url"):
            return {"file_data": {"mime_type": mime, "file_uri": data}}
        return None

    def _convert_conversation_to_gemini(
        self, conversation: ConversationContext, context: BuildContext | None = None
    ) -> tuple[list[dict[str, Any]], str | None]:
        contents: list[dict[str, Any]] = []
        system_parts: list[str] = []

        # Map tool call ids to their (flattened) tool names so functionResponse
        # parts can reference the matching functionCall by name. Gemini requires
        # functionResponse.name to equal the functionCall.name; the call id is
        # not a valid name. ToolResultBlock.name is None on the openresponses
        # path (output items carry only call_id), so the id -> name map built
        # from assistant messages is the reliable source.
        call_id_to_name: dict[str, str] = {}
        for msg in conversation.messages:
            if msg.role != "assistant":
                continue
            for block in msg.content:
                if isinstance(block, (ToolUseBlock, CustomToolUseBlock)) and block.id:
                    call_id_to_name[block.id] = flatten_history_tool_name(
                        context.namespace_map if context else None, block.name
                    )

        for sys_msg in conversation.system_messages:
            if sys_msg.content:
                system_parts.append(sys_msg.text_content)

        for msg in conversation.messages:
            if msg.role == "system":
                sys_text = msg.text_content
                wrapped = (
                    f"<system-prompt>\n{sys_text}\n</system-prompt>"
                    if sys_text
                    else "<system-prompt></system-prompt>"
                )
                contents.append({"role": "user", "parts": [{"text": wrapped}]})
                continue
            gemini_role = "model" if msg.role == "assistant" else "user"
            parts = self._convert_content_blocks_to_gemini_parts(
                msg.content, context, call_id_to_name
            )

            # Drop empty placeholder text parts when there is other content.
            meaningful = [p for p in parts if not (isinstance(p, dict) and p.get("text") == "")]
            if meaningful:
                parts = meaningful
            elif not meaningful:
                continue

            if parts:
                contents.append({"role": gemini_role, "parts": parts})

        # Gemini 2.5+/3 require thoughtSignature on every functionCall part in
        # the request; history produced by other providers (or by Gemini before
        # the proxy cached the signature) lacks them. Inject Google's
        # recommended dummy signature so the request passes validation instead
        # of failing with "Function call is missing a thought_signature".
        # Older models (2.0/1.5) reject inconsistent signatures, so for those
        # the previous behavior is kept: strip all when any is missing.
        if self._model_supports_thought_signatures(context.model if context else None):
            contents = self._inject_dummy_thought_signatures(contents)
        elif not self._all_tool_uses_have_thought_signature(conversation):
            contents = self._strip_thought_signatures(contents)

        system = "\n".join(system_parts) if system_parts else None
        return contents, system

    @staticmethod
    def _model_supports_thought_signatures(model: str | None) -> bool:
        """Whether the target Gemini model supports thought signatures.

        Thought signatures were introduced with Gemini 2.5 and are strictly
        required for function calling on Gemini 3. Older models (2.0/1.5) do
        not support the field.
        """
        if not model:
            return False
        return "gemini-2.5" in model or "gemini-3" in model

    @staticmethod
    def _inject_dummy_thought_signatures(
        contents: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Add Google's recommended dummy thought signature to parts missing one.

        ``base64("skip_thought_signature_validator")`` is the approach Google
        recommends for conversation history from models that do not produce
        thought signatures; it passes Gemini's validation while real
        signatures (from Gemini responses, re-attached by the adapter cache)
        are preserved verbatim.
        """
        for msg in contents:
            for part in msg.get("parts", []):
                if (
                    isinstance(part, dict)
                    and "functionCall" in part
                    and not part.get("thoughtSignature")
                ):
                    part["thoughtSignature"] = _DUMMY_THOUGHT_SIGNATURE
        return contents

    @staticmethod
    def _all_tool_uses_have_thought_signature(
        conversation: ConversationContext,
    ) -> bool:
        """Check if all ToolUseBlocks in the conversation have thought_signature."""
        has_any_tool_use = False
        for msg in conversation.messages:
            for block in msg.content:
                if isinstance(block, ToolUseBlock):
                    has_any_tool_use = True
                    if not block.extra.get("thought_signature"):
                        return False
        # If there are no tool uses, we can include thoughtSignatures (trivially true).
        return has_any_tool_use

    @staticmethod
    def _strip_thought_signatures(contents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove thoughtSignature from all functionCall parts in contents."""
        for msg in contents:
            parts = msg.get("parts", [])
            for part in parts:
                if isinstance(part, dict) and "functionCall" in part:
                    part.pop("thoughtSignature", None)
        return contents

    def _convert_content_blocks_to_gemini_parts(
        self,
        blocks: list[ContentBlock],
        context: BuildContext | None = None,
        call_id_to_name: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        from llm_proxy.models import (
            AudioBlock,
            DocumentBlock,
            FileBlock,
            ImageBlock,
            VideoBlock,
        )

        parts: list[dict[str, Any]] = []
        policy = context.unsupported_block_policy if context else "drop"

        for block in blocks:
            # Try lossless conversion before applying policy.
            converted = try_convert_block(block)
            if converted is not None:
                block = converted

            if isinstance(block, TextBlock):
                if block.text:
                    model = context.model if context else None
                    if model and GeminiRequestBuilderMixin._is_gemini_image_model(model):
                        parts.extend(self._parse_gemini_markdown_images(block.text))
                    else:
                        parts.append({"text": block.text})
            elif isinstance(block, ImageBlock):
                part = self._make_media_part(
                    block.source.type,
                    block.source.data,
                    block.source.media_type,
                    "image/png",
                )
                if part:
                    parts.append(part)
            elif isinstance(block, AudioBlock):
                part = self._make_media_part(
                    block.source.type,
                    block.source.data,
                    block.source.media_type,
                    "audio/mp3",
                )
                if part:
                    parts.append(part)
            elif isinstance(block, VideoBlock):
                part = self._make_media_part(
                    block.source.type,
                    block.source.data,
                    block.source.media_type,
                    "video/mp4",
                )
                if part:
                    parts.append(part)
            elif isinstance(block, DocumentBlock):
                if block.source.type in ("text", "content") and block.source.data:
                    doc_data = block.source.data
                    doc_text = doc_data if isinstance(doc_data, str) else str(doc_data)
                    if doc_text:
                        parts.append({"text": doc_text})
                else:
                    # Treat document sources like other media: base64 -> inline_data,
                    # url/file_id -> file_data. Plain text documents stay in the branch
                    # above so they are emitted as regular text parts.
                    part = self._make_media_part(
                        block.source.type,
                        block.source.data,
                        block.source.media_type,
                        "application/pdf",
                    )
                    if part:
                        parts.append(part)
            elif isinstance(block, FileBlock):
                part = self._make_file_part(block.file_data, block.file_id)
                if part:
                    parts.append(part)
            elif isinstance(block, CustomToolUseBlock):
                # Custom (freeform) tools are bridged to function declarations
                # with a {"content": string} schema (see request_builder), so
                # the raw input is re-wrapped to keep history consistent.
                parts.append(
                    {
                        "functionCall": {
                            "name": flatten_history_tool_name(
                                context.namespace_map if context else None, block.name
                            ),
                            "args": {"content": block.input},
                        }
                    }
                )
            elif isinstance(block, ToolUseBlock):
                func_call_part: dict[str, Any] = {
                    "functionCall": {
                        "name": flatten_history_tool_name(
                            context.namespace_map if context else None, block.name
                        ),
                        "args": block.input,
                    }
                }
                # thoughtSignature lives at the PART level (sibling of
                # functionCall), NOT inside the functionCall object. Gemini
                # 3.x rejects the field inside functionCall with "Unknown name
                # thoughtSignature at ...function_call: Cannot find field".
                if block.extra.get("thought_signature"):
                    func_call_part["thoughtSignature"] = block.extra["thought_signature"]
                parts.append(func_call_part)
            elif isinstance(block, ThinkingBlock):
                if not block.thinking:
                    continue
                thought_part: dict[str, Any] = {"thought": True, "text": block.thinking}
                if block.signature is not None:
                    thought_part["signature"] = block.signature
                parts.append(thought_part)
            elif isinstance(block, ToolResultBlock):
                raw_name = (
                    getattr(block, "name", None)
                    or (call_id_to_name or {}).get(block.tool_use_id)
                    or block.tool_use_id
                )
                func_name = flatten_history_tool_name(
                    context.namespace_map if context else None, raw_name
                )
                response_content: dict[str, Any] | str
                if isinstance(block.content, str):
                    try:
                        parsed = orjson.loads(block.content)
                        if isinstance(parsed, dict):
                            response_content = parsed
                        else:
                            response_content = {"content": block.content}
                    except orjson.JSONDecodeError, TypeError:
                        response_content = {"content": block.content}
                elif isinstance(block.content, list):
                    text_parts: list[str] = []
                    for sub in block.content:
                        if isinstance(sub, TextBlock) and sub.text:
                            text_parts.append(sub.text)
                    response_content = (
                        {"content": "\n".join(text_parts)} if text_parts else {"content": ""}
                    )
                else:
                    response_content = {"content": str(block.content)}
                parts.append(
                    {
                        "functionResponse": {
                            "name": func_name,
                            "response": response_content,
                        }
                    }
                )
            else:
                # Handle unsupported block types via shared degradation logic.
                # If a block is declared as supported but fell through unhandled,
                # degrade it anyway as a safe fallback rather than silently dropping.
                is_supported = (
                    self.supported_content_blocks and type(block) in self.supported_content_blocks
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
                    parts.append({"text": degraded})

        return parts if parts else [{"text": ""}]

    # ── nano-banana markdown image parsing ────────────────────────────

    # Regex for markdown image syntax: ![alt](data:image/<type>;base64,<data>)
    # Only data: URIs are matched — HTTP URLs are never treated as images.
    _MARKDOWN_IMAGE_RE: re.Pattern[str] = re.compile(r"!\[[^\]]*\]\((data:[^)]+)\)")

    @staticmethod
    def _decode_data_uri(data_uri: str) -> tuple[str, str] | None:
        """Decode a data: URI into (mime_type, base64_data).

        Returns None when the URI does not contain base64-encoded data.
        Missing padding (``=``) is handled transparently — the returned
        *b64* string is the original, unmodified value from the URI.
        """
        if not data_uri.startswith("data:"):
            return None
        # Strip the "data:" prefix
        rest = data_uri[5:]
        if ";base64," not in rest:
            return None
        mime_type, b64 = rest.split(";base64,", 1)
        if not mime_type or not b64:
            raise ValidationError("Malformed data URI: missing media type or base64 payload")
        # Add missing padding just for validation, then strip it back
        missing = len(b64) % 4
        padded = b64 + "=" * (4 - missing) if missing else b64
        try:
            base64.b64decode(padded, validate=True)
        except Exception as exc:
            raise ValidationError(
                f"Invalid base64 image data in data URI (media type {mime_type})"
            ) from exc
        return mime_type, b64

    @classmethod
    def _parse_gemini_markdown_images(cls, text: str) -> list[dict[str, Any]]:
        """Parse ``![alt](data:image/…;base64,…)`` from *text*.

        Each data URI is emitted as a Gemini ``inline_data`` part.
        Surrounding text stays as ``text`` parts.  If no markdown images
        are found the original text is returned as a single text part.

        This mirrors the behaviour of new-api's ``CovertOpenAI2Gemini``
        and exists purely to support Gemini image-generation models
        (nano banana) through the ``/v1/chat/completions`` endpoint.
        """
        parts: list[dict[str, Any]] = []
        last_end = 0
        found = False

        for m in cls._MARKDOWN_IMAGE_RE.finditer(text):
            found = True
            start, end = m.start(), m.end()
            data_uri = m.group(1)

            # Emit any text before this match
            if start > last_end:
                parts.append({"text": text[last_end:start]})

            # Decode the data URI into a Gemini inline_data part.
            # Invalid base64 in free-text markdown degrades gracefully to
            # literal text instead of failing the whole request (unlike
            # explicit image inputs, where malformed data is a 400).
            try:
                decoded = cls._decode_data_uri(data_uri)
            except ValidationError:
                decoded = None
            if decoded is not None:
                mime_type, b64 = decoded
                parts.append(
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": b64,
                        }
                    }
                )
            else:
                # Could not decode — keep the original markdown as text
                parts.append({"text": m.group(0)})

            last_end = end

        if not found:
            return [{"text": text}]

        # Emit any trailing text
        if last_end < len(text):
            parts.append({"text": text[last_end:]})

        return parts
