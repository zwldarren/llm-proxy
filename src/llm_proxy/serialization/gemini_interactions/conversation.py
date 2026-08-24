"""Conversation conversion for the Gemini Interactions API.

The Interactions API replaces the legacy ``contents[]`` array with a single
``input`` field accepting a string, a list of Content items, or a list of
typed Steps. The proxy runs stateless (``store=false``), so multi-turn
history is replayed as a Step array: user turns → ``user_input`` steps,
assistant turns → ``model_output`` steps, tool calls → ``function_call``
steps, tool results → ``function_result`` steps.
"""

import logging
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
    FileBlock,
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
from llm_proxy.serialization.gemini.conversation import GeminiConversationMixin
from llm_proxy.serialization.gemini_interactions.request_builder import (
    GeminiInteractionsRequestBuilderMixin,
)
from llm_proxy.serialization.responses_toolkit.namespace import (
    flatten_history_tool_name,
)

logger = logging.getLogger(__name__)


class GeminiInteractionsConversationMixin:
    """Convert ConversationContext to Interactions ``input`` Steps."""

    provider_name: str = "gemini-interactions"
    supported_content_blocks: frozenset[type[ContentBlock]] = frozenset(
        {
            TextBlock,
            ImageBlock,
            AudioBlock,
            VideoBlock,
            DocumentBlock,
            FileBlock,
            ToolUseBlock,
            CustomToolUseBlock,
            ToolResultBlock,
        }
    )

    # ── Content item builders ─────────────────────────────────────────

    # Dialect-independent helpers shared with the legacy generateContent
    # dialect (see serialization/gemini/conversation.py).
    _decode_data_uri = staticmethod(GeminiConversationMixin._decode_data_uri)

    # Regex for markdown image syntax: ![alt](data:image/<type>;base64,<data>)
    # Only data: URIs are matched — HTTP URLs are never treated as images.
    _MARKDOWN_IMAGE_RE: re.Pattern[str] = GeminiConversationMixin._MARKDOWN_IMAGE_RE

    def _parse_gemini_markdown_images(self, text: str) -> list[dict[str, Any]]:
        """Parse ``![alt](data:image/…;base64,…)`` into image content items.

        Mirrors the legacy nano-banana markdown handling so image-generation
        models keep working through the chat endpoint under the Interactions
        variant.
        """
        items: list[dict[str, Any]] = []
        last_end = 0

        for m in self._MARKDOWN_IMAGE_RE.finditer(text):
            start, end = m.start(), m.end()
            data_uri = m.group(1)
            if start > last_end:
                items.append({"type": "text", "text": text[last_end:start]})
            try:
                decoded = self._decode_data_uri(data_uri)
            except ValidationError:
                decoded = None
            if decoded is not None:
                mime_type, b64 = decoded
                items.append({"type": "image", "mime_type": mime_type, "data": b64})
            else:
                items.append({"type": "text", "text": m.group(0)})
            last_end = end

        if not items:
            return [{"type": "text", "text": text}]
        if last_end < len(text):
            items.append({"type": "text", "text": text[last_end:]})
        return items

    def _block_to_content_item(
        self,
        block: ContentBlock,
        context: BuildContext | None,
    ) -> list[dict[str, Any]]:
        """Convert a content block into one or more Interactions Content items."""
        converted = try_convert_block(block)
        if converted is not None:
            block = converted

        if isinstance(block, TextBlock):
            if not block.text:
                return []
            model = context.model if context else None
            if model and GeminiInteractionsRequestBuilderMixin._is_gemini_image_model(model):
                return self._parse_gemini_markdown_images(block.text)
            return [{"type": "text", "text": block.text}]
        if isinstance(block, ImageBlock):
            if block.source.type == "base64":
                return [
                    {
                        "type": "image",
                        "data": block.source.data,
                        "mime_type": block.source.media_type or "image/png",
                    }
                ]
            if block.source.type in ("url", "file_id") and block.source.data:
                # HTTP(S) URLs are downloaded by the adapter before sending;
                # file_ids (Files API URIs) are passed through as uri.
                return [{"type": "image", "uri": block.source.data}]
            return []
        if isinstance(block, AudioBlock):
            if block.source.type == "base64":
                return [
                    {
                        "type": "audio",
                        "data": block.source.data,
                        "mime_type": block.source.media_type or "audio/mp3",
                    }
                ]
            if block.source.type in ("url", "file_id") and block.source.data:
                return [{"type": "audio", "uri": block.source.data}]
            return []
        if isinstance(block, VideoBlock):
            if block.source.type == "base64":
                return [
                    {
                        "type": "video",
                        "data": block.source.data,
                        "mime_type": block.source.media_type or "video/mp4",
                    }
                ]
            # Video URLs/file ids are fetched server-side; pass through as uri.
            if block.source.type in ("url", "file_id") and block.source.data:
                return [{"type": "video", "uri": block.source.data}]
            return []
        if isinstance(block, DocumentBlock):
            if block.source.type in ("text", "content") and block.source.data:
                doc_data = block.source.data
                doc_text = doc_data if isinstance(doc_data, str) else str(doc_data)
                return [{"type": "text", "text": doc_text}] if doc_text else []
            if block.source.type == "base64":
                return [
                    {
                        "type": "document",
                        "data": block.source.data,
                        "mime_type": block.source.media_type or "application/pdf",
                    }
                ]
            if block.source.type in ("url", "file_id") and block.source.data:
                return [{"type": "document", "uri": block.source.data}]
            return []
        if isinstance(block, FileBlock):
            if block.file_data:
                decoded = self._decode_data_uri(block.file_data)
                if decoded is not None:
                    mime_type, b64 = decoded
                    return [{"type": "document", "mime_type": mime_type, "data": b64}]
                return [{"type": "document", "uri": block.file_data}]
            if block.file_id:
                return [{"type": "document", "uri": block.file_id}]
            return []
        if isinstance(block, (ToolUseBlock, CustomToolUseBlock)):
            # Tool calls become their own steps; see _conversation_to_steps.
            return []
        if isinstance(block, ToolResultBlock):
            return []
        if isinstance(block, ThinkingBlock):
            # Old thoughts are not replayed to the model (no supported step
            # type for them); only the signature cache matters, which lives
            # on ToolUseBlocks.
            return []

        is_supported = (
            self.supported_content_blocks and type(block) in self.supported_content_blocks
        )
        policy = context.unsupported_block_policy if context else "drop"
        if not is_supported and not should_degrade_block(
            policy,
            block,
            self.provider_name,
            supported_blocks=self.supported_content_blocks,
        ):
            return []
        degraded = degrade_block_to_text(block)
        return [{"type": "text", "text": degraded}] if degraded else []

    # ── Tool call / result step helpers ───────────────────────────────

    @staticmethod
    def _tool_result_content(content: Any) -> Any:
        """Normalize ToolResultBlock.content into an Interactions result.

        Accepted forms per the API reference: array of Content, object, or
        string. JSON strings that parse to an object are kept as objects;
        plain text becomes a Content array.
        """
        if isinstance(content, str):
            try:
                parsed = orjson.loads(content)
            except orjson.JSONDecodeError, TypeError:
                parsed = None
            if isinstance(parsed, dict):
                return parsed
            return [{"type": "text", "text": content}]
        if isinstance(content, list):
            text_parts = [sub.text for sub in content if isinstance(sub, TextBlock) and sub.text]
            if text_parts:
                return [{"type": "text", "text": "\n".join(text_parts)}]
            return [{"type": "text", "text": ""}]
        return [{"type": "text", "text": str(content)}]

    # ── Conversation → input ──────────────────────────────────────────

    def _conversation_to_steps(
        self, conversation: ConversationContext, context: BuildContext | None
    ) -> list[dict[str, Any]]:
        """Convert a ConversationContext into an Interactions Step array."""
        steps: list[dict[str, Any]] = []

        # Map tool call ids to their (flattened) tool names so function_result
        # steps can reference the matching function_call by name; the call id
        # is not a valid name on the openresponses path.
        call_id_to_name: dict[str, str] = {}
        for msg in conversation.messages:
            if msg.role != "assistant":
                continue
            for block in msg.content:
                if isinstance(block, (ToolUseBlock, CustomToolUseBlock)) and block.id:
                    call_id_to_name[block.id] = flatten_history_tool_name(
                        context.namespace_map if context else None, block.name
                    )

        for msg in conversation.messages:
            if msg.role == "system":
                sys_text = msg.text_content
                wrapped = (
                    f"<system-prompt>\n{sys_text}\n</system-prompt>"
                    if sys_text
                    else "<system-prompt></system-prompt>"
                )
                steps.append({"type": "user_input", "content": [{"type": "text", "text": wrapped}]})
                continue

            if msg.role == "user":
                content = self._blocks_to_content_items(msg.content, context)
                if content:
                    steps.append({"type": "user_input", "content": content})
                continue

            if msg.role == "assistant":
                # Order matters: thought/model_output/function_call steps must
                # stay in their original chronological order, and consecutive
                # content blocks merge into a single model_output step.
                #
                # Stateless replay: the live API requires a thought step
                # (with a valid signature) before a model_output step when the
                # turn ends with a tool call. Clients like codex strip
                # thoughts from history, so reconstruct the thought from the
                # tool call's cached signature (the same one the original
                # thought carried) unless the message already has a
                # ThinkingBlock of its own.
                replay_signature: str | None = None
                if not any(isinstance(b, ThinkingBlock) for b in msg.content):
                    replay_signature = next(
                        (
                            b.extra.get("thought_signature")
                            for b in msg.content
                            if isinstance(b, (ToolUseBlock, CustomToolUseBlock))
                            and b.extra.get("thought_signature")
                        ),
                        None,
                    )
                thought_emitted = False
                pending_content: list[dict[str, Any]] = []

                def _flush_model_output(
                    _replay_signature: str | None = replay_signature,
                ) -> None:
                    nonlocal pending_content, thought_emitted
                    if pending_content:
                        if _replay_signature and not thought_emitted:
                            steps.append({"type": "thought", "signature": _replay_signature})
                            thought_emitted = True
                        steps.append({"type": "model_output", "content": pending_content})
                        pending_content = []

                for block in msg.content:
                    if isinstance(block, (ToolUseBlock, CustomToolUseBlock)):
                        _flush_model_output()
                        if isinstance(block, CustomToolUseBlock):
                            arguments: Any = {"content": block.input}
                        else:
                            arguments = block.input
                        step: dict[str, Any] = {
                            "type": "function_call",
                            "id": block.id or f"{block.name}_call",
                            "name": flatten_history_tool_name(
                                context.namespace_map if context else None, block.name
                            ),
                            "arguments": arguments,
                        }
                        # Stateless replay REQUIRES the thought signature on
                        # function_call steps (the live API rejects steps
                        # without it); the adapter re-attaches cached
                        # signatures via block.extra before building.
                        signature = block.extra.get("thought_signature")
                        if signature:
                            step["signature"] = signature
                        steps.append(step)
                        continue
                    if isinstance(block, ThinkingBlock):
                        _flush_model_output()
                        # Stateless replay REQUIRES resending thought blocks
                        # exactly as received (they carry the signatures the
                        # model needs to continue its reasoning).
                        thought_step: dict[str, Any] = {"type": "thought"}
                        if block.signature is not None:
                            thought_step["signature"] = block.signature
                        if block.thinking:
                            thought_step["summary"] = [{"type": "text", "text": block.thinking}]
                        steps.append(thought_step)
                        continue
                    content = self._blocks_to_content_items([block], context)
                    pending_content.extend(content)

                _flush_model_output()
                continue

            if msg.role == "tool":
                for block in msg.content:
                    if not isinstance(block, ToolResultBlock):
                        content = self._blocks_to_content_items([block], context)
                        if content:
                            steps.append({"type": "user_input", "content": content})
                        continue
                    raw_name = (
                        getattr(block, "name", None)
                        or call_id_to_name.get(block.tool_use_id)
                        or block.tool_use_id
                    )
                    steps.append(
                        {
                            "type": "function_result",
                            "call_id": block.tool_use_id,
                            "name": flatten_history_tool_name(
                                context.namespace_map if context else None, raw_name
                            ),
                            "result": self._tool_result_content(block.content),
                        }
                    )

        # Stateless replay: the live API validates only the CURRENT turn (the
        # steps after the last user_input) and requires its FIRST
        # function_call to carry a thought signature; an unsigned one is
        # rejected ("Function call is missing a thought_signature"). Clients
        # like codex strip thoughts and regenerate call ids on resume/rewind,
        # so the adapter's signature cache cannot always be hit. Degrade the
        # trailing unsigned tool turn to a user_input step (tool calls +
        # results as text) so the request still succeeds — the model loses
        # the structured tool-call history but keeps the information. Only
        # Gemini 3 series enforces the signature (2.5 treats it as optional),
        # so the fallback is gated on the model.
        if steps and steps[-1].get("type") == "function_result":
            turn_start = next(
                (
                    i + 1
                    for i in range(len(steps) - 1, -1, -1)
                    if steps[i].get("type") == "user_input"
                ),
                0,
            )
            first_fc = next(
                (
                    i
                    for i in range(turn_start, len(steps))
                    if steps[i].get("type") == "function_call"
                ),
                None,
            )
            if (
                first_fc is not None
                and "signature" not in steps[first_fc]
                and self._is_gemini_3_series(context.model if context else None)
            ):
                # Render every function_call/function_result pair of the
                # current turn to text; model_output/thought steps are kept
                # (once the turn no longer ends with a tool call they are
                # valid without a preceding thought).
                fr_by_call_id: dict[str, dict[str, Any]] = {}
                for s in steps[turn_start:]:
                    if s.get("type") == "function_result" and s.get("call_id"):
                        fr_by_call_id.setdefault(s["call_id"], s)
                degraded = [
                    self._tool_turn_to_text(s, fr_by_call_id[s["id"]])
                    for s in steps[turn_start:]
                    if s.get("type") == "function_call" and s.get("id") in fr_by_call_id
                ]
                if degraded:
                    fc_id = steps[first_fc].get("id")
                    kept = [
                        s
                        for s in steps[turn_start:]
                        if s.get("type") not in ("function_call", "function_result")
                    ]
                    del steps[turn_start:]
                    steps.extend(kept)
                    steps.append(
                        {
                            "type": "user_input",
                            "content": [{"type": "text", "text": "\n\n".join(degraded)}],
                        }
                    )
                    logger.warning(
                        "Gemini Interactions: the current turn's first function_call "
                        "%r has no thought signature (client regenerated the call id?); "
                        "degraded the trailing tool turn to a user_input step",
                        fc_id,
                    )

        return steps

    @staticmethod
    def _is_gemini_3_series(model: str | None) -> bool:
        """Whether the target model is a Gemini 3 series model.

        Gemini 3 strictly requires a thought signature on the first
        function_call of the current turn; Gemini 2.5 treats it as optional
        (per the thought-signatures docs), so the degradation fallback is
        gated on 3-series models.
        """
        return bool(model) and "gemini-3" in model

    @staticmethod
    def _tool_turn_to_text(fc: dict[str, Any], fr: dict[str, Any]) -> str:
        """Render a function_call + function_result pair as plain text."""
        name = fc.get("name", "")
        arguments = fc.get("arguments")
        if isinstance(arguments, dict):
            arguments = orjson.dumps(arguments).decode()
        result = fr.get("result")
        if isinstance(result, str):
            result_text = result
        elif isinstance(result, list):
            result_text = "\n".join(
                c.get("text", "") for c in result if isinstance(c, dict) and c.get("text")
            )
        elif isinstance(result, dict):
            result_text = orjson.dumps(result).decode()
        else:
            result_text = ""
        parts = [f"Tool call {name}:"]
        if arguments:
            parts.append(str(arguments))
        parts.append("Result:")
        parts.append(result_text or "(no output)")
        return "\n".join(parts)

    def _blocks_to_content_items(
        self, blocks: list[ContentBlock], context: BuildContext | None
    ) -> list[dict[str, Any]]:
        """Convert a block list into Interactions Content items."""
        items: list[dict[str, Any]] = []
        for block in blocks:
            items.extend(self._block_to_content_item(block, context))
        # Drop empty placeholder text items when there is other content.
        meaningful = [i for i in items if not (i.get("type") == "text" and i.get("text") == "")]
        return meaningful or items

    def _convert_conversation_to_input(
        self,
        conversation: ConversationContext,
        context: BuildContext | None = None,
    ) -> list[dict[str, Any]]:
        """Build the ``input`` field of an Interactions request.

        Returns a Step array (user_input/model_output/function_call/
        function_result); a bare string is only used for single text turns by
        callers that prefer it.
        """
        steps = self._conversation_to_steps(conversation, context)
        return steps or [{"type": "user_input", "content": [{"type": "text", "text": ""}]}]

    def _system_instruction_text(self, conversation: ConversationContext) -> str | None:
        """Join system message text into the top-level ``system_instruction``."""
        parts: list[str] = []
        for sys_msg in conversation.system_messages:
            if sys_msg.content:
                parts.append(sys_msg.text_content)
        return "\n".join(parts) if parts else None
