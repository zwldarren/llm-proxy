"""Cohere v2 chat: ``InternalRequest`` conversation -> ``messages`` wire array.

``POST /v2/chat`` takes one chronological ``messages`` list with four roles —
``system``, ``user``, ``assistant`` and ``tool`` — which replaced v1's
``preamble`` / ``message`` / ``chat_history`` triple
(docs.cohere.com/docs/migrating-v1-to-v2). Content is either a plain string or a
list of blocks; the user and system roles know ``text`` and ``image_url``, while
the assistant role additionally carries ``tool_calls``, ``tool_plan`` and
``citations`` (docs.cohere.com/reference/chat).

This module renders the conversation only. Which body fields exist, and what
they are called, is the serializer's business.
"""

import json
from typing import Any

from llm_proxy.models import (
    ContentBlock,
    ConversationContext,
    ImageBlock,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from llm_proxy.serialization._shared_degradation import (
    degrade_block_to_text,
    should_degrade_block,
)
from llm_proxy.serialization.context import BuildContext
from llm_proxy.serialization.responses_toolkit.namespace import flatten_history_tool_name

#: Roles that v2 folds into the single leading ``system`` message: the API has
#: one system slot, and OpenAI's ``developer`` spelling has no v2 equivalent.
_SYSTEM_ROLES = frozenset({"system", "developer"})


def _media_url(source: Any, default_media_type: str = "image/png") -> str:
    """Render a media source as the URL an ``image_url`` block takes.

    Byte sources become a ``data:`` URI (Cohere's documented inline form);
    remote URLs and file ids pass through unchanged — the same projection the
    Chat Completions dialects make in ``serialization/openai/converter.py``.
    """
    if source.type == "base64":
        return f"data:{source.media_type or default_media_type};base64,{source.data}"
    return source.data


class CohereConversationMixin:
    """Convert a ``ConversationContext`` to Cohere v2 chat messages."""

    provider_name: str = "cohere"
    supported_content_blocks: frozenset[type[ContentBlock]] = frozenset(
        {
            TextBlock,
            ImageBlock,
            ToolUseBlock,
            ToolResultBlock,
            ThinkingBlock,
        }
    )

    def _build_cohere_messages(
        self, request: Any, context: BuildContext | None = None
    ) -> list[dict[str, Any]]:
        """Render the request's conversation as Cohere's ``messages`` array."""
        conversation: ConversationContext = request.conversation
        policy = context.unsupported_block_policy if context else "drop"
        namespace_map = context.namespace_map if context else None

        messages: list[dict[str, Any]] = []
        system_text = self._cohere_system_text(conversation)
        if system_text:
            # v2 requires the system turn to lead the array.
            messages.append({"role": "system", "content": system_text})

        for message in conversation.messages:
            if message.role in _SYSTEM_ROLES:
                # Already merged into the leading system message above: v2 has
                # no mid-conversation system slot.
                continue
            messages.extend(self._cohere_messages_for(message, policy, namespace_map))
        return messages

    def _cohere_system_text(self, conversation: ConversationContext) -> str | None:
        """The single system instruction v2 allows, folded from every system turn.

        System and developer messages (including mid-conversation ones, which
        v2 cannot place) are concatenated in order: dropping them would lose the
        instruction, and v2 has exactly one slot for all of them.
        """
        texts: list[str] = []
        for system_message in conversation.system_messages:
            texts.extend(
                block.text for block in system_message.content if isinstance(block, TextBlock)
            )
        for message in conversation.messages:
            if message.role in _SYSTEM_ROLES:
                texts.extend(
                    block.text for block in message.content if isinstance(block, TextBlock)
                )
        joined = "\n\n".join(text for text in texts if text)
        return joined or None

    def _cohere_messages_for(
        self,
        message: Message,
        policy: str,
        namespace_map: dict[str, list[str]] | None,
    ) -> list[dict[str, Any]]:
        """Render one internal message as one or more v2 messages.

        A tool result inside a non-tool turn becomes its own ``role: "tool"``
        message, because v2 states tool output per ``tool_call_id`` rather than
        inline in the conversation turn.
        """
        assistant = message.role == "assistant"
        parts: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []
        tool_plan: str | None = None
        tool_messages: list[dict[str, Any]] = []

        for block in message.content:
            match block:
                case TextBlock():
                    parts.append({"type": "text", "text": block.text})
                case ThinkingBlock() if assistant:
                    # v2 replays the model's own reasoning as a content block on
                    # the assistant turn.
                    parts.append({"type": "thinking", "thinking": block.thinking})
                case ImageBlock():
                    image_url: dict[str, Any] = {"url": _media_url(block.source)}
                    # ``detail`` is Cohere's own auto/low/high enum; an explicit
                    # null is off-schema, so it is omitted when unset.
                    if block.detail is not None:
                        image_url["detail"] = block.detail
                    parts.append({"type": "image_url", "image_url": image_url})
                case ToolUseBlock():
                    tool_calls.append(
                        {
                            "id": block.id,
                            "type": "function",
                            "function": {
                                # History calls carry the short name; the
                                # definitions sent upstream are flattened, and a
                                # model echoes what it saw.
                                "name": flatten_history_tool_name(namespace_map, block.name),
                                "arguments": json.dumps(block.input or {}),
                            },
                        }
                    )
                    if tool_plan is None:
                        # The chain-of-thought plan Cohere returns beside a tool
                        # call. The dialect has no first-class internal block for
                        # it, so the parser parks it on the call it belongs to and
                        # this path hands it back where it arrived.
                        stored = block.extra.get("cohere_tool_plan")
                        tool_plan = stored if isinstance(stored, str) else None
                case ToolResultBlock():
                    tool_messages.append(self._cohere_tool_message(block, policy))
                case _:
                    if self._cohere_dropped(block, policy):
                        continue
                    degraded = degrade_block_to_text(block)
                    if degraded:
                        parts.append({"type": "text", "text": degraded})

        rendered: list[dict[str, Any]] = []
        if assistant:
            message_body: dict[str, Any] = {"role": "assistant"}
            if parts:
                message_body["content"] = self._cohere_content(parts)
            if tool_calls:
                message_body["tool_calls"] = tool_calls
            if tool_plan:
                message_body["tool_plan"] = tool_plan
            if len(message_body) > 1:
                rendered.append(message_body)
        elif parts:
            rendered.append({"role": "user", "content": self._cohere_content(parts)})
        rendered.extend(tool_messages)
        return rendered

    @staticmethod
    def _cohere_content(parts: list[dict[str, Any]]) -> str | list[dict[str, Any]]:
        """v2 accepts a bare string for plain text; blocks otherwise."""
        if len(parts) == 1 and parts[0].get("type") == "text":
            return parts[0]["text"]
        return parts

    def _cohere_dropped(self, block: ContentBlock, policy: str) -> bool:
        """Whether an unsupported block is dropped instead of degraded to text."""
        is_supported = bool(self.supported_content_blocks) and type(block) in (
            self.supported_content_blocks
        )
        if is_supported:
            return False
        return not should_degrade_block(
            policy,
            block,
            self.provider_name,
            supported_blocks=self.supported_content_blocks,
        )

    def _cohere_tool_message(self, block: ToolResultBlock, policy: str) -> dict[str, Any]:
        """Render one tool result as v2's ``role: "tool"`` message."""
        content_raw = block.content
        if isinstance(content_raw, str):
            content = content_raw
        else:
            texts: list[str] = []
            for sub_block in content_raw:
                if isinstance(sub_block, TextBlock):
                    texts.append(sub_block.text)
                    continue
                if self._cohere_dropped(sub_block, policy):
                    continue
                degraded = degrade_block_to_text(sub_block)
                if degraded:
                    texts.append(degraded)
            content = "\n".join(texts)
        return {
            "role": "tool",
            # v2 matches tool output to the call it answers by id; there is no
            # positional pairing.
            "tool_call_id": block.tool_use_id,
            "content": content,
        }


__all__ = ["CohereConversationMixin"]
