"""Replay stored Responses items back into the unified conversation.

The inverse of materialize (``conversation_to_input_items``): converts the
input/output items of a stored response into unified messages and prepends
them to the live conversation. Used for ``previous_response_id`` continuations
and for resolving ``item_reference`` entries that point at items stored with
the previous response.
"""

from types import SimpleNamespace
from typing import Any

from llm_proxy.models import ConversationContext, Message, SystemMessage
from llm_proxy.observability.logger import get_logger
from llm_proxy.protocols.openresponses.serializer import (
    _dispatch_input_item,
    _flush_pending_turn,
)

logger = get_logger(__name__)


def replay_stored_response(
    stored: dict[str, Any],
    conversation: ConversationContext,
    unresolved_refs: list[tuple[int, str]] | None = None,
) -> int:
    """Prepend a stored response's items to the conversation.

    Both item lists go through the same dispatch logic used at request-parse
    time, so every item type (custom_tool_call, local_shell_call,
    tool_search_call/output, web_search_call, agent_message, compaction,
    item_reference, ...) round-trips exactly as it would in a request body —
    including rehydration of proxy-produced compaction blobs and preservation
    of the assistant ``phase`` label.

    When ``unresolved_refs`` (``(message_index, ref_id)`` pairs recorded at
    parse time) is given, the referenced items are spliced into the
    conversation at their recorded positions after the prepend.

    Returns the number of messages prepended.
    """
    prev_messages: list[Message] = []
    prev_system: list[SystemMessage] = []

    prev_instructions = stored.get("instructions")
    if prev_instructions:
        prev_system.append(SystemMessage.from_text(role="system", text=prev_instructions))

    pending_blocks: list = []
    pending_ws: list = []
    pending_phase: list = [None]
    # item_reference entries resolve against items seen earlier in the
    # stored input/output.
    seen_items: dict[str, dict[str, Any]] = {}
    # additional_tools items inside a stored turn are turn-scoped
    # injections; absorb them into a throwaway holder (the current request
    # re-declares its own tools) instead of polluting InternalRequest.tools
    # with raw dicts.
    tools_sink = SimpleNamespace(tools=None)

    for source in (stored.get("input"), stored.get("output")):
        if isinstance(source, str):
            source = [{"type": "message", "role": "user", "content": source}] if source else []
        if not isinstance(source, list):
            continue
        for item in source:
            if isinstance(item, dict):
                item_dict = item
            elif hasattr(item, "model_dump"):
                item_dict = item.model_dump()
            else:
                continue
            _dispatch_input_item(
                dict(item_dict),
                prev_messages,
                prev_system,
                pending_blocks,
                pending_ws,
                pending_phase,
                tools_sink,
                seen_items,
                None,
            )

    _flush_pending_turn(pending_blocks, pending_ws, prev_messages, pending_phase)

    if prev_system:
        conversation.system_messages = [*prev_system, *conversation.system_messages]
    # Prepend previous messages before current messages
    conversation.messages = prev_messages + conversation.messages

    prepended_count = len(prev_messages)
    if unresolved_refs:
        _splice_item_references(stored, conversation, unresolved_refs, prepended_count)
    return prepended_count


def _splice_item_references(
    stored: dict[str, Any],
    conversation: ConversationContext,
    unresolved_refs: list[tuple[int, str]],
    prepended_count: int,
) -> None:
    """Splice item_reference targets from a stored response into the conversation.

    Parse-time resolution of ``item_reference`` only sees items within the
    current request body; references to items stored with a previous response
    are recorded as ``(message_index, ref_id)`` pairs (the message boundary
    where the reference appeared in the new input). The referenced items are
    converted through the same dispatch logic used at parse time and inserted
    at their recorded positions (shifted by the number of prepended messages
    and by earlier insertions).
    """
    catalog: dict[str, dict[str, Any]] = {}
    for source in (stored.get("input"), stored.get("output")):
        if isinstance(source, list):
            for item in source:
                if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"]:
                    catalog[item["id"]] = item
    if not catalog:
        return

    inserted = 0
    for msg_index, ref_id in sorted(unresolved_refs):
        item = catalog.get(ref_id)
        if item is None:
            logger.warning(f"item_reference '{ref_id}' not found in previous response; skipping")
            continue
        produced: list[Message] = []
        pending_blocks: list = []
        pending_ws: list = []
        pending_phase: list = [None]
        # additional_tools items inside a resolved reference are turn-scoped
        # injections from the *previous* turn; absorb them into a throwaway
        # holder (the current request declares its own tools) instead of
        # polluting InternalRequest.tools with raw dicts — same treatment as
        # replay_stored_response's tools_sink.
        tools_sink = SimpleNamespace(tools=None)
        try:
            _dispatch_input_item(
                dict(item),
                produced,
                conversation.system_messages,
                pending_blocks,
                pending_ws,
                pending_phase,
                tools_sink,
                {},
            )
            _flush_pending_turn(pending_blocks, pending_ws, produced, pending_phase)
        except Exception:
            logger.warning(f"Failed to resolve item_reference '{ref_id}'", exc_info=True)
            continue
        if not produced:
            continue
        pos = min(prepended_count + msg_index + inserted, len(conversation.messages))
        conversation.messages[pos:pos] = produced
        inserted += len(produced)
