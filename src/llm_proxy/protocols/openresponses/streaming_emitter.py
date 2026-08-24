"""Content emitter for OpenResponses streaming format.

Handles content type transitions and emits appropriate SSE events during
streaming. Extracted from OpenResponsesStreamingTransformer to separate
content emission concerns from streaming orchestration.
"""

from typing import Any

from llm_proxy.models.tools import is_web_search_tool_name
from llm_proxy.protocols.openresponses.serializer import match_custom_tool_name
from llm_proxy.protocols.openresponses.streaming_events import StreamingEventFactory
from llm_proxy.serialization.responses_toolkit import generate_item_id

# Item types that carry tool calls rather than text content. Text/reasoning
# deltas must never be attributed to these items; when the model interleaves
# text between tool calls, the open tool-call item is closed first.
_TOOL_CALL_ITEM_TYPES = (
    "function_call",
    "custom_tool_call",
    "web_search_call",
    "tool_search_call",
)


class StreamingContentEmitter:
    """Emits content events for OpenResponses streaming.

    Manages content type transitions (reasoning/text/refusal) and delegates
    event creation to StreamingEventFactory. Reads and writes shared streaming
    state to track item positions and accumulated content.

    Args:
        state: Shared streaming state for reading/writing item and content data
        factory: Event factory for creating SSE events
    """

    def __init__(
        self,
        state: Any,  # OpenResponsesStreamingState - avoids circular import
        factory: StreamingEventFactory,
    ) -> None:
        self.state = state
        self._factory = factory

    def _emit_reasoning_content(self, content: str) -> str:
        """Emit reasoning content events."""
        return self._emit_content(content, "reasoning_text")

    def _emit_text_content(self, content: str) -> str:
        """Emit text content events."""
        return self._emit_content(content, "output_text")

    def _emit_refusal_content(self, content: str) -> str:
        """Emit refusal content events."""
        return self._emit_content(content, "refusal")

    def _emit_content(self, content: str, content_type: str) -> str:
        """Emit content events for a given content type.

        Handles content type transitions, creates output items and content parts
        as needed, and emits the appropriate delta event.

        Args:
            content: The content text to emit
            content_type: One of "reasoning_text", "output_text", or "refusal"

        Returns:
            SSE events string
        """
        events = ""

        # Transition to a new item if content type changed
        if (
            self.state.current_content_type is not None
            and self.state.current_content_type != content_type
        ):
            events += self._close_current_item()

        # Tool-call items hold no text content. When the current slot is a
        # tool-call item (the model interleaves text between tool calls),
        # close it so the delta starts a fresh message/reasoning item instead
        # of referencing the tool-call item id. Loop because the next slot
        # may also hold a tool-call item.
        current_pending = self.state.pending_items.get(self.state.current_item_index)
        while current_pending is not None and current_pending.get("type") in _TOOL_CALL_ITEM_TYPES:
            events += self._close_current_item()
            current_pending = self.state.pending_items.get(self.state.current_item_index)

        self.state.current_content_type = content_type

        if self.state.current_item_index not in self.state.pending_items:
            item_id = generate_item_id()
            item_type = "reasoning" if content_type == "reasoning_text" else "message"
            self.state.pending_items[self.state.current_item_index] = {
                "id": item_id,
                "type": item_type,
            }
            events += self._factory._create_output_item_added_event(
                item_id=item_id,
                item_type=item_type,
            )

        item_idx = self.state.current_item_index
        content_idx = self.state.current_content_index.get(item_idx, 0)
        key = (item_idx, content_idx)
        if key not in self.state.accumulated_text:
            item_id = self.state.pending_items[item_idx]["id"]
            if content_type == "reasoning_text":
                # Reasoning text streams via the summary part lifecycle
                # (response.reasoning_summary_part.added), the event family
                # whose names are identical in the OpenResponses spec and the
                # OpenAI Responses API.
                events += self._factory._create_reasoning_summary_part_added_event(
                    item_index=item_idx,
                    summary_index=0,
                    item_id=item_id,
                )
            else:
                events += self._factory._create_content_part_added_event(
                    item_index=item_idx,
                    content_index=content_idx,
                    content_type=content_type,
                    item_id=item_id,
                )
            self.state.accumulated_text[key] = ""
            self.state.content_types[key] = content_type

        item_id = self.state.pending_items[item_idx]["id"]
        if content_type == "reasoning_text":
            events += self._factory._create_reasoning_summary_text_delta_event(
                item_index=item_idx,
                summary_index=0,
                text=content,
                item_id=item_id,
            )
        elif content_type == "refusal":
            events += self._factory._create_refusal_delta_event(
                item_index=item_idx,
                content_index=content_idx,
                text=content,
                item_id=item_id,
            )
        else:
            events += self._factory._create_output_text_delta_event(
                item_index=item_idx,
                content_index=content_idx,
                text=content,
                item_id=item_id,
            )

        self.state.accumulated_text[key] += content

        return events

    def _close_current_item(self) -> str:
        """Close the current item when transitioning between content types.

        Emits done events for the current content part and item,
        then increments the item index for the next item.

        Returns:
            SSE events string for closing the current item
        """
        events = ""
        item_idx = self.state.current_item_index

        if item_idx not in self.state.pending_items:
            self.state.current_item_index += 1
            return events

        item = self.state.pending_items[item_idx]
        item_id = item["id"]
        item_type = item["type"]

        if item_type in _TOOL_CALL_ITEM_TYPES:
            # Tool-call items close via the tool-call event family
            # (function_call_arguments.done), not text content events.
            events += self._close_tool_call_item(item_idx)
            return events

        content_idx = self.state.current_content_index.get(item_idx, 0)
        key = (item_idx, content_idx)
        accumulated = self.state.accumulated_text.get(key, "")
        content_type = self.state.content_types.get(key, "output_text")

        if accumulated:
            events += self._factory._create_content_done_event(
                item_index=item_idx,
                content_index=content_idx,
                text=accumulated,
                item_id=item_id,
                content_type=content_type,
                item_type=item_type,
            )

        if content_type == "reasoning_text":
            events += self._factory._create_reasoning_summary_part_done_event(
                item_index=item_idx,
                summary_index=0,
                text=accumulated,
                item_id=item_id,
            )
        else:
            events += self._factory._create_content_part_done_event(
                item_index=item_idx,
                content_index=content_idx,
                item_id=item_id,
                content_type=content_type,
                text=accumulated,
            )

        events += self._factory._create_output_item_done_event(
            item_id=item_id,
            item_index=item_idx,
            item_type=item_type,
            status="completed",
            content_type=content_type,
            text=accumulated,
        )

        self.state.closed_items.add(item_idx)
        self.state.current_item_index += 1

        return events

    def _close_tool_call_item(self, item_idx: int) -> str:
        """Close an open tool-call item with the tool-call event family.

        Mirrors the tool-call closing logic in ``_process_finish`` so an item
        interrupted by text/reasoning content is closed with
        ``function_call_arguments.done`` + ``output_item.done`` instead of
        text events. ``web_search_call`` items are left open: their
        ``output_item.added`` is deferred to ``_process_finish`` (where the
        complete query is available), so only the index is advanced.

        Args:
            item_idx: Output index of the tool-call item to close

        Returns:
            SSE events string for closing the item
        """
        events = ""
        item = self.state.pending_items.get(item_idx)
        if item is None:
            self.state.current_item_index += 1
            return events

        item_type = item["type"]
        if item_type == "web_search_call" or item_idx in self.state.web_search_tool_indices:
            self.state.current_item_index += 1
            return events

        item_id = item["id"]
        if item_type == "function_call":
            arguments = self.state.accumulated_tool_args.get(item_idx, "")
            events += self._factory._create_function_call_arguments_done_event(
                item_index=item_idx,
                arguments=arguments,
                item_id=item_id,
            )

        events += self._factory._create_output_item_done_event(
            item_id=item_id,
            item_index=item_idx,
            item_type=item_type,
            status="completed",
        )

        self.state.closed_items.add(item_idx)
        self.state.current_item_index += 1

        return events

    def _get_output_index_for_tool_call(self, provider_index: int | None) -> int:
        """Get or create a unique output index for a tool call.

        Maps the provider's tool_call.index to a unique output item index that
        doesn't collide with closed/pending content items (reasoning/message).
        This ensures tool calls work correctly when they arrive interleaved with
        text content from providers like DeepSeek that may send both in the same
        chunk or have tool call indices that reset.

        Args:
            provider_index: The tool_call index from the provider (may be None or 0-based)

        Returns:
            A unique output item index for this tool call
        """
        if provider_index is not None and provider_index in self.state.tool_call_index_map:
            return self.state.tool_call_index_map[provider_index]

        output_idx = self.state.current_item_index
        while (
            output_idx in self.state.pending_items
            or output_idx in self.state.closed_items
            or output_idx in self.state.tool_call_index_map.values()
        ):
            output_idx += 1

        if provider_index is not None:
            self.state.tool_call_index_map[provider_index] = output_idx

        return output_idx

    def _emit_tool_calls(self, tool_calls: list[dict[str, Any]]) -> str:
        """Emit tool call events.

        Skips event emission for server-side web_search tool calls
        (handled internally by the backend, not shown to the client).

        Args:
            tool_calls: List of tool call deltas

        Returns:
            SSE events string
        """
        events = ""
        self.state.has_tool_calls = True

        # Close any open content item before processing tool calls to prevent index
        # collision with content items (reasoning/message) that may already be closed.
        if (
            self.state.current_item_index in self.state.pending_items
            and self.state.pending_items.get(self.state.current_item_index, {}).get("type")
            not in ("function_call", "custom_tool_call", "web_search_call", "tool_search_call")
        ):
            events += self._close_current_item()

        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue

            tc_index = self._get_output_index_for_tool_call(tc.get("index"))
            tc_id = tc.get("id")
            function = tc.get("function", {})
            thought_sig = tc.get("thought_signature")
            tool_name = function.get("name", "")

            is_web_search = is_web_search_tool_name(tool_name)
            # Tolerant matching: models often echo the short history name
            # (``exec``) rather than the flattened tool-definition name
            # (``functions__exec``) that custom_tool_names carries.
            is_custom = match_custom_tool_name(tool_name, self.state.custom_tool_names) is not None
            is_tool_search = tool_name == "tool_search"
            if is_tool_search:
                item_type = "tool_search_call"
            elif is_custom:
                item_type = "custom_tool_call"
            else:
                item_type = "function_call"

            if tc_id and tc_index not in self.state.tool_call_ids:
                item_id = generate_item_id()
                self.state.tool_call_ids[tc_index] = tc_id
                self.state.tool_call_names[tc_index] = tool_name
                self.state.accumulated_tool_args[tc_index] = ""
                if thought_sig:
                    self.state.tool_call_thought_signatures[tc_index] = thought_sig

                if is_web_search and not self.state.intercept_web_search:
                    # Native provider-executed search: count for billing.
                    self.state.native_web_search_call_count += 1

                if is_web_search and self.state.intercept_web_search:
                    # Server-side interceptor will execute the search;
                    # skip client emission.
                    self.state.web_search_tool_indices.add(tc_index)
                elif is_web_search and not self.state.web_search_as_function:
                    # Builtin {"type": "web_search"} declaration: the client
                    # expects the search to have been executed server-side, so
                    # report it as a completed web_search_call. output_item.added
                    # is deferred to _process_finish where the complete query
                    # is available.
                    self.state.pending_items[tc_index] = {
                        "id": item_id,
                        "type": "web_search_call",
                    }
                else:
                    # Emit the call back as a regular output item — including
                    # web_search declared as a client function tool (e.g. Hermes
                    # Agent), where a web_search_call would make the client
                    # believe the search already ran server-side and silently
                    # end the turn.
                    self.state.pending_items[tc_index] = {
                        "id": item_id,
                        "type": item_type,
                    }
                    events += self._factory._create_output_item_added_event(
                        item_id=item_id,
                        item_type=item_type,
                        call_id=tc_id,
                        name=tool_name,
                        thought_signature=thought_sig,
                        item_index=tc_index,
                    )

            args = function.get("arguments")
            if args:
                if tc_index not in self.state.accumulated_tool_args:
                    self.state.accumulated_tool_args[tc_index] = ""
                self.state.accumulated_tool_args[tc_index] += args

                # Skip function_call_arguments.delta for web_search, custom,
                # and tool_search calls — content emitted in output_item.done.
                # Check the *registered* pending item type (state-based) rather
                # than per-chunk flags: argument-only chunks omit the tool name,
                # and web_search calls emitted client-side (intercept_web_search
                # False) are not in web_search_tool_indices. Emitting a delta for
                # such items would reference an item_id before output_item.added.
                pending_item = self.state.pending_items.get(tc_index)
                pending_type = pending_item.get("type") if pending_item else None
                if tc_index not in self.state.web_search_tool_indices and pending_type not in (
                    "web_search_call",
                    "custom_tool_call",
                    "tool_search_call",
                ):
                    if pending_item is None:
                        # No item registered for this index (arguments chunk
                        # without a call id): cannot emit a delta referencing an
                        # unknown item; arguments are still emitted via
                        # function_call_arguments.done in _process_finish.
                        continue
                    events += self._factory._create_function_call_arguments_delta_event(
                        item_index=tc_index,
                        text=args,
                        item_id=pending_item["id"],
                    )

        return events
