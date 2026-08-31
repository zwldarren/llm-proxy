"""Event factory for OpenResponses streaming format.

Creates SSE events conforming to the OpenResponses protocol specification.
Extracted from OpenResponsesStreamingTransformer to separate event creation
concerns from streaming orchestration.
"""

import time
from typing import Any

import orjson

from llm_proxy.core.reasoning_cache import try_cache_reasoning_from_responses_output
from llm_proxy.observability.logger import get_logger

logger = get_logger(__name__)


def _create_output_text_part(text: str) -> dict[str, Any]:
    """Create a schema-complete OpenResponses output_text content part."""
    return {
        "type": "output_text",
        "text": text,
        "annotations": [],
        "logprobs": [],
    }


def _create_summary_text_part(text: str) -> dict[str, Any]:
    """Create a schema-complete summary_text content part."""
    return {
        "type": "summary_text",
        "text": text,
    }


def _reasoning_encrypted_content(state: Any, item_index: int) -> str | None:
    """Encrypted content for a reasoning item (include-gated or signature fallback).

    The include-gated payload (``reasoning.encrypted_content`` requested) wins.
    Otherwise an Anthropic-origin thinking signature is bridged into
    ``encrypted_content`` — the only Responses field that round-trips it — so
    replaying the item keeps multi-turn extended thinking working.
    """
    if state.include_reasoning_encrypted:
        encrypted = (
            state.reasoning_encrypted_contents.get(item_index) or state.reasoning_encrypted_content
        )
        if encrypted:
            return encrypted
    return state.reasoning_signatures.get(item_index)


def _build_openresponses_response(
    *,
    response_id: str,
    created_at: int,
    status: str,
    model: str,
    output: list[dict[str, Any]],
    input_tokens: int,
    output_tokens: int,
    completed_at: int | None = None,
    incomplete_details: dict[str, Any] | None = None,
    reasoning: dict[str, Any] | None = None,
    cached_tokens: int = 0,
    reasoning_tokens: int = 0,
    usage_none: bool = False,
    error: dict[str, Any] | None = None,
    store: bool = False,
    previous_response_id: str | None = None,
    instructions: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = "auto",
    truncation: str = "disabled",
    parallel_tool_calls: bool = True,
    text: Any | None = None,
    top_p: float = 1.0,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    top_logprobs: int = 0,
    temperature: float = 1.0,
    max_output_tokens: int | None = None,
    max_tool_calls: int | None = None,
    background: bool = False,
    service_tier: str = "default",
    metadata: dict[str, str] | None = None,
    safety_identifier: str | None = None,
    prompt_cache_key: str | None = None,
) -> dict[str, Any]:
    """Build a schema-complete OpenResponses response payload for streaming completion."""
    if usage_none:
        usage: dict[str, Any] | None = None
    else:
        usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_tokens_details": {"cached_tokens": cached_tokens},
            "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
        }
    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "completed_at": completed_at,
        "status": status,
        "model": model,
        "previous_response_id": previous_response_id,
        "instructions": instructions,
        "output": output,
        "error": error,
        "tools": tools or [],
        "tool_choice": tool_choice,
        "truncation": truncation,
        "parallel_tool_calls": parallel_tool_calls,
        "text": text or {"format": {"type": "text"}},
        "top_p": top_p,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "top_logprobs": top_logprobs,
        "temperature": temperature,
        "reasoning": reasoning,
        "usage": usage,
        "max_output_tokens": max_output_tokens,
        "max_tool_calls": max_tool_calls,
        "store": store,
        "background": background,
        "service_tier": service_tier,
        "metadata": metadata or {},
        "safety_identifier": safety_identifier,
        "prompt_cache_key": prompt_cache_key,
        "incomplete_details": incomplete_details,
    }


class StreamingEventFactory:
    """Factory for creating OpenResponses SSE events.

    Encapsulates all event creation logic, building properly formatted
    SSE event strings conforming to the OpenResponses protocol specification.
    Each event increments the sequence number to maintain event ordering.

    Args:
        state: Shared streaming state for reading sequence numbers and item data
        response_id: The response ID for this streaming session
        model: The model name for this streaming session
    """

    def __init__(
        self,
        state: Any,  # OpenResponsesStreamingState - avoids circular import
        response_id: str,
        model: str,
    ) -> None:
        self.state = state
        self.response_id = response_id
        self.model = model

    def _echo_kwargs(self) -> dict[str, Any]:
        """Request-field echo for response snapshots, read from the state.

        Mirrors the non-streaming formatter's echo logic so streaming
        snapshots carry the same effective request configuration (spec
        ResponseResource required fields).
        """
        s = self.state
        return {
            "previous_response_id": getattr(s, "previous_response_id", None),
            "instructions": getattr(s, "instructions", None),
            "tools": getattr(s, "raw_tools", None),
            "tool_choice": (
                s.tool_choice if getattr(s, "tool_choice", None) is not None else "auto"
            ),
            "truncation": getattr(s, "truncation", None) or "disabled",
            "parallel_tool_calls": (
                True if getattr(s, "parallel_tool_calls", None) is None else s.parallel_tool_calls
            ),
            "text": getattr(s, "text", None),
            "top_p": s.top_p if getattr(s, "top_p", None) is not None else 1.0,
            "presence_penalty": (
                s.presence_penalty if getattr(s, "presence_penalty", None) is not None else 0.0
            ),
            "frequency_penalty": (
                s.frequency_penalty if getattr(s, "frequency_penalty", None) is not None else 0.0
            ),
            "top_logprobs": getattr(s, "top_logprobs", None) or 0,
            "temperature": s.temperature if getattr(s, "temperature", None) is not None else 1.0,
            "max_output_tokens": getattr(s, "max_output_tokens", None),
            "max_tool_calls": getattr(s, "max_tool_calls", None),
            "background": bool(getattr(s, "background", None)),
            "service_tier": getattr(s, "service_tier", None) or "default",
            "metadata": getattr(s, "metadata", None),
            "safety_identifier": getattr(s, "safety_identifier", None),
            "prompt_cache_key": getattr(s, "prompt_cache_key", None),
        }

    def _restore_name_and_namespace(self, name: str) -> tuple[str, str | None]:
        """Restore original tool name and namespace from the namespace mapping.

        Accepts both the flattened definition name (``mcp__github__create_issue``)
        and the original short name (``create_issue`` — models often echo the
        short history name). If the given name is a flattened key in the
        namespace_map, return the original short name and its namespace.
        Otherwise, return the name unchanged with no namespace.
        """
        from llm_proxy.serialization.responses_toolkit.namespace import (
            restore_tool_name,
        )

        return restore_tool_name(self.state.namespace_map, name)

    def _set_call_name(self, item: dict[str, Any], name: str) -> None:
        """Set a call item's name, restoring the namespace when flattened."""
        restored_name, namespace = self._restore_name_and_namespace(name)
        item["name"] = restored_name
        if namespace:
            item["namespace"] = namespace

    def _set_tool_call_args(self, item: dict[str, Any], item_index: int) -> None:
        """Set tool-call-specific arguments on an item from accumulated state."""
        item_type = item.get("type")
        args = self.state.accumulated_tool_args.get(item_index, "")
        if item_type == "custom_tool_call":
            from llm_proxy.protocols.openresponses.serializer import (
                unwrap_custom_tool_arguments,
            )

            item["input"] = unwrap_custom_tool_arguments(args)
        elif item_type == "tool_search_call":
            item["execution"] = "client"
            try:
                item["arguments"] = orjson.loads(args) if args else {}
            except orjson.JSONDecodeError:
                item["arguments"] = {}
        else:
            item["arguments"] = args

    def _create_sse_event(self, event_type: str, data: dict[str, Any]) -> str:
        """Create an SSE event string.

        Args:
            event_type: The event type name
            data: The event data

        Returns:
            SSE formatted string
        """
        self.state.sequence_number += 1
        data["sequence_number"] = self.state.sequence_number
        data_str = orjson.dumps(data).decode()
        # Spec: the event field MUST match the type in the event body.
        return f"event: {event_type}\ndata: {data_str}\n\n"

    def _create_response_created_event(self) -> str:
        """Create response.created event.

        Returns:
            SSE event string
        """
        data = {
            "type": "response.created",
            "response": _build_openresponses_response(
                response_id=self.response_id,
                created_at=self.state.created_at,
                completed_at=None,
                status="in_progress",
                model=self.model,
                output=[],
                input_tokens=0,
                output_tokens=0,
                reasoning=self.state.reasoning,
                usage_none=True,
                store=self.state.store,
                **self._echo_kwargs(),
            ),
        }
        return self._create_sse_event("response.created", data)

    def _create_response_in_progress_event(self) -> str:
        """Create response.in_progress event.

        Returns:
            SSE event string
        """
        data = {
            "type": "response.in_progress",
            "response": _build_openresponses_response(
                response_id=self.response_id,
                created_at=self.state.created_at,
                completed_at=None,
                status="in_progress",
                model=self.model,
                output=[],
                input_tokens=self.state.input_tokens,
                output_tokens=self.state.output_tokens,
                reasoning=self.state.reasoning,
                usage_none=True,
                store=self.state.store,
                **self._echo_kwargs(),
            ),
        }
        return self._create_sse_event("response.in_progress", data)

    def _create_output_item_added_event(
        self,
        item_id: str,
        item_type: str,
        call_id: str | None = None,
        name: str | None = None,
        thought_signature: str | None = None,
        action: dict[str, Any] | None = None,
        item_index: int | None = None,
    ) -> str:
        """Create response.output_item.added event.

        Args:
            item_id: Unique item ID
            item_type: Item type (message, function_call, reasoning, web_search_call)
            call_id: Tool call ID for function_call items
            name: Function name for function_call items
            thought_signature: Thought signature for Google OpenAI compatibility
            action: Action data for web_search_call items
            item_index: Explicit output index (for parallel tool calls).
                Falls back to ``current_item_index`` when not provided.

        Returns:
            SSE event string
        """
        item: dict[str, Any] = {
            "id": item_id,
            "type": item_type,
        }

        if item_type in ("function_call", "tool_search_call", "custom_tool_call") and call_id:
            item["call_id"] = call_id
            item["status"] = "in_progress"
            if item_type == "function_call":
                item["arguments"] = ""
            elif item_type == "tool_search_call":
                item["execution"] = "client"
                item["arguments"] = {}
            elif item_type == "custom_tool_call":
                item["input"] = ""
            if name:
                self._set_call_name(item, name)
            if thought_signature:
                item["thought_signature"] = thought_signature
        elif item_type == "message":
            item["status"] = "in_progress"
            item["role"] = "assistant"
            item["content"] = []
            item["phase"] = "final_answer"
        elif item_type == "reasoning":
            item["status"] = "in_progress"
            item["content"] = []
            item["summary"] = []
            if self.state.include_reasoning_encrypted and self.state.reasoning_encrypted_content:
                self.state.reasoning_encrypted_contents[self.state.current_item_index] = (
                    self.state.reasoning_encrypted_content
                )
        elif item_type == "web_search_call":
            item["status"] = "completed"
            if action:
                item["action"] = action

        output_index = item_index if item_index is not None else self.state.current_item_index
        data = {
            "type": "response.output_item.added",
            "output_index": output_index,
            "item": item,
        }
        return self._create_sse_event("response.output_item.added", data)

    @staticmethod
    def _build_content_part(content_type: str, text: str = "") -> dict[str, Any]:
        """Create a schema-complete content part dict for the given type."""
        part: dict[str, Any] = {"type": content_type}
        if content_type == "output_text":
            part["annotations"] = []
            part["logprobs"] = []
            part["text"] = text
        elif content_type == "refusal":
            part["refusal"] = text
        elif content_type == "reasoning_text":
            part["text"] = text
            part["annotations"] = []
            part["logprobs"] = []
        return part

    def _create_content_part_added_event(
        self,
        item_index: int,
        content_index: int,
        content_type: str,
        item_id: str | None = None,
    ) -> str:
        """Create response.content_part.added event.

        Args:
            item_index: Index of the parent item
            content_index: Index of the content part
            content_type: Type of content (output_text, reasoning_text, refusal)
            item_id: Optional item ID for the parent item

        Returns:
            SSE event string
        """
        part = self._build_content_part(content_type)

        data: dict[str, Any] = {
            "type": "response.content_part.added",
            "output_index": item_index,
            "content_index": content_index,
            "part": part,
        }
        if item_id:
            data["item_id"] = item_id
        return self._create_sse_event("response.content_part.added", data)

    def _create_output_text_delta_event(
        self,
        item_index: int,
        content_index: int,
        text: str,
        item_id: str | None = None,
    ) -> str:
        """Create response.output_text.delta event.

        Args:
            item_index: Index of the parent item
            content_index: Index of the content part
            text: The text delta
            item_id: Optional item ID for the parent item

        Returns:
            SSE event string
        """
        data: dict[str, Any] = {
            "type": "response.output_text.delta",
            "output_index": item_index,
            "content_index": content_index,
            "delta": text,
            "logprobs": [],
        }
        if item_id:
            data["item_id"] = item_id
        return self._create_sse_event("response.output_text.delta", data)

    def _create_reasoning_summary_part_added_event(
        self,
        item_index: int,
        summary_index: int,
        item_id: str | None = None,
    ) -> str:
        """Create response.reasoning_summary_part.added event.

        Args:
            item_index: Index of the parent item
            summary_index: Index of the summary part
            item_id: Optional item ID for the parent item

        Returns:
            SSE event string
        """
        data: dict[str, Any] = {
            "type": "response.reasoning_summary_part.added",
            "output_index": item_index,
            "summary_index": summary_index,
            "part": _create_summary_text_part(""),
        }
        if item_id:
            data["item_id"] = item_id
        return self._create_sse_event("response.reasoning_summary_part.added", data)

    def _create_reasoning_summary_text_delta_event(
        self,
        item_index: int,
        summary_index: int,
        text: str,
        item_id: str | None = None,
    ) -> str:
        """Create response.reasoning_summary_text.delta event.

        Args:
            item_index: Index of the parent item
            summary_index: Index of the summary part
            text: The text delta
            item_id: Optional item ID for the parent item

        Returns:
            SSE event string
        """
        data: dict[str, Any] = {
            "type": "response.reasoning_summary_text.delta",
            "output_index": item_index,
            "summary_index": summary_index,
            "delta": text,
        }
        if item_id:
            data["item_id"] = item_id
        return self._create_sse_event("response.reasoning_summary_text.delta", data)

    def _create_reasoning_summary_text_done_event(
        self,
        item_index: int,
        summary_index: int,
        text: str,
        item_id: str | None = None,
    ) -> str:
        """Create response.reasoning_summary_text.done event.

        Args:
            item_index: Index of the parent item
            summary_index: Index of the summary part
            text: The complete accumulated text
            item_id: Optional item ID for the parent item

        Returns:
            SSE event string
        """
        data: dict[str, Any] = {
            "type": "response.reasoning_summary_text.done",
            "output_index": item_index,
            "summary_index": summary_index,
            "text": text,
        }
        if item_id:
            data["item_id"] = item_id
        return self._create_sse_event("response.reasoning_summary_text.done", data)

    def _create_reasoning_summary_part_done_event(
        self,
        item_index: int,
        summary_index: int,
        text: str,
        item_id: str | None = None,
    ) -> str:
        """Create response.reasoning_summary_part.done event.

        Args:
            item_index: Index of the parent item
            summary_index: Index of the summary part
            text: The complete accumulated text
            item_id: Optional item ID for the parent item

        Returns:
            SSE event string
        """
        data: dict[str, Any] = {
            "type": "response.reasoning_summary_part.done",
            "output_index": item_index,
            "summary_index": summary_index,
            "part": _create_summary_text_part(text),
        }
        if item_id:
            data["item_id"] = item_id
        return self._create_sse_event("response.reasoning_summary_part.done", data)

    def _create_function_call_arguments_delta_event(
        self,
        item_index: int,
        text: str,
        item_id: str | None = None,
    ) -> str:
        """Create response.function_call_arguments.delta event.

        Args:
            item_index: Index of the function call item
            text: The arguments delta

        Returns:
            SSE event string
        """
        data: dict[str, Any] = {
            "type": "response.function_call_arguments.delta",
            "output_index": item_index,
            "delta": text,
        }
        if item_id:
            data["item_id"] = item_id
        return self._create_sse_event("response.function_call_arguments.delta", data)

    def _create_function_call_arguments_done_event(
        self,
        item_index: int,
        arguments: str,
        item_id: str | None = None,
    ) -> str:
        data: dict[str, Any] = {
            "type": "response.function_call_arguments.done",
            "output_index": item_index,
            "arguments": arguments,
        }
        if item_id:
            data["item_id"] = item_id
        return self._create_sse_event("response.function_call_arguments.done", data)

    def _create_refusal_delta_event(
        self,
        item_index: int,
        content_index: int,
        text: str,
        item_id: str | None = None,
    ) -> str:
        """Create response.refusal.delta event.

        Args:
            item_index: Index of the parent item
            content_index: Index of the content part
            text: The refusal delta
            item_id: Optional item ID for the parent item

        Returns:
            SSE event string
        """
        data: dict[str, Any] = {
            "type": "response.refusal.delta",
            "output_index": item_index,
            "content_index": content_index,
            "delta": text,
        }
        if item_id:
            data["item_id"] = item_id
        return self._create_sse_event("response.refusal.delta", data)

    def _create_refusal_done_event(
        self,
        item_index: int,
        content_index: int,
        text: str,
        item_id: str | None = None,
    ) -> str:
        """Create response.refusal.done event.

        Args:
            item_index: Index of the parent item
            content_index: Index of the content part
            text: The complete refusal text
            item_id: Optional item ID for the parent item

        Returns:
            SSE event string
        """
        data: dict[str, Any] = {
            "type": "response.refusal.done",
            "output_index": item_index,
            "content_index": content_index,
            "refusal": text,
        }
        if item_id:
            data["item_id"] = item_id
        return self._create_sse_event("response.refusal.done", data)

    def _create_output_text_done_event(
        self,
        item_index: int,
        content_index: int,
        text: str,
        item_id: str | None = None,
    ) -> str:
        """Create response.output_text.done event.

        Args:
            item_index: Index of the parent item
            content_index: Index of the content part
            text: The complete accumulated text
            item_id: Optional item ID for the parent item

        Returns:
            SSE event string
        """
        data: dict[str, Any] = {
            "type": "response.output_text.done",
            "output_index": item_index,
            "content_index": content_index,
            "text": text,
            "logprobs": [],
        }
        if item_id:
            data["item_id"] = item_id
        return self._create_sse_event("response.output_text.done", data)

    def _create_content_done_event(
        self,
        item_index: int,
        content_index: int,
        text: str,
        item_id: str | None = None,
        content_type: str = "output_text",
        item_type: str = "message",
    ) -> str:
        """Create the appropriate done event based on content type and item type.

        Dispatches to refusal.done, reasoning_summary_text.done, or
        output_text.done.
        """
        if content_type == "refusal":
            return self._create_refusal_done_event(
                item_index=item_index,
                content_index=content_index,
                text=text,
                item_id=item_id,
            )
        if item_type == "reasoning":
            return self._create_reasoning_summary_text_done_event(
                item_index=item_index,
                summary_index=0,
                text=text,
                item_id=item_id,
            )
        return self._create_output_text_done_event(
            item_index=item_index,
            content_index=content_index,
            text=text,
            item_id=item_id,
        )

    def _create_content_part_done_event(
        self,
        item_index: int,
        content_index: int,
        item_id: str | None = None,
        content_type: str = "output_text",
        text: str = "",
    ) -> str:
        """Create response.content_part.done event.

        Args:
            item_index: Index of the parent item
            content_index: Index of the content part
            item_id: Optional item ID for the parent item
            content_type: Type of content (output_text, reasoning_text, refusal)
            text: The complete text for this content part

        Returns:
            SSE event string
        """
        part = self._build_content_part(content_type, text)

        data: dict[str, Any] = {
            "type": "response.content_part.done",
            "output_index": item_index,
            "content_index": content_index,
            "part": part,
        }
        if item_id:
            data["item_id"] = item_id
        return self._create_sse_event("response.content_part.done", data)

    def _create_output_item_done_event(
        self,
        item_id: str,
        item_index: int,
        item_type: str,
        status: str = "completed",
        content_type: str = "output_text",
        text: str = "",
    ) -> str:
        """Create response.output_item.done event.

        Args:
            item_id: Unique item ID
            item_index: Index of the item
            item_type: Type of item
            status: Item status
            content_type: Type of content (for message/reasoning items)
            text: Accumulated text content

        Returns:
            SSE event string
        """
        item: dict[str, Any] = {
            "id": item_id,
            "type": item_type,
            "status": status,
        }

        if item_type == "message":
            item["role"] = "assistant"
            item["content"] = []
            item["phase"] = "final_answer"
            if text:
                if content_type == "output_text":
                    item["content"].append(_create_output_text_part(text))
                else:
                    item["content"].append(
                        {
                            "type": content_type,
                            "text": text,
                        }
                    )
        elif item_type == "reasoning":
            item["content"] = []
            item["summary"] = []
            if text:
                item["summary"].append(_create_summary_text_part(text))
            encrypted = _reasoning_encrypted_content(self.state, item_index)
            if encrypted:
                item["encrypted_content"] = encrypted
        elif item_type in ("custom_tool_call", "function_call", "tool_search_call"):
            self._set_tool_call_args(item, item_index)
            if item_index in self.state.tool_call_ids:
                item["call_id"] = self.state.tool_call_ids[item_index]
            if item_index in self.state.tool_call_names:
                name = self.state.tool_call_names[item_index]
                self._set_call_name(item, name)
            if item_index in self.state.tool_call_thought_signatures:
                item["thought_signature"] = self.state.tool_call_thought_signatures[item_index]
        elif item_type == "web_search_call":
            pending = self.state.pending_items.get(item_index, {})
            action = pending.get("action")
            if action:
                item["action"] = action

        data = {
            "type": "response.output_item.done",
            "output_index": item_index,
            "item": item,
        }
        return self._create_sse_event("response.output_item.done", data)

    def _create_response_completed_event(
        self,
        status: str = "completed",
        input_tokens: int = 0,
        output_tokens: int = 0,
        incomplete_reason: str | None = None,
    ) -> str:
        """Create response.completed event.

        Args:
            status: Response status
            input_tokens: Total input tokens
            output_tokens: Total output tokens
            incomplete_reason: Spec incomplete reason ("max_output_tokens" /
                "content_filter") when status is "incomplete"

        Returns:
            SSE event string
        """
        incomplete_details = (
            {"reason": incomplete_reason}
            if status == "incomplete" and incomplete_reason is not None
            else None
        )
        response_payload = _build_openresponses_response(
            response_id=self.response_id,
            created_at=self.state.created_at,
            completed_at=int(time.time()),
            status=status,
            model=self.model,
            output=self._build_final_output(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            incomplete_details=incomplete_details,
            reasoning=self.state.reasoning,
            cached_tokens=self.state.cached_tokens,
            reasoning_tokens=self.state.reasoning_tokens,
            store=self.state.store,
            **self._echo_kwargs(),
        )
        # Keep the completed snapshot so the streaming processor can persist it
        # (store=true) for follow-up previous_response_id continuations and
        # GET /v1/responses/{id}.
        self.state.final_response_payload = response_payload
        data = {
            "type": "response.completed",
            "response": response_payload,
        }
        return self._create_sse_event("response.completed", data)

    def _create_response_failed_event(
        self,
        error_code: str = "server_error",
        error_message: str = "An error occurred during response generation",
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> str:
        """Create the spec's streaming error events.

        Per the spec, an error incurred while streaming is emitted as an
        ``error`` event (ErrorStreamingEvent, with an ErrorPayload of
        type/code/message/param) and MUST be followed by a ``response.failed``
        event whose response snapshot carries the error in ``response.error``.
        """
        error_payload = {
            "type": error_code,
            "code": error_code,
            "message": error_message,
            "param": None,
        }
        # Error event (spec ErrorStreamingEvent).
        error_event = self._create_sse_event("error", {"type": "error", "error": error_payload})
        # response.failed event with the failed snapshot (spec
        # ResponseFailedStreamingEvent; the error lives in response.error).
        failed_event = self._create_sse_event(
            "response.failed",
            {
                "type": "response.failed",
                "response": _build_openresponses_response(
                    response_id=self.response_id,
                    created_at=self.state.created_at,
                    completed_at=int(time.time()),
                    status="failed",
                    model=self.model,
                    output=self._build_final_output(),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    reasoning=self.state.reasoning,
                    usage_none=True,
                    store=self.state.store,
                    error={"code": error_code, "message": error_message},
                    **self._echo_kwargs(),
                ),
            },
        )
        return error_event + failed_event

    def _create_response_incomplete_event(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        reason: str = "max_output_tokens",
    ) -> str:
        """Create response.incomplete event.

        Emitted when the response is incomplete (e.g., max_tokens reached).

        Args:
            input_tokens: Total input tokens
            output_tokens: Total output tokens
            reason: Spec incomplete reason ("max_output_tokens" or "content_filter")

        Returns:
            SSE event string
        """
        response_payload = _build_openresponses_response(
            response_id=self.response_id,
            created_at=self.state.created_at,
            completed_at=int(time.time()),
            status="incomplete",
            model=self.model,
            output=self._build_final_output(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            incomplete_details={"reason": reason},
            reasoning=self.state.reasoning,
            cached_tokens=self.state.cached_tokens,
            reasoning_tokens=self.state.reasoning_tokens,
            store=self.state.store,
            **self._echo_kwargs(),
        )
        # Keep the incomplete snapshot persistable, mirroring the completed path.
        self.state.final_response_payload = response_payload
        data = {
            "type": "response.incomplete",
            "response": response_payload,
        }
        return self._create_sse_event("response.incomplete", data)

    def _build_final_output(self) -> list[dict[str, Any]]:
        """Build the final response output array from accumulated streaming state."""
        logger.debug(
            f"Streaming._build_final_output called: response_id={self.response_id}, "
            f"pending_items={len(self.state.pending_items)}"
        )
        output: list[dict[str, Any]] = []

        for item_index, item in sorted(self.state.pending_items.items()):
            item_id = item["id"]
            item_type = item["type"]

            if item_type in ("custom_tool_call", "function_call", "tool_search_call"):
                call_item: dict[str, Any] = {
                    "type": item_type,
                    "id": item_id,
                    "call_id": self.state.tool_call_ids.get(item_index, item_id),
                    "status": "completed",
                }
                self._set_call_name(call_item, self.state.tool_call_names.get(item_index, ""))
                self._set_tool_call_args(call_item, item_index)
                if item_index in self.state.tool_call_thought_signatures:
                    call_item["thought_signature"] = self.state.tool_call_thought_signatures[
                        item_index
                    ]
                output.append(call_item)
                continue

            if item_type == "web_search_call":
                ws_item: dict[str, Any] = {
                    "type": "web_search_call",
                    "id": item_id,
                    "status": "completed",
                }
                action = item.get("action")
                if action:
                    ws_item["action"] = action
                output.append(ws_item)
                continue

            content_index = self.state.current_content_index.get(item_index, 0)
            key = (item_index, content_index)
            text = self.state.accumulated_text.get(key, "")
            content_type = self.state.content_types.get(key, "output_text")

            if item_type == "reasoning":
                reasoning_item: dict[str, Any] = {
                    "type": "reasoning",
                    "id": item_id,
                    "status": "completed",
                    "summary": [],
                }
                if text:
                    reasoning_item["summary"].append(_create_summary_text_part(text))
                summary_text = self.state.reasoning_summary_text.get((item_index, 0), "")
                # An upstream may stream both raw reasoning deltas and summary
                # deltas for the same item; when both accumulate to identical
                # text, emit it once rather than as duplicate summary parts.
                if summary_text and summary_text != text:
                    reasoning_item["summary"].append(_create_summary_text_part(summary_text))
                encrypted = _reasoning_encrypted_content(self.state, item_index)
                if encrypted:
                    reasoning_item["encrypted_content"] = encrypted
                output.append(reasoning_item)
                continue

            content: list[dict[str, Any]] = []
            if text:
                if content_type == "refusal":
                    content.append({"type": "refusal", "refusal": text})
                else:
                    content.append(_create_output_text_part(text))

            output.append(
                {
                    "type": "message",
                    "id": item_id,
                    "status": "completed",
                    "role": "assistant",
                    "content": content,
                    "phase": "final_answer",
                }
            )

        # Cache reasoning keyed by call_id for next-turn restoration.
        # Best-effort: a cache write failure must never fail the stream.
        try_cache_reasoning_from_responses_output(
            output, self.response_id, logger_prefix="Streaming"
        )
        return output
