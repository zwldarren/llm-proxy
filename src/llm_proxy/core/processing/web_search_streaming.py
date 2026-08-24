"""Web search processing for streaming responses.

Handles detection and execution of web search tool calls
during streaming, generating appropriate SSE events, and
building continuation requests for multi-turn web search.
"""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import orjson

from llm_proxy.core.adapter import BaseAdapter
from llm_proxy.core.utils import quiet_aclose
from llm_proxy.models import (
    InternalRequest,
    Message,
    ServerToolUseBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)
from llm_proxy.models.conversation import ConversationContext
from llm_proxy.observability.event_context import EventContext
from llm_proxy.observability.logger import get_logger

logger = get_logger(__name__)

MAX_CONTINUATION_DEPTH = 5


def _normalize_tool_name(name: str) -> str:
    """Normalize a tool name for case/separator-insensitive comparison."""
    return name.lower().replace("_", "").replace("-", "")


def _absolute_block_cursor(transformer: Any) -> int | None:
    """Return the transformer's absolute next-block/item cursor, or None when
    the transformer tracks no cursor.

    The original transformer's cursor equals the length of its accumulated
    output; a continuation transformer's accumulated output is relative to its
    own start index, so the cursor is the only absolute reference. Anthropic
    tracks it as ``_current_block_index``, OpenResponses as
    ``state.current_item_index``.
    """
    block_cursor = getattr(transformer, "_current_block_index", None)
    if block_cursor is not None:
        return block_cursor
    src_state = getattr(transformer, "state", None)
    if src_state is not None:
        return src_state.current_item_index
    return None


class WebSearchStreamProcessor:
    """Process web search tool calls in streaming responses."""

    async def process_streaming_web_search(
        self,
        transformer: Any,
        web_search_interceptor: Any,
        web_search_tool_config: Any,
    ) -> tuple[str | None, list[tuple[ServerToolUseBlock, Any]], int]:
        """Process web search tool calls in streaming response.

        Inspects accumulated transformer output for web search tool use blocks,
        executes searches via the interceptor, and generates SSE events.

        Does NOT emit server_tool_use SSE events because the original tool_use
        blocks from the provider stream already serve as the tool call indicator.
        Only emits web_search_tool_result blocks + message_delta with usage.

        Anthropic and OpenResponses transformers both implement
        ``_web_search_result_block`` to emit the result content block
        (``web_search_tool_result`` for Anthropic, ``web_search_call`` for
        OpenResponses). If the transformer lacks this helper, web search
        events cannot be emitted.

        Returns:
            Tuple of (SSE events str or None, list of (ServerToolUseBlock, result) pairs,
            total_search_count)
        """
        accumulated_output = transformer.get_accumulated_output()
        if not accumulated_output:
            return None, [], 0

        web_search_blocks: list[tuple[int, ToolUseBlock | ServerToolUseBlock]] = []
        for idx, block in enumerate(accumulated_output):
            is_ws = isinstance(block, (ToolUseBlock, ServerToolUseBlock))
            if is_ws and _normalize_tool_name(block.name) == "websearch":
                web_search_blocks.append((idx, block))

        if not web_search_blocks:
            return None, [], 0

        if not hasattr(transformer, "_web_search_result_block"):
            logger.warning(
                f"Streaming web search not supported for transformer {type(transformer).__name__}"
            )
            return None, [], 0

        # Web-search result blocks are emitted at explicit content-block/item
        # indices. ``len(accumulated_output)`` is only absolute for the
        # original transformer: continuation transformers start with an empty
        # accumulated output, so their length is relative to their own start
        # index. Start from the transformer's cursor to keep indices
        # collision-free across multi-turn continuations.
        block_cursor = _absolute_block_cursor(transformer)
        current_index = block_cursor if block_cursor is not None else len(accumulated_output)
        total_search_count = 0
        events = []
        search_results: list[tuple[ServerToolUseBlock, Any]] = []
        search_state: dict[str, int] = {"count": 0}

        for idx, tool_use in web_search_blocks:
            server_tool_use = ServerToolUseBlock(
                id=tool_use.id,
                name=tool_use.name,
                input=tool_use.input,
            )

            result = await web_search_interceptor.execute_search(
                server_tool_use, web_search_tool_config, search_state=search_state
            )

            query = tool_use.input.get("query", "") if isinstance(tool_use.input, dict) else ""

            if result.result_block.is_error:
                error_code = orjson.loads(result.result_block.content).get(
                    "error_code", "unavailable"
                )
                events.append(
                    transformer._web_search_result_block(
                        current_index,
                        tool_use.id,
                        [
                            orjson.dumps(
                                {
                                    "type": "web_search_tool_result_error",
                                    "error_code": error_code,
                                }
                            ).decode()
                        ],
                        is_error=True,
                        query=query,
                    )
                )
            else:
                events.append(
                    transformer._web_search_result_block(
                        current_index,
                        tool_use.id,
                        result.result_block.content,
                        is_error=False,
                        query=query,
                    )
                )
                total_search_count += result.web_search_count
            current_index += 1

            if isinstance(accumulated_output[idx], ToolUseBlock):
                accumulated_output[idx] = server_tool_use

            search_results.append((server_tool_use, result))

        if total_search_count > 0 and hasattr(transformer, "_message_delta_with_usage"):
            usage = {
                "output_tokens": 0,
                "server_tool_use": {"web_search_requests": total_search_count},
            }
            events.append(transformer._message_delta_with_usage(usage))

        return (
            "".join(events) if events else None,
            search_results,
            total_search_count,
        )

    @staticmethod
    def build_continuation_request(
        original_request: InternalRequest,
        accumulated_output: list[Any],
        search_results: list[tuple[ServerToolUseBlock, Any]],
        web_search_interceptor: Any,
        *,
        stream: bool = True,
    ) -> InternalRequest:
        """Build a continuation request with web search results in conversation.

        Constructs a new InternalRequest that includes the original conversation
        plus the assistant's tool calls and the tool results from web search.
        This is sent back to the provider so the model can generate a text
        response based on the search results.

        Args:
            original_request: The original InternalRequest
            accumulated_output: Content blocks from the provider's response
            search_results: List of (ServerToolUseBlock, WebSearchExecutionResult) pairs
            web_search_interceptor: The web search interceptor for decoding results
            stream: Whether the continuation request should stream (default
                True for the streaming path; the non-streaming path passes
                False so the follow-up call returns a complete response).

        Returns:
            New InternalRequest ready for continuation
        """
        # Start with original conversation messages
        new_messages = list(original_request.conversation.messages)

        # Build assistant message content from accumulated output.
        # Only include web_search tool calls — non-web_search tool calls
        # are handled by the client and have no matching tool results here.
        # Including them would produce an invalid conversation where the
        # assistant message has tool_calls without corresponding tool messages,
        # which strict providers (e.g. DeepSeek) reject with:
        #   "An assistant message with 'tool_calls' must be followed by
        #    tool messages responding to each 'tool_call_id'."
        assistant_content: list[Any] = []
        for block in accumulated_output:
            if isinstance(block, TextBlock):
                assistant_content.append(TextBlock(text=block.text))
            elif (
                isinstance(block, (ServerToolUseBlock, ToolUseBlock))
                and _normalize_tool_name(block.name) == "websearch"
            ):
                if isinstance(block, ServerToolUseBlock):
                    assistant_content.append(
                        ToolUseBlock(id=block.id, name=block.name, input=block.input)
                    )
                else:
                    assistant_content.append(block)
            # Skip other block types (thinking, non-web_search tool calls, etc.)

        if assistant_content:
            new_messages.append(Message(role="assistant", content=assistant_content))

        # Build tool result messages (one per web_search call). Tool results
        # use the canonical internal ``role="tool"`` representation (what the
        # protocol parsers produce for tool messages); provider serializers
        # map it to their own wire format (Anthropic: user message with a
        # tool_result block, Gemini Interactions: function_result step,
        # OpenAI: tool role message). A ``role="user"`` message carrying a
        # ToolResultBlock would be dropped by serializers that only look for
        # tool results on ``role="tool"`` messages (e.g. the Gemini
        # Interactions converter), silently losing the search results.
        for server_tool_use, exec_result in search_results:
            if exec_result.result_block.is_error:
                decoded = exec_result.result_block.content
                new_messages.append(
                    Message(
                        role="tool",
                        content=[
                            ToolResultBlock(
                                tool_use_id=server_tool_use.id,
                                content=decoded,
                                is_error=True,
                            )
                        ],
                    )
                )
            else:
                decoded = web_search_interceptor.decode_search_results(exec_result.result_block)
                new_messages.append(
                    Message(
                        role="tool",
                        content=[
                            ToolResultBlock(
                                tool_use_id=server_tool_use.id,
                                content=decoded,
                                is_error=False,
                            )
                        ],
                    )
                )

        new_conversation = ConversationContext(
            system_messages=list(original_request.conversation.system_messages),
            messages=new_messages,
        )

        return InternalRequest(
            model=original_request.model,
            conversation=new_conversation,
            tools=original_request.tools,
            tool_choice=original_request.tool_choice,
            params=original_request.params,
            stream=stream,
            stream_options=original_request.stream_options,
            metadata=original_request.metadata,
            extra=original_request.extra,
            user_facing_model=original_request.user_facing_model,
        )

    @staticmethod
    def needs_continuation(
        accumulated_output: list[Any],
    ) -> bool:
        """Check whether the model needs continuation after web search.

        Returns True if there are web_search tool use blocks without
        any trailing text blocks, indicating the model ended its turn
        after issuing search calls and is waiting for results.

        Args:
            accumulated_output: Content blocks from the provider's response

        Returns:
            True if continuation is needed
        """

        # Find the last web_search block index
        last_ws_idx = -1
        for idx, block in enumerate(accumulated_output):
            is_ws = isinstance(block, (ToolUseBlock, ServerToolUseBlock))
            if is_ws and _normalize_tool_name(block.name) == "websearch":
                last_ws_idx = idx

        if last_ws_idx < 0:
            return False

        # Check if any text block appears after the last web_search block
        for block in accumulated_output[last_ws_idx + 1 :]:
            if isinstance(block, TextBlock) and block.text.strip():
                return False

        return True

    async def generate_continuation(
        self,
        state: ContinuationState,
        *,
        web_search_interceptor: Any,
        web_search_tool_config: Any,
        current_adapter: BaseAdapter,
        tracing_registry: Any,
        event_context: EventContext | None,
        cancel_token: asyncio.Event | None,
        proxy_web_search_active: bool = False,
    ) -> AsyncGenerator[str]:
        """Yield SSE chunks from web search processing and continuations.

        The continuation loop: while the model ended its turn waiting for
        search results, inject the results, re-call the provider, and pump
        the follow-up stream through a fresh continuation transformer.

        Mutates *state* in-place so the caller can use the updated
        transformer, depth, and stream_request for finalize/usage-merge.
        """
        (
            ws_events,
            ws_results,
            total_search_count,
        ) = await self.process_streaming_web_search(
            state.transformer, web_search_interceptor, web_search_tool_config
        )
        if ws_events:
            await tracing_registry.on_stream_chunk(state.stream_request, ws_events, event_context)
            yield ws_events

        # Write web search count to event_context for billing
        if total_search_count > 0 and event_context is not None:
            event_context.web_search_requests = total_search_count

        state.depth = 0

        while ws_results and state.depth < MAX_CONTINUATION_DEPTH:
            accumulated = state.transformer.get_accumulated_output()
            if not self.needs_continuation(accumulated):
                break

            continuation_req = self.build_continuation_request(
                original_request=state.stream_request,
                accumulated_output=accumulated,
                search_results=ws_results,
                web_search_interceptor=web_search_interceptor,
            )
            state.stream_request = continuation_req

            continuation_stream = await current_adapter.stream_chat_completion(
                continuation_req, cancel_token=cancel_token
            )

            # Absolute start index for the next continuation transformer.
            # ``len(accumulated)`` is only absolute for the first
            # continuation: each continuation transformer starts with an empty
            # accumulated output, so its length is relative to its own start
            # index. The transformer's cursor tracks the absolute position
            # instead. Anthropic's ``_current_block_index`` is the next block
            # index — web-search result blocks are emitted at explicit indices
            # and do not advance it — while OpenResponses' ``current_item_index``
            # is left pointing at the last emitted web-search result item.
            block_cursor = _absolute_block_cursor(state.transformer)
            if block_cursor is not None and hasattr(state.transformer, "_current_block_index"):
                next_index = block_cursor + len(ws_results)
            elif block_cursor is not None:
                next_index = block_cursor + 1
            else:
                next_index = len(accumulated) + len(ws_results)
            cont_cls = type(state.transformer)
            if not hasattr(cont_cls, "continuation"):
                raise TypeError(
                    f"Transformer {cont_cls.__name__} does not implement 'continuation'"
                )
            cont_kwargs = dict(
                # Echo the client-requested alias (see InternalRequest.echo_model)
                # so continuation chunks agree with the main stream; the upstream
                # call itself still uses the resolved provider model name.
                model=state.stream_request.echo_model,
                request_id=state.transformer.response_id,
                start_index=next_index,
            )
            src_state = getattr(state.transformer, "state", None)
            if src_state is not None:
                cont_kwargs["intercept_web_search"] = proxy_web_search_active
            cont_transformer = cont_cls.continuation(**cont_kwargs)

            if src_state is not None:
                for idx, item in src_state.pending_items.items():
                    if item.get("type") == "web_search_call":
                        cont_transformer.state.pending_items[idx] = dict(item)
                        cont_transformer.state.closed_items.add(idx)

            try:
                async for chunk in continuation_stream:
                    if not isinstance(chunk, (str, dict)):
                        continue
                    if cancel_token and cancel_token.is_set():
                        break
                    transformed = cont_transformer.transform(chunk)
                    if transformed:
                        await tracing_registry.on_stream_chunk(
                            state.stream_request, transformed, event_context
                        )
                        yield transformed

                (
                    ws_events,
                    ws_results,
                    ws_search_count,
                ) = await self.process_streaming_web_search(
                    cont_transformer, web_search_interceptor, web_search_tool_config
                )
                if ws_events:
                    await tracing_registry.on_stream_chunk(
                        state.stream_request, ws_events, event_context
                    )
                    yield ws_events

                # Accumulate web search count for billing
                if ws_search_count > 0 and event_context is not None:
                    current_count = event_context.web_search_requests or 0
                    event_context.web_search_requests = current_count + ws_search_count

                state.transformer = cont_transformer
                state.depth += 1
            finally:
                with suppress(Exception, asyncio.CancelledError):
                    await asyncio.shield(quiet_aclose(continuation_stream))


@dataclass
class ContinuationState:
    """Mutable state shared between the streaming processor and the
    web-search continuation loop.

    The continuation phase may replace the transformer and the stream
    request; this object carries those mutations back to the caller so the
    finalize / usage-merge phases use the correct objects.
    """

    transformer: Any
    stream_request: Any
    depth: int = 0


def sum_usage_dicts(target: dict[str, Any], source: dict[str, Any]) -> None:
    """Sum token keys from source into target, in place.

    The original turn and the web-search continuation are two INDEPENDENT
    upstream calls, each billed separately by the provider, so the correct
    totals are sums — max() would undercount output/cache tokens (and input
    tokens too: the continuation's re-sent conversation is real billed
    input). Anthropic-style keys and OpenAI-style keys are summed under
    their own vocabulary.
    """
    for key in (
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "reasoning_tokens",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    ):
        if key in target or key in source:
            target[key] = target.get(key, 0) + source.get(key, 0)


def sum_usage(target: Usage, source: Usage) -> None:
    """Sum source into target in place (independent billed calls).

    Counterpart of ``sum_usage_dicts`` for the canonical ``Usage`` record;
    None-valued fields are left untouched (unknown on one side of the sum).
    """
    target.input_tokens += source.input_tokens
    target.output_tokens += source.output_tokens
    for field in (
        "total_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "reasoning_tokens",
        "web_search_requests",
    ):
        target_value = getattr(target, field)
        source_value = getattr(source, field)
        if target_value is not None and source_value is not None:
            setattr(target, field, target_value + source_value)


def merge_continuation_usage(original: Any, continuation: Any) -> None:
    """Merge pending stop_reason and usage from original into continuation transformer.

    Transformers from different protocols expose different internal state
    (e.g. ``OpenAIStreamingTransformer`` has no ``_pending_stop_reason`` or
    ``_has_pending_usage``), so every access here is guarded to stay
    protocol-agnostic and avoid ``AttributeError`` during web search
    continuation merges.
    """
    orig_stop = getattr(original, "_pending_stop_reason", None)
    cont_stop = getattr(continuation, "_pending_stop_reason", None)
    if orig_stop and not cont_stop:
        continuation._pending_stop_reason = orig_stop

    orig_usage = getattr(original, "_pending_usage", None)
    if orig_usage:
        cont_usage = getattr(continuation, "_pending_usage", None)
        if cont_usage:
            sum_usage_dicts(cont_usage, orig_usage)
        else:
            continuation._pending_usage = orig_usage
        if hasattr(continuation, "_has_pending_usage"):
            continuation._has_pending_usage = True
