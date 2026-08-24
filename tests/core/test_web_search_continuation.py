"""Standalone tests for the web-search continuation loop.

The continuation subsystem lives behind WebSearchStreamProcessor's interface;
these tests cross exactly that seam with minimal fakes — no StreamingProcessor,
no protocol endpoint, no SSE plumbing.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_proxy.core.processing.web_search_streaming import (
    ContinuationState,
    WebSearchStreamProcessor,
)
from llm_proxy.models import (
    ConversationContext,
    InternalRequest,
    InternalResponse,
    Message,
    ServerToolUseBlock,
    TextBlock,
    ToolUseBlock,
    Usage,
)


class _FakeTransformer:
    """Minimal transformer satisfying the continuation seam's contract.

    Tracks an Anthropic-style ``_current_block_index`` cursor so the
    continuation loop's absolute-index math is exercised; ``created`` records
    every instance (original + continuations) for index assertions.
    """

    created: list = []

    def __init__(self, request_id: str = "resp-1", start_index: int = 0, **_kwargs: Any):
        self.response_id = request_id
        self.start_index = start_index
        self._output: list[Any] = []
        self._current_block_index = start_index
        self.result_indices: list[int] = []
        _FakeTransformer.created.append(self)

    @classmethod
    def continuation(cls, **kwargs: Any) -> _FakeTransformer:
        return cls(**kwargs)

    def get_accumulated_output(self) -> list[Any]:
        return self._output

    def transform(self, chunk: Any) -> str | None:
        # A content chunk: append text and emit one frame.
        delta = chunk["choices"][0]["delta"]
        if delta.get("content"):
            self._output.append(TextBlock(text=delta["content"]))
            self._current_block_index += 1
        for tc in delta.get("tool_calls") or []:
            self._output.append(ToolUseBlock(id=tc["id"], name=tc["function"]["name"], input={}))
            self._current_block_index += 1
        return "data: {}\n\n"

    def _web_search_result_block(self, index: int, *_args: Any, **_kwargs: Any) -> str:
        self.result_indices.append(index)
        return "event: content_block_start\ndata: {}\n\n"


def _fake_interceptor() -> MagicMock:
    result_block = MagicMock()
    result_block.is_error = False
    result_block.content = "search results text"

    exec_result = MagicMock()
    exec_result.result_block = result_block
    exec_result.web_search_count = 1

    interceptor = MagicMock()
    interceptor.execute_search = AsyncMock(return_value=exec_result)
    interceptor.decode_search_results = MagicMock(return_value="decoded results")
    return interceptor


def _fake_adapter() -> MagicMock:
    async def _stream(_request: Any, cancel_token: Any = None):
        yield {"choices": [{"delta": {"content": "final answer"}}]}

    adapter = MagicMock()
    adapter.stream_chat_completion = AsyncMock(
        side_effect=lambda req, cancel_token=None: _stream(req)
    )
    return adapter


def _request() -> InternalRequest:
    return InternalRequest(
        model="glm-5",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="hi")])]
        ),
        stream=True,
    )


@pytest.mark.asyncio
async def test_generate_continuation_runs_one_round_and_carries_state() -> None:
    processor = WebSearchStreamProcessor()
    transformer = _FakeTransformer()
    transformer._output.append(ToolUseBlock(id="tu_1", name="web_search", input={"query": "news"}))
    transformer._current_block_index = 1
    state = ContinuationState(transformer=transformer, stream_request=_request())

    tracing = MagicMock()
    tracing.on_stream_chunk = AsyncMock()
    adapter = _fake_adapter()

    chunks = [
        chunk
        async for chunk in processor.generate_continuation(
            state,
            web_search_interceptor=_fake_interceptor(),
            web_search_tool_config=None,
            current_adapter=adapter,
            tracing_registry=tracing,
            event_context=None,
            cancel_token=None,
        )
    ]

    # One web-search event frame + one continuation content frame.
    assert len(chunks) == 2
    # State carried back: one continuation round, swapped transformer and request.
    assert state.depth == 1
    assert state.transformer is not transformer
    assert state.stream_request is not None
    # The continuation request carries the assistant tool call + tool result.
    adapter_request = adapter.stream_chat_completion.call_args.args[0]
    roles = [m.role for m in adapter_request.conversation.messages]
    assert roles == ["user", "assistant", "tool"]


@pytest.mark.asyncio
async def test_generate_continuation_no_search_blocks_is_noop() -> None:
    processor = WebSearchStreamProcessor()
    transformer = _FakeTransformer()
    transformer._output.append(TextBlock(text="plain answer, no tools"))
    state = ContinuationState(transformer=transformer, stream_request=_request())

    tracing = MagicMock()
    tracing.on_stream_chunk = AsyncMock()
    adapter = _fake_adapter()

    chunks = [
        chunk
        async for chunk in processor.generate_continuation(
            state,
            web_search_interceptor=_fake_interceptor(),
            web_search_tool_config=None,
            current_adapter=adapter,
            tracing_registry=tracing,
            event_context=None,
            cancel_token=None,
        )
    ]

    assert chunks == []
    assert state.depth == 0
    assert state.transformer is transformer
    adapter.stream_chat_completion.assert_not_called()


@pytest.mark.asyncio
async def test_generate_continuation_multi_turn_keeps_absolute_indices() -> None:
    """Multi-turn web search must keep content-block/item indices absolute.

    Each continuation transformer starts with an empty accumulated output, so
    ``len(accumulated)`` is relative to its own start index. The start index
    of each new continuation — and the indices of the web-search result
    blocks — must be derived from the transformer's absolute cursor,
    otherwise later turns collide with earlier turns' block indices in the
    SSE stream.
    """
    _FakeTransformer.created.clear()
    processor = WebSearchStreamProcessor()
    transformer = _FakeTransformer()
    transformer._output.append(ToolUseBlock(id="tu_1", name="web_search", input={"query": "news"}))
    transformer._current_block_index = 1
    state = ContinuationState(transformer=transformer, stream_request=_request())

    tracing = MagicMock()
    tracing.on_stream_chunk = AsyncMock()

    streams = [
        # First continuation: text, then another web_search call.
        [
            {"choices": [{"delta": {"content": "here is what I found"}}]},
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "id": "tu_2",
                                    "function": {"name": "web_search", "arguments": ""},
                                }
                            ]
                        }
                    }
                ]
            },
        ],
        # Second continuation: final text only.
        [{"choices": [{"delta": {"content": "final answer"}}]}],
    ]

    async def _stream(_request: Any, cancel_token: Any = None):
        for chunk in streams.pop(0):
            yield chunk

    adapter = MagicMock()
    adapter.stream_chat_completion = AsyncMock(side_effect=_stream)

    chunks = [
        chunk
        async for chunk in processor.generate_continuation(
            state,
            web_search_interceptor=_fake_interceptor(),
            web_search_tool_config=None,
            current_adapter=adapter,
            tracing_registry=tracing,
            event_context=None,
            cancel_token=None,
        )
    ]

    # 1 ws-result frame (turn 1) + 2 frames (cont 1) + 1 ws-result frame
    # (cont 1) + 1 frame (cont 2).
    assert len(chunks) == 5
    assert state.depth == 2
    created = _FakeTransformer.created
    # Absolute start indices: turn 1 output (1 block) + 1 ws result = 2;
    # turn 2 output (2 blocks) + 1 ws result = 5.
    assert [t.start_index for t in created] == [0, 2, 5]
    # Turn 2's web-search result block lands after turn 2's own blocks
    # (absolute index 4), not at a relative index that collides with turn 1's
    # result block (index 1).
    assert created[1].result_indices == [4]
    assert created[2].result_indices == []


# ── Non-streaming continuation (RetryExecutor._continue_web_search) ──────


def _make_retry_executor() -> Any:
    """Build a RetryExecutor with a mocked parameter-override service."""
    from llm_proxy.core.processing.stages.parameter_override import (
        ParameterOverrideService,
    )
    from llm_proxy.core.processing.stages.request_execution import RetryExecutor

    service = ParameterOverrideService(serializer=MagicMock())
    return RetryExecutor(param_override_service=service)


def _make_state() -> Any:
    from llm_proxy.observability.event_context import EventContext

    state = MagicMock()
    state.trace_id = "trace-1"
    state.event_context = EventContext(
        request_id="req-1", trace_id="trace-1", model="gemini-3.1-flash-lite"
    )
    return state


def _make_context(interceptor: Any) -> Any:
    context = MagicMock()
    context.web_search_interceptor = interceptor
    context.web_search_tool_config = None
    return context


def _make_interceptor() -> Any:
    """Interceptor whose first injection returns one search pair and whose
    follow-up injection returns none."""
    result_block = MagicMock()
    result_block.is_error = False
    result_block.content = "search results text"

    exec_result = MagicMock()
    exec_result.result_block = result_block
    exec_result.web_search_count = 1

    interceptor = MagicMock()
    interceptor.decode_search_results = MagicMock(return_value="decoded results")

    async def _inject(response: Any, **kwargs: Any) -> Any:
        if response.output and any(isinstance(b, ServerToolUseBlock) for b in response.output):
            # First round: the model called web_search; return the pair.
            return response, [(response.output[0], exec_result)]
        # Follow-up rounds: no more web_search calls.
        return response, []

    interceptor.inject_results_into_response = AsyncMock(side_effect=_inject)
    return interceptor


def _make_adapter(final_text: str = "The capital of France is Paris.") -> Any:
    async def _chat(request: Any) -> Any:
        return InternalResponse(
            id="resp_cont",
            model=request.model,
            output=[TextBlock(text=final_text)],
            usage=Usage(input_tokens=100, output_tokens=10, total_tokens=110),
        )

    adapter = MagicMock()
    adapter.chat_completion = AsyncMock(side_effect=_chat)
    return adapter


@pytest.mark.asyncio
async def test_continue_web_search_replaces_raw_blocks_with_final_answer() -> None:
    """Non-streaming web search must re-call the provider with the injected
    results and return the model's final answer instead of the raw
    server_tool_use/web_search_result blocks."""

    executor = _make_retry_executor()
    interceptor = _make_interceptor()
    adapter = _make_adapter()
    state = _make_state()
    context = _make_context(interceptor)

    original = InternalRequest(
        model="gemini-3.1-flash-lite",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="Search for X")])]
        ),
    )
    response = InternalResponse(
        id="resp_1",
        model="gemini-3.1-flash-lite",
        output=[ServerToolUseBlock(id="call_1", name="web_search", input={"query": "X"})],
        usage=Usage(input_tokens=50, output_tokens=5, total_tokens=55),
    )

    # The caller (RetryExecutor.execute) records the first round's count
    # before handing off to the continuation loop.
    state.event_context.web_search_requests = 1

    result = await executor._continue_web_search(
        response=response,
        search_results=[(response.output[0], MagicMock())],
        original_request=original,
        adapter=adapter,
        context=context,
        state=state,
    )

    # The final response is the continuation's answer, not the raw blocks.
    assert result.output == [TextBlock(text="The capital of France is Paris.")]
    # The continuation request carried the assistant tool call + tool result.
    cont_request = adapter.chat_completion.call_args.args[0]
    roles = [m.role for m in cont_request.conversation.messages]
    assert roles == ["user", "assistant", "tool"]
    # Usage from both upstream calls is summed.
    assert result.usage is not None
    assert result.usage.input_tokens == 150
    assert result.usage.output_tokens == 15
    assert result.usage.total_tokens == 165
    # Web search count is reported on the final response.
    assert result.provider_info["server_tool_use"]["web_search_requests"] == 1
    assert state.event_context.web_search_requests == 1


@pytest.mark.asyncio
async def test_continue_web_search_noop_without_search_results() -> None:
    """No continuation when the model produced a plain answer."""

    executor = _make_retry_executor()
    interceptor = _make_interceptor()
    adapter = _make_adapter()
    state = _make_state()
    context = _make_context(interceptor)

    original = InternalRequest(
        model="gemini-3.1-flash-lite",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="hi")])]
        ),
    )
    response = InternalResponse(
        id="resp_1",
        model="gemini-3.1-flash-lite",
        output=[TextBlock(text="plain answer")],
        usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
    )

    result = await executor._continue_web_search(
        response=response,
        search_results=[],
        original_request=original,
        adapter=adapter,
        context=context,
        state=state,
    )

    assert result is response
    adapter.chat_completion.assert_not_called()
