"""Tests for PreviousResponseResolutionStage pipeline stage."""

from unittest.mock import MagicMock

import pytest

from llm_proxy.core.processing.base import RequestContext, ServiceDependencies
from llm_proxy.core.processing.stages.base import PipelineState
from llm_proxy.core.processing.stages.previous_response import (
    PreviousResponseResolutionStage,
)


class FakeResponseStore:
    """Fake response store for testing that returns a configurable payload."""

    def __init__(self, payload: dict | None = None, should_fail: bool = False):
        self._payload = payload
        self._should_fail = should_fail
        self.retrieve_calls: list[tuple[str, str]] = []

    async def retrieve(self, api_key_name: str, response_id: str) -> dict | None:
        self.retrieve_calls.append((api_key_name, response_id))
        if self._should_fail:
            raise RuntimeError("simulated store failure")
        return self._payload


@pytest.mark.asyncio
async def test_stage_no_op_when_no_response_store():
    """Stage is a no-op when context has no response_store."""
    stage = PreviousResponseResolutionStage()
    state = PipelineState(
        raw_data={},
        unified_request=None,
        req=None,
        strategy=None,
        trace_id="t",
        event_context=MagicMock(),
    )
    ctx = RequestContext(
        orchestrator=MagicMock(),
        services=MagicMock(),
        response_store=None,
    )
    await stage.process(state, ctx)
    # Should not raise, should not modify state


@pytest.mark.asyncio
async def test_stage_no_op_when_no_previous_response_id():
    """Stage is a no-op when unified_request has no previous_response_id in extra."""
    stage = PreviousResponseResolutionStage()
    fake_req = MagicMock()
    fake_req.extra = {}
    state = PipelineState(
        raw_data={},
        unified_request=fake_req,
        req=None,
        strategy=None,
        trace_id="t",
        event_context=MagicMock(),
    )
    ctx = RequestContext(
        orchestrator=MagicMock(),
        services=MagicMock(),
        response_store=FakeResponseStore(payload={"output": []}),
    )
    await stage.process(state, ctx)
    # No assertion needed; just verify no error raised


@pytest.mark.asyncio
async def test_stage_no_op_when_unified_request_has_no_extra():
    """Stage is a no-op when unified_request has no extra attribute."""
    stage = PreviousResponseResolutionStage()
    fake_req = MagicMock(spec=[])  # no extra attr
    state = PipelineState(
        raw_data={},
        unified_request=fake_req,
        req=None,
        strategy=None,
        trace_id="t",
        event_context=MagicMock(),
    )
    ctx = RequestContext(
        orchestrator=MagicMock(),
        services=MagicMock(),
        response_store=FakeResponseStore(payload={"output": []}),
    )
    await stage.process(state, ctx)


@pytest.mark.asyncio
async def test_stage_skips_when_no_api_key():
    """Stage skips lookup when req has no api_key_name identity."""
    from llm_proxy.core.identity import RequestIdentity

    stage = PreviousResponseResolutionStage()
    fake_req = MagicMock()
    fake_req.extra = {"previous_response_id": "prev123"}

    # Create a req with state that has identity but no api_key_name
    fake_state_req = MagicMock()
    fake_state_req.state = MagicMock()
    fake_state_req.state.identity = RequestIdentity(api_key_name=None)

    state = PipelineState(
        raw_data={},
        unified_request=fake_req,
        req=fake_state_req,
        strategy=None,
        trace_id="t",
        event_context=MagicMock(),
    )
    response_store = FakeResponseStore(payload={"output": []})
    ctx = RequestContext(
        orchestrator=MagicMock(),
        services=MagicMock(),
        response_store=response_store,
    )
    await stage.process(state, ctx)
    # Should not call retrieve since api_key_name is None
    assert len(response_store.retrieve_calls) == 0


@pytest.mark.asyncio
async def test_stage_prepends_previous_output():
    """Stage prepends previous response items to conversation when previous_response_id is valid."""
    from llm_proxy.core.identity import RequestIdentity
    from llm_proxy.models import ConversationContext, InternalRequest, Message, TextBlock

    stage = PreviousResponseResolutionStage()
    prev_response = {
        "instructions": "Be helpful",
        "input": "What is 2+2?",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "4"}],
                "status": "completed",
                "id": "msg_prev1",
            }
        ],
    }
    req = InternalRequest(
        model="gpt-4",
        conversation=ConversationContext(system_messages=[], messages=[]),
        extra={"previous_response_id": "resp_prev"},
    )
    req.conversation.messages.append(
        Message(role="user", content=[TextBlock(text="Now multiply by 3")])
    )

    # Create a req with state that has identity with api_key_name
    fake_state_req = MagicMock()
    fake_state_req.state = MagicMock()
    fake_state_req.state.identity = RequestIdentity(api_key_name="test-key")

    state = PipelineState(
        raw_data={},
        unified_request=req,
        req=fake_state_req,
        strategy=None,
        trace_id="t",
        event_context=MagicMock(),
    )
    response_store = FakeResponseStore(payload=prev_response)
    ctx = RequestContext(
        orchestrator=MagicMock(),
        services=MagicMock(),
        response_store=response_store,
    )
    await stage.process(state, ctx)

    # After prepend: prev user, prev assistant, current user = 3 messages
    assert len(req.conversation.messages) == 3
    assert req.conversation.messages[0].role == "user"
    assert req.conversation.messages[1].role == "assistant"
    assert req.conversation.messages[2].role == "user"
    # Should have instructions from previous response
    assert len(req.conversation.system_messages) == 1
    assert req.conversation.system_messages[0].text_content == "Be helpful"
    # Verify retrieve was called
    assert len(response_store.retrieve_calls) == 1
    assert response_store.retrieve_calls[0] == ("test-key", "resp_prev")


@pytest.mark.asyncio
async def test_stage_marks_previous_response_materialized():
    """Materializing a proxy-stored response disables native request passthrough.

    The raw protocol body still carries the proxy-local previous_response_id
    and none of the materialized items; BaseAdapter.allows_native_request
    reads this flag to fall back to the rebuilt body.
    """
    from llm_proxy.core.identity import RequestIdentity
    from llm_proxy.models import ConversationContext, InternalRequest, Message, TextBlock

    stage = PreviousResponseResolutionStage()
    prev_response = {
        "input": "hi",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hello"}],
                "status": "completed",
                "id": "msg_prev1",
            }
        ],
    }
    req = InternalRequest(
        model="gpt-4",
        conversation=ConversationContext(system_messages=[], messages=[]),
        extra={"previous_response_id": "resp_prev"},
    )
    req.conversation.messages.append(Message(role="user", content=[TextBlock(text="again")]))

    fake_state_req = MagicMock()
    fake_state_req.state = MagicMock()
    fake_state_req.state.identity = RequestIdentity(api_key_name="test-key")

    state = PipelineState(
        raw_data={},
        unified_request=req,
        req=fake_state_req,
        strategy=None,
        trace_id="t",
        event_context=MagicMock(),
    )
    ctx = RequestContext(
        orchestrator=MagicMock(),
        services=MagicMock(),
        response_store=FakeResponseStore(payload=prev_response),
    )
    await stage.process(state, ctx)

    assert req.previous_response_materialized is True
    # The id is consumed locally; the rebuilt body must not carry it.
    assert "previous_response_id" not in req.extra


def _materialization_harness(adapter):
    """Run the stage against a stored response with the given adapter attached."""
    from llm_proxy.core.identity import RequestIdentity
    from llm_proxy.models import ConversationContext, InternalRequest, Message, TextBlock

    prev_response = {
        "input": "hi",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hello"}],
                "status": "completed",
                "id": "msg_prev1",
            }
        ],
    }
    req = InternalRequest(
        model="deepseek-v4-pro",
        conversation=ConversationContext(system_messages=[], messages=[]),
        extra={"previous_response_id": "resp_prev"},
    )
    req.conversation.messages.append(Message(role="user", content=[TextBlock(text="again")]))

    fake_state_req = MagicMock()
    fake_state_req.state = MagicMock()
    fake_state_req.state.identity = RequestIdentity(api_key_name="test-key")

    state = PipelineState(
        raw_data={},
        unified_request=req,
        req=fake_state_req,
        strategy=None,
        trace_id="t",
        event_context=MagicMock(),
    )
    state.adapter = adapter
    ctx = RequestContext(
        orchestrator=MagicMock(),
        services=MagicMock(),
        response_store=FakeResponseStore(payload=prev_response),
    )
    return state, ctx, req


@pytest.mark.asyncio
async def test_stage_disables_native_streaming_for_chat_upstream():
    """Materialization with a non-native-Responses upstream (e.g. DeepSeek)
    disables native handling on BOTH sides: the rebuilt body is Chat
    Completions-shaped, so a native Responses stream cannot consume it — the
    whole request must fall back to translation."""
    chat_adapter = MagicMock()
    chat_adapter._target_endpoint.return_value = "chat_completions"

    state, ctx, req = _materialization_harness(chat_adapter)
    await PreviousResponseResolutionStage().process(state, ctx)

    assert req.previous_response_materialized is True
    assert req.native_request_disabled is True


@pytest.mark.asyncio
async def test_stage_keeps_native_streaming_for_responses_upstream():
    """Materialization with a native Responses upstream (OpenAI) rebuilds a
    Responses-shaped body, so the stream side may stay native."""
    native_adapter = MagicMock()
    native_adapter._target_endpoint.return_value = "responses"

    state, ctx, req = _materialization_harness(native_adapter)
    await PreviousResponseResolutionStage().process(state, ctx)

    assert req.previous_response_materialized is True
    assert req.native_request_disabled is False


@pytest.mark.asyncio
async def test_stage_prepends_codex_item_types_from_prev_input():
    """The previous-response stage must reconstruct Codex item types
    (local_shell_call, custom_tool_call/output, compaction, agent_message) from
    a stored previous input, and skip hosted tools with no Chat Completions
    equivalent."""
    from llm_proxy.core.identity import RequestIdentity
    from llm_proxy.models import ConversationContext, InternalRequest, Message, TextBlock
    from llm_proxy.models.content_blocks import ThinkingBlock, ToolResultBlock, ToolUseBlock

    stage = PreviousResponseResolutionStage()
    prev_response = {
        "input": [
            {"type": "message", "role": "user", "content": "list files"},
            {
                "type": "local_shell_call",
                "call_id": "call_ls",
                "status": "completed",
                "action": {"type": "exec", "command": ["ls"]},
            },
            {"type": "function_call_output", "call_id": "call_ls", "output": "a.txt"},
            {
                "type": "web_search_call",
                "status": "completed",
                "action": {"type": "search", "query": "q"},
            },
            {"type": "compaction", "encrypted_content": "ENC"},
            {
                "type": "agent_message",
                "author": "a",
                "recipient": "b",
                "content": [{"type": "input_text", "text": "hello"}],
            },
        ],
        "output": [],
    }
    req = InternalRequest(
        model="gpt-4",
        conversation=ConversationContext(system_messages=[], messages=[]),
        extra={"previous_response_id": "resp_prev"},
    )
    req.conversation.messages.append(Message(role="user", content=[TextBlock(text="next")]))

    fake_state_req = MagicMock()
    fake_state_req.state = MagicMock()
    fake_state_req.state.identity = RequestIdentity(api_key_name="test-key")

    state = PipelineState(
        raw_data={},
        unified_request=req,
        req=fake_state_req,
        strategy=None,
        trace_id="t",
        event_context=MagicMock(),
    )
    ctx = RequestContext(
        orchestrator=MagicMock(),
        services=MagicMock(),
        response_store=FakeResponseStore(payload=prev_response),
    )
    await stage.process(state, ctx)

    roles = [m.role for m in req.conversation.messages]
    # user, assistant(local_shell), tool(result), assistant(web_search +
    # compaction), tool(web_search placeholder), user(agent_message),
    # user(current). web_search_call becomes the same placeholder the
    # request-parse path emits (continuations must match first-turn parsing).
    assert roles == ["user", "assistant", "tool", "assistant", "tool", "user", "user"]
    assert isinstance(req.conversation.messages[1].content[0], ToolUseBlock)
    assert req.conversation.messages[1].content[0].name == "local_shell"
    assert isinstance(req.conversation.messages[2].content[0], ToolResultBlock)
    merged = req.conversation.messages[3].content
    assert any(isinstance(b, ThinkingBlock) and b.encrypted_content == "ENC" for b in merged)


@pytest.mark.asyncio
async def test_stage_raises_when_retrieve_returns_none():
    """Stage raises previous_response_not_found when the response is missing.

    Spec: when the referenced response is not available, the server MUST fail
    the turn with an error whose code is previous_response_not_found.
    """
    from llm_proxy.core.exceptions import NotFoundError
    from llm_proxy.core.identity import RequestIdentity

    stage = PreviousResponseResolutionStage()
    fake_req = MagicMock()
    fake_req.extra = {"previous_response_id": "prev_nonexistent"}
    fake_req.conversation = MagicMock()
    fake_req.conversation.messages = []
    fake_req.conversation.system_messages = []

    fake_state_req = MagicMock()
    fake_state_req.state = MagicMock()
    fake_state_req.state.identity = RequestIdentity(api_key_name="test-key")

    state = PipelineState(
        raw_data={},
        unified_request=fake_req,
        req=fake_state_req,
        strategy=None,
        trace_id="t",
        event_context=MagicMock(),
    )
    response_store = FakeResponseStore(payload=None)  # returns None
    ctx = RequestContext(
        orchestrator=MagicMock(),
        services=MagicMock(),
        response_store=response_store,
    )
    with pytest.raises(NotFoundError) as exc_info:
        await stage.process(state, ctx)
    assert exc_info.value.code == "previous_response_not_found"
    # OpenAI parity: previous_response_not_found is an HTTP 400 error.
    assert exc_info.value.status_code == 400
    assert len(response_store.retrieve_calls) == 1


@pytest.mark.asyncio
async def test_stage_forwards_to_native_upstream_when_not_found():
    """A missing previous response is forwarded, not failed, when the selected
    upstream speaks the Responses API natively (it may hold the id server-side)."""
    from llm_proxy.core.identity import RequestIdentity

    stage = PreviousResponseResolutionStage()
    fake_req = MagicMock()
    fake_req.extra = {"previous_response_id": "resp_upstream_only"}
    fake_req.conversation = MagicMock()
    fake_req.conversation.messages = []
    fake_req.conversation.system_messages = []

    fake_state_req = MagicMock()
    fake_state_req.state = MagicMock()
    fake_state_req.state.identity = RequestIdentity(api_key_name="test-key")

    native_adapter = MagicMock()
    native_adapter._target_endpoint.return_value = "responses"

    state = PipelineState(
        raw_data={},
        unified_request=fake_req,
        req=fake_state_req,
        strategy=None,
        trace_id="t",
        event_context=MagicMock(),
    )
    state.adapter = native_adapter
    response_store = FakeResponseStore(payload=None)
    ctx = RequestContext(
        orchestrator=MagicMock(),
        services=MagicMock(),
        response_store=response_store,
    )
    # Must not raise; the id stays in extra so the native upstream resolves it.
    await stage.process(state, ctx)
    assert fake_req.extra["previous_response_id"] == "resp_upstream_only"


@pytest.mark.asyncio
async def test_stage_raises_when_store_disabled_and_chat_upstream():
    """With response storage disabled, a previous_response_id that can only be
    resolved locally must fail loudly for non-native upstreams instead of being
    silently stripped by the chat request builder (losing the client's context
    without any error)."""
    from llm_proxy.core.exceptions import NotFoundError

    stage = PreviousResponseResolutionStage()
    fake_req = MagicMock()
    fake_req.extra = {"previous_response_id": "resp_local_only"}

    chat_adapter = MagicMock()
    chat_adapter._target_endpoint.return_value = "chat_completions"

    state = PipelineState(
        raw_data={},
        unified_request=fake_req,
        req=None,
        strategy=None,
        trace_id="t",
        event_context=MagicMock(),
    )
    state.adapter = chat_adapter
    ctx = RequestContext(
        orchestrator=MagicMock(),
        services=MagicMock(),
        response_store=None,  # Redis disabled
    )
    with pytest.raises(NotFoundError) as exc_info:
        await stage.process(state, ctx)
    assert exc_info.value.code == "previous_response_not_found"
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_stage_forwards_when_store_disabled_and_native_upstream():
    """With response storage disabled, a native Responses upstream may still
    hold the referenced id server-side, so the id is forwarded (not failed)."""
    stage = PreviousResponseResolutionStage()
    fake_req = MagicMock()
    fake_req.extra = {"previous_response_id": "resp_upstream_only"}

    native_adapter = MagicMock()
    native_adapter._target_endpoint.return_value = "responses"

    state = PipelineState(
        raw_data={},
        unified_request=fake_req,
        req=None,
        strategy=None,
        trace_id="t",
        event_context=MagicMock(),
    )
    state.adapter = native_adapter
    ctx = RequestContext(
        orchestrator=MagicMock(),
        services=MagicMock(),
        response_store=None,  # Redis disabled
    )
    # Must not raise; the id stays in extra so the native upstream resolves it.
    await stage.process(state, ctx)
    assert fake_req.extra["previous_response_id"] == "resp_upstream_only"


@pytest.mark.asyncio
async def test_stage_raises_for_chat_upstream_when_not_found():
    """A missing previous response still fails for non-native upstreams."""
    from llm_proxy.core.exceptions import NotFoundError
    from llm_proxy.core.identity import RequestIdentity

    stage = PreviousResponseResolutionStage()
    fake_req = MagicMock()
    fake_req.extra = {"previous_response_id": "resp_missing"}
    fake_req.conversation = MagicMock()
    fake_req.conversation.messages = []
    fake_req.conversation.system_messages = []

    fake_state_req = MagicMock()
    fake_state_req.state = MagicMock()
    fake_state_req.state.identity = RequestIdentity(api_key_name="test-key")

    chat_adapter = MagicMock()
    chat_adapter._target_endpoint.return_value = "chat_completions"

    state = PipelineState(
        raw_data={},
        unified_request=fake_req,
        req=fake_state_req,
        strategy=None,
        trace_id="t",
        event_context=MagicMock(),
    )
    state.adapter = chat_adapter
    ctx = RequestContext(
        orchestrator=MagicMock(),
        services=MagicMock(),
        response_store=FakeResponseStore(payload=None),
    )
    with pytest.raises(NotFoundError) as exc_info:
        await stage.process(state, ctx)
    assert exc_info.value.code == "previous_response_not_found"
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_stage_resolves_item_references_against_previous_response():
    """item_reference entries pointing at items stored with the previous
    response are spliced into the materialized conversation at the position
    where the reference appeared in the new input."""
    from llm_proxy.core.identity import RequestIdentity
    from llm_proxy.models import ConversationContext, InternalRequest, Message, TextBlock
    from llm_proxy.models.content_blocks import ToolUseBlock

    stage = PreviousResponseResolutionStage()
    prev_response = {
        "id": "resp_prev",
        "input": [
            {"type": "message", "role": "user", "content": "run ls"},
            {
                "type": "function_call",
                "id": "item_fc",
                "call_id": "call_123",
                "name": "exec",
                "arguments": '{"cmd": "ls"}',
            },
        ],
        "output": [],
    }
    req = InternalRequest(
        model="gpt-4",
        conversation=ConversationContext(system_messages=[], messages=[]),
        extra={"previous_response_id": "resp_prev"},
    )
    # New input derived two messages with an item_reference between them
    # (recorded at message boundary 1 by the serializer).
    req.conversation.messages.append(Message(role="user", content=[TextBlock(text="first")]))
    req.conversation.messages.append(Message(role="user", content=[TextBlock(text="second")]))
    req._unresolved_item_references = [(1, "item_fc")]

    fake_state_req = MagicMock()
    fake_state_req.state = MagicMock()
    fake_state_req.state.identity = RequestIdentity(api_key_name="test-key")

    state = PipelineState(
        raw_data={},
        unified_request=req,
        req=fake_state_req,
        strategy=None,
        trace_id="t",
        event_context=MagicMock(),
    )
    ctx = RequestContext(
        orchestrator=MagicMock(),
        services=MagicMock(),
        response_store=FakeResponseStore(payload=prev_response),
    )
    await stage.process(state, ctx)

    roles = [m.role for m in req.conversation.messages]
    # user(prev), assistant(prev fc), user(first), assistant(referenced fc), user(second)
    assert roles == ["user", "assistant", "user", "assistant", "user"]
    referenced = req.conversation.messages[3]
    assert isinstance(referenced.content[0], ToolUseBlock)
    assert referenced.content[0].name == "exec"
    assert referenced.content[0].id == "call_123"


@pytest.mark.asyncio
async def test_stage_strips_previous_response_id_after_prepend():
    """After a successful prepend, previous_response_id is dropped from extra.

    The prior context is materialized in the conversation; forwarding the id to
    a native Responses provider would double-apply the previous turn.
    """
    from llm_proxy.core.identity import RequestIdentity
    from llm_proxy.models import ConversationContext, InternalRequest, Message, TextBlock

    stage = PreviousResponseResolutionStage()
    prev_response = {
        "id": "resp_prev",
        "input": "What is 2+2?",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "4"}],
                "status": "completed",
                "id": "msg_prev1",
            }
        ],
    }
    req = InternalRequest(
        model="gpt-4",
        conversation=ConversationContext(system_messages=[], messages=[]),
        extra={"previous_response_id": "resp_prev"},
    )
    req.conversation.messages.append(
        Message(role="user", content=[TextBlock(text="Now multiply by 3")])
    )

    fake_state_req = MagicMock()
    fake_state_req.state = MagicMock()
    fake_state_req.state.identity = RequestIdentity(api_key_name="test-key")

    state = PipelineState(
        raw_data={},
        unified_request=req,
        req=fake_state_req,
        strategy=None,
        trace_id="t",
        event_context=MagicMock(),
    )
    ctx = RequestContext(
        orchestrator=MagicMock(),
        services=MagicMock(),
        response_store=FakeResponseStore(payload=prev_response),
    )
    await stage.process(state, ctx)

    assert "previous_response_id" not in req.extra
    assert len(req.conversation.messages) == 3


@pytest.mark.asyncio
async def test_stage_handles_retrieve_exception():
    """Stage handles exceptions from response store gracefully."""
    from llm_proxy.core.identity import RequestIdentity

    stage = PreviousResponseResolutionStage()
    fake_req = MagicMock()
    fake_req.extra = {"previous_response_id": "prev_error"}
    fake_req.conversation = MagicMock()
    fake_req.conversation.messages = []
    fake_req.conversation.system_messages = []

    fake_state_req = MagicMock()
    fake_state_req.state = MagicMock()
    fake_state_req.state.identity = RequestIdentity(api_key_name="test-key")

    state = PipelineState(
        raw_data={},
        unified_request=fake_req,
        req=fake_state_req,
        strategy=None,
        trace_id="t",
        event_context=MagicMock(),
    )
    response_store = FakeResponseStore(should_fail=True)
    ctx = RequestContext(
        orchestrator=MagicMock(),
        services=MagicMock(),
        response_store=response_store,
    )
    # Should not raise
    await stage.process(state, ctx)


@pytest.mark.asyncio
async def test_stage_prepends_reasoning_from_prev_input():
    """Reasoning items in previous input should be reconstructed as ThinkingBlocks."""
    from llm_proxy.core.identity import RequestIdentity
    from llm_proxy.models import ConversationContext, InternalRequest, Message, TextBlock
    from llm_proxy.models.content_blocks import ThinkingBlock

    stage = PreviousResponseResolutionStage()
    prev_response = {
        "id": "resp_prev",
        "input": [
            {"type": "message", "role": "user", "content": "search for files"},
            {
                "type": "reasoning",
                "content": [{"type": "reasoning_text", "text": "I'll search for Python files."}],
                "summary": [],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Looking..."}],
            },
        ],
        "output": [],
    }
    req = InternalRequest(
        model="deepseek",
        conversation=ConversationContext(system_messages=[], messages=[]),
        extra={"previous_response_id": "resp_prev"},
    )
    req.conversation.messages.append(Message(role="user", content=[TextBlock(text="continue")]))

    fake_state_req = MagicMock()
    fake_state_req.state = MagicMock()
    fake_state_req.state.identity = RequestIdentity(api_key_name="test-key")

    state = PipelineState(
        raw_data={},
        unified_request=req,
        req=fake_state_req,
        strategy=None,
        trace_id="t",
        event_context=MagicMock(),
    )
    ctx = RequestContext(
        orchestrator=MagicMock(),
        services=MagicMock(),
        response_store=FakeResponseStore(payload=prev_response),
    )
    await stage.process(state, ctx)

    # user, merged assistant turn (reasoning + text), user = 3 messages.
    # The dispatch merges reasoning + assistant message into one assistant
    # turn, matching request-parse-time behavior.
    assert len(req.conversation.messages) == 3
    assistant = req.conversation.messages[1]
    assert assistant.role == "assistant"
    assert isinstance(assistant.content[0], ThinkingBlock)
    assert assistant.content[0].thinking == "I'll search for Python files."
    assert assistant.content[0].encrypted_content is None
    assert assistant.content[1].text == "Looking..."


@pytest.mark.asyncio
async def test_stage_prepends_reasoning_from_prev_output():
    """Reasoning items in previous output should be reconstructed as ThinkingBlocks."""
    from llm_proxy.core.identity import RequestIdentity
    from llm_proxy.models import ConversationContext, InternalRequest, Message, TextBlock
    from llm_proxy.models.content_blocks import ThinkingBlock

    stage = PreviousResponseResolutionStage()
    prev_response = {
        "id": "resp_prev",
        "input": [{"type": "message", "role": "user", "content": "what is 2+2"}],
        "output": [
            {
                "type": "reasoning",
                "id": "rsn_test",
                "status": "completed",
                "content": [{"type": "reasoning_text", "text": "Simple arithmetic."}],
                "summary": [],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "4"}],
                "status": "completed",
            },
        ],
    }
    req = InternalRequest(
        model="deepseek",
        conversation=ConversationContext(system_messages=[], messages=[]),
        extra={"previous_response_id": "resp_prev"},
    )
    req.conversation.messages.append(Message(role="user", content=[TextBlock(text="thanks")]))

    fake_state_req = MagicMock()
    fake_state_req.state = MagicMock()
    fake_state_req.state.identity = RequestIdentity(api_key_name="test-key")

    state = PipelineState(
        raw_data={},
        unified_request=req,
        req=fake_state_req,
        strategy=None,
        trace_id="t",
        event_context=MagicMock(),
    )
    ctx = RequestContext(
        orchestrator=MagicMock(),
        services=MagicMock(),
        response_store=FakeResponseStore(payload=prev_response),
    )
    await stage.process(state, ctx)

    # user (prev), merged assistant turn (reasoning + text), user (current)
    assert len(req.conversation.messages) == 3
    assistant = req.conversation.messages[1]
    assert assistant.role == "assistant"
    assert isinstance(assistant.content[0], ThinkingBlock)
    assert assistant.content[0].thinking == "Simple arithmetic."
    assert assistant.content[0].encrypted_content is None


@pytest.mark.asyncio
async def test_repair_encrypted_reasoning_restores_from_cache():
    """Encrypted ThinkingBlocks in current conversation are repaired with cached reasoning."""
    from llm_proxy.core.identity import RequestIdentity
    from llm_proxy.models import ConversationContext, InternalRequest, Message, TextBlock
    from llm_proxy.models.content_blocks import ThinkingBlock, ToolUseBlock

    stage = PreviousResponseResolutionStage()
    prev_response = {
        "id": "resp_prev",
        "output": [
            {
                "type": "reasoning",
                "id": "rsn_1",
                "status": "completed",
                "content": [{"type": "reasoning_text", "text": "Let me analyze the code."}],
                "summary": [],
            },
            {
                "type": "function_call",
                "call_id": "call_test",
                "name": "exec",
                "arguments": '{"cmd": "ls"}',
            },
        ],
    }
    req = InternalRequest(
        model="deepseek",
        conversation=ConversationContext(system_messages=[], messages=[]),
        extra={"previous_response_id": "resp_prev"},
    )
    # Current conversation has an encrypted ThinkingBlock (from Codex)
    req.conversation.messages.append(
        Message(
            role="assistant",
            content=[
                ThinkingBlock(thinking="", encrypted_content="encrypted_blob"),
                TextBlock(text="I'll now execute the command."),
                ToolUseBlock(id="call_test", name="exec", input={"cmd": "ls"}),
            ],
        )
    )
    req.conversation.messages.append(
        Message(role="user", content=[TextBlock(text="next question")])
    )

    fake_state_req = MagicMock()
    fake_state_req.state = MagicMock()
    fake_state_req.state.identity = RequestIdentity(api_key_name="test-key")

    state = PipelineState(
        raw_data={},
        unified_request=req,
        req=fake_state_req,
        strategy=None,
        trace_id="t",
        event_context=MagicMock(),
    )
    ctx = RequestContext(
        orchestrator=MagicMock(),
        services=MagicMock(),
        response_store=FakeResponseStore(payload=prev_response),
    )
    await stage.process(state, ctx)

    # The encrypted ThinkingBlock should now have the cached reasoning.
    # After prepend: [merged prev turn (reasoning + function_call),
    # current-encrypted-assistant, user]
    assert len(req.conversation.messages) == 3
    assistant_msg = req.conversation.messages[1]
    assert isinstance(assistant_msg.content[0], ThinkingBlock)
    assert assistant_msg.content[0].thinking == "Let me analyze the code."
    assert assistant_msg.content[0].encrypted_content == "encrypted_blob"


@pytest.mark.asyncio
async def test_repair_encrypted_reasoning_falls_back_when_no_cache():
    """When no reasoning in cached response, encrypted blocks stay empty."""
    from llm_proxy.core.identity import RequestIdentity
    from llm_proxy.models import ConversationContext, InternalRequest, Message
    from llm_proxy.models.content_blocks import ThinkingBlock

    stage = PreviousResponseResolutionStage()
    prev_response: dict = {"id": "resp_prev", "input": [], "output": []}  # No reasoning in output
    req = InternalRequest(
        model="deepseek",
        conversation=ConversationContext(system_messages=[], messages=[]),
        extra={"previous_response_id": "resp_prev"},
    )
    req.conversation.messages.append(
        Message(
            role="assistant",
            content=[
                ThinkingBlock(thinking="", encrypted_content="encrypted_no_cache"),
            ],
        )
    )

    fake_state_req = MagicMock()
    fake_state_req.state = MagicMock()
    fake_state_req.state.identity = RequestIdentity(api_key_name="test-key")

    state = PipelineState(
        raw_data={},
        unified_request=req,
        req=fake_state_req,
        strategy=None,
        trace_id="t",
        event_context=MagicMock(),
    )
    ctx = RequestContext(
        orchestrator=MagicMock(),
        services=MagicMock(),
        response_store=FakeResponseStore(payload=prev_response),
    )
    await stage.process(state, ctx)

    # Thinking should still be empty (no cache to restore from)
    # No prepended items, so original message stays at index 0
    assert req.conversation.messages[0].content[0].thinking == ""
    assert req.conversation.messages[0].content[0].encrypted_content == "encrypted_no_cache"


@pytest.mark.asyncio
async def test_fallback_rerun_rematerializes_previous_response():
    """Fallback re-parses from the pristine body, so the previous-response
    materialization must be re-applied for the fallback provider — otherwise
    the rebuilt body would carry an unresolvable proxy-local id."""
    from unittest.mock import AsyncMock

    from llm_proxy.config.types.provider import ProviderConfig
    from llm_proxy.core.identity import RequestIdentity
    from llm_proxy.core.processing.fallback import setup_fallback_provider
    from llm_proxy.core.processing.stages.parameter_override import (
        ParameterOverrideService,
    )
    from llm_proxy.core.provider_selector import ProviderSelectionResult
    from llm_proxy.protocols.registry import get_protocol_serializer

    prev_response = {
        "id": "resp_prev",
        "input": "hi",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hello"}],
                "status": "completed",
                "id": "msg_prev1",
            }
        ],
    }

    service = ParameterOverrideService(get_protocol_serializer("openresponses"))
    store = FakeResponseStore(payload=prev_response)
    context = RequestContext(
        orchestrator=MagicMock(),
        services=ServiceDependencies(adapter_factory=AsyncMock(return_value=MagicMock())),
        response_store=store,
    )

    req = MagicMock()
    req.state = MagicMock()
    req.state.request_id = "req-1"
    req.state.identity = RequestIdentity(api_key_name="test-key")

    pristine_raw = {
        "model": "fast",
        "input": "again",
        "previous_response_id": "resp_prev",
        "stream": False,
    }
    initial_request = service._serializer.parse_request(pristine_raw)

    selection = ProviderSelectionResult(
        provider_name="openai",
        provider_config=ProviderConfig(type="openai", api_key="test-key"),
        provider_model_name="gpt-4o",
        priority=1,
        parameter_overrides=None,
    )
    result = await setup_fallback_provider(
        selection, req, initial_request, pristine_raw, context, service
    )

    assert result is not None
    _, new_request = result
    # The stage re-run materialized the stored response for the fresh request.
    assert store.retrieve_calls == [("test-key", "resp_prev")]
    assert new_request.previous_response_materialized is True
    assert "previous_response_id" not in (new_request.extra or {})
