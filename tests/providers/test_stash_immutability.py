"""Regression: outbound body building must never pollute the raw stash.

The wire-compatible rebuild shortcut (serializer ``compatible_protocols``)
reuses the client's raw body. Downstream mutators (reasoning-field
normalization, reasoning-echo injection) write into nested message dicts in
place — so the wire-reuse tier must hand them a fully detached copy. The stash is the same object
as ``PipelineState.original_raw_data`` when no parameter overrides ran, and
the fallback chain re-parses from it as the pristine client body (ADR-0008):
pollution means the next provider attempt sees reasoning the client never
sent.
"""

from llm_proxy.models import (
    ConversationContext,
    InternalRequest,
    Message,
    TextBlock,
)
from llm_proxy.providers.deepseek.adapter import DeepSeekAdapter
from llm_proxy.providers.openrouter.adapter import OpenRouterAdapter

_TOOL_CALLS = [
    {
        "id": "call_1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": "{}"},
    }
]


def _request_with_stash(model: str, raw_messages: list[dict]) -> InternalRequest:
    request = InternalRequest(
        model=model,
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="hi")])]
        ),
    )
    request.metadata.protocol_name = "openai"
    request._raw_protocol_data = {
        "model": "client-alias",
        "messages": raw_messages,
    }
    return request


class TestFastPathStashImmutability:
    def test_reasoning_echo_does_not_pollute_stash(self):
        """DeepSeek echo injects reasoning_content into the outbound body only."""
        stash_messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": _TOOL_CALLS},
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        ]
        request = _request_with_stash("deepseek-v4-pro", stash_messages)
        adapter = DeepSeekAdapter(api_key="sk-test")

        body = adapter._build_request_body(request)

        # Outbound body got the placeholder echo…
        assert body["messages"][1]["reasoning_content"]
        # …but the stashed raw body is byte-identical to what the client sent.
        assert stash_messages[1] == {
            "role": "assistant",
            "content": None,
            "tool_calls": _TOOL_CALLS,
        }
        assert "reasoning_content" not in request._raw_protocol_data["messages"][1]

    def test_reasoning_rename_does_not_pollute_stash(self):
        """OpenRouter's reasoning_content→reasoning rename stays off the stash."""
        stash_messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "done",
                "reasoning_content": "thought",
                "tool_calls": _TOOL_CALLS,
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        ]
        # Model without echo markers: isolates the rename mutator.
        request = _request_with_stash("openai/gpt-5", stash_messages)
        adapter = OpenRouterAdapter(api_key="sk-test")

        body = adapter._build_request_body(request)

        # Outbound body renamed the field for OpenRouter…
        assert body["messages"][1]["reasoning"] == "thought"
        assert "reasoning_content" not in body["messages"][1]
        # …but the stash kept the client's original field.
        assert request._raw_protocol_data["messages"][1]["reasoning_content"] == "thought"
        assert "reasoning" not in request._raw_protocol_data["messages"][1]
