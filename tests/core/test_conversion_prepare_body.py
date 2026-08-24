"""Tests for the conversion seam's native body preparation."""

import pytest

from llm_proxy.core.conversion import prepare_native_body
from llm_proxy.models import ConversationContext, InternalRequest, Message, TextBlock
from llm_proxy.providers.anthropic import AnthropicAdapter
from llm_proxy.providers.deepseek import DeepSeekAdapter
from llm_proxy.providers.openai import OpenAIAdapter


def _request(raw: dict, protocol_name: str) -> InternalRequest:
    req = InternalRequest(
        model="routed-model-id",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="hi")])]
        ),
    )
    req.metadata.protocol_name = protocol_name
    req._raw_protocol_data = raw
    return req


def test_prepare_native_body_shared_preparation():
    """Copy + None-strip + routed model + stream flag, hook delegated."""
    raw = {
        "model": "client-alias",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        "stop_sequences": None,
        "future_field": {"nested": True},
    }
    adapter = AnthropicAdapter(api_key="k", base_url="https://api.anthropic.com")
    req = _request(raw, "anthropic")

    body = prepare_native_body(adapter, req, stream=True)

    assert body == {
        "model": "routed-model-id",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        "future_field": {"nested": True},
        "stream": True,
    }
    # The stashed raw body is untouched (None kept, alias kept, no stream).
    assert raw["model"] == "client-alias"
    assert "stream" not in raw
    assert "stop_sequences" in raw


def test_prepare_native_body_without_stream_flag():
    adapter = AnthropicAdapter(api_key="k", base_url="https://api.anthropic.com")
    raw = {"model": "client-alias", "messages": [], "stream": False}
    body = prepare_native_body(adapter, _request(raw, "anthropic"))
    # stream=None leaves the client's own flag untouched, adds nothing.
    assert body["stream"] is False


def test_prepare_native_body_openai_hook_normalizes_text_and_strips_ids():
    adapter = OpenAIAdapter(api_key="k")
    raw = {
        "model": "client-alias",
        "input": [
            {"type": "message", "id": "item_abc", "role": "user", "content": []},
            {"type": "item_reference", "id": "resp_1"},
        ],
        "text": {"type": "json_object"},
    }
    body = prepare_native_body(adapter, _request(raw, "openresponses"))
    # Legacy flat text shape wrapped under format.
    assert body["text"] == {"format": {"type": "json_object"}}
    # Codex item ids stripped; item_reference ids preserved (lookup keys).
    assert "id" not in body["input"][0]
    assert body["input"][1]["id"] == "resp_1"
    # Raw stash untouched.
    assert raw["text"] == {"type": "json_object"}
    assert raw["input"][0]["id"] == "item_abc"


def test_prepare_native_body_deepseek_hook_scopes_to_anthropic_shape():
    adapter = DeepSeekAdapter(api_key="k")
    # Responses-shaped body: messages absent, hook is a no-op.
    raw = {"model": "client-alias", "input": [{"type": "message", "id": "x"}]}
    body = prepare_native_body(adapter, _request(raw, "openresponses"))
    assert body["input"] == [{"type": "message", "id": "x"}]


def test_prepare_native_body_requires_stashed_raw():
    """A missing raw stash fails loudly instead of sending a bare model/stream body."""
    adapter = AnthropicAdapter(api_key="k", base_url="https://api.anthropic.com")
    req = _request({"model": "client-alias", "messages": []}, "anthropic")
    req._raw_protocol_data = None
    with pytest.raises(ValueError, match="_raw_protocol_data"):
        prepare_native_body(adapter, req)


def test_prepare_native_body_hook_repairs_dangling_tool_use():
    adapter = AnthropicAdapter(api_key="k", base_url="https://api.anthropic.com")
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "run"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "running"},
                {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {}},
            ],
        },
    ]
    raw = {"model": "client-alias", "messages": messages}
    body = prepare_native_body(adapter, _request(raw, "anthropic"))
    # Dangling tool_use turn dropped (Anthropic would 400).
    assert body["messages"] == [{"role": "user", "content": [{"type": "text", "text": "run"}]}]
    # The raw stash keeps the original two messages.
    assert len(raw["messages"]) == 2
