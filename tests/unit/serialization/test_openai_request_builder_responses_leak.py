"""Tests that OpenResponses-derived fields do not leak into Chat Completions bodies.

Those requests carry Responses-API-only fields (e.g. ``include``, ``reasoning`` object,
``previous_response_id``) and ``developer`` role messages. When the proxy
translates them to a Chat Completions upstream request, only the fields that
have a Chat Completions equivalent should survive, and ``developer`` should be
degraded to ``system`` for generic Chat Completions providers.
"""

import orjson
import pytest

from llm_proxy.models import (
    ConversationContext,
    CustomTool,
    CustomToolUseBlock,
    FunctionTool,
    InternalRequest,
    Message,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
)
from llm_proxy.models.params import GenerationParams, OpenAISpecificParams
from llm_proxy.serialization.context import BuildContext
from llm_proxy.serialization.openai.components.request_builder import (
    OpenAIRequestBuilder,
)


@pytest.fixture
def builder():
    return OpenAIRequestBuilder()


def _codex_like_request() -> InternalRequest:
    """Build an InternalRequest that mirrors a Codex /v1/responses request."""
    return InternalRequest(
        model="deepseek-v4-pro",
        conversation=ConversationContext(
            system_messages=[
                SystemMessage.from_text(role="system", text="Top-level instruction."),
            ],
            messages=[
                Message(role="developer", content=[TextBlock(text="Dev prompt 1")]),
                Message(role="user", content=[TextBlock(text="User prompt 1")]),
                Message(role="developer", content=[TextBlock(text="Dev prompt 2")]),
                Message(role="user", content=[TextBlock(text="User prompt 2")]),
            ],
        ),
        params=GenerationParams(
            openai=OpenAISpecificParams(
                store=False,
                prompt_cache_key="019f4bed-3694-7901-9135-e74a41b44318",
                parallel_tool_calls=True,
                reasoning_effort="xhigh",
            )
        ),
        stream=True,
        extra={
            "include": ["reasoning.encrypted_content"],
            "reasoning": {"effort": "xhigh", "summary": "auto"},
            "previous_response_id": "prev_123",
            "background": False,
            "max_tool_calls": 10,
            "truncation": "disabled",
        },
    )


def test_responses_only_extra_keys_are_not_leaked(builder: OpenAIRequestBuilder):
    """Responses-only keys from request.extra must not reach the provider body."""
    request = _codex_like_request()
    context = BuildContext.from_request(
        request,
        provider_name="openai-compatible",
        base_url="https://example.com/v1",
        target_endpoint="chat_completions",
    )
    body = builder.build(request, context)

    for key in (
        "include",
        "reasoning",
        "previous_response_id",
        "background",
        "max_tool_calls",
        "truncation",
    ):
        assert key not in body, f"Responses-only extra key '{key}' leaked into chat body"


def test_chat_completions_equivalent_params_are_kept(builder: OpenAIRequestBuilder):
    """OpenAI params that are valid for Chat Completions must remain.

    ``prompt_cache_key`` is the exception: it is only forwarded to upstreams
    known to accept it, so it is dropped for the unknown example.com gateway.
    """
    request = _codex_like_request()
    context = BuildContext.from_request(
        request,
        provider_name="openai-compatible",
        base_url="https://example.com/v1",
        target_endpoint="chat_completions",
    )
    body = builder.build(request, context)

    assert body.get("store") is False
    assert "prompt_cache_key" not in body
    assert body.get("reasoning_effort") == "xhigh"
    assert body.get("parallel_tool_calls") is True


def test_prompt_cache_key_forwarded_to_known_compatible_upstream(
    builder: OpenAIRequestBuilder,
):
    """prompt_cache_key reaches api.openai.com chat bodies (known compatible)."""
    request = _codex_like_request()
    context = BuildContext.from_request(
        request,
        provider_name="openai",
        base_url="https://api.openai.com/v1",
        target_endpoint="chat_completions",
    )
    body = builder.build(request, context)

    assert body.get("prompt_cache_key") == "019f4bed-3694-7901-9135-e74a41b44318"


def test_developer_role_is_degraded_to_system_for_compatible_providers(
    builder: OpenAIRequestBuilder,
):
    """Developer messages must become system messages for chat completions providers."""
    request = _codex_like_request()
    context = BuildContext.from_request(
        request,
        provider_name="openai-compatible",
        base_url="https://example.com/v1",
        target_endpoint="chat_completions",
    )
    body = builder.build(request, context)

    roles = [msg["role"] for msg in body["messages"]]
    assert "developer" not in roles
    assert roles.count("system") == 3  # 1 system + 2 degraded developer messages
    assert roles.count("user") == 2


def test_developer_role_is_preserved_for_responses_endpoint(
    builder: OpenAIRequestBuilder,
):
    """Developer messages are kept as-is when the target endpoint is Responses."""
    request = _codex_like_request()
    context = BuildContext.from_request(
        request,
        provider_name="openai",
        base_url="https://api.openai.com/v1",
        target_endpoint="responses",
    )
    body = builder.build(request, context)

    roles = [msg["role"] for msg in body["messages"]]
    assert "developer" in roles
    assert roles.count("system") == 1
    assert roles.count("user") == 2


def test_custom_tool_is_converted_to_function_for_chat_completions(builder: OpenAIRequestBuilder):
    """Custom tools are converted to function tools for Chat Completions.

    A Chat Completions provider cannot handle ``type: "custom"`` tools. The
    request builder converts them to ``function`` tools with a single
    ``content`` string parameter and embeds grammar in the description.
    This mirrors LiteLLM's ``convert_custom_tool_to_function_tool``.
    """
    request = InternalRequest(
        model="deepseek-v4-pro",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="hi")])],
        ),
        tools=[
            FunctionTool(name="exec_command"),
            CustomTool(name="apply_patch", format_type="freeform"),
        ],
        stream=False,
    )
    context = BuildContext.from_request(
        request,
        provider_name="openai-compatible",
        base_url="https://example.com/v1",
        target_endpoint="chat_completions",
    )
    body = builder.build(request, context)

    assert body.get("tools") is not None
    assert len(body["tools"]) == 2
    assert all(t["type"] == "function" for t in body["tools"])

    # The converted apply_patch should be a function tool with a content param
    apply_patch = [t for t in body["tools"] if t["function"]["name"] == "apply_patch"]
    assert len(apply_patch) == 1
    fn = apply_patch[0]["function"]
    assert "content" in fn["parameters"]["properties"]


def test_custom_tool_is_kept_for_responses_endpoint(builder: OpenAIRequestBuilder):
    """The OpenAI Responses API supports custom tools natively, so they survive."""
    request = InternalRequest(
        model="gpt-4",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="hi")])],
        ),
        tools=[CustomTool(name="apply_patch", format_type="freeform")],
        stream=False,
    )
    context = BuildContext.from_request(
        request,
        provider_name="openai",
        base_url="https://api.openai.com/v1",
        target_endpoint="responses",
    )
    body = builder.build(request, context)

    assert body.get("tools") is not None
    assert len(body["tools"]) == 1
    assert body["tools"][0]["type"] == "custom"
    # Flat format: custom tool should have "name" at top level, not nested.
    assert body["tools"][0]["name"] == "apply_patch"


def test_custom_tool_call_in_history_is_converted_to_function_for_chat_completions(
    builder: OpenAIRequestBuilder,
):
    """Prior custom_tool_call items must become function tool calls on Chat Completions.

    Codex sends earlier ``custom_tool_call`` items (e.g. its ``exec`` tool) back in
    the conversation. When the upstream target is Chat Completions these must be
    serialized as ``type: "function"`` tool calls wrapped in the
    ``{"content": ...}`` bridge envelope — strict OpenAI-compatible providers
    reject ``type: "custom"`` with "unknown variant `custom`, expected `function`".
    """
    raw_input = "const r = await tools.exec_command({ cmd: 'ls' });\ntext(r.output);"
    request = InternalRequest(
        model="deepseek-v4-flash",
        conversation=ConversationContext(
            messages=[
                Message(role="user", content=[TextBlock(text="Explore the repo")]),
                Message(
                    role="assistant",
                    content=[
                        CustomToolUseBlock(
                            id="call_00_4ubgNDYhyFoVnj4GH0cu6632",
                            name="exec",
                            input=raw_input,
                        ),
                    ],
                ),
                Message(
                    role="user",
                    content=[
                        ToolResultBlock(
                            tool_use_id="call_00_4ubgNDYhyFoVnj4GH0cu6632",
                            content="done",
                        )
                    ],
                ),
            ],
        ),
        stream=True,
    )
    context = BuildContext.from_request(
        request,
        provider_name="openai-compatible",
        base_url="https://example.com/v1",
        target_endpoint="chat_completions",
    )
    body = builder.build(request, context)

    # No message in the body may carry a custom tool call.
    for msg in body["messages"]:
        assert "custom" not in msg

    assistant = [m for m in body["messages"] if m["role"] == "assistant"]
    assert len(assistant) == 1
    tool_call = assistant[0]["tool_calls"][0]
    assert tool_call["id"] == "call_00_4ubgNDYhyFoVnj4GH0cu6632"
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "exec"
    # The freeform input is re-wrapped in the {"content": ...} bridge envelope
    # that mirrors the converted tool definition's single ``content`` parameter.
    assert orjson.loads(tool_call["function"]["arguments"]) == {"content": raw_input}

    # The tool result still references the same call id.
    tool_msgs = [m for m in body["messages"] if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["tool_call_id"] == "call_00_4ubgNDYhyFoVnj4GH0cu6632"


def test_custom_tool_call_in_history_is_kept_for_responses_endpoint(
    builder: OpenAIRequestBuilder,
):
    """Custom tool calls stay ``type: "custom"`` when targeting the Responses API."""
    request = InternalRequest(
        model="gpt-4",
        conversation=ConversationContext(
            messages=[
                Message(
                    role="assistant",
                    content=[
                        CustomToolUseBlock(
                            id="call_1",
                            name="exec",
                            input="const x = 1;",
                        ),
                    ],
                ),
            ],
        ),
        stream=False,
    )
    context = BuildContext.from_request(
        request,
        provider_name="openai",
        base_url="https://api.openai.com/v1",
        target_endpoint="responses",
    )
    body = builder.build(request, context)

    assistant = [m for m in body["messages"] if m["role"] == "assistant"]
    assert len(assistant) == 1
    tool_call = assistant[0]["tool_calls"][0]
    assert tool_call["type"] == "custom"
    assert tool_call["custom"]["name"] == "exec"
    assert tool_call["custom"]["input"] == "const x = 1;"


def test_developer_role_is_degraded_for_openai_chat_compatible(
    builder: OpenAIRequestBuilder,
):
    """Developer messages are degraded when routed through chat_completions.

    OpenAI's own Chat Completions endpoint accepts ``developer``, but the proxy's
    ``openai-compatible`` adapter targets generic providers that may not. We keep
    the degradation for any Chat Completions endpoint.
    """
    request = _codex_like_request()
    context = BuildContext.from_request(
        request,
        provider_name="openai",
        base_url="https://api.openai.com/v1",
        target_endpoint="chat_completions",
    )
    body = builder.build(request, context)

    roles = [msg["role"] for msg in body["messages"]]
    assert "developer" not in roles
    assert roles.count("system") == 3
    assert roles.count("user") == 2
