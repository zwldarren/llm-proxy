# tests/test_serialization/test_anthropic.py
"""Tests for Anthropic serializer."""

import pytest

from llm_proxy.models import (
    AnthropicSpecificParams,
    ConversationContext,
    GenerationParams,
    InternalRequest,
    InternalResponse,
    Message,
    TextBlock,
    ThinkingBlock,
    ThinkingConfig,
    ToolChoice,
    ToolResultBlock,
    ToolUseBlock,
)
from llm_proxy.models.tools import CustomTool, FunctionTool
from llm_proxy.protocols.anthropic.serializer import AnthropicProtocolSerializer
from llm_proxy.serialization.anthropic.serializer import AnthropicProviderSerializer
from llm_proxy.serialization.anthropic.streaming_converter import AnthropicChunkConverter


class AnthropicSerializer(AnthropicProtocolSerializer, AnthropicProviderSerializer):
    """Combined serializer for testing both protocol and provider methods."""

    pass


@pytest.fixture
def serializer():
    """Create an AnthropicSerializer instance directly."""
    return AnthropicSerializer()


def test_serializer_registered():
    """Test Anthropic serializer is registered."""
    from llm_proxy.protocols.registry import get_protocol_serializer
    from llm_proxy.serialization.providers import get_provider_serializer

    protocol_serializer = get_protocol_serializer("anthropic")
    assert isinstance(protocol_serializer, AnthropicProtocolSerializer)

    provider_serializer = get_provider_serializer("anthropic")
    assert isinstance(provider_serializer, AnthropicProviderSerializer)


def test_parse_simple_request(serializer):
    """Test parsing a simple Anthropic request."""
    data = {
        "model": "claude-3-opus",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hello, world!"}],
    }

    request = serializer.parse_request(data)

    assert request.model == "claude-3-opus"
    assert len(request.conversation.messages) == 1
    assert request.conversation.messages[0].role == "user"


def test_parse_request_with_system(serializer):
    """Test parsing request with system message."""
    data = {
        "model": "claude-3-opus",
        "max_tokens": 1024,
        "system": "You are a helpful assistant.",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    request = serializer.parse_request(data)

    assert len(request.conversation.system_messages) == 1
    assert request.conversation.system_messages[0].text_content == "You are a helpful assistant."


def test_format_response(serializer):
    """Test formatting response to Anthropic format."""
    response = InternalResponse(
        id="test-id", model="claude-3-opus", output=[TextBlock(text="Hello!")]
    )

    result = serializer.format_response(response)

    assert result["id"] == "test-id"
    assert result["model"] == "claude-3-opus"
    assert result["type"] == "message"
    assert len(result["content"]) == 1


def test_format_response_emits_stop_sequence(serializer):
    """Regression: stop_sequence must appear in the formatted Anthropic response."""
    response = InternalResponse(
        id="test-id",
        model="claude-3-opus",
        output=[TextBlock(text="Hello!")],
        finish_reason="stop_sequence",
        provider_info={"stop_sequence": "END"},
    )

    result = serializer.format_response(response)

    assert result["stop_reason"] == "stop_sequence"
    assert result["stop_sequence"] == "END"


def test_parse_provider_response_preserves_stop_sequence_round_trip(serializer):
    """Regression: stop_sequence stop_reason must survive a provider -> protocol round trip."""
    provider_response = {
        "id": "test-id",
        "model": "claude-3-opus",
        "content": [{"type": "text", "text": "Hello!"}],
        "stop_reason": "stop_sequence",
        "stop_sequence": "END",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }

    response = serializer.parse_provider_response(provider_response, model="claude-3-opus")
    assert response.finish_reason == "stop_sequence"
    assert response.provider_info.get("stop_sequence") == "END"

    result = serializer.format_response(response)
    assert result["stop_reason"] == "stop_sequence"
    assert result["stop_sequence"] == "END"


def test_build_provider_request(serializer):
    """Test building provider request."""
    request = InternalRequest(
        model="claude-3-opus",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="Hello")])]
        ),
        params=GenerationParams(max_tokens=1024),
    )

    body = serializer.build_provider_request(request)

    assert body["model"] == "claude-3-opus"
    assert body["max_tokens"] == 1024
    assert len(body["messages"]) == 1


def test_build_provider_request_keeps_custom_tool(serializer):
    """Custom tools (e.g. Codex CLI ``exec``) must not be dropped.

    Regression: the Anthropic serializer previously had no ``CustomTool``
    branch, so Responses API custom tools inside namespaces (Codex CLI's
    ``functions.exec``) were silently omitted from the outbound body. The
    model then never saw ``exec``, kept calling ``wait`` on non-existent
    cells, and the client looped forever without a result.
    """
    request = InternalRequest(
        model="kimi-k2.7-code",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="read the README")])]
        ),
        params=GenerationParams(),
        tools=[
            CustomTool(
                name="functions__exec",
                description="Run JavaScript code to orchestrate tool calls",
            ),
            FunctionTool(
                name="functions__wait",
                description="Waits on a yielded exec cell",
                parameters={"type": "object"},
            ),
        ],
    )

    body = serializer.build_provider_request(request)

    assert body["tools"], "tools must not be empty"
    names = [t["name"] for t in body["tools"]]
    assert "functions__exec" in names, "custom tool must be forwarded to the provider"
    exec_def = next(t for t in body["tools"] if t["name"] == "functions__exec")
    assert exec_def["description"].startswith("Run JavaScript code to orchestrate tool calls")
    # Schema-less custom tools carry raw text input (e.g. Codex exec takes
    # raw JavaScript source). Anthropic-compatible endpoints require an object
    # schema, so the raw text is wrapped in a single ``input`` property and
    # unwrapped on the response side (unwrap_custom_tool_arguments). The
    # original definition is embedded in the description so the model knows
    # the tool is freeform.
    assert exec_def["input_schema"] == {
        "type": "object",
        "properties": {
            "input": {
                "type": "string",
                "description": (
                    "Raw string input for the original custom tool. "
                    "Preserve formatting exactly and follow the original "
                    "tool definition embedded in the description."
                ),
            }
        },
        "required": ["input"],
    }
    assert "Original tool definition:" in exec_def["description"]
    assert '"type":"custom"' in exec_def["description"]
    assert "functions__wait" in names


def test_custom_tool_embeds_original_definition_with_grammar(serializer):
    """Grammar/format metadata must survive the custom-tool bridge.

    The bridged function tool embeds the original definition (including the
    freeform grammar hint) in its description so the model knows the tool is
    freeform and how its input must be formatted — mirrors cc-switch's
    "Original tool definition:" trick.
    """
    request = InternalRequest(
        model="kimi-k2.7-code",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="edit the file")])]
        ),
        params=GenerationParams(),
        tools=[
            CustomTool(
                name="apply_patch",
                description="Use the apply_patch tool to edit files.",
                format_type="grammar",
                grammar_definition="start: begin_patch hunk+ end_patch",
                grammar_syntax="lark",
            ),
        ],
    )

    body = serializer.build_provider_request(request)

    tool_def = body["tools"][0]
    assert tool_def["input_schema"]["properties"]["input"]["type"] == "string"
    assert "Original tool definition:" in tool_def["description"]
    assert '"type":"custom"' in tool_def["description"]
    assert '"type":"grammar"' in tool_def["description"]
    assert "start: begin_patch hunk+ end_patch" in tool_def["description"]
    assert '"syntax":"lark"' in tool_def["description"]


def test_function_tool_parameters_normalized_to_object(serializer):
    """Function tools with null/non-object parameters must not 400.

    Anthropic requires ``input_schema.type == "object"``; OpenAI Responses
    function tools may carry ``parameters: null`` or ``{"type": null}``.
    Mirrors cc-switch's ``normalize_function_parameters``.
    """
    request = InternalRequest(
        model="kimi-k2.7-code",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="hi")])]
        ),
        params=GenerationParams(),
        tools=[
            FunctionTool(name="no_params", parameters=None),
            FunctionTool(name="null_type", parameters={"type": None, "properties": {}}),
            FunctionTool(
                name="ok",
                parameters={"type": "object", "properties": {"x": {"type": "string"}}},
            ),
        ],
    )

    body = serializer.build_provider_request(request)

    by_name = {t["name"]: t["input_schema"] for t in body["tools"]}
    assert by_name["no_params"] == {"type": "object", "properties": {}}
    assert by_name["null_type"]["type"] == "object"
    assert by_name["ok"] == {"type": "object", "properties": {"x": {"type": "string"}}}


def test_messages_normalized_leading_assistant_prepends_user(serializer):
    """Anthropic requires the first message to be ``user``.

    Compacted/resumed sessions (e.g. Codex via /v1/responses) may start with
    an assistant turn; a placeholder user message is prepended instead of
    failing. Mirrors cc-switch's ``ensure_leading_user_message``.
    """
    request = InternalRequest(
        model="kimi-k2.7-code",
        conversation=ConversationContext(
            messages=[
                Message(role="assistant", content=[TextBlock(text="Let me check.")]),
                Message(role="user", content=[TextBlock(text="ok")]),
            ]
        ),
        params=GenerationParams(),
    )

    body = serializer.build_provider_request(request)

    assert body["messages"][0]["role"] == "user"
    assert body["messages"][0]["content"][0]["text"] == "(continuing the conversation)"
    assert body["messages"][1]["role"] == "assistant"


def test_messages_normalized_drops_incomplete_tool_turn(serializer):
    """A trailing assistant ``tool_use`` without its ``tool_result`` pair is dropped.

    Anthropic 400s on an unanswered tool_use; compacted sessions may replay
    one. Mirrors cc-switch's ``drop_incomplete_tool_turns``.
    """
    request = InternalRequest(
        model="kimi-k2.7-code",
        conversation=ConversationContext(
            messages=[
                Message(role="user", content=[TextBlock(text="run it")]),
                Message(
                    role="assistant",
                    content=[ToolUseBlock(id="toolu_1", name="exec", input={"input": "ls"})],
                ),
            ]
        ),
        params=GenerationParams(),
    )

    body = serializer.build_provider_request(request)

    assert [m["role"] for m in body["messages"]] == ["user"]


def test_messages_normalized_keeps_complete_tool_pair(serializer):
    """A complete tool_use → tool_result pair survives normalization."""
    request = InternalRequest(
        model="kimi-k2.7-code",
        conversation=ConversationContext(
            messages=[
                Message(role="user", content=[TextBlock(text="run it")]),
                Message(
                    role="assistant",
                    content=[ToolUseBlock(id="toolu_1", name="exec", input={"input": "ls"})],
                ),
                Message(
                    role="user",
                    content=[ToolResultBlock(tool_use_id="toolu_1", content="done")],
                ),
            ]
        ),
        params=GenerationParams(),
    )

    body = serializer.build_provider_request(request)

    roles = [m["role"] for m in body["messages"]]
    assert roles == ["user", "assistant", "user"]
    assert body["messages"][1]["content"][0]["type"] == "tool_use"
    assert body["messages"][2]["content"][0]["type"] == "tool_result"


def test_thinking_enabled_keeps_temperature_for_non_claude(serializer):
    """Non-Claude models behind a custom base_url accept temperature/top_p
    alongside thinking, so they are passed through unchanged."""
    request = InternalRequest(
        model="kimi-k2.7-code",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="hi")])]
        ),
        params=GenerationParams(
            temperature=0.7,
            top_p=0.9,
            thinking=ThinkingConfig(type="enabled", budget_tokens=4000),
        ),
    )

    body = serializer.build_provider_request(request)

    assert body["thinking"] == {"type": "enabled", "budget_tokens": 4000}
    assert body["temperature"] == 0.7
    assert body["top_p"] == 0.9


def test_thinking_enabled_drops_temperature_for_claude(serializer):
    """Claude models reject temperature/top_p while thinking is enabled.

    They are silently dropped instead of surfacing a 400. Mirrors cc-switch.
    """
    request = InternalRequest(
        model="claude-sonnet-4-6",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="hi")])]
        ),
        params=GenerationParams(
            temperature=0.7,
            top_p=0.9,
            thinking=ThinkingConfig(type="enabled", budget_tokens=4000),
        ),
    )

    body = serializer.build_provider_request(request)

    assert body["thinking"] == {"type": "enabled", "budget_tokens": 4000}
    assert "temperature" not in body
    assert "top_p" not in body


def test_forced_tool_choice_keeps_thinking_for_non_claude(serializer):
    """Non-Claude models accept forced tool_choice alongside thinking, so
    thinking is preserved."""
    request = InternalRequest(
        model="kimi-k2.7-code",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="hi")])]
        ),
        params=GenerationParams(
            temperature=0.7,
            thinking=ThinkingConfig(type="enabled", effort="high"),
        ),
        tools=[FunctionTool(name="exec", parameters={"type": "object"})],
        tool_choice=ToolChoice(mode="required"),
    )

    body = serializer.build_provider_request(request)

    assert body["tool_choice"] == {"type": "any"}
    assert body["thinking"]["type"] == "enabled"
    assert "output_config" in body
    assert body["temperature"] == 0.7


def test_forced_tool_choice_disables_thinking_for_claude(serializer):
    """Claude models reject a forced tool_choice while thinking is enabled.

    The caller's explicit tool constraint wins: thinking is disabled (and
    output_config dropped) instead of failing the request. Mirrors cc-switch.
    """
    request = InternalRequest(
        model="claude-sonnet-4-6",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="hi")])]
        ),
        params=GenerationParams(
            temperature=0.7,
            thinking=ThinkingConfig(type="enabled", effort="high"),
        ),
        tools=[FunctionTool(name="exec", parameters={"type": "object"})],
        tool_choice=ToolChoice(mode="required"),
    )

    body = serializer.build_provider_request(request)

    assert body["tool_choice"] == {"type": "any"}
    assert body["thinking"] == {"type": "disabled"}
    assert "output_config" not in body
    assert body["temperature"] == 0.7


def test_auto_tool_choice_keeps_thinking(serializer):
    """A non-forced tool_choice does not disable thinking."""
    request = InternalRequest(
        model="kimi-k2.7-code",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="hi")])]
        ),
        params=GenerationParams(thinking=ThinkingConfig(type="enabled", effort="high")),
        tools=[FunctionTool(name="exec", parameters={"type": "object"})],
        tool_choice=ToolChoice(mode="auto"),
    )

    body = serializer.build_provider_request(request)

    assert body["tool_choice"] == {"type": "auto"}
    assert body["thinking"]["type"] == "enabled"
    assert "output_config" in body


def test_parse_provider_response(serializer):
    """Test parsing provider response."""
    provider_response = {
        "id": "test-id",
        "model": "claude-3-opus",
        "content": [{"type": "text", "text": "Hello!"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }

    response = serializer.parse_provider_response(provider_response, model="claude-3-opus")

    assert response.id == "test-id"
    assert response.model == "claude-3-opus"
    assert len(response.output) == 1
    assert response.output[0].text == "Hello!"
    assert response.finish_reason == "stop"


def test_parse_thinking_block(serializer):
    """Test parsing thinking block."""
    blocks = serializer.parse_content_blocks([{"type": "thinking", "thinking": "Let me think..."}])

    assert len(blocks) == 1
    assert isinstance(blocks[0], ThinkingBlock)
    assert blocks[0].thinking == "Let me think..."


def test_round_trip(serializer):
    """Test round-trip: request -> build_provider_request -> parse_request."""
    original_data = {
        "model": "claude-3-opus",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hello"}],
    }

    request = serializer.parse_request(original_data)
    provider_body = serializer.build_provider_request(request)

    assert provider_body["model"] == "claude-3-opus"
    assert len(provider_body["messages"]) == 1


def test_parse_tool_use(serializer):
    """Test parsing tool use in assistant message."""
    data = {
        "model": "claude-3-opus",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me check that."},
                    {
                        "type": "tool_use",
                        "id": "toolu_123",
                        "name": "get_weather",
                        "input": {"location": "Boston"},
                    },
                ],
            }
        ],
    }

    request = serializer.parse_request(data)

    assert len(request.conversation.messages) == 1
    msg = request.conversation.messages[0]
    assert msg.role == "assistant"
    assert len(msg.content) == 2
    assert isinstance(msg.content[1], ToolUseBlock)
    assert msg.content[1].id == "toolu_123"
    assert msg.content[1].name == "get_weather"
    assert msg.content[1].input == {"location": "Boston"}


def test_format_response_with_thinking(serializer):
    """Test formatting response with thinking block."""
    response = InternalResponse(
        id="test-id",
        model="claude-3-opus",
        output=[
            ThinkingBlock(thinking="Let me think...", signature="sig123"),
            TextBlock(text="Hello!"),
        ],
    )

    result = serializer.format_response(response)

    assert len(result["content"]) == 2
    assert result["content"][0]["type"] == "thinking"
    assert result["content"][0]["thinking"] == "Let me think..."
    assert result["content"][1]["type"] == "text"
    assert result["content"][1]["text"] == "Hello!"


def test_map_finish_reason(serializer):
    """Test finish reason mapping."""
    assert serializer._map_finish_reason("stop") == "end_turn"
    assert serializer._map_finish_reason("tool_calls") == "tool_use"
    assert serializer._map_finish_reason("length") == "max_tokens"
    assert serializer._map_finish_reason("unknown") == "unknown"


def test_disable_parallel_tool_use_converts_to_parallel_tool_calls(serializer):
    """Test that disable_parallel_tool_use=true sets parallel_tool_calls=false for OpenAI."""
    data = {
        "model": "claude-3-opus",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hello"}],
        "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
    }

    request = serializer.parse_request(data)

    # Should set disable_parallel_tool_use=True in Anthropic params
    assert request.params.anthropic is not None
    assert request.params.anthropic.disable_parallel_tool_use is True
    # Should set parallel_tool_calls=False in extra for cross-provider routing
    assert request.extra.get("parallel_tool_calls") is False


def test_disable_parallel_tool_use_top_level_converts(serializer):
    """Test that top-level disable_parallel_tool_use=true sets parallel_tool_calls=false."""
    data = {
        "model": "claude-3-opus",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hello"}],
        "disable_parallel_tool_use": True,
    }

    request = serializer.parse_request(data)

    # Should set disable_parallel_tool_use=True in Anthropic params
    assert request.params.anthropic is not None
    assert request.params.anthropic.disable_parallel_tool_use is True
    # Should set parallel_tool_calls=False in extra
    assert request.extra.get("parallel_tool_calls") is False


def test_build_provider_request_with_disable_parallel_tool_use(serializer):
    """Test that disable_parallel_tool_use is built correctly in provider request."""
    from llm_proxy.models import AnthropicSpecificParams

    request = InternalRequest(
        model="claude-3-opus",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="Hello")])]
        ),
        params=GenerationParams(
            max_tokens=1024,
            anthropic=AnthropicSpecificParams(disable_parallel_tool_use=True),
        ),
    )

    body = serializer.build_provider_request(request)

    assert body["disable_parallel_tool_use"] is True


def test_parse_request_with_output_config(serializer):
    """Test parsing Anthropic request with output_config.effort."""
    data = {
        "model": "claude-opus-4-7",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hello"}],
        "output_config": {"effort": "medium"},
    }

    request = serializer.parse_request(data)

    assert request.model == "claude-opus-4-7"
    assert request.params.thinking is not None
    assert request.params.thinking.type == "enabled"
    assert request.params.thinking.effort == "medium"
    assert request.params.anthropic is not None
    assert request.params.anthropic.output_config == {"effort": "medium"}


def test_build_provider_request_with_output_config_effort(serializer):
    """Test building provider request from ThinkingConfig with effort."""
    from llm_proxy.models.types import ThinkingConfig

    request = InternalRequest(
        model="claude-opus-4-7",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="Hello")])]
        ),
        params=GenerationParams(
            max_tokens=1024,
            thinking=ThinkingConfig(type="enabled", effort="high"),
        ),
    )

    body = serializer.build_provider_request(request)

    assert body["output_config"] == {"effort": "high"}
    assert body["thinking"] == {"type": "enabled"}


def test_build_provider_request_with_output_config_max(serializer):
    """Test building provider request with max effort."""
    from llm_proxy.models.types import ThinkingConfig

    request = InternalRequest(
        model="claude-opus-4-7",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="Hello")])]
        ),
        params=GenerationParams(
            max_tokens=1024,
            thinking=ThinkingConfig(type="enabled", effort="max"),
        ),
    )

    body = serializer.build_provider_request(request)

    assert body["output_config"] == {"effort": "max"}
    assert body["thinking"] == {"type": "enabled"}


def test_round_trip_output_config(serializer):
    """Test round-trip of output_config.effort through parse and build."""
    data = {
        "model": "claude-opus-4-7",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hello"}],
        "output_config": {"effort": "xhigh"},
    }

    request = serializer.parse_request(data)
    body = serializer.build_provider_request(request)

    assert body["output_config"] == {"effort": "xhigh"}
    assert body["thinking"] == {"type": "enabled"}


def test_output_config_preserved_with_extra_fields(serializer):
    """Test that other fields in output_config are preserved."""
    request = InternalRequest(
        model="claude-opus-4-7",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="Hello")])]
        ),
        params=GenerationParams(
            max_tokens=1024,
            anthropic=AnthropicSpecificParams(
                output_config={"effort": "medium", "custom_key": "value"}
            ),
        ),
    )

    body = serializer.build_provider_request(request)

    # The anthropic-specific output_config should override the one from thinking
    assert body["output_config"] == {"effort": "medium", "custom_key": "value"}


def test_parse_request_with_context_management(serializer):
    """context_management is an Anthropic-only field (beta
    context-management-2025-06-27); it must be parsed into AnthropicSpecificParams
    and not leak into extra."""
    data = {
        "model": "claude-opus-4-7",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hello"}],
        "context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]},
    }

    request = serializer.parse_request(data)

    assert request.params.anthropic is not None
    assert request.params.anthropic.context_management == {
        "edits": [{"type": "clear_tool_uses_20250919"}]
    }
    # Must not also leak into extra.
    assert "context_management" not in (request.extra or {})


def test_build_provider_request_preserves_context_management(serializer):
    """Anthropic -> Anthropic must keep context_management in the body."""
    request = InternalRequest(
        model="claude-opus-4-7",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="Hello")])]
        ),
        params=GenerationParams(
            max_tokens=1024,
            anthropic=AnthropicSpecificParams(
                context_management={"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]}
            ),
        ),
    )

    body = serializer.build_provider_request(request)

    assert body["context_management"] == {
        "edits": [{"type": "clear_thinking_20251015", "keep": "all"}]
    }


def test_build_tools_with_web_search_tool(serializer):
    """Test _build_tools formats WebSearchTool for Anthropic API."""
    from llm_proxy.models.tools import WebSearchTool

    request = InternalRequest(
        model="claude-opus-4-7",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="Search for news")])]
        ),
        params=GenerationParams(max_tokens=1024),
        tools=[
            WebSearchTool(name="web_search", type="web_search_20250305"),
        ],
    )
    body = serializer.build_provider_request(request)

    assert "tools" in body
    assert len(body["tools"]) == 1
    assert body["tools"][0] == {"type": "web_search_20250305", "name": "web_search"}


def test_build_tools_with_dict_passthrough(serializer):
    """Test _build_tools passes through raw dict tools."""
    request = InternalRequest(
        model="claude-opus-4-7",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="Hello")])]
        ),
        params=GenerationParams(max_tokens=1024),
        tools=[
            {"type": "custom_tool", "name": "custom", "extra": "value"},
            FunctionTool(
                name="get_weather",
                parameters={"type": "object"},
            ),
        ],
    )
    body = serializer.build_provider_request(request)

    assert "tools" in body
    assert len(body["tools"]) == 2
    assert body["tools"][0] == {"type": "custom_tool", "name": "custom", "extra": "value"}
    assert body["tools"][1]["name"] == "get_weather"


def test_parse_web_search_tool_result_block(serializer):
    """parse_content_blocks must handle web_search_tool_result type.

    In multi-turn conversations, the client sends back conversation history
    containing web_search_tool_result blocks from the previous response.
    Without this parsing, those blocks are silently dropped, causing
    empty user messages that make downstream providers return 400.
    """
    from llm_proxy.models.content_blocks.anthropic_builtin import WebSearchToolResultBlock

    blocks = serializer.parse_content_blocks(
        [
            {
                "type": "web_search_tool_result",
                "tool_use_id": "toolu_ws_001",
                "content": [
                    {
                        "type": "web_search_result",
                        "url": "https://example.com",
                        "title": "Example",
                        "encrypted_content": "base64...",
                    },
                ],
                "is_error": False,
            },
        ]
    )

    assert len(blocks) == 1
    assert isinstance(blocks[0], WebSearchToolResultBlock)
    assert blocks[0].tool_use_id == "toolu_ws_001"
    assert blocks[0].is_error is False
    assert isinstance(blocks[0].content, list)
    assert len(blocks[0].content) == 1


def test_parse_web_search_result_block(serializer):
    """parse_content_blocks must handle web_search_result type.

    Individual search result items inside web_search_tool_result.content
    must parse into WebSearchResultContentBlock with url, title, and
    encoded_content fields preserved.
    """
    from llm_proxy.models.content_blocks.anthropic_builtin import WebSearchResultContentBlock

    blocks = serializer.parse_content_blocks(
        [
            {
                "type": "web_search_result",
                "url": "https://docs.example.com",
                "title": "Documentation",
                "encoded_content": "ZW5jcnlwdGVk...",
                "page_age": "2025-01-15",
            },
        ]
    )

    assert len(blocks) == 1
    assert isinstance(blocks[0], WebSearchResultContentBlock)
    assert blocks[0].url == "https://docs.example.com"
    assert blocks[0].title == "Documentation"
    assert blocks[0].encoded_content == "ZW5jcnlwdGVk..."
    assert blocks[0].page_age == "2025-01-15"


def test_parse_web_search_tool_result_with_error(serializer):
    """web_search_tool_result with is_error=True must be parsed correctly."""
    from llm_proxy.models.content_blocks.anthropic_builtin import WebSearchToolResultBlock

    blocks = serializer.parse_content_blocks(
        [
            {
                "type": "web_search_tool_result",
                "tool_use_id": "toolu_ws_err",
                "content": "Search failed: rate limited",
                "is_error": True,
            },
        ]
    )

    assert len(blocks) == 1
    assert isinstance(blocks[0], WebSearchToolResultBlock)
    assert blocks[0].tool_use_id == "toolu_ws_err"
    assert blocks[0].is_error is True
    assert blocks[0].content == "Search failed: rate limited"


def test_web_search_round_trip_through_parse_request(serializer):
    """Full request parsing must preserve web_search content in conversation history.

    Simulates a multi-turn scenario where a client sends an Anthropic message
    containing a web_search_tool_result block. The parsed InternalRequest must
    preserve this content so it can be converted to the downstream provider's format.
    """
    from llm_proxy.models.content_blocks.anthropic_builtin import WebSearchToolResultBlock

    data = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "Search for latest docs"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me search for that."},
                    {
                        "type": "server_tool_use",
                        "id": "toolu_ws_001",
                        "name": "web_search",
                        "input": {"query": "latest docs"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": "toolu_ws_001",
                        "content": [
                            {
                                "type": "web_search_result",
                                "url": "https://example.com",
                                "title": "Example",
                                "encrypted_content": "base64...",
                            },
                        ],
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Based on the search results, here's what I found...",
                    },
                ],
            },
        ],
    }

    request = serializer.parse_request(data)
    conv = request.conversation

    # Message 0: user query
    assert conv.messages[0].role == "user"
    assert len(conv.messages[0].content) == 1

    # Message 1: assistant with server_tool_use
    assert conv.messages[1].role == "assistant"
    assert len(conv.messages[1].content) == 2
    assert conv.messages[1].content[1].name == "web_search"

    # Message 2: user with web_search_tool_result (MUST NOT be empty/dropped)
    assert conv.messages[2].role == "user"
    assert len(conv.messages[2].content) == 1, (
        "web_search_tool_result was dropped from user message content"
    )
    assert isinstance(conv.messages[2].content[0], WebSearchToolResultBlock)
    ws_result = conv.messages[2].content[0]
    assert ws_result.tool_use_id == "toolu_ws_001"
    assert isinstance(ws_result.content, list)
    assert len(ws_result.content) == 1

    # Message 3: assistant text response
    assert conv.messages[3].role == "assistant"
    assert len(conv.messages[3].content) == 1


def test_web_search_options_from_openai_to_anthropic_tool(serializer):
    """web_search_options in OpenAI params should become a web_search tool.

    When an OpenAI protocol request includes web_search_options and routes
    to the Anthropic provider, the serializer must convert it to Anthropic's
    native web_search tool, preserving mappable fields like user_location.
    search_context_size is OpenAI-specific and has no Anthropic equivalent.
    """
    from llm_proxy.models import OpenAISpecificParams

    request = InternalRequest(
        model="claude-sonnet-4-6",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="latest news")])]
        ),
        params=GenerationParams(
            openai=OpenAISpecificParams(
                web_search_options={
                    "search_context_size": "medium",
                    "user_location": {
                        "type": "approximate",
                        "approximate": {
                            "country": "GB",
                            "city": "London",
                            "region": "London",
                            "timezone": "Europe/London",
                        },
                    },
                }
            ),
        ),
        tools=[],
    )
    body = serializer.build_provider_request(request)

    assert "tools" in body
    assert len(body["tools"]) == 1
    assert body["tools"][0]["type"] == "web_search_20250305"
    assert body["tools"][0]["name"] == "web_search"
    # user_location should be mapped from OpenAI nested format to Anthropic flat format
    assert body["tools"][0]["user_location"] == {
        "type": "approximate",
        "country": "GB",
        "city": "London",
        "region": "London",
        "timezone": "Europe/London",
    }
    # search_context_size is OpenAI-specific and is not passed to Anthropic
    assert "search_context_size" not in body["tools"][0]


def test_build_provider_request_degrades_audio_file_id_to_text(serializer):
    """AudioBlock with file_id should be degraded to text, not crash."""
    from llm_proxy.models.content_blocks import AudioBlock
    from llm_proxy.models.types import AudioSource

    request = InternalRequest(
        model="claude-sonnet-4-6",
        conversation=ConversationContext(
            messages=[
                Message(
                    role="user",
                    content=[
                        TextBlock(text="Listen to this"),
                        AudioBlock(
                            source=AudioSource(
                                type="file_id", data="file_abc123", media_type="audio/mpeg"
                            )
                        ),
                    ],
                )
            ]
        ),
        params=GenerationParams(max_tokens=1024),
    )
    body = serializer.build_provider_request(request)
    messages = body["messages"]
    assert len(messages) == 1
    content = messages[0]["content"]
    assert any(c.get("type") == "text" and "file_abc123" in c.get("text", "") for c in content)


def test_build_provider_request_audio_base64_and_url_still_works(serializer):
    """AudioBlock with base64 and url should still produce native audio blocks."""
    from llm_proxy.models.content_blocks import AudioBlock
    from llm_proxy.models.types import AudioSource

    request = InternalRequest(
        model="claude-sonnet-4-6",
        conversation=ConversationContext(
            messages=[
                Message(
                    role="user",
                    content=[
                        AudioBlock(
                            source=AudioSource(
                                type="base64", data="base64data", media_type="audio/wav"
                            )
                        ),
                        AudioBlock(
                            source=AudioSource(
                                type="url",
                                data="https://example.com/audio.wav",
                                media_type="audio/wav",
                            )
                        ),
                    ],
                )
            ]
        ),
        params=GenerationParams(max_tokens=1024),
    )
    body = serializer.build_provider_request(request)
    content = body["messages"][0]["content"]
    types = [c.get("type") for c in content]
    assert types.count("audio") == 2
    assert any(c["source"]["type"] == "base64" for c in content if c["type"] == "audio")
    assert any(c["source"]["type"] == "url" for c in content if c["type"] == "audio")


# ---------------------------------------------------------------------------
# Mid-conversation system message conversion tests
# ---------------------------------------------------------------------------


class TestMidConversationSystemConversion:
    """Anthropic's mid_conv_system content blocks must degrade correctly when routed
    to OpenAI-compatible providers, and system-role messages must convert to user
    with XML wrapping for providers that don't support native mid-conversation system."""

    @staticmethod
    def _make_unified_with_mid_conv_system():
        """Build an InternalRequest from Anthropic protocol input containing
        a mid_conv_system content block, simulating cross-protocol routing."""
        from llm_proxy.protocols.anthropic.serializer import AnthropicProtocolSerializer

        protocol = AnthropicProtocolSerializer()
        return protocol.parse_request(
            {
                "model": "claude-sonnet-4-6",
                "max_tokens": 1024,
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Got it, proceeding."}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "mid_conv_system",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "From now on, use only French.",
                                    },
                                ],
                            },
                        ],
                    },
                ],
            }
        )

    def test_mid_conv_system_block_preserved_in_anthropic_round_trip(self, serializer):
        """MidConversationSystemBlock should serialize back to mid_conv_system
        when the target provider is Anthropic (native support)."""
        request = self._make_unified_with_mid_conv_system()

        body = serializer.build_provider_request(request)
        messages = body["messages"]

        # Message 0: user "Hello"
        # Message 1: assistant
        # Message 2: user with mid_conv_system
        assert len(messages) == 3
        user_msg = messages[2]
        assert user_msg["role"] == "user"
        content_types = [c.get("type") for c in user_msg["content"]]
        assert "mid_conv_system" in content_types, (
            f"mid_conv_system block was not preserved: {content_types}"
        )
        mid_sys = [c for c in user_msg["content"] if c["type"] == "mid_conv_system"][0]
        assert mid_sys["content"][0]["text"] == "From now on, use only French."

    def test_mid_conv_system_block_degraded_for_openai_converter(self):
        """When routing Anthropic→OpenAI, MidConversationSystemBlock must be converted
        to plain text (content preserved, no XML wrapping at block level)."""
        from llm_proxy.serialization.openai.converter import format_conversation

        request = self._make_unified_with_mid_conv_system()

        openai_messages = format_conversation(request.conversation)

        # Messages should be: system→user "Hello"→assistant→user "From now on..."
        assert len(openai_messages) == 3
        user_msg = openai_messages[2]
        assert user_msg["role"] == "user"
        content = user_msg["content"]
        # Plain text, NOT XML-wrapped (block-level degradation preserves raw text)
        if isinstance(content, str):
            assert "From now on, use only French." in content
        else:
            texts = [c["text"] for c in content if c.get("type") == "text"]
            combined = " ".join(texts)
            assert "From now on, use only French." in combined

    def test_system_role_message_converts_to_user_xml_for_openai_provider(self):
        """When ConversationContext.messages contains a Message(role='system'),
        OpenAI provider serializer must also convert it to role='user' with
        <system-prompt> XML wrapping (existing behavior from converter.py)."""
        from llm_proxy.serialization.providers.chat_completions import OpenAIProviderSerializer

        provider = OpenAIProviderSerializer()
        request = InternalRequest(
            model="gpt-4",
            conversation=ConversationContext(
                messages=[
                    Message(role="user", content=[TextBlock(text="Hello")]),
                    Message(
                        role="system",
                        content=[TextBlock(text="Use only French from now on.")],
                    ),
                    Message(role="user", content=[TextBlock(text="How are you?")]),
                ]
            ),
            params=GenerationParams(max_tokens=1024),
        )

        body = provider.build_provider_request(request)
        messages = body["messages"]

        assert len(messages) == 3
        sys_msg = messages[1]
        assert sys_msg["role"] == "user", (
            f"Expected role='user' for system message, got '{sys_msg['role']}'"
        )
        assert "<system-prompt>" in sys_msg["content"]
        assert "Use only French from now on." in sys_msg["content"]
        assert "</system-prompt>" in sys_msg["content"]

    def test_non_system_roles_unchanged_by_anthropic_serializer(self, serializer):
        """Messages with user/assistant/tool roles must not be affected by
        the system role conversion logic."""
        request = InternalRequest(
            model="claude-sonnet-4-6",
            conversation=ConversationContext(
                messages=[
                    Message(role="user", content=[TextBlock(text="Hi")]),
                    Message(role="assistant", content=[TextBlock(text="Hello!")]),
                ]
            ),
            params=GenerationParams(max_tokens=1024),
        )

        body = serializer.build_provider_request(request)
        messages = body["messages"]

        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"


# ---------------------------------------------------------------------------
# Streaming chunk converter regression tests
# ---------------------------------------------------------------------------


class TestAnthropicChunkConverter:
    """Regression tests for AnthropicChunkConverter event handlers."""

    @pytest.fixture
    def converter(self):
        """Create a fresh AnthropicChunkConverter."""
        return AnthropicChunkConverter(model="claude-sonnet-4-7", request_id="msg_01")

    def test_handle_ping_returns_none(self, converter):
        """ping keepalive events must be swallowed without raising."""
        assert converter.convert_chunk({"type": "ping"}) is None

    def test_thinking_delta_emits_reasoning_content(self, converter):
        """thinking_delta must read the 'thinking' field, not 'text'."""
        chunk = converter.convert_chunk(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "step one"},
            }
        )
        assert chunk is not None
        assert chunk["choices"][0]["delta"]["reasoning_content"] == "step one"

    def test_signature_delta_buffered_until_content_block_stop(self, converter):
        """signature_delta accumulates and is emitted on content_block_stop."""
        converter.convert_chunk(
            {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking"}}
        )
        converter.convert_chunk(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "hmm"},
            }
        )
        converter.convert_chunk(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "sigA"},
            }
        )
        stop_chunk = converter.convert_chunk({"type": "content_block_stop", "index": 0})
        assert stop_chunk is not None
        assert stop_chunk["choices"][0]["delta"]["reasoning_signature"] == "sigA"

    def test_usage_normalized_into_openai_dialect_details(self, converter):
        """Anthropic-native cache/thinking counters are normalized into the
        OpenAI-dialect details objects so canonical-channel consumers read a
        single dialect; the lossless passthrough keys are preserved."""
        converter.convert_chunk(
            {
                "type": "message_start",
                "message": {
                    "id": "m1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-x",
                    "content": [],
                    "usage": {"input_tokens": 10, "cache_read_input_tokens": 70},
                },
            }
        )
        converter.convert_chunk(
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {
                    "output_tokens": 5,
                    "output_tokens_details": {"thinking_tokens": 7},
                },
            }
        )
        final = converter.convert_chunk({"type": "message_stop"})
        assert final is not None
        usage = final["usage"]
        assert usage["prompt_tokens"] == 80  # 10 input + 70 cache read
        assert usage["prompt_tokens_details"] == {"cached_tokens": 70}
        assert usage["completion_tokens_details"] == {"reasoning_tokens": 7}
        # Lossless Anthropic-native passthrough is preserved.
        assert usage["cache_read_input_tokens"] == 70
        assert usage["output_tokens_details"] == {"thinking_tokens": 7}

    def test_thinking_and_signature_full_flow(self, converter):
        """Multiple thinking deltas plus signature are all preserved."""
        converter.convert_chunk(
            {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking"}}
        )
        for fragment in ("let me", " think"):
            converter.convert_chunk(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": fragment},
                }
            )
        converter.convert_chunk(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "signature123"},
            }
        )
        sig_chunk = converter.convert_chunk({"type": "content_block_stop", "index": 0})

        reasoning_chunks = [
            converter.convert_chunk(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": fragment},
                }
            )
            for fragment in ("let me", " think")
        ]

        assert [c["choices"][0]["delta"]["reasoning_content"] for c in reasoning_chunks] == [
            "let me",
            " think",
        ]
        assert sig_chunk["choices"][0]["delta"]["reasoning_signature"] == "signature123"


def test_format_response_clamps_negative_cache_subtraction(serializer):
    """Regression: a provider violating the "input_tokens includes cache"
    invariant must clamp at 0 instead of emitting a negative token count."""
    from llm_proxy.models.types import Usage

    response = InternalResponse(
        id="test-id",
        model="claude-3-opus",
        output=[TextBlock(text="Hello!")],
        usage=Usage(
            input_tokens=100,
            output_tokens=5,
            cache_read_input_tokens=1000,
            cache_creation_input_tokens=200,
        ),
    )

    result = serializer.format_response(response)

    assert result["usage"]["input_tokens"] == 0
    assert result["usage"]["cache_read_input_tokens"] == 1000
    assert result["usage"]["cache_creation_input_tokens"] == 200
