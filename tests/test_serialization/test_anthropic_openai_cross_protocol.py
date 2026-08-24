"""Regression tests for OpenAI/Responses protocol -> Anthropic provider conversions."""

from typing import Any

import pytest

from llm_proxy.models import (
    AnthropicSpecificParams,
    ConversationContext,
    GenerationParams,
    InternalRequest,
    Message,
    OpenAISpecificParams,
    RequestMetadata,
    SystemMessage,
    TextBlock,
)
from llm_proxy.models.content_blocks.anthropic_builtin import CacheControl
from llm_proxy.serialization.anthropic.serializer import AnthropicProviderSerializer


@pytest.fixture
def serializer():
    """Create an Anthropic provider serializer."""
    return AnthropicProviderSerializer()


def _single_user_request(**kwargs: Any) -> InternalRequest:
    """Build an InternalRequest with a single user 'hi' message and defaults."""
    return InternalRequest(
        model=kwargs.get("model", "claude-3-5-sonnet"),
        conversation=kwargs.get(
            "conversation",
            ConversationContext(messages=[Message(role="user", content=[TextBlock(text="hi")])]),
        ),
        params=kwargs.get("params", GenerationParams()),
        tools=kwargs.get("tools"),
        tool_choice=kwargs.get("tool_choice"),
        metadata=kwargs.get("metadata", RequestMetadata()),
        extra=kwargs.get("extra", {}),
    )


def test_developer_messages_merge_into_anthropic_system(serializer):
    """OpenAI ``developer`` messages must not leak into the Anthropic message list.

    Anthropic has no ``developer`` role; developer instructions (e.g. from
    Codex via /v1/responses) are merged into the top-level ``system``
    parameter, mirroring ``normalize_developer_roles``. Regression test for
    the upstream 400 ``role 'developer' is not allowed``.
    """
    body = serializer.build_provider_request(
        _single_user_request(
            conversation=ConversationContext(
                system_messages=[SystemMessage.from_text(role="system", text="Top-level.")],
                messages=[
                    Message(role="developer", content=[TextBlock(text="Dev prompt 1")]),
                    Message(role="user", content=[TextBlock(text="User 1")]),
                    Message(role="developer", content=[TextBlock(text="Dev prompt 2")]),
                ],
            ),
        )
    )

    roles = [m["role"] for m in body["messages"]]
    assert "developer" not in roles
    assert roles == ["user"]
    system_texts = [b["text"] for b in body["system"]]
    assert "Top-level." in system_texts
    assert any("Dev prompt 1" in t and "Dev prompt 2" in t for t in system_texts)


def test_developer_only_request_still_emits_system(serializer):
    """A request with only developer messages gets them as the system param."""
    body = serializer.build_provider_request(
        _single_user_request(
            conversation=ConversationContext(
                messages=[
                    Message(role="developer", content=[TextBlock(text="Dev only")]),
                    Message(role="user", content=[TextBlock(text="hi")]),
                ],
            ),
        )
    )

    assert [m["role"] for m in body["messages"]] == ["user"]
    assert body["system"] == [{"type": "text", "text": "Dev only"}]


def test_openai_user_and_metadata_reach_anthropic_metadata(serializer):
    """OpenAI ``user`` + request-level ``metadata`` merge into Anthropic ``metadata``.

    The Anthropic Messages API only accepts ``metadata.user_id``, so non-``user_id``
    OpenAI metadata keys (e.g. ``session_id``) are dropped to avoid an upstream 400.
    Priority: explicit Anthropic ``metadata.user_id`` > OpenAI ``user`` field.
    """
    body = serializer.build_provider_request(
        _single_user_request(
            params=GenerationParams(
                openai=OpenAISpecificParams(metadata={"session_id": "sess-1"}),
            ),
            metadata=RequestMetadata(user="user-123"),
        )
    )

    # session_id is dropped (Anthropic only accepts user_id); user fills user_id.
    assert body["metadata"] == {"user_id": "user-123"}


def test_anthropic_metadata_user_id_wins_over_openai_user(serializer):
    """Explicit Anthropic ``metadata.user_id`` takes priority over OpenAI ``user``."""
    body = serializer.build_provider_request(
        _single_user_request(
            params=GenerationParams(
                anthropic=AnthropicSpecificParams(metadata={"user_id": "anthropic-user"}),
            ),
            metadata=RequestMetadata(user="openai-user"),
        )
    )

    assert body["metadata"] == {"user_id": "anthropic-user"}


def test_openai_service_tier_maps_to_anthropic(serializer):
    """OpenAI ``service_tier`` values map to Anthropic ``service_tier`` values."""
    body = serializer.build_provider_request(
        _single_user_request(
            params=GenerationParams(
                openai=OpenAISpecificParams(service_tier="auto"),
            ),
        )
    )

    assert body["service_tier"] == "auto"


def test_openai_service_tier_default_maps_to_standard_only(serializer):
    """OpenAI ``service_tier: default`` maps to Anthropic ``standard_only``."""
    body = serializer.build_provider_request(
        _single_user_request(
            params=GenerationParams(
                openai=OpenAISpecificParams(service_tier="default"),
            ),
        )
    )

    assert body["service_tier"] == "standard_only"


def test_anthropic_service_tier_takes_priority_over_openai(serializer):
    """Anthropic-native ``service_tier`` wins over OpenAI value."""
    body = serializer.build_provider_request(
        _single_user_request(
            params=GenerationParams(
                anthropic=AnthropicSpecificParams(service_tier="auto"),
                openai=OpenAISpecificParams(service_tier="default"),
            ),
        )
    )

    assert body["service_tier"] == "auto"


def test_openai_cache_control_on_text_block(serializer):
    """OpenAI text content parts with ``cache_control`` preserve cache control."""
    body = serializer.build_provider_request(
        _single_user_request(
            conversation=ConversationContext(
                messages=[
                    Message(
                        role="user",
                        content=[
                            TextBlock(
                                text="hi",
                                cache_control=CacheControl(type="ephemeral"),
                            )
                        ],
                    )
                ]
            ),
        )
    )

    assert body["messages"][0]["content"][0] == {
        "type": "text",
        "text": "hi",
        "cache_control": {"type": "ephemeral"},
    }


def test_openai_prompt_cache_breakpoint_on_text_block():
    """OpenAI ``prompt_cache_breakpoint`` becomes Anthropic ``cache_control``."""
    # The OpenAI chat parser converts ``prompt_cache_breakpoint`` to a
    # TextBlock with an ephemeral CacheControl marker.
    from llm_proxy.serialization.content_parsers import parse_text_block

    block = parse_text_block({"type": "text", "text": "hi", "prompt_cache_breakpoint": True})
    assert block is not None
    assert block.cache_control is not None
    assert block.cache_control.type == "ephemeral"


def test_openai_parallel_tool_calls_false_reaches_tool_choice(serializer):
    """OpenAI ``parallel_tool_calls: false`` becomes ``tool_choice.disable_parallel_tool_use``.

    It must not leak as a top-level body field.
    """
    from llm_proxy.models.tools import FunctionTool

    body = serializer.build_provider_request(
        _single_user_request(
            params=GenerationParams(
                openai=OpenAISpecificParams(parallel_tool_calls=False),
            ),
            tools=[
                FunctionTool(
                    name="get_weather",
                    description="weather tool",
                    parameters={"type": "object"},
                )
            ],
            extra={"disable_parallel_tool_use": True},
        )
    )

    assert body["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    assert "disable_parallel_tool_use" not in body


def test_openai_parallel_tool_calls_false_dropped_without_tools(serializer):
    """``parallel_tool_calls: false`` is discarded when no tools are present."""
    body = serializer.build_provider_request(
        _single_user_request(
            params=GenerationParams(
                openai=OpenAISpecificParams(parallel_tool_calls=False),
            ),
            extra={"disable_parallel_tool_use": True},
        )
    )

    assert "tool_choice" not in body
    assert "disable_parallel_tool_use" not in body


def test_openai_parallel_tool_calls_false_merges_with_existing_tool_choice(
    serializer,
):
    """Disable-parallel flag merges into an existing tool_choice dict."""
    from llm_proxy.models.tools import ToolChoice

    body = serializer.build_provider_request(
        _single_user_request(
            params=GenerationParams(
                openai=OpenAISpecificParams(parallel_tool_calls=False),
            ),
            tools=[
                {
                    "name": "get_weather",
                    "description": "weather tool",
                    "parameters": {"type": "object"},
                }
            ],
            tool_choice=ToolChoice(mode="auto"),
            extra={"disable_parallel_tool_use": True},
        )
    )

    assert body["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}


def test_responses_extra_fields_do_not_leak_into_anthropic_body(serializer):
    """Responses-only fields must not leak as top-level Anthropic body keys."""
    extra: dict[str, Any] = {
        "reasoning": {"effort": "low"},
        "previous_response_id": "resp_1",
        "truncation": "auto",
        "include": ["reasoning.encrypted_content"],
        "background": True,
        "max_tool_calls": 10,
        "_system_blocks": [{"type": "text", "text": "sys"}],
        "disable_parallel_tool_use": True,
        "parallel_tool_calls": False,
    }
    body = serializer.build_provider_request(
        InternalRequest(
            model="claude-3-5-sonnet",
            conversation=ConversationContext(
                system_messages=[SystemMessage.from_text(role="system", text="sys")],
                messages=[Message(role="user", content=[TextBlock(text="hi")])],
            ),
            extra=extra,
        )
    )

    assert "reasoning" not in body
    assert "previous_response_id" not in body
    assert "truncation" not in body
    assert "include" not in body
    assert "background" not in body
    assert "max_tool_calls" not in body
    assert "_system_blocks" not in body
    assert "disable_parallel_tool_use" not in body
    assert "parallel_tool_calls" not in body

    # System blocks should be consumed by the system-building logic, not leaked.
    assert body["system"] == [{"type": "text", "text": "sys"}]


def test_unknown_extra_fields_still_pass_through(serializer):
    """Legitimate Anthropic extra fields (e.g. beta keys) are still forwarded."""
    body = serializer.build_provider_request(
        _single_user_request(
            extra={"speed": "fast"},
        )
    )

    assert body["speed"] == "fast"


def test_openai_function_role_maps_to_anthropic_tool_result(serializer):
    """Deprecated OpenAI ``role: function`` becomes Anthropic user + tool_result.

    When the tool_result is orphaned (no preceding assistant ``tool_use``),
    message normalization drops it and inserts a placeholder user message —
    Anthropic 400s on both orphaned tool_results and empty message lists.
    """
    body = serializer.build_provider_request(
        InternalRequest(
            model="claude-3-5-sonnet",
            conversation=ConversationContext(
                messages=[
                    Message(
                        role="function",
                        content=[TextBlock(text="sunny")],
                        name="get_weather",
                    )
                ]
            ),
        )
    )

    assert body["messages"] == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "(continuing the conversation)"}],
        }
    ]


def test_unwrap_custom_tool_arguments_shapes():
    """The custom-tool bridge wrappers are unwrapped; other shapes pass through.

    The Anthropic bridge declares a single ``input`` property (legacy
    ``content`` wrapper still unwrapped for backward compatibility). A
    ``command`` key is NOT unwrapped — a custom tool whose native input
    legitimately contains ``command`` must receive it verbatim.
    """
    from llm_proxy.protocols.openresponses.serializer import (
        unwrap_custom_tool_arguments,
    )

    # Bridge wrappers.
    assert unwrap_custom_tool_arguments('{"input": "ls -la"}') == "ls -la"
    assert unwrap_custom_tool_arguments('{"content": "legacy"}') == "legacy"
    # Legacy raw-input wrapper from history blocks (models may still echo it).
    assert unwrap_custom_tool_arguments('{"value": "raw js"}') == "raw js"
    # Plain JSON string (string input_schema).
    assert unwrap_custom_tool_arguments('"plain string"') == "plain string"
    # Native passthrough shapes are returned verbatim.
    assert (
        unwrap_custom_tool_arguments('{"patch": "*** Begin Patch"}')
        == '{"patch": "*** Begin Patch"}'
    )
    # Misfire protection: ``command`` is not a bridge wrapper key.
    assert unwrap_custom_tool_arguments('{"command": "ls"}') == '{"command": "ls"}'
    # Non-JSON and empty input fall through unchanged.
    assert unwrap_custom_tool_arguments("not json") == "not json"
    assert unwrap_custom_tool_arguments("") == ""
