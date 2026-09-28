"""Regression tests for Anthropic /v1/messages wire compatibility fixes.

Covers gaps found by an audit against the official Anthropic Messages API:
- top-level ``container`` as ContainerParams object (skills beta)
- ``max_tokens: 0`` (cache pre-warm) must reach the upstream unmodified
- server-side tool result blocks survive a full internal round-trip
- unknown content block types fall back to raw passthrough, not drop
- tool_use / server_tool_use / tool_result fidelity (caller, toolset_name)
- image ``transformations`` and official ``tool_reference`` shape
- non-streaming response: stop_details, usage.service_tier,
  usage.output_tokens_details
- streaming: citations_delta and raw server-tool-result blocks pass through
  the converter → transformer chain, error events raise ProviderError
- document / search_result citations config + context survive the round-trip
- container_upload cache_control round-trip
- streaming: container info in message_delta passes the converter →
  transformer chain
- body-level ``betas`` merges into the ``anthropic-beta`` header
- legacy ``output_format`` body field aliases ``output_config.format``
- system text blocks keep per-block citations (via ``_system_blocks`` stash)
"""

import pytest

from llm_proxy.core.exceptions import ProviderError
from llm_proxy.protocols.anthropic.handler import anthropic_protocol
from llm_proxy.protocols.anthropic.schemas import MessagesRequest
from llm_proxy.protocols.anthropic.serializer import AnthropicProtocolSerializer
from llm_proxy.protocols.anthropic.streaming import AnthropicStreamingTransformer
from llm_proxy.providers.anthropic.client_headers import merge_client_headers
from llm_proxy.serialization.anthropic.serializer import AnthropicProviderSerializer
from llm_proxy.serialization.anthropic.streaming_converter import AnthropicChunkConverter
from llm_proxy.serialization.context import BuildContext


@pytest.fixture
def protocol() -> AnthropicProtocolSerializer:
    return AnthropicProtocolSerializer()


@pytest.fixture
def provider() -> AnthropicProviderSerializer:
    return AnthropicProviderSerializer()


def build(provider: AnthropicProviderSerializer, request_data: dict) -> dict:
    request = AnthropicProtocolSerializer().parse_request(request_data)
    return provider.build_provider_request(request, BuildContext(model=request_data["model"]))


def test_container_accepts_params_object():
    """Official ``container`` is ``string | ContainerParams``; object must parse."""
    MessagesRequest(
        model="claude-sonnet-4-5",
        max_tokens=10,
        messages=[{"role": "user", "content": "x"}],
        container={
            "id": "c_1",
            "skills": [{"skill_id": "pdf", "type": "anthropic", "version": "latest"}],
        },
    )


def test_max_tokens_zero_reaches_upstream(provider):
    body = build(
        provider, {"model": "m", "max_tokens": 0, "messages": [{"role": "user", "content": "warm"}]}
    )
    assert body["max_tokens"] == 0


def _thinking_request(model: str, max_tokens: int | None, budget_tokens: int | None = None) -> dict:
    thinking: dict = {"type": "enabled"}
    if budget_tokens is not None:
        thinking["budget_tokens"] = budget_tokens
    data: dict = {
        "model": model,
        "messages": [{"role": "user", "content": "x"}],
        "thinking": thinking,
    }
    if max_tokens is not None:
        data["max_tokens"] = max_tokens
    return data


class TestThinkingBudgetValidation:
    """thinking.budget_tokens bounds are enforced provider-side (Anthropic's
    API imposes >= 1024 and < max_tokens), gated on Claude models so
    third-party Anthropic-compatible upstreams keep accepting what they
    previously did."""

    def test_budget_below_minimum_rejected(self, provider):
        from llm_proxy.core.exceptions import ValidationError

        with pytest.raises(ValidationError, match=">= 1024"):
            build(provider, _thinking_request("claude-sonnet-4-5", 2000, budget_tokens=1000))

    def test_budget_at_minimum_accepted(self, provider):
        body = build(provider, _thinking_request("claude-sonnet-4-5", 2000, budget_tokens=1024))
        assert body["thinking"]["budget_tokens"] == 1024

    def test_budget_at_max_tokens_rejected(self, provider):
        from llm_proxy.core.exceptions import ValidationError

        with pytest.raises(ValidationError, match="less than max_tokens"):
            build(provider, _thinking_request("claude-sonnet-4-5", 2000, budget_tokens=2000))

    def test_budget_none_accepted(self, provider):
        body = build(provider, _thinking_request("claude-sonnet-4-5", 2000))
        assert body["thinking"]["type"] == "enabled"

    def test_budget_below_minimum_accepted_for_non_claude_model(self, provider):
        """The 1024 floor is Anthropic-API-specific: compatible third-party
        upstreams (DeepSeek, GLM, Kimi Code, ...) may not impose it."""
        body = build(provider, _thinking_request("deepseek-chat", 2000, budget_tokens=500))
        assert body["thinking"]["budget_tokens"] == 500

    def test_budget_below_minimum_accepted_for_non_thinking_request(self, provider):
        """The floor only applies while thinking is enabled (``adaptive`` or
        ``enabled``); a disabled thinking block passes through."""
        data = _thinking_request("claude-sonnet-4-5", 2000, budget_tokens=500)
        data["thinking"]["type"] = "disabled"
        assert build(provider, data)


def test_server_tool_results_survive_roundtrip(protocol, provider):
    body = build(
        provider,
        {
            "model": "m",
            "max_tokens": 1,
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "server_tool_use",
                            "id": "s1",
                            "name": "web_fetch",
                            "input": {},
                            "caller": {"type": "code_execution_20250825", "tool_id": "srvtoolu_9"},
                        },
                        {
                            "type": "server_tool_use",
                            "id": "s2",
                            "name": "code_execution",
                            "input": {},
                        },
                        {"type": "tool_use", "id": "tu1", "name": "n", "input": {}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "web_fetch_tool_result",
                            "tool_use_id": "s1",
                            "content": "x",
                            "caller": {"type": "code_execution_20250825", "tool_id": "srvtoolu_9"},
                        },
                        {
                            "type": "code_execution_tool_result",
                            "tool_use_id": "s2",
                            "content": "y",
                            "is_error": True,
                            "caller": {"type": "code_execution_20250825", "tool_id": "srvtoolu_2"},
                        },
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu1",
                            "content": "ok",
                            "toolset_name": "fam",
                        },
                        {"type": "browser_state", "tabs": []},
                    ],
                },
            ],
        },
    )
    types = [c["type"] for m in body["messages"] for c in m["content"]]
    assert {
        "web_fetch_tool_result",
        "code_execution_tool_result",
        "tool_result",
        "browser_state",
    } <= set(types)
    server_uses = [
        c for m in body["messages"] for c in m["content"] if c.get("type") == "server_tool_use"
    ]
    # ``caller`` is official on server_tool_use; ``toolset_name`` is not.
    assert any(c.get("caller", {}).get("type") == "code_execution_20250825" for c in server_uses)
    assert all("toolset_name" not in c for c in server_uses)
    web_fetch_results = [
        c
        for m in body["messages"]
        for c in m["content"]
        if c.get("type") == "web_fetch_tool_result"
    ]
    exec_results = [
        c
        for m in body["messages"]
        for c in m["content"]
        if c.get("type") == "code_execution_tool_result"
    ]
    # ``caller`` is official on web_fetch results only; ``is_error`` is not
    # part of any server tool result block shape.
    assert any(
        c.get("caller") == {"type": "code_execution_20250825", "tool_id": "srvtoolu_9"}
        for c in web_fetch_results
    )
    assert all("caller" not in c and "is_error" not in c for c in exec_results)
    tool_results = [
        c for m in body["messages"] for c in m["content"] if c.get("type") == "tool_result"
    ]
    assert any(c.get("toolset_name") == "fam" for c in tool_results)


def test_unknown_block_falls_back_to_raw_passthrough(protocol, provider):
    body = build(
        provider,
        {
            "model": "m",
            "max_tokens": 1,
            "messages": [
                {"role": "user", "content": [{"type": "some_future_block", "payload": 1}]}
            ],
        },
    )
    assert body["messages"][0]["content"][0] == {"type": "some_future_block", "payload": 1}


def test_tool_reference_official_shape(protocol, provider):
    body = build(
        provider,
        {
            "model": "m",
            "max_tokens": 1,
            "messages": [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "t", "name": "n", "input": {}}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t",
                            "content": [
                                {
                                    "type": "tool_reference",
                                    "tool_name": "my_tool",
                                    "cache_control": {"type": "ephemeral"},
                                },
                            ],
                        }
                    ],
                },
            ],
        },
    )
    tool_result = [
        c for m in body["messages"] for c in m["content"] if c.get("type") == "tool_result"
    ][0]
    ref = [
        c
        for c in tool_result["content"]
        if isinstance(c, dict) and c.get("type") == "tool_reference"
    ][0]
    assert ref["tool_name"] == "my_tool"
    assert ref["cache_control"] == {"type": "ephemeral"}
    assert "tool_id" not in ref


def test_image_transformations_roundtrip(protocol, provider):
    body = build(
        provider,
        {
            "model": "m",
            "max_tokens": 1,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": "x"},
                            "transformations": {"oversized_image": "error"},
                        }
                    ],
                }
            ],
        },
    )
    assert body["messages"][0]["content"][0]["transformations"] == {"oversized_image": "error"}


def test_non_streaming_response_native_fields(protocol, provider):
    raw = {
        "id": "1",
        "type": "message",
        "role": "assistant",
        "model": "claude-x",
        "content": [{"type": "text", "text": "Hi"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "stop_details": {"type": "refusal", "category": "cyber"},
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "service_tier": "standard",
            "output_tokens_details": {"thinking_tokens": 3},
            "cache_read_input_tokens": 4,
            "cache_creation_input_tokens": 2,
        },
    }
    out = protocol.format_response(provider.parse_provider_response(raw, model="claude-x"))
    assert out["stop_details"] == {"type": "refusal", "category": "cyber"}
    # usage fold/restore invariant: internal total (10+4+2) re-splits into the
    # official wire shape (input_tokens excludes cache tokens)
    assert out["usage"]["input_tokens"] == 10
    assert out["usage"]["cache_read_input_tokens"] == 4
    assert out["usage"]["cache_creation_input_tokens"] == 2


def _run_stream(events: list[dict]) -> str:
    converter = AnthropicChunkConverter(model="claude-x", request_id="msg_1")
    transformer = AnthropicStreamingTransformer(model="claude-x", request_id="msg_1")
    frames: list[str] = []
    for event in events:
        chunk = converter.convert_chunk(event)
        if chunk is not None:
            frames.append(transformer.transform(dict(chunk)) or "")
    frames.append(transformer.finalize())
    return "".join(frames)


def test_stream_citations_delta_passthrough():
    sse = _run_stream(
        [
            {"type": "message_start", "message": {"id": "m", "usage": {"input_tokens": 1}}},
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "citations_delta",
                    "citation": {
                        "type": "char_location",
                        "cited_text": "T",
                        "document_index": 0,
                        "start_char_index": 0,
                        "end_char_index": 1,
                    },
                },
            },
            {"type": "message_stop"},
        ]
    )
    assert '"citations_delta"' in sse
    assert '"cited_text":"T"' in sse


def test_stream_web_search_tool_result_block_passthrough():
    sse = _run_stream(
        [
            {"type": "message_start", "message": {"id": "m", "usage": {}}},
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srvtoolu_1",
                    "content": [
                        {
                            "title": "R",
                            "url": "u",
                            "type": "web_search_result",
                            "encrypted_content": "e",
                        }
                    ],
                },
            },
            {"type": "message_stop"},
        ]
    )
    assert '"web_search_tool_result"' in sse
    assert '"srvtoolu_1"' in sse


def test_stream_error_event_raises_provider_error():
    with pytest.raises(ProviderError) as exc_info:
        AnthropicChunkConverter().convert_chunk(
            {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
        )
    assert exc_info.value.error_type == "overloaded_error"


def test_stream_stop_sequence_and_usage_details():
    sse = _run_stream(
        [
            {
                "type": "message_start",
                "message": {"id": "m", "usage": {"input_tokens": 10, "cache_read_input_tokens": 5}},
            },
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": "STOP"},
                "usage": {
                    "output_tokens": 7,
                    "service_tier": "standard",
                    "output_tokens_details": {"thinking_tokens": 3},
                },
            },
            {"type": "message_stop"},
        ]
    )
    assert '"stop_sequence":"STOP"' in sse
    assert '"thinking_tokens":3' in sse
    # service_tier belongs to the full Usage object (message_start), not to
    # the terminal MessageDeltaUsage — it must not reach the stream here.
    assert '"service_tier"' not in sse
    # cache keys ride the terminal message_delta usage (MessageDeltaUsage)
    assert '"cache_read_input_tokens":5' in sse


def test_header_merge_matrix():
    # Claude Code: marker injected, user-profile-id forwarded
    cc = {"Content-Type": "application/json", "anthropic-version": "2023-06-01", "x-api-key": "k"}
    merge_client_headers(
        cc,
        {
            "user-agent": "claude-cli/2.1",
            "anthropic-beta": "context-management-2025-06-27",
            "anthropic-user-profile-id": "u-1",
            "anthropic-version": "2023-06-01",
        },
    )
    assert cc["anthropic-beta"] == "claude-code-20250219,context-management-2025-06-27"
    assert cc["anthropic-user-profile-id"] == "u-1"

    # Plain SDK client: beta list untouched, no marker injected
    plain = {"anthropic-version": "2023-06-01", "x-api-key": "k"}
    merge_client_headers(
        plain, {"user-agent": "anthropic-sdk-python/0.60", "anthropic-beta": "pdfs-2024-09-25"}
    )
    assert plain["anthropic-beta"] == "pdfs-2024-09-25"

    # Newer client wire version flows through
    upgraded = {"anthropic-version": "2023-06-01", "x-api-key": "k"}
    merge_client_headers(upgraded, {"user-agent": "other/1", "anthropic-version": "2030-01-01"})
    assert upgraded["anthropic-version"] == "2030-01-01"
    assert "anthropic-beta" not in upgraded


def test_document_citations_context_roundtrip(protocol, provider):
    body = build(
        provider,
        {
            "model": "m",
            "max_tokens": 1,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": "x",
                            },
                            "citations": {"enabled": True},
                            "context": "budget context",
                            "title": "t",
                        }
                    ],
                }
            ],
        },
    )
    doc = body["messages"][0]["content"][0]
    assert doc["citations"] == {"enabled": True}
    assert doc["context"] == "budget context"


def test_search_result_citations_roundtrip(protocol, provider):
    body = build(
        provider,
        {
            "model": "m",
            "max_tokens": 1,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "search_result",
                            "source": "https://example.com",
                            "title": "R",
                            "content": [{"type": "text", "text": "c"}],
                            "citations": {"enabled": False},
                        }
                    ],
                }
            ],
        },
    )
    search = body["messages"][0]["content"][0]
    assert search["citations"] == {"enabled": False}
    assert search["content"][0]["text"] == "c"


def test_container_upload_cache_control_roundtrip(protocol, provider):
    body = build(
        provider,
        {
            "model": "m",
            "max_tokens": 1,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "container_upload",
                            "file_id": "f",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ],
        },
    )
    block = body["messages"][0]["content"][0]
    assert block["cache_control"] == {"type": "ephemeral"}


def test_system_block_citations_preserved(provider):
    citation = {
        "type": "char_location",
        "cited_text": "T",
        "document_index": 0,
        "document_title": None,
        "start_char_index": 0,
        "end_char_index": 1,
    }
    body = build(
        provider,
        {
            "model": "m",
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "q"}],
            "system": [{"type": "text", "text": "S", "citations": [citation]}],
        },
    )
    block = body["system"][0]
    assert block["citations"] == [citation]
    assert block["text"] == "S"


def test_body_betas_merge_into_beta_header(protocol):
    from llm_proxy.providers.anthropic.client_headers import (
        clear_client_headers,
        get_client_headers,
    )

    clear_client_headers()
    try:
        # Body-level ``betas`` is merged by the protocol's on_parse_request
        # hook (parse_request itself stays a pure wire-to-internal
        # conversion, ADR-0009).
        anthropic_protocol.on_parse_request(
            {
                "model": "m",
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "x"}],
                "betas": [
                    "context-management-2025-06-27",
                    "interleaved-thinking-2025-05-14",
                ],
            }
        )
        assert get_client_headers()["anthropic-beta"] == (
            "context-management-2025-06-27,interleaved-thinking-2025-05-14"
        )
    finally:
        clear_client_headers()


def test_body_betas_merge_with_captured_header(protocol):
    from llm_proxy.providers.anthropic.client_headers import (
        capture_client_headers,
        clear_client_headers,
        get_client_headers,
    )

    try:
        capture_client_headers({"anthropic-beta": "interleaved-thinking-2025-05-14"})
        anthropic_protocol.on_parse_request(
            {
                "model": "m",
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "x"}],
                "betas": ["context-management-2025-06-27", "interleaved-thinking-2025-05-14"],
            }
        )
        assert get_client_headers()["anthropic-beta"] == (
            "interleaved-thinking-2025-05-14,context-management-2025-06-27"
        )
    finally:
        clear_client_headers()


def test_output_format_alias_reaches_output_config(provider):
    fmt = {"type": "json_schema", "schema": {"type": "object"}}
    body = build(
        provider,
        {
            "model": "m",
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "x"}],
            "output_format": fmt,
        },
    )
    assert body["output_config"]["format"] == fmt
    assert "output_format" not in body


def test_output_format_defers_to_output_config_format(provider):
    fmt = {"type": "json_schema", "schema": {"type": "object"}}
    body = build(
        provider,
        {
            "model": "m",
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "x"}],
            "output_config": {"format": {"type": "json_schema", "schema": {"type": "string"}}},
            "output_format": fmt,
        },
    )
    assert body["output_config"]["format"] == {"type": "json_schema", "schema": {"type": "string"}}


def test_stream_container_reaches_message_delta():
    sse = _run_stream(
        [
            {"type": "message_start", "message": {"id": "m", "usage": {"input_tokens": 1}}},
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": "end_turn",
                    "container": {"id": "c_1", "expires_at": "2030-01-01T00:00:00Z"},
                },
                "usage": {"output_tokens": 3},
            },
            {"type": "message_stop"},
        ]
    )
    assert '"container":{"id":"c_1"' in sse
    assert '"stop_reason":"end_turn"' in sse


def test_tool_definition_cache_control_roundtrip(provider):
    """Official tools accept a cache_control breakpoint; the rebuild path must
    not drop it (FunctionTool previously had no cache_control field)."""
    body = build(
        provider,
        {
            "model": "claude-opus-5",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "d",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                    "strict": True,
                }
            ],
        },
    )
    assert body["tools"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_server_tool_response_inclusion_roundtrip(provider):
    """web_search_20260318 / web_fetch_20260318 accept ``response_inclusion``;
    the rebuild path must keep it on both tool families."""
    body = build(
        provider,
        {
            "model": "claude-opus-5",
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "x"}],
            "tools": [
                {
                    "type": "web_search_20260318",
                    "name": "web_search",
                    "max_uses": 2,
                    "response_inclusion": "full",
                },
                {
                    "type": "web_fetch_20260318",
                    "name": "web_fetch",
                    "response_inclusion": "excluded",
                },
            ],
        },
    )
    assert body["tools"][0]["response_inclusion"] == "full"
    assert body["tools"][1]["response_inclusion"] == "excluded"


def test_non_streaming_beta_usage_and_diagnostics(provider, protocol):
    """fast-mode ``usage.speed``, compaction ``usage.iterations`` and the
    cache-diagnostics beta ``diagnostics`` object survive the converted path."""
    raw = {
        "id": "1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5",
        "content": [{"type": "text", "text": "hi"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "speed": "fast",
            "iterations": [{"type": "message", "input_tokens": 10, "output_tokens": 5}],
        },
        "diagnostics": {"cache_miss_reason": None},
    }
    internal = provider.parse_provider_response(raw, model="claude-opus-5")
    out = protocol.format_response(internal)
    assert out["usage"]["speed"] == "fast"
    assert out["usage"]["iterations"] == [
        {"type": "message", "input_tokens": 10, "output_tokens": 5}
    ]
    assert out["diagnostics"] == {"cache_miss_reason": None}


def test_non_streaming_diagnostics_null_state_preserved(provider, protocol):
    """``diagnostics: null`` is a meaningful "no divergence" state, not absence."""
    raw = {
        "id": "1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5",
        "content": [{"type": "text", "text": "hi"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "diagnostics": None,
    }
    internal = provider.parse_provider_response(raw, model="claude-opus-5")
    out = protocol.format_response(internal)
    assert "diagnostics" in out
    assert out["diagnostics"] is None


def test_stream_beta_usage_and_diagnostics_pass_chain():
    """Diagnostics replay inside message_start; speed/iterations ride the
    converter → transformer chain into the SSE stream."""
    sse = _run_stream(
        [
            {
                "type": "message_start",
                "message": {
                    "id": "m",
                    "usage": {"input_tokens": 1},
                    "diagnostics": {"cache_miss_reason": None},
                },
            },
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {
                    "output_tokens": 3,
                    "speed": "fast",
                    "iterations": [{"type": "message", "input_tokens": 1, "output_tokens": 3}],
                },
            },
            {"type": "message_stop"},
        ]
    )
    assert '"diagnostics":{"cache_miss_reason":null}' in sse
    assert '"speed":"fast"' in sse
    assert '"iterations":[{"type":"message"' in sse
    assert '"stop_reason":"end_turn"' in sse


def _message_delta_payloads(sse: str) -> list[dict]:
    """Parse the ``message_delta`` events out of a converted SSE stream."""
    import orjson

    payloads = []
    for frame in sse.split("\n\n"):
        if "event: message_delta" not in frame:
            continue
        for line in frame.split("\n"):
            if line.startswith("data: "):
                payloads.append(orjson.loads(line[len("data: ") :]))
    return payloads


def _message_start_usage(sse: str) -> dict:
    """Parse the ``message_start`` usage out of a converted SSE stream."""
    import orjson

    for frame in sse.split("\n\n"):
        if "event: message_start" not in frame:
            continue
        for line in frame.split("\n"):
            if line.startswith("data: "):
                return orjson.loads(line[len("data: ") :])["message"]["usage"]
    raise AssertionError("no message_start event")


def test_stream_input_tokens_exclude_cache_counters():
    """Regression: Claude Code counted the cached prompt twice.

    Canonical ``input_tokens`` INCLUDES cache reads/writes, while the Anthropic
    wire reports them separately and expects ``input_tokens`` to exclude them.
    Emitting the inclusive count beside ``cache_read_input_tokens`` made any
    client that sums the three show roughly twice the real prompt size.
    """
    transformer = AnthropicStreamingTransformer(model="deepseek-v4.1-flash", request_id="msg_1")
    chunk = {
        "id": "chatcmpl-x",
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": "deepseek-v4.1-flash",
        "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 178883,
            "completion_tokens": 2801,
            "total_tokens": 181684,
            "cache_read_input_tokens": 178304,
            "prompt_tokens_details": {"cached_tokens": 178304},
        },
    }
    sse = (transformer.transform(chunk) or "") + transformer.finalize()

    start_usage = _message_start_usage(sse)
    assert start_usage["input_tokens"] == 579
    assert start_usage["cache_read_input_tokens"] == 178304

    deltas = _message_delta_payloads(sse)
    assert deltas, "expected a terminal message_delta"
    delta_usage = deltas[-1]["usage"]
    assert delta_usage["input_tokens"] == 579
    assert delta_usage["cache_read_input_tokens"] == 178304
    # Real split: exclusive input + cache read + output.
    assert (
        delta_usage["input_tokens"]
        + delta_usage["cache_read_input_tokens"]
        + delta_usage["output_tokens"]
        == 181684
    )
    # The canonical StreamingUsage total must not add the cache read twice.
    usage = transformer.get_usage()
    assert usage is not None
    assert usage.total_tokens == 181684


def test_stream_nested_openai_cache_read_splits_input_tokens():
    """An OpenAI-family provider reports cache reads only through the nested
    ``prompt_tokens_details.cached_tokens`` dialect; the Anthropic wire must
    still lift them to the flat field and subtract them from ``input_tokens``."""
    transformer = AnthropicStreamingTransformer(model="gpt-4o", request_id="msg_1")
    chunk = {
        "id": "chatcmpl-x",
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": "gpt-4o",
        "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "total_tokens": 1050,
            "prompt_tokens_details": {"cached_tokens": 400},
        },
    }
    sse = (transformer.transform(chunk) or "") + transformer.finalize()

    start_usage = _message_start_usage(sse)
    assert start_usage["input_tokens"] == 600
    assert start_usage["cache_read_input_tokens"] == 400
    delta_usage = _message_delta_payloads(sse)[-1]["usage"]
    assert delta_usage["input_tokens"] == 600
    assert delta_usage["cache_read_input_tokens"] == 400


_INCLUSIVE_PROMPT = 178_883
_OUTPUT_TOKENS = 2_801
_CACHE_READ = 178_304

# Canonical usage dialects every provider family produces on the wire. The
# internal invariant is shared: ``prompt_tokens``/``input_tokens`` is the
# INCLUSIVE prompt total and the cache count is a subset expressed either flat
# or in the OpenAI-nested / DeepSeek-top-level dialect.
_PROVIDER_USAGE_DIALECTS = {
    "anthropic-native": {
        "prompt_tokens": _INCLUSIVE_PROMPT,
        "completion_tokens": _OUTPUT_TOKENS,
        "total_tokens": _INCLUSIVE_PROMPT + _OUTPUT_TOKENS,
        "cache_read_input_tokens": _CACHE_READ,
        "cache_creation_input_tokens": 0,
        "prompt_tokens_details": {"cached_tokens": _CACHE_READ},
    },
    "ollama": {
        "prompt_tokens": _INCLUSIVE_PROMPT,
        "completion_tokens": _OUTPUT_TOKENS,
        "total_tokens": _INCLUSIVE_PROMPT + _OUTPUT_TOKENS,
        "cache_read_input_tokens": _CACHE_READ,
        "prompt_tokens_details": {"cached_tokens": _CACHE_READ},
    },
    "gemini": {
        "prompt_tokens": _INCLUSIVE_PROMPT,
        "completion_tokens": _OUTPUT_TOKENS,
        "total_tokens": _INCLUSIVE_PROMPT + _OUTPUT_TOKENS,
        "cache_read_input_tokens": _CACHE_READ,
    },
    "openai-nested": {
        "prompt_tokens": _INCLUSIVE_PROMPT,
        "completion_tokens": _OUTPUT_TOKENS,
        "total_tokens": _INCLUSIVE_PROMPT + _OUTPUT_TOKENS,
        "prompt_tokens_details": {"cached_tokens": _CACHE_READ},
    },
    "deepseek-top-level": {
        "prompt_tokens": _INCLUSIVE_PROMPT,
        "completion_tokens": _OUTPUT_TOKENS,
        "total_tokens": _INCLUSIVE_PROMPT + _OUTPUT_TOKENS,
        "prompt_cache_hit_tokens": _CACHE_READ,
        "prompt_cache_miss_tokens": _INCLUSIVE_PROMPT - _CACHE_READ,
    },
    "cache-creation-only": {
        "prompt_tokens": _INCLUSIVE_PROMPT,
        "completion_tokens": _OUTPUT_TOKENS,
        "total_tokens": _INCLUSIVE_PROMPT + _OUTPUT_TOKENS,
        "cache_creation_input_tokens": _CACHE_READ,
    },
    "no-cache": {
        "prompt_tokens": _INCLUSIVE_PROMPT,
        "completion_tokens": _OUTPUT_TOKENS,
        "total_tokens": _INCLUSIVE_PROMPT + _OUTPUT_TOKENS,
    },
}


def _usage_total(usage: dict) -> int:
    return (
        usage.get("input_tokens", 0)
        + (usage.get("cache_read_input_tokens") or 0)
        + (usage.get("cache_creation_input_tokens") or 0)
    )


@pytest.mark.parametrize("dialect", sorted(_PROVIDER_USAGE_DIALECTS))
def test_stream_input_tokens_consistent_across_provider_dialects(dialect: str):
    """Every provider's canonical usage must split into the Anthropic wire so
    that ``input + cache_read + cache_creation`` equals the real prompt size.

    A client summing those three (Claude Code) must never see the cached prompt
    counted twice, whatever dialect the upstream reported it in.
    """
    usage = _PROVIDER_USAGE_DIALECTS[dialect]
    transformer = AnthropicStreamingTransformer(model="m", request_id="msg_1")
    chunk = {
        "id": "chatcmpl-x",
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": "m",
        "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"}],
        "usage": usage,
    }
    sse = (transformer.transform(chunk) or "") + transformer.finalize()

    assert _usage_total(_message_start_usage(sse)) == _INCLUSIVE_PROMPT
    assert _usage_total(_message_delta_payloads(sse)[-1]["usage"]) == _INCLUSIVE_PROMPT

    streaming_usage = transformer.get_usage()
    assert streaming_usage is not None
    assert streaming_usage.total_tokens == _INCLUSIVE_PROMPT + _OUTPUT_TOKENS


def test_ollama_streaming_converter_usage_flows_to_anthropic_wire():
    """End to end for the Ollama provider: its native ``done`` chunk folds
    ``prompt_eval_cached_count`` into canonical usage, which must then reach the
    Anthropic wire split rather than double-counted."""
    from llm_proxy.serialization.ollama.streaming import OllamaChunkConverter

    converter = OllamaChunkConverter(model="llama3.2", request_id="msg_1")
    canonical = converter.convert_chunk(
        {
            "model": "llama3.2",
            "done": True,
            "done_reason": "stop",
            "message": {"role": "assistant", "content": ""},
            "prompt_eval_count": _INCLUSIVE_PROMPT,
            "eval_count": _OUTPUT_TOKENS,
            "prompt_eval_cached_count": _CACHE_READ,
        }
    )
    transformer = AnthropicStreamingTransformer(model="llama3.2", request_id="msg_1")
    sse = (transformer.transform(canonical) or "") + transformer.finalize()

    start_usage = _message_start_usage(sse)
    assert start_usage["input_tokens"] == _INCLUSIVE_PROMPT - _CACHE_READ
    assert start_usage["cache_read_input_tokens"] == _CACHE_READ
    assert _usage_total(_message_delta_payloads(sse)[-1]["usage"]) == _INCLUSIVE_PROMPT


def test_non_streaming_input_tokens_exclude_nested_openai_cache_read():
    """Non-streaming counterpart: an OpenAI-family provider's nested cache read
    is lifted to the flat Anthropic field and removed from ``input_tokens``,
    matching the streaming formatter."""
    from llm_proxy.models import TextBlock
    from llm_proxy.models.internal import InternalResponse
    from llm_proxy.models.types import PromptTokensDetails, Usage

    response = InternalResponse(
        id="msg_1",
        model="gpt-4o",
        output=[TextBlock(text="hi")],
        usage=Usage(
            input_tokens=1000,
            output_tokens=50,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=400),
        ),
    )
    usage = AnthropicProtocolSerializer().format_response(response)["usage"]
    assert usage["input_tokens"] == 600
    assert usage["cache_read_input_tokens"] == 400


def test_non_streaming_safeguard_results_roundtrip(provider, protocol):
    """Server-side auto mode: ``safeguard_results`` must survive the converted
    (non-passthrough) response path, with the tool-use ids it keys on intact."""
    results = [
        {
            "type": "dangerous_tool_use",
            "status": {
                "type": "available",
                "tool_uses": {
                    "toolu_01V9Z5KXn3SU71Fzr5cquHLi": {
                        "type": "evaluated",
                        "outcome": "not_flagged",
                    }
                },
            },
        }
    ]
    raw = {
        "id": "1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_01V9Z5KXn3SU71Fzr5cquHLi",
                "name": "Bash",
                "input": {"command": "echo hello"},
            }
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 7, "output_tokens": 3},
        "safeguard_results": results,
    }
    out = protocol.format_response(provider.parse_provider_response(raw, model="claude-opus-5"))
    assert out["safeguard_results"] == results
    # The results refer to tool uses by id — those ids must not be rewritten.
    assert out["content"][0]["id"] == "toolu_01V9Z5KXn3SU71Fzr5cquHLi"


def test_non_streaming_safeguard_results_absent_stays_absent(provider, protocol):
    """Absence is not synthesized: a response without the field must not gain it."""
    raw = {
        "id": "1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5",
        "content": [{"type": "text", "text": "hi"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    out = protocol.format_response(provider.parse_provider_response(raw, model="claude-opus-5"))
    assert "safeguard_results" not in out


def test_stream_safeguard_results_reaches_terminal_message_delta():
    """The streamed field rides the converter → transformer chain and lands
    inside the terminal ``message_delta`` delta, where Claude Code reads it."""
    results = [
        {
            "type": "dangerous_tool_use",
            "status": {
                "type": "available",
                "tool_uses": {
                    "toolu_01V9Z5KXn3SU71Fzr5cquHLi": {
                        "type": "evaluated",
                        "outcome": "not_flagged",
                    }
                },
            },
        }
    ]
    sse = _run_stream(
        [
            {"type": "message_start", "message": {"id": "m", "usage": {"input_tokens": 1}}},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "safeguard_results": results},
                "usage": {"output_tokens": 3},
            },
            {"type": "message_stop"},
        ]
    )
    deltas = _message_delta_payloads(sse)
    assert deltas, "expected a terminal message_delta"
    assert deltas[-1]["delta"]["safeguard_results"] == results
    assert "toolu_01V9Z5KXn3SU71Fzr5cquHLi" in sse


def test_stream_safeguard_results_survives_without_stop_reason_or_usage():
    """Degenerate upstream: results with no stop_reason and no usage still get
    a terminal delta rather than being dropped."""
    results = [{"type": "dangerous_tool_use", "status": {"type": "available"}}]
    sse = _run_stream(
        [
            {"type": "message_start", "message": {"id": "m", "usage": {"input_tokens": 1}}},
            {"type": "message_delta", "delta": {"safeguard_results": results}, "usage": {}},
            {"type": "message_stop"},
        ]
    )
    deltas = _message_delta_payloads(sse)
    assert deltas[-1]["delta"]["safeguard_results"] == results
