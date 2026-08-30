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
        protocol.parse_request(
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
        protocol.parse_request(
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
