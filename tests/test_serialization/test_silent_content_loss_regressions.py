"""Regression tests for content blocks that were silently lost during serialization.

Covers:
* Bug 3b: DocumentBlock sent to Ollama was silently dropped; it is now extracted
  as text for plain-text sources or degraded to a placeholder.
* Bug 3c: Anthropic tool_result blocks containing text + image content were sent to
  OpenAI Responses with an empty function_call_output.output string; the text is now
  preserved and images are degraded to placeholders.
"""

from llm_proxy.protocols.registry import get_protocol_serializer
from llm_proxy.serialization import get_provider_serializer
from llm_proxy.serialization.context import BuildContext


def _chat_convert(protocol: str, provider: str, raw: dict) -> dict:
    """Parse a protocol request and build the provider request body."""
    proto = get_protocol_serializer(protocol)
    internal = proto.parse_request(raw)
    prov = get_provider_serializer(provider)
    target_endpoint = "responses" if provider in {"openai"} else "chat_completions"
    ctx = BuildContext.from_request(
        internal,
        provider_name=provider,
        target_endpoint=target_endpoint,
        unknown_fields_policy="ignore",
        unsupported_block_policy="drop",
        supported_content_blocks=prov.supported_content_blocks,
    )
    return prov.build_provider_request(internal, ctx)


class TestOllamaDocumentBlockRegression:
    def test_document_text_source_is_not_dropped(self):
        body = _chat_convert(
            "anthropic",
            "ollama",
            {
                "model": "m",
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "document",
                                "source": {
                                    "type": "text",
                                    "media_type": "text/plain",
                                    "data": "aGVsbG8=",
                                },
                            }
                        ],
                    }
                ],
            },
        )
        assert body["messages"], "DocumentBlock text source must not produce an empty messages list"
        assert body["messages"][0]["content"] == "aGVsbG8="

    def test_document_base64_text_is_decoded(self):
        import base64

        body = _chat_convert(
            "anthropic",
            "ollama",
            {
                "model": "m",
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "hello"},
                            {
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": "text/plain",
                                    "data": base64.b64encode(b"doc").decode(),
                                },
                                "title": "readme.md",
                            },
                        ],
                    }
                ],
            },
        )
        assert body["messages"][0]["content"] == "hello doc"

    def test_document_base64_pdf_is_degraded_not_dropped(self):
        body = _chat_convert(
            "anthropic",
            "ollama",
            {
                "model": "m",
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": "application/pdf",
                                    "data": "JVBERi0xLg==",
                                },
                                "title": "report.pdf",
                            }
                        ],
                    }
                ],
            },
        )
        assert body["messages"], "Binary document must not be silently dropped"
        assert "[Document: report.pdf]" in body["messages"][0]["content"]


class TestOpenAIToolResultImageRegression:
    def test_tool_result_with_text_and_image_preserves_content(self):
        body = _chat_convert(
            "anthropic",
            "openai",
            {
                "model": "m",
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "t1",
                                "content": [
                                    {"type": "text", "text": "result"},
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/png",
                                            "data": "iVBOR=",
                                        },
                                    },
                                ],
                            }
                        ],
                    }
                ],
            },
        )
        input_items = body["input"]
        function_call_outputs = [
            item for item in input_items if item.get("type") == "function_call_output"
        ]
        assert len(function_call_outputs) == 1
        assert function_call_outputs[0]["call_id"] == "t1"
        output = function_call_outputs[0]["output"]
        assert output, "function_call_output.output must not be empty"
        assert "result" in output
        assert "[Image: image/png]" in output


class TestContextManagementDroppedForChatCompletions:
    """``context_management`` must not leak into Chat Completions bodies.

    ``context_management`` is supported by both Anthropic Messages (beta
    ``context-management-2025-06-27``, object shape ``{edits: [...]}``) and the
    OpenAI Responses API (array shape ``[{type, compact_threshold}]``), but has
    no Chat Completions equivalent. For Anthropic-origin requests it is modeled
    on ``AnthropicSpecificParams`` (so Anthropic->Anthropic preserves it); for
    any request routed to an OpenAI-compatible Chat Completions provider it must
    be dropped, since the proxy cannot perform server-side context editing on a
    third-party provider.
    """

    def test_context_management_dropped_for_openrouter(self):
        body = _chat_convert(
            "anthropic",
            "openrouter",
            {
                "model": "kimi-k3",
                "max_tokens": 1000,
                "stream": True,
                "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "high"},
                "context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]},
            },
        )
        assert "context_management" not in body
        # adaptive + output_config.effort must surface as reasoning_effort, not as
        # an invalid `thinking: {type: adaptive}`.
        assert body.get("reasoning_effort") == "high"
        assert "thinking" not in body

    def test_context_management_dropped_for_deepseek(self):
        body = _chat_convert(
            "anthropic",
            "deepseek",
            {
                "model": "deepseek-chat",
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": "hi"}],
                "context_management": {"edits": [{"type": "clear_thinking_20251015"}]},
            },
        )
        assert "context_management" not in body

    def test_openai_client_context_management_dropped(self):
        # Defensive: an OpenAI Chat Completions client that oddly sends
        # context_management must also have it stripped.
        body = _chat_convert(
            "openai",
            "openrouter",
            {
                "model": "kimi-k3",
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": "hi"}],
                "context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]},
            },
        )
        assert "context_management" not in body
