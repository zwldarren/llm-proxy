"""Tests for DeepSeek provider adapter.

DeepSeek uses an OpenAI-compatible Chat Completions API with a few
provider-specific extensions:

- ``thinking: {type: "enabled"|"disabled"}`` — toggle thinking mode
- ``reasoning_effort`` — "high" (default) or "max"
- ``user_id`` — custom user identifier for safety/KVCache/scheduling isolation
- ``reasoning_content`` in assistant responses (thinking mode only)
- ``insufficient_system_resource`` finish reason
- Deprecated: ``frequency_penalty``, ``presence_penalty`` (no-op)
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_proxy.core.adapter import get_adapter, list_providers
from llm_proxy.models import (
    ConversationContext,
    GenerationParams,
    InternalRequest,
    InternalResponse,
    Message,
    OpenAISpecificParams,
    TextBlock,
    ThinkingBlock,
    ThinkingConfig,
    ToolUseBlock,
)
from llm_proxy.providers.deepseek.adapter import DeepSeekAdapter
from llm_proxy.providers.openai_compatible._base import OpenAICompatibleBase

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class MockResponse:
    """Mock HTTP response compatible with httpx2's sync json()."""

    def __init__(self, status_code: int, json_data: dict | None = None):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.headers = MagicMock()

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


# ---------------------------------------------------------------------------
# Registration / basic properties
# ---------------------------------------------------------------------------


def test_deepseek_adapter_is_registered():
    """DeepSeek adapter is listed among registered providers."""
    assert "deepseek" in list_providers()


def test_deepseek_adapter_can_be_created():
    """DeepSeek adapter instantiates with the correct class name."""
    adapter = get_adapter("deepseek", api_key="sk-test")
    assert adapter.__class__.__name__ == "DeepSeekAdapter"


def test_deepseek_default_base_url():
    """Default base URL points to DeepSeek's API."""
    adapter = get_adapter("deepseek", api_key="sk-test")
    assert adapter._base_url == "https://api.deepseek.com/v1"


def test_deepseek_custom_base_url():
    """Custom base URL overrides the default."""
    adapter = get_adapter("deepseek", api_key="sk-test", base_url="https://custom.deepseek.com/v1")
    assert adapter._base_url == "https://custom.deepseek.com/v1"


def test_deepseek_provider_name():
    """Provider name is 'deepseek'."""
    adapter = get_adapter("deepseek", api_key="sk-test")
    assert adapter._provider_name == "deepseek"


def test_deepseek_is_openai_compatible_subclass():
    """DeepSeek adapter inherits from OpenAICompatibleBase."""
    assert issubclass(DeepSeekAdapter, OpenAICompatibleBase)


def test_deepseek_inherits_native_passthrough_base():
    """The passthrough machinery comes from NativePassthroughChatBase."""
    from llm_proxy.providers.openai_compatible._native import NativePassthroughChatBase

    assert issubclass(DeepSeekAdapter, NativePassthroughChatBase)


# ---------------------------------------------------------------------------
# Native endpoint routing (inherited machinery + site-root derivation)
# ---------------------------------------------------------------------------


def test_anthropic_messages_url_defaults_to_site_root():
    """Anthropic endpoint lives at the root, not under the /v1 chat alias."""
    adapter = get_adapter("deepseek", api_key="sk-test")
    assert adapter._anthropic_messages_url() == "https://api.deepseek.com/anthropic/v1/messages"


def test_responses_url_defaults_to_site_root():
    """Responses endpoint lives at the root, not under the /v1 chat alias."""
    adapter = get_adapter("deepseek", api_key="sk-test")
    assert adapter._responses_url() == "https://api.deepseek.com/responses"


def test_native_urls_follow_relay_base_url():
    """A custom base_url moves both native endpoints (relay support)."""
    adapter = get_adapter("deepseek", api_key="sk-test", base_url="https://ds.relay.example.com/v1")
    assert adapter._anthropic_messages_url() == "https://ds.relay.example.com/anthropic/v1/messages"
    assert adapter._responses_url() == "https://ds.relay.example.com/responses"


def test_native_urls_tolerate_root_base_url_without_alias():
    """A base_url without the /v1 alias is used as the site root as-is."""
    adapter = get_adapter("deepseek", api_key="sk-test", base_url="https://ds.relay.example.com")
    assert adapter._anthropic_messages_url() == "https://ds.relay.example.com/anthropic/v1/messages"
    assert adapter._responses_url() == "https://ds.relay.example.com/responses"


def test_native_urls_endpoint_base_urls_override_wins():
    """Per-endpoint overrides beat the derived site-root URLs."""
    adapter = get_adapter(
        "deepseek",
        api_key="sk-test",
        endpoint_base_urls={
            "anthropic_messages": "https://relay.example.com/a/",
            "responses": "https://relay.example.com/r/",
        },
    )
    assert adapter._anthropic_messages_url() == "https://relay.example.com/a"
    assert adapter._responses_url() == "https://relay.example.com/r"


# ---------------------------------------------------------------------------
# Reasoning field handling
# ---------------------------------------------------------------------------


def test_deepseek_uses_reasoning_content_by_default():
    """DeepSeek uses ``reasoning_content`` (standard OpenAI field).

    DeepSeek's _REASONING_FIELD is None, meaning it auto-detects and defaults
    to reasoning_content. Unlike OpenRouter which explicitly uses ``reasoning``.
    """
    adapter = DeepSeekAdapter(api_key="sk-test")
    assert adapter._REASONING_FIELD is None
    preferred = adapter._reasoning_field_preference()
    assert preferred is None  # None = auto-detect


def test_deepseek_request_body_defaults_to_reasoning_content():
    """By default, DeepSeek request bodies emit ``reasoning_content``."""
    adapter = DeepSeekAdapter(api_key="sk-test")
    request = InternalRequest(
        model="deepseek-v4-pro",
        conversation=ConversationContext(
            messages=[
                Message(
                    role="assistant",
                    content=[ThinkingBlock(thinking="Step-by-step reasoning...")],
                )
            ]
        ),
    )
    body = adapter._build_request_body(request)
    assistant_msg = body["messages"][0]
    assert "reasoning_content" in assistant_msg
    assert assistant_msg["reasoning_content"] == "Step-by-step reasoning..."
    assert "reasoning" not in assistant_msg


def test_deepseek_stream_chunk_normalizes_reasoning_to_reasoning_content():
    """Streaming chunks with ``reasoning`` are normalized to ``reasoning_content``."""
    adapter = DeepSeekAdapter(api_key="sk-test")
    chunk = {"choices": [{"delta": {"content": "answer", "reasoning": "thinking..."}}]}
    try:
        normalized = adapter._stream_transform_chunk(chunk, {})
        delta = normalized["choices"][0]["delta"]
        assert "reasoning_content" in delta
        assert delta["reasoning_content"] == "thinking..."
        assert "reasoning" not in delta
    finally:
        # The transform learns the provider's field from the chunk; this
        # synthetic ``reasoning`` fixture must not leak the preference into
        # the shared serializer builder (DeepSeek's real wire format is
        # ``reasoning_content``).
        adapter._get_request_builder().clear_reasoning_field_preference(adapter._base_url)


# ---------------------------------------------------------------------------
# Chat completion request body
# ---------------------------------------------------------------------------


@pytest.fixture
def deepseek_adapter():
    """Create a DeepSeek adapter for testing."""
    return DeepSeekAdapter(api_key="sk-test")


class TestDeepSeekRequestBody:
    """DeepSeek-specific request body construction."""

    def test_basic_request(self, deepseek_adapter):
        """Basic user message produces a valid request body."""
        request = InternalRequest(
            model="deepseek-v4-pro",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Hello")])]
            ),
        )
        body = deepseek_adapter._build_request_body(request)
        assert body["model"] == "deepseek-v4-pro"
        assert body["messages"][0]["role"] == "user"
        assert body["messages"][0]["content"] == "Hello"

    def test_thinking_enabled_emits_reasoning_effort(self, deepseek_adapter):
        """ThinkingConfig(type=enabled) → reasoning_effort: medium.

        The unified thinking layer converts ``ThinkingConfig(type="enabled")``
        into a ``reasoning_effort`` value. DeepSeek accepts both
        ``reasoning_effort`` and ``thinking: {type}``; the proxy uses the
        simpler ``reasoning_effort`` form when the effort can be derived.
        """
        request = InternalRequest(
            model="deepseek-v4-pro",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Puzzle")])]
            ),
            params=GenerationParams(thinking=ThinkingConfig(type="enabled")),
        )
        body = deepseek_adapter._build_request_body(request)
        assert body["reasoning_effort"] == "medium"
        assert "thinking" not in body

    def test_thinking_disabled_emits_reasoning_effort_none(self, deepseek_adapter):
        """ThinkingConfig(type=disabled) → reasoning_effort: none."""
        request = InternalRequest(
            model="deepseek-v4-pro",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Quick fact")])]
            ),
            params=GenerationParams(thinking=ThinkingConfig(type="disabled")),
        )
        body = deepseek_adapter._build_request_body(request)
        assert body["reasoning_effort"] == "none"
        assert "thinking" not in body

    def test_thinking_budget_low_emits_reasoning_effort_low(self, deepseek_adapter):
        """ThinkingConfig budget_tokens=4000 → reasoning_effort: low."""
        request = InternalRequest(
            model="deepseek-v4-pro",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Puzzle")])]
            ),
            params=GenerationParams(thinking=ThinkingConfig(type="enabled", budget_tokens=4000)),
        )
        body = deepseek_adapter._build_request_body(request)
        assert body["reasoning_effort"] == "low"

    def test_thinking_budget_high_emits_reasoning_effort_high(self, deepseek_adapter):
        """ThinkingConfig budget_tokens=32000 → reasoning_effort: high."""
        request = InternalRequest(
            model="deepseek-v4-pro",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Puzzle")])]
            ),
            params=GenerationParams(thinking=ThinkingConfig(type="enabled", budget_tokens=32000)),
        )
        body = deepseek_adapter._build_request_body(request)
        assert body["reasoning_effort"] == "high"

    def test_reasoning_effort_high(self, deepseek_adapter):
        """``reasoning_effort: high`` maps from ThinkingConfig effort."""
        request = InternalRequest(
            model="deepseek-v4-pro",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Solve")])]
            ),
            params=GenerationParams(thinking=ThinkingConfig(type="enabled", effort="high")),
        )
        body = deepseek_adapter._build_request_body(request)
        assert body["reasoning_effort"] == "high"

    def test_reasoning_effort_max(self, deepseek_adapter):
        """``reasoning_effort: max`` for complex agent requests."""
        request = InternalRequest(
            model="deepseek-v4-pro",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Complex task")])]
            ),
            params=GenerationParams(thinking=ThinkingConfig(type="enabled", effort="max")),
        )
        body = deepseek_adapter._build_request_body(request)
        assert body["reasoning_effort"] == "max"

    def test_reasoning_effort_from_openai_params(self, deepseek_adapter):
        """``reasoning_effort`` from OpenAISpecificParams is forwarded."""
        request = InternalRequest(
            model="deepseek-v4-pro",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Think hard")])]
            ),
            params=GenerationParams(openai=OpenAISpecificParams(reasoning_effort="high")),
        )
        body = deepseek_adapter._build_request_body(request)
        assert body["reasoning_effort"] == "high"

    def test_user_id_with_passthrough_policy(self):
        """``user_id`` from extra reaches the body when field policy is passthrough.

        DeepSeek's ``user_id`` parameter is provider-specific (not part of the
        standard OpenAI schema), so it must be sent via ``extra`` and the adapter
        must be configured with ``unknown_fields_policy="passthrough"``.
        """
        adapter = DeepSeekAdapter(api_key="sk-test", unknown_fields_policy="passthrough")
        request = InternalRequest(
            model="deepseek-v4-pro",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Hello")])]
            ),
            extra={"user_id": "customer-abc-123"},
        )
        body = adapter._build_request_body(request)
        assert body["user_id"] == "customer-abc-123"

    def test_max_tokens(self, deepseek_adapter):
        """``max_tokens`` is included when set."""
        request = InternalRequest(
            model="deepseek-v4-pro",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Hello")])]
            ),
            params=GenerationParams(max_tokens=1024),
        )
        body = deepseek_adapter._build_request_body(request)
        assert body["max_tokens"] == 1024

    def test_stop_strings(self, deepseek_adapter):
        """Stop sequences are forwarded."""
        request = InternalRequest(
            model="deepseek-v4-pro",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Hello")])]
            ),
            params=GenerationParams(stop=["END", "STOP"]),
        )
        body = deepseek_adapter._build_request_body(request)
        assert body["stop"] == ["END", "STOP"]

    def test_seed(self, deepseek_adapter):
        """Seed parameter is forwarded for reproducibility."""
        request = InternalRequest(
            model="deepseek-v4-pro",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Hello")])]
            ),
            params=GenerationParams(seed=42),
        )
        body = deepseek_adapter._build_request_body(request)
        assert body["seed"] == 42

    def test_response_format_json_object(self, deepseek_adapter):
        """``response_format: {type: json_object}`` for JSON output mode."""
        from llm_proxy.models.types import ResponseFormat

        request = InternalRequest(
            model="deepseek-v4-pro",
            conversation=ConversationContext(
                messages=[
                    Message(role="system", content=[TextBlock(text="Output JSON only.")]),
                    Message(role="user", content=[TextBlock(text="List planets")]),
                ]
            ),
            params=GenerationParams(response_format=ResponseFormat(type="json_object")),
        )
        body = deepseek_adapter._build_request_body(request)
        assert body["response_format"] == {"type": "json_object"}

    def test_stream_options_include_usage(self, deepseek_adapter):
        """``stream_options.include_usage`` is forwarded."""
        from llm_proxy.models.types import StreamOptions

        request = InternalRequest(
            model="deepseek-v4-pro",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Hello")])]
            ),
            stream_options=StreamOptions(include_usage=True),
        )
        body = deepseek_adapter._build_request_body(request)
        assert body["stream_options"]["include_usage"]

    def test_no_thinking_params_by_default(self, deepseek_adapter):
        """When thinking is not configured, no thinking params leak into the body."""
        request = InternalRequest(
            model="deepseek-v4-pro",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Hello")])]
            ),
        )
        body = deepseek_adapter._build_request_body(request)
        assert "thinking" not in body
        assert "reasoning_effort" not in body

    # ------------------------------------------------------------------
    # Thinking-mode reasoning_content echo (multi-provider mixing)
    # ------------------------------------------------------------------

    def test_tool_call_without_reasoning_gets_placeholder(self, deepseek_adapter):
        """A tool-call assistant message with no reasoning gets the placeholder.

        DeepSeek's thinking mode requires ``reasoning_content`` to be echoed
        back for every assistant message that made tool calls (HTTP 400
        otherwise). Turns served by another provider (e.g. Ollama) can carry
        tool calls without any reasoning — the adapter must not forward such a
        message bare.
        """
        request = InternalRequest(
            model="deepseek-v4-pro",
            conversation=ConversationContext(
                messages=[
                    Message(
                        role="assistant",
                        content=[
                            ToolUseBlock(
                                id="echo_call_ollama",
                                name="get_weather",
                                input={"city": "Shanghai"},
                            )
                        ],
                    )
                ]
            ),
        )
        body = deepseek_adapter._build_request_body(request)
        assistant_msg = body["messages"][0]
        # Minimal factual placeholder naming the tool (real reasoning is
        # preferred via cache/ThinkingBlock history; this only fills the gap).
        assert assistant_msg["reasoning_content"] == "Calling get_weather."

    def test_tool_call_with_real_reasoning_preserved(self, deepseek_adapter):
        """Real reasoning on a tool-call assistant message is preserved unchanged."""
        request = InternalRequest(
            model="deepseek-v4-pro",
            conversation=ConversationContext(
                messages=[
                    Message(
                        role="assistant",
                        content=[
                            ThinkingBlock(thinking="Let me check the weather first."),
                            ToolUseBlock(
                                id="echo_call_ds", name="get_weather", input={"city": "Beijing"}
                            ),
                        ],
                    )
                ]
            ),
        )
        body = deepseek_adapter._build_request_body(request)
        assistant_msg = body["messages"][0]
        assert assistant_msg["reasoning_content"] == "Let me check the weather first."

    def test_non_tool_call_assistant_message_untouched(self, deepseek_adapter):
        """Content-only assistant messages never get an injected placeholder.

        DeepSeek only requires reasoning_content echo for tool-call turns;
        non-tool-call reasoning is ignored by the API (per its docs).
        """
        request = InternalRequest(
            model="deepseek-v4-pro",
            conversation=ConversationContext(
                messages=[Message(role="assistant", content=[TextBlock(text="Done.")])]
            ),
        )
        body = deepseek_adapter._build_request_body(request)
        assistant_msg = body["messages"][0]
        assert "reasoning_content" not in assistant_msg

    def test_mixed_provider_conversation_echoes_reasoning(self, deepseek_adapter):
        """Mixed deepseek+ollama history stays valid: every tool-call assistant
        message carries reasoning_content, real reasoning is kept, and
        content-only messages are untouched."""
        request = InternalRequest(
            model="deepseek-v4-pro",
            conversation=ConversationContext(
                messages=[
                    Message(
                        role="assistant",
                        content=[
                            ThinkingBlock(thinking="Let me check the weather."),
                            ToolUseBlock(
                                id="echo_call_ds", name="get_weather", input={"city": "Beijing"}
                            ),
                        ],
                    ),
                    Message(
                        role="assistant",
                        content=[
                            ToolUseBlock(
                                id="echo_call_ollama",
                                name="get_weather",
                                input={"city": "Shanghai"},
                            )
                        ],
                    ),
                    Message(role="assistant", content=[TextBlock(text="Done.")]),
                ]
            ),
        )
        body = deepseek_adapter._build_request_body(request)
        assistant_msgs = [m for m in body["messages"] if m["role"] == "assistant"]
        assert assistant_msgs[0]["reasoning_content"] == "Let me check the weather."
        assert assistant_msgs[1]["reasoning_content"] == "Calling get_weather."
        assert "reasoning_content" not in assistant_msgs[2]

    def test_stream_body_echoes_reasoning(self, deepseek_adapter):
        """The streaming body path (used by stream_chat_completion) applies the
        same reasoning echo guarantee via _build_request_body."""
        request = InternalRequest(
            model="deepseek-v4-pro",
            conversation=ConversationContext(
                messages=[
                    Message(
                        role="assistant",
                        content=[
                            ToolUseBlock(
                                id="echo_call_ollama",
                                name="get_weather",
                                input={"city": "Shanghai"},
                            )
                        ],
                    )
                ]
            ),
            stream=True,
        )
        body = deepseek_adapter._stream_body(request)
        assert body["stream"]
        assert body["messages"][0]["reasoning_content"] == "Calling get_weather."

    def test_thinking_disabled_skips_placeholder(self, deepseek_adapter):
        """Thinking mode disabled ⇒ no validation ⇒ no injected placeholder.

        With ``thinking.type == "disabled"`` (or ``reasoning_effort: none``)
        DeepSeek performs no reasoning-content validation, so injecting a
        placeholder would only pollute the model's context.
        """
        request = InternalRequest(
            model="deepseek-v4-pro",
            conversation=ConversationContext(
                messages=[
                    Message(
                        role="assistant",
                        content=[
                            ToolUseBlock(
                                id="echo_call_ollama",
                                name="get_weather",
                                input={"city": "Shanghai"},
                            )
                        ],
                    )
                ]
            ),
            params=GenerationParams(thinking=ThinkingConfig(type="disabled")),
        )
        body = deepseek_adapter._build_request_body(request)
        assistant_msg = body["messages"][0]
        assert "reasoning_content" not in assistant_msg

    def test_reasoning_effort_none_skips_placeholder(self, deepseek_adapter):
        """``reasoning_effort: none`` disables thinking the same way."""
        request = InternalRequest(
            model="deepseek-v4-pro",
            conversation=ConversationContext(
                messages=[
                    Message(
                        role="assistant",
                        content=[
                            ToolUseBlock(
                                id="echo_call_ollama",
                                name="get_weather",
                                input={"city": "Shanghai"},
                            )
                        ],
                    )
                ]
            ),
            params=GenerationParams(thinking=ThinkingConfig(type="enabled", effort="none")),
        )
        body = deepseek_adapter._build_request_body(request)
        assistant_msg = body["messages"][0]
        assert "reasoning_content" not in assistant_msg

    def test_unnamed_tool_call_gets_generic_placeholder(self, deepseek_adapter):
        """A tool call without a function name falls back to a generic placeholder."""
        request = InternalRequest(
            model="deepseek-v4-pro",
            conversation=ConversationContext(
                messages=[
                    Message(
                        role="assistant",
                        content=[ToolUseBlock(id="echo_call_x", name="", input={"a": 1})],
                    )
                ]
            ),
        )
        body = deepseek_adapter._build_request_body(request)
        assistant_msg = body["messages"][0]
        assert assistant_msg["reasoning_content"] == "Tool call."


class TestReasoningEchoAcrossAdapters:
    """The echo guarantee is shared: base + OpenRouter apply it model-gated.

    Generic OpenAI-compatible and OpenRouter adapters serve many models; the
    DeepSeek-style reasoning echo is only applied when the model name matches
    a known marker (``deepseek``, ``kimi``), so non-thinking models are never
    polluted with injected reasoning.
    """

    def _request(self, model: str):
        return InternalRequest(
            model=model,
            conversation=ConversationContext(
                messages=[
                    Message(
                        role="assistant",
                        content=[
                            ToolUseBlock(
                                id="echo_call_1", name="get_weather", input={"city": "Shanghai"}
                            )
                        ],
                    )
                ]
            ),
        )

    def test_generic_compatible_deepseek_model_gets_echo(self):
        """A deepseek-named model on the generic adapter gets reasoning_content."""
        adapter = OpenAICompatibleBase(api_key="test-key", base_url="https://example.com/v1")
        body = adapter._build_request_body(self._request("deepseek-v4-pro"))
        assert body["messages"][0]["reasoning_content"] == "Calling get_weather."

    def test_generic_compatible_non_deepseek_model_untouched(self):
        """Non-deepseek models on the generic adapter are never injected."""
        adapter = OpenAICompatibleBase(api_key="test-key", base_url="https://example.com/v1")
        for model in ("gpt-4o", "claude-3-5-sonnet"):
            body = adapter._build_request_body(self._request(model))
            assert "reasoning_content" not in body["messages"][0]

    def test_openrouter_deepseek_model_gets_reasoning_field(self):
        """OpenRouter expects ``reasoning``; deepseek models get the echo there."""
        from llm_proxy.providers.openrouter.adapter import OpenRouterAdapter

        adapter = OpenRouterAdapter(api_key="sk-or-test")
        body = adapter._build_request_body(self._request("deepseek/deepseek-v4-pro"))
        assistant_msg = body["messages"][0]
        assert assistant_msg["reasoning"] == "Calling get_weather."
        assert "reasoning_content" not in assistant_msg

    def test_openrouter_non_deepseek_model_untouched(self):
        """Non-deepseek models via OpenRouter never receive injected reasoning."""
        from llm_proxy.providers.openrouter.adapter import OpenRouterAdapter

        adapter = OpenRouterAdapter(api_key="sk-or-test")
        for model in ("anthropic/claude-3-5-sonnet", "openai/gpt-4o"):
            body = adapter._build_request_body(self._request(model))
            assert "reasoning" not in body["messages"][0]
            assert "reasoning_content" not in body["messages"][0]

    def test_generic_compatible_thinking_disabled_skips(self):
        """Thinking disabled skips the echo on the generic adapter as well."""
        adapter = OpenAICompatibleBase(api_key="test-key", base_url="https://example.com/v1")
        request = self._request("deepseek-v4-pro")
        request.params.thinking = ThinkingConfig(type="disabled")
        body = adapter._build_request_body(request)
        assert "reasoning_content" not in body["messages"][0]


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


class TestDeepSeekResponseParsing:
    """DeepSeek-specific response parsing."""

    @pytest.fixture
    def adapter(self):
        return DeepSeekAdapter(api_key="sk-test")

    def _make_response(self, status=200, **overrides):
        defaults = {
            "id": "chatcmpl-deepseek-123",
            "object": "chat.completion",
            "model": "deepseek-v4-pro",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
            "system_fingerprint": "fp_test",
        }
        defaults.update(overrides)
        return MockResponse(status, json_data=defaults)

    @pytest.mark.asyncio
    async def test_basic_response(self, adapter):
        """Basic chat completion response is parsed correctly."""
        mock_client = MagicMock()
        mock_client.post = AsyncMock(
            return_value=self._make_response(
                choices=[
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello, world!"},
                        "finish_reason": "stop",
                    }
                ]
            )
        )

        with patch(
            "llm_proxy.providers.openai_compatible._base.AsyncSession",
            return_value=mock_client,
        ):
            adapter._http_client = mock_client
            request = InternalRequest(
                model="deepseek-v4-pro",
                conversation=ConversationContext(
                    messages=[Message(role="user", content=[TextBlock(text="Hi")])]
                ),
            )
            response = await adapter.chat_completion(request)
            assert isinstance(response, InternalResponse)
            assert len(response.output) == 1
            assert isinstance(response.output[0], TextBlock)
            assert response.output[0].text == "Hello, world!"
            assert response.model == "deepseek-v4-pro"
            assert response.finish_reason == "stop"

    def test_reasoning_content_in_response(self, adapter):
        """``reasoning_content`` in response → ThinkingBlock + TextBlock."""
        serializer = adapter._get_serializer()
        response_data = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "model": "deepseek-v4-pro",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "The answer is 4.",
                        "reasoning_content": "Let me count: 2+2=4",
                    },
                    "finish_reason": "stop",
                }
            ],
        }
        result = serializer.parse_provider_response(response_data, model="deepseek-v4-pro")
        assert len(result.output) >= 2
        # First block should be the thinking/reasoning
        thinking_blocks = [b for b in result.output if isinstance(b, ThinkingBlock)]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0].thinking == "Let me count: 2+2=4"
        # Text should also be present
        text_blocks = [b for b in result.output if isinstance(b, TextBlock)]
        assert len(text_blocks) == 1
        assert text_blocks[0].text == "The answer is 4."

    def test_insufficient_system_resource_finish_reason(self, adapter):
        """DeepSeek-specific ``insufficient_system_resource`` finish reason passes through.

        This finish reason indicates the inference system ran out of resources.
        The proxy preserves the original value so callers can distinguish it
        from a generic error.
        """
        serializer = adapter._get_serializer()
        response_data = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "model": "deepseek-v4-pro",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Partial response...",
                    },
                    "finish_reason": "insufficient_system_resource",
                }
            ],
        }
        result = serializer.parse_provider_response(response_data, model="deepseek-v4-pro")
        assert result.finish_reason == "insufficient_system_resource"

    def test_tool_calls_in_response(self, adapter):
        """Tool calls in DeepSeek response are parsed correctly."""
        serializer = adapter._get_serializer()
        response_data = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "model": "deepseek-v4-pro",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "Beijing"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
        result = serializer.parse_provider_response(response_data, model="deepseek-v4-pro")
        assert result.finish_reason == "tool_calls"
        tool_use_blocks = [b for b in result.output if isinstance(b, ToolUseBlock)]
        assert len(tool_use_blocks) == 1
        assert tool_use_blocks[0].name == "get_weather"
        assert tool_use_blocks[0].input == {"location": "Beijing"}

    def test_usage_with_reasoning_tokens(self, adapter):
        """DeepSeek's completion_tokens_details.reasoning_tokens is mapped."""
        serializer = adapter._get_serializer()
        response_data = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "model": "deepseek-v4-pro",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Cached response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "total_tokens": 110,
                "prompt_cache_hit_tokens": 80,
                "prompt_cache_miss_tokens": 20,
                "completion_tokens_details": {"reasoning_tokens": 50},
            },
        }
        result = serializer.parse_provider_response(response_data, model="deepseek-v4-pro")
        assert result.usage is not None
        assert result.usage.input_tokens == 100
        assert result.usage.output_tokens == 10
        assert result.usage.total_tokens == 110
        # DeepSeek reports cache hits as top-level prompt_cache_hit_tokens;
        # the generic OpenAI parser folds them into
        # prompt_tokens_details.cached_tokens (see fold_deepseek_cache_hits),
        # which the billing layer maps to cached_prompt_tokens so the
        # cache-read rate applies instead of the full input rate.
        assert result.usage.prompt_tokens_details is not None
        assert result.usage.prompt_tokens_details.cached_tokens == 80
        assert result.usage.completion_tokens_details is not None
        assert result.usage.completion_tokens_details.reasoning_tokens == 50

    def test_system_fingerprint_in_response(self, adapter):
        """System fingerprint from DeepSeek is captured in provider_info."""
        serializer = adapter._get_serializer()
        response_data = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "model": "deepseek-v4-pro",
            "system_fingerprint": "fp_deepseek_2025",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "OK"},
                    "finish_reason": "stop",
                }
            ],
        }
        result = serializer.parse_provider_response(response_data, model="deepseek-v4-pro")
        assert "system_fingerprint" in result.provider_info
        assert result.provider_info["system_fingerprint"] == "fp_deepseek_2025"


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class TestDeepSeekStreaming:
    """DeepSeek streaming tests."""

    @pytest.fixture
    def adapter(self):
        return DeepSeekAdapter(api_key="sk-test")

    def test_stream_body_includes_stream_true(self, adapter):
        """Streaming request body includes ``stream: true``."""
        request = InternalRequest(
            model="deepseek-v4-pro",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Hello")])]
            ),
            stream=True,
        )
        body = adapter._stream_body(request)
        assert body["stream"]

    def test_streaming_chunk_with_reasoning_content(self, adapter):
        """Streaming chunks with reasoning_content are passed through."""
        chunk = {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "model": "deepseek-v4-pro",
            "choices": [
                {
                    "index": 0,
                    "delta": {"reasoning_content": "Let me think...", "role": "assistant"},
                    "finish_reason": None,
                }
            ],
        }
        result = adapter._stream_transform_chunk(chunk, {"model": "deepseek-v4-pro"})
        assert result is not None
        delta = result["choices"][0]["delta"]
        assert "reasoning_content" in delta
        assert delta["reasoning_content"] == "Let me think..."

    def test_streaming_chunk_with_content(self, adapter):
        """Streaming chunks with text content are passed through."""
        chunk = {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "model": "deepseek-v4-pro",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Hello", "role": "assistant"},
                    "finish_reason": None,
                }
            ],
        }
        result = adapter._stream_transform_chunk(chunk, {"model": "deepseek-v4-pro"})
        assert result is not None
        delta = result["choices"][0]["delta"]
        assert delta["content"] == "Hello"

    def test_streaming_chunk_with_tool_call_delta(self, adapter):
        """Streaming tool call deltas are passed through."""
        chunk = {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "model": "deepseek-v4-pro",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_abc",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"loc'},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }
        result = adapter._stream_transform_chunk(chunk, {"model": "deepseek-v4-pro"})
        assert result is not None
        tool_calls = result["choices"][0]["delta"]["tool_calls"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "get_weather"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestDeepSeekErrorHandling:
    """DeepSeek-specific error response handling."""

    @pytest.fixture
    def adapter(self):
        return DeepSeekAdapter(api_key="sk-test")

    def test_401_authentication_error(self):
        """401 → authentication error."""
        from llm_proxy.core.errors.handler import ErrorHandler

        handler = ErrorHandler()
        error = handler.create_provider_error(
            message="Authentication failed",
            error_type="authentication_error",
            status_code=401,
            provider_name="deepseek",
        )
        assert error.status_code == 401
        assert "Authentication" in error.message

    def test_402_insufficient_balance(self):
        """402 Payment Required — DeepSeek-specific error code."""
        from llm_proxy.core.errors.handler import ErrorHandler

        handler = ErrorHandler()
        error = handler.create_provider_error(
            message="Insufficient Balance",
            error_type="insufficient_balance",
            status_code=402,
            provider_name="deepseek",
        )
        assert error.status_code == 402
        assert "Insufficient Balance" in error.message

    def test_429_rate_limit(self):
        """429 → rate limit error."""
        from llm_proxy.core.errors.handler import ErrorHandler

        handler = ErrorHandler()
        error = handler.create_provider_error(
            message="Rate limit exceeded",
            error_type="rate_limit_error",
            status_code=429,
            provider_name="deepseek",
        )
        assert error.status_code == 429

    def test_503_server_overloaded(self):
        """503 → service unavailable."""
        from llm_proxy.core.errors.handler import ErrorHandler

        handler = ErrorHandler()
        error = handler.create_provider_error(
            message="Server overloaded",
            error_type="api_error",
            status_code=503,
            provider_name="deepseek",
        )
        assert error.status_code == 503
