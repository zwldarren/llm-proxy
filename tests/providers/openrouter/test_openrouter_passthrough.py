"""OpenRouter native Responses passthrough, app attribution and header forwarding.

OpenRouter differs from a plain Chat Completions upstream in three ways this
module pins down:

* it exposes a native OpenAI Responses endpoint, so ``/v1/responses`` clients
  are forwarded verbatim instead of being flattened to Chat Completions
  (which silently dropped ``reasoning.mode``/``context``, ``provider``,
  ``plugins``, ``usage``, ...);
* it accepts the unified ``reasoning`` object on its Chat Completions wire,
  so the builder emits that rather than a bare ``reasoning_effort``;
* it tracks the calling application through attribution headers and accepts
  per-request control headers that must survive the proxy hop.
"""

from unittest.mock import AsyncMock, patch

import pytest
from providers.helpers import MockStreamResponse, make_request, make_sse_events

import llm_proxy.protocols.openresponses.serializer  # noqa: F401 — registration
import llm_proxy.serialization.providers.chat_completions  # noqa: F401 — registration
from llm_proxy.core.conversion import plan_conversion, prepare_native_body
from llm_proxy.models import (
    ConversionTier,
    InternalEmbeddingRequest,
    InternalImageRequest,
    InternalSpeechRequest,
    InternalTranscriptionRequest,
)
from llm_proxy.protocols.registry import get_protocol_serializer
from llm_proxy.providers.anthropic.client_headers import (
    capture_client_headers as capture_anthropic_client_headers,
)
from llm_proxy.providers.anthropic.client_headers import (
    clear_client_headers as clear_anthropic_client_headers,
)
from llm_proxy.providers.openai.client_headers import (
    capture_client_headers,
    clear_client_headers,
)
from llm_proxy.providers.openrouter.adapter import OpenRouterAdapter


@pytest.fixture
def adapter():
    return OpenRouterAdapter(api_key="sk-or-test")


@pytest.fixture(autouse=True)
def _clear_captured_headers():
    """Client headers are a contextvar; never leak them between tests."""
    yield
    clear_client_headers()
    clear_anthropic_client_headers()


# ---------------------------------------------------------------------------
# Capability declarations
# ---------------------------------------------------------------------------


class TestNativeProtocols:
    def test_declares_anthropic_and_openresponses_native(self, adapter):
        assert adapter.native_protocols == frozenset({"anthropic", "openresponses"})

    def test_responses_url_hangs_off_the_chat_root(self, adapter):
        assert adapter._responses_url(model="openai/gpt-5.6") == (
            "https://openrouter.ai/api/v1/responses"
        )

    def test_responses_request_takes_the_native_tier(self, adapter):
        raw = {
            "model": "openai/gpt-5.6",
            "input": "Prove that there are infinitely many primes.",
            "reasoning": {"mode": "pro", "effort": "high", "context": "all_turns"},
            "provider": {"sort": "throughput", "zdr": True},
            "plugins": [{"id": "web"}],
            "usage": {"include": True},
            "stream": False,
        }
        request = get_protocol_serializer("openresponses").parse_request(raw)
        request.metadata.protocol_name = "openresponses"
        request._raw_protocol_data = raw

        plan = plan_conversion(adapter, request)
        assert plan.response_mode == ConversionTier.NATIVE_PASSTHROUGH
        assert plan.stream_mode == ConversionTier.NATIVE_PASSTHROUGH

        # Verbatim: nothing the Chat Completions translation used to drop is lost.
        body = prepare_native_body(adapter, request)
        assert body["reasoning"] == {"mode": "pro", "effort": "high", "context": "all_turns"}
        assert body["provider"] == {"sort": "throughput", "zdr": True}
        assert body["plugins"] == [{"id": "web"}]
        assert body["usage"] == {"include": True}
        assert body["model"] == "openai/gpt-5.6"

    def test_chat_protocol_is_not_native(self, adapter):
        """only /v1/responses is forwarded verbatim; chat keeps the translation."""
        assert not adapter.supports_native_request("openai", None)


class TestNativeMessages:
    """Anthropic-protocol clients are forwarded to OpenRouter's Messages API."""

    def test_messages_url_hangs_off_the_chat_root(self, adapter):
        assert adapter._anthropic_messages_url(model="anthropic/claude-sonnet-4.5") == (
            "https://openrouter.ai/api/v1/messages"
        )

    def test_anthropic_request_takes_the_native_tier(self, adapter):
        raw = {
            "model": "claude-alias",
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "thinking": {"type": "enabled", "budget_tokens": 4096},
            "context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]},
            "provider": {"sort": "throughput"},
            "stream": False,
        }
        request = make_request(raw, model="anthropic/claude-sonnet-4.5", protocol_name="anthropic")

        plan = plan_conversion(adapter, request)
        assert plan.response_mode == ConversionTier.NATIVE_PASSTHROUGH
        assert plan.stream_mode == ConversionTier.NATIVE_PASSTHROUGH

        # Verbatim: the double translation (Messages -> Chat Completions ->
        # Messages) is what dropped thinking and context management.
        body = prepare_native_body(adapter, request)
        assert body["thinking"] == {"type": "enabled", "budget_tokens": 4096}
        assert body["context_management"] == {"edits": [{"type": "clear_tool_uses_20250919"}]}
        assert body["provider"] == {"sort": "throughput"}
        assert body["max_tokens"] == 2048
        assert body["model"] == "anthropic/claude-sonnet-4.5"

    @pytest.mark.asyncio
    async def test_native_completion_posts_verbatim_and_bills_reported_cost(
        self, adapter, mock_response_cls
    ):
        upstream = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "anthropic/claude-sonnet-4.5",
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 4,
                "cache_read_input_tokens": 2,
                "cost": 0.00042,
                "is_byok": False,
            },
        }
        raw = {
            "model": "claude-alias",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "stream": False,
        }
        request = make_request(raw, model="anthropic/claude-sonnet-4.5", protocol_name="anthropic")

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response_cls(json_data=upstream))
        with patch.object(adapter, "_get_client", return_value=mock_client):
            result = await adapter.chat_completion(request)

        call = mock_client.post.call_args
        assert call.args[0] == "https://openrouter.ai/api/v1/messages"
        assert call.kwargs["json"]["model"] == "anthropic/claude-sonnet-4.5"

        assert result.provider_info["_raw_response_body"] is upstream
        assert result.provider_info["openrouter_cost"] == 0.00042
        # Anthropic input_tokens excludes cached tokens; billing sees the fold.
        assert result.usage.input_tokens == 12
        # The Chat Completions unknown-field dump must not fire on an Anthropic
        # envelope (it would spill content/role/stop_reason into provider_info).
        assert "content" not in result.provider_info
        assert "stop_reason" not in result.provider_info

    def test_protocol_headers_forwarded_without_the_client_fingerprint(self, adapter):
        capture_anthropic_client_headers(
            {
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "structured-outputs-2025-11-13",
                "x-app": "claude-code",
                "user-agent": "claude-cli/2.0.1 (external, cli)",
                "x-stainless-runtime": "node",
            }
        )

        headers = adapter._build_headers()

        assert headers["anthropic-version"] == "2023-06-01"
        assert headers["anthropic-beta"] == "structured-outputs-2025-11-13"
        # The router's upstream is not the client's original target.
        assert "x-app" not in headers
        assert "user-agent" not in headers
        assert "x-stainless-runtime" not in headers

    @pytest.mark.asyncio
    async def test_messages_stream_forwards_raw_sse(self, adapter):
        sse_events = make_sse_events(
            [
                (
                    "message_start",
                    '{"type":"message_start","message":{"id":"msg_s","type":"message",'
                    '"role":"assistant","content":[],"model":"anthropic/claude-sonnet-4.5",'
                    '"usage":{"input_tokens":10,"output_tokens":1}}}',
                ),
                (
                    "content_block_delta",
                    '{"type":"content_block_delta","index":0,'
                    '"delta":{"type":"text_delta","text":"hi"}}',
                ),
                ("message_stop", '{"type":"message_stop"}'),
            ]
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=MockStreamResponse(sse_events))

        raw = {
            "model": "claude-alias",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "stream": False,
        }
        request = make_request(raw, model="anthropic/claude-sonnet-4.5", protocol_name="anthropic")

        with patch.object(adapter, "_get_client", return_value=mock_client):
            stream_gen = await adapter.stream_chat_completion_native(request)
            frames = [frame async for frame in stream_gen]

        call = mock_client.post.call_args
        assert call.args[0] == "https://openrouter.ai/api/v1/messages"
        assert call.kwargs["json"]["stream"] is True
        assert raw["stream"] is False  # the stash is never coerced in place
        assert any("message_start" in frame for frame in frames)
        assert any("content_block_delta" in frame for frame in frames)
        assert any("message_stop" in frame for frame in frames)


# ---------------------------------------------------------------------------
# App attribution
# ---------------------------------------------------------------------------


class TestAppAttribution:
    def test_defaults_to_the_llm_proxy_project(self, adapter):
        headers = adapter._build_headers()
        assert headers["HTTP-Referer"] == "https://github.com/zwldarren/llm-proxy"
        assert headers["X-OpenRouter-Title"] == "LLM Proxy"

    def test_typed_config_overrides_defaults(self):
        adapter = OpenRouterAdapter(
            api_key="sk-or-test",
            app_attribution={
                "url": "https://my.app",
                "title": "My App",
                "categories": ["cli-agent", "cloud-agent"],
                "visibility": "hidden",
            },
        )
        headers = adapter._build_headers()
        assert headers["HTTP-Referer"] == "https://my.app"
        assert headers["X-OpenRouter-Title"] == "My App"
        assert headers["X-OpenRouter-Categories"] == "cli-agent,cloud-agent"
        assert headers["X-OpenRouter-App-Visibility"] == "hidden"

    def test_custom_headers_win_over_typed_config(self):
        adapter = OpenRouterAdapter(
            api_key="sk-or-test",
            custom_headers={"HTTP-Referer": "https://explicit.example"},
            app_attribution={"url": "https://my.app", "title": "My App"},
        )
        headers = adapter._build_headers()
        assert headers["HTTP-Referer"] == "https://explicit.example"
        # The title still comes from the typed config: only the collision loses.
        assert headers["X-OpenRouter-Title"] == "My App"

    def test_legacy_x_title_is_respected(self):
        adapter = OpenRouterAdapter(
            api_key="sk-or-test",
            custom_headers={"X-Title": "Legacy Name"},
        )
        headers = adapter._build_headers()
        assert headers["X-Title"] == "Legacy Name"
        assert "X-OpenRouter-Title" not in headers

    def test_visibility_ships_with_the_default_referer(self):
        """A hidden app still needs an ``HTTP-Referer`` to be attributed at all."""
        adapter = OpenRouterAdapter(
            api_key="sk-or-test",
            app_attribution={"visibility": "hidden"},
        )
        headers = adapter._build_headers()
        assert headers["HTTP-Referer"] == "https://github.com/zwldarren/llm-proxy"
        assert headers["X-OpenRouter-App-Visibility"] == "hidden"

    def test_categories_accept_a_list_or_a_string(self):
        from_list = OpenRouterAdapter(
            api_key="k", app_attribution={"categories": ["cli-agent", "cloud-agent"]}
        )._build_headers()
        from_string = OpenRouterAdapter(
            api_key="k", app_attribution={"categories": "cli-agent"}
        )._build_headers()
        assert from_list["X-OpenRouter-Categories"] == "cli-agent,cloud-agent"
        assert from_string["X-OpenRouter-Categories"] == "cli-agent"


# ---------------------------------------------------------------------------
# Client control headers
# ---------------------------------------------------------------------------


class TestClientHeaderForwarding:
    def test_openrouter_control_headers_are_forwarded(self, adapter):
        capture_client_headers(
            {
                "X-OpenRouter-Metadata": "enabled",
                "X-OpenRouter-Cache": "true",
                "X-OpenRouter-Cache-TTL": "600",
            }
        )
        headers = adapter._build_headers()
        assert headers["X-OpenRouter-Metadata"] == "enabled"
        assert headers["X-OpenRouter-Cache"] == "true"
        assert headers["X-OpenRouter-Cache-TTL"] == "600"

    def test_client_fingerprint_headers_are_not_forwarded_to_a_router(self, adapter):
        """Codex headers describe the client's original target, not OpenRouter."""
        capture_client_headers(
            {
                "originator": "codex_cli_rs",
                "X-Codex-Beta-Features": "shell_tool",
                "User-Agent": "codex/1.0",
                "X-OpenRouter-Cache": "true",
            }
        )
        headers = adapter._build_headers()
        assert headers["X-OpenRouter-Cache"] == "true"
        assert "originator" not in headers
        assert "X-Codex-Beta-Features" not in headers
        assert "User-Agent" not in headers

    def test_client_cannot_spoof_auth_or_attribution(self, adapter):
        capture_client_headers(
            {
                "Authorization": "Bearer attacker",
                "HTTP-Referer": "https://attacker.example",
                "X-OpenRouter-Title": "Attacker App",
                "X-Title": "Attacker App",
                "X-OpenRouter-Categories": "malware",
                "X-OpenRouter-App-Visibility": "hidden",
            }
        )
        headers = adapter._build_headers()
        # auth comes from the provider config, attribution from the operator.
        assert headers["Authorization"] == "Bearer sk-or-test"
        assert headers["HTTP-Referer"] == "https://github.com/zwldarren/llm-proxy"
        assert headers["X-OpenRouter-Title"] == "LLM Proxy"
        assert "X-Title" not in headers
        assert "X-OpenRouter-Categories" not in headers
        assert "X-OpenRouter-App-Visibility" not in headers


# ---------------------------------------------------------------------------
# Chat Completions extras that OpenRouter documents
# ---------------------------------------------------------------------------


class TestChatExemptExtraKeys:
    def test_documented_fields_survive_the_ignore_policy(self, adapter):
        """``unknown_fields_policy="ignore"`` must not eat OpenRouter's own fields."""
        raw = {
            "model": "openai/gpt-5.6",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning": {"mode": "pro"},
            "reasoning_effort": "high",
            "provider": {"sort": "throughput", "zdr": True},
            "models": ["a/b"],
            "plugins": [{"id": "web"}],
            "usage": {"include": True},
            "min_p": 0.05,
            "definitely_not_a_field": 1,
        }
        request = get_protocol_serializer("openai").parse_request(raw)
        request.metadata.protocol_name = "openai"

        body = adapter._build_request_body(request)

        assert body["reasoning"] == {"mode": "pro", "effort": "high"}
        assert "reasoning_effort" not in body
        assert body["provider"] == {"sort": "throughput", "zdr": True}
        assert body["models"] == ["a/b"]
        assert body["plugins"] == [{"id": "web"}]
        assert body["usage"] == {"include": True}
        assert body["min_p"] == 0.05
        # Genuinely unknown fields are still dropped by the default policy.
        assert "definitely_not_a_field" not in body


# ---------------------------------------------------------------------------
# Per-request control headers that do not carry the x-openrouter- prefix
# ---------------------------------------------------------------------------


class TestSessionAndBetaHeaders:
    def test_session_id_header_is_forwarded(self, adapter):
        """Sticky routing / cache affinity may be set by header, not only body."""
        capture_client_headers({"x-session-id": "sess-123"})
        assert adapter._build_headers()["x-session-id"] == "sess-123"

    def test_anthropic_beta_header_is_forwarded(self, adapter):
        """Without structured-outputs-* OpenRouter strips ``strict`` from tools."""
        capture_client_headers({"x-anthropic-beta": "structured-outputs-2025-11-13"})
        assert adapter._build_headers()["x-anthropic-beta"] == "structured-outputs-2025-11-13"


# ---------------------------------------------------------------------------
# Documented non-OpenAI fields on the non-chat endpoints
# ---------------------------------------------------------------------------


class TestEndpointExemptExtraKeys:
    def test_image_generation_extras_survive_the_ignore_policy(self, adapter):
        request = InternalImageRequest(
            model="google/gemini-2.5-flash-image",
            prompt="a cat",
            extra={
                "resolution": "2K",
                "aspect_ratio": "16:9",
                "seed": 7,
                "provider": {"sort": "price"},
            },
        )

        body = adapter._build_outbound_body(request, request_type="image_generation").json_body

        assert body["resolution"] == "2K"
        assert body["aspect_ratio"] == "16:9"
        assert body["seed"] == 7
        assert body["provider"] == {"sort": "price"}

    def test_speech_extras_survive_the_ignore_policy(self, adapter):
        request = InternalSpeechRequest(
            model="fish-audio/s2.1-pro",
            input="hello",
            voice="reference",
            extra={
                "input_references": [{"type": "input_audio"}],
                "provider": {"options": {"openai": {"instructions": "warm"}}},
            },
        )

        body = adapter._build_outbound_body(request, request_type="speech").json_body

        assert body["input_references"] == [{"type": "input_audio"}]
        assert body["provider"] == {"options": {"openai": {"instructions": "warm"}}}

    def test_embedding_extras_survive_the_ignore_policy(self, adapter):
        request = InternalEmbeddingRequest(
            model="openai/text-embedding-3-small",
            input="hello",
            extra={
                "input_type": "search_query",
                "session_id": "sess-1",
                "trace": {"trace_id": "t-1"},
                "provider": {"order": ["openai"]},
            },
        )

        body = adapter._build_outbound_body(request, request_type="embedding").json_body

        assert body["input_type"] == "search_query"
        assert body["session_id"] == "sess-1"
        assert body["trace"] == {"trace_id": "t-1"}
        assert body["provider"] == {"order": ["openai"]}

    @pytest.mark.asyncio
    async def test_transcription_provider_options_reach_the_wire(self, adapter, monkeypatch):
        """The JSON STT body is finalized by the adapter, not the chokepoint."""
        sent: dict = {}

        async def _post(url, headers=None, json=None, **kwargs):
            sent.update(json)

            class _Response:
                status_code = 200
                headers = {"content-type": "application/json"}

                def json(self):
                    return {"text": "hi"}

            return _Response()

        client = await adapter._get_client()
        monkeypatch.setattr(client, "post", _post)

        await adapter.transcription(
            InternalTranscriptionRequest(
                model="openai/whisper-large-v3",
                file=b"audio",
                filename="clip.wav",
                extra={"provider": {"options": {"groq": {"prompt": "vocab"}}}},
            )
        )

        assert sent["provider"] == {"options": {"groq": {"prompt": "vocab"}}}
