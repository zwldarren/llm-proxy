"""Tests for client fingerprint header forwarding on native passthrough paths.

Every ``NativePassthroughChatBase`` adapter (qwen, kimi-code, deepseek,
moonshot, minimax, zai/zhipu coding, ...) merges the client fingerprint
headers captured by the anthropic/openresponses protocol layers into upstream
requests, so Claude Code / Codex identity survives the native passthrough.
Covered here:

* the merge itself: both protocol captures, ``claude-code-20250219`` beta
  marker injection, no capture → unchanged headers, provider headers win;
* inheritance through adapters with their own ``_build_headers`` (GLMBase
  ``x-api-key``, KimiCodeAdapter);
* end-to-end: native completions and streams carry the merged headers.
"""

from unittest.mock import AsyncMock, patch

import pytest

from llm_proxy.providers.anthropic.client_headers import (
    CLAUDE_CODE_BETA,
)
from llm_proxy.providers.anthropic.client_headers import (
    capture_client_headers as capture_anthropic_headers,
)
from llm_proxy.providers.anthropic.client_headers import (
    clear_client_headers as clear_anthropic_headers,
)
from llm_proxy.providers.deepseek import DeepSeekAdapter
from llm_proxy.providers.kimi_code import KimiCodeAdapter
from llm_proxy.providers.openai.client_headers import (
    capture_client_headers as capture_openai_headers,
)
from llm_proxy.providers.openai.client_headers import (
    clear_client_headers as clear_openai_headers,
)
from llm_proxy.providers.zhipu_coding import ZhipuCodingAdapter
from providers.helpers import (
    MockStreamResponse,
    make_request,
    make_sse_events,
    raw_anthropic,
    raw_responses,
)


@pytest.fixture(autouse=True)
def _clean_contextvars():
    clear_anthropic_headers()
    clear_openai_headers()
    yield
    clear_anthropic_headers()
    clear_openai_headers()


@pytest.fixture
def adapter() -> DeepSeekAdapter:
    return DeepSeekAdapter(api_key="sk-upstream")


class TestNativePassthroughHeaderMerge:
    def test_merges_anthropic_client_headers_and_beta_marker(self, adapter):
        capture_anthropic_headers(
            {
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "oauth-2025-04-20",
                "x-app": "claude-code",
                "user-agent": "claude-cli/2.0.14",
                "x-client-request-id": "req_1",
                "x-stainless-runtime": "node",
            }
        )
        headers = adapter._build_headers()
        assert headers["anthropic-version"] == "2023-06-01"
        assert headers["x-app"] == "claude-code"
        assert headers["user-agent"] == "claude-cli/2.0.14"
        assert headers["x-client-request-id"] == "req_1"
        assert headers["x-stainless-runtime"] == "node"
        # Beta is rebuilt to guarantee the claude-code marker.
        assert headers["anthropic-beta"] == "claude-code-20250219,oauth-2025-04-20"
        # Provider auth is never touched by the merge.
        assert headers["Authorization"] == "Bearer sk-upstream"

    def test_injects_beta_marker_when_client_sent_none(self, adapter):
        capture_anthropic_headers({"x-app": "claude-code"})
        headers = adapter._build_headers()
        assert headers["anthropic-beta"] == CLAUDE_CODE_BETA

    def test_merges_openai_responses_client_headers(self, adapter):
        capture_openai_headers(
            {
                "originator": "codex",
                "openai-beta": "responses-v1",
                "conversation_id": "conv_1",
                "session_id": "sess_1",
                "chatgpt-account-id": "acct_1",
                "user-agent": "codex/1.2.3",
                "x-codex-app": "cli",
                "x-stainless-runtime": "node",
            }
        )
        headers = adapter._build_headers()
        assert headers["originator"] == "codex"
        assert headers["openai-beta"] == "responses-v1"
        assert headers["conversation_id"] == "conv_1"
        assert headers["session_id"] == "sess_1"
        assert headers["chatgpt-account-id"] == "acct_1"
        assert headers["user-agent"] == "codex/1.2.3"
        assert headers["x-codex-app"] == "cli"
        assert headers["x-stainless-runtime"] == "node"
        # No anthropic capture → no beta injection on the Responses path.
        assert "anthropic-beta" not in headers

    def test_without_capture_is_unchanged(self, adapter):
        headers = adapter._build_headers()
        assert headers == {
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-upstream",
        }

    def test_client_headers_never_override_provider_headers(self):
        adapter = DeepSeekAdapter(api_key="sk-upstream", custom_headers={"x-app": "custom-app"})
        capture_anthropic_headers({"x-app": "claude-code", "user-agent": "claude-cli/2.0"})
        headers = adapter._build_headers()
        # Case-insensitive collision: provider's own header is kept.
        assert headers["x-app"] == "custom-app"
        assert headers["user-agent"] == "claude-cli/2.0"

    def test_both_protocol_captures_merge_without_clobbering(self, adapter):
        capture_anthropic_headers({"x-app": "claude-code", "user-agent": "claude-cli/2.0"})
        capture_openai_headers({"originator": "codex", "user-agent": "codex/1.2.3"})
        headers = adapter._build_headers()
        # The anthropic capture wins the shared ``user-agent`` slot (merged first).
        assert headers["x-app"] == "claude-code"
        assert headers["originator"] == "codex"
        assert headers["user-agent"] == "claude-cli/2.0"
        assert headers["anthropic-beta"] == CLAUDE_CODE_BETA


class TestInheritedBuildHeaders:
    def test_glm_base_chain_merges_and_keeps_x_api_key(self):
        adapter = ZhipuCodingAdapter(api_key="sk-upstream")
        capture_anthropic_headers({"anthropic-version": "2023-06-01", "x-app": "claude-code"})
        headers = adapter._build_headers()
        assert headers["x-api-key"] == "sk-upstream"
        assert headers["anthropic-version"] == "2023-06-01"
        assert headers["x-app"] == "claude-code"
        assert headers["anthropic-beta"] == CLAUDE_CODE_BETA

    def test_kimi_code_adapter_merges_and_keeps_x_api_key(self):
        adapter = KimiCodeAdapter(api_key="sk-upstream")
        capture_anthropic_headers({"x-app": "claude-code", "user-agent": "claude-cli/2.0"})
        headers = adapter._build_headers()
        assert headers["x-api-key"] == "sk-upstream"
        assert headers["x-app"] == "claude-code"
        assert headers["user-agent"] == "claude-cli/2.0"
        assert headers["anthropic-beta"] == CLAUDE_CODE_BETA


class TestNativeRequestsCarryClientHeaders:
    @pytest.mark.asyncio
    async def test_anthropic_native_completion_sends_merged_headers(
        self, adapter, mock_response_cls
    ):
        upstream = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 7, "output_tokens": 3},
        }
        raw = raw_anthropic()
        req = make_request(raw, model="deepseek-v4", protocol_name="anthropic")

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response_cls(json_data=upstream))
        capture_anthropic_headers({"anthropic-version": "2023-06-01", "x-app": "claude-code"})
        with patch.object(adapter, "_get_client", return_value=mock_client):
            await adapter.chat_completion(req)

        sent = mock_client.post.call_args.kwargs
        assert sent["headers"]["anthropic-version"] == "2023-06-01"
        assert sent["headers"]["x-app"] == "claude-code"
        assert sent["headers"]["anthropic-beta"] == CLAUDE_CODE_BETA
        assert sent["headers"]["Authorization"] == "Bearer sk-upstream"

    @pytest.mark.asyncio
    async def test_openresponses_native_completion_sends_merged_headers(
        self, adapter, mock_response_cls
    ):
        upstream = {
            "id": "resp_1",
            "object": "response",
            "model": "deepseek-v4",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hi"}],
                }
            ],
            "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        }
        raw = raw_responses()
        req = make_request(raw, model="deepseek-v4", protocol_name="openresponses")

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response_cls(json_data=upstream))
        capture_openai_headers({"originator": "codex", "user-agent": "codex/1.2.3"})
        with patch.object(adapter, "_get_client", return_value=mock_client):
            await adapter.chat_completion(req)

        sent = mock_client.post.call_args.kwargs
        assert sent["headers"]["originator"] == "codex"
        assert sent["headers"]["user-agent"] == "codex/1.2.3"
        assert "anthropic-beta" not in sent["headers"]

    @pytest.mark.asyncio
    async def test_native_stream_sends_merged_headers(self, adapter):
        sse_events = make_sse_events(
            [
                (
                    "message_start",
                    '{"type":"message_start","message":{"id":"msg_s","type":"message",'
                    '"role":"assistant","content":[],"model":"deepseek-v4",'
                    '"usage":{"input_tokens":10,"output_tokens":1}}}',
                ),
                ("message_stop", '{"type":"message_stop"}'),
            ]
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=MockStreamResponse(sse_events))

        raw = raw_anthropic()
        req = make_request(raw, model="deepseek-v4", protocol_name="anthropic")
        capture_anthropic_headers({"anthropic-version": "2023-06-01", "x-app": "claude-code"})
        with patch.object(adapter, "_get_client", return_value=mock_client):
            stream_gen = await adapter.stream_chat_completion_native(req)
            frames = [frame async for frame in stream_gen]

        sent = mock_client.post.call_args.kwargs
        assert sent["headers"]["anthropic-version"] == "2023-06-01"
        assert sent["headers"]["x-app"] == "claude-code"
        assert sent["headers"]["anthropic-beta"] == CLAUDE_CODE_BETA
        assert any("message_start" in frame for frame in frames)
