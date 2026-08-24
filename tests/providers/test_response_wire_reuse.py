"""End-to-end tests for the response-side WIRE_REUSE tier (chat completions).

When the provider answers in the client's own protocol
(``serializer.compatible_protocols``), the non-streaming response rides
verbatim — mirroring the streaming path's fidelity. Load-bearing behavior
is pinned here: reasoning-field rename, model aliasing, usage extraction
for billing (parity with the parsed tier), the reasoning cache write, the
``_post_process_chat_response`` hook (provider cost metadata), the
``response_passthrough`` kill switch, and the ``native_request_disabled``
veto (the web-search continuation needs a parsed InternalResponse).
"""

import copy
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_proxy.core import reasoning_cache
from llm_proxy.models import (
    ConversationContext,
    ConversionTier,
    GenerationParams,
    InternalRequest,
    Message,
    TextBlock,
)
from llm_proxy.providers.deepseek.adapter import DeepSeekAdapter
from llm_proxy.providers.openai.adapter import OpenAIAdapter
from llm_proxy.providers.openrouter.adapter import OpenRouterAdapter


@pytest.fixture(autouse=True)
def _clear_reasoning_cache():
    reasoning_cache.clear()
    yield
    reasoning_cache.clear()


def _chat_request(model: str = "openai/gpt-x") -> InternalRequest:
    req = InternalRequest(
        model=model,
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="hi")])]
        ),
        params=GenerationParams(),
    )
    req.metadata.protocol_name = "openai"
    return req


def _upstream_body() -> dict:
    return {
        "id": "chatcmpl-9",
        "model": "openai/gpt-x-internal",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    # Raw provider field name — the streaming path renames
                    # this to reasoning_content, so the verbatim tier must too.
                    "reasoning": "thinking hard",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5, "cost": 0.01},
    }


class TestChatWireReuseResponse:
    @pytest.mark.asyncio
    async def test_verbatim_body_rename_cache_and_cost(self, mock_response_cls):
        adapter = OpenRouterAdapter(api_key="k")
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response_cls(json_data=_upstream_body()))
        req = _chat_request()
        # Set by the routing layer when an alias is involved; the verbatim
        # body is then masked to echo the client-facing name.
        req.user_facing_model = "openai/gpt-x"

        with patch.object(adapter, "_get_client", return_value=mock_client):
            result = await adapter.chat_completion(req)

        assert req.response_tier == ConversionTier.WIRE_REUSE
        raw = result.provider_info["_raw_response_body"]
        msg = raw["choices"][0]["message"]
        # Streaming parity: reasoning field renamed, model alias rewritten.
        assert msg["reasoning_content"] == "thinking hard"
        assert "reasoning" not in msg
        assert raw["model"] == "openai/gpt-x"
        # Verbatim tier never produces parsed output blocks.
        assert result.output == []
        # Reasoning cache written from the wire shape, keyed by tool-call id.
        assert reasoning_cache.get("call_1") == "thinking hard"
        # The OpenRouter cost hook runs on the verbatim path (billing).
        assert result.provider_info["openrouter_cost"] == 0.01
        # Usage still parsed for billing.
        assert result.usage.total_tokens == 5

    @pytest.mark.asyncio
    async def test_model_alias_injected_when_upstream_omits_model(self, mock_response_cls):
        """Streaming parity: the transformer sets the alias even when the
        upstream omits ``model``, so the verbatim tier must too."""
        body = _upstream_body()
        del body["model"]
        adapter = OpenRouterAdapter(api_key="k")
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response_cls(json_data=body))
        req = _chat_request()
        req.user_facing_model = "openai/gpt-x"

        with patch.object(adapter, "_get_client", return_value=mock_client):
            result = await adapter.chat_completion(req)

        raw = result.provider_info["_raw_response_body"]
        assert raw["model"] == "openai/gpt-x"

    @pytest.mark.asyncio
    async def test_kill_switch_forces_parsed_response(self, mock_response_cls):
        adapter = OpenRouterAdapter(api_key="k", response_passthrough=False)
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response_cls(json_data=_upstream_body()))
        req = _chat_request()

        with patch.object(adapter, "_get_client", return_value=mock_client):
            result = await adapter.chat_completion(req)

        assert req.response_tier == ConversionTier.FULL_CONVERSION
        assert "_raw_response_body" not in result.provider_info
        assert result.output  # parsed blocks present
        # The parsed path's cache write covers the same pairing.
        assert reasoning_cache.get("call_1") == "thinking hard"

    @pytest.mark.asyncio
    async def test_native_request_disabled_forces_parsed_response(self, mock_response_cls):
        """Post-parse mutations (web search, role normalization) veto the
        verbatim response: consumers like the web-search continuation need
        parsed output blocks."""
        adapter = OpenRouterAdapter(api_key="k")
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response_cls(json_data=_upstream_body()))
        req = _chat_request()
        req.native_request_disabled = True

        with patch.object(adapter, "_get_client", return_value=mock_client):
            result = await adapter.chat_completion(req)

        assert req.response_tier == ConversionTier.FULL_CONVERSION
        assert "_raw_response_body" not in result.provider_info
        assert result.output

    def test_usage_parity_with_parsed_tier(self):
        """The same upstream body must yield the same Usage on both tiers
        (DeepSeek cache-hit folding and server_tool_use included)."""
        body = {
            "id": "chatcmpl-1",
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "total_tokens": 110,
                "prompt_cache_hit_tokens": 64,
                "prompt_cache_miss_tokens": 36,
                "server_tool_use": {"web_search_requests": 2},
            },
        }
        adapter = DeepSeekAdapter(api_key="k")
        req = _chat_request(model="deepseek-chat")

        wire_usage, _extras = adapter._parse_passthrough_usage(copy.deepcopy(body), req)
        parsed = adapter._parse_response(
            adapter._get_serializer(), copy.deepcopy(body), model="deepseek-chat", request=req
        )

        assert wire_usage == parsed.usage
        assert wire_usage.prompt_tokens_details.cached_tokens == 64
        assert wire_usage.web_search_requests == 2


class TestOpenAINativeNonStreamReasoningCache:
    """The openai adapter's native (verbatim) non-streaming branch never
    builds parsed blocks, so the reasoning cache is written from the raw
    output items — the non-stream counterpart of the terminal-snapshot
    stream write."""

    @pytest.mark.asyncio
    async def test_native_passthrough_feeds_cache(self):
        adapter = OpenAIAdapter(api_key="k")
        raw = {"model": "gpt-alias", "input": [{"role": "user", "content": "hi"}]}
        req = InternalRequest(
            model="gpt-5",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            params=GenerationParams(),
        )
        req.metadata.protocol_name = "openresponses"
        req._raw_protocol_data = raw

        upstream = {
            "id": "resp_1",
            "status": "completed",
            "output": [
                {
                    "type": "reasoning",
                    "content": [{"type": "reasoning_text", "text": "hidden chain"}],
                },
                {"type": "function_call", "call_id": "call_7", "name": "f"},
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }
        fake_response = MagicMock()
        fake_response.json.return_value = upstream
        fake_response.headers = {}

        with patch.object(
            adapter, "_post_json_response_with_retry", new=AsyncMock(return_value=fake_response)
        ):
            result = await adapter.chat_completion(req)

        assert result.provider_info["_raw_response_body"] is upstream
        assert reasoning_cache.get("call_7") == "hidden chain"
