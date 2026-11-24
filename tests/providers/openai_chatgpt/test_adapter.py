"""Tests for OpenAIChatGPTAdapter."""

from unittest.mock import AsyncMock, MagicMock

import pytest

import llm_proxy.providers  # noqa: F401
from llm_proxy.core.adapter import get_adapter, list_provider_types, list_providers
from llm_proxy.core.exceptions import ProviderError
from llm_proxy.models import (
    ConversationContext,
    InternalRequest,
    Message,
    TextBlock,
)
from llm_proxy.providers.openai_chatgpt import OpenAIChatGPTAdapter
from llm_proxy.serialization.openai.serializer import OpenAIResponsesProviderSerializer
from providers.helpers import MockStreamResponse, make_request


def _sse_response_payload(text: str = "Hello from ChatGPT plan") -> str:
    import orjson

    item_event = orjson.dumps(
        {
            "type": "response.output_item.done",
            "item": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            },
        }
    ).decode("utf-8")
    done_event = orjson.dumps(
        {
            "type": "response.completed",
            "response": {
                "id": "resp_oauth_1",
                "status": "completed",
                "model": "gpt-5.4",
                "output": [],
                "usage": {"input_tokens": 12, "output_tokens": 7, "total_tokens": 19},
            },
        }
    ).decode("utf-8")
    return (
        f"event: response.output_item.done\ndata: {item_event}\n\n"
        f"event: response.completed\ndata: {done_event}\n\n"
    )


class TestOpenAIChatGPTAdapter:
    def test_registration_branding_and_serializer(self):
        assert "openai-chatgpt" in list_providers()
        info = next(i for i in list_provider_types() if i.type == "openai-chatgpt")
        assert info.name_en == "OpenAI (ChatGPT OAuth)"
        assert info.icon_id == "openai"
        assert info.icon_variant == "mono"

        adapter = get_adapter("openai-chatgpt")
        assert isinstance(adapter, OpenAIChatGPTAdapter)
        assert adapter._base_url == "https://api.openai.com/v1"
        assert adapter.native_protocols == frozenset({"openresponses"})
        assert isinstance(adapter._get_serializer(), OpenAIResponsesProviderSerializer)

    def test_native_body_hook_enforces_stream_true_and_store_false(self):
        adapter = OpenAIChatGPTAdapter()
        body = adapter.native_body_hook(
            {
                "model": "gpt-5.4",
                "stream": False,
                "store": True,
                "input": [{"id": "item_1", "type": "message", "role": "user", "content": "hi"}],
            }
        )
        assert body["stream"] is True
        assert body["store"] is False
        assert "id" not in body["input"][0]

    def test_parse_error_response_handles_detail_payload(self):
        adapter = OpenAIChatGPTAdapter()
        err = adapter._parse_error_response(429, {"detail": "ChatGPT plan weekly cap reached"})
        assert err.status_code == 429
        assert "ChatGPT plan weekly cap reached" in err.message

    @pytest.mark.asyncio
    async def test_chat_completion_forces_stream_store_false_and_aggregates_sse(self):
        sse_text = _sse_response_payload("Plan reply")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"x-request-id": "req_oauth_1"}
        mock_response.json = MagicMock(side_effect=ValueError("SSE body"))
        mock_response.text = sse_text

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        adapter = OpenAIChatGPTAdapter(
            provider_name="my-chatgpt",
            http_client=mock_client,
            token_provider=lambda: "oauth_access_token_1",
        )
        req = InternalRequest(
            model="gpt-5.4",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Hi")])]
            ),
            stream=False,
        )
        res = await adapter.chat_completion(req)

        assert res.output == [TextBlock(text="Plan reply")]
        assert res.usage is not None
        assert res.usage.input_tokens == 12
        assert res.usage.output_tokens == 7
        assert res.provider_info["_rate_limit_headers"]["x-request-id"] == "req_oauth_1"

        call_kwargs = mock_client.post.call_args
        assert call_kwargs.args[0] == "https://api.openai.com/v1/responses"
        assert call_kwargs.kwargs["headers"]["Authorization"] == "Bearer oauth_access_token_1"
        assert call_kwargs.kwargs["json"]["stream"] is True
        assert call_kwargs.kwargs["json"]["store"] is False

    @pytest.mark.asyncio
    async def test_chat_completion_refreshes_once_on_401_and_retries(self):
        unauthorized = MagicMock()
        unauthorized.status_code = 401
        unauthorized.headers = {}
        unauthorized.json = MagicMock(return_value={"detail": "token expired"})
        unauthorized.text = '{"detail":"token expired"}'

        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.headers = {}
        ok_resp.json = MagicMock(side_effect=ValueError("SSE"))
        ok_resp.text = _sse_response_payload("Retried OK")

        mock_client = MagicMock()
        mock_client.post = AsyncMock(side_effect=[unauthorized, ok_resp])

        tokens_seen: list[bool] = []

        def _token_provider(*, force_refresh: bool = False, failed_token: str | None = None):
            tokens_seen.append(force_refresh)
            return "fresh_tok" if force_refresh else "stale_tok"

        adapter = OpenAIChatGPTAdapter(
            provider_name="my-chatgpt",
            http_client=mock_client,
            max_retries=1,
            token_provider=_token_provider,
        )
        req = InternalRequest(
            model="gpt-5.4",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Hi")])]
            ),
        )
        res = await adapter.chat_completion(req)
        assert res.output == [TextBlock(text="Retried OK")]
        assert tokens_seen == [False, True]
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_stream_chat_completion_native_retries_once_on_401(self):
        bad_stream = MockStreamResponse(status_code=401, lines=[])
        bad_stream.aread = AsyncMock()
        bad_stream.text = '{"detail":"expired"}'
        good_stream = MockStreamResponse(
            status_code=200,
            lines=[
                b"event: response.output_text.delta",
                b'data: {"type":"response.output_text.delta","delta":"hi"}',
                b"",
            ],
        )
        mock_client = MagicMock()
        mock_client.post = AsyncMock(side_effect=[bad_stream, good_stream])

        def _token_provider(*, force_refresh: bool = False, failed_token: str | None = None):
            return "tok_new" if force_refresh else "tok_old"

        adapter = OpenAIChatGPTAdapter(
            provider_name="my-chatgpt",
            http_client=mock_client,
            max_retries=1,
            token_provider=_token_provider,
        )
        req = make_request(
            {"model": "gpt-5.4", "input": "Hi", "store": True, "stream": False},
            model="gpt-5.4",
            protocol_name="openresponses",
        )
        req.stream = True
        blocks = [b async for b in await adapter.stream_chat_completion_native(req)]
        assert len(blocks) == 1
        assert "response.output_text.delta" in blocks[0]
        assert mock_client.post.call_count == 2
        sent_body = mock_client.post.call_args_list[1].kwargs["json"]
        assert sent_body["stream"] is True
        assert sent_body["store"] is False

        # Verify persistent 401 does not loop beyond one refresh.
        always_401_client = MagicMock()
        always_401_client.post = AsyncMock(return_value=bad_stream)
        adapter2 = OpenAIChatGPTAdapter(
            provider_name="my-chatgpt",
            http_client=always_401_client,
            max_retries=1,
            token_provider=_token_provider,
        )
        with pytest.raises(ProviderError) as exc_info:
            async for _ in await adapter2.stream_chat_completion_native(req):
                pass
        assert exc_info.value.status_code == 401
        assert always_401_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_list_models_parses_chatgpt_visibility_catalog_and_standard_fallback(self):
        chatgpt_resp = MagicMock()
        chatgpt_resp.status_code = 200
        chatgpt_resp.raise_for_status = MagicMock()
        chatgpt_resp.json = MagicMock(
            return_value={
                "models": [
                    {
                        "slug": "gpt-5.4",
                        "display_name": "GPT-5.4",
                        "visibility": "list",
                        "context_window": 200000,
                    },
                    {
                        "slug": "internal-hidden",
                        "display_name": "Hidden",
                        "visibility": "hide",
                    },
                ]
            }
        )
        client = MagicMock()
        client.request = AsyncMock(return_value=chatgpt_resp)

        adapter = OpenAIChatGPTAdapter(
            provider_name="my-chatgpt",
            token_provider=lambda: "oauth_tok",
        )
        models = await adapter.list_models(client=client)
        assert [m.id for m in models] == ["gpt-5.4"]
        assert models[0].name == "GPT-5.4"
        assert models[0].context_length == 200000

        standard_resp = MagicMock()
        standard_resp.status_code = 200
        standard_resp.raise_for_status = MagicMock()
        standard_resp.json = MagicMock(
            return_value={"data": [{"id": "gpt-4o", "owned_by": "openai"}]}
        )
        client2 = MagicMock()
        client2.request = AsyncMock(return_value=standard_resp)
        models2 = await adapter.list_models(client=client2)
        assert [m.id for m in models2] == ["gpt-4o"]
