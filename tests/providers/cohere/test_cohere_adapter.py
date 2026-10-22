"""Cohere adapter tests: registration, endpoints, and the four request paths."""

import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_proxy.core.adapter import get_adapter, list_provider_types, list_providers
from llm_proxy.core.exceptions import ProviderError
from llm_proxy.core.request_type import RequestType
from llm_proxy.models import (
    ConversationContext,
    InternalEmbeddingRequest,
    InternalRequest,
    InternalRerankRequest,
    InternalSpeechRequest,
    InternalTranscriptionRequest,
    Message,
    TextBlock,
)
from llm_proxy.providers.cohere import CohereAdapter
from llm_proxy.providers.cohere.serializer import COHERE_CHAT_EXTRA_KEYS

SSE_LINES = [
    b'data: {"type":"message-start","id":"c14c","delta":{"message":{"role":"assistant"}}}\n',
    b"event: content-delta\n",
    b'data: {"type":"content-delta","index":0,"delta":{"message":{"content":{"text":"Hi"}}}}\n',
    b"\n",
    b'data: {"type":"message-end","delta":{"finish_reason":"COMPLETE",'
    b'"usage":{"tokens":{"input_tokens":4,"output_tokens":1}}}}\n',
]


@pytest.fixture
def adapter() -> CohereAdapter:
    return CohereAdapter(api_key="test-key")


def _chat_request() -> InternalRequest:
    request = InternalRequest(
        model="command-a-plus-05-2026",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="hi")])]
        ),
    )
    request.metadata.protocol_name = "openai"
    return request


class TestDeclaration:
    def test_registered_under_its_type(self):
        assert "cohere" in list_providers()
        assert isinstance(get_adapter("cohere", api_key="k"), CohereAdapter)

    def test_provider_name_is_the_registry_name(self, adapter):
        assert adapter.provider_name == "cohere"

    def test_default_base_url(self, adapter):
        assert adapter._base_url == "https://api.cohere.com"

    def test_catalog_branding(self):
        info = next(info for info in list_provider_types() if info.type == "cohere")

        assert info.name_en == "Cohere"
        assert info.name_zh == "Cohere"
        assert info.icon_id == "cohere"
        assert info.icon_variant == "color"

    def test_no_native_passthrough_protocols(self, adapter):
        """Cohere serves neither Anthropic Messages nor OpenAI Responses."""
        assert adapter.native_protocols == frozenset()
        for protocol in ("openai", "anthropic", "openresponses"):
            assert adapter.supports_native_request(protocol) is False
            assert adapter.supports_native_streaming(protocol) is False

    def test_model_listing_stays_on_the_documented_v1_route(self, adapter):
        assert adapter._models_url() == "https://api.cohere.com/v1/models"

    def test_cohere_only_fields_are_exempt_from_the_field_policy(self, adapter):
        exempt = adapter.EXEMPT_EXTRA_KEYS[RequestType.CHAT]

        assert exempt == COHERE_CHAT_EXTRA_KEYS
        assert {"documents", "citation_options", "thinking"} <= exempt | {"thinking"}

    def test_bearer_auth(self, adapter):
        assert adapter._build_headers()["Authorization"] == "Bearer test-key"


class TestChatCompletion:
    @pytest.mark.asyncio
    async def test_round_trip(self, adapter, mock_response_cls):
        upstream = {
            "id": "c14c",
            "finish_reason": "COMPLETE",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
            "usage": {"tokens": {"input_tokens": 4, "output_tokens": 2}},
        }
        client = MagicMock()
        client.post = AsyncMock(return_value=mock_response_cls(json_data=upstream))

        with patch.object(adapter, "_get_client", return_value=client):
            result = await adapter.chat_completion(_chat_request())

        call = client.post.call_args
        assert call.args[0] == "https://api.cohere.com/v2/chat"
        assert call.kwargs["json"]["model"] == "command-a-plus-05-2026"
        assert call.kwargs["json"]["stream"] is False
        assert result.output[0].text == "hello"
        assert result.usage.total_tokens == 6

    @pytest.mark.asyncio
    async def test_cohere_error_bodies_keep_their_message(self, adapter, mock_response_cls):
        upstream = {"message": "invalid api token", "id": "e1"}
        client = MagicMock()
        client.post = AsyncMock(
            return_value=mock_response_cls(
                status_code=401, json_data=upstream, text_data=json.dumps(upstream)
            )
        )

        with (
            patch.object(adapter, "_get_client", return_value=client),
            pytest.raises(ProviderError) as caught,
        ):
            await adapter.chat_completion(_chat_request())

        assert caught.value.message == "invalid api token"
        assert caught.value.error_type == "authentication_error"
        assert caught.value.status_code == 401


class TestStreaming:
    @pytest.mark.asyncio
    async def test_event_stream_becomes_canonical_chunks_ending_in_done(self, adapter):
        response = _stream_response(SSE_LINES)

        @asynccontextmanager
        async def fake_streaming_post(*_args, **_kwargs):
            yield response

        with (
            patch.object(adapter, "_get_client", return_value=MagicMock()),
            patch.object(adapter._transport, "streaming_post", fake_streaming_post),
        ):
            stream = await adapter.stream_chat_completion(_chat_request())
            chunks = [chunk async for chunk in stream]

        # Cohere never sends [DONE]: the adapter appends it after the terminal
        # event it converts itself.
        assert chunks[-1] == "[DONE]"
        content = [c for c in chunks[:-1] if c["choices"][0]["delta"].get("content")]
        assert content[0]["choices"][0]["delta"]["content"] == "Hi"
        terminal = chunks[-2]
        assert terminal["choices"][0]["finish_reason"] == "stop"
        assert terminal["usage"]["total_tokens"] == 5

    @pytest.mark.asyncio
    async def test_the_request_body_asks_for_a_stream(self, adapter):
        response = _stream_response(
            [b'data: {"type":"message-end","delta":{"finish_reason":"COMPLETE"}}\n']
        )
        captured: dict = {}

        @asynccontextmanager
        async def fake_streaming_post(*_args, **kwargs):
            captured.update(kwargs)
            yield response

        with (
            patch.object(adapter, "_get_client", return_value=MagicMock()),
            patch.object(adapter._transport, "streaming_post", fake_streaming_post),
        ):
            stream = await adapter.stream_chat_completion(_chat_request())
            [_ async for _ in stream]

        assert captured["json"]["stream"] is True
        assert captured["json"]["messages"] == [{"role": "user", "content": "hi"}]


class TestEmbeddings:
    @pytest.mark.asyncio
    async def test_request_and_response(self, adapter, mock_response_cls):
        upstream = {
            "id": "e1",
            "embeddings": {"float": [[0.1], [0.2]]},
            "meta": {"tokens": {"input_tokens": 3}},
        }
        client = MagicMock()
        client.post = AsyncMock(return_value=mock_response_cls(json_data=upstream))
        request = InternalEmbeddingRequest(model="embed-v4.0", input=["a", "b"])

        with patch.object(adapter, "_get_client", return_value=client):
            result = await adapter.embeddings(request)

        call = client.post.call_args
        assert call.args[0] == "https://api.cohere.com/v2/embed"
        assert call.kwargs["json"]["texts"] == ["a", "b"]
        assert call.kwargs["json"]["input_type"] == "search_document"
        assert [item.embedding for item in result.data] == [[0.1], [0.2]]


class TestRerank:
    @pytest.mark.asyncio
    async def test_documents_v2_does_not_echo_are_projected_back(self, adapter, mock_response_cls):
        upstream = {
            "id": "r1",
            "results": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.1}],
        }
        client = MagicMock()
        client.post = AsyncMock(return_value=mock_response_cls(json_data=upstream))
        request = InternalRerankRequest(
            model="rerank-v5.0-pro", query="capital?", documents=["paris", "berlin"]
        )

        with patch.object(adapter, "_get_client", return_value=client):
            result = await adapter.rerank(request)

        call = client.post.call_args
        assert call.args[0] == "https://api.cohere.com/v2/rerank"
        assert call.kwargs["json"]["documents"] == ["paris", "berlin"]
        assert [ranked.document.text for ranked in result.results] == ["berlin", "paris"]

    @pytest.mark.asyncio
    async def test_return_documents_false_still_drops_the_projection(
        self, adapter, mock_response_cls
    ):
        upstream = {"results": [{"index": 0, "relevance_score": 0.9}]}
        client = MagicMock()
        client.post = AsyncMock(return_value=mock_response_cls(json_data=upstream))
        request = InternalRerankRequest(
            model="rerank-v5.0-pro", query="q", documents=["paris"], return_documents=False
        )

        with patch.object(adapter, "_get_client", return_value=client):
            result = await adapter.rerank(request)

        assert result.results[0].document is None

    @pytest.mark.asyncio
    async def test_relevant_spans_survive_the_document_projection(self, adapter, mock_response_cls):
        # The spans are the upstream's answer to a request field the adapter
        # forwards; projecting the documents back must not take them with it,
        # and ``return_documents: false`` must not either (spans are not
        # documents).
        spans = [{"start": 0, "end": 5, "score": 0.9}]
        upstream = {"results": [{"index": 0, "relevance_score": 0.9, "relevant_spans": spans}]}
        client = MagicMock()
        client.post = AsyncMock(return_value=mock_response_cls(json_data=upstream))

        for return_documents in (None, False):
            request = InternalRerankRequest(
                model="rerank-v5.0-pro",
                query="q",
                documents=["paris"],
                return_documents=return_documents,
            )
            with patch.object(adapter, "_get_client", return_value=client):
                result = await adapter.rerank(request)
            assert result.results[0].relevant_spans == spans

    @pytest.mark.asyncio
    async def test_meta_is_projected_onto_the_shared_usage_echo(self, adapter, mock_response_cls):
        # v2 answers ``meta.tokens``/``billed_units``; the client-facing rerank
        # wire reports ``total_tokens``/``search_units``/``cost`` (see
        # docs/api/rerank.md), so the vendor envelope is re-shaped and not just
        # echoed.
        upstream = {
            "id": "r1",
            "results": [{"index": 0, "relevance_score": 0.9}],
            "meta": {
                "tokens": {"input_tokens": 120, "output_tokens": 0},
                "billed_units": {"search_units": 1},
                "cost": 0.001,
            },
        }
        client = MagicMock()
        client.post = AsyncMock(return_value=mock_response_cls(json_data=upstream))
        request = InternalRerankRequest(model="rerank-v5.0-pro", query="q", documents=["paris"])

        with patch.object(adapter, "_get_client", return_value=client):
            result = await adapter.rerank(request)

        assert result.provider_info["rerank_usage"] == {
            "total_tokens": 120,
            "search_units": 1,
            "cost": 0.001,
        }
        assert result.provider_info["openrouter_cost"] == 0.001

    @pytest.mark.asyncio
    async def test_a_meta_block_without_counters_echoes_nothing(self, adapter, mock_response_cls):
        upstream = {
            "results": [{"index": 0, "relevance_score": 0.9}],
            "meta": {"api_version": {"version": "2"}},
        }
        client = MagicMock()
        client.post = AsyncMock(return_value=mock_response_cls(json_data=upstream))
        request = InternalRerankRequest(model="rerank-v5.0-pro", query="q", documents=["paris"])

        with patch.object(adapter, "_get_client", return_value=client):
            result = await adapter.rerank(request)

        assert "rerank_usage" not in result.provider_info


class TestTranscription:
    @pytest.mark.asyncio
    async def test_multipart_form_carries_only_the_documented_fields(
        self, adapter, mock_response_cls
    ):
        response = mock_response_cls(json_data={"text": "hello"})
        response.headers = {"content-type": "application/json"}
        client = MagicMock()
        client.post = AsyncMock(return_value=response)
        request = InternalTranscriptionRequest(
            model="cohere-transcribe-03-2026",
            file=b"audio",
            filename="a.wav",
            language="en",
            prompt="keep this out",
            response_format="verbose_json",
        )

        with patch.object(adapter, "_get_client", return_value=client):
            result = await adapter.transcription(request)

        call = client.post.call_args
        assert call.args[0] == "https://api.cohere.com/v2/audio/transcriptions"
        assert call.kwargs["data"] == {"model": "cohere-transcribe-03-2026", "language": "en"}
        assert call.kwargs["files"] == {"file": ("a.wav", b"audio")}
        assert "Content-Type" not in call.kwargs["headers"]
        assert result.text == "hello"

    @pytest.mark.asyncio
    async def test_a_missing_language_is_refused_locally(self, adapter):
        request = InternalTranscriptionRequest(
            model="cohere-transcribe-03-2026", file=b"audio", filename="a.wav"
        )

        with pytest.raises(ProviderError, match="requires 'language'"):
            await adapter.transcription(request)

    @pytest.mark.asyncio
    async def test_streaming_transcription_is_refused(self, adapter):
        request = InternalTranscriptionRequest(
            model="cohere-transcribe-03-2026", file=b"audio", filename="a.wav", language="en"
        )

        with pytest.raises(ProviderError, match="does not support streaming"):
            await adapter.stream_transcription(request)

    @pytest.mark.asyncio
    async def test_speech_has_no_native_endpoint(self, adapter):
        request = InternalSpeechRequest(model="tts", input="hi", voice="alloy")

        with pytest.raises(NotImplementedError):
            await adapter.speech(request)


class TestModelListing:
    @pytest.mark.asyncio
    async def test_catalog_pagination_uses_cohere_query_names(self, adapter, mock_response_cls):
        pages = [
            mock_response_cls(
                json_data={
                    "models": [{"name": "command-a-plus-05-2026", "context_length": 128000}],
                    "next_page_token": "p2",
                }
            ),
            mock_response_cls(json_data={"models": [{"name": "embed-v4.0"}]}),
        ]
        client = MagicMock()
        client.request = AsyncMock(side_effect=pages)

        with patch.object(adapter, "_get_client", return_value=client):
            models = await adapter.list_models()

        first = client.request.call_args_list[0]
        second = client.request.call_args_list[1]
        assert first.kwargs["params"] == {"page_size": 1000}
        assert second.kwargs["params"] == {"page_size": 1000, "page_token": "p2"}
        assert [model.id for model in models] == ["command-a-plus-05-2026", "embed-v4.0"]
        assert models[0].context_length == 128000


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _stream_response(lines: list[bytes]):
    response = MagicMock()
    response.status_code = 200
    response.headers = {}
    response.iter_lines = lambda: _aiter(lines)
    return response


async def _aiter(lines):
    for line in lines:
        yield line
