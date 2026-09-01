"""Tests for Gemini provider adapter."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest

from llm_proxy.core.adapter import get_adapter, list_providers
from llm_proxy.core.exceptions import ProviderError
from llm_proxy.http.client import download_image_as_base64
from llm_proxy.models import (
    ConversationContext,
    InternalEmbeddingRequest,
    InternalRequest,
    InternalResponse,
    Message,
    ToolUseBlock,
)
from llm_proxy.models.content_blocks import ImageBlock, TextBlock
from llm_proxy.models.embedding import InternalEmbeddingResponse
from llm_proxy.models.image import ImageEditSource, ImageSize, InternalImageEditRequest
from llm_proxy.models.types import ImageSource
from llm_proxy.providers.gemini import GeminiAdapter  # noqa: F401 - triggers registration
from llm_proxy.serialization.gemini.serializer import GeminiProviderSerializer
from llm_proxy.serialization.gemini.streaming_converter import GeminiStreamingTransformer


def test_gemini_adapter_is_registered():
    """Test that Gemini adapter is registered."""
    assert "gemini" in list_providers()


def test_gemini_adapter_can_be_created():
    """Test that Gemini adapter can be instantiated."""
    adapter = get_adapter("gemini", api_key="test-key")
    assert adapter.__class__.__name__ == "GeminiAdapter"


class TestGeminiStreamingUsage:
    """Tests for Gemini streaming usage extraction."""

    def test_usage_metadata_extracted_in_streaming_chunk(self):
        """Test that usageMetadata is extracted from Gemini streaming chunk."""
        chunk = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "Hello"}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 100,
                "candidatesTokenCount": 50,
                "totalTokenCount": 150,
            },
        }

        transformer = GeminiStreamingTransformer(model="gemini-2.0-flash")
        result_str = transformer.transform_chunk(chunk)
        assert result_str is not None

        result = orjson.loads(result_str.replace("data: ", "").strip())
        chunk_data = result

        assert "usage" in chunk_data
        assert chunk_data["usage"]["prompt_tokens"] == 100
        assert chunk_data["usage"]["completion_tokens"] == 50

    def test_usage_with_cached_tokens(self):
        """Test that cachedContentTokenCount is extracted."""
        chunk = {
            "candidates": [
                {
                    "content": {"parts": [{"text": ""}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 1000,
                "candidatesTokenCount": 200,
                "totalTokenCount": 1200,
                "cachedContentTokenCount": 500,
            },
        }

        transformer = GeminiStreamingTransformer(model="gemini-2.0-flash")
        result_str = transformer.transform_chunk(chunk)
        assert result_str is not None

        result = orjson.loads(result_str.replace("data: ", "").strip())
        chunk_data = result

        assert "usage" in chunk_data
        assert chunk_data["usage"]["prompt_tokens"] == 1000
        assert chunk_data["usage"]["completion_tokens"] == 200

    def test_no_usage_when_not_present(self):
        """Test that no usage field when usageMetadata is absent."""
        chunk = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "Hello"}]},
                }
            ],
        }

        transformer = GeminiStreamingTransformer(model="gemini-2.0-flash")
        result_str = transformer.transform_chunk(chunk)
        assert result_str is not None

        result = orjson.loads(result_str.replace("data: ", "").strip())
        chunk_data = result

        assert "usage" not in chunk_data

    def test_empty_candidates_returns_none(self):
        """Test that empty candidates returns None (filtered out)."""
        chunk = {
            "candidates": [],
            "usageMetadata": {
                "promptTokenCount": 0,
                "candidatesTokenCount": 0,
            },
        }

        transformer = GeminiStreamingTransformer(model="gemini-2.0-flash")
        result_str = transformer.transform_chunk(chunk)
        assert result_str is None


class TestDownloadedImagesUsedInRequest:
    """Tests to verify downloaded images are used in requests, not discarded."""

    @pytest.mark.asyncio
    async def test_chat_completion_uses_downloaded_images(self):
        """Verify downloaded images are patched into Gemini request contents."""
        adapter = GeminiAdapter(api_key="test-key")

        message_with_url = Message(
            role="user",
            content=[
                TextBlock(text="What's in this image?"),
                ImageBlock(
                    source=ImageSource(
                        type="url", data="https://example.com/image.png", media_type=None
                    )
                ),
            ],
        )
        request = InternalRequest(
            model="gemini-2.0-flash",
            conversation=ConversationContext(messages=[message_with_url]),
        )

        patched_contents = [
            {
                "role": "user",
                "parts": [
                    {"text": "What's in this image?"},
                    {"inline_data": {"mime_type": "image/png", "data": "ABC123"}},
                ],
            }
        ]

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "A test image"}]}}],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
            },
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(
                adapter, "_download_images_in_gemini_contents", return_value=patched_contents
            ),
            patch.object(adapter, "_get_client", return_value=mock_client),
        ):
            await adapter.chat_completion(request)

        post_call = mock_client.post.call_args
        body = post_call.kwargs["json"]

        contents = body.get("contents", [])
        assert len(contents) > 0, "Request body should have contents"

        user_content = contents[0].get("parts", [])
        image_parts = [p for p in user_content if "inline_data" in p]
        assert len(image_parts) > 0, (
            "Request should contain inline_data from downloaded image, not the original URL"
        )
        assert image_parts[0]["inline_data"]["data"] == "ABC123"

    @pytest.mark.asyncio
    async def test_stream_chat_completion_uses_downloaded_images(self):
        """Verify downloaded images are patched into streaming Gemini request."""
        adapter = GeminiAdapter(api_key="test-key")

        message_with_url = Message(
            role="user",
            content=[
                TextBlock(text="Describe this"),
                ImageBlock(
                    source=ImageSource(
                        type="url", data="https://example.com/photo.jpg", media_type=None
                    )
                ),
            ],
        )
        request = InternalRequest(
            model="gemini-2.0-flash",
            conversation=ConversationContext(messages=[message_with_url]),
        )

        patched_contents = [
            {
                "role": "user",
                "parts": [
                    {"text": "Describe this"},
                    {"inline_data": {"mime_type": "image/jpeg", "data": "XYZ789"}},
                ],
            }
        ]

        async def async_iter_lines():
            for line in [
                b'data: {"candidates":[{"content":{"parts":[{"text":"A"}]}}]}',
                b"data: [DONE]",
            ]:
                yield line

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = async_iter_lines()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(
                adapter, "_download_images_in_gemini_contents", return_value=patched_contents
            ),
            patch.object(adapter, "_get_client", return_value=mock_client),
        ):
            stream_gen = await adapter.stream_chat_completion(request)
            async for _chunk in stream_gen:
                pass

        post_call = mock_client.post.call_args
        body = post_call.kwargs["json"]

        contents = body.get("contents", [])
        assert len(contents) > 0, "Request body should have contents"

        user_content = contents[0].get("parts", [])
        image_parts = [p for p in user_content if "inline_data" in p]
        assert len(image_parts) > 0, "Request should contain inline_data from downloaded image"
        assert image_parts[0]["inline_data"]["data"] == "XYZ789"


class TestGeminiEmbeddingsResponseFormat:
    """Tests for Gemini embeddings response format parsing."""

    def test_single_embedding_response(self):
        """Parse single embedding response: {"embedding": {"values": [...]}}."""
        serializer = GeminiProviderSerializer()
        response = {"embedding": {"values": [0.1, 0.2, 0.3]}}
        result = serializer.parse_provider_embedding_response(response, model="test-model")

        assert isinstance(result, InternalEmbeddingResponse)
        assert result.model == "test-model"
        assert len(result.data) == 1
        assert result.data[0].embedding == [0.1, 0.2, 0.3]
        assert result.data[0].index == 0

    def test_batch_embedding_response(self):
        """Parse batch embedding response: {"embeddings": [{"values": [...]}, ...]}."""
        serializer = GeminiProviderSerializer()
        response = {
            "embeddings": [
                {"values": [0.1, 0.2]},
                {"values": [0.3, 0.4]},
                {"values": [0.5, 0.6]},
            ]
        }
        result = serializer.parse_provider_embedding_response(response, model="test-model")

        assert len(result.data) == 3
        assert result.data[0].embedding == [0.1, 0.2]
        assert result.data[0].index == 0
        assert result.data[1].embedding == [0.3, 0.4]
        assert result.data[1].index == 1
        assert result.data[2].embedding == [0.5, 0.6]
        assert result.data[2].index == 2

    def test_empty_response(self):
        """Handle response without embedding or embeddings key."""
        serializer = GeminiProviderSerializer()
        result = serializer.parse_provider_embedding_response({}, model="test-model")

        assert isinstance(result, InternalEmbeddingResponse)
        assert result.data == []

    def test_embedding_not_dict(self):
        """Handle embedding field that is not a dict."""
        serializer = GeminiProviderSerializer()
        result = serializer.parse_provider_embedding_response(
            {"embedding": "invalid"}, model="test-model"
        )

        assert len(result.data) == 1
        assert result.data[0].embedding == []

    def test_single_embedding_usage_parsed(self):
        """usageMetadata is parsed into response usage for billing."""
        serializer = GeminiProviderSerializer()
        response = {
            "embedding": {"values": [0.1, 0.2, 0.3]},
            "usageMetadata": {"promptTokenCount": 7, "totalTokenCount": 7},
        }
        result = serializer.parse_provider_embedding_response(response, model="test-model")

        assert result.usage is not None
        assert result.usage.input_tokens == 7
        assert result.usage.output_tokens == 0
        assert result.usage.total_tokens == 7

    def test_batch_embedding_usage_parsed(self):
        """batchEmbedContents usageMetadata is parsed into response usage."""
        serializer = GeminiProviderSerializer()
        response = {
            "embeddings": [{"values": [0.1]}, {"values": [0.2]}],
            "usageMetadata": {"promptTokenCount": 12, "totalTokenCount": 12},
        }
        result = serializer.parse_provider_embedding_response(response, model="test-model")

        assert result.usage is not None
        assert result.usage.input_tokens == 12
        assert result.usage.total_tokens == 12

    def test_no_usage_metadata_keeps_usage_none(self):
        """Responses without usageMetadata keep usage None (no crash)."""
        serializer = GeminiProviderSerializer()
        result = serializer.parse_provider_embedding_response(
            {"embedding": {"values": [0.1]}}, model="test-model"
        )

        assert result.usage is None


class TestGeminiEmbeddingsRequest:
    """Tests for Gemini embeddings request building and execution."""

    @pytest.mark.asyncio
    async def test_single_string_uses_embedcontent(self):
        """Single string input should use :embedContent endpoint."""
        adapter = GeminiAdapter(api_key="test-key")

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": {"values": [0.1, 0.2]}}
        mock_client.post = AsyncMock(return_value=mock_response)

        request = InternalEmbeddingRequest(model="test-model", input="hello world")

        with patch.object(adapter, "_get_client", return_value=mock_client):
            await adapter.embeddings(request)

        url = mock_client.post.call_args.args[0]
        body = mock_client.post.call_args.kwargs["json"]

        assert ":embedContent" in url
        assert "batchEmbedContents" not in url
        assert body == {"content": {"parts": [{"text": "hello world"}]}}

    @pytest.mark.asyncio
    async def test_multiple_inputs_uses_batchembedcontents(self):
        """List input should use :batchEmbedContents endpoint."""
        adapter = GeminiAdapter(api_key="test-key")

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embeddings": [{"values": [0.1]}, {"values": [0.2]}, {"values": [0.3]}]
        }
        mock_client.post = AsyncMock(return_value=mock_response)

        request = InternalEmbeddingRequest(model="test-model", input=["a", "b", "c"])

        with patch.object(adapter, "_get_client", return_value=mock_client):
            result = await adapter.embeddings(request)

        url = mock_client.post.call_args.args[0]
        body = mock_client.post.call_args.kwargs["json"]

        assert ":batchEmbedContents" in url
        assert "embedContent" not in url
        assert len(body["requests"]) == 3
        assert body["requests"][0]["content"]["parts"][0]["text"] == "a"
        assert body["requests"][1]["content"]["parts"][0]["text"] == "b"
        assert body["requests"][2]["content"]["parts"][0]["text"] == "c"
        assert len(result.data) == 3

    @pytest.mark.asyncio
    async def test_single_element_list_uses_embedcontent(self):
        """Single-element list should use :embedContent (treated as single input)."""
        adapter = GeminiAdapter(api_key="test-key")

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": {"values": [0.1, 0.2]}}
        mock_client.post = AsyncMock(return_value=mock_response)

        request = InternalEmbeddingRequest(model="test-model", input=["only one"])

        with patch.object(adapter, "_get_client", return_value=mock_client):
            result = await adapter.embeddings(request)

        url = mock_client.post.call_args.args[0]

        assert ":embedContent" in url
        assert "batchEmbedContents" not in url
        assert len(result.data) == 1
        assert result.data[0].index == 0

    @pytest.mark.asyncio
    async def test_dimensions_in_single_request(self):
        """Dimensions should be included in single embedContent request."""
        adapter = GeminiAdapter(api_key="test-key")

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": {"values": [0.1]}}
        mock_client.post = AsyncMock(return_value=mock_response)

        request = InternalEmbeddingRequest(model="test-model", input="hello", dimensions=512)

        with patch.object(adapter, "_get_client", return_value=mock_client):
            await adapter.embeddings(request)

        body = mock_client.post.call_args.kwargs["json"]
        assert body["embedContentConfig"]["outputDimensionality"] == 512

    @pytest.mark.asyncio
    async def test_dimensions_in_batch_request(self):
        """Dimensions should be included in each batch request item."""
        adapter = GeminiAdapter(api_key="test-key")

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embeddings": [{"values": [0.1]}, {"values": [0.2]}]}
        mock_client.post = AsyncMock(return_value=mock_response)

        request = InternalEmbeddingRequest(model="test-model", input=["a", "b"], dimensions=256)

        with patch.object(adapter, "_get_client", return_value=mock_client):
            await adapter.embeddings(request)

        body = mock_client.post.call_args.kwargs["json"]
        assert body["requests"][0]["embedContentConfig"]["outputDimensionality"] == 256
        assert body["requests"][1]["embedContentConfig"]["outputDimensionality"] == 256

    @pytest.mark.asyncio
    async def test_batch_fallback_when_batchembedcontents_fails(self):
        """Fall back to individual embedContent calls when batchEmbedContents fails."""

        adapter = GeminiAdapter(api_key="test-key")

        mock_client = AsyncMock()
        mock_success = MagicMock()
        mock_success.status_code = 200
        mock_success.json.return_value = {
            "embedding": {"values": [0.1]},
            "usageMetadata": {"promptTokenCount": 3, "totalTokenCount": 3},
        }

        mock_fail = MagicMock()
        mock_fail.status_code = 400
        mock_fail.text = AsyncMock(return_value="Bad Request")

        # First call (batch) fails, subsequent calls (single) succeed
        mock_client.post = AsyncMock(side_effect=[mock_fail, mock_success, mock_success])

        request = InternalEmbeddingRequest(model="test-model", input=["a", "b"])

        with patch.object(adapter, "_get_client", return_value=mock_client):
            result = await adapter.embeddings(request)

        assert mock_client.post.call_count == 3
        assert len(result.data) == 2
        assert result.data[0].embedding == [0.1]
        assert result.data[0].index == 0
        assert result.data[1].embedding == [0.1]
        assert result.data[1].index == 1
        # Usage is aggregated across the individual embedContent calls
        assert result.usage is not None
        assert result.usage.input_tokens == 6
        assert result.usage.total_tokens == 6

    @pytest.mark.asyncio
    async def test_batch_not_retried_on_non_fallback_status(self):
        """Non-fallback status codes should still raise, not fall back."""
        adapter = GeminiAdapter(api_key="test-key")

        mock_client = AsyncMock()
        mock_fail = MagicMock()
        mock_fail.status_code = 500
        mock_fail.text = AsyncMock(return_value="Internal Server Error")
        mock_client.post = AsyncMock(return_value=mock_fail)

        request = InternalEmbeddingRequest(model="test-model", input=["a", "b"])

        with (
            pytest.raises(ProviderError),
            patch.object(adapter, "_get_client", return_value=mock_client),
        ):
            await adapter.embeddings(request)

    @pytest.mark.asyncio
    async def test_models_prefix_removed_from_url(self):
        """Models with models/ prefix should have it stripped in the URL."""
        adapter = GeminiAdapter(api_key="test-key")

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": {"values": [0.1]}}
        mock_client.post = AsyncMock(return_value=mock_response)

        request = InternalEmbeddingRequest(model="models/my-embed-model", input="test")

        with patch.object(adapter, "_get_client", return_value=mock_client):
            await adapter.embeddings(request)

        url = mock_client.post.call_args.args[0]
        assert "/models/my-embed-model:embedContent" in url
        assert "models/models/" not in url


class TestGeminiEmbeddingsProviderNameCasing:
    """Regression tests for provider name casing in embeddings."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "provider_name",
        [
            "Gemini",
            "GEMINI",
            "GeMiNi",
            "gemini",
        ],
    )
    async def test_embeddings_works_with_various_provider_name_casing(self, provider_name):
        """Embedding requests should work regardless of provider_name casing."""
        adapter = GeminiAdapter(provider_name=provider_name, api_key="test-key")

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": {"values": [0.1, 0.2]}}
        mock_client.post = AsyncMock(return_value=mock_response)

        request = InternalEmbeddingRequest(model="test-model", input="hello world")

        with patch.object(adapter, "_get_client", return_value=mock_client):
            result = await adapter.embeddings(request)

        # Verify the URL uses the normalized (lowercase) provider name
        url = mock_client.post.call_args.args[0]
        assert ":embedContent" in url, f"URL should contain :embedContent, got: {url}"
        assert "models/" in url, f"URL should contain models/, got: {url}"

        assert isinstance(result, InternalEmbeddingResponse)
        assert len(result.data) == 1
        assert result.data[0].embedding == [0.1, 0.2]


class TestParseDataUrl:
    """Tests for GeminiAdapter._parse_data_url."""

    def test_valid_data_url(self):
        adapter = GeminiAdapter(api_key="test-key")
        mime, data = adapter._parse_data_url("data:image/png;base64,ABC123")
        assert mime == "image/png"
        assert data == "ABC123"

    def test_invalid_data_url_falls_back_to_image_png(self):
        adapter = GeminiAdapter(api_key="test-key")
        mime, data = adapter._parse_data_url("not-a-data-url")
        assert mime == "image/png"
        assert data == "not-a-data-url"

    def test_index_error_in_data_url_falls_back(self):
        """Regression: ensure both ValueError and IndexError are caught."""
        adapter = GeminiAdapter(api_key="test-key")
        mime, data = adapter._parse_data_url("data:")
        assert mime == "image/png"
        assert data == ""

    def test_data_url_without_comma_falls_back(self):
        adapter = GeminiAdapter(api_key="test-key")
        mime, data = adapter._parse_data_url("data:image/png;base64")
        assert mime == "image/png"
        assert data == ""


class TestDownloadImagesMimeType:
    """Tests that non-image HTTP URLs keep their correct MIME type."""

    @pytest.mark.asyncio
    async def test_preserves_audio_content_type(self):
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = b"audio data"
        mock_response.headers = {"content-type": "audio/mpeg"}
        mock_client.get.return_value = mock_response

        result = await download_image_as_base64(mock_client, "https://example.com/audio.mp3")
        assert result is not None
        data_url, mime = result
        assert mime == "audio/mpeg"
        assert data_url.startswith("data:audio/mpeg;base64,")

    @pytest.mark.asyncio
    async def test_infers_mime_from_extension_when_no_header(self):
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = b"pdf data"
        mock_response.headers = {}
        mock_client.get.return_value = mock_response

        result = await download_image_as_base64(mock_client, "https://example.com/doc.pdf")
        assert result is not None
        data_url, mime = result
        assert mime == "application/pdf"
        assert data_url.startswith("data:application/pdf;base64,")

    @pytest.mark.asyncio
    async def test_strips_charset_from_content_type(self):
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = b"pdf data"
        mock_response.headers = {"content-type": "application/pdf; charset=utf-8"}
        mock_client.get.return_value = mock_response

        result = await download_image_as_base64(mock_client, "https://example.com/doc.pdf")
        assert result is not None
        data_url, mime = result
        assert mime == "application/pdf"


class TestThoughtSignatureCache:
    """Tests for thought_signature caching and re-attachment."""

    def setup_method(self):
        """Clear class-level cache before each test to ensure isolation."""
        GeminiAdapter._thought_signature_cache.clear()

    def test_cache_thought_signatures_from_response(self):
        adapter = GeminiAdapter(api_key="test-key")
        response = InternalResponse(
            id="resp-1",
            model="gemini-2.0-flash",
            output=[
                ToolUseBlock(
                    id="call_1",
                    name="get_weather",
                    input={"city": "NYC"},
                    extra={"thought_signature": "sig_abc"},
                ),
                ToolUseBlock(
                    id="call_2",
                    name="get_time",
                    input={"tz": "EST"},
                    extra={},
                ),
            ],
        )
        adapter._cache_thought_signatures(response.output)
        assert adapter._thought_signature_cache == {"call_1": "sig_abc"}
        assert "call_2" not in adapter._thought_signature_cache

    def test_cache_thought_signatures_from_response_empty(self):
        adapter = GeminiAdapter(api_key="test-key")
        response = InternalResponse(id="resp-1", model="m", output=[])
        adapter._cache_thought_signatures(response.output)
        assert adapter._thought_signature_cache == {}

    def test_cache_thought_signatures_from_transformer(self):
        adapter = GeminiAdapter(api_key="test-key")
        transformer = MagicMock()
        transformer._accumulated_output = [
            ToolUseBlock(
                id="call_x",
                name="search",
                input={"q": "test"},
                extra={"thought_signature": "sig_xyz"},
            ),
        ]
        adapter._cache_thought_signatures(transformer._accumulated_output)
        assert adapter._thought_signature_cache == {"call_x": "sig_xyz"}

    def test_cache_thought_signatures_from_transformer_no_attr(self):
        adapter = GeminiAdapter(api_key="test-key")
        transformer = MagicMock(spec=[])
        adapter._cache_thought_signatures(getattr(transformer, "_accumulated_output", []))
        assert adapter._thought_signature_cache == {}

    def test_cache_evicts_at_max(self):
        adapter = GeminiAdapter(api_key="test-key")
        adapter._MAX_THOUGHT_SIGNATURE_CACHE = 3
        for i in range(4):
            response = InternalResponse(
                id=f"resp-{i}",
                model="m",
                output=[
                    ToolUseBlock(
                        id=f"call_{i}", name="n", input={}, extra={"thought_signature": f"sig_{i}"}
                    )
                ],
            )
            adapter._cache_thought_signatures(response.output)
        # LRU eviction: only the oldest entry (call_0) is evicted
        assert len(adapter._thought_signature_cache) == 3
        assert "call_0" not in adapter._thought_signature_cache
        assert "call_3" in adapter._thought_signature_cache

    def test_enrich_conversation_with_thought_signatures(self):
        adapter = GeminiAdapter(api_key="test-key")
        adapter._thought_signature_cache = {"call_1": "sig_abc"}

        block = ToolUseBlock(id="call_1", name="get_weather", input={"city": "NYC"}, extra={})
        conversation = ConversationContext(messages=[Message(role="assistant", content=[block])])
        adapter._enrich_conversation_with_thought_signatures(conversation)
        assert block.extra["thought_signature"] == "sig_abc"

    def test_enrich_skips_when_already_present(self):
        adapter = GeminiAdapter(api_key="test-key")
        adapter._thought_signature_cache = {"call_1": "sig_abc"}

        block = ToolUseBlock(
            id="call_1", name="get_weather", input={}, extra={"thought_signature": "sig_existing"}
        )
        conversation = ConversationContext(messages=[Message(role="assistant", content=[block])])
        adapter._enrich_conversation_with_thought_signatures(conversation)
        assert block.extra["thought_signature"] == "sig_existing"

    def test_enrich_skips_when_cache_empty(self):
        adapter = GeminiAdapter(api_key="test-key")
        block = ToolUseBlock(id="call_1", name="n", input={}, extra={})
        conversation = ConversationContext(messages=[Message(role="assistant", content=[block])])
        adapter._enrich_conversation_with_thought_signatures(conversation)
        assert "thought_signature" not in block.extra

    def test_enrich_skips_non_tool_use_blocks(self):
        adapter = GeminiAdapter(api_key="test-key")
        adapter._thought_signature_cache = {"call_1": "sig_abc"}

        text_block = TextBlock(text="hello")
        conversation = ConversationContext(
            messages=[Message(role="assistant", content=[text_block])]
        )
        adapter._enrich_conversation_with_thought_signatures(conversation)
        assert "thought_signature" not in text_block.__dict__


class TestGeminiImageSizeMapping:
    """Regression tests for Gemini image generation size configuration."""

    def test_image_size_maps_to_image_config(self):
        adapter = GeminiAdapter(api_key="test-key")
        body = adapter._build_gemini_image_request(
            prompt="a cat",
            model="gemini-3.1-flash-image",
            size=ImageSize(width=1024, height=768),
        )
        assert "imageConfig" in body["generationConfig"]
        assert body["generationConfig"]["imageConfig"]["aspectRatio"] == "4:3"
        assert body["generationConfig"]["imageConfig"]["imageSize"] == "1K"

    def test_image_size_uses_supported_k_shorthand(self):
        adapter = GeminiAdapter(api_key="test-key")
        body = adapter._build_gemini_image_request(
            prompt="a cat",
            model="gemini-3.1-flash-image",
            size=ImageSize(width=512, height=512),
        )
        # Gemini ImageConfig.imageSize supports 512/1K/2K/4K (not "0.5K").
        assert body["generationConfig"]["imageConfig"]["imageSize"] == "512"

    def test_image_size_maps_unsupported_ratio_to_nearest(self):
        """1792x1024 (7:4) is not a Gemini aspect ratio; map to 16:9."""
        adapter = GeminiAdapter(api_key="test-key")
        body = adapter._build_gemini_image_request(
            prompt="a cat",
            model="gemini-3.1-flash-image",
            size=ImageSize(width=1792, height=1024),
        )
        assert body["generationConfig"]["imageConfig"]["aspectRatio"] == "16:9"
        assert body["generationConfig"]["imageConfig"]["imageSize"] == "2K"

    def test_image_size_maps_portrait_unsupported_ratio(self):
        """1024x1792 (4:7) maps to 9:16."""
        adapter = GeminiAdapter(api_key="test-key")
        body = adapter._build_gemini_image_request(
            prompt="a cat",
            model="gemini-3.1-flash-image",
            size=ImageSize(width=1024, height=1792),
        )
        assert body["generationConfig"]["imageConfig"]["aspectRatio"] == "9:16"

    def test_image_size_4k_for_large(self):
        adapter = GeminiAdapter(api_key="test-key")
        body = adapter._build_gemini_image_request(
            prompt="a cat",
            model="gemini-3.1-flash-image",
            size=ImageSize(width=4096, height=4096),
        )
        assert body["generationConfig"]["imageConfig"]["imageSize"] == "4K"

    def test_image_size_none_omits_image_config(self):
        adapter = GeminiAdapter(api_key="test-key")
        body = adapter._build_gemini_image_request(
            prompt="a cat",
            model="gemini-3.1-flash-image",
            size=None,
        )
        assert "imageConfig" not in body["generationConfig"]
        assert body["generationConfig"]["responseModalities"] == ["IMAGE"]

    def test_image_edit_mask_with_file_id_is_preserved(self):
        """A mask referenced by file_id must not be silently dropped."""
        adapter = GeminiAdapter(api_key="test-key")
        request = InternalImageEditRequest(
            model="gemini-2.5-flash-image",
            prompt="edit",
            images=[ImageEditSource(image_url="https://example.com/base.png")],
            mask=ImageEditSource(file_id="files/abc123"),
        )
        body, files = adapter._build_image_edit_raw(request)
        assert files == {}
        parts = body["contents"][0]["parts"]
        mask_parts = [p for p in parts if p.get("file_data", {}).get("file_uri") == "files/abc123"]
        assert len(mask_parts) == 1

    def test_image_edit_mask_with_http_url_is_preserved(self):
        """A mask referenced by HTTP URL must not be silently dropped."""
        adapter = GeminiAdapter(api_key="test-key")
        request = InternalImageEditRequest(
            model="gemini-2.5-flash-image",
            prompt="edit",
            images=[ImageEditSource(image_url="https://example.com/base.png")],
            mask=ImageEditSource(image_url="https://example.com/mask.png"),
        )
        body, _files = adapter._build_image_edit_raw(request)
        parts = body["contents"][0]["parts"]
        mask_parts = [
            p
            for p in parts
            if p.get("file_data", {}).get("file_uri") == "https://example.com/mask.png"
        ]
        assert len(mask_parts) == 1

    def test_stream_image_edit_raises_clean_provider_error(self):
        """Streaming edits fail with a 400 invalid_request_error, not a 500."""
        adapter = GeminiAdapter(api_key="test-key")
        request = InternalImageEditRequest(
            model="gemini-2.5-flash-image",
            prompt="edit",
            images=[ImageEditSource(image_url="https://example.com/base.png")],
        )
        with pytest.raises(ProviderError) as exc_info:
            asyncio.run(adapter.stream_image_edit(request))
        assert exc_info.value.status_code == 400
        assert exc_info.value.error_type == "invalid_request_error"


class TestGeminiErrorResponseParsing:
    """Provider error body parsing, including actionable geo-block hints."""

    _LOCATION_BODY = {
        "error": {
            "message": "User location is not supported for the API use.",
            "status": "FAILED_PRECONDITION",
        }
    }

    def test_location_block_error_gets_actionable_hint(self):
        """Google's geo-block error surfaces with guidance, not as a proxy bug."""
        adapter = GeminiAdapter(api_key="test-key")
        error = adapter._parse_error_response(400, self._LOCATION_BODY)

        assert error.status_code == 400
        assert error.provider_name == "gemini"
        assert error.error_type == "api_error"
        assert error.message.startswith("User location is not supported for the API use.")
        assert "egress IP" in error.message
        assert "supported region" in error.message
        # original_error must survive so the API layer passes the upstream
        # body through verbatim (status FAILED_PRECONDITION, request_id...).
        assert error.original_error == self._LOCATION_BODY

    def test_unrelated_error_message_unchanged(self):
        """Other FAILED_PRECONDITION errors are not rewritten."""
        adapter = GeminiAdapter(api_key="test-key")
        body = {
            "error": {
                "message": "Model does not support this capability",
                "status": "FAILED_PRECONDITION",
            }
        }
        error = adapter._parse_error_response(400, body)

        assert error.message == "Model does not support this capability"
        assert "egress" not in error.message


class TestGeminiStreamingErrorPassthrough:
    """ProviderError raised mid-stream keeps its metadata (status, type, body).

    Regression test: the generators used to funnel every exception through
    _handle_http_error, which re-wraps ProviderError as a generic
    "Gemini request failed: ..." api_error with status_code=None — turning
    upstream 4xx errors into 500s and dropping the original error body.
    """

    @staticmethod
    def _error_client() -> MagicMock:
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = TestGeminiErrorResponseParsing._LOCATION_BODY
        mock_response.aread = AsyncMock()
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        return mock_client

    async def test_stream_chat_preserves_provider_error_metadata(self):
        adapter = GeminiAdapter(api_key="test-key")
        request = InternalRequest(
            model="gemini-2.0-flash",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
        )

        with patch.object(adapter, "_get_client", return_value=self._error_client()):
            stream_gen = await adapter.stream_chat_completion(request)
            with pytest.raises(ProviderError) as exc_info:
                async for _chunk in stream_gen:
                    pass

        error = exc_info.value
        assert error.status_code == 400
        assert error.error_type == "api_error"
        assert "Gemini request failed" not in error.message
        assert error.message.startswith("User location is not supported for the API use.")
        assert error.original_error == TestGeminiErrorResponseParsing._LOCATION_BODY

    async def test_stream_speech_preserves_provider_error_metadata(self):
        from llm_proxy.models.audio import InternalSpeechRequest

        adapter = GeminiAdapter(api_key="test-key")
        request = InternalSpeechRequest(model="gemini-2.5-flash", input="hello", voice="Puck")

        with patch.object(adapter, "_get_client", return_value=self._error_client()):
            stream_gen = await adapter.stream_speech(request)
            with pytest.raises(ProviderError) as exc_info:
                async for _chunk in stream_gen:
                    pass

        error = exc_info.value
        assert error.status_code == 400
        assert "Gemini request failed" not in error.message

    async def test_stream_image_generation_preserves_provider_error_metadata(self):
        from llm_proxy.models.image import InternalImageRequest

        adapter = GeminiAdapter(api_key="test-key")
        request = InternalImageRequest(model="gemini-2.5-flash-image", prompt="a cat")

        with patch.object(adapter, "_get_client", return_value=self._error_client()):
            stream_gen = await adapter.stream_image_generation(request)
            with pytest.raises(ProviderError) as exc_info:
                async for _chunk in stream_gen:
                    pass

        error = exc_info.value
        assert error.status_code == 400
        assert "Gemini request failed" not in error.message


class TestGeminiNoSpaceSseTolerance:
    """Gemini-compatible relays may omit the space after the SSE field colon.

    Regression: ``data:{...}`` lines used to fall through to orjson with the
    ``data:`` prefix intact, so every chunk was silently dropped (empty
    stream, no usage). The parser must accept both SSE spellings.
    """

    @staticmethod
    def _stream_response(lines: list[bytes]) -> MagicMock:
        mock_response = MagicMock()
        mock_response.status_code = 200

        async def _iter_lines():
            for line in lines:
                yield line

        mock_response.iter_lines = _iter_lines
        return mock_response

    @staticmethod
    def _chat_request() -> InternalRequest:
        return InternalRequest(
            model="gemini-2.0-flash",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
        )

    async def test_stream_chat_parses_no_space_sse_frames(self):
        from contextlib import asynccontextmanager

        adapter = GeminiAdapter(api_key="test-key")
        response_mock = self._stream_response(
            [
                b'data:{"candidates":[{"content":{"parts":[{"text":"hi"}]}}],'
                b'"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":1}}',
                b"data:[DONE]",
            ]
        )

        @asynccontextmanager
        async def fake_streaming_post(*args, **kwargs):
            yield response_mock

        with (
            patch.object(adapter, "_get_client", return_value=MagicMock()),
            patch.object(adapter._transport, "streaming_post", fake_streaming_post),
        ):
            stream_gen = await adapter.stream_chat_completion(self._chat_request())
            chunks = [chunk async for chunk in stream_gen]

        content_chunks = [chunk for chunk in chunks if chunk != "[DONE]"]
        assert content_chunks, "no-space SSE frames must still produce chunks"
        assert any(chunk.get("choices") for chunk in content_chunks)
