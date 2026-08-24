"""Tests for ChutesAdapter native implementation."""

import base64
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_proxy.core.exceptions import ProviderError
from llm_proxy.models import ConversationContext, InternalRequest, Message, TextBlock
from llm_proxy.models.image import ImageSize, InternalImageRequest, InternalImageResponse
from llm_proxy.providers.chutes import ChutesAdapter
from llm_proxy.providers.chutes.serializer import (
    ChutesProviderSerializer,
    _decode_base64_embedding,
    _parse_embedding_item,
)
from llm_proxy.serialization.context import BuildContext


class TestChutesAdapter:
    """Test suite for ChutesAdapter."""

    def test_default_initialization(self):
        """Test default provider initialization."""
        provider = ChutesAdapter(api_key="test-key")
        assert provider.provider_name == "chutes"
        assert provider._base_url == "https://llm.chutes.ai/v1"

    def test_custom_base_url(self):
        """Test custom base URL configuration."""
        provider = ChutesAdapter(api_key="test-key", base_url="https://custom.chutes.ai/v1")
        assert provider._base_url == "https://custom.chutes.ai/v1"

    def test_normalize_model_name_with_prefix(self):
        """Test model name normalization strips prefixes."""
        provider = ChutesAdapter(api_key="test-key")
        result = provider._normalize_model_name("chutes/Qwen/Qwen3-Embedding-8B")
        assert result == "Qwen/Qwen3-Embedding-8B"
        result = provider._normalize_model_name("openai/Qwen/Qwen3-Embedding-8B")
        assert result == "Qwen/Qwen3-Embedding-8B"

    def test_normalize_model_name_without_prefix(self):
        """Test model name normalization passes through plain names."""
        provider = ChutesAdapter(api_key="test-key")
        result = provider._normalize_model_name("Qwen/Qwen3-Embedding-8B")
        assert result == "Qwen/Qwen3-Embedding-8B"
        assert provider._normalize_model_name("gpt-4") == "gpt-4"

    def test_resolve_embeddings_base_url_known_model(self):
        """Test embedding URL resolution for known models."""
        provider = ChutesAdapter(api_key="test-key")
        url = provider._resolve_embeddings_base_url("Qwen/Qwen3-Embedding-8B")
        assert url == "https://chutes-qwen-qwen3-embedding-8b.chutes.ai/v1"

        url = provider._resolve_embeddings_base_url("Qwen/Qwen3-Embedding-0.6B")
        assert url == "https://chutes-qwen-qwen3-embedding-0-6b.chutes.ai/v1"

    def test_resolve_embeddings_base_url_unknown_model_raises(self):
        """Test embedding URL resolution raises for unknown models."""
        provider = ChutesAdapter(api_key="test-key")
        with pytest.raises(ProviderError) as exc_info:
            provider._resolve_embeddings_base_url("unknown/model")
        assert "only support models" in str(exc_info.value).lower()

    def test_build_headers_includes_auth(self):
        """Test that headers include authorization."""
        provider = ChutesAdapter(api_key="test-key")
        headers = provider._build_headers()
        assert headers["Authorization"] == "Bearer test-key"


class TestChutesProviderSerializerEmbedding:
    """Test suite for ChutesProviderSerializer embedding methods."""

    def test_decode_base64_embedding_valid(self):
        """Test base64 decoding of embedding vector."""
        import base64
        import struct

        raw_bytes = struct.pack("<fff", 1.0, 2.0, 3.0)
        b64_value = base64.b64encode(raw_bytes).decode()

        result = _decode_base64_embedding(b64_value)
        assert len(result) == 3
        assert abs(result[0] - 1.0) < 0.001
        assert abs(result[1] - 2.0) < 0.001
        assert abs(result[2] - 3.0) < 0.001

    def test_decode_base64_embedding_invalid(self):
        """Test base64 decoding returns empty list for invalid input."""
        assert _decode_base64_embedding("not-valid-base64!!!") == []

    def test_decode_base64_embedding_empty(self):
        """Test base64 decoding returns empty list for empty input."""
        assert _decode_base64_embedding("") == []

    def test_parse_embedding_item_dict_format(self):
        """Test parsing embedding item in dict format."""
        item = {"embedding": [0.1, 0.2, 0.3], "index": 5}
        embedding, index = _parse_embedding_item(item)
        assert embedding == [0.1, 0.2, 0.3]
        assert index == 5

    def test_parse_embedding_item_base64_format(self):
        """Test parsing embedding item with base64-encoded embedding."""
        import base64
        import struct

        raw_bytes = struct.pack("<ff", 0.5, 1.5)
        b64_value = base64.b64encode(raw_bytes).decode()
        item = {"embedding": b64_value, "index": 0}

        embedding, index = _parse_embedding_item(item)
        assert len(embedding) == 2
        assert abs(embedding[0] - 0.5) < 0.001
        assert abs(embedding[1] - 1.5) < 0.001

    def test_convert_embedding_response(self):
        """Test converting full embedding response."""
        serializer = ChutesProviderSerializer()
        response = {
            "object": "list",
            "data": [
                {"embedding": [0.1, 0.2], "index": 0},
                {"embedding": [0.3, 0.4], "index": 1},
            ],
            "model": "test-model",
            "usage": {"prompt_tokens": 10, "total_tokens": 20},
        }

        result = serializer.parse_provider_embedding_response(response, model="test-model")
        assert len(result.data) == 2
        assert result.data[0].embedding == [0.1, 0.2]
        assert result.data[1].embedding == [0.3, 0.4]
        assert result.usage is not None
        assert result.usage.input_tokens == 10
        assert result.usage.total_tokens == 20


class TestChutesImageModelConfig:
    """Test suite for Chutes image model configuration."""

    def test_image_model_config_has_z_image_turbo(self):
        """Test that z-image-turbo has model_in_url pattern."""
        from llm_proxy.providers.chutes.adapter import _IMAGE_MODEL_CONFIGS

        assert "z-image-turbo" in _IMAGE_MODEL_CONFIGS
        assert _IMAGE_MODEL_CONFIGS["z-image-turbo"]["url_pattern"] == "model_in_url"

    def test_image_model_config_has_qwen_image(self):
        """Test that Qwen-Image-2512 has model_in_body pattern."""
        from llm_proxy.providers.chutes.adapter import _IMAGE_MODEL_CONFIGS

        assert "Qwen-Image-2512" in _IMAGE_MODEL_CONFIGS
        config = _IMAGE_MODEL_CONFIGS["Qwen-Image-2512"]
        assert config["url_pattern"] == "model_in_body"
        assert config.get("default_params", {}).get("guidance_scale") == 7.5
        assert config.get("default_params", {}).get("num_inference_steps") == 50


class TestChutesGetImageModelConfig:
    """Test suite for _get_image_model_config method."""

    def test_get_image_model_config_known_model(self):
        """Test getting config for known model."""
        provider = ChutesAdapter(api_key="test-key")
        config = provider._get_image_model_config("z-image-turbo")
        assert config["url_pattern"] == "model_in_url"

    def test_get_image_model_config_unknown_model(self):
        """Test getting config for unknown model returns default."""
        provider = ChutesAdapter(api_key="test-key")
        config = provider._get_image_model_config("unknown-model")
        # Should return default config with model_in_body
        assert config["url_pattern"] == "model_in_body"

    def test_get_image_model_config_normalizes_model_name(self):
        """Test that model name is normalized before lookup."""
        provider = ChutesAdapter(api_key="test-key")
        config = provider._get_image_model_config("chutes/z-image-turbo")
        assert config["url_pattern"] == "model_in_url"

    def test_build_chat_raw_normalizes_model_name(self):
        """Regression: chat path must strip the chutes/ prefix before sending upstream."""
        provider = ChutesAdapter(api_key="test-key")
        request = InternalRequest(
            model="chutes/Qwen/Qwen3-Embedding-8B",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
        )
        body = provider._build_chat_raw(request, BuildContext())
        assert body["model"] == "Qwen/Qwen3-Embedding-8B"


class TestChutesResolveImageUrl:
    """Test suite for _resolve_image_url method."""

    def test_resolve_image_url_model_in_url(self):
        """Test URL for model_in_url pattern."""
        provider = ChutesAdapter(api_key="test-key")
        config = {"url_pattern": "model_in_url"}
        url = provider._resolve_image_url("z-image-turbo", config)
        assert url == "https://chutes-z-image-turbo.chutes.ai/generate"

    def test_resolve_image_url_model_in_body(self):
        """Test URL for model_in_body pattern."""
        provider = ChutesAdapter(api_key="test-key")
        config = {"url_pattern": "model_in_body"}
        url = provider._resolve_image_url("Qwen-Image-2512", config)
        assert url == "https://image.chutes.ai/generate"


class TestChutesBuildImageRequestBody:
    """Test suite for _build_chutes_image_request_body method."""

    def test_build_body_basic_prompt(self):
        """Test building body with just prompt."""
        provider = ChutesAdapter(api_key="test-key")
        request = InternalImageRequest(model="test-model", prompt="a white cat")
        config = {"url_pattern": "model_in_body"}

        body = provider._build_chutes_image_request_body(request, config)
        assert body["prompt"] == "a white cat"
        assert body["model"] == "test-model"

    def test_build_body_size_conversion(self):
        """Test size to width/height conversion."""
        provider = ChutesAdapter(api_key="test-key")
        request = InternalImageRequest(
            model="test-model", prompt="test", size=ImageSize(width=1024, height=768)
        )
        config = {"url_pattern": "model_in_body"}

        body = provider._build_chutes_image_request_body(request, config)
        assert body["width"] == 1024
        assert body["height"] == 768

    def test_build_body_default_size(self):
        """Test default size when not specified."""
        provider = ChutesAdapter(api_key="test-key")
        request = InternalImageRequest(model="test-model", prompt="test")
        config = {"url_pattern": "model_in_body"}

        body = provider._build_chutes_image_request_body(request, config)
        assert body["width"] == 1024
        assert body["height"] == 1024

    def test_build_body_with_extra(self):
        """Test extra parameters are merged by the dispatch.

        _build_chutes_image_request_body no longer merges extra directly;
        the dispatch's _finalize_body handles that.
        """
        provider = ChutesAdapter(api_key="test-key", unknown_fields_policy="passthrough")
        request = InternalImageRequest(
            model="test-model",
            prompt="test",
            extra={"negative_prompt": "blur", "guidance_scale": 8.0},
        )

        outbound = provider._build_outbound_body(request, request_type="image_generation")
        assert outbound.json_body is not None
        assert outbound.json_body["negative_prompt"] == "blur"
        assert outbound.json_body["guidance_scale"] == 8.0

    def test_build_body_with_default_params(self):
        """Test default_params from config are included."""
        provider = ChutesAdapter(api_key="test-key")
        request = InternalImageRequest(model="test-model", prompt="test")
        config = {
            "url_pattern": "model_in_body",
            "default_params": {"guidance_scale": 7.5, "num_inference_steps": 50},
        }

        body = provider._build_chutes_image_request_body(request, config)
        assert body["guidance_scale"] == 7.5
        assert body["num_inference_steps"] == 50

    def test_build_body_extra_overrides_defaults(self):
        """Test extra overrides default_params via dispatch merge."""
        provider = ChutesAdapter(
            api_key="test-key",
            unknown_fields_policy="passthrough",
            endpoint_base_urls={"image_generation": "https://chutes-test-model.chutes.ai/generate"},
        )
        request = InternalImageRequest(
            model="test-model",
            prompt="test",
            extra={"guidance_scale": 10.0},
        )
        # Use dispatch which merges extra into the raw body
        outbound = provider._build_outbound_body(request, request_type="image_generation")
        assert outbound.json_body is not None
        assert outbound.json_body["guidance_scale"] == 10.0

    def test_build_body_model_in_url_no_model_in_body(self):
        """Test model_in_url pattern doesn't include model in body."""
        provider = ChutesAdapter(api_key="test-key")
        request = InternalImageRequest(model="z-image-turbo", prompt="test")
        config = {"url_pattern": "model_in_url"}

        body = provider._build_chutes_image_request_body(request, config)
        assert "model" not in body


class TestChutesParseImageResponse:
    """Test suite for _parse_chutes_image_response method."""

    def test_parse_png_response(self):
        """Test parsing PNG binary response."""
        provider = ChutesAdapter(api_key="test-key")
        # Create minimal PNG-like binary data
        png_data = b"\x89PNG\r\n\x1a\n" + b"fake_png_content"

        response = provider._parse_chutes_image_response(png_data, "test-model")

        assert isinstance(response, InternalImageResponse)
        assert len(response.data) == 1
        assert response.data[0].b64_json is not None
        assert response.data[0].url is None
        # Verify it's valid base64
        decoded = base64.b64decode(response.data[0].b64_json)
        assert decoded == png_data

    def test_parse_response_includes_created_timestamp(self):
        """Test that response includes created timestamp."""
        provider = ChutesAdapter(api_key="test-key")
        png_data = b"\x89PNG\r\n\x1a\n" + b"fake_png_content"

        before = int(time.time())
        response = provider._parse_chutes_image_response(png_data, "test-model")
        after = int(time.time())

        assert before <= response.created <= after


class TestChutesImageGeneration:
    """Test suite for image_generation method."""

    @pytest.mark.asyncio
    async def test_image_generation_model_in_url(self):
        """Test image generation with model_in_url pattern."""
        provider = ChutesAdapter(api_key="test-key")
        request = InternalImageRequest(model="z-image-turbo", prompt="a white cat")

        # Mock the HTTP client
        fake_png = b"\x89PNG\r\n\x1a\n" + b"fake_content"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = fake_png
        mock_response.aread = AsyncMock(return_value=fake_png)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch.object(provider, "_get_client", return_value=mock_client):
            response = await provider.image_generation(request)

            # Verify URL was correct
            call_args = mock_client.post.call_args
            assert "chutes-z-image-turbo.chutes.ai/generate" in call_args[0][0]

            # Verify response
            assert len(response.data) == 1
            assert response.data[0].b64_json is not None

    @pytest.mark.asyncio
    async def test_image_generation_model_in_body(self):
        """Test image generation with model_in_body pattern."""
        provider = ChutesAdapter(api_key="test-key", unknown_fields_policy="passthrough")
        request = InternalImageRequest(
            model="Qwen-Image-2512",
            prompt="a white cat",
            size=ImageSize(width=512, height=512),
            extra={"negative_prompt": "blur"},
        )

        fake_jpeg = b"\xff\xd8\xff\xe0" + b"fake_jpeg_content"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = fake_jpeg
        mock_response.aread = AsyncMock(return_value=fake_jpeg)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch.object(provider, "_get_client", return_value=mock_client):
            response = await provider.image_generation(request)

            # Verify URL was correct
            call_args = mock_client.post.call_args
            assert "image.chutes.ai/generate" in call_args[0][0]

            # Verify body had correct parameters
            body = call_args[1]["json"]
            assert body["model"] == "Qwen-Image-2512"
            assert body["prompt"] == "a white cat"
            assert body["width"] == 512
            assert body["height"] == 512
            assert body["negative_prompt"] == "blur"

            # Verify response
            assert len(response.data) == 1
            assert response.data[0].b64_json is not None
