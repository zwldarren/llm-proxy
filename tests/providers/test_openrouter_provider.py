"""Tests for OpenRouterAdapter native implementation."""

import base64

import pytest

from llm_proxy.models import TextBlock
from llm_proxy.providers.openrouter import OpenRouterAdapter
from llm_proxy.providers.openrouter.adapter import _infer_audio_format


class TestOpenRouterAdapter:
    """Test suite for OpenRouterAdapter."""

    def test_default_initialization(self):
        """Test default provider initialization."""
        provider = OpenRouterAdapter(api_key="test-key")
        assert provider.provider_name == "openrouter"
        assert provider._base_url == "https://openrouter.ai/api/v1"

    def test_custom_base_url(self):
        """Test custom base URL configuration."""
        provider = OpenRouterAdapter(
            api_key="test-key", base_url="https://custom.openrouter.com/v1"
        )
        assert provider._base_url == "https://custom.openrouter.com/v1"

    def test_custom_headers(self):
        """Test custom headers are stored."""
        provider = OpenRouterAdapter(
            api_key="test-key",
            custom_headers={"HTTP-Referer": "https://myapp.com", "X-Title": "MyApp"},
        )
        assert provider._custom_headers["HTTP-Referer"] == "https://myapp.com"
        assert provider._custom_headers["X-Title"] == "MyApp"

    def test_build_headers_includes_custom_headers(self):
        """Test that custom headers are included in request headers."""
        provider = OpenRouterAdapter(
            api_key="test-key", custom_headers={"HTTP-Referer": "https://myapp.com"}
        )
        headers = provider._build_headers()
        assert headers["Authorization"] == "Bearer test-key"
        assert headers["HTTP-Referer"] == "https://myapp.com"

    def test_post_process_chat_response_converts_reasoning_to_reasoning_content(self):
        """Test that reasoning field is extracted as ThinkingBlock by the serializer."""
        from llm_proxy.models import ThinkingBlock
        from llm_proxy.serialization.providers import get_provider_serializer

        serializer = get_provider_serializer("openrouter")

        # OpenRouter response with 'reasoning' field
        response = {
            "id": "resp_test",
            "model": "test-model",
            "choices": [
                {
                    "message": {
                        "content": "Hello",
                        "reasoning": "Let me think about this...",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

        result = serializer.parse_provider_response(response, model="test-model")

        # Check that reasoning was added as ThinkingBlock
        thinking_blocks = [b for b in result.output if isinstance(b, ThinkingBlock)]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0].thinking == "Let me think about this..."

    def test_post_process_chat_response_preserves_reasoning_content_if_present(self):
        """Test that reasoning_content takes precedence over reasoning."""
        from llm_proxy.models import ThinkingBlock
        from llm_proxy.serialization.providers import get_provider_serializer

        serializer = get_provider_serializer("openrouter")

        # Response with both reasoning and reasoning_content (reasoning_content takes precedence)
        response = {
            "id": "resp_test",
            "model": "test-model",
            "choices": [
                {
                    "message": {
                        "content": "Hello",
                        "reasoning": "Old reasoning",
                        "reasoning_content": "New reasoning content",
                    },
                    "finish_reason": "stop",
                }
            ],
        }

        result = serializer.parse_provider_response(response, model="test-model")

        # Check that reasoning_content was added as ThinkingBlock (takes precedence)
        thinking_blocks = [b for b in result.output if isinstance(b, ThinkingBlock)]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0].thinking == "New reasoning content"

    def test_normalize_reasoning_for_request_converts_reasoning_content_to_reasoning(self):
        """Test that reasoning_content in assistant messages is converted to reasoning."""
        from llm_proxy.serialization.openai.components.request_builder import OpenAIRequestBuilder

        builder = OpenAIRequestBuilder()
        body = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {
                    "role": "assistant",
                    "reasoning_content": "Let me think...",
                    "content": "Hi there",
                },
            ],
            "model": "test-model",
        }

        result = builder.normalize_reasoning_for_request(
            body, base_url="https://test.example.com", preferred="reasoning"
        )

        assistant_msg = next(m for m in result["messages"] if m["role"] == "assistant")
        assert "reasoning_content" not in assistant_msg
        assert assistant_msg.get("reasoning") == "Let me think..."

    def test_extract_unknown_response_fields_identifies_unknown_fields(self):
        """Test that unknown response fields are extracted correctly."""
        from llm_proxy.serialization.providers.chat_completions import OpenAIProviderSerializer

        serializer = OpenAIProviderSerializer()

        response = {
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [{"message": {"content": "Hello"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "id_provider": "anthropic",
            "cost": 0.001,
            "native_tokens_prompt": 100,
            "native_tokens_completion": 50,
        }

        unknown = serializer.extract_unknown_response_fields(response)

        assert "id" not in unknown
        assert "model" not in unknown
        assert "choices" not in unknown
        assert "usage" not in unknown
        assert unknown["id_provider"] == "anthropic"
        assert unknown["cost"] == 0.001
        assert unknown["native_tokens_prompt"] == 100
        assert unknown["native_tokens_completion"] == 50

    def test_extract_unknown_response_fields_returns_empty_for_known_only(self):
        """Test that no unknown fields are returned when response has only known fields."""
        from llm_proxy.serialization.providers.chat_completions import OpenAIProviderSerializer

        serializer = OpenAIProviderSerializer()

        response = {
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [{"message": {"content": "Hello"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "system_fingerprint": "fp_123",
            "object": "chat.completion",
            "created": 1234567890,
        }

        unknown = serializer.extract_unknown_response_fields(response)

        assert unknown == {}

    async def test_unknown_fields_preserved_in_provider_info(self, monkeypatch):
        """Test that unknown fields from OpenRouter response are preserved in provider_info."""
        from llm_proxy.models import ConversationContext, InternalRequest, Message

        provider = OpenRouterAdapter(api_key="test-key")

        openrouter_response = {
            "id": "gen-123",
            "model": "anthropic/claude-3-opus",
            "choices": [
                {
                    "message": {"content": "Hello"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "id_provider": "anthropic",
            "cost": 0.0025,
            "native_tokens_prompt": 150,
            "native_tokens_completion": 75,
        }

        async def mock_post(*args, **kwargs):
            class MockResponse:
                status_code = 200

                def raise_for_status(self):
                    pass

                def json(self):
                    return openrouter_response

            return MockResponse()

        client = await provider._get_client()
        monkeypatch.setattr(client, "post", mock_post)

        request = InternalRequest(
            model="anthropic/claude-3-opus",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Hello")])]
            ),
        )
        result = await provider.chat_completion(request)

        assert result.provider_info["provider"] == "openai"
        assert result.provider_info["id_provider"] == "anthropic"
        assert result.provider_info["cost"] == 0.0025
        assert result.provider_info["native_tokens_prompt"] == 150
        assert result.provider_info["native_tokens_completion"] == 75


class TestOpenRouterSTT:
    """Tests for OpenRouter STT (JSON-based, not multipart)."""

    def test_infer_audio_format_wav(self):
        """.wav filename infers wav format."""
        assert _infer_audio_format("audio.wav") == "wav"

    def test_infer_audio_format_mp3(self):
        """.mp3 filename infers mp3 format."""
        assert _infer_audio_format("audio.mp3") == "mp3"

    def test_infer_audio_format_unknown(self):
        """Unknown extension raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported audio format extension"):
            _infer_audio_format("audio.xyz")

    def test_infer_audio_format_no_extension(self):
        """No extension raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported audio format extension"):
            _infer_audio_format("audio")

    def test_infer_audio_format_flac(self):
        """.flac filename infers flac format."""
        assert _infer_audio_format("audio.flac") == "flac"

    def test_infer_audio_format_pcm(self):
        """.pcm filename infers pcm16 format."""
        assert _infer_audio_format("audio.pcm") == "pcm16"

    async def test_transcription_builds_json_body(self, monkeypatch):
        """transcription() sends JSON with input_audio instead of multipart."""
        from llm_proxy.models import InternalTranscriptionRequest

        provider = OpenRouterAdapter(api_key="test-key")

        captured_url = None
        captured_headers = None
        captured_body = None

        async def mock_post(url, headers=None, json=None, **kwargs):
            nonlocal captured_url, captured_headers, captured_body
            captured_url = url
            captured_headers = headers
            captured_body = json

            class MockResponse:
                status_code = 200
                headers = {"content-type": "application/json"}

                def raise_for_status(self):
                    pass

                def json(self):
                    return {"text": "Hello world"}

            return MockResponse()

        client = await provider._get_client()
        monkeypatch.setattr(client, "post", mock_post)

        request = InternalTranscriptionRequest(
            model="openai/whisper-1",
            file=b"fake_audio_data",
            filename="audio.wav",
        )

        result = await provider.transcription(request)

        # Verify JSON body format
        assert captured_body is not None
        assert captured_body["model"] == "openai/whisper-1"
        assert "input_audio" in captured_body
        assert captured_body["input_audio"]["format"] == "wav"
        # Verify base64 encoding
        assert captured_body["input_audio"]["data"] == base64.b64encode(b"fake_audio_data").decode()
        # Verify content type is JSON
        assert captured_headers["Content-Type"] == "application/json"
        # Verify result
        assert result.text == "Hello world"

    async def test_transcription_with_language(self, monkeypatch):
        """transcription() includes language parameter when provided."""
        from llm_proxy.models import InternalTranscriptionRequest

        provider = OpenRouterAdapter(api_key="test-key")

        captured_body = None

        async def mock_post(url, headers=None, json=None, **kwargs):
            nonlocal captured_body
            captured_body = json

            class MockResponse:
                status_code = 200
                headers = {"content-type": "application/json"}

                def raise_for_status(self):
                    pass

                def json(self):
                    return {"text": "Bonjour"}

            return MockResponse()

        client = await provider._get_client()
        monkeypatch.setattr(client, "post", mock_post)

        request = InternalTranscriptionRequest(
            model="openai/whisper-1",
            file=b"audio_data",
            filename="audio.mp3",
            language="fr",
        )

        result = await provider.transcription(request)

        assert captured_body["language"] == "fr"
        assert captured_body["input_audio"]["format"] == "mp3"
        assert result.text == "Bonjour"


class TestOpenRouterImageGeneration:
    """Tests for OpenRouterAdapter image generation."""

    def test_images_endpoint_override(self):
        """OpenRouter uses /images (not /images/generations)."""
        provider = OpenRouterAdapter(api_key="test-key")
        assert provider.IMAGES_ENDPOINT == "/images"

    async def test_image_generation_url(self):
        """The image generation URL uses the overridden endpoint."""
        provider = OpenRouterAdapter(api_key="test-key")
        url = provider._image_generation_url(model="bytedance-seed/seedream-4.5")
        assert url == "https://openrouter.ai/api/v1/images"

    async def test_image_generation_extracts_openrouter_cost(self, monkeypatch):
        """usage.cost from the response is stored in provider_info."""

        provider = OpenRouterAdapter(api_key="test-key")

        async def _fake_post(*args, **kwargs):
            return {
                "created": 1748372400,
                "data": [{"b64_json": "abc123"}],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 4175,
                    "total_tokens": 4175,
                    "cost": 0.04,
                },
            }

        monkeypatch.setattr(provider, "_post_json_with_retry", _fake_post)

        from llm_proxy.models.image import InternalImageRequest

        request = InternalImageRequest(
            model="bytedance-seed/seedream-4.5",
            prompt="a red panda",
        )

        result = await provider.image_generation(request)

        assert result.provider_info.get("openrouter_cost") == 0.04
        assert len(result.data) == 1
        assert result.data[0].b64_json == "abc123"

    async def test_image_generation_no_cost(self, monkeypatch):
        """When usage.cost is absent, provider_info is not polluted."""
        provider = OpenRouterAdapter(api_key="test-key")

        async def _fake_post(*args, **kwargs):
            return {
                "created": 1748372400,
                "data": [{"b64_json": "abc123"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 100, "total_tokens": 100},
            }

        monkeypatch.setattr(provider, "_post_json_with_retry", _fake_post)

        from llm_proxy.models.image import InternalImageRequest

        request = InternalImageRequest(
            model="bytedance-seed/seedream-4.5",
            prompt="a red panda",
        )

        result = await provider.image_generation(request)

        assert "openrouter_cost" not in result.provider_info


class TestOpenRouterChatPostProcess:
    """Regression tests for OpenRouter-specific response metadata extraction."""

    def test_chat_response_extracts_cost_details_and_is_byok(self):
        from llm_proxy.models import InternalResponse, TextBlock

        provider = OpenRouterAdapter(api_key="test-key")
        response = {
            "id": "resp-1",
            "model": "openai/gpt-5",
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "cost": 0.005,
                "cost_details": {"prompt": 0.002, "completion": 0.003},
                "is_byok": True,
            },
        }
        internal = InternalResponse(
            id="resp-1",
            model="requested-model",
            output=[TextBlock(text="hi")],
            usage=None,
            finish_reason="stop",
            provider_info={"provider": "openrouter"},
        )
        result = provider._post_process_chat_response(response, internal)
        assert result.provider_info["openrouter_cost"] == 0.005
        assert result.provider_info["openrouter_cost_details"] == {
            "prompt": 0.002,
            "completion": 0.003,
        }
        assert result.provider_info["openrouter_is_byok"] is True
