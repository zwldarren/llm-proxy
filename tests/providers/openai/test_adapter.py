"""Tests for OpenAI adapter."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_proxy.core.exceptions import ProviderError
from llm_proxy.models import (
    ConversationContext,
    InternalEmbeddingRequest,
    InternalRequest,
    Message,
    TextBlock,
    ToolUseBlock,
)
from llm_proxy.models.image import ImageEditSource, InternalImageEditRequest, InternalImageRequest
from llm_proxy.providers.openai.adapter import OpenAIAdapter
from llm_proxy.serialization.openai.serializer import (
    OpenAIResponsesProviderSerializer,
)


class TestOpenAIAdapter:
    """Tests for OpenAIAdapter."""

    def test_provider_name(self):
        """Test provider name is correct."""
        adapter = OpenAIAdapter(api_key="test-key")
        assert adapter.provider_name == "openai"

    def test_custom_provider_name(self):
        """Test custom provider name can be set."""
        adapter = OpenAIAdapter(
            api_key="test-key",
            provider_name="custom_responses",
        )
        assert adapter.provider_name == "custom_responses"

    def test_default_base_url(self):
        """Test default base URL is correct."""
        adapter = OpenAIAdapter(api_key="test-key")
        assert adapter._base_url == "https://api.openai.com/v1"

    def test_custom_base_url(self):
        """Test custom base URL can be set."""
        adapter = OpenAIAdapter(
            api_key="test-key",
            base_url="https://custom.api.com/v1",
        )
        assert adapter._base_url == "https://custom.api.com/v1"

    def test_build_request_body(self):
        """Test request body building (responses passthrough)."""
        adapter = OpenAIAdapter(api_key="test-key")
        request = {
            "model": "gpt-5",
            "input": "Hello, world!",
        }
        body = adapter._build_responses_passthrough_body(request, stream=False)

        assert body["model"] == "gpt-5"
        assert body["input"] == "Hello, world!"
        assert body["stream"] is False

    def test_build_request_body_with_stream(self):
        """Test request body building with stream=True (responses passthrough)."""
        adapter = OpenAIAdapter(api_key="test-key")
        request = {
            "model": "gpt-5",
            "input": "Hello, world!",
        }
        body = adapter._build_responses_passthrough_body(request, stream=True)

        assert body["stream"] is True

    def test_parse_response(self):
        """Test response parsing."""
        response = {
            "id": "resp_123",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Hello!"}],
                }
            ],
        }
        # The response is passed through as-is (no transformation needed)
        assert response == response

    def test_parse_error_response(self):
        """Test error response parsing."""
        adapter = OpenAIAdapter(api_key="test-key")
        error_body = {
            "error": {
                "type": "invalid_request_error",
                "code": "invalid_api_key",
                "message": "Invalid API key",
            }
        }
        error = adapter._parse_error_response(401, error_body)

        assert error.message == "Invalid API key"
        assert error.error_type == "invalid_request_error"
        assert error.code == "invalid_api_key"
        assert error.status_code == 401

    def test_from_provider_format_text(self):
        """Test converting response to InternalResponse with text."""
        response = {
            "id": "resp_123",
            "model": "gpt-5",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Hello, world!"}],
                }
            ],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            },
        }
        serializer = OpenAIResponsesProviderSerializer()
        llm_response = serializer.parse_provider_response(response, model="gpt-5")

        assert len(llm_response.output) == 1
        assert isinstance(llm_response.output[0], TextBlock)
        assert llm_response.output[0].text == "Hello, world!"
        assert llm_response.model == "gpt-5"
        assert llm_response.finish_reason == "stop"
        assert llm_response.usage is not None
        assert llm_response.usage.input_tokens == 100

    def test_from_provider_format_tool_calls(self):
        """Test converting response with tool calls."""
        response = {
            "id": "resp_123",
            "model": "gpt-5",
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_123",
                    "name": "get_weather",
                    "arguments": '{"location": "NYC"}',
                }
            ],
        }
        serializer = OpenAIResponsesProviderSerializer()
        llm_response = serializer.parse_provider_response(response)

        assert len(llm_response.output) == 1
        assert isinstance(llm_response.output[0], ToolUseBlock)
        tool_use = llm_response.output[0]
        assert tool_use.name == "get_weather"
        assert tool_use.id == "call_123"
        assert tool_use.input == {"location": "NYC"}
        assert llm_response.finish_reason == "tool_calls"

    async def test_embeddings(self, monkeypatch):
        """Test embeddings returns correct response."""
        adapter = OpenAIAdapter(api_key="test-key")
        request = InternalEmbeddingRequest(model="text-embedding-3-small", input="test")

        mock_response = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.1, 0.2, 0.3],
                }
            ],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        }

        call_args_list = []

        async def mock_post(*args, **kwargs):
            call_args_list.append(args)

            class MockResponse:
                status_code = 200
                ok = True
                headers = {}

                def json(self):
                    return mock_response

                async def text(self):
                    return ""

            return MockResponse()

        client = await adapter._get_client()
        monkeypatch.setattr(client, "post", mock_post)
        result = await adapter.embeddings(request)
        assert result.model == "text-embedding-3-small"
        assert len(result.data) == 1
        assert result.data[0].embedding == [0.1, 0.2, 0.3]
        assert result.usage is not None
        assert result.usage.total_tokens == 1

        # Verify request was made correctly
        assert len(call_args_list) == 1
        assert "/embeddings" in str(call_args_list[0][0])  # URL contains embeddings endpoint


@pytest.mark.asyncio
async def test_chat_completion_captures_rate_limit_headers(monkeypatch):
    """OpenAI adapter chat completion captures upstream rate-limit headers."""
    adapter = OpenAIAdapter(api_key="test-key")

    mock_response = {
        "id": "resp_rl",
        "model": "gpt-5",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "Hello!"}],
            }
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }

    class _MockResponse:
        status_code = 200
        headers = {
            "x-ratelimit-limit-requests": "1000",
            "x-ratelimit-remaining-requests": "999",
            "retry-after": "1",
        }

        def json(self):
            return mock_response

    client = await adapter._get_client()
    monkeypatch.setattr(client, "post", AsyncMock(return_value=_MockResponse()))

    request = InternalRequest(
        model="gpt-5",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="hi")])]
        ),
    )
    result = await adapter.chat_completion(request)

    assert result.provider_info.get("_rate_limit_headers") == {
        "x-ratelimit-limit-requests": "1000",
        "x-ratelimit-remaining-requests": "999",
        "retry-after": "1",
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestOpenAIStreamingErrorPassthrough:
    """ProviderError raised mid-stream keeps its metadata (status, type, message).

    Regression: the stream generators used to funnel every exception through
    _handle_http_error, which re-wraps ProviderError as a generic
    "openai request failed: ..." api_error with status_code=None — turning
    upstream 4xx errors into 500s and dropping the original error body.
    """

    @staticmethod
    def _error_client() -> MagicMock:
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.headers = {}
        mock_response.text = (
            '{"error": {"message": "bad request", "type": "invalid_request_error"}}'
        )
        mock_response.aread = AsyncMock()
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        return mock_client

    @staticmethod
    def _request() -> InternalRequest:
        return InternalRequest(
            model="gpt-5",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
        )

    async def test_stream_chat_preserves_provider_error_metadata(self):
        adapter = OpenAIAdapter(api_key="test-key")
        with patch.object(adapter, "_get_client", return_value=self._error_client()):
            stream_gen = await adapter.stream_chat_completion(self._request())
            with pytest.raises(ProviderError) as exc_info:
                async for _chunk in stream_gen:
                    pass

        error = exc_info.value
        assert error.status_code == 400
        assert error.error_type == "invalid_request_error"
        assert error.message == "bad request"
        assert "request failed" not in error.message

    async def test_stream_chat_native_preserves_provider_error_metadata(self):
        adapter = OpenAIAdapter(api_key="test-key")
        with patch.object(adapter, "_get_client", return_value=self._error_client()):
            stream_gen = await adapter.stream_chat_completion_native(self._request())
            with pytest.raises(ProviderError) as exc_info:
                async for _chunk in stream_gen:
                    pass

        error = exc_info.value
        assert error.status_code == 400
        assert error.error_type == "invalid_request_error"
        assert error.message == "bad request"
        assert "request failed" not in error.message

    async def test_stream_image_generation_preserves_provider_error_metadata(self):
        """Exercises the shared ImageCapabilityMixin._stream_image_request path."""
        adapter = OpenAIAdapter(api_key="test-key")
        request = InternalImageRequest(model="dall-e-3", prompt="a cat")
        with patch.object(adapter, "_get_client", return_value=self._error_client()):
            stream_gen = await adapter.stream_image_generation(request)
            with pytest.raises(ProviderError) as exc_info:
                async for _chunk in stream_gen:
                    pass

        error = exc_info.value
        assert error.status_code == 400
        assert error.error_type == "invalid_request_error"
        assert error.message == "bad request"
        assert "request failed" not in error.message

    async def test_image_edit_uploads_send_multipart_form_data(self):
        """Regression: image edits with uploaded files must POST multipart.

        ``_build_outbound_body`` returns ``form_data`` + ``files`` (json_body
        is None) when the request carries uploaded image bytes. A previous
        implementation only handled json_body and raised
        "_build_outbound_body returned no json_body for image_edit" — the
        standard OpenAI /v1/images/edits flow (multipart upload) was broken.
        """
        adapter = OpenAIAdapter(api_key="test-key")
        request = InternalImageEditRequest(
            model="gpt-image-1",
            prompt="add a hat",
            images=[
                ImageEditSource(
                    file=b"fake-png-bytes",
                    filename="input.png",
                    content_type="image/png",
                )
            ],
            mask=ImageEditSource(
                file=b"fake-mask-bytes",
                filename="mask.png",
                content_type="image/png",
            ),
        )

        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"data": [], "created": 1234567890}
        client = MagicMock()
        client.post = AsyncMock(return_value=response)
        adapter._get_client = AsyncMock(return_value=client)
        adapter._check_response_status = AsyncMock()

        async def fake_with_retry(op, *a, **kw):
            return await op()

        adapter._with_retry = fake_with_retry

        result = await adapter.image_edit(request)

        assert result is not None
        call_kwargs = client.post.call_args.kwargs
        assert client.post.await_args.args[0] == "https://api.openai.com/v1/images/edits"
        assert call_kwargs["files"] == [
            (
                "image[]",
                ("input.png", b"fake-png-bytes", "image/png"),
            ),
            (
                "mask",
                ("mask.png", b"fake-mask-bytes", "image/png"),
            ),
        ]
        assert call_kwargs["data"]["model"] == "gpt-image-1"
        assert call_kwargs["data"]["prompt"] == "add a hat"
        # The JSON Content-Type header must be dropped so httpx sets the
        # multipart boundary itself.
        assert "Content-Type" not in call_kwargs["headers"]

    async def test_image_edit_file_id_still_posts_json(self):
        """Image edits referencing a File API id (no upload) keep JSON body."""
        adapter = OpenAIAdapter(api_key="test-key")
        request = InternalImageEditRequest(
            model="gpt-image-1",
            prompt="add a hat",
            images=[ImageEditSource(file_id="file_123")],
        )
        captured: dict = {}

        async def fake_post(_url, _headers, body):
            captured["body"] = body
            return {"data": [], "created": 1}

        adapter._post_json_with_retry = fake_post

        await adapter.image_edit(request)

        assert captured["body"]["images"] == [{"file_id": "file_123"}]
