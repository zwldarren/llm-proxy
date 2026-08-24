"""Tests for OpenAI provider adapter."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_proxy.core.adapter import get_adapter, list_providers
from llm_proxy.core.exceptions import ProviderError
from llm_proxy.models import (
    ConversationContext,
    GenerationParams,
    InternalRequest,
    InternalResponse,
    Message,
    OpenAISpecificParams,
    TextBlock,
)
from llm_proxy.providers.openai_compatible._base import OpenAICompatibleBase


class MockResponse:
    """Mock HTTP response for httpx2.

    For non-streaming responses, json() is synchronous in httpx2.
    """

    def __init__(self, status_code: int, json_data: dict | None = None):
        self.status_code = status_code
        self._json_data = json_data or {}

    def json(self):
        """Return JSON data (synchronous for non-streaming responses)."""
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


def test_openai_adapter_is_registered():
    """Test that OpenAI adapter is registered."""
    assert "deepseek" in list_providers()
    assert "openai-compatible" in list_providers()


def test_deepseek_adapter_is_registered():
    """Test that DeepSeek adapter is registered."""
    assert "deepseek" in list_providers()


def test_deepseek_adapter_can_be_created():
    """Test that DeepSeek adapter can be instantiated."""
    adapter = get_adapter("deepseek", api_key="test-key")
    assert adapter.__class__.__name__ == "DeepSeekAdapter"


@pytest.fixture
def openai_adapter():
    """Create an OpenAI adapter for testing."""
    return OpenAICompatibleBase(
        api_key="test-key",
        base_url="https://api.openai.com/v1",
    )


@pytest.mark.asyncio
async def test_openai_chat_completion(openai_adapter):
    """Test OpenAI chat completion request."""
    mock_response = MockResponse(
        200,
        json_data={
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello, world!",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch(
        "llm_proxy.providers.openai_compatible._base.AsyncSession", return_value=mock_client
    ):
        openai_adapter._http_client = mock_client

        request = InternalRequest(
            model="gpt-4",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Hello")])]
            ),
        )
        response = await openai_adapter.chat_completion(request)

        assert isinstance(response, InternalResponse)
        assert len(response.output) == 1
        assert isinstance(response.output[0], TextBlock)
        assert response.output[0].text == "Hello, world!"
        assert response.model == "gpt-4"
        assert response.finish_reason == "stop"
        assert response.usage is not None
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 5


class TestOpenAIAdapterNewParams:
    """Tests for new request parameters in adapter."""

    def test_build_request_body_with_audio_params(self, openai_adapter):
        """Test audio parameters are included in request body."""
        request = InternalRequest(
            model="gpt-4o-audio-preview",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Hello")])]
            ),
            params=GenerationParams(
                openai=OpenAISpecificParams(
                    audio={"format": "wav", "voice": "alloy"},
                    modalities=["text", "audio"],
                )
            ),
        )
        body = openai_adapter._build_request_body(request)

        assert body["audio"] == {"format": "wav", "voice": "alloy"}
        assert body["modalities"] == ["text", "audio"]

    def test_build_request_body_with_reasoning_effort(self, openai_adapter):
        """Test reasoning_effort is included in request body."""
        request = InternalRequest(
            model="o1",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Solve")])]
            ),
            params=GenerationParams(openai=OpenAISpecificParams(reasoning_effort="high")),
        )
        body = openai_adapter._build_request_body(request)

        assert body["reasoning_effort"] == "high"

    def test_build_request_body_with_web_search_options(self, openai_adapter):
        """Test web_search_options is included in request body."""
        request = InternalRequest(
            model="gpt-4o",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Search")])]
            ),
            params=GenerationParams(
                openai=OpenAISpecificParams(web_search_options={"search_context_size": "high"})
            ),
        )
        body = openai_adapter._build_request_body(request)

        assert body["web_search_options"] == {"search_context_size": "high"}

    def test_build_request_body_with_service_tier(self, openai_adapter):
        """Test service_tier is included in request body."""
        request = InternalRequest(
            model="gpt-4o",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Hello")])]
            ),
            params=GenerationParams(openai=OpenAISpecificParams(service_tier="flex")),
        )
        body = openai_adapter._build_request_body(request)

        assert body["service_tier"] == "flex"

    def test_build_request_body_with_all_new_params(self, openai_adapter):
        """Test all new parameters are included in request body."""
        request = InternalRequest(
            model="gpt-4o",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Hello")])]
            ),
            params=GenerationParams(
                openai=OpenAISpecificParams(
                    audio={"format": "wav", "voice": "alloy"},
                    modalities=["text", "audio"],
                    reasoning_effort="high",
                    prediction={"type": "content", "content": "test"},
                    web_search_options={"search_context_size": "high"},
                    prompt_cache_key="key-123",
                    prompt_cache_retention="24h",
                    safety_identifier="user-123",
                    service_tier="flex",
                    verbosity="low",
                    store=True,
                    metadata={"key": "value"},
                    logprobs=True,
                    top_logprobs=5,
                )
            ),
        )
        body = openai_adapter._build_request_body(request)

        assert body["audio"] == {"format": "wav", "voice": "alloy"}
        assert body["modalities"] == ["text", "audio"]
        assert body["reasoning_effort"] == "high"
        assert body["prediction"] == {"type": "content", "content": "test"}
        assert body["web_search_options"] == {"search_context_size": "high"}
        assert body["service_tier"] == "flex"
        assert body["verbosity"] == "low"
        assert body["store"] is True
        assert body["metadata"] == {"key": "value"}
        assert body["prompt_cache_key"] == "key-123"
        assert body["prompt_cache_retention"] == "24h"
        assert body["safety_identifier"] == "user-123"
        assert body["logprobs"] is True
        assert body["top_logprobs"] == 5

    def test_build_request_body_omits_none_params(self, openai_adapter):
        """Test that None parameters are omitted from request body."""
        request = InternalRequest(
            model="gpt-4o",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Hello")])]
            ),
        )
        body = openai_adapter._build_request_body(request)

        assert "audio" not in body
        assert "modalities" not in body
        assert "reasoning_effort" not in body
        assert "web_search_options" not in body
        assert "service_tier" not in body

    def test_build_request_body_keeps_unknown_extra_when_policy_passthrough(self, openai_adapter):
        """Test passthrough policy keeps unknown extra fields."""
        adapter = OpenAICompatibleBase(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            unknown_fields_policy="passthrough",
        )
        request = InternalRequest(
            model="gpt-4o",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Hello")])]
            ),
            extra={"x_custom_flag": True},
        )

        body = adapter._build_request_body(request)
        assert body["x_custom_flag"] is True

    def test_build_request_body_ignores_unknown_extra_when_policy_ignore(self):
        """Test ignore policy removes unknown extra fields before upstream request."""
        adapter = OpenAICompatibleBase(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            unknown_fields_policy="ignore",
        )
        request = InternalRequest(
            model="gpt-4o",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Hello")])]
            ),
            extra={"x_custom_flag": True, "x_debug_hint": "a"},
        )

        body = adapter._build_request_body(request)
        assert "x_custom_flag" not in body
        assert "x_debug_hint" not in body

    def test_build_request_body_errors_unknown_extra_when_policy_error(self):
        """Test error policy rejects unknown extra fields with a validation error."""
        adapter = OpenAICompatibleBase(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            unknown_fields_policy="error",
        )
        request = InternalRequest(
            model="gpt-4o",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Hello")])]
            ),
            extra={"x_custom_flag": True},
        )

        with pytest.raises(ProviderError, match="unknown request fields"):
            adapter._build_request_body(request)


class TestReasoningFieldAutoDetection:
    """Tests for automatic reasoning field detection and conversion."""

    def test_detect_reasoning_field_from_response(self):
        """Test detecting 'reasoning' field from provider response."""
        from llm_proxy.providers.reasoning import detect_reasoning_field_in_message

        message = {"role": "assistant", "content": "Hello", "reasoning": "Let me think..."}
        result = detect_reasoning_field_in_message(message)
        assert result == "reasoning"

    def test_detect_reasoning_content_field_from_response(self):
        """Test detecting 'reasoning_content' field from provider response."""
        from llm_proxy.providers.reasoning import detect_reasoning_field_in_message

        message = {"role": "assistant", "content": "Hello", "reasoning_content": "Let me think..."}
        result = detect_reasoning_field_in_message(message)
        assert result == "reasoning_content"

    def test_detect_no_reasoning_field(self):
        """Test when no reasoning field is present."""
        from llm_proxy.providers.reasoning import detect_reasoning_field_in_message

        message = {"role": "assistant", "content": "Hello"}
        result = detect_reasoning_field_in_message(message)
        assert result is None

    def test_normalize_reasoning_in_stream_chunk(self):
        """Test converting 'reasoning' to 'reasoning_content' in stream chunk."""
        from llm_proxy.providers.reasoning import normalize_reasoning_in_stream_chunk

        chunk = {"choices": [{"delta": {"content": "Hello", "reasoning": "Let me think..."}}]}
        normalized = normalize_reasoning_in_stream_chunk(chunk)
        delta = normalized["choices"][0]["delta"]
        assert "reasoning_content" in delta
        assert delta["reasoning_content"] == "Let me think..."
        assert "reasoning" not in delta

    def test_normalize_preserves_reasoning_content(self):
        """Test that 'reasoning_content' is preserved in stream chunk."""
        from llm_proxy.providers.reasoning import normalize_reasoning_in_stream_chunk

        chunk = {
            "choices": [{"delta": {"content": "Hello", "reasoning_content": "Let me think..."}}]
        }
        normalized = normalize_reasoning_in_stream_chunk(chunk)
        delta = normalized["choices"][0]["delta"]
        assert delta["reasoning_content"] == "Let me think..."
        assert "reasoning" not in delta

    def test_normalize_reasoning_for_request_default(self, openai_adapter):
        """Test that 'reasoning_content' is preserved by default (no cache)."""
        from llm_proxy.models import ThinkingBlock

        request = InternalRequest(
            model="gpt-4",
            conversation=ConversationContext(
                messages=[
                    Message(role="assistant", content=[ThinkingBlock(thinking="Previous thought")])
                ]
            ),
        )
        body = openai_adapter._build_request_body(request)
        assistant_msg = body["messages"][0]
        assert "reasoning_content" in assistant_msg
        assert assistant_msg["reasoning_content"] == "Previous thought"

    def test_normalize_reasoning_for_request_with_cache(self, openai_adapter):
        """Test that cached preference converts reasoning_content to reasoning."""
        from llm_proxy.models import ThinkingBlock
        from llm_proxy.serialization.openai.components.request_builder import OpenAIRequestBuilder

        base_url = "https://api.openai.com/v1"
        from llm_proxy.serialization.context import BuildContext

        builder = OpenAIRequestBuilder()
        builder.set_reasoning_field_preference(base_url, "reasoning")

        try:
            request = InternalRequest(
                model="gpt-4",
                conversation=ConversationContext(
                    messages=[
                        Message(
                            role="assistant",
                            content=[ThinkingBlock(thinking="Previous thought")],
                        )
                    ]
                ),
            )
            body = builder.build(request, BuildContext.from_request(request, base_url=base_url))
            assistant_msg = body["messages"][0]
            assert "reasoning" in assistant_msg
            assert assistant_msg["reasoning"] == "Previous thought"
            assert "reasoning_content" not in assistant_msg
        finally:
            builder.set_reasoning_field_preference(base_url, "reasoning_content")

    def test_adapter_detects_reasoning_field_from_response(self):
        """Generic OpenAI-compatible adapter switches to `reasoning` after
        the upstream provider returns it in a response."""
        from llm_proxy.models import ThinkingBlock

        adapter = OpenAICompatibleBase(
            api_key="test-key",
            base_url="https://reasoning-provider.example.com/v1",
        )
        request = InternalRequest(
            model="test",
            conversation=ConversationContext(
                messages=[
                    Message(
                        role="assistant",
                        content=[ThinkingBlock(thinking="Previous thought")],
                    )
                ]
            ),
        )

        # First request: default to reasoning_content.
        body = adapter._build_request_body(request)
        assert "reasoning_content" in body["messages"][0]
        assert "reasoning" not in body["messages"][0]

        # Simulate a provider response that uses `reasoning`.
        response = {
            "id": "resp_1",
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "hi",
                        "reasoning": "I think",
                    },
                    "finish_reason": "stop",
                }
            ],
        }
        adapter._parse_response(
            adapter._get_serializer(),
            response,
            model="test",
            request_id="req_1",
            base_url=adapter._base_url,
        )

        # Next request: should now use `reasoning`.
        try:
            body = adapter._build_request_body(request)
            assert "reasoning" in body["messages"][0]
            assert body["messages"][0]["reasoning"] == "Previous thought"
            assert "reasoning_content" not in body["messages"][0]
        finally:
            # Restore the default preference to avoid leaking state to other
            # tests. The cache lives on the shared serializer builder, so the
            # cleanup must use the adapter's own builder.
            adapter._get_request_builder().clear_reasoning_field_preference(adapter._base_url)

    def test_openrouter_adapter_uses_reasoning_field(self):
        """OpenRouter always expects assistant reasoning as `reasoning`."""
        from llm_proxy.models import ThinkingBlock
        from llm_proxy.providers.openrouter.adapter import OpenRouterAdapter

        adapter = OpenRouterAdapter(api_key="test-key")
        request = InternalRequest(
            model="test",
            conversation=ConversationContext(
                messages=[
                    Message(
                        role="assistant",
                        content=[ThinkingBlock(thinking="Previous thought")],
                    )
                ]
            ),
        )
        body = adapter._build_request_body(request)
        assert "reasoning" in body["messages"][0]
        assert body["messages"][0]["reasoning"] == "Previous thought"
        assert "reasoning_content" not in body["messages"][0]

    def test_nanogpt_adapter_uses_reasoning_field(self):
        """NanoGPT always expects assistant reasoning as `reasoning`."""
        from llm_proxy.models import ThinkingBlock
        from llm_proxy.providers.nanogpt.adapter import NanoGPTAdapter

        adapter = NanoGPTAdapter(api_key="test-key")
        request = InternalRequest(
            model="test",
            conversation=ConversationContext(
                messages=[
                    Message(
                        role="assistant",
                        content=[ThinkingBlock(thinking="Previous thought")],
                    )
                ]
            ),
        )
        body = adapter._build_request_body(request)
        assert "reasoning" in body["messages"][0]
        assert body["messages"][0]["reasoning"] == "Previous thought"
        assert "reasoning_content" not in body["messages"][0]

    def test_stream_chunk_teaches_reasoning_preference(self):
        """A streamed chunk carrying `reasoning` renames for the client AND
        teaches the preference, so the next request converts the client's
        `reasoning_content` echo back to the provider's field."""
        from llm_proxy.models import ThinkingBlock

        base_url = "https://streaming-reasoning.example.com/v1"
        adapter = OpenAICompatibleBase(api_key="test-key", base_url=base_url)
        request = InternalRequest(
            model="test",
            conversation=ConversationContext(
                messages=[
                    Message(
                        role="assistant",
                        content=[ThinkingBlock(thinking="Previous thought")],
                    )
                ]
            ),
        )

        # First request: no detection has run yet, default to reasoning_content.
        body = adapter._build_request_body(request)
        assert "reasoning_content" in body["messages"][0]
        assert "reasoning" not in body["messages"][0]

        # A chunk carrying the raw provider field (`reasoning`) is renamed
        # for the client and teaches the preference.
        chunk = {
            "choices": [{"index": 0, "delta": {"reasoning": "I think"}, "finish_reason": None}]
        }
        try:
            transformed = adapter._stream_transform_chunk(chunk, {"model": "test"})
            delta = transformed["choices"][0]["delta"]
            assert delta["reasoning_content"] == "I think"
            assert "reasoning" not in delta

            # Next request: the echo now uses the provider's field.
            body = adapter._build_request_body(request)
            assert "reasoning" in body["messages"][0]
            assert body["messages"][0]["reasoning"] == "Previous thought"
            assert "reasoning_content" not in body["messages"][0]
        finally:
            adapter._get_request_builder().clear_reasoning_field_preference(base_url)

    @pytest.mark.asyncio
    async def test_wire_reuse_response_teaches_reasoning_preference(self, mock_response_cls):
        """The verbatim (wire-reuse) response tier renames `reasoning` for
        the client AND learns the preference, so a later request converts
        the client's `reasoning_content` echo back to the provider's field."""
        from llm_proxy.models import ConversionTier, ThinkingBlock

        base_url = "https://wire-reuse-reasoning.example.com/v1"
        adapter = OpenAICompatibleBase(api_key="test-key", base_url=base_url)
        req = InternalRequest(
            model="test",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            params=GenerationParams(),
        )
        req.metadata.protocol_name = "openai"

        upstream = {
            "id": "chatcmpl-9",
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "hi",
                        # Raw provider field name — the verbatim tier must
                        # rename it for the client.
                        "reasoning": "thinking hard",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response_cls(json_data=upstream))

        with patch.object(adapter, "_get_client", return_value=mock_client):
            result = await adapter.chat_completion(req)

        assert req.response_tier == ConversionTier.WIRE_REUSE
        raw = result.provider_info["_raw_response_body"]
        assert raw["choices"][0]["message"]["reasoning_content"] == "thinking hard"
        assert "reasoning" not in raw["choices"][0]["message"]

        # The learned preference now drives the request side.
        request2 = InternalRequest(
            model="test",
            conversation=ConversationContext(
                messages=[
                    Message(
                        role="assistant",
                        content=[ThinkingBlock(thinking="Previous thought")],
                    )
                ]
            ),
        )
        try:
            body = adapter._build_request_body(request2)
            assert body["messages"][0]["reasoning"] == "Previous thought"
            assert "reasoning_content" not in body["messages"][0]
        finally:
            adapter._get_request_builder().clear_reasoning_field_preference(base_url)

    def test_reasoning_preference_is_per_model(self):
        """Models on the same base URL keep their own reasoning field.

        A gateway can serve ``reasoning``-field and ``reasoning_content``-field
        models side by side; one model's response must never clobber another
        model's learned preference, and never-seen models keep the default.
        """
        from llm_proxy.serialization.openai.components.request_builder import (
            OpenAIRequestBuilder,
        )

        base_url = "https://mixed-models.example.com/v1"
        builder = OpenAIRequestBuilder()
        builder.set_reasoning_field_preference(base_url, "reasoning", model="model-a")
        builder.set_reasoning_field_preference(base_url, "reasoning_content", model="model-b")

        assert builder.get_reasoning_field_preference(base_url, "model-a") == "reasoning"
        assert builder.get_reasoning_field_preference(base_url, "model-b") == "reasoning_content"
        # Never-seen models default to OpenAI's reasoning_content — the
        # base_url-level fallback exists only for model-less lookups.
        assert builder.get_reasoning_field_preference(base_url, "model-c") == "reasoning_content"

    def test_model_stream_chunk_teaches_only_its_own_model(self):
        """A streamed `reasoning` chunk for model A never changes the field
        used for model B's echo on the same base URL."""
        from llm_proxy.models import ThinkingBlock

        base_url = "https://per-model-streaming.example.com/v1"
        adapter = OpenAICompatibleBase(api_key="test-key", base_url=base_url)
        chunk = {
            "choices": [{"index": 0, "delta": {"reasoning": "I think"}, "finish_reason": None}]
        }
        try:
            adapter._stream_transform_chunk(chunk, {"model": "model-a"})

            body_a = adapter._build_request_body(
                InternalRequest(
                    model="model-a",
                    conversation=ConversationContext(
                        messages=[
                            Message(
                                role="assistant",
                                content=[ThinkingBlock(thinking="Previous thought")],
                            )
                        ]
                    ),
                )
            )
            assert body_a["messages"][0]["reasoning"] == "Previous thought"
            assert "reasoning_content" not in body_a["messages"][0]

            body_b = adapter._build_request_body(
                InternalRequest(
                    model="model-b",
                    conversation=ConversationContext(
                        messages=[
                            Message(
                                role="assistant",
                                content=[ThinkingBlock(thinking="Previous thought")],
                            )
                        ]
                    ),
                )
            )
            assert body_b["messages"][0]["reasoning_content"] == "Previous thought"
            assert "reasoning" not in body_b["messages"][0]
        finally:
            adapter._get_request_builder().clear_reasoning_field_preference(base_url)

    def test_wire_reuse_learning_is_per_model(self):
        """The verbatim response tier learns per model: model A's `reasoning`
        response does not affect model B's echo field."""
        from llm_proxy.models import ThinkingBlock
        from llm_proxy.providers.reasoning import detect_reasoning_field_in_response_body

        base_url = "https://mixed-wire-reuse.example.com/v1"
        adapter = OpenAICompatibleBase(api_key="test-key", base_url=base_url)

        # Learn `reasoning` from a wire-reuse body for model A (the same
        # learning the adapter's chat_completion WIRE_REUSE branch runs).
        body_a_response = {
            "model": "model-a-upstream",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi", "reasoning": "think"},
                    "finish_reason": "stop",
                }
            ],
        }
        try:
            adapter._record_reasoning_field_preference(
                detect_reasoning_field_in_response_body(body_a_response),
                model="model-a",
                response_model=body_a_response.get("model"),
            )

            body_a = adapter._build_request_body(
                InternalRequest(
                    model="model-a",
                    conversation=ConversationContext(
                        messages=[
                            Message(
                                role="assistant",
                                content=[ThinkingBlock(thinking="Previous thought")],
                            )
                        ]
                    ),
                )
            )
            assert body_a["messages"][0]["reasoning"] == "Previous thought"

            body_b = adapter._build_request_body(
                InternalRequest(
                    model="model-b",
                    conversation=ConversationContext(
                        messages=[
                            Message(
                                role="assistant",
                                content=[ThinkingBlock(thinking="Previous thought")],
                            )
                        ]
                    ),
                )
            )
            assert body_b["messages"][0]["reasoning_content"] == "Previous thought"
        finally:
            adapter._get_request_builder().clear_reasoning_field_preference(base_url)


class TestPromptCacheKeyGating:
    """prompt_cache_key is only forwarded to known-compatible upstreams
    (strict gateways reject unknown fields with HTTP 400)."""

    def _body_for_base_url(self, base_url: str) -> dict:
        adapter = OpenAICompatibleBase(
            api_key="test-key",
            base_url=base_url,
        )
        request = InternalRequest(
            model="gpt-4o",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Hello")])]
            ),
            params=GenerationParams(
                openai=OpenAISpecificParams(
                    prompt_cache_key="key-123",
                )
            ),
        )
        return adapter._build_request_body(request)

    def test_forwarded_to_api_openai_com(self):
        body = self._body_for_base_url("https://api.openai.com/v1")
        assert body["prompt_cache_key"] == "key-123"

    def test_forwarded_to_kimi_coding_path(self):
        body = self._body_for_base_url("https://api.kimi.com/coding")
        assert body["prompt_cache_key"] == "key-123"
        body = self._body_for_base_url("https://api.kimi.com/coding/v1")
        assert body["prompt_cache_key"] == "key-123"

    def test_dropped_for_unknown_gateway(self):
        body = self._body_for_base_url("https://api.example-gateway.com/v1")
        assert "prompt_cache_key" not in body

    def test_dropped_for_kimi_non_coding_path(self):
        body = self._body_for_base_url("https://api.kimi.com/v1")
        assert "prompt_cache_key" not in body
