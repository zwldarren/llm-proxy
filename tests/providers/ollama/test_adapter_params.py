"""Tests for Ollama adapter parameter mapping."""

import logging
from unittest.mock import patch

import pytest

from llm_proxy.core.adapter import AdapterConfig
from llm_proxy.core.exceptions import ProviderError
from llm_proxy.models import (
    AnthropicSpecificParams,
    ConversationContext,
    GenerationParams,
    InternalEmbeddingRequest,
    InternalRequest,
    Message,
    OpenAISpecificParams,
    ResponseFormat,
    TextBlock,
    ThinkingConfig,
)
from llm_proxy.providers.ollama.adapter import OllamaAdapter


class TestBuildRequestBody:
    """Tests for _build_request_body parameter mapping."""

    @pytest.fixture
    def provider(self):
        return OllamaAdapter()

    def test_format_json_object(self, provider):
        """Test response_format json_object maps to format: json."""
        request = InternalRequest(
            model="test",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            params=GenerationParams(response_format=ResponseFormat(type="json_object")),
        )
        body = provider._build_request_body(request)
        assert body["format"] == "json"

    def test_format_json_schema(self, provider):
        """Test response_format json_schema maps to format schema."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        request = InternalRequest(
            model="test",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            params=GenerationParams(
                response_format=ResponseFormat(type="json_schema", json_schema=schema)
            ),
        )
        body = provider._build_request_body(request)
        assert body["format"] == schema

    def test_thinking_enabled(self, provider):
        """Test Anthropic thinking enabled maps to think: true."""
        request = InternalRequest(
            model="test",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            params=GenerationParams(thinking=ThinkingConfig(type="enabled", budget_tokens=None)),
        )
        body = provider._build_request_body(request)
        assert body["think"] is True

    def test_thinking_disabled(self, provider):
        """Test Anthropic thinking disabled maps to think: false."""
        request = InternalRequest(
            model="test",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            params=GenerationParams(thinking=ThinkingConfig(type="disabled", budget_tokens=None)),
        )
        body = provider._build_request_body(request)
        assert body["think"] is False

    def test_thinking_with_budget_low(self, provider):
        """Test thinking with low budget maps to think: low."""
        request = InternalRequest(
            model="test",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            params=GenerationParams(thinking=ThinkingConfig(type="enabled", budget_tokens=2000)),
        )
        body = provider._build_request_body(request)
        assert body["think"] == "low"

    def test_thinking_with_budget_high(self, provider):
        """Test thinking with high budget maps to think: high."""
        request = InternalRequest(
            model="test",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            params=GenerationParams(thinking=ThinkingConfig(type="enabled", budget_tokens=30000)),
        )
        body = provider._build_request_body(request)
        assert body["think"] == "high"

    def test_reasoning_effort_levels(self, provider):
        """Test OpenAI reasoning_effort maps to think levels."""
        cases = [
            ("none", False),
            ("minimal", "low"),
            ("low", "low"),
            ("medium", "medium"),
            ("high", "high"),
            ("xhigh", "max"),
        ]
        for effort, expected in cases:
            request = InternalRequest(
                model="test",
                conversation=ConversationContext(
                    messages=[Message(role="user", content=[TextBlock(text="hi")])]
                ),
                params=GenerationParams(openai=OpenAISpecificParams(reasoning_effort=effort)),
            )
            body = provider._build_request_body(request)
            assert body["think"] == expected, f"Failed for effort={effort}"

    def test_reasoning_effort_takes_precedence(self, provider):
        """Test reasoning_effort when unified thinking is not set."""
        request = InternalRequest(
            model="test",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            params=GenerationParams(
                openai=OpenAISpecificParams(reasoning_effort="low"),
            ),
        )
        body = provider._build_request_body(request)
        assert body["think"] == "low"

    def test_logprobs_enabled(self, provider):
        """Test logprobs parameter mapping."""
        request = InternalRequest(
            model="test",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            params=GenerationParams(openai=OpenAISpecificParams(logprobs=True, top_logprobs=5)),
        )
        body = provider._build_request_body(request)
        assert body["logprobs"] is True
        assert body["top_logprobs"] == 5

    def test_ollama_options_from_extra_body(self, provider):
        """Test Ollama options extracted from extra_body."""
        request = InternalRequest(
            model="test",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            extra={
                "num_ctx": 4096,
                "num_batch": 512,
                "seed": 42,
                "unknown_param": "value",
            },
        )
        body = provider._build_request_body(request)
        assert body["options"]["num_ctx"] == 4096
        assert body["options"]["num_batch"] == 512
        assert body["options"]["seed"] == 42
        # With default 'ignore' policy, unknown params are stripped
        assert "unknown_param" not in body

    def test_ollama_options_from_extra_body_passthrough(self, provider):
        """Test Ollama options with passthrough policy."""
        adapter = OllamaAdapter(unknown_fields_policy="passthrough")
        request = InternalRequest(
            model="test",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            extra={
                "num_ctx": 4096,
                "num_batch": 512,
                "seed": 42,
                "unknown_param": "value",
            },
        )
        body = adapter._build_request_body(request)
        assert body["options"]["num_ctx"] == 4096
        assert body["options"]["num_batch"] == 512
        assert body["options"]["seed"] == 42
        # With passthrough, unknown params go to top level
        assert body["unknown_param"] == "value"

    def test_runner_options_from_extra_body(self, provider):
        """Runner options (main_gpu/use_mmap/draft_num_predict) reach options."""
        request = InternalRequest(
            model="test",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            extra={
                "main_gpu": 0,
                "use_mmap": False,
                "draft_num_predict": 64,
            },
        )
        body = provider._build_request_body(request)
        assert body["options"]["main_gpu"] == 0
        assert body["options"]["use_mmap"] is False
        assert body["options"]["draft_num_predict"] == 64

    def test_stale_options_not_routed(self, provider):
        """Options removed from the Ollama request API are not routed.

        mirostat/tfs_z/epsilon_cutoff/eta_cutoff are Modelfile-only parameters
        since Ollama 0.8; sending them in options is ignored server-side, so
        they must not be advertised as native request options.
        """
        request = InternalRequest(
            model="test",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            extra={"mirostat": 2, "tfs_z": 1.0, "epsilon_cutoff": 0.1},
        )
        body = provider._build_request_body(request)
        assert "mirostat" not in body.get("options", {})
        assert "tfs_z" not in body.get("options", {})
        assert "epsilon_cutoff" not in body.get("options", {})
        # Default ignore policy strips them from the body entirely
        assert "mirostat" not in body
        assert "tfs_z" not in body

    def test_truncate_and_shift_survive_ignore_policy(self, provider):
        """Native top-level truncate/shift pass through the default policy."""
        request = InternalRequest(
            model="test",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            extra={"truncate": False, "shift": True},
        )
        body = provider._build_request_body(request)
        assert body["truncate"] is False
        assert body["shift"] is True

    def test_truncation_disabled_maps_to_truncate_false(self, provider):
        """OpenResponses truncation=disabled maps to Ollama truncate: false."""
        request = InternalRequest(
            model="test",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            extra={"truncation": "disabled"},
        )
        body = provider._build_request_body(request)
        assert body["truncate"] is False
        # The OpenResponses-only key itself never leaks into the body
        assert "truncation" not in body

    def test_truncation_auto_omits_truncate(self, provider):
        """truncation=auto (the default) leaves truncate unset."""
        request = InternalRequest(
            model="test",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            extra={"truncation": "auto"},
        )
        body = provider._build_request_body(request)
        assert "truncate" not in body

    def test_keep_alive_at_top_level(self, provider):
        """Test keep_alive goes to top level, not options."""
        adapter = OllamaAdapter(unknown_fields_policy="passthrough")
        request = InternalRequest(
            model="test",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            extra={"keep_alive": "5m"},
        )
        body = adapter._build_request_body(request)
        assert body["keep_alive"] == "5m"
        assert "keep_alive" not in body.get("options", {})

    def test_keep_alive_at_top_level_with_ignore_policy(self, provider):
        """Test keep_alive is preserved even with default ignore policy."""
        request = InternalRequest(
            model="test",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            extra={"keep_alive": "5m"},
        )
        body = provider._build_request_body(request)
        assert body["keep_alive"] == "5m"
        assert "keep_alive" not in body.get("options", {})

    def test_common_sampling_params_mapped_to_options(self, provider):
        """seed/penalties from standard protocol params must reach options.

        The OpenAI protocol parses these into params.common (they never land
        in request.extra), so the builder must map them explicitly — Ollama
        natively supports all three in options.
        """
        request = InternalRequest(
            model="test",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            params=GenerationParams(
                seed=42,
                frequency_penalty=0.5,
                presence_penalty=0.7,
            ),
        )
        body = provider._build_request_body(request)
        assert body["options"]["seed"] == 42
        assert body["options"]["frequency_penalty"] == 0.5
        assert body["options"]["presence_penalty"] == 0.7

    def test_anthropic_top_k_mapped_to_options(self, provider):
        """Anthropic-protocol top_k must map to Ollama options.top_k."""
        request = InternalRequest(
            model="test",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            params=GenerationParams(anthropic=AnthropicSpecificParams(top_k=40)),
        )
        body = provider._build_request_body(request)
        assert body["options"]["top_k"] == 40

    def test_extra_seed_overrides_params_seed(self, provider):
        """Native extra options win over params-derived options on conflict."""
        request = InternalRequest(
            model="test",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            params=GenerationParams(seed=1),
            extra={"seed": 2},
        )
        body = provider._build_request_body(request)
        assert body["options"]["seed"] == 2


class TestChatCompletionLogprobs:
    """Non-streaming chat_completion forwards logprobs to the parser."""

    @pytest.fixture
    def provider(self):
        return OllamaAdapter()

    def _mock_response(self, mock_response_cls):
        return mock_response_cls(
            json_data={
                "model": "llama3.2",
                "message": {"role": "assistant", "content": "hi"},
                "done": True,
                "done_reason": "stop",
                "logprobs": [{"token": "hi", "logprob": -0.5, "bytes": [104, 105]}],
            }
        )

    @pytest.mark.asyncio
    async def test_logprobs_requested_are_parsed(
        self, provider, mock_response_cls, make_mock_client
    ):
        """logprobs: true reaches the parser and populates response.logprobs."""
        request = InternalRequest(
            model="llama3.2",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            params=GenerationParams(openai=OpenAISpecificParams(logprobs=True)),
        )
        mock_client = make_mock_client(self._mock_response(mock_response_cls))

        with (
            patch.object(provider, "_get_client", return_value=mock_client),
            patch.object(provider, "_download_images_in_conversation"),
        ):
            result = await provider.chat_completion(request)

        assert result.logprobs is not None
        assert result.logprobs.content[0].token == "hi"

    @pytest.mark.asyncio
    async def test_logprobs_not_requested_are_skipped(
        self, provider, mock_response_cls, make_mock_client
    ):
        """Without logprobs the payload is not parsed."""
        request = InternalRequest(
            model="llama3.2",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
        )
        mock_client = make_mock_client(self._mock_response(mock_response_cls))

        with (
            patch.object(provider, "_get_client", return_value=mock_client),
            patch.object(provider, "_download_images_in_conversation"),
        ):
            result = await provider.chat_completion(request)

        assert result.logprobs is None


class TestEmbeddingBody:
    """Ollama /api/embed body building."""

    @pytest.fixture
    def provider(self):
        return OllamaAdapter()

    def test_encoding_format_not_sent(self, provider):
        """encoding_format is OpenAI-only; Ollama /api/embed ignores it."""
        request = InternalEmbeddingRequest(
            model="nomic-embed-text",
            input="hello",
            encoding_format="base64",
        )
        body = provider._build_outbound_body(request, request_type="embedding").json_body
        assert "encoding_format" not in body

    def test_dimensions_sent_when_requested(self, provider):
        """dimensions is a native /api/embed parameter."""
        request = InternalEmbeddingRequest(
            model="nomic-embed-text",
            input="hello",
            dimensions=256,
        )
        body = provider._build_outbound_body(request, request_type="embedding").json_body
        assert body["dimensions"] == 256

    def test_native_extra_params_survive_ignore_policy(self, provider):
        """keep_alive/truncate/options are native /api/embed params."""
        request = InternalEmbeddingRequest(
            model="nomic-embed-text",
            input="hello",
            extra={"keep_alive": "5m", "truncate": True, "options": {"num_ctx": 2048}},
        )
        body = provider._build_outbound_body(request, request_type="embedding").json_body
        assert body["keep_alive"] == "5m"
        assert body["truncate"] is True
        assert body["options"] == {"num_ctx": 2048}

    def test_unknown_extra_still_stripped(self, provider):
        """Non-native extra keys are still removed by the ignore policy."""
        request = InternalEmbeddingRequest(
            model="nomic-embed-text",
            input="hello",
            extra={"unknown_param": "value"},
        )
        body = provider._build_outbound_body(request, request_type="embedding").json_body
        assert "unknown_param" not in body


class TestFieldPolicyExemptions:
    """Keys the request builder explicitly handles are exempt from the
    unknown-fields policy, so ``unknown_fields_policy: error`` does not
    reject them."""

    def _provider_with_error_policy(self):
        return OllamaAdapter(
            config=AdapterConfig(
                provider_name="ollama",
                extra={"unknown_fields_policy": "error"},
            )
        )

    def test_responses_only_keys_exempt_from_error_policy(self):
        """store/metadata are deliberately dropped by the builder, not errors."""

        provider = self._provider_with_error_policy()
        request = InternalRequest(
            model="test",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            extra={"store": True, "metadata": {"k": "v"}},
        )
        body = provider._build_request_body(request)
        # Builder drops them; the error policy must not reject them either.
        assert "store" not in body
        assert "metadata" not in body

    def test_native_top_level_keys_exempt_from_error_policy(self):
        """keep_alive/truncate/shift survive the error policy."""
        provider = self._provider_with_error_policy()
        request = InternalRequest(
            model="test",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            extra={"keep_alive": "5m", "truncate": False, "shift": True},
        )
        body = provider._build_request_body(request)
        assert body["keep_alive"] == "5m"
        assert body["truncate"] is False
        assert body["shift"] is True

    def test_truly_unknown_key_still_rejected(self):
        """A genuinely unknown key still raises under the error policy."""
        provider = self._provider_with_error_policy()
        request = InternalRequest(
            model="test",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            extra={"definitely_not_an_ollama_param": 1},
        )
        with pytest.raises(ProviderError):
            provider._build_request_body(request)


class TestToolChoiceWarning:
    """Ollama has no tool_choice; the drop must be surfaced, not silent."""

    @pytest.fixture
    def provider(self):
        return OllamaAdapter()

    def test_tool_choice_logs_warning(self, provider, caplog):
        request = InternalRequest(
            model="test",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            tool_choice="none",
        )
        with caplog.at_level(logging.WARNING):
            provider._build_request_body(request)
        assert any("tool_choice" in record.message for record in caplog.records)

    def test_no_warning_without_tool_choice(self, provider, caplog):
        request = InternalRequest(
            model="test",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
        )
        with caplog.at_level(logging.WARNING):
            provider._build_request_body(request)
        assert not any("tool_choice" in record.message for record in caplog.records)


class TestConvertLogprobs:
    """Tests for convert_logprobs method in OllamaProviderSerializer."""

    @pytest.fixture
    def serializer(self):
        from llm_proxy.serialization.ollama.serializer import OllamaProviderSerializer

        return OllamaProviderSerializer()

    def test_empty_logprobs(self, serializer):
        """Test empty input returns None."""
        assert serializer.convert_logprobs(None) is None
        assert serializer.convert_logprobs([]) is None

    def test_simple_logprobs(self, serializer):
        """Test simple logprobs conversion."""
        ollama_logprobs = [
            {"token": "Hello", "logprob": -0.5, "bytes": [72, 101, 108, 108, 111]},
        ]
        result = serializer.convert_logprobs(ollama_logprobs)
        assert result is not None
        assert "content" in result
        assert len(result["content"]) == 1
        assert result["content"][0]["token"] == "Hello"
        assert result["content"][0]["logprob"] == -0.5
        assert result["content"][0]["bytes"] == [72, 101, 108, 108, 111]

    def test_logprobs_with_top_logprobs(self, serializer):
        """Test logprobs with top_logprobs."""
        ollama_logprobs = [
            {
                "token": "Hello",
                "logprob": -0.5,
                "top_logprobs": [
                    {"token": "Hello", "logprob": -0.5},
                    {"token": "Hi", "logprob": -1.2},
                ],
            },
        ]
        result = serializer.convert_logprobs(ollama_logprobs)
        assert result is not None
        assert len(result["content"][0]["top_logprobs"]) == 2
        assert result["content"][0]["top_logprobs"][0]["token"] == "Hello"
        assert result["content"][0]["top_logprobs"][1]["token"] == "Hi"

    def test_logprobs_missing_optional_fields(self, serializer):
        """Test logprobs handles missing optional fields."""
        ollama_logprobs = [
            {"token": "test", "logprob": -0.1},
        ]
        result = serializer.convert_logprobs(ollama_logprobs)
        assert result is not None
        assert "bytes" not in result["content"][0]
        assert "top_logprobs" not in result["content"][0]
