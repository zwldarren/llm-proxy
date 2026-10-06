"""Tests for the Langfuse tracing handler and its SDK data builders."""

from unittest.mock import MagicMock, patch

import pytest

from llm_proxy.core.request_type import RequestType
from llm_proxy.observability.event_context import EventContext
from llm_proxy.observability.tracing.handlers.providers.langfuse import LangfuseTracingHandler
from llm_proxy.observability.tracing.handlers.providers.langfuse.attributes import (
    build_cost_details,
    build_metadata,
    build_request_input_data,
    build_response_output_data,
    build_usage_details,
    extract_tool_uses,
)


@pytest.fixture
def mock_langfuse_cls():
    """Return a patched Langfuse class suitable for handler.create_handler."""
    with patch(
        "llm_proxy.observability.tracing.handlers.providers.langfuse.handler.Langfuse"
    ) as cls:
        yield cls


@pytest.fixture
def mock_propagate():
    """Patch ``propagate_attributes`` so trace-level attributes can be asserted.

    The SDK's real function attaches OpenTelemetry baggage and needs an active
    span; the handler passes user/session/trace-name through it instead of the
    (silently ignored) ``observation.update(trace_user_id=...)`` kwargs.
    """
    with patch(
        "llm_proxy.observability.tracing.handlers.providers.langfuse.handler.propagate_attributes"
    ) as propagate:
        yield propagate


@pytest.fixture
def mock_client(mock_langfuse_cls):
    """Return a mock Langfuse client instance."""
    client = MagicMock()
    mock_langfuse_cls.return_value = client
    return client


def _make_generation():
    """Create a fake SDK generation observation."""
    gen = MagicMock()
    gen.trace_id = "trace-123"
    gen.id = "obs-456"
    return gen


def _make_request_with_conversation():
    """Build a mock InternalRequest with a simple user message."""
    msg = MagicMock()
    msg.role = "user"
    msg.content = [{"type": "text", "text": "hello"}]
    msg.name = None

    conversation = MagicMock()
    conversation.system_messages = []
    conversation.messages = [msg]

    request = MagicMock()
    request.conversation = conversation
    request.model = "gpt-4"
    request.request_type = RequestType.CHAT
    request.stream = False
    request.tools = None
    request.tool_choice = None
    return request


def _make_streaming_request():
    """Build a mock InternalRequest with stream=True."""
    request = _make_request_with_conversation()
    request.stream = True
    return request


class TestLangfuseTracingHandler:
    def test_create_handler_eu_default(self, mock_client, mock_langfuse_cls):
        settings = {
            "public_key": "pk-test",
            "secret_key": "sk-test",
        }
        handler = LangfuseTracingHandler.create_handler(settings)
        assert handler._base_url == "https://cloud.langfuse.com"
        mock_langfuse_cls.assert_called_once()
        call_kwargs = mock_langfuse_cls.call_args.kwargs
        assert call_kwargs["public_key"] == "pk-test"
        assert call_kwargs["secret_key"] == "sk-test"
        assert call_kwargs["base_url"] == "https://cloud.langfuse.com"

    def test_create_handler_region_is_ignored(self, mock_client, mock_langfuse_cls):
        settings = {
            "public_key": "pk-test",
            "secret_key": "sk-test",
            "region": "us",
        }
        handler = LangfuseTracingHandler.create_handler(settings)
        assert handler._base_url == "https://cloud.langfuse.com"
        call_kwargs = mock_langfuse_cls.call_args.kwargs
        assert call_kwargs["base_url"] == "https://cloud.langfuse.com"

    def test_create_handler_self_hosted(self, mock_client, mock_langfuse_cls):
        settings = {
            "public_key": "pk-test",
            "secret_key": "sk-test",
            "base_url": "http://localhost:3000",
        }
        handler = LangfuseTracingHandler.create_handler(settings)
        assert handler._base_url == "http://localhost:3000"
        assert mock_langfuse_cls.call_args.kwargs["base_url"] == "http://localhost:3000"

    def test_create_handler_passes_timeout(self, mock_client, mock_langfuse_cls):
        settings = {
            "public_key": "pk-test",
            "secret_key": "sk-test",
            "timeout": 30,
        }
        LangfuseTracingHandler.create_handler(settings)
        assert mock_langfuse_cls.call_args.kwargs["timeout"] == 30

    def test_create_handler_scopes_sample_rate_to_a_project_provider(
        self, mock_client, mock_langfuse_cls
    ):
        """A sub-1.0 rate must not land on the process-wide sampler."""
        settings = {"public_key": "pk-sampled", "secret_key": "sk-test", "sample_rate": 0.5}
        LangfuseTracingHandler.create_handler(settings)
        first_provider = mock_langfuse_cls.call_args.kwargs["tracer_provider"]
        assert first_provider is not None
        assert mock_langfuse_cls.call_args.kwargs["sample_rate"] == 0.5

        # A rebuild for the same project reuses the provider (no leaked thread).
        LangfuseTracingHandler.create_handler(settings)
        assert mock_langfuse_cls.call_args.kwargs["tracer_provider"] is first_provider

    def test_create_handler_keeps_default_provider_at_full_rate(
        self, mock_client, mock_langfuse_cls
    ):
        LangfuseTracingHandler.create_handler(
            {"public_key": "pk-full", "secret_key": "sk-test", "sample_rate": 1.0}
        )
        assert "tracer_provider" not in mock_langfuse_cls.call_args.kwargs

    def test_create_handler_rejects_invalid_sample_rate(self, mock_langfuse_cls):
        with pytest.raises(ValueError, match="sample_rate"):
            LangfuseTracingHandler.create_handler(
                {"public_key": "pk", "secret_key": "sk", "sample_rate": 1.5}
            )

    def test_missing_public_key_raises(self):
        settings = {"secret_key": "sk-test"}
        with pytest.raises(ValueError, match="public_key"):
            LangfuseTracingHandler.create_handler(settings)

    def test_missing_secret_key_raises(self):
        settings = {"public_key": "pk-test"}
        with pytest.raises(ValueError, match="secret_key"):
            LangfuseTracingHandler.create_handler(settings)

    def test_empty_public_key_raises(self):
        settings = {"public_key": "", "secret_key": "sk-test"}
        with pytest.raises(ValueError, match="public_key"):
            LangfuseTracingHandler.create_handler(settings)

    def test_empty_secret_key_raises(self):
        settings = {"public_key": "pk-test", "secret_key": ""}
        with pytest.raises(ValueError, match="secret_key"):
            LangfuseTracingHandler.create_handler(settings)

    def test_validate_config_true_with_valid_keys(self):
        settings = {"public_key": "pk-test", "secret_key": "sk-test"}
        assert LangfuseTracingHandler.validate_config(settings) is True

    def test_validate_config_false_when_public_key_missing(self):
        settings = {"secret_key": "sk-test"}
        assert LangfuseTracingHandler.validate_config(settings) is False

    def test_validate_config_false_when_secret_key_missing(self):
        settings = {"public_key": "pk-test"}
        assert LangfuseTracingHandler.validate_config(settings) is False

    def test_validate_config_false_when_keys_empty(self):
        assert LangfuseTracingHandler.validate_config({"public_key": "", "secret_key": ""}) is False
        assert (
            LangfuseTracingHandler.validate_config({"public_key": "pk", "secret_key": ""}) is False
        )
        assert (
            LangfuseTracingHandler.validate_config({"public_key": "", "secret_key": "sk"}) is False
        )

    def test_handler_provider_metadata(self):
        assert LangfuseTracingHandler.provider_name == "langfuse"
        assert "public_key" in LangfuseTracingHandler.required_settings
        assert "secret_key" in LangfuseTracingHandler.required_settings
        assert set(LangfuseTracingHandler.optional_settings) == {
            "base_url",
            "timeout",
            "sample_rate",
            "version",
        }
        field_names = {f["name"] for f in LangfuseTracingHandler.field_metadata}
        assert "base_url" in field_names
        assert "sample_rate" in field_names
        assert "version" in field_names
        assert "region" not in field_names

    def test_name_from_settings(self, mock_client, mock_langfuse_cls):
        settings = {
            "public_key": "pk-test",
            "secret_key": "sk-test",
            "name": "my-langfuse",
        }
        handler = LangfuseTracingHandler.create_handler(settings)
        assert handler.name == "my-langfuse"

    async def test_on_request_start_creates_generation(self, mock_client, mock_propagate):
        settings = {"public_key": "pk-test", "secret_key": "sk-test"}
        handler = LangfuseTracingHandler.create_handler(settings)

        gen = _make_generation()
        mock_client.start_as_current_observation.return_value.__enter__.return_value = gen

        request = _make_request_with_conversation()
        context = EventContext(
            request_id="req-1",
            trace_id="trace-1",
            model="gpt-4",
            user_id="user-42",
            session_id="session-42",
            metadata={"endpoint": "/v1/chat/completions"},
        )

        await handler.on_request_start(request, context)

        mock_client.start_as_current_observation.assert_called_once()
        call_kwargs = mock_client.start_as_current_observation.call_args.kwargs
        assert call_kwargs["as_type"] == "generation"
        assert call_kwargs["name"] == "chat completions"
        assert call_kwargs["model"] == "gpt-4"
        assert call_kwargs["end_on_exit"] is False
        # "trace-1" is not a 32-char hex Langfuse trace id, so the generated id
        # is left to the SDK rather than handed to it (it would raise).
        assert call_kwargs["trace_context"] is None
        assert isinstance(call_kwargs["input"], list)

        # Trace-level attributes must go through propagate_attributes: SDK v4
        # ignores them as observation.update() kwargs.
        propagate_kwargs = mock_propagate.call_args.kwargs
        assert propagate_kwargs["trace_name"] == "chat completions"
        assert propagate_kwargs["user_id"] == "user-42"
        assert propagate_kwargs["session_id"] == "session-42"

        assert handler.get_trace_id() == "trace-123"
        assert handler.get_observation_id() == "obs-456"

    async def test_on_request_start_links_caller_trace_id(self, mock_client, mock_propagate):
        """A caller-supplied x-langfuse-trace-id nests the observation in its trace."""
        settings = {"public_key": "pk-test", "secret_key": "sk-test"}
        handler = LangfuseTracingHandler.create_handler(settings)

        gen = _make_generation()
        mock_client.start_as_current_observation.return_value.__enter__.return_value = gen

        request = _make_request_with_conversation()
        context = EventContext(
            request_id="req-1",
            trace_id="0123456789abcdef0123456789abcdef",
            model="gpt-4",
            metadata={"endpoint": "/v1/chat/completions"},
        )

        await handler.on_request_start(request, context)

        call_kwargs = mock_client.start_as_current_observation.call_args.kwargs
        assert call_kwargs["trace_context"] == {"trace_id": "0123456789abcdef0123456789abcdef"}

    async def test_on_stream_end_still_closes_a_released_generation(
        self, mock_client, mock_propagate
    ):
        """A config change mid-request must not lose the in-flight trace.

        Registry invalidation calls ``release()`` (flush) rather than
        ``shutdown()``; the handler dropped from the registry must still end the
        generation it created, otherwise the span is never exported.
        """
        settings = {"public_key": "pk-test", "secret_key": "sk-test"}
        handler = LangfuseTracingHandler.create_handler(settings)

        gen = _make_generation()
        mock_client.start_as_current_observation.return_value.__enter__.return_value = gen

        request = _make_request_with_conversation()
        context = EventContext(
            request_id="req-1",
            trace_id="trace-1",
            model="gpt-4",
            metadata={"endpoint": "/v1/chat/completions"},
        )

        await handler.on_request_start(request, context)
        await handler.release()
        await handler.on_stream_end(request, context)

        gen.end.assert_called_once()
        mock_client.flush.assert_called_once()
        mock_client.shutdown.assert_not_called()

    async def test_on_request_end_updates_generation(self, mock_client):
        settings = {"public_key": "pk-test", "secret_key": "sk-test"}
        handler = LangfuseTracingHandler.create_handler(settings)

        gen = _make_generation()
        mock_client.start_as_current_observation.return_value.__enter__.return_value = gen

        request = _make_request_with_conversation()
        response = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "world"
        response.output = [text_block]
        response.finish_reason = "stop"
        response.model = "gpt-4"

        context = EventContext(
            request_id="req-1",
            trace_id="trace-1",
            model="gpt-4",
            prompt_tokens=5,
            completion_tokens=10,
            total_tokens=15,
            cost_usd=0.002,
            metadata={"endpoint": "/v1/chat/completions"},
        )

        await handler.on_request_start(request, context)
        await handler.on_request_end(request, response, context)

        update_calls = gen.update.call_args_list
        last_update_kwargs = update_calls[-1].kwargs
        assert last_update_kwargs["output"]["role"] == "assistant"
        assert last_update_kwargs["output"]["content"] == "world"
        assert last_update_kwargs["model"] == "gpt-4"
        assert last_update_kwargs["usage_details"]["input"] == 5
        assert last_update_kwargs["usage_details"]["total"] == 15
        assert last_update_kwargs["cost_details"]["total"] == 0.002
        gen.end.assert_called_once()

    async def test_on_request_end_records_tool_observations(self, mock_client):
        settings = {"public_key": "pk-test", "secret_key": "sk-test"}
        handler = LangfuseTracingHandler.create_handler(settings)

        gen = _make_generation()
        tool_obs = MagicMock()
        gen.start_observation.return_value = tool_obs
        mock_client.start_as_current_observation.return_value.__enter__.return_value = gen

        request = _make_request_with_conversation()
        response = MagicMock()
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "call_1"
        tool_block.name = "get_weather"
        tool_block.input = {"location": "NYC"}
        response.output = [tool_block]
        response.finish_reason = "tool_calls"
        response.model = "gpt-4"

        context = EventContext(
            request_id="req-1",
            trace_id="trace-1",
            model="gpt-4",
            prompt_tokens=5,
            completion_tokens=10,
            total_tokens=15,
            metadata={"endpoint": "/v1/chat/completions"},
        )

        await handler.on_request_start(request, context)
        await handler.on_request_end(request, response, context)

        gen.start_observation.assert_called_once()
        call_kwargs = gen.start_observation.call_args.kwargs
        assert call_kwargs["as_type"] == "tool"
        assert call_kwargs["name"] == "get_weather"
        assert call_kwargs["input"] == {"location": "NYC"}
        assert call_kwargs["metadata"]["tool_call_id"] == "call_1"
        assert call_kwargs["metadata"]["result_observed"] is False
        tool_obs.end.assert_called_once()

    async def test_on_error_marks_generation_error(self, mock_client):
        settings = {"public_key": "pk-test", "secret_key": "sk-test"}
        handler = LangfuseTracingHandler.create_handler(settings)

        gen = _make_generation()
        mock_client.start_as_current_observation.return_value.__enter__.return_value = gen

        request = _make_request_with_conversation()
        context = EventContext(
            request_id="req-1",
            trace_id="trace-1",
            model="gpt-4",
            user_id="user-42",
            metadata={"endpoint": "/v1/chat/completions"},
        )

        await handler.on_request_start(request, context)
        await handler.on_error(request, ValueError("boom"), context)

        update_calls = gen.update.call_args_list
        last_update_kwargs = update_calls[-1].kwargs
        assert last_update_kwargs["level"] == "ERROR"
        assert last_update_kwargs["status_message"] == "boom"
        gen.end.assert_called_once()

    async def test_shutdown_flushes_client(self, mock_client):
        settings = {"public_key": "pk-test", "secret_key": "sk-test"}
        handler = LangfuseTracingHandler.create_handler(settings)

        await handler.shutdown()

        mock_client.flush.assert_called_once()
        mock_client.shutdown.assert_called_once()
        assert handler.enabled is False


class TestLangfuseSDKDataBuilders:
    def test_build_request_input_data_returns_messages(self):
        msg = MagicMock()
        msg.role = "user"
        msg.content = [{"type": "text", "text": "hello"}]
        msg.name = None

        conversation = MagicMock()
        conversation.system_messages = []
        conversation.messages = [msg]

        request = MagicMock()
        request.conversation = conversation
        request.tools = None

        data = build_request_input_data(request)
        assert isinstance(data, list)
        assert data[0]["role"] == "user"
        assert "hello" in str(data)

    def test_build_request_input_data_returns_none_without_conversation(self):
        request = MagicMock()
        request.conversation = None
        assert build_request_input_data(request) is None

    def test_build_response_output_data(self):
        response = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "world"
        response.output = [text_block]
        response.finish_reason = "stop"
        response.model = "gpt-4"

        data = build_response_output_data(response)
        assert data is not None
        assert data["role"] == "assistant"
        assert data["content"] == "world"
        assert data["finish_reason"] == "stop"
        assert data["model"] == "gpt-4"

    def test_build_usage_details(self):
        context = EventContext(
            request_id="req-1",
            trace_id="trace-1",
            model="gpt-4",
            prompt_tokens=5,
            completion_tokens=10,
            total_tokens=15,
        )
        details = build_usage_details(context)
        assert details == {"input": 5, "output": 10, "total": 15}

    def test_build_usage_details_are_exclusive_buckets(self):
        """Cache/audio/reasoning details are subtracted from input/output.

        Langfuse treats each usage key as a mutually exclusive bucket; passing an
        inclusive input plus its detail keys double-counts (and double-prices)
        the tokens.
        """
        context = EventContext(
            request_id="req-1",
            trace_id="trace-1",
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cache_read_input_tokens=30,
            cache_creation_input_tokens=10,
            audio_input_tokens=5,
            audio_output_tokens=4,
            reasoning_tokens=20,
        )
        details = build_usage_details(context)
        assert details["input"] == 55  # 100 - 30 - 10 - 5
        assert details["output"] == 26  # 50 - 4 - 20
        assert details["total"] == 150
        assert details["cache_read_input_tokens"] == 30
        assert details["cache_creation_input_tokens"] == 10
        assert details["input_audio_tokens"] == 5
        assert details["output_audio_tokens"] == 4
        assert details["output_reasoning_tokens"] == 20

    def test_build_usage_details_reads_canonical_cache_fields(self):
        """Dialect aliases are normalized at capture time; tracing reads canonical
        ``cache_read_input_tokens`` only (CONTEXT.md "Canonical usage record")."""
        context = EventContext(
            request_id="req-1",
            trace_id="trace-1",
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=10,
            total_tokens=110,
            cache_read_input_tokens=25,
        )
        details = build_usage_details(context)
        assert details["cache_read_input_tokens"] == 25
        assert details["input"] == 75

    def test_build_usage_details_none_without_token_data(self):
        context = EventContext(request_id="req-1", trace_id="trace-1", model="gpt-4")
        assert build_usage_details(context) is None

    def test_build_cost_details(self):
        context = EventContext(
            request_id="req-1",
            trace_id="trace-1",
            model="gpt-4",
            cost_usd=0.002,
            provider_reported_cost=0.0015,
        )
        details = build_cost_details(context)
        # Only the proxy-computed total is a real cost bucket; the
        # provider-reported cost rides in metadata (see build_metadata).
        assert details == {"total": 0.002}

    def test_build_cost_details_none_without_cost(self):
        context = EventContext(request_id="req-1", trace_id="trace-1", model="gpt-4")
        assert build_cost_details(context) is None

    def test_build_metadata(self):
        context = EventContext(
            request_id="req-1",
            trace_id="trace-1",
            model="gpt-4",
            provider="openai",
            user_id="user-1",
            session_id="session-1",
            request_type=RequestType.CHAT,
            metadata={"endpoint": "/v1/chat/completions"},
        )
        metadata = build_metadata(context)
        assert metadata["request_id"] == "req-1"
        assert metadata["provider"] == "openai"
        assert metadata["endpoint"] == "/v1/chat/completions"
        assert metadata["endpoint_name"] == "chat completions"

    def test_build_response_output_data_with_tool_call(self):
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "call_1"
        tool_block.name = "get_weather"
        tool_block.input = {"location": "NYC"}

        response = MagicMock()
        response.output = [tool_block]
        response.finish_reason = "tool_calls"
        response.model = "gpt-4"

        result = build_response_output_data(response)
        assert result["role"] == "assistant"
        assert result["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_build_request_input_data_includes_tools(self):
        msg = MagicMock()
        msg.role = "user"
        msg.content = [{"type": "text", "text": "hello"}]
        msg.name = None

        conversation = MagicMock()
        conversation.system_messages = []
        conversation.messages = [msg]

        tool = MagicMock()
        tool.name = "get_weather"
        tool.description = "Get the weather"
        tool.parameters = {"type": "object", "properties": {}}

        request = MagicMock()
        request.conversation = conversation
        request.tools = [tool]

        data = build_request_input_data(request)
        assert isinstance(data, dict)
        assert isinstance(data["tools"], list)
        assert data["tools"][0]["type"] == "function"
        assert data["tools"][0]["function"]["name"] == "get_weather"
        assert data["tools"][0]["function"]["description"] == "Get the weather"
        assert isinstance(data["messages"], list)
        assert data["messages"][0]["role"] == "user"

    def test_extract_tool_uses_returns_tool_use_blocks(self):
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "hi"

        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "call_1"
        tool_block.name = "get_weather"
        tool_block.input = {"location": "NYC"}

        tool_uses = extract_tool_uses([text_block, tool_block])
        assert len(tool_uses) == 1
        assert tool_uses[0]["id"] == "call_1"
        assert tool_uses[0]["name"] == "get_weather"
        assert tool_uses[0]["input"] == {"location": "NYC"}

    def test_extract_tool_uses_empty_for_no_output(self):
        assert extract_tool_uses(None) == []
        assert extract_tool_uses([]) == []
