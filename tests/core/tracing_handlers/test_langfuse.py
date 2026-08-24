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

    async def test_on_request_start_creates_generation(self, mock_client):
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
        assert isinstance(call_kwargs["input"], list)

        assert handler.get_trace_id() == "trace-123"
        assert handler.get_observation_id() == "obs-456"

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

    def test_build_usage_details_with_cache_and_audio(self):
        context = EventContext(
            request_id="req-1",
            trace_id="trace-1",
            model="gpt-4",
            prompt_tokens=5,
            completion_tokens=10,
            total_tokens=15,
            cache_read_input_tokens=2,
            audio_input_tokens=1,
            audio_output_tokens=1,
            reasoning_tokens=3,
        )
        details = build_usage_details(context)
        assert details["cache_read_input_tokens"] == 2
        assert details["audio_input_tokens"] == 1
        assert details["audio_output_tokens"] == 1
        assert details["reasoning_tokens"] == 3

    def test_build_cost_details(self):
        context = EventContext(
            request_id="req-1",
            trace_id="trace-1",
            model="gpt-4",
            cost_usd=0.002,
            provider_reported_cost=0.0015,
        )
        details = build_cost_details(context)
        assert details["total"] == 0.002
        assert details["provider_reported"] == 0.0015

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
