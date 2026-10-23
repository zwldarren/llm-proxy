"""Tests for the Langfuse tracing handler and its SDK data builders."""

from unittest.mock import MagicMock, patch

import pytest

from llm_proxy.core.request_type import RequestType
from llm_proxy.models import (
    ImageData,
    ImageEditSource,
    ImageSize,
    InternalDecisionRequest,
    InternalDecisionResponse,
    InternalImageEditRequest,
    InternalImageRequest,
    InternalImageResponse,
    InternalRerankRequest,
    InternalRerankResponse,
    InternalSpeechRequest,
    InternalSpeechResponse,
    InternalSystemOneRequest,
    InternalSystemOneResponse,
    InternalTranscriptionRequest,
    InternalTranscriptionResponse,
    InternalTranslationRequest,
    InternalTranslationResponse,
    RerankDocument,
    RerankResult,
)
from llm_proxy.models.types import (
    CompletionTokensDetails,
    PromptTokensDetails,
    Usage,
)
from llm_proxy.observability.event_context import EventContext
from llm_proxy.observability.tracing.handlers.providers.langfuse import LangfuseTracingHandler
from llm_proxy.observability.tracing.handlers.providers.langfuse.attributes import (
    _endpoint_display_name,
    build_audio_transcript_input_data,
    build_audio_transcript_output_data,
    build_cost_details,
    build_decisions_input_data,
    build_decisions_output_data,
    build_embedding_input_data,
    build_embedding_output_data,
    build_embedding_usage_details,
    build_image_input_data,
    build_image_output_data,
    build_metadata,
    build_request_input_data,
    build_rerank_input_data,
    build_rerank_output_data,
    build_response_output_data,
    build_speech_input_data,
    build_speech_output_data,
    build_systemone_input_data,
    build_systemone_output_data,
    build_usage_details,
    extract_tool_uses,
    observation_profile,
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


def _make_embedding_request(texts: str | list[str] | None = None):
    """Build a mock InternalEmbeddingRequest: no conversation, input is the text."""
    request = MagicMock()
    request.conversation = None
    request.model = "gemini-embedding-001"
    request.request_type = RequestType.EMBEDDING
    request.stream = False
    request.tools = None
    # ``None`` keeps ``build_model_parameters`` from reading MagicMock fields.
    request.params = None
    request.input = ["hello world"] if texts is None else texts
    return request


def _make_embedding_response(count: int = 2, dimensions: int = 768):
    """Build a mock InternalEmbeddingResponse carrying ``count`` vectors."""
    response = MagicMock()
    response.output = []
    response.model = "gemini-embedding-001"
    data = []
    for index in range(count):
        item = MagicMock()
        item.index = index
        item.embedding = [float(index * 1000 + offset) for offset in range(dimensions)]
        data.append(item)
    response.data = data
    return response


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
            usage=Usage(
                input_tokens=5,
                output_tokens=10,
                total_tokens=15,
            ),
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
            usage=Usage(
                input_tokens=5,
                output_tokens=10,
                total_tokens=15,
            ),
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


class TestLangfuseEmbeddingObservations:
    """Embedding requests are their own observation type, with content.

    A Langfuse ``embedding`` observation carries the embedded text as input and
    the response shape as output; without that the trace of a retrieval
    pipeline shows a generation with no content at all.
    """

    async def test_embedding_request_opens_embedding_observation(self, mock_client, mock_propagate):
        settings = {"public_key": "pk-test", "secret_key": "sk-test"}
        handler = LangfuseTracingHandler.create_handler(settings)

        gen = _make_generation()
        mock_client.start_as_current_observation.return_value.__enter__.return_value = gen

        request = _make_embedding_request(["hello world", "second text"])
        context = EventContext(
            request_id="req-emb-1",
            trace_id="trace-1",
            model="gemini-embedding-001",
            request_type=RequestType.EMBEDDING,
            metadata={"endpoint": "/v1/embeddings"},
        )

        await handler.on_request_start(request, context)

        call_kwargs = mock_client.start_as_current_observation.call_args.kwargs
        assert call_kwargs["as_type"] == "embedding"
        assert call_kwargs["name"] == "embeddings"
        assert call_kwargs["model"] == "gemini-embedding-001"
        assert call_kwargs["input"] == ["hello world", "second text"]
        assert call_kwargs["end_on_exit"] is False

    async def test_chat_request_keeps_generation_type(self, mock_client, mock_propagate):
        settings = {"public_key": "pk-test", "secret_key": "sk-test"}
        handler = LangfuseTracingHandler.create_handler(settings)

        gen = _make_generation()
        mock_client.start_as_current_observation.return_value.__enter__.return_value = gen

        request = _make_request_with_conversation()
        context = EventContext(
            request_id="req-1",
            trace_id="trace-1",
            model="gpt-4",
            request_type=RequestType.CHAT,
            metadata={"endpoint": "/v1/chat/completions"},
        )

        await handler.on_request_start(request, context)

        assert mock_client.start_as_current_observation.call_args.kwargs["as_type"] == "generation"

    async def test_request_type_falls_back_to_the_request(self, mock_client, mock_propagate):
        """A bare event context (defaults to CHAT) must not hide an embedding."""
        settings = {"public_key": "pk-test", "secret_key": "sk-test"}
        handler = LangfuseTracingHandler.create_handler(settings)

        gen = _make_generation()
        mock_client.start_as_current_observation.return_value.__enter__.return_value = gen

        request = _make_embedding_request("single text")
        context = EventContext(
            request_id="req-emb-2",
            trace_id="trace-1",
            model="gemini-embedding-001",
            metadata={"endpoint": "/v1/embeddings"},
        )

        await handler.on_request_start(request, context)

        call_kwargs = mock_client.start_as_current_observation.call_args.kwargs
        assert call_kwargs["as_type"] == "embedding"
        assert call_kwargs["input"] == "single text"

    async def test_on_request_end_records_embedding_output(self, mock_client):
        settings = {"public_key": "pk-test", "secret_key": "sk-test"}
        handler = LangfuseTracingHandler.create_handler(settings)

        gen = _make_generation()
        mock_client.start_as_current_observation.return_value.__enter__.return_value = gen

        request = _make_embedding_request()
        response = _make_embedding_response(count=12, dimensions=768)
        context = EventContext(
            request_id="req-emb-3",
            trace_id="trace-1",
            model="gemini-embedding-001",
            request_type=RequestType.EMBEDDING,
            # Gemini embeddings bill input tokens only: no output bucket.
            usage=Usage(input_tokens=9),
            metadata={"endpoint": "/v1/embeddings"},
        )

        await handler.on_request_start(request, context)
        await handler.on_request_end(request, response, context)

        last_update_kwargs = gen.update.call_args_list[-1].kwargs
        output = last_update_kwargs["output"]
        assert output["count"] == 12
        assert output["dimensions"] == 768
        assert output["model"] == "gemini-embedding-001"
        assert output["preview_truncated"] is True
        assert len(output["embeddings"]) == 5
        assert output["embeddings"][0]["preview"] == [float(i) for i in range(8)]
        assert last_update_kwargs["usage_details"] == {"input": 9, "total": 9}
        # Embeddings never produce tool observations.
        gen.start_observation.assert_not_called()
        gen.end.assert_called_once()


class TestLangfuseObservationProfiles:
    """Every request type the pipeline serves has a profile with content."""

    def test_every_request_type_has_a_profile(self):
        for request_type in RequestType:
            profile = observation_profile(request_type)
            assert profile.build_input is not None
            assert profile.build_output is not None

    def test_only_embedding_uses_its_own_observation_type(self):
        assert observation_profile(RequestType.EMBEDDING).as_type == "embedding"
        for request_type in RequestType:
            if request_type is not RequestType.EMBEDDING:
                # Langfuse prices usage/cost on generation and embedding only.
                assert observation_profile(request_type).as_type == "generation"

    def test_unknown_or_missing_type_falls_back_to_a_metadata_only_profile(self):
        default = observation_profile(None)
        assert observation_profile("not-a-request-type") is default
        assert observation_profile(object()) is default
        assert default.as_type == "generation"
        assert default.build_input(MagicMock()) is None
        assert default.build_output(MagicMock()) is None

    def test_wire_strings_resolve_like_enum_members(self):
        assert observation_profile("rerank") is observation_profile(RequestType.RERANK)

    def test_endpoint_display_name_ignores_version_and_alias_segments(self):
        assert _endpoint_display_name("/v1/rerank") == "rerank"
        assert _endpoint_display_name("/v2/rerank") == "rerank"
        assert _endpoint_display_name("/v1/v1/chat/completions") == "chat completions"
        assert _endpoint_display_name("/v1/v1/decisions") == "decisions"
        assert _endpoint_display_name("/v1/audio/transcriptions") == "audio transcriptions"


class TestLangfuseNonConversationPayloads:
    """Rerank, images, audio, systemone and decisions carry their own content."""

    def test_rerank_input_carries_query_and_documents(self):
        request = InternalRerankRequest(
            model="rerank-v3.5",
            query="what is a proxy?",
            documents=["a", RerankDocument(text="b", image="data:image/png;base64,AAAA")],
            top_n=1,
        )
        data = build_rerank_input_data(request)
        assert data["query"] == "what is a proxy?"
        assert data["documents"][0] == "a"
        # A structured document keeps its keys; an inline image becomes a marker.
        assert data["documents"][1]["text"] == "b"
        assert data["documents"][1]["image"] == "<data-uri 26 chars>"
        assert data["top_n"] == 1

    def test_rerank_input_caps_and_truncates_documents(self):
        request = InternalRerankRequest(model="rerank-v3.5", query="q", documents=["x" * 5000] * 50)
        data = build_rerank_input_data(request)
        assert len(data["documents"]) == 20
        assert data["document_count"] == 50
        assert data["documents_truncated"] is True
        assert data["documents"][0].startswith("x" * 100)
        assert "truncated" in data["documents"][0]

    def test_rerank_output_reports_the_ranking(self):
        response = InternalRerankResponse(
            model="rerank-v3.5",
            results=[RerankResult(index=1, relevance_score=0.93)],
            id="resp_1",
        )
        data = build_rerank_output_data(response)
        assert data["results"] == [{"index": 1, "relevance_score": 0.93}]
        assert data["count"] == 1
        assert data["model"] == "rerank-v3.5"
        assert data["id"] == "resp_1"
        assert "preview_truncated" not in data

    def test_rerank_output_none_without_results(self):
        assert build_rerank_output_data(InternalRerankResponse(model="m", results=[])) is None

    def test_image_input_carries_prompt_and_parameters(self):
        request = InternalImageRequest(
            model="gpt-image-1", prompt="a cat", n=2, size=ImageSize(1024, 768)
        )
        data = build_image_input_data(request)
        assert data["prompt"] == "a cat"
        assert data["n"] == 2
        assert data["size"] == "1024x768"

    def test_image_edit_input_never_carries_image_bytes(self):
        request = InternalImageEditRequest(
            model="gpt-image-1",
            prompt="add a hat",
            images=[ImageEditSource(file=b"PNGDATA", filename="a.png", content_type="image/png")],
            mask=ImageEditSource(image_url="data:image/png;base64,BBBB"),
        )
        data = build_image_input_data(request)
        assert data["images"] == [{"filename": "a.png", "content_type": "image/png", "bytes": 7}]
        assert data["mask"] == {"image_url": "<data-uri 26 chars>"}

    def test_image_output_keeps_urls_and_summarizes_base64(self):
        response = InternalImageResponse(
            created=1700000000,
            model="gpt-image-1",
            data=[ImageData(url="https://x/y.png"), ImageData(b64_json="QUJD")],
        )
        data = build_image_output_data(response)
        assert data["count"] == 2
        assert data["images"][0] == {"url": "https://x/y.png"}
        assert data["images"][1] == {"b64_json_length": 4}
        assert data["created"] == 1700000000

    def test_speech_payload_keeps_the_script_not_the_audio(self):
        request = InternalSpeechRequest(model="tts-1", input="hello there", voice="alloy")
        assert build_speech_input_data(request) == {
            "input": "hello there",
            "voice": "alloy",
            "response_format": "mp3",
            "speed": 1.0,
        }
        response = InternalSpeechResponse(content=b"ID3audio", content_type="audio/mpeg")
        assert build_speech_output_data(response) == {
            "content_type": "audio/mpeg",
            "bytes": 8,
        }

    def test_audio_transcript_payloads_describe_the_upload(self):
        request = InternalTranscriptionRequest(
            model="whisper-1", file=b"RIFFdata", filename="meeting.mp3", language="de"
        )
        data = build_audio_transcript_input_data(request)
        assert data["filename"] == "meeting.mp3"
        assert data["bytes"] == 8
        assert data["language"] == "de"

        response = InternalTranscriptionResponse(
            text="hallo", language="de", duration=12.5, segments=[{"text": "hallo"}]
        )
        assert build_audio_transcript_output_data(response) == {
            "text": "hallo",
            "language": "de",
            "duration": 12.5,
            "segment_count": 1,
        }

    def test_translation_shares_the_transcript_builders(self):
        request = InternalTranslationRequest(model="whisper-1", file=b"RIFF", filename="a.mp3")
        assert build_audio_transcript_input_data(request)["filename"] == "a.mp3"
        assert build_audio_transcript_output_data(InternalTranslationResponse(text="hello")) == {
            "text": "hello",
            "language": "english",
        }

    def test_systemone_payloads(self):
        request = InternalSystemOneRequest(
            model="jev-latest", state={"context": "x"}, questions={"q1": {"type": "bool"}}
        )
        assert build_systemone_input_data(request) == {
            "state": {"context": "x"},
            "questions": {"q1": {"type": "bool"}},
        }
        response = InternalSystemOneResponse(model="jev-latest", answers={"q1": True})
        assert build_systemone_output_data(response) == {
            "answers": {"q1": True},
            "model": "jev-latest",
        }

    def test_decisions_payloads(self):
        request = InternalDecisionRequest(
            model="m", input=[{"role": "user", "content": "evidence"}], questions=[{"name": "q"}]
        )
        data = build_decisions_input_data(request)
        assert data["input"][0]["content"] == "evidence"
        assert data["questions"] == [{"name": "q"}]

        response = InternalDecisionResponse(model="m", answers=[{"name": "q", "answer": True}])
        assert build_decisions_output_data(response) == {
            "answers": [{"name": "q", "answer": True}],
            "model": "m",
        }

    def test_decisions_input_summarizes_inline_images_and_bounds_text(self):
        # ``input`` is raw client JSON: an ``input_image`` part carries a base64
        # data URL, which must never reach the trace.
        request = InternalDecisionRequest(
            model="m",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "x" * 5000},
                        {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
                    ],
                }
            ],
            questions={"instructions": "y" * 5000},
        )
        data = build_decisions_input_data(request)
        parts = data["input"][0]["content"]
        assert parts[0]["text"].endswith("…<truncated 3000 chars>")
        assert parts[1]["image_url"] == "<data-uri 26 chars>"
        assert data["questions"]["instructions"].endswith("…<truncated 3000 chars>")

    def test_rerank_input_bounds_extra_document_keys(self):
        document = RerankDocument(text="b", extra={"meta": "z" * 5000, "thumb": "data:x"})
        request = InternalRerankRequest(model="m", query="q", documents=[document])
        rendered = build_rerank_input_data(request)["documents"][0]
        # Every leaf is bounded, not just the modelled ``text``/``image`` keys.
        assert rendered["meta"].endswith("…<truncated 3000 chars>")
        assert rendered["thumb"] == "<data-uri 6 chars>"

    def test_decisions_output_none_without_answers(self):
        assert build_decisions_output_data(InternalDecisionResponse(model="m", answers=[])) is None

    async def test_rerank_request_opens_a_generation_with_content(
        self, mock_client, mock_propagate
    ):
        settings = {"public_key": "pk-test", "secret_key": "sk-test"}
        handler = LangfuseTracingHandler.create_handler(settings)

        gen = _make_generation()
        mock_client.start_as_current_observation.return_value.__enter__.return_value = gen

        request = InternalRerankRequest(model="rerank-v3.5", query="q", documents=["a", "b"])
        context = EventContext(
            request_id="req-rerank",
            trace_id="trace-1",
            model="rerank-v3.5",
            request_type=RequestType.RERANK,
            usage=Usage(input_tokens=12),
            metadata={"endpoint": "/v1/rerank"},
        )
        response = InternalRerankResponse(
            model="rerank-v3.5", results=[RerankResult(index=0, relevance_score=0.8)]
        )

        await handler.on_request_start(request, context)
        await handler.on_request_end(request, response, context)

        start_kwargs = mock_client.start_as_current_observation.call_args.kwargs
        assert start_kwargs["as_type"] == "generation"
        assert start_kwargs["name"] == "rerank"
        assert start_kwargs["input"] == {"query": "q", "documents": ["a", "b"]}

        last_update_kwargs = gen.update.call_args_list[-1].kwargs
        assert last_update_kwargs["output"]["count"] == 1
        assert last_update_kwargs["usage_details"] == {"input": 12, "output": 0, "total": 12}
        gen.end.assert_called_once()

    async def test_image_request_opens_a_generation_with_the_prompt(
        self, mock_client, mock_propagate
    ):
        settings = {"public_key": "pk-test", "secret_key": "sk-test"}
        handler = LangfuseTracingHandler.create_handler(settings)

        gen = _make_generation()
        mock_client.start_as_current_observation.return_value.__enter__.return_value = gen

        request = InternalImageRequest(model="gpt-image-1", prompt="a cat")
        context = EventContext(
            request_id="req-img",
            trace_id="trace-1",
            model="gpt-image-1",
            request_type=RequestType.IMAGE_GENERATION,
            metadata={"endpoint": "/v1/images/generations"},
        )
        response = InternalImageResponse(created=1, data=[ImageData(url="https://x/y.png")])

        await handler.on_request_start(request, context)
        await handler.on_request_end(request, response, context)

        assert (
            mock_client.start_as_current_observation.call_args.kwargs["input"]["prompt"] == "a cat"
        )
        assert gen.update.call_args_list[-1].kwargs["output"]["images"] == [
            {"url": "https://x/y.png"}
        ]


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

    def test_build_embedding_input_data_passes_text_through(self):
        assert build_embedding_input_data(_make_embedding_request("hello")) == "hello"
        assert build_embedding_input_data(_make_embedding_request(["a", "b"])) == ["a", "b"]

    def test_build_embedding_input_data_bounds_a_retrieval_batch(self):
        data = build_embedding_input_data(_make_embedding_request(["x" * 5000] + ["t"] * 40))
        assert data[0].endswith("…<truncated 3000 chars>")
        # The head of the batch plus a marker, so the span stays bounded.
        assert len(data) == 21
        assert data[-1] == "<truncated 21 texts>"

    def test_build_embedding_input_data_none_when_empty(self):
        assert build_embedding_input_data(_make_embedding_request("")) is None
        assert build_embedding_input_data(_make_embedding_request([])) is None
        # Token-id inputs are not text: nothing readable to show.
        assert build_embedding_input_data(_make_embedding_request([1, 2, 3])) is None

    def test_build_embedding_output_data_keeps_shape_and_preview(self):
        data = build_embedding_output_data(_make_embedding_response(count=2, dimensions=768))
        assert data["object"] == "list"
        assert data["count"] == 2
        assert data["dimensions"] == 768
        assert "preview_truncated" not in data
        first = data["embeddings"][0]
        assert first["index"] == 0
        assert first["dimensions"] == 768
        assert first["preview"] == [float(i) for i in range(8)]

    def test_build_embedding_output_data_truncates_large_batches(self):
        """Full vectors would dwarf the span: only the leading ones are previewed."""
        data = build_embedding_output_data(_make_embedding_response(count=100, dimensions=3072))
        assert data["count"] == 100
        assert data["preview_truncated"] is True
        assert len(data["embeddings"]) == 5

    def test_build_embedding_output_data_reports_base64_without_vectors(self):
        item = MagicMock()
        item.index = 0
        item.embedding = "AAECAwQFBgc="
        response = MagicMock()
        response.model = "gemini-embedding-001"
        response.data = [item]

        data = build_embedding_output_data(response)
        assert "dimensions" not in data
        assert data["embeddings"] == [
            {"index": 0, "encoding": "base64", "encoded_length": len("AAECAwQFBgc=")}
        ]

    def test_build_embedding_output_data_none_without_vectors(self):
        response = MagicMock()
        response.data = []
        assert build_embedding_output_data(response) is None

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
            usage=Usage(
                input_tokens=5,
                output_tokens=10,
                total_tokens=15,
            ),
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
            usage=Usage(
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
                cache_read_input_tokens=30,
                cache_creation_input_tokens=10,
                reasoning_tokens=20,
                prompt_tokens_details=PromptTokensDetails(audio_tokens=5),
                completion_tokens_details=CompletionTokensDetails(audio_tokens=4),
            ),
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
            usage=Usage(
                input_tokens=100,
                output_tokens=10,
                total_tokens=110,
                cache_read_input_tokens=25,
            ),
        )
        details = build_usage_details(context)
        assert details["cache_read_input_tokens"] == 25
        assert details["input"] == 75

    def test_build_usage_details_none_without_token_data(self):
        context = EventContext(request_id="req-1", trace_id="trace-1", model="gpt-4")
        assert build_usage_details(context) is None

    def test_build_embedding_usage_details_drops_output_buckets(self):
        """An embedding bills input only; a zero ``output`` row prices nothing."""
        context = EventContext(
            request_id="req-1",
            trace_id="trace-1",
            model="gemini-embedding-001",
            usage=Usage(input_tokens=9),
        )
        assert build_embedding_usage_details(context) == {"input": 9, "total": 9}
        assert build_usage_details(context) == {"input": 9, "output": 0, "total": 9}

    def test_build_embedding_usage_details_none_without_usage(self):
        context = EventContext(request_id="req-1", trace_id="trace-1", model="m")
        assert build_embedding_usage_details(context) is None

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
