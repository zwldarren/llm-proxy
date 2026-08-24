"""Tests for OpenAI adapter."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_proxy.core.adapter import get_adapter, list_providers
from llm_proxy.models import ConversationContext, InternalRequest, Message, TextBlock
from llm_proxy.providers.openai.adapter import OpenAIAdapter
from llm_proxy.serialization.openai.serializer import (
    OpenAIResponsesProviderSerializer,
)


class MockAsyncIterator:
    """Async iterator for test bytes chunks."""

    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk


def test_openai_adapter_is_registered():
    """Test that openai adapter is registered."""
    assert "openai" in list_providers()


def test_openai_adapter_can_be_created():
    """Test that OpenAI adapter can be instantiated."""
    adapter = get_adapter("openai", api_key="test-key")
    assert adapter.__class__.__name__ == "OpenAIAdapter"


@pytest.fixture
def openai_adapter():
    """Create an OpenAI adapter for testing."""
    return OpenAIAdapter(
        api_key="test-key",
        base_url="https://api.openai.com/v1",
    )


class TestStreamingOutputType:
    """Test that streaming output is dict (canonical OpenAI chunks)."""

    @pytest.mark.asyncio
    async def test_stream_yields_dicts_not_strings(self, openai_adapter):
        """Streaming chunks should be dicts (canonical OpenAI format), not SSE strings."""
        test_chunks = [
            b"event: response.created\n",
            b'data: {"type":"response.created","response":{"id":"resp_123"}}\n',
            b"event: response.output_item.added\n",
            b'data: {"type":"response.output_item.added","output_index":0}\n',
            b"event: response.content_part.added\n",
            b'data: {"type":"response.content_part.added","content_index":0}\n',
            b"event: response.output_text.delta\n",
            b'data: {"type":"response.output_text.delta","delta":"Hello"}\n',
            b"event: response.output_text.delta\n",
            b'data: {"type":"response.output_text.delta","delta":" world"}\n',
            b"event: response.output_text.done\n",
            b'data: {"type":"response.output_text.done","text":"Hello world"}\n',
            b"event: response.completed\n",
            (
                b'data: {"type":"response.completed",'
                b'"response":{"id":"resp_123","status":"completed"}}\n'
            ),
            b"data: [DONE]\n",
        ]

        class MockResponse:
            status_code = 200

            def json(self):
                return {}

            def iter_lines(self):
                return MockAsyncIterator(test_chunks)

        mock_response = MockResponse()
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch.object(openai_adapter, "_get_client", return_value=mock_client):
            request = InternalRequest(
                model="gpt-4o",
                conversation=ConversationContext(
                    messages=[Message(role="user", content=[TextBlock(text="hi")])]
                ),
                stream=True,
            )

            chunks = []
            async for chunk in await openai_adapter.stream_chat_completion(request):
                chunks.append(chunk)

            # Chunks should be dict (canonical format) except final "[DONE]" string.
            dict_count = 0
            done_count = 0
            for i, chunk in enumerate(chunks):
                if chunk == "[DONE]":
                    done_count += 1
                elif isinstance(chunk, dict):
                    dict_count += 1
                else:
                    pytest.fail(
                        f"Chunk {i} is {type(chunk).__name__}, expected dict or '[DONE]': {chunk!r}"
                    )
            assert dict_count > 0, "Expected at least one dict chunk"
            assert done_count == 1, f"Expected exactly one [DONE], got {done_count}"

    @pytest.mark.asyncio
    async def test_stream_handles_invalid_json_gracefully(self, openai_adapter):
        """Invalid JSON in stream should be skipped gracefully."""
        test_chunks = [
            b"event: response.created\n",
            b'data: {"type":"response.created","response":{"id":"resp_123"}}\n',
            b"data: this is not valid json\n",
            b"event: response.completed\n",
            (
                b'data: {"type":"response.completed",'
                b'"response":{"id":"resp_123","status":"completed"}}\n'
            ),
            b"data: [DONE]\n",
        ]

        class MockResponse:
            status_code = 200

            def json(self):
                return {}

            def iter_lines(self):
                return MockAsyncIterator(test_chunks)

        mock_response = MockResponse()
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch.object(openai_adapter, "_get_client", return_value=mock_client):
            request = InternalRequest(
                model="gpt-4o",
                conversation=ConversationContext(
                    messages=[Message(role="user", content=[TextBlock(text="hi")])]
                ),
                stream=True,
            )

            chunks = []
            async for chunk in await openai_adapter.stream_chat_completion(request):
                chunks.append(chunk)

            # All chunks should be dicts or "[DONE]".
            for i, chunk in enumerate(chunks):
                assert isinstance(chunk, (dict, str)), (
                    f"Chunk {i} is {type(chunk).__name__}, expected dict or str: {chunk!r}"
                )

            # Invalid JSON must be skipped, not forwarded as a raw string.
            invalid_yielded = any(isinstance(c, str) and c != "[DONE]" for c in chunks)
            assert not invalid_yielded, "Invalid JSON should be skipped, not yielded as a string"

    @pytest.mark.asyncio
    async def test_stream_handles_non_sse_lines(self, openai_adapter):
        """Non-SSE lines (comment lines) should be skipped."""
        test_chunks = [
            b"event: response.created\n",
            b'data: {"type":"response.created","response":{"id":"resp_123"}}\n',
            b": this is a comment line\n",
            b"event: response.completed\n",
            (
                b'data: {"type":"response.completed",'
                b'"response":{"id":"resp_123","status":"completed"}}\n'
            ),
            b"data: [DONE]\n",
        ]

        class MockResponse:
            status_code = 200

            def json(self):
                return {}

            def iter_lines(self):
                return MockAsyncIterator(test_chunks)

        mock_response = MockResponse()
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch.object(openai_adapter, "_get_client", return_value=mock_client):
            request = InternalRequest(
                model="gpt-4o",
                conversation=ConversationContext(
                    messages=[Message(role="user", content=[TextBlock(text="hi")])]
                ),
                stream=True,
            )

            chunks = []
            async for chunk in await openai_adapter.stream_chat_completion(request):
                chunks.append(chunk)

            # All chunks should be dicts or "[DONE]".
            for i, chunk in enumerate(chunks):
                assert isinstance(chunk, (dict, str)), (
                    f"Chunk {i} is {type(chunk).__name__}, expected dict or str: {chunk!r}"
                )

            # Non-SSE comment lines must be skipped, not forwarded as strings.
            comment_yielded = any(isinstance(c, str) and c != "[DONE]" for c in chunks)
            assert not comment_yielded, "Comment lines should be skipped, not yielded as strings"


class TestToolChoiceConversion:
    """Test tool_choice conversion in OpenAIResponsesProviderSerializer."""

    _serializer = OpenAIResponsesProviderSerializer()

    def test_tool_choice_named(self):
        from llm_proxy.models.tools import ToolChoiceNamed

        result = self._serializer._build_tool_choice(ToolChoiceNamed(name="get_weather"))
        assert result == {"type": "function", "name": "get_weather"}

    def test_tool_choice_allowed_tools(self):
        from llm_proxy.models.tools import AllowedToolsConfig, ToolChoiceAllowedTools

        result = self._serializer._build_tool_choice(
            ToolChoiceAllowedTools(
                allowed_tools=AllowedToolsConfig(
                    mode="auto",
                    tools=[{"type": "function", "name": "get"}],
                )
            )
        )
        # The full allowed_tools shape is forwarded so the hard constraint is
        # enforced provider-side (spec: allowed_tools MUST be enforced).
        assert result == {
            "type": "allowed_tools",
            "tools": [{"type": "function", "name": "get"}],
            "mode": "auto",
        }

    def test_tool_choice_custom(self):
        from llm_proxy.models.tools import ToolChoiceCustom

        result = self._serializer._build_tool_choice(ToolChoiceCustom(name="my_tool"))
        assert result == {"type": "custom", "name": "my_tool"}

    def test_tool_choice_string(self):
        result = self._serializer._build_tool_choice("required")
        assert result == "required"

    def test_tool_choice_none(self):
        result = self._serializer._build_tool_choice(None)
        assert result is None


class TestStreamOptionsForwarding:
    """stream_options is normalized to the fields the Responses API accepts."""

    def _build_request(self, stream_options: dict | None):
        from llm_proxy.protocols.openresponses.serializer import (
            OpenResponsesProtocolSerializer,
        )
        from llm_proxy.serialization.context import BuildContext

        unified = OpenResponsesProtocolSerializer().parse_request(
            {
                "model": "gpt-5.2",
                "input": "hi",
                "stream": True,
                "stream_options": stream_options,
            }
        )
        context = BuildContext.from_request(
            unified,
            provider_name="openai",
            base_url="https://api.openai.com/v1",
            target_endpoint="responses",
        )
        return OpenAIResponsesProviderSerializer().build_provider_request(unified, context)

    def test_include_obfuscation_forwarded(self):
        body = self._build_request({"include_obfuscation": True})
        assert body["stream_options"] == {"include_obfuscation": True}

    def test_include_obfuscation_false_forwarded(self):
        body = self._build_request({"include_obfuscation": False})
        assert body["stream_options"] == {"include_obfuscation": False}

    def test_include_usage_is_dropped(self):
        """The Responses API rejects include_usage with 400 Unknown parameter.

        The schema drops the unknown ``include_usage`` field and applies the
        ``include_obfuscation`` default, so only the supported field is sent.
        """
        body = self._build_request({"include_usage": True})
        assert body["stream_options"] == {"include_obfuscation": True}
        assert "include_usage" not in body["stream_options"]

    def test_mixed_options_keep_only_obfuscation(self):
        body = self._build_request({"include_usage": True, "include_obfuscation": False})
        assert body["stream_options"] == {"include_obfuscation": False}

    def test_no_stream_options_sends_nothing(self):
        """Without stream_options, nothing is sent (the Responses API always
        includes usage in response.completed, so no usage request is needed)."""
        from llm_proxy.protocols.openresponses.serializer import (
            OpenResponsesProtocolSerializer,
        )
        from llm_proxy.serialization.context import BuildContext

        unified = OpenResponsesProtocolSerializer().parse_request(
            {"model": "gpt-5.2", "input": "hi", "stream": True}
        )
        context = BuildContext.from_request(
            unified,
            provider_name="openai",
            base_url="https://api.openai.com/v1",
            target_endpoint="responses",
        )
        body = OpenAIResponsesProviderSerializer().build_provider_request(unified, context)
        assert "stream_options" not in body


class TestNativeResponsesStreaming:
    """Native SSE passthrough for the OpenResponses protocol."""

    def test_supports_native_streaming_only_for_openresponses(self, openai_adapter):
        """Native passthrough applies to the openresponses protocol only."""
        assert openai_adapter.supports_native_streaming("openresponses") is True
        assert openai_adapter.supports_native_streaming("openai") is False
        assert openai_adapter.supports_native_streaming("anthropic") is False

    @pytest.mark.asyncio
    async def test_native_stream_yields_raw_sse_blocks(self, openai_adapter):
        """stream_chat_completion_native forwards upstream SSE blocks verbatim,
        preserving Codex item types the chat-chunk path cannot represent."""
        test_chunks = [
            b"event: response.created\n",
            b'data: {"type":"response.created","response":{"id":"resp_123"}}\n',
            b"\n",
            b"event: response.output_item.added\n",
            (
                b'data: {"type":"response.output_item.added","output_index":0,'
                b'"item":{"type":"custom_tool_call","id":"ctc_1","call_id":"call_1",'
                b'"name":"apply_patch","status":"in_progress"}}\n'
            ),
            b"\n",
            b"event: response.custom_tool_call_arguments.delta\n",
            b'data: {"type":"response.custom_tool_call_arguments.delta","delta":"{\\"patch\\":"}\n',
            b"\n",
            b"event: response.completed\n",
            (
                b'data: {"type":"response.completed","response":{"id":"resp_123",'
                b'"status":"completed","output":[]}}\n'
            ),
            b"\n",
            b"data: [DONE]\n",
            b"\n",
        ]

        class MockResponse:
            status_code = 200

            def json(self):
                return {}

            def iter_lines(self):
                return MockAsyncIterator(test_chunks)

        mock_response = MockResponse()
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch.object(openai_adapter, "_get_client", return_value=mock_client):
            request = InternalRequest(
                model="gpt-5.2",
                conversation=ConversationContext(
                    messages=[Message(role="user", content=[TextBlock(text="hi")])]
                ),
                stream=True,
            )
            blocks = []
            async for block in await openai_adapter.stream_chat_completion_native(request):
                blocks.append(block)

        joined = "".join(blocks)
        assert "event: response.created" in joined
        assert "event: response.custom_tool_call_arguments.delta" in joined
        assert '"type":"custom_tool_call"' in joined
        assert 'data: {"type":"response.completed"' in joined
        assert "data: [DONE]" in joined

    @pytest.mark.asyncio
    async def test_native_stream_handles_http_error(self, openai_adapter):
        """HTTP errors during native streaming are wrapped as ProviderError."""

        class MockResponse:
            status_code = 429

            def json(self):
                return {"error": {"message": "rate limited"}}

            def iter_lines(self):
                return MockAsyncIterator([b"data: [DONE]\n"])

        mock_response = MockResponse()
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        from llm_proxy.core.exceptions import ProviderError

        with patch.object(openai_adapter, "_get_client", return_value=mock_client):
            request = InternalRequest(
                model="gpt-5.2",
                conversation=ConversationContext(
                    messages=[Message(role="user", content=[TextBlock(text="hi")])]
                ),
                stream=True,
            )
            with pytest.raises(ProviderError):
                stream = await openai_adapter.stream_chat_completion_native(request)
                async for _ in stream:
                    pass


class TestNativeResponseRoundTrip:
    """Non-streaming native upstream responses must round-trip losslessly."""

    def test_codex_item_types_survive_round_trip(self):
        """custom_tool_call / local_shell_call / unknown items are preserved
        through the InternalResponse round-trip."""
        from llm_proxy.protocols.openresponses.serializer import (
            OpenResponsesProtocolSerializer,
        )

        upstream = {
            "id": "resp_upstream1",
            "object": "response",
            "created_at": 1750000000,
            "status": "completed",
            "model": "gpt-5.6",
            "output": [
                {
                    "type": "message",
                    "id": "msg_1",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "patching"}],
                    "phase": "final_answer",
                },
                {
                    "type": "custom_tool_call",
                    "id": "ctc_1",
                    "call_id": "call_abc",
                    "name": "apply_patch",
                    "input": '{"patch": "--- a/foo\\n+++ b/foo\\n"}',
                },
                {
                    "type": "local_shell_call",
                    "id": "lsc_1",
                    "call_id": "call_ghi",
                    "status": "completed",
                    "action": {"type": "exec", "command": ["ls"]},
                },
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [{"type": "summary_text", "text": "Thinking"}],
                },
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 5},
            },
        }

        parsed = OpenAIResponsesProviderSerializer().parse_provider_response(
            upstream, model="gpt-5.6"
        )
        formatted = OpenResponsesProtocolSerializer().format_response(parsed)

        types = [i.get("type") for i in formatted["output"]]
        assert types == [
            "message",
            "custom_tool_call",
            "local_shell_call",
            "reasoning",
        ], f"round-trip dropped items: {types}"
        ctc = formatted["output"][1]
        assert ctc["name"] == "apply_patch"
        assert '"patch": "--- a/foo' in ctc["input"]
        assert formatted["output"][2]["action"]["command"] == ["ls"]


class TestParallelToolCallsForwarding:
    """parallel_tool_calls must reach the native Responses upstream body."""

    def _build_body(self, value: bool | None):
        from llm_proxy.protocols.openresponses.serializer import (
            OpenResponsesProtocolSerializer,
        )
        from llm_proxy.serialization.context import BuildContext

        raw = {"model": "gpt-5.2", "input": "hi"}
        if value is not None:
            raw["parallel_tool_calls"] = value
        unified = OpenResponsesProtocolSerializer().parse_request(raw)
        context = BuildContext.from_request(
            unified,
            provider_name="openai",
            base_url="https://api.openai.com/v1",
            target_endpoint="responses",
        )
        return OpenAIResponsesProviderSerializer().build_provider_request(unified, context)

    def test_false_forwarded(self):
        assert self._build_body(False)["parallel_tool_calls"] is False

    def test_true_forwarded(self):
        assert self._build_body(True)["parallel_tool_calls"] is True

    def test_omitted_sends_nothing(self):
        assert "parallel_tool_calls" not in self._build_body(None)


class TestUpstreamFailedStatus:
    """A 2xx response object with status failed/cancelled must surface as a
    failed response, never as a bogus completion."""

    @staticmethod
    def _upstream(status: str, error: dict | None = None) -> dict:
        return {
            "id": "resp_fail1",
            "object": "response",
            "created_at": 1750000000,
            "status": status,
            "model": "gpt-5.6",
            "output": [],
            "error": error,
            "usage": {
                "input_tokens": 5,
                "output_tokens": 1,
                "total_tokens": 6,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }

    @staticmethod
    def _format(upstream: dict) -> dict:
        from llm_proxy.protocols.openresponses.handler import (
            clear_format_context,
            set_format_context,
        )
        from llm_proxy.protocols.openresponses.serializer import (
            OpenResponsesProtocolSerializer,
        )

        parsed = OpenAIResponsesProviderSerializer().parse_provider_response(
            upstream, model="gpt-5.6"
        )
        set_format_context({"model": "gpt-5.6", "input": "hi"})
        try:
            return OpenResponsesProtocolSerializer().format_response(parsed)
        finally:
            clear_format_context()

    def test_failed_becomes_failed_with_upstream_error(self):
        error = {
            "code": "server_error",
            "message": "upstream exploded",
            "type": "server_error",
            "param": None,
        }
        formatted = self._format(self._upstream("failed", error))
        assert formatted["status"] == "failed"
        assert formatted["error"] == error

    def test_failed_without_error_payload_gets_default(self):
        formatted = self._format(self._upstream("failed"))
        assert formatted["status"] == "failed"
        assert formatted["error"]["code"] == "provider_error"

    def test_cancelled_becomes_failed(self):
        formatted = self._format(self._upstream("cancelled"))
        assert formatted["status"] == "failed"


class TestIncompleteReasonPreserved:
    """incomplete_details.reason from the upstream (max_output_tokens /
    content_filter) must not be collapsed to a generic length."""

    def test_content_filter_reason_preserved(self):
        from llm_proxy.protocols.openresponses.handler import (
            clear_format_context,
            set_format_context,
        )
        from llm_proxy.protocols.openresponses.serializer import (
            OpenResponsesProtocolSerializer,
        )

        upstream = {
            "id": "resp_inc1",
            "object": "response",
            "created_at": 1750000000,
            "status": "incomplete",
            "model": "gpt-5.6",
            "output": [],
            "incomplete_details": {"reason": "content_filter"},
            "usage": {
                "input_tokens": 5,
                "output_tokens": 1,
                "total_tokens": 6,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }
        parsed = OpenAIResponsesProviderSerializer().parse_provider_response(
            upstream, model="gpt-5.6"
        )
        set_format_context({"model": "gpt-5.6", "input": "hi"})
        try:
            formatted = OpenResponsesProtocolSerializer().format_response(parsed)
        finally:
            clear_format_context()
        assert formatted["status"] == "incomplete"
        assert formatted["incomplete_details"] == {"reason": "content_filter"}

    def test_missing_reason_defaults_to_length(self):
        from llm_proxy.protocols.openresponses.serializer import (
            OpenResponsesProtocolSerializer,
        )

        upstream = {
            "id": "resp_inc2",
            "object": "response",
            "created_at": 1750000000,
            "status": "incomplete",
            "model": "gpt-5.6",
            "output": [],
            "usage": {
                "input_tokens": 5,
                "output_tokens": 1,
                "total_tokens": 6,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }
        parsed = OpenAIResponsesProviderSerializer().parse_provider_response(
            upstream, model="gpt-5.6"
        )
        formatted = OpenResponsesProtocolSerializer().format_response(parsed)
        assert formatted["status"] == "incomplete"
        assert formatted["incomplete_details"] == {"reason": "length"}


class TestWebSearchActionPreserved:
    """Non-streaming native passthrough keeps the upstream web_search_call
    action (query/queries/sources) instead of dropping the sources."""

    def test_upstream_action_forwarded_verbatim(self):
        from llm_proxy.protocols.openresponses.handler import (
            clear_format_context,
            set_format_context,
        )
        from llm_proxy.protocols.openresponses.serializer import (
            OpenResponsesProtocolSerializer,
        )

        upstream = {
            "id": "resp_ws1",
            "object": "response",
            "created_at": 1750000000,
            "status": "completed",
            "model": "gpt-5.6",
            "output": [
                {
                    "type": "web_search_call",
                    "id": "ws_1",
                    "status": "completed",
                    "action": {
                        "type": "search",
                        "query": "llm proxy",
                        "queries": ["llm proxy"],
                        "sources": [{"url": "https://a.com", "title": "A"}],
                    },
                }
            ],
            "usage": {
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }
        parsed = OpenAIResponsesProviderSerializer().parse_provider_response(
            upstream, model="gpt-5.6"
        )
        set_format_context({"model": "gpt-5.6", "input": "hi"})
        try:
            formatted = OpenResponsesProtocolSerializer().format_response(parsed)
        finally:
            clear_format_context()
        item = formatted["output"][0]
        assert item["type"] == "web_search_call"
        assert item["action"]["sources"] == [{"url": "https://a.com", "title": "A"}]
        assert item["action"]["queries"] == ["llm proxy"]

    def test_interceptor_path_still_include_gated(self):
        """Proxy-executed searches keep the include-gated sources behavior."""
        from llm_proxy.models.content_blocks.anthropic_builtin import (
            WebSearchToolResultBlock,
        )
        from llm_proxy.models.content_blocks.extended import ServerToolUseBlock
        from llm_proxy.models.internal import InternalResponse
        from llm_proxy.protocols.openresponses.handler import (
            clear_format_context,
            set_format_context,
        )
        from llm_proxy.protocols.openresponses.serializer import (
            OpenResponsesProtocolSerializer,
        )

        response = InternalResponse(
            id="resp_ws2",
            model="gpt-5.6",
            output=[
                ServerToolUseBlock(
                    id="ws_2",
                    name="web_search",
                    input={"query": "q"},
                ),
                WebSearchToolResultBlock(
                    tool_use_id="ws_2",
                    content=[{"url": "https://b.com", "title": "B"}],
                ),
            ],
        )
        set_format_context(
            {"model": "gpt-5.6", "input": "hi", "include": ["web_search_call.action.sources"]}
        )
        try:
            formatted = OpenResponsesProtocolSerializer().format_response(response)
        finally:
            clear_format_context()
        item = formatted["output"][0]
        assert item["type"] == "web_search_call"
        assert item["action"]["sources"] == [{"url": "https://b.com", "title": "B"}]


class TestRawResponsesFieldsForwarded:
    """Unknown /v1/responses fields reach the native upstream verbatim and are
    stripped from the Chat Completions translation."""

    def _native_body(self, raw: dict) -> dict:
        from llm_proxy.protocols.openresponses.serializer import (
            OpenResponsesProtocolSerializer,
        )
        from llm_proxy.serialization.context import BuildContext

        unified = OpenResponsesProtocolSerializer().parse_request(raw)
        context = BuildContext.from_request(
            unified,
            provider_name="openai",
            base_url="https://api.openai.com/v1",
            target_endpoint="responses",
        )
        return OpenAIResponsesProviderSerializer().build_provider_request(unified, context)

    def test_context_management_and_prompt_cache_options_forwarded(self):
        body = self._native_body(
            {
                "model": "gpt-5.6",
                "input": "hi",
                "context_management": ["auto"],
                "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
            }
        )
        assert body["context_management"] == ["auto"]
        assert body["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}

    def test_rebuilt_keys_win_over_raw_fields(self):
        """A schema-modeled field is never overridden by the raw passthrough."""
        body = self._native_body(
            {"model": "gpt-5.6", "input": "hi", "temperature": 0.5, "temperature_extra": 1}
        )
        assert body["temperature"] == 0.5
        assert "temperature_extra" in body

    def test_chat_translation_strips_raw_fields(self):
        from llm_proxy.protocols.openresponses.serializer import (
            OpenResponsesProtocolSerializer,
        )
        from llm_proxy.serialization.context import BuildContext
        from llm_proxy.serialization.openai.components.request_builder import (
            OpenAIRequestBuilder,
        )

        unified = OpenResponsesProtocolSerializer().parse_request(
            {
                "model": "gpt-5.6",
                "input": "hi",
                "context_management": ["auto"],
                "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
            }
        )
        context = BuildContext.from_request(
            unified,
            provider_name="openai-compatible",
            base_url="https://api.example.com/v1",
            target_endpoint="chat_completions",
        )
        body = OpenAIRequestBuilder().build(unified, context)
        assert "responses_raw_fields" not in body
        assert "context_management" not in body
        assert "prompt_cache_options" not in body

    def test_chat_translation_strips_carrier_even_without_raw_fields(self):
        from llm_proxy.protocols.openresponses.serializer import (
            OpenResponsesProtocolSerializer,
        )
        from llm_proxy.serialization.context import BuildContext
        from llm_proxy.serialization.openai.components.request_builder import (
            OpenAIRequestBuilder,
        )

        unified = OpenResponsesProtocolSerializer().parse_request(
            {"model": "gpt-5.6", "input": "hi"}
        )
        context = BuildContext.from_request(
            unified,
            provider_name="openai-compatible",
            base_url="https://api.example.com/v1",
            target_endpoint="chat_completions",
        )
        body = OpenAIRequestBuilder().build(unified, context)
        assert "responses_raw_fields" not in body
