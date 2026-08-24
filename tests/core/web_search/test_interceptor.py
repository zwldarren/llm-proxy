"""Tests for web search interceptor."""

import pytest

from llm_proxy.models import (
    InternalRequest,
    InternalResponse,
    Message,
    ServerToolUseBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from llm_proxy.models.content_blocks.anthropic_builtin import WebSearchToolResultBlock
from llm_proxy.models.tools import UserLocation, WebSearchTool
from llm_proxy.web_search.interceptor import WebSearchInterceptor
from llm_proxy.web_search.provider import (
    SearchResult,
    WebSearchProvider,
    WebSearchResponse,
    WebSearchToolConfig,
)


class MockWebSearchProvider(WebSearchProvider):
    """Mock web search provider for testing."""

    def __init__(self):
        self.search_results = [
            SearchResult(
                url="https://example.com/result1",
                title="Result 1",
                snippet="This is result 1",
                page_age="2024-01-15",
            ),
            SearchResult(
                url="https://example.com/result2",
                title="Result 2",
                snippet="This is result 2",
            ),
        ]

    async def search(self, query: str, config: WebSearchToolConfig | None = None, **kwargs):
        return WebSearchResponse(
            results=self.search_results,
            search_id="ws_test_123",
        )

    async def close(self):
        pass


class TestWebSearchInterceptor:
    """Tests for WebSearchInterceptor."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock web search provider."""
        return MockWebSearchProvider()

    @pytest.fixture
    def interceptor(self, mock_provider):
        """Create an interceptor with mock provider."""
        return WebSearchInterceptor(mock_provider)

    def test_has_web_search_tool_with_web_search_tool(self, interceptor):
        """Test detecting WebSearchTool in tools list."""
        tools = [
            WebSearchTool(name="web_search", type="web_search_20250305"),
        ]

        assert interceptor.has_web_search_tool(tools) is True

    def test_has_web_search_tool_with_dict(self, interceptor):
        """Test detecting web_search tool in dict format."""
        tools = [
            {"type": "web_search_20250305", "name": "web_search"},
        ]

        assert interceptor.has_web_search_tool(tools) is True

    def test_has_web_search_tool_empty(self, interceptor):
        """Test with no web_search tool."""
        tools = [
            {"type": "function", "name": "other_tool"},
        ]

        assert interceptor.has_web_search_tool(tools) is False

    def test_has_web_search_tool_none(self, interceptor):
        """Test with None tools."""
        assert interceptor.has_web_search_tool(None) is False

    def test_convert_web_search_to_function(self, interceptor):
        """Test converting web_search to function tool."""
        function_tool = interceptor.convert_web_search_to_function()

        assert function_tool.name == "web_search"
        assert function_tool.parameters is not None
        assert "query" in function_tool.parameters["properties"]
        assert function_tool.parameters["required"] == ["query"]
        assert function_tool.description is not None
        assert "web" in function_tool.description.lower()

    def test_is_web_search_tool_use_with_tool_use_block(self, interceptor):
        """Test detecting web_search in ToolUseBlock (non-Anthropic providers)."""

        tool_use = ToolUseBlock(
            id="toolu_123",
            name="web_search",
            input={"query": "test query"},
        )

        assert interceptor.is_web_search_server_tool_use(tool_use) is True

        other_tool = ToolUseBlock(
            id="toolu_456",
            name="other_function",
            input={},
        )

        assert interceptor.is_web_search_server_tool_use(other_tool) is False

    def test_extract_web_search_tool_config(self, interceptor):
        """Test extracting tool config from WebSearchTool."""
        tools = [
            WebSearchTool(
                name="web_search",
                type="web_search_20250305",
                max_uses=5,
                allowed_domains=["example.com"],
                blocked_domains=["spam.com"],
                user_location=UserLocation(
                    city="San Francisco",
                    country="US",
                    region="California",
                    timezone="America/Los_Angeles",
                ),
            ),
        ]

        config = interceptor.extract_web_search_tool_config(tools)

        assert config is not None
        assert config.max_uses == 5
        assert config.allowed_domains == ["example.com"]
        assert config.blocked_domains == ["spam.com"]
        assert config.user_location is not None

    def test_filter_web_search_tools(self, interceptor):
        """Test removing web_search tools from tools list."""
        tools = [
            {"type": "web_search_20250305", "name": "web_search"},
            {"type": "function", "name": "other_tool"},
        ]

        filtered = interceptor.filter_web_search_tools(tools)

        assert len(filtered) == 1
        assert filtered[0]["name"] == "other_tool"

    def test_filter_web_search_tools_all_removed(self, interceptor):
        """Test when all tools are web_search."""
        tools = [
            {"type": "web_search_20250305", "name": "web_search"},
        ]

        filtered = interceptor.filter_web_search_tools(tools)

        assert filtered is None

    def test_is_web_search_server_tool_use(self, interceptor):
        """Test detecting web_search server_tool_use block."""
        web_search_block = ServerToolUseBlock(
            id="srvtoolu_123",
            name="web_search",
            input={"query": "test query"},
        )
        other_block = ServerToolUseBlock(
            id="srvtoolu_456",
            name="other_tool",
            input={},
        )

        assert interceptor.is_web_search_server_tool_use(web_search_block) is True
        assert interceptor.is_web_search_server_tool_use(other_block) is False

    def test_is_web_search_server_tool_use_case_insensitive(self, interceptor):
        """Test detecting web_search with different casing (e.g., WebSearch from some models)."""
        # Test various casing variations that models might output
        web_search_camel_case = ServerToolUseBlock(
            id="srvtoolu_123",
            name="WebSearch",  # CamelCase as some models output
            input={"query": "test query"},
        )
        web_search_upper = ServerToolUseBlock(
            id="srvtoolu_456",
            name="WEB_SEARCH",
            input={"query": "test query"},
        )
        web_search_mixed = ServerToolUseBlock(
            id="srvtoolu_789",
            name="Web_Search",
            input={"query": "test query"},
        )

        # All variations should be detected as web_search
        assert interceptor.is_web_search_server_tool_use(web_search_camel_case) is True
        assert interceptor.is_web_search_server_tool_use(web_search_upper) is True
        assert interceptor.is_web_search_server_tool_use(web_search_mixed) is True

    def test_is_web_search_tool_use_case_insensitive(self, interceptor):
        """Test detecting web_search in ToolUseBlock with different casing."""
        # Test various casing variations for ToolUseBlock (non-Anthropic providers)
        tool_use_camel_case = ToolUseBlock(
            id="toolu_123",
            name="WebSearch",  # CamelCase as some models output
            input={"query": "test query"},
        )
        tool_use_upper = ToolUseBlock(
            id="toolu_456",
            name="WEB_SEARCH",
            input={"query": "test query"},
        )

        # All variations should be detected as web_search
        assert interceptor.is_web_search_server_tool_use(tool_use_camel_case) is True
        assert interceptor.is_web_search_server_tool_use(tool_use_upper) is True

    @pytest.mark.asyncio
    async def test_execute_search(self, interceptor):
        """Test executing a search."""
        from llm_proxy.web_search import WebSearchExecutionResult

        tool_use = ServerToolUseBlock(
            id="srvtoolu_123",
            name="web_search",
            input={"query": "test query"},
        )

        result = await interceptor.execute_search(tool_use)

        assert isinstance(result, WebSearchExecutionResult)
        assert result.web_search_count == 1
        assert result.tool_use_block == tool_use
        assert isinstance(result.result_block, WebSearchToolResultBlock)
        assert result.result_block.tool_use_id == "srvtoolu_123"
        assert result.result_block.is_error is False
        assert isinstance(result.result_block.content, list)
        assert len(result.result_block.content) == 2

    @pytest.mark.asyncio
    async def test_execute_search_missing_query(self, interceptor):
        """Test search with missing query."""
        tool_use = ServerToolUseBlock(
            id="srvtoolu_123",
            name="web_search",
            input={},
        )

        result = await interceptor.execute_search(tool_use)

        assert result.web_search_count == 0
        assert result.result_block.is_error is True
        # Content is a JSON string for errors
        import orjson

        error_dict = orjson.loads(result.result_block.content)
        assert error_dict["type"] == "web_search_tool_result_error"
        assert error_dict["error_code"] == "invalid_input"

    @pytest.mark.asyncio
    async def test_execute_search_max_uses_exceeded(self, interceptor):
        """Test max_uses enforcement."""
        tool_use = ServerToolUseBlock(
            id="srvtoolu_123",
            name="web_search",
            input={"query": "test query"},
        )
        tool_config = WebSearchToolConfig(max_uses=1)
        search_state = {"count": 0}

        # First search should succeed
        result1 = await interceptor.execute_search(
            tool_use, tool_config, "request_1", search_state=search_state
        )
        assert result1.result_block.is_error is False
        assert result1.web_search_count == 1

        # Second search should fail due to max_uses
        result2 = await interceptor.execute_search(
            tool_use, tool_config, "request_1", search_state=search_state
        )
        assert result2.result_block.is_error is True
        assert result2.web_search_count == 0
        # Content is a JSON string for errors
        import orjson

        error_dict = orjson.loads(result2.result_block.content)
        assert error_dict["error_code"] == "max_uses_exceeded"

    @pytest.mark.asyncio
    async def test_inject_results_into_response(self, interceptor):
        """Test injecting results into a response."""
        tool_use = ServerToolUseBlock(
            id="srvtoolu_123",
            name="web_search",
            input={"query": "test query"},
        )

        response = InternalResponse(
            id="msg_123",
            model="test-model",
            output=[tool_use],
        )

        modified, _ = await interceptor.inject_results_into_response(response)

        # Should have original tool_use block plus result block
        assert len(modified.output) == 2
        assert isinstance(modified.output[1], WebSearchToolResultBlock)

    @pytest.mark.asyncio
    async def test_inject_results_no_web_search_blocks(self, interceptor):
        """Test with no web_search blocks in response."""
        response = InternalResponse(
            id="msg_123",
            model="test-model",
            output=[],
        )

        modified, _ = await interceptor.inject_results_into_response(response)

        assert modified is response  # Should return same object

    @pytest.mark.asyncio
    async def test_inject_results_updates_usage(self, interceptor):
        """Test that usage is tracked in provider_info."""
        tool_use = ServerToolUseBlock(
            id="srvtoolu_123",
            name="web_search",
            input={"query": "test query"},
        )

        response = InternalResponse(
            id="msg_123",
            model="test-model",
            output=[tool_use],
        )

        modified, _ = await interceptor.inject_results_into_response(response)

        # Should have usage tracking in provider_info
        assert "server_tool_use" in modified.provider_info
        assert modified.provider_info["server_tool_use"]["web_search_requests"] == 1

    @pytest.mark.asyncio
    async def test_inject_results_aggregates_multiple_searches(self, interceptor):
        """Test that multiple searches are aggregated correctly."""
        tool_use1 = ServerToolUseBlock(
            id="srvtoolu_1",
            name="web_search",
            input={"query": "query 1"},
        )
        tool_use2 = ServerToolUseBlock(
            id="srvtoolu_2",
            name="web_search",
            input={"query": "query 2"},
        )

        response = InternalResponse(
            id="msg_123",
            model="test-model",
            output=[tool_use1, tool_use2],
        )

        modified, _ = await interceptor.inject_results_into_response(response)

        # Should aggregate search counts
        assert modified.provider_info["server_tool_use"]["web_search_requests"] == 2

    @pytest.mark.asyncio
    async def test_inject_results_converts_tool_use_to_server_tool_use(self, interceptor):
        """ToolUseBlock for web_search must be converted to ServerToolUseBlock.

        Non-Anthropic providers return ToolUseBlock for web_search calls.
        When injected into the response (destined for Anthropic protocol),
        they must become ServerToolUseBlock so the client receives the
        correct 'server_tool_use' content block type, not 'tool_use'.
        """
        tool_use = ToolUseBlock(
            id="toolu_ws_001",
            name="web_search",
            input={"query": "test query"},
        )

        response = InternalResponse(
            id="msg_001",
            model="test-model",
            output=[tool_use],
        )

        modified, _ = await interceptor.inject_results_into_response(response)

        assert len(modified.output) == 2
        assert isinstance(modified.output[0], ServerToolUseBlock)
        assert modified.output[0].id == "toolu_ws_001"
        assert modified.output[0].name == "web_search"
        assert modified.output[0].input == {"query": "test query"}
        assert isinstance(modified.output[1], WebSearchToolResultBlock)
        assert modified.output[1].tool_use_id == "toolu_ws_001"

    @pytest.mark.asyncio
    async def test_inject_results_preserves_server_tool_use(self, interceptor):
        """ServerToolUseBlock should remain as-is (no double conversion)."""
        server_tool_use = ServerToolUseBlock(
            id="srvtoolu_001",
            name="web_search",
            input={"query": "test query"},
        )

        response = InternalResponse(
            id="msg_001",
            model="test-model",
            output=[server_tool_use],
        )

        modified, _ = await interceptor.inject_results_into_response(response)

        assert len(modified.output) == 2
        assert isinstance(modified.output[0], ServerToolUseBlock)
        assert modified.output[0] is server_tool_use


class TestWebSearchStreamProcessor:
    """Tests for WebSearchStreamProcessor."""

    def _make_transformer(self):
        from llm_proxy.protocols.anthropic.streaming import AnthropicStreamingTransformer

        return AnthropicStreamingTransformer(
            model="claude-3-5-sonnet",
            request_id="msg_test123",
        )

    def _make_mock_interceptor(self):
        """Create a mock interceptor with a working mock provider."""
        interceptor = WebSearchInterceptor(MockWebSearchProvider())
        return interceptor

    @pytest.mark.asyncio
    async def test_no_duplicate_server_tool_use_events(self):
        """process_streaming_web_search must NOT emit server_tool_use SSE events.

        The original tool_use blocks from the provider stream already serve
        as the tool call indicator. Only web_search_tool_result events should
        be emitted to prevent the 'Duplicate value for tool_call_id' error.
        """
        from llm_proxy.core.processing.web_search_streaming import WebSearchStreamProcessor

        transformer = self._make_transformer()
        interceptor = self._make_mock_interceptor()
        processor = WebSearchStreamProcessor()

        transformer._accumulated_output = [
            ToolUseBlock(
                id="toolu_ws_001",
                name="web_search",
                input={"query": "test query"},
            ),
        ]

        events, _, _ = await processor.process_streaming_web_search(transformer, interceptor, None)

        assert events is not None
        assert "event: content_block_start" in events
        assert "event: content_block_stop" in events
        assert "web_search_tool_result" in events
        assert '"type":"server_tool_use"' not in events

    @pytest.mark.asyncio
    async def test_replaces_tool_use_with_server_tool_use_in_accumulated(self):
        """ToolUseBlock in accumulated_output must be replaced with ServerToolUseBlock.

        After processing, the accumulated output should contain ServerToolUseBlock
        so the finalizer has the correct block type for usage tracking.
        """
        from llm_proxy.core.processing.web_search_streaming import WebSearchStreamProcessor

        transformer = self._make_transformer()
        interceptor = self._make_mock_interceptor()
        processor = WebSearchStreamProcessor()

        transformer._accumulated_output = [
            ToolUseBlock(
                id="toolu_ws_001",
                name="web_search",
                input={"query": "test query"},
            ),
        ]

        await processor.process_streaming_web_search(transformer, interceptor, None)

        assert len(transformer._accumulated_output) == 1
        assert isinstance(transformer._accumulated_output[0], ServerToolUseBlock)
        assert transformer._accumulated_output[0].id == "toolu_ws_001"

    @pytest.mark.asyncio
    async def test_multiple_web_searches_no_duplicates(self):
        """Multiple web_search calls must each produce unique tool_call_ids."""
        from llm_proxy.core.processing.web_search_streaming import WebSearchStreamProcessor

        transformer = self._make_transformer()
        interceptor = self._make_mock_interceptor()
        processor = WebSearchStreamProcessor()

        transformer._accumulated_output = [
            ToolUseBlock(
                id="toolu_ws_001",
                name="web_search",
                input={"query": "query 1"},
            ),
            ToolUseBlock(
                id="toolu_ws_002",
                name="web_search",
                input={"query": "query 2"},
            ),
        ]

        events, _, _ = await processor.process_streaming_web_search(transformer, interceptor, None)

        assert events is not None
        assert '"type":"server_tool_use"' not in events
        assert "toolu_ws_001" in events
        assert "toolu_ws_002" in events
        assert "web_search_tool_result" in events
        assert isinstance(transformer._accumulated_output[0], ServerToolUseBlock)
        assert isinstance(transformer._accumulated_output[1], ServerToolUseBlock)

    @pytest.mark.asyncio
    async def test_no_web_search_blocks_returns_none(self):
        """When no web_search blocks exist, process_streaming_web_search returns None."""
        from llm_proxy.core.processing.web_search_streaming import WebSearchStreamProcessor

        transformer = self._make_transformer()
        interceptor = self._make_mock_interceptor()
        processor = WebSearchStreamProcessor()

        transformer._accumulated_output = []

        result, results, _ = await processor.process_streaming_web_search(
            transformer, interceptor, None
        )

        assert result is None
        assert results == []

    def test_needs_continuation_true_with_only_web_search(self):
        """needs_continuation True when only web_search blocks present."""
        from llm_proxy.core.processing.web_search_streaming import WebSearchStreamProcessor

        processor = WebSearchStreamProcessor()
        accumulated = [
            ToolUseBlock(id="toolu_1", name="web_search", input={"query": "q1"}),
        ]
        assert processor.needs_continuation(accumulated) is True

    def test_needs_continuation_false_with_trailing_text(self):
        """needs_continuation False when text follows last web_search."""
        from llm_proxy.core.processing.web_search_streaming import WebSearchStreamProcessor

        processor = WebSearchStreamProcessor()
        accumulated = [
            ToolUseBlock(id="toolu_1", name="web_search", input={"query": "q1"}),
            TextBlock(text="Here are the results..."),
        ]
        assert processor.needs_continuation(accumulated) is False

    def test_needs_continuation_false_with_no_web_search(self):
        """needs_continuation False when no web_search blocks."""
        from llm_proxy.core.processing.web_search_streaming import WebSearchStreamProcessor

        processor = WebSearchStreamProcessor()
        accumulated = [TextBlock(text="Hello")]
        assert processor.needs_continuation(accumulated) is False

    def test_needs_continuation_true_with_text_before_web_search(self):
        """needs_continuation True when text precedes web_search but none follows."""
        from llm_proxy.core.processing.web_search_streaming import WebSearchStreamProcessor

        processor = WebSearchStreamProcessor()
        accumulated = [
            TextBlock(text="Let me search..."),
            ToolUseBlock(id="toolu_1", name="web_search", input={"query": "q1"}),
        ]
        assert processor.needs_continuation(accumulated) is True

    def test_build_continuation_request_includes_context(self):
        """build_continuation_request prepends orig msgs, appends assistant+user."""
        from llm_proxy.core.processing.web_search_streaming import WebSearchStreamProcessor
        from llm_proxy.models.conversation import ConversationContext

        processor = WebSearchStreamProcessor()
        interceptor = self._make_mock_interceptor()

        orig_request = InternalRequest(
            model="claude-3-5-sonnet",
            conversation=ConversationContext(
                system_messages=[],
                messages=[Message(role="user", content=[TextBlock(text="Search for X")])],
            ),
        )

        accumulated = [
            ToolUseBlock(id="toolu_ws_001", name="web_search", input={"query": "X"}),
        ]

        server_tool_use = ServerToolUseBlock(
            id="toolu_ws_001", name="web_search", input={"query": "X"}
        )
        from llm_proxy.web_search.provider import WebSearchExecutionResult

        ws_result = WebSearchExecutionResult(
            tool_use_block=server_tool_use,
            result_block=WebSearchToolResultBlock(
                tool_use_id="toolu_ws_001",
                content=[
                    {
                        "type": "web_search_result",
                        "url": "https://example.com",
                        "title": "Example",
                        "encoded_content": interceptor._encode_content("result snippet"),
                    }
                ],
                is_error=False,
            ),
            web_search_count=1,
        )

        continuation_req = processor.build_continuation_request(
            original_request=orig_request,
            accumulated_output=accumulated,
            search_results=[(server_tool_use, ws_result)],
            web_search_interceptor=interceptor,
        )

        messages = continuation_req.conversation.messages
        assert len(messages) == 3
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"
        assert len(messages[1].content) == 1
        assert isinstance(messages[1].content[0], ToolUseBlock)
        assert messages[1].content[0].name == "web_search"
        assert messages[2].role == "tool"
        assert len(messages[2].content) == 1
        assert isinstance(messages[2].content[0], ToolResultBlock)

    @pytest.mark.asyncio
    async def test_build_continuation_request_accumulates_across_rounds(self):
        """build_continuation_request accumulates all prior rounds when called sequentially."""
        from llm_proxy.core.processing.web_search_streaming import WebSearchStreamProcessor
        from llm_proxy.models.conversation import ConversationContext

        processor = WebSearchStreamProcessor()
        interceptor = self._make_mock_interceptor()

        orig_request = InternalRequest(
            model="claude-3-5-sonnet",
            conversation=ConversationContext(
                system_messages=[],
                messages=[Message(role="user", content=[TextBlock(text="Do research")])],
            ),
        )

        # Round 1: model issues web_search query
        accumulated_1 = [
            ToolUseBlock(id="toolu_1", name="web_search", input={"query": "query 1"}),
        ]
        server_tool_1 = ServerToolUseBlock(
            id="toolu_1", name="web_search", input={"query": "query 1"}
        )
        from llm_proxy.web_search.provider import WebSearchExecutionResult

        ws_result_1 = WebSearchExecutionResult(
            tool_use_block=server_tool_1,
            result_block=WebSearchToolResultBlock(
                tool_use_id="toolu_1",
                content=[
                    {
                        "type": "web_search_result",
                        "url": "https://example.com/1",
                        "title": "Result 1",
                        "encoded_content": interceptor._encode_content("snippet 1"),
                    }
                ],
                is_error=False,
            ),
            web_search_count=1,
        )

        cont_req_1 = processor.build_continuation_request(
            original_request=orig_request,
            accumulated_output=accumulated_1,
            search_results=[(server_tool_1, ws_result_1)],
            web_search_interceptor=interceptor,
        )

        # Round 2: model issues another web_search query (simulating the fix:
        # using the previous continuation request as the base)
        accumulated_2 = [
            TextBlock(text="Based on that..."),
            ToolUseBlock(id="toolu_2", name="web_search", input={"query": "query 2"}),
        ]
        server_tool_2 = ServerToolUseBlock(
            id="toolu_2", name="web_search", input={"query": "query 2"}
        )
        ws_result_2 = WebSearchExecutionResult(
            tool_use_block=server_tool_2,
            result_block=WebSearchToolResultBlock(
                tool_use_id="toolu_2",
                content=[
                    {
                        "type": "web_search_result",
                        "url": "https://example.com/2",
                        "title": "Result 2",
                        "encoded_content": interceptor._encode_content("snippet 2"),
                    }
                ],
                is_error=False,
            ),
            web_search_count=1,
        )

        cont_req_2 = processor.build_continuation_request(
            original_request=cont_req_1,  # Accumulated context
            accumulated_output=accumulated_2,
            search_results=[(server_tool_2, ws_result_2)],
            web_search_interceptor=interceptor,
        )

        messages = cont_req_2.conversation.messages
        # original user + assistant(q1) + user(r1) + assistant(TextBlock, q2) + user(r2)
        assert len(messages) == 5, f"Expected 5 messages, got {len(messages)}"

        assert messages[0].role == "user"
        assert messages[1].role == "assistant"
        assert isinstance(messages[1].content[0], ToolUseBlock)
        assert messages[1].content[0].name == "web_search"
        assert messages[2].role == "tool"
        assert isinstance(messages[2].content[0], ToolResultBlock)

        assert messages[3].role == "assistant"
        assert len(messages[3].content) == 2
        assert isinstance(messages[3].content[0], TextBlock)
        assert isinstance(messages[3].content[1], ToolUseBlock)

        assert messages[4].role == "tool"
        assert isinstance(messages[4].content[0], ToolResultBlock)

    def test_build_continuation_excludes_non_web_search_tool_calls(self):
        """Non-web_search tool calls are excluded from the assistant message.

        If the model called both get_weather and web_search in the same turn,
        the continuation must only include web_search in the assistant message
        because only web_search has a matching tool result. Including the
        non-web_search tool call would produce an invalid conversation where
        tool_calls outnumber tool messages — strict providers like DeepSeek
        reject this with:
          "An assistant message with 'tool_calls' must be followed by
           tool messages responding to each 'tool_call_id'."
        """
        from llm_proxy.core.processing.web_search_streaming import WebSearchStreamProcessor
        from llm_proxy.models.conversation import ConversationContext

        processor = WebSearchStreamProcessor()
        interceptor = self._make_mock_interceptor()

        orig_request = InternalRequest(
            model="deepseek-chat",
            conversation=ConversationContext(
                system_messages=[],
                messages=[
                    Message(
                        role="user",
                        content=[TextBlock(text="check weather and search news")],
                    )
                ],
            ),
        )

        # Simulate accumulated output after process_streaming_web_search:
        # web_search ToolUseBlock was replaced with ServerToolUseBlock,
        # but get_weather ToolUseBlock remains.
        server_tool_ws = ServerToolUseBlock(
            id="call_ws", name="web_search", input={"query": "AI news"}
        )
        accumulated = [
            TextBlock(text="Let me check both weather and news"),
            ToolUseBlock(
                id="call_weather",
                name="get_weather",
                input={"location": "Tokyo"},
            ),
            server_tool_ws,
        ]
        from llm_proxy.web_search.provider import WebSearchExecutionResult

        ws_result = WebSearchExecutionResult(
            tool_use_block=server_tool_ws,
            result_block=WebSearchToolResultBlock(
                tool_use_id="call_ws",
                content=[
                    {
                        "type": "web_search_result",
                        "url": "https://example.com",
                        "title": "AI News",
                        "encoded_content": interceptor._encode_content("latest AI news"),
                    }
                ],
                is_error=False,
            ),
            web_search_count=1,
        )

        continuation_req = processor.build_continuation_request(
            original_request=orig_request,
            accumulated_output=accumulated,
            search_results=[(server_tool_ws, ws_result)],
            web_search_interceptor=interceptor,
        )

        messages = continuation_req.conversation.messages
        # original user + assistant(TextBlock + web_search only) + tool_result
        assert len(messages) == 3, f"Expected 3 messages, got {len(messages)}"

        # Assistant message must contain text and ONLY web_search tool call
        assert messages[1].role == "assistant"
        assert len(messages[1].content) == 2
        assert isinstance(messages[1].content[0], TextBlock)
        assert messages[1].content[0].text == "Let me check both weather and news"
        assert isinstance(messages[1].content[1], ToolUseBlock)
        assert messages[1].content[1].name == "web_search"
        # Non-web_search tool call must NOT be present
        tool_names = [b.name for b in messages[1].content if isinstance(b, ToolUseBlock)]
        assert tool_names == ["web_search"], f"Only web_search expected, got: {tool_names}"

        # Tool result must be present
        assert messages[2].role == "tool"
        assert isinstance(messages[2].content[0], ToolResultBlock)
        assert messages[2].content[0].tool_use_id == "call_ws"
