"""Tests for OpenAI adapter web search built-in tool serialization."""

from llm_proxy.models import (
    ConversationContext,
    InternalRequest,
    Message,
    TextBlock,
)
from llm_proxy.models.params import GenerationParams
from llm_proxy.models.tools import OpenAIToolSearchTool, OpenAIWebSearchTool
from llm_proxy.serialization.openai.serializer import OpenAIResponsesProviderSerializer


class TestOpenAIWebSearchToolSerialization:
    """Test web_search built-in tool conversion to OpenAI Responses API."""

    _serializer = OpenAIResponsesProviderSerializer()

    def test_build_tools_with_openai_web_search(self):
        request = InternalRequest(
            model="gpt-4o-mini",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Search for news")])]
            ),
            params=GenerationParams(),
            tools=[
                OpenAIWebSearchTool(
                    search_context_size="high",
                    allowed_domains=["example.com"],
                    user_location={
                        "type": "approximate",
                        "country": "US",
                        "city": "New York",
                    },
                ),
            ],
        )

        body = self._serializer.build_provider_request(request)

        assert "tools" in body
        assert len(body["tools"]) == 1
        tool = body["tools"][0]
        assert tool["type"] == "web_search"
        assert tool["search_context_size"] == "high"
        assert tool["filters"]["allowed_domains"] == ["example.com"]
        assert tool["user_location"]["country"] == "US"
        assert tool["user_location"]["city"] == "New York"

    def test_build_tools_skips_none_fields(self):
        request = InternalRequest(
            model="gpt-4o-mini",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Search")])]
            ),
            params=GenerationParams(),
            tools=[OpenAIWebSearchTool()],
        )

        body = self._serializer.build_provider_request(request)

        assert body["tools"][0] == {"type": "web_search"}


class TestOpenAIToolSearchToolSerialization:
    """Test tool_search built-in tool conversion to OpenAI Responses API.

    Regression: ``OpenAIToolSearchTool`` was silently dropped by
    ``_build_tools`` (no matching branch), so /v1/responses requests routing
    to the OpenAI provider never exposed the tool_search tool to the model.
    """

    _serializer = OpenAIResponsesProviderSerializer()

    def test_build_tools_emits_tool_search(self):
        request = InternalRequest(
            model="gpt-5",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="find shell tools")])]
            ),
            params=GenerationParams(),
            tools=[OpenAIToolSearchTool()],
        )

        body = self._serializer.build_provider_request(request)

        assert "tools" in body
        assert len(body["tools"]) == 1
        assert body["tools"][0] == {"type": "tool_search"}

    def test_build_tools_keeps_tool_search_alongside_function(self):
        from llm_proxy.models import FunctionTool

        request = InternalRequest(
            model="gpt-5",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            params=GenerationParams(),
            tools=[
                OpenAIToolSearchTool(),
                FunctionTool(name="exec_command", parameters={"type": "object"}),
            ],
        )

        body = self._serializer.build_provider_request(request)

        tool_types = [t["type"] for t in body["tools"]]
        assert tool_types == ["tool_search", "function"]

    def test_parse_provider_response_web_search_call(self):
        response = {
            "id": "resp_123",
            "model": "gpt-4o-mini",
            "status": "completed",
            "output": [
                {
                    "type": "web_search_call",
                    "id": "ws_abc",
                    "status": "completed",
                    "action": {"type": "search", "query": "latest AI news"},
                }
            ],
        }
        llm_response = self._serializer.parse_provider_response(response, model="gpt-4o-mini")

        assert len(llm_response.output) == 1
        block = llm_response.output[0]
        assert block.name == "web_search"
        assert block.input == {"query": "latest AI news"}
