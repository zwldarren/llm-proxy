"""Tests for OpenAI adapter web search built-in tool serialization."""

from llm_proxy.models import (
    ConversationContext,
    InternalRequest,
    Message,
    TextBlock,
)
from llm_proxy.models.params import GenerationParams, OpenAISpecificParams
from llm_proxy.models.tools import (
    AnthropicWebSearchTool,
    FunctionTool,
    OpenAIToolSearchTool,
    OpenAIWebSearchTool,
    UserLocation,
)
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


class TestAnthropicWebSearchToolSerialization:
    """A Messages-protocol client's hosted search must not vanish here.

    Regression: ``_build_tools`` matched only ``OpenAIWebSearchTool``, so a
    Claude Code client (``web_search_20250305``) routed to an OpenAI model lost
    web search entirely — the built body carried no ``tools`` key at all.
    """

    _serializer = OpenAIResponsesProviderSerializer()

    def test_anthropic_web_search_becomes_the_responses_hosted_tool(self):
        request = InternalRequest(
            model="gpt-5.6",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="search")])]
            ),
            params=GenerationParams(),
            tools=[
                AnthropicWebSearchTool(
                    allowed_domains=["good.example"],
                    blocked_domains=["bad.example"],
                    user_location=UserLocation(country="US", city="New York"),
                )
            ],
        )

        body = self._serializer.build_provider_request(request)

        assert body["tools"] == [
            {
                "type": "web_search",
                "filters": {
                    "allowed_domains": ["good.example"],
                    "blocked_domains": ["bad.example"],
                },
                "user_location": {"type": "approximate", "country": "US", "city": "New York"},
            }
        ]

    def test_max_uses_has_no_responses_equivalent(self):
        request = InternalRequest(
            model="gpt-5.6",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="search")])]
            ),
            params=GenerationParams(),
            tools=[AnthropicWebSearchTool(max_uses=3, response_inclusion="excluded")],
        )

        body = self._serializer.build_provider_request(request)

        assert body["tools"] == [{"type": "web_search"}]


class TestWebSearchOptionsTranslation:
    """``web_search_options`` is a Chat Completions field; this wire is Responses.

    Regression: the option was dropped with the rest of the unknown Chat
    Completions fields, so a client asking for web search reached OpenAI with no
    search at all whenever the proxy had no interceptor of its own to fall back
    on (``WebSearchStage`` only translates while an interceptor exists).
    """

    _serializer = OpenAIResponsesProviderSerializer()

    def _request(self, **params):
        return InternalRequest(
            model="gpt-5.6",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="search")])]
            ),
            params=GenerationParams(
                openai=OpenAISpecificParams(web_search_options=params.get("options"))
            ),
        )

    def test_options_dict_becomes_a_hosted_tool(self):
        body = self._serializer.build_provider_request(
            self._request(options={"search_context_size": "low"})
        )

        assert body["tools"] == [{"type": "web_search", "search_context_size": "low"}]

    def test_bare_true_uses_defaults(self):
        body = self._serializer.build_provider_request(self._request(options=True))

        assert body["tools"] == [{"type": "web_search"}]

    def test_interception_bridge_is_not_duplicated(self):
        """The proxy's own ``web_search`` function tool means search is owned
        by the interceptor; adding the hosted tool back would search upstream too."""
        request = self._request(options={"search_context_size": "low"})
        request.tools = [
            FunctionTool(name="web_search", parameters={"type": "object", "properties": {}})
        ]

        body = self._serializer.build_provider_request(request)

        assert body["tools"] == [body["tools"][0]]
        assert body["tools"][0]["type"] == "function"


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
