"""Regression tests for ``/v1/responses`` ``tool_choice`` parsing.

The ``ResponsesRequest`` schema parses ``tool_choice`` into pydantic models
(``FunctionToolChoice`` / ``AllowedToolChoice``). Previously ``_parse_tool_choice``
only handled ``str`` / ``dict`` and silently returned ``None`` for pydantic
objects, dropping a user's forced function tool choice. These tests guard the
fix across pydantic-object, dict (Responses + chat-completions shapes) and
string inputs, plus the end-to-end OpenAI provider body output.
"""

import pytest

from llm_proxy.models import ToolChoice, ToolChoiceFunction
from llm_proxy.models.tools import ToolChoiceAllowedTools, ToolChoiceCustom
from llm_proxy.protocols.openresponses import (
    OpenResponsesProtocolSerializer,
)
from llm_proxy.protocols.openresponses.schemas import (
    AllowedToolChoice,
    FunctionToolChoice,
    ResponsesRequest,
)
from llm_proxy.protocols.openresponses.serializer import (
    _parse_tool_choice,
)
from llm_proxy.serialization.context import BuildContext
from llm_proxy.serialization.openai.serializer import OpenAIResponsesProviderSerializer

# ---------------------------------------------------------------------------
# _parse_tool_choice: unit-level coverage
# ---------------------------------------------------------------------------


class TestParseToolChoice:
    def test_function_pydantic_model_is_not_dropped(self):
        """A pydantic ``FunctionToolChoice`` (parsed by the schema) must become a
        ``ToolChoiceFunction`` carrying the function name, not ``None``."""
        result = _parse_tool_choice(FunctionToolChoice(type="function", name="get"))
        assert isinstance(result, ToolChoiceFunction)
        assert result.name == "get"

    def test_function_responses_dict_shape(self):
        """The Responses API wire shape ``{type:function, name}`` (no nested
        ``function`` object) must be parsed correctly."""
        result = _parse_tool_choice({"type": "function", "name": "get"})
        assert isinstance(result, ToolChoiceFunction)
        assert result.name == "get"

    def test_function_chat_completions_dict_shape_still_supported(self):
        """The chat completions shape ``{type:function, function:{name}}`` must
        still be parsed (backwards compatibility for other call sites)."""
        result = _parse_tool_choice({"type": "function", "function": {"name": "get"}})
        assert isinstance(result, ToolChoiceFunction)
        assert result.name == "get"

    @pytest.mark.parametrize("mode", ["auto", "any", "none", "required"])
    def test_string_modes(self, mode):
        result = _parse_tool_choice(mode)
        assert isinstance(result, ToolChoice)
        assert result.mode == mode

    def test_allowed_tools_pydantic_model_single_tool(self):
        """An ``AllowedToolChoice`` pydantic object with a single tool collapses to
        a ``ToolChoiceFunction`` carrying that tool's name."""
        result = _parse_tool_choice(
            AllowedToolChoice(
                type="allowed_tools",
                mode="auto",
                tools=[FunctionToolChoice(type="function", name="get")],
            )
        )
        assert isinstance(result, ToolChoiceFunction)
        assert result.name == "get"

    def test_allowed_tools_dict_multiple_tools(self):
        """Multiple allowed tools are preserved as a ``ToolChoiceAllowedTools``."""
        result = _parse_tool_choice(
            {
                "type": "allowed_tools",
                "mode": "required",
                "tools": [
                    {"type": "function", "name": "get"},
                    {"type": "function", "name": "put"},
                ],
            }
        )
        assert isinstance(result, ToolChoiceAllowedTools)
        assert result.allowed_tools.mode == "required"
        assert len(result.allowed_tools.tools) == 2

    def test_allowed_tools_dict_empty_tools_with_mode(self):
        """An ``allowed_tools`` choice with a plain mode and no tools degrades to
        a ``ToolChoice`` mode."""
        result = _parse_tool_choice({"type": "allowed_tools", "mode": "auto"})
        assert isinstance(result, ToolChoice)
        assert result.mode == "auto"

    def test_custom_dict_shape(self):
        result = _parse_tool_choice({"type": "custom", "name": "patch"})
        assert isinstance(result, ToolChoiceCustom)
        assert result.name == "patch"

    def test_none_returns_none(self):
        assert _parse_tool_choice(None) is None


# ---------------------------------------------------------------------------
# End-to-end: parse_request -> OpenAI provider body (responses endpoint)
# ---------------------------------------------------------------------------


_proto_serializer = OpenResponsesProtocolSerializer()
_provider_serializer = OpenAIResponsesProviderSerializer()


def _openai_body(request: ResponsesRequest) -> dict:
    internal = _proto_serializer.parse_request(request.model_dump())
    ctx = BuildContext(
        provider_name="openai",
        model=request.model,
        target_endpoint="responses",
        supported_content_blocks=_provider_serializer.supported_content_blocks,
    )
    return _provider_serializer._build_provider_request(internal, ctx)


class TestToolChoiceRoundtrip:
    def test_function_tool_choice_reaches_provider_body(self):
        """Forced function tool choice must survive into the OpenAI Responses
        provider body instead of being silently dropped."""
        request = ResponsesRequest(
            model="gpt-5",
            input="hi",
            tools=[
                {
                    "type": "function",
                    "name": "get",
                    "description": "d",
                    "parameters": {"type": "object"},
                }
            ],
            tool_choice={"type": "function", "name": "get"},
        )
        body = _openai_body(request)
        assert body["tool_choice"] == {"type": "function", "name": "get"}

    def test_auto_string_tool_choice_reaches_provider_body(self):
        request = ResponsesRequest(
            model="gpt-5",
            input="hi",
            tools=[
                {
                    "type": "function",
                    "name": "get",
                    "parameters": {"type": "object"},
                }
            ],
            tool_choice="auto",
        )
        body = _openai_body(request)
        assert body["tool_choice"] == "auto"

    def test_function_tool_choice_with_pydantic_model_input(self):
        """Passing a pydantic ``FunctionToolChoice`` directly (as the schema would
        produce internally) must also reach the provider body."""
        request = ResponsesRequest(
            model="gpt-5",
            input="hi",
            tools=[
                {
                    "type": "function",
                    "name": "get",
                    "parameters": {"type": "object"},
                }
            ],
            tool_choice=FunctionToolChoice(type="function", name="get"),
        )
        body = _openai_body(request)
        assert body["tool_choice"] == {"type": "function", "name": "get"}
