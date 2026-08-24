# tests/unit/models/test_tools.py
"""Tests for ToolDefinition types in unified protocol format."""

from llm_proxy.models.tools import (
    CodeExecutionTool,
    FunctionTool,
    ToolChoice,
    ToolChoiceFunction,
    ToolChoiceSpec,
    ToolDefinition,
)


class TestFunctionTool:
    def test_create_minimal(self):
        tool = FunctionTool(name="get_weather", parameters={"type": "object", "properties": {}})
        assert tool.name == "get_weather"
        assert tool.parameters == {"type": "object", "properties": {}}
        assert tool.description is None
        assert tool.strict is False

    def test_create_with_all_fields(self):
        tool = FunctionTool(
            name="get_weather",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}},
            description="Get weather for a city",
            strict=True,
        )
        assert tool.name == "get_weather"
        assert tool.parameters == {"type": "object", "properties": {"city": {"type": "string"}}}
        assert tool.description == "Get weather for a city"
        assert tool.strict is True


class TestCodeExecutionTool:
    def test_create_minimal(self):
        tool = CodeExecutionTool()
        assert tool.type == "code_execution_20250522"
        assert tool.name == "code_execution"

    def test_create_with_type(self):
        tool = CodeExecutionTool(type="code_execution_20250825")
        assert tool.type == "code_execution_20250825"


class TestToolDefinition:
    def test_function_tool_is_valid(self):
        tool: ToolDefinition = FunctionTool(
            name="test",
            parameters={"type": "object", "properties": {}},
        )
        assert isinstance(tool, FunctionTool)

    def test_code_execution_tool_is_valid(self):
        tool: ToolDefinition = CodeExecutionTool()
        assert isinstance(tool, CodeExecutionTool)


class TestToolChoice:
    def test_create_auto(self):
        choice = ToolChoice(mode="auto")
        assert choice.mode == "auto"

    def test_create_none(self):
        choice = ToolChoice(mode="none")
        assert choice.mode == "none"

    def test_create_required(self):
        choice = ToolChoice(mode="required")
        assert choice.mode == "required"


class TestToolChoiceFunction:
    def test_create(self):
        choice = ToolChoiceFunction(name="get_weather")
        assert choice.type == "function"
        assert choice.name == "get_weather"


class TestToolChoiceSpec:
    def test_accepts_tool_choice_auto(self):
        spec: ToolChoiceSpec = ToolChoice(mode="auto")
        assert isinstance(spec, ToolChoice)

    def test_accepts_tool_choice_none(self):
        spec: ToolChoiceSpec = ToolChoice(mode="none")
        assert isinstance(spec, ToolChoice)

    def test_accepts_tool_choice_required(self):
        spec: ToolChoiceSpec = ToolChoice(mode="required")
        assert isinstance(spec, ToolChoice)

    def test_accepts_tool_choice_function(self):
        spec: ToolChoiceSpec = ToolChoiceFunction(name="get_weather")
        assert isinstance(spec, ToolChoiceFunction)

    def test_accepts_string(self):
        spec: ToolChoiceSpec = "auto"
        assert spec == "auto"

    def test_accepts_function_name_string(self):
        spec: ToolChoiceSpec = "get_weather"
        assert spec == "get_weather"
