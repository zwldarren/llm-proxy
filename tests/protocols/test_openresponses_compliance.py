"""Compliance-focused tests for OpenResponses API against the Open Responses specification.

These tests verify that the implementation correctly handles the six compliance scenarios:
1. Basic Text Response
2. Streaming Response
3. System Prompt
4. Tool Calling
5. Image Input
6. Multi-turn Conversation
"""

import sys

import orjson

from llm_proxy.models import (
    CustomTool,
    InternalResponse,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
)
from llm_proxy.models.content_blocks import FileBlock, ImageBlock
from llm_proxy.models.content_blocks.anthropic_builtin import ToolSearchToolResultBlock
from llm_proxy.models.tools.openai_builtin import OpenAIToolSearchTool
from llm_proxy.models.types import ChoiceLogprobs, TokenLogprob, Usage
from llm_proxy.protocols.openresponses import (
    OpenResponsesProtocolSerializer,
)
from llm_proxy.protocols.openresponses.handler import openresponses_protocol
from llm_proxy.protocols.openresponses.schemas import (
    FunctionCallOutputItemParam,
    FunctionToolParam,
    ResponsesRequest,
    ResponsesResponse,
)
from llm_proxy.protocols.openresponses.serializer import (
    _parse_tool,
)
from llm_proxy.protocols.openresponses.streaming import OpenResponsesStreamingTransformer
from llm_proxy.serialization.context import BuildContext
from llm_proxy.serialization.format_context import FormatContext
from llm_proxy.serialization.gemini.serializer import GeminiProviderSerializer
from llm_proxy.serialization.openai.components.request_builder import OpenAIRequestBuilder

_serializer = OpenResponsesProtocolSerializer()


def _build_chat_completions_body(request: ResponsesRequest) -> dict:
    """Parse a Responses request and build the Chat Completions body a
    chat-compatible provider would receive (target_endpoint='chat_completions')."""
    internal = _serializer.parse_request(request.model_dump())
    context = BuildContext(
        stream=request.stream,
        model=request.model,
        provider_name="chat-compatible",
        target_endpoint="chat_completions",
    )
    return OpenAIRequestBuilder().build(internal, context)


class TestBasicTextResponse:
    """Verify a simple user message and validate the ResponseResource schema."""

    def test_simple_string_input(self):
        request = ResponsesRequest(model="gpt-4", input="Hello")
        result = _serializer.parse_request(request.model_dump())
        assert result.model == "gpt-4"
        assert len(result.conversation.messages) == 1
        assert result.conversation.messages[0].role == "user"

    def test_response_schema_compliance(self):
        internal = InternalResponse(
            id="resp_abc123",
            model="gpt-4",
            output=[TextBlock(text="Hello!")],
            usage=Usage(input_tokens=5, output_tokens=2, total_tokens=7),
            finish_reason="stop",
        )
        result = _serializer.format_response(internal)

        # Validate required response fields per spec
        assert result["id"] == "resp_abc123"
        assert result["object"] == "response"
        assert isinstance(result["created_at"], int)
        assert isinstance(result["completed_at"], int)
        assert result["status"] == "completed"
        assert result["model"] == "gpt-4"

        # Validate output structure
        assert len(result["output"]) > 0
        msg = result["output"][0]
        assert msg["type"] == "message"
        assert msg["role"] == "assistant"
        assert msg["status"] == "completed"
        assert len(msg["content"]) > 0
        assert msg["content"][0]["type"] == "output_text"
        assert msg["content"][0]["text"] == "Hello!"

        # Validate usage
        assert result["usage"]["input_tokens"] == 5
        assert result["usage"]["output_tokens"] == 2
        assert result["usage"]["total_tokens"] == 7

    def test_response_validates_against_pydantic_model(self):
        internal = InternalResponse(
            id="resp_abc123",
            model="gpt-4",
            output=[TextBlock(text="Hi")],
            finish_reason="stop",
        )
        result = _serializer.format_response(internal)
        response = ResponsesResponse(**result)
        assert response.id == "resp_abc123"
        assert response.object == "response"
        assert response.status == "completed"

    def test_response_defaults_match_spec(self):
        """Spec-required numbers must default (never null) on the response."""
        internal = InternalResponse(
            id="resp_defaults",
            model="gpt-4",
            output=[TextBlock(text="Hi")],
            finish_reason="stop",
        )
        result = _serializer.format_response(internal)
        # Spec: top_p/temperature are required numbers on ResponseResource.
        assert result["top_p"] == 1.0
        assert result["temperature"] == 1.0
        # Spec: parallel_tool_calls is required and defaults to true.
        assert result["parallel_tool_calls"] is True
        # Spec: usage requires both details objects with their own required
        # inner fields, even when the provider supplied no usage at all.
        assert result["usage"] == {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        }


# =============================================================================
# Scenario 2: Streaming Response
# =============================================================================


class TestStreamingResponse:
    """Check SSE streaming events and ensure the final response is present."""

    def test_streaming_request_sets_stream_flag(self):
        request = ResponsesRequest(model="gpt-4", input="Hello", stream=True)
        result = _serializer.parse_request(request.model_dump())
        assert result.stream is True

    def test_streaming_transformer_available(self):
        transformer = openresponses_protocol.get_streaming_transformer()
        assert transformer is not None

    def test_streaming_response_completed(self):
        """Verify completed streaming response has all required fields."""

        internal = InternalResponse(
            id="resp_stream_1",
            model="gpt-4",
            output=[TextBlock(text="Streamed text")],
            usage=Usage(input_tokens=3, output_tokens=2, total_tokens=5),
            finish_reason="stop",
        )
        result = OpenResponsesProtocolSerializer().format_response(internal)
        assert result["id"] == "resp_stream_1"
        assert result["status"] == "completed"
        assert result["completed_at"] is not None


# =============================================================================
# Scenario 3: System Prompt
# =============================================================================


class TestSystemPrompt:
    """Confirm that a system-role message can be included in the input."""

    def test_system_message_in_input(self):
        request = ResponsesRequest(
            model="gpt-4",
            input=[
                {"type": "message", "role": "system", "content": "You are a helpful assistant"},
                {"type": "message", "role": "user", "content": "Hello"},
            ],
        )
        result = _serializer.parse_request(request.model_dump())
        assert len(result.conversation.system_messages) == 1
        assert result.conversation.system_messages[0].text_content == "You are a helpful assistant"

    def test_system_message_with_list_content(self):
        """System messages can have list content (e.g., multiple text parts)."""
        request = ResponsesRequest(
            model="gpt-4",
            input=[
                {
                    "type": "message",
                    "role": "system",
                    "content": [
                        {"type": "input_text", "text": "Rule 1"},
                        {"type": "input_text", "text": "Rule 2"},
                    ],
                },
                {"type": "message", "role": "user", "content": "Hello"},
            ],
        )
        result = _serializer.parse_request(request.model_dump())
        assert len(result.conversation.system_messages) == 1
        assert "Rule 1" in result.conversation.system_messages[0].text_content
        assert "Rule 2" in result.conversation.system_messages[0].text_content

    def test_instructions_field(self):
        """The `instructions` field should be converted to a system message."""
        request = ResponsesRequest(
            model="gpt-4",
            input="Hello",
            instructions="You are a helpful assistant",
        )
        result = _serializer.parse_request(request.model_dump())
        assert len(result.conversation.system_messages) == 1
        assert result.conversation.system_messages[0].text_content == "You are a helpful assistant"

    def test_instructions_in_response(self):
        """The `instructions` field should be reflected in the response."""
        internal = InternalResponse(
            id="test",
            model="gpt-4",
            output=[TextBlock(text="Hi")],
        )

        result = _serializer.format_response(internal, FormatContext(instructions="Be helpful"))
        assert result["instructions"] == "Be helpful"


# =============================================================================
# Scenario 4: Tool Calling
# =============================================================================


class TestToolCalling:
    """Test the ability to define a function tool and confirm function_call output."""

    def test_flat_tool_definition(self):
        """FunctionToolParam is flat: name/description/parameters at top level."""
        tool = {
            "type": "function",
            "name": "get_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
            },
        }
        result = _parse_tool(tool)
        assert result.name == "get_weather"
        assert result.description == "Get the current weather"
        assert result.parameters["type"] == "object"
        assert result.strict is False  # Default should be False per spec

    def test_nested_tool_definition(self):
        """Some clients/tools still use nested 'function' key format."""
        tool = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather",
                "parameters": {"type": "object"},
            },
        }
        result = _parse_tool(tool)
        assert result.name == "get_weather"
        assert result.description == "Get the current weather"

    def test_tool_calling_request_conversion(self):
        request = ResponsesRequest(
            model="gpt-4",
            input="What's the weather?",
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get the current weather",
                    "parameters": {"type": "object"},
                }
            ],
        )
        result = _serializer.parse_request(request.model_dump())
        assert result.tools is not None
        assert len(result.tools) == 1
        assert result.tools[0].name == "get_weather"

    def test_function_call_in_response(self):
        """Verify function_call output items in the response."""
        internal = InternalResponse(
            id="test",
            model="gpt-4",
            output=[ToolUseBlock(id="call_abc123", name="get_weather", input='{"location": "SF"}')],
            finish_reason="stop",
        )
        result = _serializer.format_response(internal)
        function_calls = [o for o in result["output"] if o["type"] == "function_call"]
        assert len(function_calls) == 1
        assert function_calls[0]["name"] == "get_weather"
        assert function_calls[0]["call_id"] == "call_abc123"
        assert function_calls[0]["arguments"] == '{"location": "SF"}'
        assert function_calls[0]["status"] == "completed"

    def test_builtin_tools_are_skipped(self):
        """Built-in tools (file_search, code_interpreter) should not be forwarded.
        web_search should be parsed into OpenAIWebSearchTool for native passthrough/interception."""
        from llm_proxy.models.tools import OpenAIWebSearchTool

        request = ResponsesRequest(
            model="gpt-4",
            input="Search the web",
            tools=[
                {"type": "web_search"},
                {"type": "file_search"},
                {"type": "code_interpreter"},
                {"type": "function", "name": "my_tool", "description": "My tool"},
            ],
        )
        result = _serializer.parse_request(request.model_dump())
        assert result.tools is not None
        assert len(result.tools) == 2
        assert isinstance(result.tools[0], OpenAIWebSearchTool)
        assert result.tools[1].name == "my_tool"

    def test_function_call_output_output_accepts_string_or_content_items(self):
        """FunctionCallOutputItemParam.output may be a plain string OR an array
        of structured content items (input_text / input_image /
        encrypted_content) per the OpenAI Responses API and Codex wire format."""
        str_item = FunctionCallOutputItemParam(call_id="call_123", output="result string")
        assert isinstance(str_item.output, str)

        array_item = FunctionCallOutputItemParam(
            call_id="call_123",
            output=[
                {"type": "input_text", "text": "command output"},
                {"type": "input_image", "image_url": "data:image/png;base64,..."},
                {"type": "encrypted_content", "encrypted_content": "opaque-blob"},
            ],
        )
        assert isinstance(array_item.output, list)
        assert len(array_item.output) == 3

    def test_codex_item_types_accepted_by_responses_request(self):
        """All Codex /v1/responses input item types must validate (no 422)."""

        items = [
            {
                "type": "local_shell_call",
                "call_id": "c",
                "status": "completed",
                "action": {"type": "exec", "command": ["ls"]},
            },
            {"type": "custom_tool_call", "call_id": "c", "name": "apply_patch", "input": "{}"},
            {"type": "custom_tool_call_output", "call_id": "c", "output": "ok"},
            {"type": "tool_search_call", "call_id": "c", "execution": "sync", "arguments": {}},
            {
                "type": "tool_search_output",
                "call_id": "c",
                "status": "completed",
                "execution": "sync",
                "tools": [],
            },
            {
                "type": "web_search_call",
                "status": "completed",
                "action": {"type": "search", "query": "q"},
            },
            {"type": "image_generation_call", "status": "completed", "result": "img"},
            {
                "type": "agent_message",
                "author": "a",
                "recipient": "b",
                "content": [{"type": "input_text", "text": "hi"}],
            },
            {"type": "additional_tools", "role": "user", "tools": []},
            {"type": "compaction", "encrypted_content": "blob"},
            {"type": "compaction_summary", "encrypted_content": "blob"},
            {"type": "compaction_trigger"},
            {"type": "context_compaction", "encrypted_content": "blob"},
            {"type": "some_future_type", "foo": "bar"},
        ]
        req = ResponsesRequest(model="gpt-4", input=items)
        assert len(req.input) == len(items)

    def test_function_tool_param_strict_default_false(self):
        """FunctionToolParam.strict should default to False per spec."""
        tool = FunctionToolParam(name="test_func")
        assert tool.strict is False

    def test_custom_tool_is_parsed(self):
        """type: 'custom' tool definitions are parsed into CustomTool (flat format)."""

        request = ResponsesRequest(
            model="gpt-4",
            input="Use custom tool",
            tools=[
                {
                    "type": "custom",
                    "name": "apply_patch",
                    "description": "Apply a patch",
                    "format": {
                        "type": "grammar",
                        "grammar": {
                            "definition": "patch-grammar",
                            "syntax": "lark",
                        },
                    },
                }
            ],
        )
        result = _serializer.parse_request(request.model_dump())
        assert result.tools is not None
        assert len(result.tools) == 1
        assert isinstance(result.tools[0], CustomTool)
        assert result.tools[0].name == "apply_patch"
        assert result.tools[0].description == "Apply a patch"
        assert result.tools[0].format_type == "grammar"
        assert result.tools[0].grammar_definition == "patch-grammar"
        assert result.tools[0].grammar_syntax == "lark"

    def test_tool_search_converted_not_dropped(self):
        """type: 'tool_search' is now converted to OpenAIToolSearchTool, not dropped."""

        request = ResponsesRequest(
            model="gpt-4",
            input="Search tools",
            tools=[
                {"type": "tool_search", "description": "Find tools"},
                {"type": "function", "name": "my_tool", "description": "My tool"},
            ],
        )
        result = _serializer.parse_request(request.model_dump())

        assert result.tools is not None
        assert len(result.tools) == 2
        assert isinstance(result.tools[0], OpenAIToolSearchTool)
        assert result.tools[0].type == "tool_search"
        assert result.tools[1].name == "my_tool"


# =============================================================================
# Scenario 5: Image Input
# =============================================================================


class TestImageInput:
    """Validate sending an image URL within user content."""

    def test_image_url_in_input(self):
        request = ResponsesRequest(
            model="gpt-4",
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "What's in this image?"},
                        {
                            "type": "input_image",
                            "image_url": "https://example.com/image.png",
                        },
                    ],
                }
            ],
        )
        result = _serializer.parse_request(request.model_dump())
        assert len(result.conversation.messages) == 1

        blocks = result.conversation.messages[0].content
        text_blocks = [b for b in blocks if isinstance(b, TextBlock)]
        image_blocks = [b for b in blocks if isinstance(b, ImageBlock)]
        assert len(text_blocks) == 1
        assert len(image_blocks) == 1
        assert image_blocks[0].source.type == "url"

    def test_image_base64_in_input(self):
        """Per OpenAI spec, input_image can be a base64 data URI."""
        request = ResponsesRequest(
            model="gpt-4",
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "What's in this image?"},
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64,iVBORw0KGgo=",
                        },
                    ],
                }
            ],
        )
        result = _serializer.parse_request(request.model_dump())

        msg = result.conversation.messages[0]
        image_blocks = [b for b in msg.content if isinstance(b, ImageBlock)]
        assert len(image_blocks) == 1
        assert image_blocks[0].source.type == "base64"
        assert image_blocks[0].source.data == "iVBORw0KGgo="

    def test_file_input_with_file_id(self):
        """Per OpenAI spec, Responses API supports input_file with file_id."""
        request = ResponsesRequest(
            model="gpt-4",
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "What is this?"},
                        {
                            "type": "input_file",
                            "file_id": "file_abc123",
                            "filename": "doc.pdf",
                        },
                    ],
                }
            ],
        )
        result = _serializer.parse_request(request.model_dump())

        msg = result.conversation.messages[0]
        file_blocks = [b for b in msg.content if isinstance(b, FileBlock)]
        assert len(file_blocks) == 1
        assert file_blocks[0].file_id == "file_abc123"
        assert file_blocks[0].filename == "doc.pdf"

    def test_file_input_with_file_url(self):
        """Per OpenAI spec, Responses API supports input_file with file_url."""
        request = ResponsesRequest(
            model="gpt-4",
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Analyze this PDF"},
                        {
                            "type": "input_file",
                            "file_url": "https://example.com/letter.pdf",
                            "filename": "letter.pdf",
                        },
                    ],
                }
            ],
        )
        result = _serializer.parse_request(request.model_dump())

        msg = result.conversation.messages[0]
        file_blocks = [b for b in msg.content if isinstance(b, FileBlock)]
        assert len(file_blocks) == 1
        assert file_blocks[0].file_data == "https://example.com/letter.pdf"
        assert file_blocks[0].filename == "letter.pdf"

    def test_file_input_with_base64(self):
        """Per OpenAI spec, Responses API supports input_file with file_data (base64)."""
        request = ResponsesRequest(
            model="gpt-4",
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Analyze this file"},
                        {
                            "type": "input_file",
                            "file_data": "data:application/pdf;base64,AAAA",
                            "filename": "report.pdf",
                        },
                    ],
                }
            ],
        )
        result = _serializer.parse_request(request.model_dump())

        msg = result.conversation.messages[0]
        file_blocks = [b for b in msg.content if isinstance(b, FileBlock)]
        assert len(file_blocks) == 1
        assert file_blocks[0].file_data == "data:application/pdf;base64,AAAA"


# =============================================================================
# Scenario 6: Multi-turn Conversation
# =============================================================================


class TestMultiTurnConversation:
    """Check that assistant plus user messages are sent as conversation history."""

    def test_conversation_with_function_call_output(self):
        request = ResponsesRequest(
            model="gpt-4",
            input=[
                {"type": "message", "role": "user", "content": "What's the weather?"},
                {
                    "type": "function_call",
                    "call_id": "call_123",
                    "name": "get_weather",
                    "arguments": '{"location": "SF"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_123",
                    "output": '{"temp": 72, "unit": "f"}',
                },
            ],
        )
        result = _serializer.parse_request(request.model_dump())
        # user message + function_call (assistant) + function_call_output (tool)
        assert len(result.conversation.messages) == 3
        assert result.conversation.messages[0].role == "user"
        assert result.conversation.messages[1].role == "assistant"
        assert result.conversation.messages[2].role == "tool"

    def test_previous_response_id_forwarded_in_extra(self):
        """previous_response_id should be stored in InternalRequest.extra."""
        request = ResponsesRequest(
            model="gpt-4",
            input="Follow up",
            previous_response_id="resp_prev123",
        )
        result = _serializer.parse_request(request.model_dump())
        assert result.extra.get("previous_response_id") == "resp_prev123"


# =============================================================================
# Multi-turn Responses API -> Chat Completions round-trip
# =============================================================================


class TestMultiTurnResponsesToChatCompletions:
    """Regression tests for round-tripping multi-turn /v1/responses input items
    (reasoning / function_call / function_call_output / assistant messages) into a
    Chat Completions body for chat-compatible providers.

    Previously the input parser dropped assistant ``output_text`` content
    (yielding an empty assistant message that providers reject as "Upstream
    request failed"), double-encoded ``function_call`` arguments, and silently
    dropped ``reasoning_text`` reasoning items.
    """

    def test_assistant_message_output_text_is_preserved(self):
        """An assistant message with output_text content must keep its text.

        Without handling output_text, the assistant message degraded to an
        empty TextBlock and the Chat Completions body emitted an assistant
        message with neither content nor tool_calls.
        """

        request = ResponsesRequest(
            model="gpt-4",
            input=[
                {"type": "message", "role": "user", "content": "Run it"},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Let me retry."}],
                },
            ],
        )
        body = _build_chat_completions_body(request)
        assistant_msgs = [m for m in body["messages"] if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        # The text must survive; no empty assistant messages allowed.
        assert assistant_msgs[0].get("content") == "Let me retry."

    def test_function_call_arguments_not_double_encoded(self):
        """function_call arguments must serialize to a JSON object, not a
        JSON-string-literal-of-a-JSON-string (double-encoded)."""

        request = ResponsesRequest(
            model="gpt-4",
            input=[
                {"type": "message", "role": "user", "content": "List runs"},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "exec_command",
                    "arguments": '{"cmd": "gh run list"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "ok",
                },
            ],
        )
        body = _build_chat_completions_body(request)
        tool_call_msgs = [
            m for m in body["messages"] if m.get("role") == "assistant" and m.get("tool_calls")
        ]
        assert len(tool_call_msgs) == 1
        args = tool_call_msgs[0]["tool_calls"][0]["function"]["arguments"]
        # Parsed arguments must be a dict, not a string.
        parsed = orjson.loads(args)
        assert isinstance(parsed, dict)
        assert parsed == {"cmd": "gh run list"}

    def test_reasoning_text_content_round_trips(self):
        """reasoning items with reasoning_text content (what the streaming
        transformer emits) must be preserved as reasoning_content, not dropped."""
        request = ResponsesRequest(
            model="gpt-4",
            input=[
                {"type": "message", "role": "user", "content": "Think"},
                {
                    "type": "reasoning",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": "planning the steps"}],
                },
            ],
        )
        internal = _serializer.parse_request(request.model_dump())
        # Reasoning must produce a ThinkingBlock carried on an assistant message.
        assistant_msgs = [m for m in internal.conversation.messages if m.role == "assistant"]
        assert len(assistant_msgs) == 1
        assert any(isinstance(b, ThinkingBlock) for b in assistant_msgs[0].content)

    def test_assistant_turn_items_are_merged(self):
        """Consecutive reasoning + assistant message + function_call items that
        belong to one assistant turn merge into a single Chat Completions
        assistant message carrying reasoning_content + content + tool_calls."""

        request = ResponsesRequest(
            model="gpt-4",
            input=[
                {"type": "message", "role": "user", "content": "Do it"},
                {
                    "type": "reasoning",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": "thinking"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Let me retry."}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "exec_command",
                    "arguments": '{"cmd": "gh run list"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "ok",
                },
            ],
        )
        body = _build_chat_completions_body(request)
        assistant_msgs = [m for m in body["messages"] if m["role"] == "assistant"]
        # reasoning + message + function_call merge into ONE assistant message.
        assert len(assistant_msgs) == 1
        merged = assistant_msgs[0]
        assert merged.get("reasoning_content") == "thinking"
        # With tool_calls present, content is emitted as structured parts.
        assert merged.get("content") == [{"type": "text", "text": "Let me retry."}]
        assert merged.get("tool_calls")
        assert merged["tool_calls"][0]["function"]["name"] == "exec_command"

    def test_no_empty_assistant_messages_on_full_second_turn(self):
        """A full Codex-style second turn (developer/user/reasoning/function_call/
        function_call_output/reasoning/assistant/function_call/function_call_output)
        must produce a valid Chat Completions body: every assistant message has
        content or tool_calls, and tool_call arguments are valid JSON objects."""

        request = ResponsesRequest(
            model="deepseek-v4-pro",
            instructions="You are a coding assistant.",
            stream=True,
            parallel_tool_calls=True,
            tool_choice="auto",
            input=[
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "dev instructions"}],
                },
                {"type": "message", "role": "user", "content": "first user msg"},
                {"type": "message", "role": "user", "content": "second user msg"},
                {
                    "type": "reasoning",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": "plan a"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_00_a",
                    "name": "exec_command",
                    "arguments": '{"cmd": "gh run list --limit 10"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_00_a",
                    "output": "error connecting to api.github.com",
                },
                {
                    "type": "reasoning",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": "plan b"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Let me retry."}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_00_b",
                    "name": "exec_command",
                    "arguments": '{"cmd": "gh run list --limit 5", "justification": "query runs"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_00_b",
                    "output": '[{"conclusion":"failure"}]',
                },
            ],
            text={"type": "text"},
            reasoning={"effort": "xhigh", "summary": "auto"},
            tools=[
                {
                    "type": "function",
                    "name": "exec_command",
                    "description": "Runs a command.",
                    "parameters": {"type": "object"},
                }
            ],
        )
        body = _build_chat_completions_body(request)

        # Every assistant message must carry content or tool_calls.
        for msg in body["messages"]:
            if msg["role"] != "assistant":
                continue
            assert msg.get("content") or msg.get("tool_calls"), (
                f"Empty assistant message would be rejected by providers: {msg!r}"
            )

        # Every tool_call argument must be a valid JSON object (not double-encoded).
        for msg in body["messages"]:
            for tc in msg.get("tool_calls", []):
                parsed = orjson.loads(tc["function"]["arguments"])
                assert isinstance(parsed, dict), (
                    f"tool_call arguments must parse to a dict, got {type(parsed).__name__}: "
                    f"{tc['function']['arguments']!r}"
                )

    def test_interleaved_thinking_stays_per_segment(self):
        """Interleaved thinking: reasoning must stay bound to its own assistant
        segment (the tool call it preceded) and NOT bleed across tool results
        into the next segment.

        Codex sends back a turn that did: reason -> call -> result -> reason ->
        call -> result. Each [reasoning + call] pair must become one assistant
        message carrying its own reasoning_content + tool_calls, so the provider
        receives the correct interleaved context.
        """
        request = ResponsesRequest(
            model="deepseek-v4-pro",
            input=[
                {"type": "message", "role": "user", "content": "list runs"},
                {
                    "type": "reasoning",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": "plan a"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_a",
                    "name": "exec_command",
                    "arguments": '{"cmd": "gh run list --limit 10"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_a",
                    "output": "error connecting to api.github.com",
                },
                {
                    "type": "reasoning",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": "plan b"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Let me retry."}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_b",
                    "name": "exec_command",
                    "arguments": '{"cmd": "gh run list --limit 5"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_b",
                    "output": '[{"conclusion":"failure"}]',
                },
            ],
            tools=[
                {
                    "type": "function",
                    "name": "exec_command",
                    "description": "Runs a command.",
                    "parameters": {"type": "object"},
                }
            ],
        )
        body = _build_chat_completions_body(request)

        assistant_msgs = [m for m in body["messages"] if m["role"] == "assistant"]
        # Two assistant segments (one per tool-call round), not one merged blob.
        assert len(assistant_msgs) == 2

        # Segment 1: reasoning "plan a" bound to call_a ONLY.
        seg1 = assistant_msgs[0]
        assert seg1.get("reasoning_content") == "plan a"
        assert [tc["id"] for tc in seg1["tool_calls"]] == ["call_a"]
        assert not seg1.get("content")

        # Segment 2: reasoning "plan b" + text + call_b.
        seg2 = assistant_msgs[1]
        assert seg2.get("reasoning_content") == "plan b"
        assert seg2.get("content") == [{"type": "text", "text": "Let me retry."}]
        assert [tc["id"] for tc in seg2["tool_calls"]] == ["call_b"]

        # Tool messages must sit between the two assistant segments.
        tool_msgs = [m for m in body["messages"] if m["role"] == "tool"]
        assert [m["tool_call_id"] for m in tool_msgs] == ["call_a", "call_b"]
        idx_seg1 = body["messages"].index(seg1)
        idx_tool_a = body["messages"].index(tool_msgs[0])
        idx_seg2 = body["messages"].index(seg2)
        assert idx_seg1 < idx_tool_a < idx_seg2

    def test_reasoning_plus_text_merges_into_one_assistant_message(self):
        """A reasoning item immediately followed by an assistant message (no tool
        call in the same segment) must merge into one assistant message carrying
        reasoning_content + content, not two separate assistant messages."""
        request = ResponsesRequest(
            model="gpt-4",
            input=[
                {"type": "message", "role": "user", "content": "Summarize"},
                {
                    "type": "reasoning",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": "step by step"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Here is the summary."}],
                },
            ],
        )
        body = _build_chat_completions_body(request)
        assistant_msgs = [m for m in body["messages"] if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0].get("reasoning_content") == "step by step"
        assert assistant_msgs[0].get("content") == "Here is the summary."

    def test_reasoning_round_trips_to_provider_reasoning_content_field(self):
        """The reasoning carried in /v1/responses reasoning items must be sent
        back to a chat-compatible provider as the ``reasoning_content`` field
        on the corresponding assistant message (the provider's own format)."""
        request = ResponsesRequest(
            model="deepseek-v4-pro",
            input=[
                {"type": "message", "role": "user", "content": "think and act"},
                {
                    "type": "reasoning",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": "deliberation"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_x",
                    "name": "exec_command",
                    "arguments": '{"cmd": "ls"}',
                },
                {"type": "function_call_output", "call_id": "call_x", "output": "done"},
            ],
            tools=[
                {
                    "type": "function",
                    "name": "exec_command",
                    "description": "Runs a command.",
                    "parameters": {"type": "object"},
                }
            ],
        )
        body = _build_chat_completions_body(request)
        assistant_msgs = [m for m in body["messages"] if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        # The provider's reasoning field is present and carries the thinking text.
        assert "reasoning_content" in assistant_msgs[0]
        assert assistant_msgs[0]["reasoning_content"] == "deliberation"

    def test_encrypted_content_only_reasoning_does_not_create_invalid_message(self):
        """A reasoning item whose content is empty (only encrypted_content, which
        a chat-compatible provider cannot use) must be dropped gracefully and
        must NOT produce an empty assistant message nor break the following
        function_call's assistant message."""
        request = ResponsesRequest(
            model="gpt-4",
            input=[
                {"type": "message", "role": "user", "content": "act"},
                {
                    "type": "reasoning",
                    "summary": [],
                    "content": [],
                    "encrypted_content": "opaque-encrypted-blob",
                },
                {
                    "type": "function_call",
                    "call_id": "call_y",
                    "name": "exec_command",
                    "arguments": '{"cmd": "pwd"}',
                },
                {"type": "function_call_output", "call_id": "call_y", "output": "/tmp"},
            ],
            tools=[
                {
                    "type": "function",
                    "name": "exec_command",
                    "description": "Runs a command.",
                    "parameters": {"type": "object"},
                }
            ],
        )
        body = _build_chat_completions_body(request)
        assistant_msgs = [m for m in body["messages"] if m["role"] == "assistant"]
        # The function_call still becomes a valid assistant message with tool_calls.
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0].get("tool_calls")
        # No empty assistant message (the encrypted-only reasoning was dropped).
        for m in assistant_msgs:
            assert m.get("content") or m.get("tool_calls")
        # encrypted_content is OpenAI-Responses-specific and must not leak into a
        # Chat Completions body.
        assert "encrypted_content" not in assistant_msgs[0]

    def test_web_search_only_turn_does_not_create_empty_assistant_message(self):
        """A turn that performed only a hosted web_search (reasoning +
        web_search_call, no function_call, no assistant text) must NOT degrade to
        an empty assistant message that providers reject as "Upstream request
        failed".

        Reproduces the Codex /v1/responses failure where the conversation works
        for several rounds (turns with function_call are valid) then suddenly
        fails on a web-search-only round: the web_search_call was skipped, leaving
        encrypted-only reasoning that flushes to an assistant message with neither
        content nor tool_calls. The proxy exposes web_search as a function tool to
        chat-completions providers, so the web_search_call must be replayed as a
        web_search tool_call with a matching (placeholder) tool result.
        """
        request = ResponsesRequest(
            model="deepseek-v4-pro",
            instructions="You are a coding assistant.",
            stream=True,
            input=[
                {"type": "message", "role": "user", "content": "research it"},
                {
                    "type": "reasoning",
                    "summary": [],
                    "content": [],
                    "encrypted_content": "opaque-encrypted-blob",
                },
                {
                    "type": "web_search_call",
                    "id": "ws_abc123",
                    "status": "completed",
                    "action": {
                        "type": "search",
                        "query": "how to fix upstream request failed",
                        "queries": ["how to fix upstream request failed"],
                    },
                },
                {
                    "type": "reasoning",
                    "summary": [],
                    "content": [],
                    "encrypted_content": "opaque-encrypted-blob-2",
                },
                {
                    "type": "web_search_call",
                    "id": "ws_def456",
                    "status": "completed",
                    "action": {
                        "type": "search",
                        "query": "deepseek chat completions tool_call",
                        "queries": [],
                    },
                },
                {"type": "message", "role": "user", "content": "now implement the fix"},
            ],
            include=["reasoning.encrypted_content"],
            tools=[{"type": "web_search", "external_web_access": False}],
            text={"type": "text"},
            reasoning={"effort": "xhigh", "summary": "auto"},
        )
        body = _build_chat_completions_body(request)

        # Every assistant message must carry content or tool_calls — an empty
        # assistant message is what providers reject as "Upstream request failed".
        assistant_msgs = [m for m in body["messages"] if m["role"] == "assistant"]
        assert assistant_msgs, "expected the web-search turn to produce an assistant message"
        for msg in assistant_msgs:
            assert msg.get("content") or msg.get("tool_calls"), (
                f"Empty assistant message would be rejected by providers: {msg!r}"
            )

        # The web_search_call items must be replayed as web_search function
        # tool_calls (the proxy exposes web_search as a function tool).
        web_search_calls = [
            tc
            for m in assistant_msgs
            for tc in m.get("tool_calls", [])
            if tc.get("function", {}).get("name") == "web_search"
        ]
        assert len(web_search_calls) == 2, (
            f"expected 2 web_search tool_calls, got {len(web_search_calls)}: {web_search_calls!r}"
        )

        # Every web_search tool_call must have a matching tool result message
        # (chat-completions providers require a result for every tool_call).
        tool_msgs = [m for m in body["messages"] if m.get("role") == "tool"]
        result_ids = {m.get("tool_call_id") for m in tool_msgs}
        call_ids = {tc.get("id") for tc in web_search_calls}
        missing = call_ids - result_ids
        assert not missing, f"web_search tool_calls without a matching tool result: {missing!r}"

    def test_encrypted_only_reasoning_turn_dropped_from_chat_completions_body(self):
        """A turn with only encrypted-only reasoning (no function_call, no text,
        no web_search) — e.g. a compaction summary or Codex reasoning with an
        empty summary — must NOT produce an assistant message in the Chat
        Completions body.

        Chat-completions providers cannot consume OpenAI-Responses-specific
        ``encrypted_content``; emitting an assistant message that carries only
        an empty ``reasoning_content`` would cause providers to reject the
        request with "Upstream request failed". The internal model still keeps
        the encrypted reasoning on the ``ThinkingBlock`` so a Responses-API
        provider can round-trip it on a later turn.
        """
        request = ResponsesRequest(
            model="deepseek-v4-pro",
            stream=True,
            input=[
                {"type": "message", "role": "user", "content": "do something"},
                {
                    "type": "reasoning",
                    "summary": [],
                    "content": [],
                    "encrypted_content": "opaque-encrypted-blob",
                },
                {"type": "compaction", "encrypted_content": "opaque-compaction-blob"},
                {"type": "message", "role": "user", "content": "continue"},
            ],
            include=["reasoning.encrypted_content"],
        )
        # The internal model must still capture the encrypted reasoning so a
        # Responses-API provider can round-trip it.
        internal = _serializer.parse_request(request.model_dump())
        thinking_blocks = [
            b
            for m in internal.conversation.messages
            if m.role == "assistant"
            for b in m.content
            if isinstance(b, ThinkingBlock) and b.encrypted_content
        ]
        assert len(thinking_blocks) == 2, (
            f"expected 2 encrypted ThinkingBlocks in internal model, got {len(thinking_blocks)!r}"
        )

        # The Chat Completions body must not contain an empty assistant
        # message. Encrypted-only reasoning (no visible text, no
        # tool_calls) carries nothing useful for chat-completions
        # providers — the encrypted blob cannot be decrypted and would
        # produce an empty message that providers reject. The Gemini
        # thought_signature leak that previously caused encrypted-only
        # items has been fixed, so this path should rarely trigger.
        body = _build_chat_completions_body(request)
        for msg in body["messages"]:
            if msg["role"] != "assistant":
                continue
            assert msg.get("content") or msg.get("tool_calls"), (
                f"Empty assistant message would be rejected by providers: {msg!r}"
            )
        assistant_msgs = [m for m in body["messages"] if m["role"] == "assistant"]
        assert assistant_msgs == [], (
            f"expected the encrypted-only reasoning turn to be dropped from the "
            f"Chat Completions body, got {assistant_msgs!r}"
        )

    def test_interleaved_reasoning_with_web_search_preserves_reasoning_content(self):
        """Interleaved thinking (reasoning interspersed with hosted web_search
        calls) must round-trip the reasoning text to a chat-completions provider.

        The proxy streams model reasoning as readable ``reasoning_text``, so when
        Codex sends the history back each reasoning item carries its text. The
        web_search calls (replayed as ``web_search`` tool_calls) give the
        assistant turn ``tool_calls``, so the turn is valid AND the reasoning is
        preserved as ``reasoning_content`` — neither the web_search replay nor
        the empty-assistant-message safety net should drop it.
        """
        request = ResponsesRequest(
            model="deepseek-v4-pro",
            stream=True,
            input=[
                {"type": "message", "role": "user", "content": "research it"},
                {
                    "type": "reasoning",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": "think step 1"}],
                },
                {
                    "type": "web_search_call",
                    "id": "ws_1",
                    "status": "completed",
                    "action": {"type": "search", "query": "q1", "queries": ["q1"]},
                },
                {
                    "type": "reasoning",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": "think step 2"}],
                },
                {
                    "type": "web_search_call",
                    "id": "ws_2",
                    "status": "completed",
                    "action": {"type": "search", "query": "q2", "queries": ["q2"]},
                },
                {"type": "message", "role": "user", "content": "now answer"},
            ],
            tools=[{"type": "web_search", "external_web_access": False}],
            text={"type": "text"},
            reasoning={"effort": "xhigh", "summary": "auto"},
        )
        body = _build_chat_completions_body(request)

        assistant_msgs = [m for m in body["messages"] if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1, (
            f"expected one assistant turn, got {len(assistant_msgs)}: {assistant_msgs!r}"
        )
        turn = assistant_msgs[0]
        # Reasoning text from both reasoning items is preserved (joined, since
        # web_search has no output item to flush between them).
        assert "think step 1" in turn["reasoning_content"]
        assert "think step 2" in turn["reasoning_content"]
        # web_search calls are replayed as tool_calls (keeps the turn non-empty).
        assert [tc["id"] for tc in turn["tool_calls"]] == ["ws_1", "ws_2"]
        assert {tc["function"]["name"] for tc in turn["tool_calls"]} == {"web_search"}
        # Each web_search tool_call has a matching tool result.
        tool_msgs = [m for m in body["messages"] if m.get("role") == "tool"]
        assert {m["tool_call_id"] for m in tool_msgs} == {"ws_1", "ws_2"}


class TestResponsesInputReasoningCapture:
    """The OpenResponses protocol parser must capture ``encrypted_content`` (and
    ``summary`` text) from reasoning input items into ``ThinkingBlock`` so it can
    be forwarded back to a Responses-API provider on the next turn."""

    def _parse(self, input_items):
        from llm_proxy.protocols.openresponses import (
            OpenResponsesProtocolSerializer,
        )

        data = {"model": "o3", "input": input_items, "stream": False}
        return OpenResponsesProtocolSerializer().parse_request(data)

    def test_encrypted_content_captured_into_thinking_block(self):
        internal = self._parse(
            [
                {"type": "message", "role": "user", "content": "act"},
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": "plan"}],
                    "encrypted_content": "OPAQUE_BLOB",
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "exec",
                    "arguments": "{}",
                },
                {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
            ]
        )
        thinking = [
            b
            for m in internal.conversation.messages
            for b in m.content
            if isinstance(b, ThinkingBlock)
        ]
        assert thinking, "reasoning item must be parsed into a ThinkingBlock"
        assert thinking[0].thinking == "plan"
        assert thinking[0].encrypted_content == "OPAQUE_BLOB"

    def test_summary_text_used_when_content_empty(self):
        internal = self._parse(
            [
                {"type": "message", "role": "user", "content": "act"},
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [{"type": "summary_text", "text": "the summary"}],
                    "content": [],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "exec",
                    "arguments": "{}",
                },
                {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
            ]
        )
        thinking = [
            b
            for m in internal.conversation.messages
            for b in m.content
            if isinstance(b, ThinkingBlock)
        ]
        assert thinking and thinking[0].thinking == "the summary"

    def test_encrypted_only_reasoning_is_preserved(self):
        internal = self._parse(
            [
                {"type": "message", "role": "user", "content": "act"},
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [],
                    "content": [],
                    "encrypted_content": "ENC_ONLY",
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "exec",
                    "arguments": "{}",
                },
                {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
            ]
        )
        thinking = [
            b
            for m in internal.conversation.messages
            for b in m.content
            if isinstance(b, ThinkingBlock)
        ]
        assert thinking and thinking[0].encrypted_content == "ENC_ONLY"
        assert thinking[0].thinking == ""


class TestGeminiThoughtSignatureRoundTrip:
    """Gemini's generateContent API uses ``thoughtSignature`` on ``functionCall``
    parts to preserve reasoning context across multi-step tool calls. The proxy
    must round-trip this signature through ``/v1/responses`` without leaking it
    as OpenAI-style ``encrypted_content`` on a reasoning item.
    """

    def _parse(self, input_items):
        data = {"model": "gemini-3.1-pro-preview", "input": input_items, "stream": False}
        return _serializer.parse_request(data)

    def _build_gemini_body(self, internal):
        context = BuildContext(
            stream=False,
            model="gemini-3.1-pro-preview",
            provider_name="gemini",
            target_endpoint="chat_completions",
            supported_content_blocks=GeminiProviderSerializer().supported_content_blocks,
        )
        return GeminiProviderSerializer().build_provider_request(internal, context)

    def test_function_call_thought_signature_parses(self):
        internal = self._parse(
            [
                {"type": "message", "role": "user", "content": "check flight"},
                {
                    "type": "function_call",
                    "call_id": "fc_check_flight",
                    "name": "check_flight",
                    "arguments": '{"flight": "AA100"}',
                    "thought_signature": "SIG_A",
                },
            ]
        )
        tool_blocks = [
            b
            for m in internal.conversation.messages
            if m.role == "assistant"
            for b in m.content
            if isinstance(b, ToolUseBlock)
        ]
        assert len(tool_blocks) == 1
        assert tool_blocks[0].extra.get("thought_signature") == "SIG_A"

    def test_function_call_thought_signature_reaches_gemini_request(self):
        internal = self._parse(
            [
                {"type": "message", "role": "user", "content": "check flight"},
                {
                    "type": "function_call",
                    "call_id": "fc_check_flight",
                    "name": "check_flight",
                    "arguments": '{"flight": "AA100"}',
                    "thought_signature": "SIG_A",
                },
            ]
        )
        body = self._build_gemini_body(internal)
        model_parts = [c for c in body["contents"] if c.get("role") == "model"][0]["parts"]
        fc_part = next(p for p in model_parts if "functionCall" in p)
        assert fc_part.get("thoughtSignature") == "SIG_A"

    def test_function_call_thought_signature_emitted_in_response(self):
        response = InternalResponse(
            id="resp_gemini_1",
            model="gemini-3.1-pro-preview",
            output=[
                ToolUseBlock(
                    id="fc_check_flight",
                    name="check_flight",
                    input={"flight": "AA100"},
                    extra={"thought_signature": "SIG_A"},
                )
            ],
            finish_reason="tool_calls",
        )
        formatted = _serializer.format_response(response)
        fc_items = [it for it in formatted["output"] if it["type"] == "function_call"]
        assert len(fc_items) == 1
        assert fc_items[0].get("thought_signature") == "SIG_A"

    def test_gemini_thought_signature_not_leaked_as_reasoning_encrypted_content(self):
        response = InternalResponse(
            id="resp_gemini_1",
            model="gemini-3.1-pro-preview",
            output=[
                ToolUseBlock(
                    id="fc_check_flight",
                    name="check_flight",
                    input={"flight": "AA100"},
                    extra={"thought_signature": "SIG_A"},
                )
            ],
            finish_reason="tool_calls",
        )
        formatted = _serializer.format_response(
            response, FormatContext(include=["reasoning.encrypted_content"])
        )
        reasoning_items = [it for it in formatted["output"] if it["type"] == "reasoning"]
        assert reasoning_items == []

    def test_encrypted_only_reasoning_dropped_from_gemini_request(self):
        internal = self._parse(
            [
                {"type": "message", "role": "user", "content": "hi"},
                {
                    "type": "reasoning",
                    "id": "rs1",
                    "content": [],
                    "summary": [],
                    "encrypted_content": "OPAQUE",
                },
            ]
        )
        body = self._build_gemini_body(internal)
        model_contents = [c for c in body["contents"] if c.get("role") == "model"]
        assert model_contents == []


# =============================================================================
# Request Parameter Forwarding
# =============================================================================


class TestRequestParamForwarding:
    """Verify that request parameters are properly forwarded to the internal request."""

    def test_store_param_forwarded(self):
        request = ResponsesRequest(model="gpt-4", input="Hello", store=True)
        result = _serializer.parse_request(request.model_dump())
        assert result.params.openai is not None
        assert result.params.openai.store is True

    def test_service_tier_forwarded(self):
        request = ResponsesRequest(model="gpt-4", input="Hello", service_tier="flex")
        result = _serializer.parse_request(request.model_dump())
        assert result.params.openai.service_tier == "flex"

    def test_reasoning_forwarded(self):
        request = ResponsesRequest(
            model="gpt-4",
            input="Hello",
            reasoning={"effort": "high", "summary": "detailed"},
        )
        result = _serializer.parse_request(request.model_dump())
        assert result.params.openai.reasoning_effort == "high"
        assert result.params.thinking is not None
        assert result.params.thinking.effort == "high"

    def test_parallel_tool_calls_forwarded(self):
        request = ResponsesRequest(model="gpt-4", input="Hello", parallel_tool_calls=True)
        result = _serializer.parse_request(request.model_dump())
        assert result.params.openai.parallel_tool_calls is True

    def test_top_logprobs_forwarded(self):
        request = ResponsesRequest(model="gpt-4", input="Hello", top_logprobs=5)
        result = _serializer.parse_request(request.model_dump())
        assert result.params.openai.top_logprobs == 5
        assert result.params.openai.logprobs is True

    def test_safety_identifier_forwarded(self):
        request = ResponsesRequest(model="gpt-4", input="Hello", safety_identifier="safety_123")
        result = _serializer.parse_request(request.model_dump())
        assert result.params.openai.safety_identifier == "safety_123"

    def test_prompt_cache_key_forwarded(self):
        request = ResponsesRequest(model="gpt-4", input="Hello", prompt_cache_key="cache_123")
        result = _serializer.parse_request(request.model_dump())
        assert result.params.openai.prompt_cache_key == "cache_123"

    def test_truncation_forwarded(self):
        request = ResponsesRequest(model="gpt-4", input="Hello", truncation="auto")
        result = _serializer.parse_request(request.model_dump())
        assert result.extra.get("truncation") == "auto"

    def test_background_forwarded(self):
        request = ResponsesRequest(model="gpt-4", input="Hello", background=True)
        result = _serializer.parse_request(request.model_dump())
        assert result.extra.get("background") is True

    def test_max_tool_calls_forwarded(self):
        request = ResponsesRequest(model="gpt-4", input="Hello", max_tool_calls=5)
        result = _serializer.parse_request(request.model_dump())
        assert result.extra.get("max_tool_calls") == 5


# =============================================================================
# Response Parameter Pass-Through
# =============================================================================


class TestResponseParameterPassthrough:
    """Verify that request parameters are reflected in the response."""

    def _make_internal(self):
        return InternalResponse(id="test", model="gpt-4", output=[TextBlock(text="Hi")])

    def test_store_reflected_in_response(self):

        result = _serializer.format_response(self._make_internal(), FormatContext(store=True))
        assert result["store"] is True

    def test_truncation_reflected_in_response(self):

        result = _serializer.format_response(
            self._make_internal(), FormatContext(truncation="auto")
        )
        assert result["truncation"] == "auto"

    def test_parallel_tool_calls_reflected_in_response(self):

        result = _serializer.format_response(
            self._make_internal(), FormatContext(parallel_tool_calls=True)
        )
        assert result["parallel_tool_calls"] is True

    def test_temperature_reflected_in_response(self):

        result = _serializer.format_response(self._make_internal(), FormatContext(temperature=0.5))
        assert result["temperature"] == 0.5

    def test_service_tier_reflected_in_response(self):

        result = _serializer.format_response(
            self._make_internal(), FormatContext(service_tier="flex")
        )
        assert result["service_tier"] == "flex"

    def test_previous_response_id_reflected_in_response(self):

        result = _serializer.format_response(
            self._make_internal(), FormatContext(previous_response_id="resp_prev")
        )
        assert result["previous_response_id"] == "resp_prev"


# =============================================================================
# completed_at / incomplete_details
# =============================================================================


class TestCompletedAtHandling:
    """Verify completed_at is correctly set for different statuses."""

    def test_completed_response_has_completed_at(self):
        internal = InternalResponse(
            id="test",
            model="gpt-4",
            output=[TextBlock(text="Hi")],
            finish_reason="stop",
        )
        result = _serializer.format_response(internal)
        assert result["status"] == "completed"
        assert result["completed_at"] is not None
        assert isinstance(result["completed_at"], int)

    def test_incomplete_response_has_correct_details(self):
        internal = InternalResponse(
            id="test",
            model="gpt-4",
            output=[TextBlock(text="Partial...")],
            finish_reason="length",
        )
        result = _serializer.format_response(internal)
        assert result["status"] == "incomplete"
        assert result["completed_at"] is not None
        assert result["incomplete_details"]["reason"] == "max_output_tokens"

    def test_failed_response_has_completed_at(self):
        internal = InternalResponse(
            id="test",
            model="gpt-4",
            output=[],
            finish_reason="error",
        )
        result = _serializer.format_response(internal)
        assert result["status"] == "failed"
        assert result["completed_at"] is not None
        assert result["error"] is not None


# =============================================================================
# include field handling
# =============================================================================


class TestIncludeFieldHandling:
    """Verify that the include field properly adds requested data."""

    def test_include_reasoning_encrypted_content_streaming(self):
        """When include contains reasoning.encrypted_content, encrypted_content
        arriving on a Chat Completions streaming delta must be attached to the
        emitted reasoning item."""
        from llm_proxy.protocols.openresponses.streaming import OpenResponsesStreamingTransformer

        transformer = OpenResponsesStreamingTransformer(
            model="o3",
            request_id="req-1",
        )
        transformer.state.include_reasoning_encrypted = True
        chunk = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "o3",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "reasoning_content": "thinking",
                        "encrypted_content": "opaque-blob",
                    },
                    "finish_reason": "stop",
                }
            ],
        }
        events = transformer._transform_chat_completions_chunk(chunk)
        assert events
        # Find the output_item.done event for the reasoning item.
        reasoning_done = None
        for line in events.strip().split("\n\n"):
            data_line = next((ln for ln in line.splitlines() if ln.startswith("data: ")), None)
            if data_line is None:
                continue
            data = orjson.loads(data_line[6:])
            if data.get("type") == "response.output_item.done":
                item = data.get("item", {})
                if item.get("type") == "reasoning":
                    reasoning_done = item
        assert reasoning_done is not None
        assert reasoning_done.get("encrypted_content") == "opaque-blob"

    def test_include_reasoning_encrypted_content(self):
        """When include contains reasoning.encrypted_content, add it to reasoning items.

        ``encrypted_content`` is sourced from the ThinkingBlock itself, not from
        ``provider_info``. Provider-level thought_signatures (Gemini) and other
        non-OpenAI metadata are not leaked as encrypted reasoning to the client.
        """

        internal = InternalResponse(
            id="test",
            model="gpt-4",
            output=[
                TextBlock(text="Answer"),
                ThinkingBlock(thinking="Let me think...", encrypted_content="encrypted_abc123"),
            ],
        )
        context = FormatContext(include=["reasoning.encrypted_content"])
        result = _serializer.format_response(internal, context)
        reasoning = [o for o in result["output"] if o["type"] == "reasoning"]
        assert len(reasoning) == 1
        assert reasoning[0]["encrypted_content"] == "encrypted_abc123"

    def test_include_logprobs(self):
        """When include contains message.output_text.logprobs, add logprobs to text parts."""

        internal = InternalResponse(
            id="test",
            model="gpt-4",
            output=[TextBlock(text="Hello")],
            logprobs=ChoiceLogprobs(content=[TokenLogprob(token="Hello", logprob=-0.5)]),
        )
        context = FormatContext(include=["message.output_text.logprobs"])
        result = _serializer.format_response(internal, context)
        msg = [o for o in result["output"] if o["type"] == "message"]
        assert len(msg) == 1
        assert "logprobs" in msg[0]["content"][0]

    def test_include_logprobs_empty_when_not_available(self):
        """When logprobs are requested but not available, include empty list."""

        internal = InternalResponse(
            id="test",
            model="gpt-4",
            output=[TextBlock(text="Hi")],
        )
        context = FormatContext(include=["message.output_text.logprobs"])
        result = _serializer.format_response(internal, context)
        msg = [o for o in result["output"] if o["type"] == "message"]
        assert msg[0]["content"][0]["logprobs"] == []

    def test_include_forwarded_in_internal_request(self):
        """The include field should be stored in InternalRequest.extra."""
        request = ResponsesRequest(
            model="gpt-4",
            input="Hello",
            include=["reasoning.encrypted_content", "message.output_text.logprobs"],
        )
        result = _serializer.parse_request(request.model_dump())
        assert result.extra.get("include") == [
            "reasoning.encrypted_content",
            "message.output_text.logprobs",
        ]


# =============================================================================
# Streaming events compliance
# =============================================================================


class TestStreamingEventsCompliance:
    """Verify streaming transformer produces spec-compliant events."""

    def test_response_incomplete_event_exists(self):
        """The streaming transformer should support response.incomplete events."""

        transformer = OpenResponsesStreamingTransformer(model="gpt-4")
        assert hasattr(transformer._factory, "_create_response_incomplete_event")

    def test_length_finish_emits_incomplete_event(self):
        """Finish reason 'length' should emit response.incomplete event."""

        transformer = OpenResponsesStreamingTransformer(model="gpt-4")
        events = transformer._transform_chunk(
            {
                "choices": [{"delta": {"content": "Hi"}, "finish_reason": "length"}],
            }
        )
        assert "response.incomplete" in events


# =============================================================================
# Previous response ID resolution
# =============================================================================


class TestPreviousResponseIdResolution:
    """Verify previous_response_id resolution from the response store."""

    def test_prepend_previous_response_reconstructs_conversation(self):
        """replay_stored_response should prepend previous output items."""
        from llm_proxy.models import ConversationContext, InternalRequest, Message, TextBlock
        from llm_proxy.protocols.openresponses import replay_stored_response

        prev_response = {
            "instructions": "Be helpful",
            "input": "What is 2+2?",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "4"}],
                    "status": "completed",
                    "id": "msg_prev1",
                }
            ],
        }
        req = InternalRequest(
            model="gpt-4",
            conversation=ConversationContext(system_messages=[], messages=[]),
            extra={"previous_response_id": "resp_prev"},
        )
        # Add current user message
        req.conversation.messages.append(
            Message(role="user", content=[TextBlock(text="Now multiply by 3")])
        )

        replay_stored_response(prev_response, req.conversation)

        # Should have: prev user, prev assistant, current user
        assert len(req.conversation.messages) == 3
        assert req.conversation.messages[0].role == "user"
        assert req.conversation.messages[1].role == "assistant"
        assert req.conversation.messages[2].role == "user"
        # Should have instructions from previous response
        assert len(req.conversation.system_messages) == 1
        assert req.conversation.system_messages[0].text_content == "Be helpful"

    def test_prepend_previous_with_function_calls(self):
        """replay_stored_response should handle function call items."""
        from llm_proxy.models import ConversationContext, InternalRequest, Message, TextBlock
        from llm_proxy.protocols.openresponses import replay_stored_response

        prev_response = {
            "input": [
                {"type": "message", "role": "user", "content": "What's the weather?"},
            ],
            "output": [
                {
                    "type": "function_call",
                    "id": "fc_prev1",
                    "call_id": "call_prev1",
                    "name": "get_weather",
                    "arguments": '{"city": "SF"}',
                    "status": "completed",
                }
            ],
        }
        req = InternalRequest(
            model="gpt-4",
            conversation=ConversationContext(system_messages=[], messages=[]),
        )
        req.conversation.messages.append(
            Message(role="user", content=[TextBlock(text="Follow up")])
        )

        replay_stored_response(prev_response, req.conversation)

        # Should have: prev user, prev function_call (assistant), current user
        assert len(req.conversation.messages) == 3
        assert req.conversation.messages[0].role == "user"
        assert req.conversation.messages[1].role == "assistant"  # function_call
        assert req.conversation.messages[2].role == "user"

    def test_prepend_previous_response_preserves_custom_tool_calls_in_output(self):
        """Stored output custom_tool_call items must survive continuation.

        Regression: the output-side loop previously dropped every item type
        except message/reasoning/function_call, so a stored custom_tool_call
        (Codex apply_patch) vanished and the follow-up custom_tool_call_output
        dangled (a tool message with no preceding tool call → upstream 400).
        """
        from llm_proxy.models import ConversationContext, InternalRequest, Message
        from llm_proxy.models.content_blocks import CustomToolUseBlock, ToolResultBlock
        from llm_proxy.protocols.openresponses import replay_stored_response

        prev_response = {
            "input": [{"type": "message", "role": "user", "content": "patch it"}],
            "output": [
                {
                    "type": "custom_tool_call",
                    "id": "ctc_1",
                    "call_id": "call_patch",
                    "name": "apply_patch",
                    "input": "***patch***",
                    "status": "completed",
                }
            ],
        }
        req = InternalRequest(
            model="gpt-5.2",
            conversation=ConversationContext(system_messages=[], messages=[]),
            extra={"previous_response_id": "resp_prev"},
        )
        # The client executed the custom tool and returns its output.
        req.conversation.messages.append(
            Message(
                role="tool",
                content=[ToolResultBlock(tool_use_id="call_patch", content="applied")],
            )
        )

        replay_stored_response(prev_response, req.conversation)

        # user, assistant(custom_tool_call), current tool output — the call is
        # present so the tool output no longer dangles.
        assert len(req.conversation.messages) == 3
        assert req.conversation.messages[1].role == "assistant"
        assert any(
            isinstance(b, CustomToolUseBlock) and b.id == "call_patch"
            for b in req.conversation.messages[1].content
        )
        assert req.conversation.messages[2].role == "tool"

    def test_prepend_previous_response_rehydrates_compaction_blob(self):
        """Proxy-produced compaction blobs in stored input are rehydrated."""
        from llm_proxy.models import ConversationContext, InternalRequest, Message, TextBlock
        from llm_proxy.protocols.openresponses import replay_stored_response
        from llm_proxy.protocols.openresponses.compaction import encode_compaction_blob

        blob = encode_compaction_blob([{"type": "message", "role": "user", "content": "old turn"}])
        prev_response = {
            "input": [{"type": "compaction", "encrypted_content": blob}],
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "ok"}],
                    "status": "completed",
                }
            ],
        }
        req = InternalRequest(
            model="gpt-5.2",
            conversation=ConversationContext(system_messages=[], messages=[]),
        )
        req.conversation.messages.append(Message(role="user", content=[TextBlock(text="next")]))

        replay_stored_response(prev_response, req.conversation)

        # The blob is unpacked back into the original user turn.
        assert req.conversation.messages[0].role == "user"
        assert req.conversation.messages[0].text_content == "old turn"
        assert req.conversation.messages[1].role == "assistant"
        assert req.conversation.messages[2].role == "user"

    def test_prepend_previous_response_preserves_assistant_phase(self):
        """The commentary/final_answer phase label survives continuation."""
        from llm_proxy.models import ConversationContext, InternalRequest
        from llm_proxy.protocols.openresponses import replay_stored_response

        prev_response = {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "phase": "commentary",
                    "content": [{"type": "output_text", "text": "working on it"}],
                    "status": "completed",
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "phase": "final_answer",
                    "content": [{"type": "output_text", "text": "done"}],
                    "status": "completed",
                },
            ],
        }
        req = InternalRequest(
            model="gpt-5.2",
            conversation=ConversationContext(system_messages=[], messages=[]),
        )

        replay_stored_response(prev_response, req.conversation)

        phases = [m.phase for m in req.conversation.messages]
        assert phases == ["commentary", "final_answer"]

    def test_prepend_previous_response_system_items_go_to_system_messages(self):
        """System items in stored input land back in system_messages."""
        from llm_proxy.models import ConversationContext, InternalRequest, Message, TextBlock
        from llm_proxy.protocols.openresponses import replay_stored_response

        prev_response = {
            "input": [
                {"type": "message", "role": "system", "content": "Be terse."},
                {"type": "message", "role": "user", "content": "hi"},
            ],
            "output": [],
        }
        req = InternalRequest(
            model="gpt-5.2",
            conversation=ConversationContext(system_messages=[], messages=[]),
        )
        req.conversation.messages.append(Message(role="user", content=[TextBlock(text="again")]))

        replay_stored_response(prev_response, req.conversation)

        assert req.conversation.system_messages[0].text_content == "Be terse."
        assert [m.role for m in req.conversation.messages] == ["user", "user"]


# =============================================================================
# Circular import fix
# =============================================================================


class TestCircularImportFix:
    """Verify the circular import issue is resolved."""

    def test_models_imports_cleanly(self):
        """Importing llm_proxy.models should not cause circular import."""
        import importlib

        # Force fresh import
        if "llm_proxy.models" in sys.modules:
            mod = sys.modules["llm_proxy.models"]
        else:
            mod = importlib.import_module("llm_proxy.models")
        assert mod is not None

    def test_request_type_still_works(self):
        """RequestType enum should still be importable."""
        from llm_proxy.core.request_type import RequestType

        assert RequestType.CHAT == "chat"
        assert RequestType.EMBEDDING == "embedding"
        assert RequestType.IMAGE_GENERATION == "image_generation"

    def test_internal_request_default_type(self):
        """InternalRequest should default to 'chat' request type."""
        from llm_proxy.models import ConversationContext, InternalRequest

        req = InternalRequest(
            model="gpt-4",
            conversation=ConversationContext(system_messages=[], messages=[]),
        )
        assert req.request_type == "chat"

    def test_embedding_request_type(self):
        """InternalEmbeddingRequest should default to 'embedding' request type."""
        from llm_proxy.models import InternalEmbeddingRequest

        req = InternalEmbeddingRequest(model="text-embedding-3-small", input="test")
        assert req.request_type == "embedding"

    def test_image_request_type(self):
        """InternalImageRequest should default to 'image_generation' request type."""
        from llm_proxy.models import InternalImageRequest

        req = InternalImageRequest(model="dall-e-3", prompt="a cat")
        assert req.request_type == "image_generation"


# =============================================================================
# Response Restoration: tool_search calls/outputs and namespace name restoration
# =============================================================================


class TestResponseRestoration:
    """Verify bidirectional restoration: tool_search calls/outputs and
    namespace name restoration in format_response()."""

    def test_tool_search_call_restored_in_response(self):
        """ToolUseBlock(name='tool_search') -> tool_search_call output item."""
        internal = InternalResponse(
            id="test",
            model="gpt-4",
            output=[
                ToolUseBlock(
                    id="call_001",
                    name="tool_search",
                    input={"query": "find tools", "tool_types": ["function"]},
                ),
            ],
        )
        result = _serializer.format_response(internal)
        tool_search_calls = [o for o in result["output"] if o["type"] == "tool_search_call"]
        assert len(tool_search_calls) == 1
        assert tool_search_calls[0]["call_id"] == "call_001"
        assert tool_search_calls[0]["execution"] == "client"
        assert tool_search_calls[0]["arguments"] == {
            "query": "find tools",
            "tool_types": ["function"],
        }

    def test_tool_search_output_restored_in_response(self):
        """ToolSearchToolResultBlock -> tool_search_output item."""

        internal = InternalResponse(
            id="test",
            model="gpt-4",
            output=[
                ToolSearchToolResultBlock(
                    tool_use_id="call_001",
                    content='[{"name": "get_weather", "type": "function"}]',
                ),
            ],
        )
        result = _serializer.format_response(internal)
        tool_search_outputs = [o for o in result["output"] if o["type"] == "tool_search_output"]
        assert len(tool_search_outputs) == 1
        assert tool_search_outputs[0]["call_id"] == "call_001"
        assert tool_search_outputs[0]["status"] == "completed"
        assert tool_search_outputs[0]["execution"] == "client"
        assert tool_search_outputs[0]["tools"] == [{"name": "get_weather", "type": "function"}]

    def test_namespace_name_restored_in_response(self):
        """Flat name 'mcp__github__list_issues' -> 'list_issues' with namespace_map."""

        internal = InternalResponse(
            id="test",
            model="gpt-4",
            output=[
                ToolUseBlock(
                    id="call_mcp",
                    name="mcp__github__list_issues",
                    input={"repo": "llm-proxy"},
                ),
            ],
        )
        context = FormatContext(
            namespace_map={"mcp__github__list_issues": ["github", "list_issues"]}
        )
        result = _serializer.format_response(internal, context)
        function_calls = [o for o in result["output"] if o["type"] == "function_call"]
        assert len(function_calls) == 1
        assert function_calls[0]["name"] == "list_issues"
        assert function_calls[0]["namespace"] == "github"

    def test_namespace_short_name_restored_in_response(self):
        """Short-name echo 'list_issues' is restored with its namespace.

        Models frequently echo the short history name instead of the flattened
        definition name; without the namespace the client (Codex) resolves the
        call against the default "functions" namespace and cannot match the
        tool.
        """

        internal = InternalResponse(
            id="test",
            model="gpt-4",
            output=[
                ToolUseBlock(
                    id="call_mcp",
                    name="list_issues",
                    input={"repo": "llm-proxy"},
                ),
            ],
        )
        context = FormatContext(
            namespace_map={"mcp__github__list_issues": ["github", "list_issues"]}
        )
        result = _serializer.format_response(internal, context)
        function_calls = [o for o in result["output"] if o["type"] == "function_call"]
        assert len(function_calls) == 1
        assert function_calls[0]["name"] == "list_issues"
        assert function_calls[0]["namespace"] == "github"

    def test_unmapped_names_passthrough(self):
        """Unmapped names pass through unchanged."""

        internal = InternalResponse(
            id="test",
            model="gpt-4",
            output=[
                ToolUseBlock(
                    id="call_norm",
                    name="get_weather",
                    input={"city": "SF"},
                ),
            ],
        )
        context = FormatContext(namespace_map={"other_tool": ["other", "tool"]})
        result = _serializer.format_response(internal, context)
        function_calls = [o for o in result["output"] if o["type"] == "function_call"]
        assert len(function_calls) == 1
        assert function_calls[0]["name"] == "get_weather"

    def test_namespace_custom_name_restored_in_response(self):
        """Flat namespaced custom tool is restored to original name as custom_tool_call."""

        internal = InternalResponse(
            id="test",
            model="gpt-4",
            output=[
                ToolUseBlock(
                    id="call_custom",
                    name="mcp__patch",
                    input={"content": "apply this patch"},
                ),
            ],
        )
        context = FormatContext(
            namespace_map={"mcp__patch": ["mcp", "patch"]},
            tools=[
                {
                    "type": "namespace",
                    "name": "mcp",
                    "tools": [{"type": "custom", "name": "patch"}],
                }
            ],
        )
        result = _serializer.format_response(internal, context)
        custom_calls = [o for o in result["output"] if o["type"] == "custom_tool_call"]
        assert len(custom_calls) == 1
        assert custom_calls[0]["name"] == "patch"
        assert custom_calls[0]["input"] == "apply this patch"
