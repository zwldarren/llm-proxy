"""Tests for OpenResponses format converters."""

from llm_proxy.models import (
    FunctionTool,
    InternalRequest,
    InternalResponse,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
)
from llm_proxy.models.types import Usage
from llm_proxy.protocols.openresponses import (
    OpenResponsesProtocolSerializer,
)
from llm_proxy.protocols.openresponses.schemas import ResponsesRequest
from llm_proxy.protocols.openresponses.serializer import (
    _convert_input_content,
    _convert_usage,
    _parse_tool,
    _parse_tool_choice,
)
from llm_proxy.serialization.responses_toolkit import (
    generate_item_id,
)

_serializer = OpenResponsesProtocolSerializer()


class TestGenerateItemId:
    """Tests for generate_item_id function."""

    def test_returns_string(self):
        result = generate_item_id()
        assert isinstance(result, str)

    def test_starts_with_item_prefix(self):
        result = generate_item_id()
        assert result.startswith("item_")

    def test_unique_ids(self):
        id1 = generate_item_id()
        id2 = generate_item_id()
        assert id1 != id2

    def test_has_correct_length(self):
        result = generate_item_id()
        assert len(result) == 29  # "item_" + 24 hex chars


class TestConvertInputContent:
    """Tests for _convert_input_content function."""

    def test_none_returns_empty_text_block(self):
        result = _convert_input_content(None)
        assert len(result) == 1
        assert isinstance(result[0], TextBlock)
        assert result[0].text == ""

    def test_string_returns_text_block(self):
        result = _convert_input_content("Hello")
        assert len(result) == 1
        assert isinstance(result[0], TextBlock)
        assert result[0].text == "Hello"

    def test_non_list_non_string_returns_string_content(self):
        result = _convert_input_content(123)
        assert len(result) == 1
        assert result[0].text == "123"

    def test_empty_list_returns_empty_text_block(self):
        result = _convert_input_content([])
        assert len(result) == 1
        assert result[0].text == ""

    def test_list_with_input_text(self):
        result = _convert_input_content([{"type": "input_text", "text": "Hello"}])
        assert len(result) == 1
        assert result[0].text == "Hello"

    def test_list_with_input_image(self):
        from llm_proxy.models.content_blocks import ImageBlock

        result = _convert_input_content(
            [{"type": "input_image", "image_url": "http://example.com/image.png"}]
        )
        assert len(result) == 1
        assert isinstance(result[0], ImageBlock)

    def test_input_image_with_base64_data_uri(self):
        from llm_proxy.models.content_blocks import ImageBlock

        result = _convert_input_content(
            [
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,iVBORw0KGgo=",
                }
            ]
        )
        assert len(result) == 1
        assert isinstance(result[0], ImageBlock)
        assert result[0].source.type == "base64"
        assert result[0].source.data == "iVBORw0KGgo="
        assert result[0].source.media_type == "image/png"

    def test_input_file_with_file_id(self):
        from llm_proxy.models.content_blocks import FileBlock

        result = _convert_input_content(
            [{"type": "input_file", "file_id": "file_abc123", "filename": "doc.pdf"}]
        )
        assert len(result) == 1
        assert isinstance(result[0], FileBlock)
        assert result[0].file_id == "file_abc123"
        assert result[0].filename == "doc.pdf"
        assert result[0].file_data is None

    def test_input_file_with_file_url(self):
        from llm_proxy.models.content_blocks import FileBlock

        result = _convert_input_content(
            [
                {
                    "type": "input_file",
                    "file_url": "https://example.com/document.pdf",
                    "filename": "document.pdf",
                }
            ]
        )
        assert len(result) == 1
        assert isinstance(result[0], FileBlock)
        assert result[0].file_data == "https://example.com/document.pdf"
        assert result[0].filename == "document.pdf"
        assert result[0].file_id is None

    def test_input_file_with_base64_data(self):
        from llm_proxy.models.content_blocks import FileBlock

        result = _convert_input_content(
            [
                {
                    "type": "input_file",
                    "file_data": "data:application/pdf;base64,AAAA",
                    "filename": "report.pdf",
                }
            ]
        )
        assert len(result) == 1
        assert isinstance(result[0], FileBlock)
        assert result[0].file_data == "data:application/pdf;base64,AAAA"
        assert result[0].filename == "report.pdf"

    def test_input_file_with_all_fields_file_id_takes_precedence(self):
        from llm_proxy.models.content_blocks import FileBlock

        result = _convert_input_content(
            [
                {
                    "type": "input_file",
                    "file_url": "https://example.com/doc.pdf",
                    "file_data": "data:application/pdf;base64,AAAA",
                    "file_id": "file_abc",
                    "filename": "doc.pdf",
                }
            ]
        )
        assert len(result) == 1
        assert isinstance(result[0], FileBlock)
        assert result[0].file_id == "file_abc"
        assert result[0].file_data == "data:application/pdf;base64,AAAA"

    def test_input_image_with_detail(self):
        from llm_proxy.models.content_blocks import ImageBlock

        result = _convert_input_content(
            [
                {
                    "type": "input_image",
                    "image_url": "https://example.com/img.png",
                    "detail": "low",
                }
            ]
        )
        assert isinstance(result[0], ImageBlock)
        assert result[0].detail == "low"

    def test_input_video_degraded_to_text(self):
        result = _convert_input_content(
            [{"type": "input_video", "video_url": "https://example.com/video.mp4"}]
        )
        assert len(result) == 1
        assert isinstance(result[0], TextBlock)

    def test_input_file_with_file_id_through_full_request(self):
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
                            "file_id": "file_abc123",
                            "filename": "data.csv",
                        },
                    ],
                }
            ],
        )
        result = _serializer.parse_request(request.model_dump())
        from llm_proxy.models.content_blocks import FileBlock

        blocks = result.conversation.messages[0].content
        files = [b for b in blocks if isinstance(b, FileBlock)]
        assert len(files) == 1
        assert files[0].file_id == "file_abc123"
        assert files[0].filename == "data.csv"

    def test_input_file_with_file_url_through_full_request(self):
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
                            "file_url": "https://example.com/doc.pdf",
                            "filename": "doc.pdf",
                        },
                    ],
                }
            ],
        )
        result = _serializer.parse_request(request.model_dump())
        from llm_proxy.models.content_blocks import FileBlock

        blocks = result.conversation.messages[0].content
        files = [b for b in blocks if isinstance(b, FileBlock)]
        assert len(files) == 1
        assert files[0].file_data == "https://example.com/doc.pdf"
        assert files[0].filename == "doc.pdf"

    def test_base64_image_through_full_request(self):
        from llm_proxy.models.content_blocks import ImageBlock

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
        blocks = result.conversation.messages[0].content
        images = [b for b in blocks if isinstance(b, ImageBlock)]
        assert len(images) == 1
        assert images[0].source.type == "base64"
        assert images[0].source.data == "iVBORw0KGgo="

    def test_multiple_file_inputs(self):
        from llm_proxy.models.content_blocks import FileBlock

        request = ResponsesRequest(
            model="gpt-4",
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Compare these files"},
                        {
                            "type": "input_file",
                            "file_id": "file_1",
                            "filename": "doc1.pdf",
                        },
                        {
                            "type": "input_file",
                            "file_id": "file_2",
                            "filename": "doc2.pdf",
                        },
                    ],
                }
            ],
        )
        result = _serializer.parse_request(request.model_dump())
        blocks = result.conversation.messages[0].content
        files = [b for b in blocks if isinstance(b, FileBlock)]
        assert len(files) == 2
        assert files[0].file_id == "file_1"
        assert files[1].file_id == "file_2"

    def test_list_with_unknown_type_skipped(self):
        result = _convert_input_content([{"type": "unknown_type", "text": "test"}])
        assert len(result) == 1
        assert result[0].text == ""

    def test_list_with_dict_having_model_dump(self):
        class MockPart:
            type = "input_text"
            text = "Mocked"

            def model_dump(self):
                return {"type": "input_text", "text": "Mocked"}

        result = _convert_input_content([MockPart()])
        assert len(result) == 1
        assert result[0].text == "Mocked"

    def test_input_audio_with_base64_data(self):
        from llm_proxy.models.content_blocks import AudioBlock

        result = _convert_input_content(
            [{"type": "input_audio", "audio_data": "base64data", "format": "wav"}]
        )
        assert len(result) == 1
        assert isinstance(result[0], AudioBlock)
        assert result[0].source.type == "base64"
        assert result[0].source.data == "base64data"
        assert result[0].source.media_type == "audio/wav"

    def test_input_audio_with_url(self):
        from llm_proxy.models.content_blocks import AudioBlock

        result = _convert_input_content(
            [
                {
                    "type": "input_audio",
                    "audio_url": "https://example.com/audio.mp3",
                    "format": "mp3",
                }
            ]
        )
        assert len(result) == 1
        assert isinstance(result[0], AudioBlock)
        assert result[0].source.type == "url"
        assert result[0].source.data == "https://example.com/audio.mp3"
        assert result[0].source.media_type == "audio/mpeg"

    def test_input_audio_with_data_uri(self):
        from llm_proxy.models.content_blocks import AudioBlock

        result = _convert_input_content(
            [
                {
                    "type": "input_audio",
                    "audio_url": "data:audio/mp3;base64,base64data",
                    "format": "mp3",
                }
            ]
        )
        assert len(result) == 1
        assert isinstance(result[0], AudioBlock)
        assert result[0].source.type == "base64"
        assert result[0].source.data == "base64data"
        assert result[0].source.media_type == "audio/mp3"

    def test_input_audio_format_mapping(self):
        from llm_proxy.models.content_blocks import AudioBlock

        test_cases = [
            ("wav", "audio/wav"),
            ("mp3", "audio/mpeg"),
            ("ogg", "audio/ogg"),
            ("flac", "audio/flac"),
            ("webm", "audio/webm"),
            ("mp4", "audio/mp4"),
        ]
        for fmt, expected_mime in test_cases:
            result = _convert_input_content(
                [{"type": "input_audio", "audio_data": "data", "format": fmt}]
            )
            assert isinstance(result[0], AudioBlock)
            assert result[0].source.media_type == expected_mime


class TestParseTool:
    """Tests for _parse_tool function."""

    def test_nested_function_key(self):
        tool = {
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object"},
            }
        }
        result = _parse_tool(tool)
        assert isinstance(result, FunctionTool)
        assert result.name == "get_weather"
        assert result.description == "Get weather"

    def test_flat_tool_format(self):
        """FunctionToolParam is flat: name/description/parameters at top level."""
        tool = {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object"},
            "type": "function",
        }
        result = _parse_tool(tool)
        assert isinstance(result, FunctionTool)
        assert result.name == "get_weather"
        assert result.description == "Get weather"

    def test_strict_tool(self):
        tool = {"name": "strict_tool", "strict": True, "type": "function"}
        result = _parse_tool(tool)
        assert result.strict is True

    def test_strict_defaults_false(self):
        tool = {"name": "loose_tool", "type": "function"}
        result = _parse_tool(tool)
        assert result.strict is False

    def test_missing_name(self):
        tool = {}
        result = _parse_tool(tool)
        assert result.name == ""
        assert result.parameters == {"type": "object"}


class TestParseToolChoice:
    """Tests for _parse_tool_choice function."""

    def test_none_returns_none(self):
        result = _parse_tool_choice(None)
        assert result is None

    def test_string_auto(self):
        from llm_proxy.models import ToolChoice

        result = _parse_tool_choice("auto")
        assert isinstance(result, ToolChoice)
        assert result.mode == "auto"

    def test_string_none(self):
        from llm_proxy.models import ToolChoice

        result = _parse_tool_choice("none")
        assert isinstance(result, ToolChoice)
        assert result.mode == "none"

    def test_string_required(self):

        result = _parse_tool_choice("required")
        assert result.mode == "required"

    def test_dict_function_type(self):
        from llm_proxy.models import ToolChoiceFunction

        result = _parse_tool_choice({"type": "function", "function": {"name": "get_weather"}})
        assert isinstance(result, ToolChoiceFunction)
        assert result.name == "get_weather"

    def test_dict_allowed_tools_auto(self):
        from llm_proxy.models import ToolChoice

        result = _parse_tool_choice({"type": "allowed_tools", "mode": "auto"})
        assert isinstance(result, ToolChoice)
        assert result.mode == "auto"

    def test_dict_allowed_tools_none(self):

        result = _parse_tool_choice({"type": "allowed_tools", "mode": "none"})
        assert result.mode == "none"

    def test_dict_allowed_tools_single_tool(self):
        from llm_proxy.models import ToolChoiceFunction

        result = _parse_tool_choice(
            {"type": "allowed_tools", "mode": "specific", "tools": [{"name": "get_weather"}]}
        )
        assert isinstance(result, ToolChoiceFunction)
        assert result.name == "get_weather"

    def test_dict_allowed_tools_multiple_tools(self):
        from llm_proxy.models.tools import ToolChoiceAllowedTools

        result = _parse_tool_choice(
            {
                "type": "allowed_tools",
                "mode": "specific",
                "tools": [{"name": "tool1"}, {"name": "tool2"}],
            }
        )
        assert isinstance(result, ToolChoiceAllowedTools)
        assert result.allowed_tools.mode == "specific"
        assert len(result.allowed_tools.tools) == 2

    def test_dict_unknown_type_returns_none(self):
        result = _parse_tool_choice({"type": "unknown"})
        assert result is None


class TestConvertUsage:
    """Tests for _convert_usage function."""

    def test_none_returns_zeros(self):
        result = _convert_usage(None)
        assert result["input_tokens"] == 0
        assert result["output_tokens"] == 0
        assert result["total_tokens"] == 0

    def test_basic_usage(self):
        result = _convert_usage(
            {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        )
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50
        assert result["total_tokens"] == 150

    def test_calculated_total(self):
        result = _convert_usage({"prompt_tokens": 100, "completion_tokens": 50})
        assert result["total_tokens"] == 150

    def test_with_token_details(self):
        result = _convert_usage(
            {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "prompt_tokens_details": {"cached_tokens": 30},
                "completion_tokens_details": {"reasoning_tokens": 20},
            }
        )
        assert result["input_tokens_details"]["cached_tokens"] == 30
        assert result["output_tokens_details"]["reasoning_tokens"] == 20

    def test_default_token_details(self):
        result = _convert_usage({"prompt_tokens": 100, "completion_tokens": 50})
        # Spec: ResponseResource.usage requires both details objects (with
        # their own required inner fields), so zero-value defaults are emitted
        # when the upstream response did not provide them; real values are
        # preserved as-is.
        assert result["input_tokens_details"] == {"cached_tokens": 0}
        assert result["output_tokens_details"] == {"reasoning_tokens": 0}

    def test_token_details_preserved(self):
        result = _convert_usage(
            {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "prompt_tokens_details": {"cached_tokens": 3},
                "completion_tokens_details": {"reasoning_tokens": 7},
            }
        )
        assert result["input_tokens_details"]["cached_tokens"] == 3
        assert result["output_tokens_details"]["reasoning_tokens"] == 7


class TestConvertOpenresponsesRequestToUnified:
    """Tests for OpenResponsesProtocolSerializer.parse_request."""

    def test_simple_string_input(self):
        request = ResponsesRequest(
            model="gpt-4",
            input="Hello",
        )
        result = _serializer.parse_request(request.model_dump())
        assert isinstance(result, InternalRequest)
        assert result.model == "gpt-4"
        assert len(result.conversation.messages) == 1
        assert result.conversation.messages[0].role == "user"

    def test_message_input_user(self):
        request = ResponsesRequest(
            model="gpt-4",
            input=[{"type": "message", "role": "user", "content": "Hello"}],
        )
        result = _serializer.parse_request(request.model_dump())
        assert len(result.conversation.messages) == 1
        assert result.conversation.messages[0].role == "user"

    def test_message_input_system(self):
        request = ResponsesRequest(
            model="gpt-4",
            input=[{"type": "message", "role": "system", "content": "You are helpful"}],
        )
        result = _serializer.parse_request(request.model_dump())
        assert len(result.conversation.system_messages) == 1

    def test_function_call_input(self):
        request = ResponsesRequest(
            model="gpt-4",
            input=[
                {
                    "type": "function_call",
                    "call_id": "call_123",
                    "name": "get_weather",
                    "arguments": "{}",
                }
            ],
        )
        result = _serializer.parse_request(request.model_dump())
        assert len(result.conversation.messages) == 1
        assert result.conversation.messages[0].role == "assistant"

    def test_function_call_output_input(self):
        request = ResponsesRequest(
            model="gpt-4",
            input=[{"type": "function_call_output", "call_id": "call_123", "output": "result"}],
        )
        result = _serializer.parse_request(request.model_dump())
        assert len(result.conversation.messages) == 1
        assert result.conversation.messages[0].role == "tool"

    def test_function_call_output_array_content_items(self):
        """Codex sends function_call_output.output as an array of content items
        (input_text / input_image / encrypted_content), not just a plain string.

        This is the shape that triggered the 422 ``Input should be a valid
        string`` rejection at the /v1/responses endpoint. The parser must accept
        it and flatten input_text items into the tool result text.
        """
        from llm_proxy.models.content_blocks import ToolResultBlock

        request = ResponsesRequest(
            model="gpt-4",
            input=[
                {
                    "type": "function_call_output",
                    "call_id": "call_123",
                    "output": [
                        {"type": "input_text", "text": "ID  NAME  STATUS"},
                        {"type": "input_text", "text": "1   ci    failure"},
                        {"type": "input_image", "image_url": "data:..."},
                        {"type": "encrypted_content", "encrypted_content": "blob"},
                    ],
                }
            ],
        )
        result = _serializer.parse_request(request.model_dump())
        assert len(result.conversation.messages) == 1
        msg = result.conversation.messages[0]
        assert msg.role == "tool"
        block = msg.content[0]
        assert isinstance(block, ToolResultBlock)
        # input_text items are joined with newlines; image/encrypted are skipped.
        assert block.content == "ID  NAME  STATUS\n1   ci    failure"

    def test_codex_turn2_payload_validates_and_parses(self):
        """Regression: a realistic Codex turn-2 /v1/responses payload (developer +
        user + reasoning + function_call + function_call_output with array
        output) must not trigger a 422 and must parse to a tool result."""
        from llm_proxy.models.content_blocks import ToolResultBlock

        request = ResponsesRequest(
            model="deepseek-v4-pro",
            instructions="You are a coding assistant.",
            stream=True,
            store=False,
            parallel_tool_calls=True,
            tool_choice="auto",
            text={"type": "text"},
            reasoning={"effort": "xhigh", "summary": "auto"},
            include=["reasoning.encrypted_content"],
            input=[
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "<permissions instructions>"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "check the latest CI run"}],
                },
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": "Let me run gh to check."}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_00_x",
                    "name": "exec_command",
                    "arguments": '{"cmd": "gh run list --limit 5"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_00_x",
                    "output": [{"type": "input_text", "text": "1  ci  failure"}],
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
        result = _serializer.parse_request(request.model_dump())
        tool_msgs = [m for m in result.conversation.messages if m.role == "tool"]
        assert tool_msgs
        assert isinstance(tool_msgs[0].content[0], ToolResultBlock)
        assert tool_msgs[0].content[0].content == "1  ci  failure"

    def test_local_shell_call_translates_to_tool_use(self):
        from llm_proxy.models.content_blocks import ToolResultBlock, ToolUseBlock

        request = ResponsesRequest(
            model="gpt-4",
            input=[
                {"type": "message", "role": "user", "content": "run ls"},
                {
                    "type": "local_shell_call",
                    "call_id": "call_ls",
                    "status": "completed",
                    "action": {"type": "exec", "command": ["ls", "-la"]},
                },
                {"type": "function_call_output", "call_id": "call_ls", "output": "file1"},
            ],
        )
        result = _serializer.parse_request(request.model_dump())
        roles = [m.role for m in result.conversation.messages]
        assert roles == ["user", "assistant", "tool"]
        use = result.conversation.messages[1].content[0]
        assert isinstance(use, ToolUseBlock)
        assert use.name == "local_shell"
        assert use.input == {"type": "exec", "command": ["ls", "-la"]}
        assert isinstance(result.conversation.messages[2].content[0], ToolResultBlock)

    def test_custom_tool_call_and_output_translated(self):
        from llm_proxy.models.content_blocks import CustomToolUseBlock, ToolResultBlock

        request = ResponsesRequest(
            model="gpt-4",
            input=[
                {
                    "type": "custom_tool_call",
                    "call_id": "call_p",
                    "name": "apply_patch",
                    "input": '{"patch":"begin"}',
                },
                {
                    "type": "custom_tool_call_output",
                    "call_id": "call_p",
                    "output": [{"type": "input_text", "text": "patched"}],
                },
            ],
        )
        result = _serializer.parse_request(request.model_dump())
        roles = [m.role for m in result.conversation.messages]
        assert roles == ["assistant", "tool"]
        use = result.conversation.messages[0].content[0]
        assert isinstance(use, CustomToolUseBlock)
        assert use.name == "apply_patch"
        assert use.input == '{"patch":"begin"}'
        res = result.conversation.messages[1].content[0]
        assert isinstance(res, ToolResultBlock)
        assert res.content == "patched"

    def test_tool_search_call_and_output_translated(self):
        import orjson

        from llm_proxy.models.content_blocks import ToolResultBlock, ToolUseBlock

        request = ResponsesRequest(
            model="gpt-4",
            input=[
                {
                    "type": "tool_search_call",
                    "call_id": "call_ts",
                    "execution": "sync",
                    "arguments": {"query": "shell"},
                },
                {
                    "type": "tool_search_output",
                    "call_id": "call_ts",
                    "status": "completed",
                    "execution": "sync",
                    "tools": [{"name": "exec_command"}],
                },
            ],
        )
        result = _serializer.parse_request(request.model_dump())
        roles = [m.role for m in result.conversation.messages]
        assert roles == ["assistant", "tool"]
        use = result.conversation.messages[0].content[0]
        assert isinstance(use, ToolUseBlock)
        assert use.name == "tool_search"
        assert use.input == {"query": "shell"}
        res = result.conversation.messages[1].content[0]
        assert isinstance(res, ToolResultBlock)
        assert orjson.loads(res.content) == [{"name": "exec_command"}]

    def test_web_search_call_replayed_other_hosted_tools_skipped(self):
        # web_search is exposed to chat-completions providers as a function
        # tool, so a ``web_search_call`` history item is replayed as a
        # ``web_search`` tool_call plus a placeholder tool result (skipping it
        # would leave an empty assistant message that providers reject as
        # "Upstream request failed" on web-search-only turns). The remaining
        # hosted tools / controls (image_generation_call, additional_tools,
        # compaction_trigger) have no Chat Completions equivalent and are still
        # skipped.
        request = ResponsesRequest(
            model="gpt-4",
            input=[
                {"type": "message", "role": "user", "content": "hi"},
                {
                    "type": "web_search_call",
                    "id": "ws_skip1",
                    "status": "completed",
                    "action": {"type": "search", "query": "q"},
                },
                {"type": "image_generation_call", "status": "completed", "result": "img"},
                {"type": "additional_tools", "role": "user", "tools": [{"name": "t"}]},
                {"type": "compaction_trigger"},
                {"type": "message", "role": "user", "content": "bye"},
            ],
        )
        result = _serializer.parse_request(request.model_dump())
        roles = [m.role for m in result.conversation.messages]
        # user(hi) -> assistant(web_search tool_call) -> tool(result) -> user(bye)
        assert roles == ["user", "assistant", "tool", "user"]
        from llm_proxy.models.content_blocks import ToolResultBlock

        assistant = result.conversation.messages[1]
        use = assistant.content[0]
        assert isinstance(use, ToolUseBlock)
        assert use.name == "web_search"
        assert use.input == {"query": "q"}
        assert use.id == "ws_skip1"
        tool_msg = result.conversation.messages[2]
        res = tool_msg.content[0]
        assert isinstance(res, ToolResultBlock)
        assert res.tool_use_id == "ws_skip1"
        assert "Web search performed" in res.content

    def test_agent_message_becomes_user_message(self):
        request = ResponsesRequest(
            model="gpt-4",
            input=[
                {
                    "type": "agent_message",
                    "author": "planner",
                    "recipient": "coder",
                    "content": [{"type": "input_text", "text": "do the thing"}],
                }
            ],
        )
        result = _serializer.parse_request(request.model_dump())
        assert len(result.conversation.messages) == 1
        msg = result.conversation.messages[0]
        assert msg.role == "user"
        assert msg.content[0].text == "do the thing"

    def test_compaction_carries_encrypted_content(self):
        from llm_proxy.models.content_blocks import ThinkingBlock

        request = ResponsesRequest(
            model="gpt-4",
            input=[
                {"type": "compaction", "encrypted_content": "ENC_BLOB"},
                {"type": "message", "role": "user", "content": "continue"},
            ],
        )
        result = _serializer.parse_request(request.model_dump())
        roles = [m.role for m in result.conversation.messages]
        assert roles == ["assistant", "user"]
        block = result.conversation.messages[0].content[0]
        assert isinstance(block, ThinkingBlock)
        assert block.encrypted_content == "ENC_BLOB"

    def test_compaction_summary_alias_accepted(self):
        from llm_proxy.models.content_blocks import ThinkingBlock

        request = ResponsesRequest(
            model="gpt-4",
            input=[{"type": "compaction_summary", "encrypted_content": "ENC"}],
        )
        result = _serializer.parse_request(request.model_dump())
        block = result.conversation.messages[0].content[0]
        assert isinstance(block, ThinkingBlock)
        assert block.encrypted_content == "ENC"

    def test_unknown_future_item_type_accepted_and_skipped(self):
        request = ResponsesRequest(
            model="gpt-4",
            input=[
                {"type": "message", "role": "user", "content": "hi"},
                {"type": "some_future_call", "payload": 42},
            ],
        )
        result = _serializer.parse_request(request.model_dump())
        roles = [m.role for m in result.conversation.messages]
        assert roles == ["user"]

    def test_with_tools(self):
        request = ResponsesRequest(
            model="gpt-4",
            input="Hello",
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )
        result = _serializer.parse_request(request.model_dump())
        assert result.tools is not None
        assert len(result.tools) == 1

    def test_parses_web_search_tools(self):
        request = ResponsesRequest(
            model="gpt-4",
            input="Hello",
            tools=[{"type": "web_search"}, {"type": "web_search_preview"}],
        )
        result = _serializer.parse_request(request.model_dump())
        assert result.tools is not None
        assert len(result.tools) == 2
        for tool in result.tools:
            assert tool.name == "web_search"
            assert tool.type in ("web_search", "web_search_preview")

    def test_mixed_function_and_web_search_tools(self):
        from llm_proxy.models.tools import FunctionTool, OpenAIWebSearchTool

        request = ResponsesRequest(
            model="gpt-4",
            input="Hello",
            tools=[
                {"type": "web_search"},
                {"type": "function", "name": "get_weather", "parameters": {"type": "object"}},
            ],
        )
        result = _serializer.parse_request(request.model_dump())

        assert len(result.tools) == 2
        assert isinstance(result.tools[0], OpenAIWebSearchTool)
        assert isinstance(result.tools[1], FunctionTool)

    def test_streaming_enabled(self):
        request = ResponsesRequest(
            model="gpt-4",
            input="Hello",
            stream=True,
        )
        result = _serializer.parse_request(request.model_dump())
        assert result.stream is True

    def test_generation_params(self):
        request = ResponsesRequest(
            model="gpt-4",
            input="Hello",
            temperature=0.5,
            top_p=0.9,
            max_output_tokens=100,
        )
        result = _serializer.parse_request(request.model_dump())
        assert result.params.temperature == 0.5
        assert result.params.top_p == 0.9
        assert result.params.max_tokens == 100

    def test_metadata_user(self):
        request = ResponsesRequest(
            model="gpt-4",
            input="Hello",
            metadata={"user": "user123"},
        )
        result = _serializer.parse_request(request.model_dump())
        assert result.user == "user123"


class TestFormatResponse:
    """Tests for OpenResponsesProtocolSerializer.format_response."""

    _serializer = OpenResponsesProtocolSerializer()

    def test_basic_response(self):
        internal = InternalResponse(
            id="resp_123",
            model="gpt-4",
            output=[TextBlock(text="Hello!")],
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            finish_reason="stop",
        )
        result = self._serializer.format_response(internal)
        assert result["id"] == "resp_123"
        assert result["model"] == "gpt-4"
        assert result["status"] == "completed"
        assert len(result["output"]) > 0

    def test_response_with_tool_calls(self):
        internal = InternalResponse(
            id="test",
            model="test",
            output=[ToolUseBlock(id="call_123", name="get_weather", input="{}")],
            finish_reason="stop",
        )
        result = self._serializer.format_response(internal)
        function_calls = [o for o in result["output"] if o["type"] == "function_call"]
        assert len(function_calls) > 0

    def test_response_with_reasoning(self):
        internal = InternalResponse(
            id="test",
            model="test",
            output=[TextBlock(text="Answer"), ThinkingBlock(thinking="Let me think...")],
            finish_reason="stop",
        )
        result = self._serializer.format_response(internal)
        reasoning = [o for o in result["output"] if o["type"] == "reasoning"]
        assert len(reasoning) > 0

    def test_finish_reason_length(self):
        internal = InternalResponse(
            id="test",
            model="test",
            output=[TextBlock(text="Partial...")],
            finish_reason="length",
        )
        result = self._serializer.format_response(internal)
        assert result["status"] == "incomplete"
        assert result["incomplete_details"]["reason"] == "max_output_tokens"

    def test_finish_reason_error(self):
        internal = InternalResponse(
            id="test",
            model="test",
            output=[],
            finish_reason="error",
        )
        result = self._serializer.format_response(internal)
        assert result["status"] == "failed"
        assert result["error"]["code"] == "provider_error"

    def test_custom_request_id(self):
        internal = InternalResponse(
            id="custom_id_123",
            model="test",
            output=[TextBlock(text="test")],
        )
        result = self._serializer.format_response(internal)
        assert result["id"] == "custom_id_123"

    def test_custom_model(self):
        internal = InternalResponse(
            id="test",
            model="custom-model",
            output=[TextBlock(text="test")],
        )
        result = self._serializer.format_response(internal)
        assert result["model"] == "custom-model"

    def test_no_output_fallsback_to_empty_message(self):
        internal = InternalResponse(
            id="test",
            model="gpt-4",
            output=[],
        )
        result = self._serializer.format_response(internal)
        assert len(result["output"]) > 0
        assert result["output"][0]["type"] == "message"

    def test_usage_conversion(self):
        internal = InternalResponse(
            id="test",
            model="test",
            output=[TextBlock(text="test")],
            usage=Usage(input_tokens=100, output_tokens=50, total_tokens=150),
        )
        result = self._serializer.format_response(internal)
        assert result["usage"]["input_tokens"] == 100
        assert result["usage"]["output_tokens"] == 50
        assert result["usage"]["total_tokens"] == 150

    def test_server_tool_use_block(self):
        from llm_proxy.models import ServerToolUseBlock

        internal = InternalResponse(
            id="test",
            model="test",
            output=[ServerToolUseBlock(id="stu_1", name="web_search", input={"query": "test"})],
            finish_reason="stop",
        )
        result = self._serializer.format_response(internal)
        assert len(result["output"]) > 0

    def test_web_search_tool_result_block(self):
        from llm_proxy.models.content_blocks.anthropic_builtin import WebSearchToolResultBlock

        internal = InternalResponse(
            id="test",
            model="test",
            output=[WebSearchToolResultBlock(tool_use_id="stu_1", content="results")],
            finish_reason="stop",
        )
        result = self._serializer.format_response(internal)
        assert len(result["output"]) > 0

    def test_web_fetch_tool_result_block(self):
        from llm_proxy.models.content_blocks.anthropic_builtin import WebFetchToolResultBlock

        internal = InternalResponse(
            id="test",
            model="test",
            output=[WebFetchToolResultBlock(tool_use_id="stu_1", content="fetched content")],
            finish_reason="stop",
        )
        result = self._serializer.format_response(internal)
        assert len(result["output"]) > 0

    def test_code_execution_tool_result_block(self):
        from llm_proxy.models.content_blocks.anthropic_builtin import CodeExecutionToolResultBlock

        internal = InternalResponse(
            id="test",
            model="test",
            output=[CodeExecutionToolResultBlock(tool_use_id="tool_1", content="output")],
            finish_reason="stop",
        )
        result = self._serializer.format_response(internal)
        assert len(result["output"]) > 0

    def test_bash_code_execution_tool_result_block(self):
        from llm_proxy.models.content_blocks.anthropic_builtin import (
            BashCodeExecutionToolResultBlock,
        )

        internal = InternalResponse(
            id="test",
            model="test",
            output=[BashCodeExecutionToolResultBlock(tool_use_id="tool_1", content="output")],
            finish_reason="stop",
        )
        result = self._serializer.format_response(internal)
        assert len(result["output"]) > 0

    def test_text_editor_code_execution_tool_result_block(self):
        from llm_proxy.models.content_blocks.anthropic_builtin import (
            TextEditorCodeExecutionToolResultBlock,
        )

        internal = InternalResponse(
            id="test",
            model="test",
            output=[TextEditorCodeExecutionToolResultBlock(tool_use_id="tool_1", content="output")],
            finish_reason="stop",
        )
        result = self._serializer.format_response(internal)
        assert len(result["output"]) > 0

    def test_tool_search_tool_result_block(self):
        from llm_proxy.models.content_blocks.anthropic_builtin import ToolSearchToolResultBlock

        internal = InternalResponse(
            id="test",
            model="test",
            output=[ToolSearchToolResultBlock(tool_use_id="tool_1", content="results")],
            finish_reason="stop",
        )
        result = self._serializer.format_response(internal)
        assert len(result["output"]) > 0

    def test_redacted_thinking_block(self):
        from llm_proxy.models import RedactedThinkingBlock

        internal = InternalResponse(
            id="test",
            model="test",
            output=[RedactedThinkingBlock(data="redacted_data")],
            finish_reason="stop",
        )
        result = self._serializer.format_response(internal)
        assert len(result["output"]) > 0

    def test_refusal_block(self):
        from llm_proxy.models import RefusalBlock

        internal = InternalResponse(
            id="test",
            model="test",
            output=[TextBlock(text="I cannot..."), RefusalBlock(refusal="I cannot do that")],
            finish_reason="stop",
        )
        result = self._serializer.format_response(internal)
        assert len(result["output"]) > 0

    def test_custom_tool_use_block(self):
        from llm_proxy.models import CustomToolUseBlock

        internal = InternalResponse(
            id="test",
            model="test",
            output=[CustomToolUseBlock(id="custom_1", name="custom_tool", input="{}")],
            finish_reason="stop",
        )
        result = self._serializer.format_response(internal)
        assert len(result["output"]) > 0

    def test_short_name_tool_use_block_matched_as_custom(self):
        """Model echoing the short history name (``exec``) instead of the
        flattened definition name (``functions__exec``) must still emit a
        custom_tool_call with unwrapped ``input`` — the custom-name check is
        a tolerant suffix match, not exact.
        """
        from llm_proxy.protocols.openresponses.handler import (
            clear_format_context,
            set_format_context,
        )

        set_format_context(
            {
                "tools": [
                    {
                        "type": "namespace",
                        "name": "functions",
                        "tools": [{"type": "custom", "name": "exec", "description": "Run JS"}],
                    }
                ]
            }
        )
        try:
            internal = InternalResponse(
                id="test",
                model="test",
                output=[
                    ToolUseBlock(
                        id="call_1",
                        name="exec",
                        input={"content": "await tools.exec_command({cmd: 'ls'})"},
                    )
                ],
                finish_reason="tool_calls",
            )
            result = self._serializer.format_response(internal)
        finally:
            clear_format_context()

        calls = [o for o in result["output"] if o["type"] == "custom_tool_call"]
        assert len(calls) == 1
        assert calls[0]["name"] == "exec"
        assert calls[0]["input"] == "await tools.exec_command({cmd: 'ls'})"
        assert "arguments" not in calls[0]

    def test_mixed_anthropic_blocks_with_openai_blocks(self):
        from llm_proxy.models import ServerToolUseBlock
        from llm_proxy.models.content_blocks.anthropic_builtin import WebSearchToolResultBlock

        internal = InternalResponse(
            id="test",
            model="test",
            output=[
                ThinkingBlock(thinking="Let me search..."),
                ServerToolUseBlock(id="stu_1", name="web_search", input={"query": "test"}),
                WebSearchToolResultBlock(tool_use_id="stu_1", content="results"),
                TextBlock(text="Based on search results..."),
            ],
            finish_reason="stop",
        )
        result = self._serializer.format_response(internal)
        assert len(result["output"]) >= 3

    def test_multiple_raw_output_items_keep_relative_order(self):
        """Raw items re-insert at their recorded positions, accounting for
        the shift introduced by earlier insertions."""
        internal = InternalResponse(
            id="resp_raw",
            model="gpt-4",
            output=[
                TextBlock(text="first"),
                ThinkingBlock(thinking="hmm"),
                ToolUseBlock(id="call_1", name="f1", input={}),
                TextBlock(text="second"),
                ToolUseBlock(id="call_2", name="f2", input={}),
            ],
            finish_reason="stop",
            raw_output=[
                (1, {"type": "local_shell_call", "id": "lsc_1"}),
                (3, {"type": "agent_message", "id": "am_1"}),
            ],
        )
        result = self._serializer.format_response(internal)
        assert [o["type"] for o in result["output"]] == [
            "message",
            "local_shell_call",
            "reasoning",
            "function_call",
            "agent_message",
            "message",
            "function_call",
        ]


class TestFormatResponseWithImageFileBlocks:
    """Tests for format_response with ImageBlock, FileBlock, DocumentBlock output blocks."""

    _serializer = OpenResponsesProtocolSerializer()

    def test_image_block_base64(self):
        from llm_proxy.models.content_blocks import ImageBlock
        from llm_proxy.models.types import ImageSource

        internal = InternalResponse(
            id="test",
            model="test",
            output=[
                ImageBlock(
                    source=ImageSource(type="base64", data="AAAA", media_type="image/png"),
                )
            ],
        )
        result = self._serializer.format_response(internal)
        output = result["output"]
        assert len(output) == 1
        assert output[0]["type"] == "message"
        assert output[0]["content"][0]["type"] == "output_text"
        assert "image/png" in output[0]["content"][0]["text"]

    def test_image_block_file_id(self):
        from llm_proxy.models.content_blocks import ImageBlock
        from llm_proxy.models.types import ImageSource

        internal = InternalResponse(
            id="test",
            model="test",
            output=[
                ImageBlock(
                    source=ImageSource(type="file_id", data="file_img_123", media_type="image/png"),
                )
            ],
        )
        result = self._serializer.format_response(internal)
        assert len(result["output"]) == 1
        assert "file_img_123" in result["output"][0]["content"][0]["text"]

    def test_document_block(self):
        from llm_proxy.models import DocumentBlock
        from llm_proxy.models.types import DocumentSource

        internal = InternalResponse(
            id="test",
            model="test",
            output=[
                DocumentBlock(
                    source=DocumentSource(type="text", data="...", media_type="application/pdf"),
                    title="report.pdf",
                )
            ],
        )
        result = self._serializer.format_response(internal)
        assert len(result["output"]) == 1
        text = result["output"][0]["content"][0]["text"]
        assert "report.pdf" in text
        assert "pdf" in text.lower()

    def test_file_block(self):
        from llm_proxy.models import FileBlock

        internal = InternalResponse(
            id="test",
            model="test",
            output=[FileBlock(file_id="file_xyz", filename="data.csv")],
        )
        result = self._serializer.format_response(internal)
        assert len(result["output"]) == 1
        text = result["output"][0]["content"][0]["text"]
        assert "data.csv" in text

    def test_file_block_minimal(self):
        from llm_proxy.models import FileBlock

        internal = InternalResponse(
            id="test",
            model="test",
            output=[FileBlock(file_id="file_abc")],
        )
        result = self._serializer.format_response(internal)
        assert len(result["output"]) == 1
        text = result["output"][0]["content"][0]["text"]
        assert "file_abc" in text

    def test_mixed_text_and_image(self):
        from llm_proxy.models.content_blocks import ImageBlock
        from llm_proxy.models.types import ImageSource

        internal = InternalResponse(
            id="test",
            model="test",
            output=[
                TextBlock(text="Here is the result:"),
                ImageBlock(
                    source=ImageSource(
                        type="url", data="https://example.com/img.png", media_type=None
                    ),
                ),
            ],
        )
        result = self._serializer.format_response(internal)
        assert len(result["output"]) == 2
        assert result["output"][0]["content"][0]["text"] == "Here is the result:"
        assert result["output"][1]["content"][0]["type"] == "output_text"


class TestIntegration:
    """Integration tests for converters."""

    def test_full_request_conversion(self):
        request = ResponsesRequest(
            model="gpt-4",
            input=[
                {"type": "message", "role": "system", "content": "Be helpful"},
                {"type": "message", "role": "user", "content": "Hi"},
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            tool_choice="auto",
            temperature=0.7,
            stream=True,
        )
        result = _serializer.parse_request(request.model_dump())
        assert result.model == "gpt-4"
        assert len(result.conversation.system_messages) == 1
        assert len(result.conversation.messages) == 1
        assert len(result.tools) == 1
        assert result.params.temperature == 0.7
        assert result.stream is True


class TestMatchCustomToolName:
    """match_custom_tool_name is deterministic, even on suffix collisions."""

    def test_exact_match_preferred(self):
        from llm_proxy.protocols.openresponses.serializer import match_custom_tool_name

        assert match_custom_tool_name("b__exec", {"a__exec", "b__exec"}) == "b__exec"

    def test_short_name_match_deterministic(self):
        from llm_proxy.protocols.openresponses.serializer import match_custom_tool_name

        # Two namespaces declare the same short name: sorted candidate order
        # keeps the result stable across processes (set iteration order does
        # not leak into the match).
        result = match_custom_tool_name("exec", {"b__exec", "a__exec"})
        assert result == "a__exec"

    def test_single_suffix_match(self):
        from llm_proxy.protocols.openresponses.serializer import match_custom_tool_name

        assert match_custom_tool_name("exec", {"functions__exec", "shell__ls"}) == (
            "functions__exec"
        )

    def test_no_match_returns_none(self):
        from llm_proxy.protocols.openresponses.serializer import match_custom_tool_name

        assert match_custom_tool_name("other", {"a__exec"}) is None


class TestConversationToInputItemsServerTools:
    """ServerToolUseBlock round-trips to its native item types."""

    def _convert(self, block):
        from llm_proxy.models import ConversationContext, Message
        from llm_proxy.protocols.openresponses.serializer import (
            conversation_to_input_items,
        )

        conversation = ConversationContext(messages=[Message(role="assistant", content=[block])])
        return conversation_to_input_items(conversation)

    def test_web_search_uses_upstream_action(self):
        from llm_proxy.models.content_blocks import ServerToolUseBlock

        block = ServerToolUseBlock(
            id="ws_1",
            name="web_search",
            input={"query": "news"},
            extra={
                "responses_action": {
                    "type": "search",
                    "query": "news",
                    "queries": ["news"],
                }
            },
        )
        items = self._convert(block)
        assert items == [
            {
                "type": "web_search_call",
                "id": "ws_1",
                "status": "completed",
                "action": {"type": "search", "query": "news", "queries": ["news"]},
            }
        ]

    def test_web_search_action_synthesized_without_extra(self):
        from llm_proxy.models.content_blocks import ServerToolUseBlock

        block = ServerToolUseBlock(id="ws_2", name="web_search", input={"query": "weather"})
        items = self._convert(block)
        assert items[0]["type"] == "web_search_call"
        assert items[0]["action"] == {
            "type": "search",
            "query": "weather",
            "queries": ["weather"],
        }

    def test_tool_search_round_trip(self):
        from llm_proxy.models.content_blocks import ServerToolUseBlock

        block = ServerToolUseBlock(
            id="ts_1", name="tool_search", input={"arguments": {"query": "files"}}
        )
        items = self._convert(block)
        assert items == [
            {
                "type": "tool_search_call",
                "id": "ts_1",
                "arguments": {"query": "files"},
            }
        ]

    def test_other_server_tool_falls_back_to_custom_tool_call(self):
        from llm_proxy.models.content_blocks import ServerToolUseBlock

        block = ServerToolUseBlock(id="st_1", name="code_interpreter", input={"code": "1+1"})
        items = self._convert(block)
        assert items[0]["type"] == "custom_tool_call"
        assert items[0]["name"] == "code_interpreter"

    def test_web_search_round_trip_through_dispatch(self):
        """The emitted web_search_call item parses back to a web search call."""
        from llm_proxy.models.content_blocks import ServerToolUseBlock

        block = ServerToolUseBlock(id="ws_3", name="web_search", input={"query": "docs"})
        items = self._convert(block)
        request = {"model": "gpt-4", "input": items}
        parsed = _serializer.parse_request(request)
        tool_uses = [
            b
            for m in parsed.conversation.messages
            for b in m.content
            if isinstance(b, ToolUseBlock)
        ]
        assert any(b.name == "web_search" and b.input.get("query") == "docs" for b in tool_uses)
