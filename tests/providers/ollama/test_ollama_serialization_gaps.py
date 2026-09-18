"""Tests for Ollama serialization gaps.

TDD: Tests written before implementation. Covers:
- Build side: unhandled block types are text-degraded instead of silently dropped
- Parse side: message.images are converted to ImageBlock
"""

from llm_proxy.models import (
    ConversationContext,
    ImageBlock,
    Message,
    ServerToolUseBlock,
    TextBlock,
)
from llm_proxy.models.tools import CustomTool, FunctionTool
from llm_proxy.serialization.context import BuildContext
from llm_proxy.serialization.ollama.serializer import OllamaProviderSerializer


class TestOllamaBuildSideGaps:
    """Unhandled blocks are text-degraded when policy is degrade."""

    def _build_message(self, blocks):
        """Helper to build an Ollama message dict from content blocks."""
        from llm_proxy.models import ConversationContext, Message

        serializer = OllamaProviderSerializer()
        ctx = BuildContext(unsupported_block_policy="degrade")
        conv = ConversationContext(
            messages=[
                Message(role="user", content=[TextBlock(text="hi")]),
                Message(role="assistant", content=blocks),
            ]
        )
        result = serializer._convert_conversation_to_ollama(conv, ctx)
        for msg in result:
            if msg.get("role") == "assistant":
                return msg
        return {}

    def test_server_tool_use_block_degraded(self):
        """ServerToolUseBlock is text-degraded (not converted to tool_calls)."""
        msg = self._build_message(
            [
                ServerToolUseBlock(id="stu_1", name="web_search", input={"query": "weather"}),
            ]
        )
        content = msg.get("content", "")
        assert "web_search" in content.lower()

    def test_redacted_thinking_block_converted_to_thinking(self):
        """RedactedThinkingBlock is converted to ThinkingBlock and rendered as thinking."""
        from llm_proxy.models import RedactedThinkingBlock

        msg = self._build_message(
            [
                RedactedThinkingBlock(data="redacted_data"),
            ]
        )
        assert msg.get("thinking") == "[Redacted thinking]"

    def test_custom_tool_use_block_wrapped_as_function_call(self):
        """CustomToolUseBlock is re-wrapped into the {"content": ...} bridge envelope."""
        from llm_proxy.models import CustomToolUseBlock

        msg = self._build_message(
            [
                CustomToolUseBlock(id="ctu_1", name="custom_fn", input='{"key": "value"}'),
            ]
        )
        tool_calls = msg.get("tool_calls")
        assert tool_calls and len(tool_calls) == 1
        function = tool_calls[0]["function"]
        assert function["name"] == "custom_fn"
        assert function["arguments"] == {"content": '{"key": "value"}'}

    def test_web_search_tool_result_as_tool_message(self):
        """WebSearchToolResultBlock is converted to a tool role message."""
        from llm_proxy.models.content_blocks.anthropic_builtin import WebSearchToolResultBlock

        serializer = OllamaProviderSerializer()
        conv = ConversationContext(
            messages=[
                Message(role="user", content=[TextBlock(text="hi")]),
                Message(
                    role="assistant",
                    content=[
                        WebSearchToolResultBlock(tool_use_id="ws_1", content="search results here"),
                    ],
                ),
            ]
        )
        result = serializer._convert_conversation_to_ollama(conv)
        tool_msgs = [m for m in result if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert "search results here" in tool_msgs[0].get("content", "")

    def test_code_execution_result_as_tool_message(self):
        """CodeExecutionToolResultBlock is converted to a tool role message."""
        from llm_proxy.models.content_blocks.anthropic_builtin import CodeExecutionToolResultBlock

        serializer = OllamaProviderSerializer()
        conv = ConversationContext(
            messages=[
                Message(role="user", content=[TextBlock(text="hi")]),
                Message(
                    role="assistant",
                    content=[
                        CodeExecutionToolResultBlock(tool_use_id="ce_1", content="code output"),
                    ],
                ),
            ]
        )
        result = serializer._convert_conversation_to_ollama(conv)
        tool_msgs = [m for m in result if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert "code output" in tool_msgs[0].get("content", "")

    def test_tool_reference_block_degraded(self):
        """ToolReferenceBlock has no conversion and is text-degraded."""
        from llm_proxy.models.content_blocks.anthropic_builtin import ToolReferenceBlock

        msg = self._build_message(
            [
                ToolReferenceBlock(tool_id="tool_1", tool_name="my_tool"),
            ]
        )
        content = msg.get("content", "")
        assert "tool" in content.lower() or "my_tool" in content


class TestOllamaParseSideImageBlock:
    """Ollama response message.images is converted to ImageBlock."""

    def test_images_converted_to_image_blocks(self):
        """message.images produces ImageBlock entries."""
        serializer = OllamaProviderSerializer()
        provider_response = {
            "model": "llava",
            "message": {
                "role": "assistant",
                "content": "Here is the image:",
                "images": ["base64imagedata"],
            },
            "done_reason": "stop",
        }
        response = serializer.parse_provider_response(provider_response, model="llava")
        image_blocks = [b for b in response.output if isinstance(b, ImageBlock)]
        assert len(image_blocks) == 1
        assert image_blocks[0].source.type == "base64"
        assert image_blocks[0].source.data == "base64imagedata"

    def test_images_without_text(self):
        """message.images without content still produces ImageBlock."""
        serializer = OllamaProviderSerializer()
        provider_response = {
            "model": "llava",
            "message": {
                "role": "assistant",
                "content": "",
                "images": ["img1", "img2"],
            },
            "done_reason": "stop",
        }
        response = serializer.parse_provider_response(provider_response, model="llava")
        image_blocks = [b for b in response.output if isinstance(b, ImageBlock)]
        assert len(image_blocks) == 2
        assert image_blocks[0].source.data == "img1"
        assert image_blocks[1].source.data == "img2"


class TestOllamaCustomToolGrammar:
    """CustomTool grammar is embedded so freeform tools stay usable.

    Codex declares freeform tools (``apply_patch``, ``exec``) as Responses
    ``type: custom`` with a ``format`` grammar. Ollama has no native grammar
    support, so the tool is bridged to a function tool; the grammar must be
    surfaced in the description (mirroring the OpenAI Chat Completions bridge)
    or the model has no way to know the expected freeform format.
    """

    def _build_tools(self, tools):
        from llm_proxy.models import InternalRequest
        from llm_proxy.serialization.context import BuildContext

        serializer = OllamaProviderSerializer()
        request = InternalRequest(
            model="qwen3-coder",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            tools=tools,
        )
        body = serializer.build_provider_request(
            request, BuildContext(provider_name="ollama", model="qwen3-coder")
        )
        return body.get("tools", [])

    def test_custom_tool_grammar_embedded_in_description(self):
        tool = CustomTool(
            name="apply_patch",
            description="Use apply_patch.",
            format_type="grammar",
            grammar_definition="start: begin_patch hunk+ end_patch",
            grammar_syntax="lark",
        )
        tools = self._build_tools([tool])
        functions = [t["function"] for t in tools]
        assert functions[0]["name"] == "apply_patch"
        assert functions[0]["parameters"]["required"] == ["content"]
        description = functions[0]["description"]
        assert "Use apply_patch." in description
        assert "start: begin_patch hunk+ end_patch" in description
        assert "```lark" in description

    def test_custom_tool_without_grammar_keeps_description(self):
        tool = CustomTool(name="exec", description="Run JS.")
        tools = self._build_tools([tool])
        assert tools[0]["function"]["description"] == "Run JS."

    def test_function_tool_description_unchanged(self):
        tool = FunctionTool(
            name="shell",
            description="Run a shell command",
            parameters={"type": "object", "properties": {}},
        )
        tools = self._build_tools([tool])
        assert tools[0]["function"]["description"] == "Run a shell command"
