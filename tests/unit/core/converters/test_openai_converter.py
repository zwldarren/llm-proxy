# tests/unit/core/converters/test_openai_converter.py
"""Tests for format_conversation function."""

from llm_proxy.models import (
    ConversationContext,
    Message,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from llm_proxy.serialization.openai import format_conversation


class TestFormatConversation:
    def test_simple_conversation(self):
        """Simple conversation should convert to OpenAI format."""
        conv = ConversationContext(
            system_messages=[SystemMessage.from_text(role="system", text="You are helpful.")],
            messages=[
                Message(role="user", content=[TextBlock(text="Hello")]),
                Message(role="assistant", content=[TextBlock(text="Hi there!")]),
            ],
        )
        messages = format_conversation(conv)

        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello"
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "Hi there!"

    def test_user_message_with_tool_result_converts_to_separate_tool_message(self):
        """User message containing ToolResultBlock should produce separate tool role message.

        This is the key fix for the bug where tool results were silently dropped
        when converting to OpenAI format. In OpenAI format, tool results must be
        separate messages with role="tool", not embedded in user messages.
        """
        conv = ConversationContext(
            messages=[
                Message(role="user", content=[TextBlock(text="What is the weather?")]),
                Message(
                    role="assistant",
                    content=[
                        TextBlock(text="Let me check."),
                        ToolUseBlock(id="call_123", name="get_weather", input={"location": "SF"}),
                    ],
                ),
                Message(
                    role="user",
                    content=[ToolResultBlock(tool_use_id="call_123", content="Sunny, 72°F")],
                ),
            ],
        )
        messages = format_conversation(conv)

        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "What is the weather?"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == [{"type": "text", "text": "Let me check."}]
        assert messages[1]["tool_calls"][0]["id"] == "call_123"
        assert messages[2]["role"] == "tool"
        assert messages[2]["tool_call_id"] == "call_123"
        assert messages[2]["content"] == "Sunny, 72°F"

    def test_user_message_with_mixed_tool_result_and_text(self):
        """User message with both text and tool_result should split into user + tool messages."""
        conv = ConversationContext(
            messages=[
                Message(
                    role="user",
                    content=[
                        TextBlock(text="Here's the result:"),
                        ToolResultBlock(tool_use_id="call_456", content="Success"),
                    ],
                ),
            ],
        )
        messages = format_conversation(conv)

        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Here's the result:"
        assert messages[1]["role"] == "tool"
        assert messages[1]["tool_call_id"] == "call_456"
        assert messages[1]["content"] == "Success"

    def test_user_message_with_multiple_tool_results(self):
        """User message with multiple tool_results should produce multiple tool messages."""
        conv = ConversationContext(
            messages=[
                Message(
                    role="user",
                    content=[
                        ToolResultBlock(tool_use_id="call_1", content="Result 1"),
                        ToolResultBlock(tool_use_id="call_2", content="Result 2"),
                    ],
                ),
            ],
        )
        messages = format_conversation(conv)

        assert len(messages) == 2
        assert messages[0]["role"] == "tool"
        assert messages[0]["tool_call_id"] == "call_1"
        assert messages[0]["content"] == "Result 1"
        assert messages[1]["role"] == "tool"
        assert messages[1]["tool_call_id"] == "call_2"
        assert messages[1]["content"] == "Result 2"

    def test_tool_result_with_error(self):
        """Tool result with is_error=True should prefix content with 'Error:'."""
        conv = ConversationContext(
            messages=[
                Message(
                    role="user",
                    content=[
                        ToolResultBlock(tool_use_id="call_err", content="Not found", is_error=True)
                    ],
                ),
            ],
        )
        messages = format_conversation(conv)

        assert len(messages) == 1
        assert messages[0]["role"] == "tool"
        assert messages[0]["tool_call_id"] == "call_err"
        assert messages[0]["content"] == "Error: Not found"

    def test_tool_role_message_preserved(self):
        """Existing tool role messages should be converted correctly."""
        conv = ConversationContext(
            messages=[
                Message(
                    role="tool",
                    content=[ToolResultBlock(tool_use_id="call_789", content="Done")],
                    name="my_tool",
                ),
            ],
        )
        messages = format_conversation(conv)

        assert len(messages) == 1
        assert messages[0]["role"] == "tool"
        assert messages[0]["tool_call_id"] == "call_789"
        assert messages[0]["content"] == "Done"
        assert messages[0]["name"] == "my_tool"

    def test_web_search_tool_result_converts_to_tool_message(self):
        """WebSearchToolResultBlock in user message must produce role='tool' message.

        In Anthropic format, web_search_tool_result blocks appear in user messages.
        When converting to OpenAI format for non-Anthropic providers, they must
        be separated into tool role messages, same as regular ToolResultBlock.
        """
        from llm_proxy.models.content_blocks.anthropic_builtin import WebSearchToolResultBlock

        conv = ConversationContext(
            messages=[
                Message(role="user", content=[TextBlock(text="Search for docs")]),
                Message(
                    role="assistant",
                    content=[
                        TextBlock(text="Let me search."),
                        ToolUseBlock(
                            id="call_ws_1",
                            name="web_search",
                            input={"query": "latest docs"},
                        ),
                    ],
                ),
                Message(
                    role="user",
                    content=[
                        WebSearchToolResultBlock(
                            tool_use_id="call_ws_1",
                            content=[TextBlock(text="Found: Example Docs")],
                        ),
                    ],
                ),
            ],
        )
        messages = format_conversation(conv)

        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Search for docs"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["tool_calls"][0]["id"] == "call_ws_1"
        assert messages[2]["role"] == "tool"
        assert messages[2]["tool_call_id"] == "call_ws_1"

    def test_web_search_tool_result_with_error(self):
        """WebSearchToolResultBlock with is_error=True must produce error-prefixed content."""
        from llm_proxy.models.content_blocks.anthropic_builtin import WebSearchToolResultBlock

        conv = ConversationContext(
            messages=[
                Message(
                    role="user",
                    content=[
                        WebSearchToolResultBlock(
                            tool_use_id="call_ws_err",
                            content="Search failed",
                            is_error=True,
                        ),
                    ],
                ),
            ],
        )
        messages = format_conversation(conv)

        assert len(messages) == 1
        assert messages[0]["role"] == "tool"
        assert messages[0]["tool_call_id"] == "call_ws_err"
        assert messages[0]["content"] == "Error: Search failed"

    def test_web_search_tool_result_with_search_result_content_blocks(self):
        """WebSearchToolResultBlock with WebSearchResultContentBlock children.

        WebSearchResultContentBlock items inside a web_search_tool_result must
        be converted to text representations like [Title](URL).
        """
        from llm_proxy.models.content_blocks.anthropic_builtin import (
            WebSearchResultContentBlock,
            WebSearchToolResultBlock,
        )

        conv = ConversationContext(
            messages=[
                Message(
                    role="user",
                    content=[
                        WebSearchToolResultBlock(
                            tool_use_id="call_ws_2",
                            content=[
                                WebSearchResultContentBlock(
                                    url="https://example.com",
                                    title="Example Site",
                                    encoded_content="base64...",
                                ),
                                WebSearchResultContentBlock(
                                    url="https://docs.example.com",
                                    title="Documentation",
                                    encoded_content="base64...",
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        )
        messages = format_conversation(conv)

        assert len(messages) == 1
        assert messages[0]["role"] == "tool"
        assert messages[0]["tool_call_id"] == "call_ws_2"
        assert "[Example Site](https://example.com)" in messages[0]["content"]
        assert "[Documentation](https://docs.example.com)" in messages[0]["content"]

    def test_server_tool_use_converts_to_tool_call(self):
        """ServerToolUseBlock in assistant message must convert to tool_calls.

        When an Anthropic conversation history contains server_tool_use blocks
        (e.g., from web_search), they must be converted to OpenAI tool_calls
        format when targeting OpenAI-compatible providers.
        """
        from llm_proxy.models import ServerToolUseBlock

        conv = ConversationContext(
            messages=[
                Message(
                    role="assistant",
                    content=[
                        TextBlock(text="Let me search for that."),
                        ServerToolUseBlock(
                            id="toolu_ws_srv",
                            name="web_search",
                            input={"query": "latest docs"},
                        ),
                    ],
                ),
            ],
        )
        messages = format_conversation(conv)

        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert "tool_calls" in messages[0]
        assert len(messages[0]["tool_calls"]) == 1
        tool_call = messages[0]["tool_calls"][0]
        assert tool_call["id"] == "toolu_ws_srv"
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "web_search"
        assert tool_call["function"]["arguments"] == '{"query":"latest docs"}'

    def test_mixed_tool_result_and_web_search_tool_result(self):
        """Both ToolResultBlock and WebSearchToolResultBlock in same message.

        Both should be converted to separate tool role messages.
        """
        from llm_proxy.models.content_blocks.anthropic_builtin import WebSearchToolResultBlock

        conv = ConversationContext(
            messages=[
                Message(
                    role="user",
                    content=[
                        ToolResultBlock(tool_use_id="call_func", content="Function result"),
                        WebSearchToolResultBlock(tool_use_id="call_ws", content="Search result"),
                    ],
                ),
            ],
        )
        messages = format_conversation(conv)

        assert len(messages) == 2
        assert messages[0]["role"] == "tool"
        assert messages[0]["tool_call_id"] == "call_func"
        assert messages[0]["content"] == "Function result"
        assert messages[1]["role"] == "tool"
        assert messages[1]["tool_call_id"] == "call_ws"
        assert messages[1]["content"] == "Search result"

    def test_full_web_search_conversation_round_trip(self):
        """Complete multi-turn web search conversation must convert without data loss.

        Simulates a full Anthropic conversation with web_search tool calls and
        results, verifying the entire conversation converts to valid OpenAI format.
        """
        from llm_proxy.models import ServerToolUseBlock
        from llm_proxy.models.content_blocks.anthropic_builtin import WebSearchToolResultBlock

        conv = ConversationContext(
            system_messages=[SystemMessage.from_text(role="system", text="You are helpful.")],
            messages=[
                Message(role="user", content=[TextBlock(text="What's new in Python 3.14?")]),
                Message(
                    role="assistant",
                    content=[
                        TextBlock(text="Let me search for that."),
                        ServerToolUseBlock(
                            id="toolu_001",
                            name="web_search",
                            input={"query": "Python 3.14 new features"},
                        ),
                    ],
                ),
                Message(
                    role="user",
                    content=[
                        WebSearchToolResultBlock(
                            tool_use_id="toolu_001",
                            content=[
                                TextBlock(
                                    text="Python 3.14 introduces several new features including..."
                                ),
                            ],
                        ),
                    ],
                ),
                Message(
                    role="assistant",
                    content=[
                        TextBlock(text="Python 3.14 introduces pattern matching improvements..."),
                    ],
                ),
            ],
        )
        messages = format_conversation(conv)

        # System + 5 messages (user, assistant+tool, tool_result, assistant)
        assert len(messages) == 5, f"Expected 5 messages, got {len(messages)}: {messages}"
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful."

        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "What's new in Python 3.14?"

        assert messages[2]["role"] == "assistant"
        assert "tool_calls" in messages[2]
        assert len(messages[2]["tool_calls"]) == 1
        assert messages[2]["tool_calls"][0]["function"]["name"] == "web_search"

        assert messages[3]["role"] == "tool"
        assert messages[3]["tool_call_id"] == "toolu_001"
        assert "Python 3.14" in messages[3]["content"]

    def test_assistant_message_with_web_search_tool_result(self):
        """Assistant message with WebSearchToolResultBlock must produce tool messages.

        Anthropic API returns server_tool_use results in the assistant message.
        When converting to OpenAI format, WebSearchToolResultBlock blocks must be
        extracted into separate tool role messages to preserve the conversation context.
        """
        from llm_proxy.models import ServerToolUseBlock
        from llm_proxy.models.content_blocks.anthropic_builtin import WebSearchToolResultBlock

        conv = ConversationContext(
            messages=[
                Message(
                    role="assistant",
                    content=[
                        TextBlock(text="Let me search for that."),
                        ServerToolUseBlock(
                            id="toolu_001",
                            name="web_search",
                            input={"query": "test"},
                        ),
                        WebSearchToolResultBlock(
                            tool_use_id="toolu_001",
                            content=[TextBlock(text="Search results here")],
                        ),
                    ],
                ),
            ],
        )
        messages = format_conversation(conv)

        assert len(messages) == 2, f"Expected 2 messages, got {len(messages)}: {messages}"
        assert messages[0]["role"] == "assistant"
        assert "tool_calls" in messages[0]
        assert messages[1]["role"] == "tool"
        assert messages[1]["tool_call_id"] == "toolu_001"

    def test_assistant_message_with_tool_result(self):
        """Assistant message with ToolResultBlock must produce tool messages.

        Similar to WebSearchToolResultBlock, regular ToolResultBlock in assistant
        messages must be extracted to preserve the conversation context.
        """
        conv = ConversationContext(
            messages=[
                Message(
                    role="assistant",
                    content=[
                        TextBlock(text="Let me check."),
                        ToolUseBlock(
                            id="call_123",
                            name="get_weather",
                            input={"location": "SF"},
                        ),
                        ToolResultBlock(
                            tool_use_id="call_123",
                            content="Sunny, 72°F",
                        ),
                    ],
                ),
            ],
        )
        messages = format_conversation(conv)

        assert len(messages) == 2, f"Expected 2 messages, got {len(messages)}: {messages}"
        assert messages[0]["role"] == "assistant"
        assert messages[0]["tool_calls"][0]["id"] == "call_123"
        assert messages[1]["role"] == "tool"
        assert messages[1]["tool_call_id"] == "call_123"
        assert messages[1]["content"] == "Sunny, 72°F"

    def test_assistant_message_with_multiple_web_search_results(self):
        """Assistant message with multiple WebSearchToolResultBlocks.

        Each web_search result must become a separate tool role message with
        its own tool_call_id, preventing the 'Duplicate value for tool_call_id' error.
        """
        from llm_proxy.models import ServerToolUseBlock
        from llm_proxy.models.content_blocks.anthropic_builtin import WebSearchToolResultBlock

        conv = ConversationContext(
            messages=[
                Message(
                    role="assistant",
                    content=[
                        TextBlock(text="Let me search multiple sources."),
                        ServerToolUseBlock(
                            id="toolu_ws_1",
                            name="web_search",
                            input={"query": "query 1"},
                        ),
                        WebSearchToolResultBlock(
                            tool_use_id="toolu_ws_1",
                            content=[TextBlock(text="Results for query 1")],
                        ),
                        ServerToolUseBlock(
                            id="toolu_ws_2",
                            name="web_search",
                            input={"query": "query 2"},
                        ),
                        WebSearchToolResultBlock(
                            tool_use_id="toolu_ws_2",
                            content=[TextBlock(text="Results for query 2")],
                        ),
                    ],
                ),
            ],
        )
        messages = format_conversation(conv)

        assert len(messages) == 3, f"Expected 3 messages, got {len(messages)}: {messages}"
        assert messages[0]["role"] == "assistant"
        assert len(messages[0]["tool_calls"]) == 2
        assert messages[1]["role"] == "tool"
        assert messages[1]["tool_call_id"] == "toolu_ws_1"
        assert messages[2]["role"] == "tool"
        assert messages[2]["tool_call_id"] == "toolu_ws_2"

    def test_assistant_message_without_tool_results_returns_single_dict(self):
        """Assistant message without tool result blocks should return a single dict.

        Ensure no regression: normal assistant messages still work as before.
        """
        conv = ConversationContext(
            messages=[
                Message(
                    role="assistant",
                    content=[
                        TextBlock(text="Hello, how can I help?"),
                    ],
                ),
            ],
        )
        messages = format_conversation(conv)

        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"] == "Hello, how can I help?"
