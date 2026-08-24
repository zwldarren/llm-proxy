# tests/unit/models/test_conversation.py
"""Tests for Message and ConversationContext."""

from llm_proxy.models.content_blocks import (
    ContentBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from llm_proxy.models.conversation import ConversationContext, Message, SystemMessage
from llm_proxy.serialization.openai import format_conversation


class TestMessage:
    """Test Message dataclass."""

    def test_create_user_message(self):
        """Create Message with user role."""
        content: list[ContentBlock] = [TextBlock(text="Hello")]
        msg = Message(role="user", content=content)
        assert msg.role == "user"
        assert len(msg.content) == 1
        assert isinstance(msg.content[0], TextBlock)
        assert msg.name is None

    def test_create_assistant_message(self):
        """Create Message with assistant role."""
        content: list[ContentBlock] = [TextBlock(text="Hi there!")]
        msg = Message(role="assistant", content=content)
        assert msg.role == "assistant"

    def test_create_tool_message(self):
        """Create Message with tool role."""
        content: list[ContentBlock] = [
            ToolResultBlock(tool_use_id="toolu_123", content="Result: 42")
        ]
        msg = Message(role="tool", content=content, name="get_weather")
        assert msg.role == "tool"
        assert msg.name == "get_weather"

    def test_text_content_property_single_block(self):
        """Extract text from single TextBlock."""
        msg = Message(role="user", content=[TextBlock(text="Hello, world!")])
        assert msg.text_content == "Hello, world!"

    def test_text_content_property_multiple_blocks(self):
        """Extract text from multiple TextBlocks."""
        msg = Message(
            role="assistant",
            content=[TextBlock(text="Hello, "), TextBlock(text="world!")],
        )
        assert msg.text_content == "Hello, world!"

    def test_text_content_property_no_text_blocks(self):
        """Return empty string when no TextBlocks present."""
        msg = Message(
            role="tool",
            content=[ToolUseBlock(id="toolu_123", name="test", input={})],
        )
        assert msg.text_content == ""

    def test_text_content_property_mixed_blocks(self):
        """Extract text only from TextBlocks, ignoring others."""
        msg = Message(
            role="assistant",
            content=[
                TextBlock(text="The result is "),
                ToolUseBlock(id="toolu_123", name="get_value", input={}),
                TextBlock(text=" 42."),
            ],
        )
        assert msg.text_content == "The result is  42."


class TestSystemMessage:
    """Test SystemMessage dataclass."""

    def test_create_system_message(self):
        """Create SystemMessage with system role."""
        msg = SystemMessage.from_text(role="system", text="You are helpful.")
        assert msg.role == "system"
        assert msg.text_content == "You are helpful."
        assert msg.name is None

    def test_create_developer_message(self):
        """Create SystemMessage with developer role."""
        msg = SystemMessage.from_text(role="developer", text="Be concise.")
        assert msg.role == "developer"
        assert msg.text_content == "Be concise."

    def test_system_message_with_name(self):
        """Create SystemMessage with name."""
        msg = SystemMessage.from_text(role="system", text="You are helpful.", name="assistant")
        assert msg.name == "assistant"

    def test_text_content_property_string(self):
        """Extract text from string content."""
        msg = SystemMessage.from_text(role="system", text="Hello, world!")
        assert msg.text_content == "Hello, world!"

    def test_text_content_property_blocks(self):
        """Extract text from ContentBlocks."""
        msg = SystemMessage(
            role="system",
            content=[TextBlock(text="You are "), TextBlock(text="helpful.")],
        )
        assert msg.text_content == "You are helpful."


class TestConversationContext:
    """Test ConversationContext dataclass."""

    def test_create_empty_context(self):
        """Create empty ConversationContext."""
        ctx = ConversationContext()
        assert ctx.system_messages == []
        assert ctx.messages == []

    def test_create_with_system_messages(self):
        """Create ConversationContext with system messages."""
        ctx = ConversationContext(
            system_messages=[SystemMessage.from_text(role="system", text="You are helpful.")]
        )
        assert len(ctx.system_messages) == 1
        assert ctx.system_messages[0].text_content == "You are helpful."

    def test_create_with_messages(self):
        """Create ConversationContext with messages."""
        messages = [
            Message(role="user", content=[TextBlock(text="Hello")]),
            Message(role="assistant", content=[TextBlock(text="Hi!")]),
        ]
        ctx = ConversationContext(messages=messages)
        assert len(ctx.messages) == 2
        assert ctx.messages[0].role == "user"
        assert ctx.messages[1].role == "assistant"

    def test_to_openai_messages_simple(self):
        """Convert to OpenAI messages format."""
        ctx = ConversationContext(
            messages=[
                Message(role="user", content=[TextBlock(text="Hello")]),
                Message(role="assistant", content=[TextBlock(text="Hi there!")]),
            ]
        )
        openai_messages = format_conversation(ctx)
        assert len(openai_messages) == 2
        assert openai_messages[0]["role"] == "user"
        assert openai_messages[0]["content"] == "Hello"
        assert openai_messages[1]["role"] == "assistant"
        assert openai_messages[1]["content"] == "Hi there!"

    def test_to_openai_messages_with_name(self):
        """Convert tool message with name to OpenAI format."""
        msg = Message(
            role="tool",
            content=[ToolResultBlock(tool_use_id="toolu_123", content="Result")],
            name="get_weather",
        )
        ctx = ConversationContext(messages=[msg])
        openai_messages = format_conversation(ctx)
        assert len(openai_messages) == 1
        assert openai_messages[0]["role"] == "tool"
        assert openai_messages[0]["tool_call_id"] == "toolu_123"
        assert openai_messages[0]["name"] == "get_weather"

    def test_to_openai_messages_with_system(self):
        """Convert system messages to OpenAI format."""
        ctx = ConversationContext(
            system_messages=[SystemMessage.from_text(role="system", text="You are helpful.")],
            messages=[
                Message(role="user", content=[TextBlock(text="Hello")]),
            ],
        )
        openai_messages = format_conversation(ctx)
        assert len(openai_messages) == 2
        assert openai_messages[0]["role"] == "system"
        assert openai_messages[0]["content"] == "You are helpful."
        assert openai_messages[1]["role"] == "user"

    def test_to_openai_messages_with_system_name(self):
        """Convert system message with name to OpenAI format."""
        ctx = ConversationContext(
            system_messages=[
                SystemMessage.from_text(role="system", text="You are helpful.", name="assistant")
            ],
            messages=[
                Message(role="user", content=[TextBlock(text="Hello")]),
            ],
        )
        openai_messages = format_conversation(ctx)
        assert openai_messages[0]["role"] == "system"
        assert openai_messages[0]["content"] == "You are helpful."
        assert openai_messages[0]["name"] == "assistant"

    def test_to_openai_messages_with_developer(self):
        """Convert developer message to OpenAI format."""
        ctx = ConversationContext(
            system_messages=[SystemMessage.from_text(role="developer", text="Be concise.")],
            messages=[
                Message(role="user", content=[TextBlock(text="Hello")]),
            ],
        )
        openai_messages = format_conversation(ctx)
        assert openai_messages[0]["role"] == "developer"
        assert openai_messages[0]["content"] == "Be concise."

    def test_roundtrip_user_assistant(self):
        """Test roundtrip with user and assistant messages."""
        ctx = ConversationContext(
            messages=[
                Message(role="user", content=[TextBlock(text="What is 2+2?")]),
                Message(role="assistant", content=[TextBlock(text="2+2 equals 4.")]),
            ]
        )
        openai = format_conversation(ctx)

        assert len(openai) == 2
        assert openai[0]["content"] == "What is 2+2?"
        assert openai[1]["content"] == "2+2 equals 4."
