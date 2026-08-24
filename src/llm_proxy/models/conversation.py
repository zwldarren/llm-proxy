# src/llm_proxy/core/unified/conversation.py
"""Message and ConversationContext for unified protocol format."""

from dataclasses import dataclass, field
from typing import Literal

from llm_proxy.models.content_blocks import ContentBlock, TextBlock


@dataclass
class SystemMessage:
    """System or developer message for unified protocol format.

    Represents a system/developer message with role, content, and optional name.
    OpenAI allows 'name' field on system/developer messages for disambiguation.

    Content is always stored as list[ContentBlock] for consistency with Message.
    Use the from_text() class method to create from a plain string.
    """

    role: Literal["system", "developer"]
    content: list[ContentBlock]
    name: str | None = None

    @classmethod
    def from_text(
        cls,
        role: Literal["system", "developer"],
        text: str,
        name: str | None = None,
    ) -> SystemMessage:
        """Create a SystemMessage from plain text.

        Args:
            role: Either "system" or "developer"
            text: The text content
            name: Optional name for the message

        Returns:
            A new SystemMessage instance with TextBlock content
        """
        return cls(
            role=role,
            content=[TextBlock(text=text)],
            name=name,
        )

    @property
    def text_content(self) -> str:
        """Extract all text from content.

        Returns:
            Concatenated text from all TextBlocks, empty string if none.
        """
        return "".join(block.text for block in self.content if isinstance(block, TextBlock))


@dataclass
class Message:
    """Message dataclass for unified protocol format.

    Represents a single message in a conversation with role, content, and optional name.
    """

    role: Literal["user", "assistant", "tool", "developer", "function", "system"]
    content: list[ContentBlock]
    name: str | None = None
    # OpenResponses assistant message phase ("commentary" | "final_answer").
    # Preserved so follow-up requests can resend it (spec 2026-04-24).
    phase: str | None = None

    @property
    def text_content(self) -> str:
        """Extract all text from TextBlocks in content.

        Returns:
            Concatenated text from all TextBlocks, empty string if none.
        """
        return "".join(block.text for block in self.content if isinstance(block, TextBlock))


@dataclass
class ConversationContext:
    """ConversationContext dataclass for unified protocol format.

    Holds system/developer messages and conversation messages. Use
    OpenAIProtocolSerializer.parse_conversation() and format_content_blocks() for
    OpenAI-compatible format conversion, or get_serializer(name) for
    protocol-specific serializers.
    """

    system_messages: list[SystemMessage] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)
