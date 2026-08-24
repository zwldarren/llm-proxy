"""Test reasoning/thinking content round-trip across protocols and providers.

These tests verify that reasoning/thinking content can be correctly:
1. Extracted from provider responses and converted to ThinkingBlock
2. Formatted into protocol-specific formats (OpenAI, Anthropic)
3. Parsed back from protocol formats to ThinkingBlock
4. Converted to provider-specific formats for requests
5. Round-tripped through streaming transformers
"""

import pytest

from llm_proxy.core.thinking import PROVIDER_REASONING_FORMAT, extract_reasoning_from_message
from llm_proxy.models import (
    ConversationContext,
    Message,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
)
from llm_proxy.protocols.anthropic.serializer import AnthropicProtocolSerializer
from llm_proxy.protocols.openai.parsing import OpenAIParsingMixin
from llm_proxy.serialization.anthropic.mixin import AnthropicContentMixin
from llm_proxy.serialization.gemini.conversation import GeminiConversationMixin
from llm_proxy.serialization.ollama.conversation import OllamaConversationMixin
from llm_proxy.serialization.openai.converter import format_conversation


class ReasoningRoundtripTestMixin:
    """Shared utilities for round-trip testing."""

    def _make_conv_with_thinking(self, thinking_text, signature=None, is_redacted=False):
        """Create conversation with a thinking block."""
        conv = ConversationContext()
        if is_redacted:
            content_block = RedactedThinkingBlock(data=thinking_text)
        else:
            content_block = ThinkingBlock(
                thinking=thinking_text,
                signature=signature,
            )
        conv.messages.append(
            Message(
                role="assistant",
                content=[
                    content_block,
                    TextBlock(text="Hello world"),
                ],
            )
        )
        return conv


class TestReasoningFieldRegistry:
    """Test the PROVIDER_REASONING_FORMAT registry."""

    def test_all_providers_have_required_fields(self):
        required_keys = {
            "message_field",
            "stream_delta_field",
            "content_block_type",
            "signature_supported",
        }
        for provider, fmt in PROVIDER_REASONING_FORMAT.items():
            missing = required_keys - set(fmt.keys())
            assert not missing, f"Provider {provider} missing keys: {missing}"

    def test_provider_format_consistency(self):
        for provider, fmt in PROVIDER_REASONING_FORMAT.items():
            if fmt["signature_supported"] and fmt["content_block_type"] is None:
                assert not fmt.get("redacted_block_type"), (
                    f"{provider}: signature_supported but content_block_type is None"
                )
            if fmt["message_field"] is None:
                assert fmt["content_block_type"] is not None, (
                    f"{provider}: no message_field and no content_block_type"
                )

    def test_extract_reasoning_from_ollama_message(self):
        msg = {"thinking": "Let me think...", "content": "Answer"}
        text, sig = extract_reasoning_from_message(msg, "ollama")
        assert text == "Let me think..."
        assert sig is None

    def test_extract_reasoning_from_anthropic_message(self):
        msg = {"content": [{"type": "thinking", "thinking": "Let me think..."}]}
        text, sig = extract_reasoning_from_message(msg, "anthropic")
        assert text is None

    def test_extract_reasoning_unknown_provider_returns_none(self):
        msg = {"reasoning_content": "Let me think...", "content": "Answer"}
        text, sig = extract_reasoning_from_message(msg, "openai")
        assert text is None
        assert sig is None


class TestOpenAIProtocolRoundTrip(ReasoningRoundtripTestMixin):
    """Test reasoning content round-trip through OpenAI protocol."""

    def test_thinking_block_to_openai_message(self):
        conv = self._make_conv_with_thinking("I need to analyze this")
        messages = format_conversation(conv)
        assert len(messages) == 1
        assert messages[0]["reasoning_content"] == "I need to analyze this"
        assert messages[0]["content"] == "Hello world"

    def test_thinking_block_with_signature_to_openai_message(self):
        conv = self._make_conv_with_thinking("think", signature="sig123")
        messages = format_conversation(conv)
        assert messages[0]["reasoning_content"] == "think"
        assert messages[0]["reasoning_signature"] == "sig123"

    def test_thinking_block_redacted_to_openai_message(self):
        conv = self._make_conv_with_thinking("hidden", is_redacted=True)
        messages = format_conversation(conv)
        assert messages[0]["reasoning_content"] == "hidden"
        assert messages[0]["reasoning_is_redacted"] is True

    def test_openai_message_parsing_roundtrip(self):
        parser = OpenAIParsingMixin()
        conv = ConversationContext()
        conv.messages.append(
            Message(
                role="assistant",
                content=[
                    ThinkingBlock(thinking="I need to analyze this"),
                    TextBlock(text="Hello world"),
                ],
            )
        )
        messages = format_conversation(conv)
        parsed_conv = parser.parse_conversation(messages)
        parsed_msg = parsed_conv.messages[-1]
        thinking_blocks = [b for b in parsed_msg.content if isinstance(b, ThinkingBlock)]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0].thinking == "I need to analyze this"

    def test_openai_message_parsing_with_signature(self):
        parser = OpenAIParsingMixin()
        messages = [
            {
                "role": "assistant",
                "content": "Hello",
                "reasoning_content": "think",
                "reasoning_signature": "sig123",
            }
        ]
        parsed_conv = parser.parse_conversation(messages)
        parsed_msg = parsed_conv.messages[-1]
        thinking_blocks = [b for b in parsed_msg.content if isinstance(b, ThinkingBlock)]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0].thinking == "think"
        assert thinking_blocks[0].signature == "sig123"

    def test_openai_message_parsing_redacted(self):
        parser = OpenAIParsingMixin()
        from llm_proxy.models import RedactedThinkingBlock

        messages = [
            {
                "role": "assistant",
                "content": "Hello",
                "reasoning_content": "[redacted]",
                "reasoning_is_redacted": True,
            }
        ]
        parsed_conv = parser.parse_conversation(messages)
        parsed_msg = parsed_conv.messages[-1]
        redacted = [b for b in parsed_msg.content if isinstance(b, RedactedThinkingBlock)]
        assert len(redacted) == 1

    def test_openai_message_with_reasoning_field(self):
        parser = OpenAIParsingMixin()
        messages = [
            {
                "role": "assistant",
                "content": "Hello",
                "reasoning_content": "I need to analyze this",
            }
        ]
        parsed_conv = parser.parse_conversation(messages)
        parsed_msg = parsed_conv.messages[-1]
        thinking_blocks = [b for b in parsed_msg.content if isinstance(b, ThinkingBlock)]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0].thinking == "I need to analyze this"

    def test_empty_reasoning_content_not_parsed(self):
        parser = OpenAIParsingMixin()
        messages = [{"role": "assistant", "content": "Hello", "reasoning_content": ""}]
        parsed_conv = parser.parse_conversation(messages)
        parsed_msg = parsed_conv.messages[-1]
        thinking_blocks = [b for b in parsed_msg.content if isinstance(b, ThinkingBlock)]
        assert len(thinking_blocks) == 0

    def test_null_reasoning_content_not_parsed(self):
        parser = OpenAIParsingMixin()
        messages = [{"role": "assistant", "content": "Hello", "reasoning_content": None}]
        parsed_conv = parser.parse_conversation(messages)
        parsed_msg = parsed_conv.messages[-1]
        thinking_blocks = [b for b in parsed_msg.content if isinstance(b, ThinkingBlock)]
        assert len(thinking_blocks) == 0


class TestAnthropicProtocolRoundTrip(ReasoningRoundtripTestMixin):
    """Test reasoning content round-trip through Anthropic protocol."""

    def setup_method(self):
        self.mixin = AnthropicContentMixin()

    def test_thinking_block_to_anthropic_format(self):
        conv = self._make_conv_with_thinking("I need to analyze this")
        for msg in conv.messages:
            formatted = self.mixin.format_content_blocks(msg.content)
            thinking_blocks = [
                b for b in formatted if isinstance(b, dict) and b.get("type") == "thinking"
            ]
            assert len(thinking_blocks) == 1
            assert thinking_blocks[0]["thinking"] == "I need to analyze this"

    def test_thinking_block_signature_to_anthropic_format(self):
        conv = self._make_conv_with_thinking("think", signature="sig123")
        for msg in conv.messages:
            formatted = self.mixin.format_content_blocks(msg.content)
            thinking_blocks = [
                b for b in formatted if isinstance(b, dict) and b.get("type") == "thinking"
            ]
            assert len(thinking_blocks) == 1
            assert thinking_blocks[0]["signature"] == "sig123"

    def test_anthropic_format_parse(self):
        content = [
            {"type": "thinking", "thinking": "I need to analyze this", "signature": "sig123"},
            {"type": "text", "text": "Hello world"},
        ]
        blocks = self.mixin.parse_content_blocks(content)
        thinking_blocks = [b for b in blocks if isinstance(b, ThinkingBlock)]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0].thinking == "I need to analyze this"
        assert thinking_blocks[0].signature == "sig123"

    def test_anthropic_format_redacted_thinking_parse(self):
        from llm_proxy.models import RedactedThinkingBlock

        content = [
            {"type": "redacted_thinking", "data": "redacted_data"},
            {"type": "text", "text": "Hello world"},
        ]
        blocks = self.mixin.parse_content_blocks(content)
        redacted = [b for b in blocks if isinstance(b, RedactedThinkingBlock)]
        assert len(redacted) == 1

    def test_anthropic_roundtrip_thinking(self):
        thinking_text = "I need to analyze this"
        signature = "sig123"
        content = [
            {"type": "thinking", "thinking": thinking_text, "signature": signature},
            {"type": "text", "text": "Hello"},
        ]
        blocks = self.mixin.parse_content_blocks(content)
        formatted = self.mixin.format_content_blocks(blocks)
        thinking_output = [
            b for b in formatted if isinstance(b, dict) and b.get("type") == "thinking"
        ]
        assert len(thinking_output) == 1
        assert thinking_output[0]["thinking"] == thinking_text
        assert thinking_output[0]["signature"] == signature

    def test_anthropic_roundtrip_redacted_thinking(self):

        content = [
            {"type": "redacted_thinking", "data": "data"},
            {"type": "text", "text": "Hello"},
        ]
        blocks = self.mixin.parse_content_blocks(content)
        formatted = self.mixin.format_content_blocks(blocks)
        redacted_output = [
            b for b in formatted if isinstance(b, dict) and b.get("type") == "redacted_thinking"
        ]
        assert len(redacted_output) == 1


class TestOllamaProtocolRoundTrip(ReasoningRoundtripTestMixin):
    """Test reasoning content round-trip through Ollama protocol."""

    def setup_method(self):
        self.mixin = OllamaConversationMixin()

    def test_thinking_block_to_ollama_format(self):
        conv = self._make_conv_with_thinking("I need to analyze this")
        messages = self.mixin._convert_conversation_to_ollama(conv)
        assert len(messages) == 1
        assert messages[0]["thinking"] == "I need to analyze this"
        assert messages[0]["content"] == "Hello world"

    def test_thinking_block_with_simple_text(self):
        conv = ConversationContext()
        conv.messages.append(
            Message(
                role="assistant",
                content=[
                    ThinkingBlock(thinking="Step 1: Check"),
                    TextBlock(text="Here's the answer"),
                ],
            )
        )
        messages = self.mixin._convert_conversation_to_ollama(conv)
        assert len(messages) == 1
        assert messages[0]["thinking"] == "Step 1: Check"
        assert messages[0]["content"] == "Here's the answer"

    def test_thinking_block_with_tool_call(self):
        conv = ConversationContext()
        conv.messages.append(
            Message(
                role="assistant",
                content=[
                    ThinkingBlock(thinking="Let me use a tool"),
                    ToolUseBlock(id="call_1", name="search", input={"q": "test"}),
                ],
            )
        )
        messages = self.mixin._convert_conversation_to_ollama(conv)
        assert len(messages) == 1
        assert messages[0]["thinking"] == "Let me use a tool"
        assert len(messages[0]["tool_calls"]) == 1


class TestGeminiProtocolRoundTrip(ReasoningRoundtripTestMixin):
    """Test reasoning content round-trip through Gemini protocol."""

    def setup_method(self):
        self.mixin = GeminiConversationMixin()

    def test_thinking_block_to_gemini_format(self):
        conv = self._make_conv_with_thinking("I need to analyze this")
        contents, _ = self.mixin._convert_conversation_to_gemini(conv)
        assert len(contents) == 1
        parts = contents[0]["parts"]
        thought_parts = [p for p in parts if p.get("thought") is True]
        assert len(thought_parts) == 1
        assert thought_parts[0]["text"] == "I need to analyze this"

    def test_thinking_block_signature_to_gemini_format(self):
        conv = self._make_conv_with_thinking("think", signature="sig123")
        contents, _ = self.mixin._convert_conversation_to_gemini(conv)
        parts = contents[0]["parts"]
        thought_parts = [p for p in parts if p.get("thought") is True]
        assert len(thought_parts) == 1
        assert thought_parts[0]["signature"] == "sig123"

    def test_thinking_block_with_text_blocks(self):
        conv = ConversationContext()
        conv.messages.append(
            Message(
                role="assistant",
                content=[
                    ThinkingBlock(thinking="Step 1: Think"),
                    TextBlock(text="Answer text here"),
                ],
            )
        )
        contents, _ = self.mixin._convert_conversation_to_gemini(conv)
        parts = contents[0]["parts"]
        thought_parts = [p for p in parts if p.get("thought") is True]
        text_parts = [p for p in parts if "text" in p and not p.get("thought")]
        assert len(thought_parts) == 1
        assert len(text_parts) == 1
        assert text_parts[0]["text"] == "Answer text here"

    def test_gemini_response_thought_part_signature_parsed(self):
        """Gemini generateContent can attach thoughtSignature to thought parts.

        The parser must preserve it on ThinkingBlock.signature so it can be
        replayed to Gemini on the next turn.
        """
        from llm_proxy.serialization.gemini.response_parser import GeminiResponseParserMixin

        parser = GeminiResponseParserMixin()
        response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"thought": True, "text": "planning", "thoughtSignature": "SIG_T"}
                        ]
                    }
                }
            ]
        }
        internal = parser.parse_provider_response(response, model="gemini-3.1-pro-preview")
        thinking = [b for b in internal.output if isinstance(b, ThinkingBlock)]
        assert len(thinking) == 1
        assert thinking[0].thinking == "planning"
        assert thinking[0].signature == "SIG_T"

    def test_gemini_thought_part_signature_reaches_gemini_request(self):
        """A ThinkingBlock carrying a Gemini thought signature must serialize
        back to a thought part with signature so multi-turn context survives."""
        conv = ConversationContext()
        conv.messages.append(
            Message(
                role="assistant",
                content=[ThinkingBlock(thinking="planning", signature="SIG_T")],
            )
        )
        contents, _ = self.mixin._convert_conversation_to_gemini(conv)
        model_parts = [c for c in contents if c.get("role") == "model"][0]["parts"]
        thought_parts = [p for p in model_parts if p.get("thought") is True]
        assert len(thought_parts) == 1
        assert thought_parts[0].get("signature") == "SIG_T"

    def test_gemini_streaming_thought_part_signature_accumulated(self):
        """Gemini streaming thought parts with thoughtSignature must accumulate
        the signature onto the final ThinkingBlock."""
        from llm_proxy.serialization.gemini.streaming_converter import GeminiStreamingTransformer

        transformer = GeminiStreamingTransformer(model="gemini-3.1-pro-preview", request_id="req-1")
        chunk = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"thought": True, "text": "planning", "thoughtSignature": "SIG_T"}
                        ]
                    },
                    "finishReason": "STOP",
                }
            ]
        }
        transformer.transform_chunk(chunk)
        blocks = transformer.get_accumulated_output()
        thinking = [b for b in blocks if isinstance(b, ThinkingBlock)]
        assert len(thinking) == 1
        assert thinking[0].thinking == "planning"
        assert thinking[0].signature == "SIG_T"


class TestCrossProtocolRoundTrip:
    """Test cross-protocol: parse in one format, format in another."""

    def test_anthropic_input_to_openai_output(self):
        mixin = AnthropicContentMixin()
        anthropic_content = [
            {"type": "thinking", "thinking": "Let me think", "signature": "sig"},
            {"type": "text", "text": "Answer"},
        ]
        blocks = mixin.parse_content_blocks(anthropic_content)
        conv = ConversationContext()
        conv.messages.append(Message(role="assistant", content=blocks))
        openai_messages = format_conversation(conv)
        assert openai_messages[0]["reasoning_content"] == "Let me think"
        assert openai_messages[0]["reasoning_signature"] == "sig"
        assert openai_messages[0]["content"] == "Answer"

    def test_openai_input_to_anthropic_output(self):
        mixin = AnthropicContentMixin()
        parser = OpenAIParsingMixin()
        openai_messages = [
            {
                "role": "assistant",
                "content": "Answer",
                "reasoning_content": "Let me think",
                "reasoning_signature": "sig",
            }
        ]
        conv = parser.parse_conversation(openai_messages)
        anthropic_formatted = mixin.format_content_blocks(conv.messages[-1].content)
        thinking_blocks = [
            b for b in anthropic_formatted if isinstance(b, dict) and b.get("type") == "thinking"
        ]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0]["thinking"] == "Let me think"
        assert thinking_blocks[0]["signature"] == "sig"

    def test_anthropic_input_to_ollama_output(self):
        mixin = AnthropicContentMixin()
        ollama_mixin = OllamaConversationMixin()
        anthropic_content = [
            {"type": "thinking", "thinking": "Let me think"},
            {"type": "text", "text": "Answer"},
        ]
        blocks = mixin.parse_content_blocks(anthropic_content)
        conv = ConversationContext()
        conv.messages.append(Message(role="assistant", content=blocks))
        ollama_messages = ollama_mixin._convert_conversation_to_ollama(conv)
        assert ollama_messages[0]["thinking"] == "Let me think"
        assert ollama_messages[0]["content"] == "Answer"

    def test_anthropic_input_to_gemini_output(self):
        mixin = AnthropicContentMixin()
        gemini_mixin = GeminiConversationMixin()
        anthropic_content = [
            {"type": "thinking", "thinking": "Let me think", "signature": "sig"},
            {"type": "text", "text": "Answer"},
        ]
        blocks = mixin.parse_content_blocks(anthropic_content)
        conv = ConversationContext()
        conv.messages.append(Message(role="assistant", content=blocks))
        contents, _ = gemini_mixin._convert_conversation_to_gemini(conv)
        thought_parts = [p for p in contents[0]["parts"] if p.get("thought") is True]
        assert len(thought_parts) == 1
        assert thought_parts[0]["text"] == "Let me think"
        assert thought_parts[0]["signature"] == "sig"

    def test_openai_input_to_gemini_output(self):
        parser = OpenAIParsingMixin()
        gemini_mixin = GeminiConversationMixin()
        openai_messages = [
            {
                "role": "assistant",
                "content": "Answer",
                "reasoning_content": "Let me think",
            }
        ]
        conv = parser.parse_conversation(openai_messages)
        contents, _ = gemini_mixin._convert_conversation_to_gemini(conv)
        thought_parts = [p for p in contents[0]["parts"] if p.get("thought") is True]
        assert len(thought_parts) == 1
        assert thought_parts[0]["text"] == "Let me think"

    def test_openai_input_to_ollama_output(self):
        parser = OpenAIParsingMixin()
        ollama_mixin = OllamaConversationMixin()
        openai_messages = [
            {
                "role": "assistant",
                "content": "Answer",
                "reasoning_content": "Let me think",
            }
        ]
        conv = parser.parse_conversation(openai_messages)
        ollama_messages = ollama_mixin._convert_conversation_to_ollama(conv)
        assert ollama_messages[0]["thinking"] == "Let me think"
        assert ollama_messages[0]["content"] == "Answer"


class TestAnthropicProtocolParserReasoningContent:
    """Test Anthropic protocol parser handles OpenAI-format reasoning_content.

    When clients send conversation history containing OpenAI-format
    reasoning_content (e.g. from a previous OpenAI-endpoint response)
    through the Anthropic protocol endpoint, the parser should convert
    it to ThinkingBlock / RedactedThinkingBlock.
    """

    def setup_method(self):
        self.serializer = AnthropicProtocolSerializer()

    def test_reasoning_content_to_thinking_block(self):
        messages = [
            {"role": "user", "content": "Think step by step"},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Here is the answer."}],
                "reasoning_content": "I need to analyze this carefully",
            },
        ]
        conv = self.serializer.parse_conversation(messages)
        assert len(conv.messages) == 2
        assistant_msg = conv.messages[-1]
        thinking_blocks = [b for b in assistant_msg.content if isinstance(b, ThinkingBlock)]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0].thinking == "I need to analyze this carefully"
        assert thinking_blocks[0].signature is None

    def test_reasoning_content_with_signature(self):
        messages = [
            {
                "role": "assistant",
                "content": "Answer",
                "reasoning_content": "thinking",
                "reasoning_signature": "sig123",
            },
        ]
        conv = self.serializer.parse_conversation(messages)
        assistant_msg = conv.messages[-1]
        thinking_blocks = [b for b in assistant_msg.content if isinstance(b, ThinkingBlock)]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0].thinking == "thinking"
        assert thinking_blocks[0].signature == "sig123"

    def test_reasoning_content_redacted(self):
        messages = [
            {
                "role": "assistant",
                "content": "Answer",
                "reasoning_content": "[redacted]",
                "reasoning_is_redacted": True,
            },
        ]
        conv = self.serializer.parse_conversation(messages)
        assistant_msg = conv.messages[-1]
        redacted = [b for b in assistant_msg.content if isinstance(b, RedactedThinkingBlock)]
        assert len(redacted) == 1
        assert redacted[0].data == "[redacted]"

    def test_empty_reasoning_content_ignored(self):
        messages = [
            {
                "role": "assistant",
                "content": "Answer",
                "reasoning_content": "",
            },
        ]
        conv = self.serializer.parse_conversation(messages)
        assistant_msg = conv.messages[-1]
        thinking = [
            b
            for b in assistant_msg.content
            if isinstance(b, (ThinkingBlock, RedactedThinkingBlock))
        ]
        assert len(thinking) == 0

    def test_null_reasoning_content_ignored(self):
        messages = [
            {
                "role": "assistant",
                "content": "Answer",
                "reasoning_content": None,
            },
        ]
        conv = self.serializer.parse_conversation(messages)
        assistant_msg = conv.messages[-1]
        thinking = [
            b
            for b in assistant_msg.content
            if isinstance(b, (ThinkingBlock, RedactedThinkingBlock))
        ]
        assert len(thinking) == 0

    def test_native_anthropic_format_still_works(self):
        """Ensure native Anthropic thinking blocks are not broken."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "native thinking", "signature": "sig"},
                    {"type": "text", "text": "Answer"},
                ],
            },
        ]
        conv = self.serializer.parse_conversation(messages)
        assistant_msg = conv.messages[-1]
        thinking_blocks = [b for b in assistant_msg.content if isinstance(b, ThinkingBlock)]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0].thinking == "native thinking"
        assert thinking_blocks[0].signature == "sig"

    def test_reasoning_content_does_not_affect_user_messages(self):
        """reasoning_content on user messages should be ignored."""
        messages = [
            {
                "role": "user",
                "content": "Hello",
                "reasoning_content": "should be ignored",
            },
        ]
        conv = self.serializer.parse_conversation(messages)
        user_msg = conv.messages[-1]
        thinking = [
            b for b in user_msg.content if isinstance(b, (ThinkingBlock, RedactedThinkingBlock))
        ]
        assert len(thinking) == 0

    def test_reasoning_content_roundtrip_through_anthropic_format(self):
        """Parse OpenAI reasoning_content, then format back to Anthropic."""
        messages = [
            {
                "role": "assistant",
                "content": "Answer",
                "reasoning_content": "thinking text",
                "reasoning_signature": "sig",
            },
        ]
        conv = self.serializer.parse_conversation(messages)
        mixin = AnthropicContentMixin()
        formatted = mixin.format_content_blocks(conv.messages[-1].content)
        thinking_blocks = [
            b for b in formatted if isinstance(b, dict) and b.get("type") == "thinking"
        ]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0]["thinking"] == "thinking text"
        assert thinking_blocks[0]["signature"] == "sig"


class TestStreamingTransformerReasoningHandling:
    """Test reasoning content handling in streaming transformers."""

    def test_anthropic_transformer_reasoning_to_thinking_delta(self):
        from llm_proxy.protocols.anthropic.streaming import AnthropicStreamingTransformer

        transformer = AnthropicStreamingTransformer(model="test", request_id="req-1")
        chunk = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "delta": {"reasoning_content": "I am thinking"},
                    "finish_reason": None,
                }
            ],
        }
        result = transformer._transform_openai_chunk(chunk)
        assert result is not None
        assert "thinking_delta" in result

    def test_anthropic_transformer_reasoning_fallback(self):
        from llm_proxy.protocols.anthropic.streaming import AnthropicStreamingTransformer

        transformer = AnthropicStreamingTransformer(model="test", request_id="req-1")
        chunk = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "delta": {"reasoning": "I am thinking via reasoning field"},
                    "finish_reason": None,
                }
            ],
        }
        result = transformer._transform_openai_chunk(chunk)
        assert result is not None
        assert "thinking_delta" in result

    def test_anthropic_transformer_signature_delta(self):
        from llm_proxy.protocols.anthropic.streaming import AnthropicStreamingTransformer

        transformer = AnthropicStreamingTransformer(model="test", request_id="req-1")
        chunk = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "delta": {"reasoning_signature": "sig123"},
                    "finish_reason": None,
                }
            ],
        }
        transformer._in_thinking_block = True
        result = transformer._transform_openai_chunk(chunk)
        assert result is not None
        assert "signature_delta" in result

    def test_anthropic_transformer_redacted_thinking(self):
        from llm_proxy.protocols.anthropic.streaming import AnthropicStreamingTransformer

        transformer = AnthropicStreamingTransformer(model="test", request_id="req-1")
        chunk = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "reasoning_content": "[redacted]",
                        "reasoning_is_redacted": True,
                    },
                    "finish_reason": None,
                }
            ],
        }
        result = transformer._transform_openai_chunk(chunk)
        assert result is not None
        assert "redacted_thinking" in result

    def test_openai_transformer_reasoning_signature_accumulation(self):
        from llm_proxy.protocols.openai.streaming import OpenAIStreamingTransformer

        transformer = OpenAIStreamingTransformer(model="test", request_id="req-1")
        chunk = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "reasoning_content": "thinking text",
                        "reasoning_signature": "sig",
                    },
                    "finish_reason": None,
                }
            ],
        }
        transformer._normalize_chunk(chunk)
        transformer._finalize_accumulation()
        blocks = transformer.get_accumulated_output()
        thinking = [b for b in blocks if isinstance(b, ThinkingBlock)]
        assert len(thinking) == 1
        assert thinking[0].thinking == "thinking text"
        assert thinking[0].signature == "sig"

    def test_openai_transformer_reasoning_field_fallback(self):
        from llm_proxy.protocols.openai.streaming import OpenAIStreamingTransformer

        transformer = OpenAIStreamingTransformer(model="test", request_id="req-1")
        chunk = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "delta": {"reasoning": "reasoning field text"},
                    "finish_reason": None,
                }
            ],
        }
        transformer._normalize_chunk(chunk)
        transformer._finalize_accumulation()
        blocks = transformer.get_accumulated_output()
        thinking = [b for b in blocks if isinstance(b, ThinkingBlock)]
        assert len(thinking) == 1
        assert thinking[0].thinking == "reasoning field text"

    def test_openai_transformer_redacted_reasoning(self):
        from llm_proxy.models.content_blocks import RedactedThinkingBlock
        from llm_proxy.protocols.openai.streaming import OpenAIStreamingTransformer

        transformer = OpenAIStreamingTransformer(model="test", request_id="req-1")
        chunk = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "reasoning_content": "hidden",
                        "reasoning_is_redacted": True,
                    },
                    "finish_reason": None,
                }
            ],
        }
        transformer._normalize_chunk(chunk)
        transformer._finalize_accumulation()
        blocks = transformer.get_accumulated_output()
        redacted = [b for b in blocks if isinstance(b, RedactedThinkingBlock)]
        assert len(redacted) == 1
        assert redacted[0].data == "hidden"

    def test_openai_transformer_passthrough_reasoning_signature(self):
        """Verify reasoning_signature survives the output cleaning."""
        from llm_proxy.protocols.openai.streaming import OpenAIStreamingTransformer

        transformer = OpenAIStreamingTransformer(model="test", request_id="req-1")
        chunk = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "delta": {"reasoning_signature": "sig123"},
                    "finish_reason": None,
                }
            ],
        }
        result = transformer.transform(chunk)
        assert result is not None
        import orjson

        data = orjson.loads(result[6:])
        delta = data["choices"][0]["delta"]
        assert "reasoning_signature" in delta
        assert delta["reasoning_signature"] == "sig123"

    def test_openai_transformer_passthrough_reasoning_is_redacted(self):
        """Verify reasoning_is_redacted survives the output cleaning."""
        from llm_proxy.protocols.openai.streaming import OpenAIStreamingTransformer

        transformer = OpenAIStreamingTransformer(model="test", request_id="req-1")
        chunk = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "reasoning_content": "[redacted]",
                        "reasoning_is_redacted": True,
                    },
                    "finish_reason": None,
                }
            ],
        }
        result = transformer.transform(chunk)
        assert result is not None
        import orjson

        data = orjson.loads(result[6:])
        delta = data["choices"][0]["delta"]
        assert "reasoning_content" in delta
        assert delta["reasoning_content"] == "[redacted]"
        assert "reasoning_is_redacted" in delta
        assert delta["reasoning_is_redacted"] is True

    def test_openai_transformer_passthrough_both_reasoning_fields(self):
        """Verify reasoning_content + signature + is_redacted all pass through."""
        from llm_proxy.protocols.openai.streaming import OpenAIStreamingTransformer

        transformer = OpenAIStreamingTransformer(model="test", request_id="req-1")
        chunk = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "reasoning_content": "thinking",
                        "reasoning_signature": "sig",
                        "reasoning_is_redacted": False,
                    },
                    "finish_reason": None,
                }
            ],
        }
        result = transformer.transform(chunk)
        assert result is not None
        import orjson

        data = orjson.loads(result[6:])
        delta = data["choices"][0]["delta"]
        assert delta.get("reasoning_content") == "thinking"
        assert delta.get("reasoning_signature") == "sig"
        assert delta.get("reasoning_is_redacted") is False

    def test_openai_transformer_encrypted_content_accumulated(self):
        """Verify encrypted_content from OpenAI Responses provider streams is
        preserved through cleaning and accumulated onto the ThinkingBlock so
        /v1/responses can emit reasoning.encrypted_content."""
        from llm_proxy.protocols.openai.streaming import OpenAIStreamingTransformer

        transformer = OpenAIStreamingTransformer(model="test", request_id="req-1")
        chunk = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "test",
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
        result = transformer.transform(chunk)
        assert result is not None
        import orjson

        data = orjson.loads(result[6:])
        delta = data["choices"][0]["delta"]
        assert delta.get("reasoning_content") == "thinking"
        assert delta.get("encrypted_content") == "opaque-blob"

        blocks = transformer.get_accumulated_output()
        thinking = [b for b in blocks if isinstance(b, ThinkingBlock)]
        assert len(thinking) == 1
        assert thinking[0].thinking == "thinking"
        assert thinking[0].encrypted_content == "opaque-blob"

    def test_openai_transformer_top_level_encrypted_content_preserved(self):
        """OpenAI Responses provider may emit encrypted_content as a top-level
        field on the final chunk; the transformer must not drop it."""
        from llm_proxy.protocols.openai.streaming import OpenAIStreamingTransformer

        transformer = OpenAIStreamingTransformer(model="test", request_id="req-1")
        # Pre-populate reasoning so the chunk is not otherwise empty.
        transformer._reasoning_buffer = "done"
        chunk = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "test",
            "encrypted_content": "final-blob",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        result = transformer.transform(chunk)
        assert result is not None
        import orjson

        data = orjson.loads(result[6:])
        assert data.get("encrypted_content") == "final-blob"

        blocks = transformer.get_accumulated_output()
        thinking = [b for b in blocks if isinstance(b, ThinkingBlock)]
        assert len(thinking) == 1
        assert thinking[0].encrypted_content == "final-blob"


class TestReasoningNormalizationRoundTrip:
    """Test reasoning field normalization utilities."""

    def test_normalize_reasoning_for_request_converts_field(self):
        from llm_proxy.serialization.openai.components.request_builder import OpenAIRequestBuilder

        builder = OpenAIRequestBuilder()
        body = {
            "messages": [
                {
                    "role": "assistant",
                    "content": "Answer",
                    "reasoning_content": "Let me think",
                }
            ]
        }
        result = builder.normalize_reasoning_for_request(
            body, base_url="https://test.example.com", preferred="reasoning"
        )
        assert result["messages"][0]["reasoning"] == "Let me think"
        assert "reasoning_content" not in result["messages"][0]


class TestEncryptedOnlyReasoningRoundTrip:
    """Encrypted-only reasoning (thinking="", encrypted_content set) should
    NOT produce reasoning_content for chat-completions providers — we cannot
    decrypt it, so there is no real reasoning to pass. The Gemini
    thought_signature leak that used to produce encrypted-only items has been
    fixed, so this path should rarely trigger in practice.
    """

    def test_encrypted_only_thinking_block_skipped(self):
        """ThinkingBlock with encrypted_content but empty thinking must NOT
        produce reasoning_content — we cannot decrypt it."""
        from llm_proxy.models import ConversationContext, Message, TextBlock, ThinkingBlock
        from llm_proxy.serialization.openai.converter import format_conversation

        conv = ConversationContext()
        conv.messages.append(
            Message(
                role="assistant",
                content=[
                    ThinkingBlock(thinking="", encrypted_content="enc_blob_abc123"),
                    TextBlock(text="Here is the answer."),
                ],
            )
        )
        messages = format_conversation(conv)
        assert len(messages) == 1
        # Encrypted-only reasoning must NOT produce reasoning_content.
        assert not messages[0].get("reasoning_content"), (
            "Encrypted-only ThinkingBlock must NOT produce reasoning_content"
        )
        # But the visible text content must be preserved.
        assert messages[0]["content"] == "Here is the answer."

    def test_encrypted_only_tool_calls_preserve_tools_no_reasoning(self):
        from llm_proxy.models import (
            ConversationContext,
            Message,
            ThinkingBlock,
            ToolUseBlock,
        )
        from llm_proxy.serialization.openai.converter import format_conversation

        conv = ConversationContext()
        conv.messages.append(
            Message(
                role="assistant",
                content=[
                    ThinkingBlock(thinking="", encrypted_content="enc_blob_abc123"),
                    ToolUseBlock(id="call_1", name="search", input={"q": "test"}),
                ],
            )
        )
        messages = format_conversation(conv)
        assert len(messages) == 1
        assert messages[0]["tool_calls"], "Tool calls must be preserved"
        # Encrypted-only reasoning with no restored text must NOT emit the
        # meaningless "tool call" placeholder; reasoning_content is omitted.
        assert "reasoning_content" not in messages[0] or not messages[0].get("reasoning_content"), (
            "Encrypted-only reasoning with tool_calls must NOT produce 'tool call' placeholder"
        )

    def test_normal_thinking_block_unchanged(self):
        """Normal ThinkingBlock (with thinking text) must still work correctly."""
        from llm_proxy.models import ConversationContext, Message, TextBlock, ThinkingBlock
        from llm_proxy.serialization.openai.converter import format_conversation

        conv = ConversationContext()
        conv.messages.append(
            Message(
                role="assistant",
                content=[
                    ThinkingBlock(thinking="I need to analyze this", encrypted_content="enc_blob"),
                    TextBlock(text="Here is the answer."),
                ],
            )
        )
        messages = format_conversation(conv)
        assert messages[0]["reasoning_content"] == "I need to analyze this", (
            "Normal reasoning content must be preserved"
        )

    def test_empty_thinking_no_encrypted_yields_no_reasoning(self):
        """ThinkingBlock with empty thinking and no encrypted_content should
        NOT produce reasoning_content."""
        from llm_proxy.models import ConversationContext, Message, TextBlock, ThinkingBlock
        from llm_proxy.serialization.openai.converter import format_conversation

        conv = ConversationContext()
        conv.messages.append(
            Message(
                role="assistant",
                content=[
                    ThinkingBlock(thinking="", encrypted_content=None),
                    TextBlock(text="Here is the answer."),
                ],
            )
        )
        messages = format_conversation(conv)
        assert "reasoning_content" not in messages[0] or not messages[0].get("reasoning_content"), (
            "Empty thinking without encrypted_content must not produce reasoning_content"
        )

    def test_encrypted_only_message_with_no_content_is_dropped(self):
        """An assistant message with only encrypted reasoning is dropped —
        it carries no useful information for chat-completions providers."""
        from llm_proxy.models import ConversationContext, Message, ThinkingBlock
        from llm_proxy.serialization.openai.converter import format_conversation

        conv = ConversationContext()
        conv.messages.append(
            Message(
                role="assistant",
                content=[
                    ThinkingBlock(thinking="", encrypted_content="enc_blob_abc123"),
                ],
            )
        )
        messages = format_conversation(conv)
        # Empty assistant message (no content, no tool_calls, no real reasoning)
        # is dropped by _is_empty_assistant_message.
        assert len(messages) == 0, "Encrypted-only assistant with no real content must be dropped"

    def test_encrypted_tool_calls_shows_placeholder_vs_restored(self):
        """Verify that encrypted ThinkingBlock yields NO reasoning_content
        (no misleading 'tool call' placeholder) UNLESS real reasoning has been
        restored from cache (via repair).

        This is the test that distinguishes the two paths:
        - Before repair → reasoning_content omitted (no placeholder)
        - After repair  → reasoning_content = "restored real text"
        """
        from llm_proxy.models import ConversationContext, Message, ThinkingBlock, ToolUseBlock
        from llm_proxy.serialization.openai.converter import format_conversation

        conv = ConversationContext()
        conv.messages.append(
            Message(
                role="assistant",
                content=[
                    ThinkingBlock(thinking="", encrypted_content="enc_blob"),
                    ToolUseBlock(id="call_1", name="search", input={"q": "test"}),
                ],
            )
        )

        # === Before repair: encrypted thinking → reasoning_content omitted ===
        messages_before = format_conversation(conv)
        assert "reasoning_content" not in messages_before[0] or not messages_before[0].get(
            "reasoning_content"
        ), (
            "Encrypted-only reasoning with tool_calls must omit reasoning_content "
            "before cache restoration (no 'tool call' placeholder)"
        )

        # === Simulate cache restoration (what _repair_encrypted_reasoning does) ===
        conv.messages[0].content[0].thinking = "Let me analyze the query first."

        # === After repair: real reasoning preserved ===
        messages_after = format_conversation(conv)
        assert messages_after[0].get("reasoning_content") == "Let me analyze the query first.", (
            "After cache restoration, reasoning_content must contain the real text, "
            f"not placeholder. Got: {messages_after[0].get('reasoning_content')}"
        )


class TestReasoningCacheFromBlocks:
    """cache_reasoning_from_blocks pairs tool calls with real reasoning."""

    @pytest.fixture(autouse=True)
    def _isolated_cache(self):
        """The reasoning cache is module-global; isolate it per test."""
        from llm_proxy.core import reasoning_cache

        reasoning_cache.clear()
        yield
        reasoning_cache.clear()

    def test_streaming_order_pairs_thinking_before_tool_call(self):
        """Streaming output order (ThinkingBlock then ToolUseBlock) is cached."""
        from llm_proxy.core.reasoning_cache import cache_reasoning_from_blocks, get

        cache_reasoning_from_blocks(
            [
                ThinkingBlock(thinking="Let me check the weather."),
                ToolUseBlock(id="call_1", name="get_weather", input={"city": "Shanghai"}),
            ],
            response_id="resp_1",
        )
        assert get("call_1") == "Let me check the weather."

    def test_parser_order_pairs_tool_call_before_thinking(self):
        """Non-streaming parser order (ToolUseBlock then ThinkingBlock) also pairs.

        The Chat Completions response parser emits tool calls before the
        reasoning block within a message, so pairing must work in both orders.
        """
        from llm_proxy.core.reasoning_cache import cache_reasoning_from_blocks, get

        cache_reasoning_from_blocks(
            [
                ToolUseBlock(id="call_1", name="get_weather", input={"city": "Shanghai"}),
                ThinkingBlock(thinking="Let me check the weather."),
            ],
            response_id="resp_1",
        )
        assert get("call_1") == "Let me check the weather."

    def test_multi_step_turn_pairs_each_tool_call(self):
        """Each tool call pairs with its nearest reasoning block."""
        from llm_proxy.core.reasoning_cache import cache_reasoning_from_blocks, get

        cache_reasoning_from_blocks(
            [
                ThinkingBlock(thinking="Reasoning for call 1."),
                ToolUseBlock(id="call_1", name="get_date", input={}),
                ThinkingBlock(thinking="Reasoning for call 2."),
                ToolUseBlock(id="call_2", name="get_weather", input={}),
            ],
            response_id="resp_1",
        )
        assert get("call_1") == "Reasoning for call 1."
        assert get("call_2") == "Reasoning for call 2."

    def test_no_thinking_means_no_cache_entries(self):
        from llm_proxy.core.reasoning_cache import cache_reasoning_from_blocks, get

        cache_reasoning_from_blocks(
            [ToolUseBlock(id="call_1", name="get_weather", input={})],
            response_id="resp_1",
        )
        assert get("call_1") is None

    def test_restored_reasoning_survives_conversation_serialization(self):
        """A bare tool-call turn serializes with reasoning restored from cache.

        This is the multi-provider mixing case: an Ollama turn produced a tool
        call with reasoning; the client's history carries the bare tool call
        (no reasoning echoed); the converter restores the real text.
        """
        from llm_proxy.core.reasoning_cache import cache_reasoning_from_blocks

        cache_reasoning_from_blocks(
            [
                ThinkingBlock(thinking="Ollama reasoned about this."),
                ToolUseBlock(id="call_ollama", name="get_weather", input={"city": "Shanghai"}),
            ],
            response_id="resp_ollama",
        )

        conv = ConversationContext()
        conv.messages.append(
            Message(
                role="assistant",
                content=[
                    ToolUseBlock(id="call_ollama", name="get_weather", input={"city": "Shanghai"})
                ],
            )
        )
        messages = format_conversation(conv)
        assert messages[0]["reasoning_content"] == "Ollama reasoned about this."
