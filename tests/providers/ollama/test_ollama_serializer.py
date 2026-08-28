"""Unit tests for OllamaProviderSerializer._convert_conversation_to_ollama."""

import base64

import pytest

from llm_proxy.core.exceptions import ValidationError
from llm_proxy.models import (
    ConversationContext,
    DocumentBlock,
    FileBlock,
    ImageBlock,
    Message,
    RefusalBlock,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from llm_proxy.models.types import DocumentSource, ImageSource
from llm_proxy.serialization.context import BuildContext
from llm_proxy.serialization.ollama.serializer import OllamaProviderSerializer


@pytest.fixture
def serializer():
    return OllamaProviderSerializer()


@pytest.fixture
def degrade_ctx():
    return BuildContext(unsupported_block_policy="degrade")


class TestBasicTextMessages:
    def test_single_user_message(self, serializer):
        conv = ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="Hello")])]
        )
        result = serializer._convert_conversation_to_ollama(conv)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"

    def test_user_and_assistant(self, serializer):
        conv = ConversationContext(
            messages=[
                Message(role="user", content=[TextBlock(text="Hi")]),
                Message(role="assistant", content=[TextBlock(text="Hello there")]),
            ]
        )
        result = serializer._convert_conversation_to_ollama(conv)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hi"
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == "Hello there"

    def test_multiple_text_blocks_in_one_message(self, serializer):
        conv = ConversationContext(
            messages=[
                Message(
                    role="user",
                    content=[TextBlock(text="Part 1"), TextBlock(text="Part 2")],
                )
            ]
        )
        result = serializer._convert_conversation_to_ollama(conv)
        assert result[0]["content"] == "Part 1 Part 2"

    def test_empty_message_is_skipped(self, serializer):
        conv = ConversationContext(messages=[Message(role="user", content=[])])
        result = serializer._convert_conversation_to_ollama(conv)
        assert len(result) == 0


class TestSystemMessages:
    def test_system_messages_at_beginning(self, serializer):
        conv = ConversationContext(
            system_messages=[SystemMessage.from_text(role="system", text="You are helpful.")],
            messages=[Message(role="user", content=[TextBlock(text="Hi")])],
        )
        result = serializer._convert_conversation_to_ollama(conv)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are helpful."
        assert result[1]["role"] == "user"

    def test_developer_role_mapped_to_system(self, serializer):
        conv = ConversationContext(
            messages=[Message(role="developer", content=[TextBlock(text="Dev prompt")])],
        )
        result = serializer._convert_conversation_to_ollama(conv)
        assert len(result) == 1
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "Dev prompt"

    def test_empty_system_message_skipped(self, serializer):
        conv = ConversationContext(
            system_messages=[SystemMessage(role="system", content=[])],
            messages=[Message(role="user", content=[TextBlock(text="Hi")])],
        )
        result = serializer._convert_conversation_to_ollama(conv)
        assert len(result) == 1
        assert result[0]["role"] == "user"


class TestImages:
    def test_base64_image(self, serializer):
        img_data = base64.b64encode(b"fake-image").decode()
        conv = ConversationContext(
            messages=[
                Message(
                    role="user",
                    content=[
                        TextBlock(text="Describe this:"),
                        ImageBlock(
                            source=ImageSource(type="base64", data=img_data, media_type="image/png")
                        ),
                    ],
                )
            ]
        )
        result = serializer._convert_conversation_to_ollama(conv)
        assert result[0]["content"] == "Describe this:"
        assert result[0]["images"] == [img_data]

    def test_data_url_image(self, serializer):
        img_data = base64.b64encode(b"fake-image").decode()
        data_url = f"data:image/png;base64,{img_data}"
        conv = ConversationContext(
            messages=[
                Message(
                    role="user",
                    content=[
                        ImageBlock(source=ImageSource(type="url", data=data_url, media_type=None)),
                    ],
                )
            ]
        )
        result = serializer._convert_conversation_to_ollama(conv)
        assert result[0]["content"] == ""
        assert result[0]["images"] == [img_data]

    def test_corrupted_data_url_base64_raises_validation_error(self, serializer):
        conv = ConversationContext(
            messages=[
                Message(
                    role="user",
                    content=[
                        ImageBlock(
                            source=ImageSource(
                                type="url",
                                data="data:image/png;base64,!!!invalid!!!",
                                media_type=None,
                            )
                        ),
                    ],
                )
            ]
        )
        with pytest.raises(ValidationError):
            serializer._convert_conversation_to_ollama(conv)

    def test_corrupted_bare_base64_raises_validation_error(self, serializer):
        conv = ConversationContext(
            messages=[
                Message(
                    role="user",
                    content=[
                        ImageBlock(
                            source=ImageSource(
                                type="url",
                                data="!!!not_base64!!!",
                                media_type=None,
                            )
                        ),
                    ],
                )
            ]
        )
        with pytest.raises(ValidationError):
            serializer._convert_conversation_to_ollama(conv)

    def test_http_url_without_base64_fallback(self, serializer):
        conv = ConversationContext(
            messages=[
                Message(
                    role="user",
                    content=[
                        ImageBlock(
                            source=ImageSource(
                                type="url",
                                data="https://example.com/img.png",
                                media_type=None,
                            )
                        ),
                    ],
                )
            ]
        )
        result = serializer._convert_conversation_to_ollama(conv)
        assert "[Image URL: https://example.com/img.png]" in result[0]["content"]


class TestToolUseBlocks:
    def test_tool_use_in_assistant_message(self, serializer):
        conv = ConversationContext(
            messages=[
                Message(
                    role="assistant",
                    content=[
                        ToolUseBlock(id="call_123", name="get_weather", input={"city": "NYC"}),
                    ],
                )
            ]
        )
        result = serializer._convert_conversation_to_ollama(conv)
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == ""
        assert "id" not in result[0]["tool_calls"][0]
        assert result[0]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert result[0]["tool_calls"][0]["function"]["arguments"] == {"city": "NYC"}

    def test_tool_use_with_text(self, serializer):
        conv = ConversationContext(
            messages=[
                Message(
                    role="assistant",
                    content=[
                        TextBlock(text="Let me check"),
                        ToolUseBlock(id="call_123", name="search", input={"q": "test"}),
                    ],
                )
            ]
        )
        result = serializer._convert_conversation_to_ollama(conv)
        assert result[0]["content"] == "Let me check"
        assert len(result[0]["tool_calls"]) == 1


class TestToolResultBlocks:
    def test_tool_result_in_user_message(self, serializer):
        conv = ConversationContext(
            messages=[
                Message(
                    role="user",
                    content=[
                        ToolResultBlock(
                            tool_use_id="call_123",
                            content="Sunny, 22C",
                        ),
                    ],
                )
            ]
        )
        result = serializer._convert_conversation_to_ollama(conv)
        # User message should be skipped (no remaining content after extracting tool results)
        # Tool result becomes a separate tool-role message
        tool_msgs = [m for m in result if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["content"] == "Sunny, 22C"
        assert "tool_call_id" not in tool_msgs[0]

    def test_tool_result_with_name_lookup(self, serializer):
        conv = ConversationContext(
            messages=[
                Message(
                    role="assistant",
                    content=[
                        ToolUseBlock(id="call_abc", name="get_temp", input={"city": "SF"}),
                    ],
                ),
                Message(
                    role="user",
                    content=[
                        ToolResultBlock(
                            tool_use_id="call_abc",
                            content="18C",
                        ),
                    ],
                ),
            ]
        )
        result = serializer._convert_conversation_to_ollama(conv)
        tool_msgs = [m for m in result if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_name"] == "get_temp"
        assert "tool_call_id" not in tool_msgs[0]

    def test_tool_result_combined_with_text(self, serializer):
        conv = ConversationContext(
            messages=[
                Message(
                    role="user",
                    content=[
                        TextBlock(text="Here is the result:"),
                        ToolResultBlock(
                            tool_use_id="call_x",
                            content="Done",
                        ),
                    ],
                )
            ]
        )
        result = serializer._convert_conversation_to_ollama(conv)
        user_msgs = [m for m in result if m["role"] == "user"]
        tool_msgs = [m for m in result if m["role"] == "tool"]
        assert len(user_msgs) == 1
        assert user_msgs[0]["content"] == "Here is the result:"
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["content"] == "Done"

    def test_tool_result_dict_content(self, serializer):
        conv = ConversationContext(
            messages=[
                Message(
                    role="user",
                    content=[
                        ToolResultBlock(
                            tool_use_id="call_z",
                            content={"key": "value"},
                        ),
                    ],
                )
            ]
        )
        result = serializer._convert_conversation_to_ollama(conv)
        tool_msgs = [m for m in result if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["content"] == "{'key': 'value'}"


class TestFileAndDocumentBlocks:
    """Ollama has no native document/audio/file parts.

    Document blocks are surfaced as text: plain-text sources are extracted and
    binary documents (e.g. PDFs) are degraded to placeholders so they are not
    silently lost. Audio and file blocks are still dropped, since Ollama cannot
    interpret them and text degradation adds no value there.
    """

    def test_file_block_is_dropped(self, serializer, degrade_ctx):
        text_content = "Hello from file"
        file_base64 = base64.b64encode(text_content.encode()).decode()
        conv = ConversationContext(
            messages=[
                Message(
                    role="user",
                    content=[
                        TextBlock(text="hello"),
                        FileBlock(
                            filename="test.txt",
                            file_data=file_base64,
                        ),
                    ],
                )
            ]
        )
        result = serializer._convert_conversation_to_ollama(conv, degrade_ctx)
        # FileBlock is silently dropped, only TextBlock content remains
        assert result[0]["content"] == "hello"

    def test_document_block_base64_text_is_extracted(self, serializer, degrade_ctx):
        conv = ConversationContext(
            messages=[
                Message(
                    role="user",
                    content=[
                        TextBlock(text="hello"),
                        DocumentBlock(
                            source=DocumentSource(
                                type="base64",
                                data=base64.b64encode(b"doc").decode(),
                                media_type="text/plain",
                            ),
                            title="readme.md",
                        ),
                    ],
                )
            ]
        )
        result = serializer._convert_conversation_to_ollama(conv, degrade_ctx)
        # base64 text/plain document is decoded into text, not dropped
        assert result[0]["content"] == "hello doc"

    def test_document_block_text_source_extracted(self, serializer, degrade_ctx):
        conv = ConversationContext(
            messages=[
                Message(
                    role="user",
                    content=[
                        DocumentBlock(
                            source=DocumentSource(
                                type="text",
                                data="plain text doc",
                                media_type="text/plain",
                            )
                        ),
                    ],
                )
            ]
        )
        result = serializer._convert_conversation_to_ollama(conv, degrade_ctx)
        # text-source document content is extracted verbatim
        assert result[0]["content"] == "plain text doc"

    def test_document_block_base64_pdf_degraded(self, serializer, degrade_ctx):
        conv = ConversationContext(
            messages=[
                Message(
                    role="user",
                    content=[
                        TextBlock(text="hello"),
                        DocumentBlock(
                            source=DocumentSource(
                                type="base64",
                                data=base64.b64encode(b"%PDF-1.4").decode(),
                                media_type="application/pdf",
                            ),
                            title="report.pdf",
                        ),
                    ],
                )
            ]
        )
        result = serializer._convert_conversation_to_ollama(conv, degrade_ctx)
        # binary PDF document degrades to a placeholder instead of being dropped
        assert result[0]["content"] == "hello [Document: report.pdf]"

    def test_audio_block_is_dropped(self, serializer, degrade_ctx):
        from llm_proxy.models import AudioBlock, AudioSource

        conv = ConversationContext(
            messages=[
                Message(
                    role="user",
                    content=[
                        TextBlock(text="hello"),
                        AudioBlock(
                            source=AudioSource(
                                type="base64",
                                data="ZmFrZQ==",
                                media_type="audio/mp3",
                            ),
                        ),
                    ],
                )
            ]
        )
        result = serializer._convert_conversation_to_ollama(conv, degrade_ctx)
        # AudioBlock is silently dropped, only TextBlock content remains
        assert result[0]["content"] == "hello"


class TestThinkingBlocks:
    def test_thinking_in_assistant_message(self, serializer):
        conv = ConversationContext(
            messages=[
                Message(
                    role="assistant",
                    content=[ThinkingBlock(thinking="Let me think about this...")],
                ),
                Message(
                    role="assistant",
                    content=[TextBlock(text="Answer")],
                ),
            ]
        )
        result = serializer._convert_conversation_to_ollama(conv)
        thinking_msg = result[0]
        assert thinking_msg["role"] == "assistant"
        assert thinking_msg["thinking"] == "Let me think about this..."
        assert thinking_msg["content"] == ""

    def test_thinking_with_text_in_same_message(self, serializer):
        conv = ConversationContext(
            messages=[
                Message(
                    role="assistant",
                    content=[
                        ThinkingBlock(thinking="Hmm..."),
                        TextBlock(text="Here is the answer"),
                    ],
                )
            ]
        )
        result = serializer._convert_conversation_to_ollama(conv)
        assert result[0]["thinking"] == "Hmm..."
        assert result[0]["content"] == "Here is the answer"


class TestMiscBlocks:
    def test_refusal_block_converted_to_text(self, serializer):
        conv = ConversationContext(
            messages=[
                Message(
                    role="assistant",
                    content=[RefusalBlock(refusal="I cannot do that.")],
                )
            ]
        )
        result = serializer._convert_conversation_to_ollama(conv)
        assert result[0]["content"] == "I cannot do that."


class TestResponseDurationMetrics:
    """Ollama native duration metrics must survive response parsing."""

    def test_duration_metrics_in_provider_info(self, serializer):
        response = {
            "message": {"role": "assistant", "content": "Hello"},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 5,
            "eval_count": 7,
            "total_duration": 1234567890,
            "load_duration": 123450000,
            "prompt_eval_duration": 234560000,
            "eval_duration": 987654321,
        }
        result = serializer.parse_provider_response(response, model="llama3.1")
        assert result.provider_info.get("ollama_metrics") == {
            "total_duration": 1234567890,
            "load_duration": 123450000,
            "prompt_eval_duration": 234560000,
            "eval_duration": 987654321,
        }


class TestResponseDoneReason:
    """Raw done_reason is preserved for observability even when it does not
    map to an OpenAI finish_reason."""

    def test_load_done_reason_preserved(self, serializer):
        """done_reason "load" (empty messages load the model) maps to
        finish_reason "stop" on the wire but stays visible in provider_info."""
        response = {
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "load",
        }
        result = serializer.parse_provider_response(response, model="llama3.2")
        assert result.finish_reason == "stop"
        assert result.provider_info.get("done_reason") == "load"

    def test_unload_done_reason_preserved(self, serializer):
        """done_reason "unload" (keep_alive=0) is preserved too."""
        response = {
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "unload",
        }
        result = serializer.parse_provider_response(response, model="llama3.2")
        assert result.finish_reason == "stop"
        assert result.provider_info.get("done_reason") == "unload"

    def test_standard_done_reason_preserved(self, serializer):
        """Standard reasons are preserved as-is."""
        response = {
            "message": {"role": "assistant", "content": "hi"},
            "done": True,
            "done_reason": "length",
        }
        result = serializer.parse_provider_response(response, model="llama3.2")
        assert result.finish_reason == "length"
        assert result.provider_info.get("done_reason") == "length"


class TestResponseLogprobs:
    """Non-streaming logprobs are parsed into InternalResponse.logprobs."""

    def _response(self):
        return {
            "model": "llama3.2",
            "message": {"role": "assistant", "content": "hi"},
            "done": True,
            "done_reason": "stop",
            "logprobs": [
                {
                    "token": "Hello",
                    "logprob": -0.5,
                    "bytes": [72, 101, 108, 108, 111],
                    "top_logprobs": [
                        {"token": "Hello", "logprob": -0.5},
                        {"token": "Hi", "logprob": -1.2},
                    ],
                },
            ],
        }

    def test_logprobs_parsed_when_requested(self, serializer):
        """logprobs=True populates the typed ChoiceLogprobs field."""
        result = serializer.parse_provider_response(
            self._response(), model="llama3.2", logprobs=True
        )
        assert result.logprobs is not None
        assert result.logprobs.content is not None
        entry = result.logprobs.content[0]
        assert entry.token == "Hello"
        assert entry.logprob == -0.5
        assert entry.bytes == [72, 101, 108, 108, 111]
        assert entry.top_logprobs is not None
        assert [t.token for t in entry.top_logprobs] == ["Hello", "Hi"]

    def test_logprobs_skipped_when_not_requested(self, serializer):
        """Without the logprobs flag the payload is not parsed."""
        result = serializer.parse_provider_response(self._response(), model="llama3.2")
        assert result.logprobs is None

    def test_logprobs_empty_list(self, serializer):
        """An empty logprobs list yields no logprobs field."""
        response = self._response()
        response["logprobs"] = []
        result = serializer.parse_provider_response(response, model="llama3.2", logprobs=True)
        assert result.logprobs is None


class TestEmbeddingResponse:
    """Ollama /api/embed response parsing."""

    def test_embeddings_parsed(self, serializer):
        response = {
            "model": "nomic-embed-text",
            "embeddings": [[0.1, 0.2], [0.3, 0.4]],
        }
        result = serializer.parse_provider_embedding_response(response)
        assert result.model == "nomic-embed-text"
        assert len(result.data) == 2
        assert result.data[0].embedding == [0.1, 0.2]
        assert result.data[1].index == 1

    def test_prompt_eval_count_parsed_as_usage(self, serializer):
        """prompt_eval_count from /api/embed becomes usage for billing."""
        response = {
            "model": "nomic-embed-text",
            "embeddings": [[0.1, 0.2]],
            "prompt_eval_count": 7,
        }
        result = serializer.parse_provider_embedding_response(response)
        assert result.usage is not None
        assert result.usage.input_tokens == 7
        assert result.usage.total_tokens == 7

    def test_no_usage_without_prompt_eval_count(self, serializer):
        response = {"model": "nomic-embed-text", "embeddings": [[0.1, 0.2]]}
        result = serializer.parse_provider_embedding_response(response)
        assert result.usage is None


class TestResponseBlockOrder:
    """Ollama response parsing must emit thinking before answer text."""

    def test_thinking_block_precedes_text_block(self, serializer):
        """Thinking must be the first block.

        Anthropic-protocol rendering preserves block order (thinking blocks
        are only valid before text blocks), and streaming emits thinking
        deltas first — non-streaming must match that order.
        """
        response = {
            "model": "qwen3",
            "message": {
                "role": "assistant",
                "content": "The answer is 42.",
                "thinking": "Let me think about this...",
            },
            "done": True,
            "done_reason": "stop",
        }
        result = serializer.parse_provider_response(response, model="qwen3")

        assert [type(block).__name__ for block in result.output] == [
            "ThinkingBlock",
            "TextBlock",
        ]

    def test_thinking_only_no_text(self, serializer):
        """A thinking-only response still yields a single ThinkingBlock."""
        response = {
            "model": "qwen3",
            "message": {"role": "assistant", "content": "", "thinking": "Hmm..."},
            "done": True,
            "done_reason": "stop",
        }
        result = serializer.parse_provider_response(response, model="qwen3")

        assert [type(block).__name__ for block in result.output] == ["ThinkingBlock"]

    def test_empty_thinking_not_emitted(self, serializer):
        """Whitespace-only thinking must not produce an empty block."""
        response = {
            "model": "qwen3",
            "message": {"role": "assistant", "content": "Answer", "thinking": "   "},
            "done": True,
            "done_reason": "stop",
        }
        result = serializer.parse_provider_response(response, model="qwen3")

        assert [type(block).__name__ for block in result.output] == ["TextBlock"]


class TestComplexConversations:
    def test_multi_turn_with_tools(self, serializer):
        conv = ConversationContext(
            system_messages=[SystemMessage.from_text(role="system", text="You are a helper.")],
            messages=[
                Message(role="user", content=[TextBlock(text="What is the weather?")]),
                Message(
                    role="assistant",
                    content=[
                        ToolUseBlock(id="call_1", name="get_weather", input={"city": "NYC"}),
                    ],
                ),
                Message(
                    role="user",
                    content=[
                        TextBlock(text="Result:"),
                        ToolResultBlock(tool_use_id="call_1", content="Sunny"),
                    ],
                ),
                Message(
                    role="assistant",
                    content=[TextBlock(text="The weather in NYC is sunny.")],
                ),
            ],
        )
        result = serializer._convert_conversation_to_ollama(conv)
        roles = [m["role"] for m in result]
        assert roles == ["system", "user", "assistant", "user", "tool", "assistant"]

    def test_images_only_message(self, serializer):
        img_data = base64.b64encode(b"img").decode()
        conv = ConversationContext(
            messages=[
                Message(
                    role="user",
                    content=[
                        ImageBlock(
                            source=ImageSource(type="base64", data=img_data, media_type="image/png")
                        ),
                    ],
                )
            ]
        )
        result = serializer._convert_conversation_to_ollama(conv)
        assert result[0]["content"] == ""
        assert result[0]["images"] == [img_data]

    def test_system_role_message_converts_to_user_xml(self, serializer):
        """Message(role='system') in conversation.messages must become
        role='user' with <system-prompt> XML wrapping in Ollama format.
        (Matches Anthropic/OpenAI provider degradation logic.)"""
        msg1 = Message(role="user", content=[TextBlock(text="Hello")])
        msg2 = Message(role="system", content=[TextBlock(text="Speak only French.")])
        msg3 = Message(role="user", content=[TextBlock(text="How are you?")])

        conv = ConversationContext(messages=[msg1, msg2, msg3])

        result = serializer._convert_conversation_to_ollama(conv)

        assert len(result) == 3
        sys_msg = result[1]
        assert sys_msg["role"] == "user", f"Expected 'user', got '{sys_msg['role']}'"
        assert "<system-prompt>" in sys_msg["content"]
        assert "Speak only French." in sys_msg["content"]
        assert "</system-prompt>" in sys_msg["content"]
