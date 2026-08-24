"""Tests for token counting utilities."""

import hashlib

from llm_proxy.billing.tokens import (
    TokenUsage,
    _batch_encode_texts,
    _compute_messages_hash,
    _count_text_content,
    _extract_text_fields,
    _extract_text_list,
    _get_cached_messages_tokens,
    _get_cached_token_count,
    _handle_bash_code_execution_tool_result,
    _handle_code_execution_tool_result,
    _handle_container_upload,
    _handle_image_url,
    _handle_search_result,
    _handle_server_tool_use,
    _handle_text_content,
    _handle_text_editor_tool_result,
    _handle_thinking,
    _handle_tool_reference,
    _handle_tool_result,
    _handle_tool_search_tool_result,
    _handle_tool_use,
    _handle_web_fetch_tool_result,
    _handle_web_search_tool_result,
    _messages_token_cache,
    _set_cached_messages_tokens,
    _set_cached_token_count,
    _token_cache,
    count_embedding_input_tokens,
    count_messages_tokens,
    count_tokens,
    count_tools_tokens,
    estimate_embedding_usage,
    estimate_usage_from_request,
    extract_tokens_from_usage,
)


class TestTokenUsage:
    """Tests for TokenUsage dataclass."""

    def test_default_values(self):
        """Test default values are zero."""
        usage = TokenUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0
        assert usage.cache_creation_input_tokens == 0
        assert usage.cache_read_input_tokens == 0
        assert usage.cached_prompt_tokens == 0
        assert usage.audio_input_tokens == 0
        assert usage.audio_output_tokens == 0

    def test_custom_values(self):
        """Test custom values are set correctly."""
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cache_creation_input_tokens=10,
            cache_read_input_tokens=20,
            cached_prompt_tokens=30,
            audio_input_tokens=5,
            audio_output_tokens=3,
        )
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
        assert usage.cache_creation_input_tokens == 10
        assert usage.cache_read_input_tokens == 20
        assert usage.cached_prompt_tokens == 30
        assert usage.audio_input_tokens == 5
        assert usage.audio_output_tokens == 3


class TestExtractTokensFromUsage:
    """Tests for extract_tokens_from_usage function."""

    def test_none_usage(self):
        """Test with None usage returns default TokenUsage."""
        result = extract_tokens_from_usage(None)
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0
        assert result.total_tokens == 0

    def test_empty_dict(self):
        """Test with empty dict returns default TokenUsage."""
        result = extract_tokens_from_usage({})
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0

    def test_openai_format(self):
        """Test OpenAI format token extraction."""
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }
        result = extract_tokens_from_usage(usage)
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50
        assert result.total_tokens == 150

    def test_anthropic_format(self):
        """Test Anthropic format token extraction."""
        usage = {
            "input_tokens": 100,
            "output_tokens": 50,
        }
        result = extract_tokens_from_usage(usage)
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50
        assert result.total_tokens == 150  # Calculated

    def test_anthropic_cache_tokens(self):
        """Test Anthropic cache token extraction."""
        usage = {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_creation_input_tokens": 30,
            "cache_read_input_tokens": 20,
        }
        result = extract_tokens_from_usage(usage)
        assert result.cache_creation_input_tokens == 30
        assert result.cache_read_input_tokens == 20

    def test_openai_cache_tokens(self):
        """Test OpenAI cache token extraction from prompt_tokens_details."""
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "prompt_tokens_details": {
                "cached_tokens": 40,
                "audio_tokens": 10,
            },
        }
        result = extract_tokens_from_usage(usage)
        assert result.cached_prompt_tokens == 40
        assert result.audio_input_tokens == 10

    def test_openai_audio_tokens(self):
        """Test OpenAI audio token extraction."""
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "completion_tokens_details": {
                "audio_tokens": 15,
            },
        }
        result = extract_tokens_from_usage(usage)
        assert result.audio_output_tokens == 15

    def test_realtime_usage_dialect(self):
        """Test Realtime API usage extraction (response.done shape).

        The Realtime API reports usage with ``input_token_details`` /
        ``output_token_details`` (singular "token") plus top-level
        ``input_tokens``/``output_tokens``.
        """
        usage = {
            "total_tokens": 1000,
            "input_tokens": 500,
            "output_tokens": 500,
            "input_token_details": {
                "cached_tokens": 100,
                "text_tokens": 300,
                "audio_tokens": 100,
                "image_tokens": 0,
            },
            "output_token_details": {
                "text_tokens": 200,
                "audio_tokens": 300,
            },
        }
        result = extract_tokens_from_usage(usage)
        assert result.prompt_tokens == 500
        assert result.completion_tokens == 500
        assert result.total_tokens == 1000
        assert result.cached_prompt_tokens == 100
        assert result.audio_input_tokens == 100
        assert result.audio_output_tokens == 300
        assert result.image_input_tokens == 0

    def test_realtime_dialect_is_fallback_only(self):
        """Test the Realtime dialect never overrides the chat dialect."""
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "prompt_tokens_details": {"cached_tokens": 40, "audio_tokens": 10},
            "completion_tokens_details": {"audio_tokens": 15},
            "input_token_details": {
                "cached_tokens": 999,
                "audio_tokens": 999,
            },
            "output_token_details": {"audio_tokens": 999},
        }
        result = extract_tokens_from_usage(usage)
        assert result.cached_prompt_tokens == 40
        assert result.audio_input_tokens == 10
        assert result.audio_output_tokens == 15

    def test_calculate_total_if_missing(self):
        """Test total is calculated when not provided."""
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
        }
        result = extract_tokens_from_usage(usage)
        assert result.total_tokens == 150

    def test_zero_values_treated_as_missing(self):
        """Test zero values are treated as missing."""
        usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        result = extract_tokens_from_usage(usage)
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0


class TestCountTokens:
    """Tests for count_tokens function."""

    def test_none_input(self):
        """Test None input returns 0."""
        assert count_tokens(None) == 0

    def test_empty_string(self):
        """Test empty string returns 0."""
        assert count_tokens("") == 0

    def test_simple_text(self):
        """Test simple text returns positive token count."""
        count = count_tokens("Hello, world!")
        assert count > 0

    def test_longer_text_more_tokens(self):
        """Test longer text has more tokens."""
        short = count_tokens("Hello")
        long = count_tokens("Hello, this is a much longer text with more words")
        assert long > short

    def test_caching(self):
        """Test that caching works for repeated texts."""
        text = "This is a test sentence"
        # Clear cache first
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        if text_hash in _token_cache:
            del _token_cache[text_hash]

        count1 = count_tokens(text)
        count2 = count_tokens(text)
        assert count1 == count2


class TestComputeMessagesHash:
    """Tests for _compute_messages_hash function."""

    def test_consistent_hash(self):
        """Test same messages produce same hash."""
        messages = [{"role": "user", "content": "Hello"}]
        hash1 = _compute_messages_hash(messages)
        hash2 = _compute_messages_hash(messages)
        assert hash1 == hash2

    def test_different_messages_different_hash(self):
        """Test different messages produce different hashes."""
        messages1 = [{"role": "user", "content": "Hello"}]
        messages2 = [{"role": "user", "content": "World"}]
        hash1 = _compute_messages_hash(messages1)
        hash2 = _compute_messages_hash(messages2)
        assert hash1 != hash2

    def test_complex_messages(self):
        """Test hash with complex nested messages."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        ]
        hash_val = _compute_messages_hash(messages)
        assert len(hash_val) == 16


class TestMessagesTokenCache:
    """Tests for message-level caching functions."""

    def test_set_and_get_cache(self):
        """Test setting and getting cached values."""
        _set_cached_messages_tokens("test_hash_123", 100)
        result = _get_cached_messages_tokens("test_hash_123")
        assert result == 100

    def test_get_nonexistent_cache(self):
        """Test getting non-existent cache returns None."""
        result = _get_cached_messages_tokens("nonexistent_hash_xyz")
        assert result is None


class TestTokenCache:
    """Tests for text-level token caching functions."""

    def test_set_and_get_cache(self):
        """Test setting and getting cached values."""
        _set_cached_token_count("test_text_hash", 50)
        result = _get_cached_token_count("test_text_hash")
        assert result == 50

    def test_get_nonexistent_cache(self):
        """Test getting non-existent cache returns None."""
        result = _get_cached_token_count("nonexistent_text_hash")
        assert result is None


class TestBatchEncodeTexts:
    """Tests for _batch_encode_texts function."""

    def test_empty_list(self):
        """Test empty list returns empty dict."""
        result = _batch_encode_texts([])
        assert result == {}

    def test_single_text(self):
        """Test single text returns correct count."""
        result = _batch_encode_texts(["Hello"])
        assert "Hello" in result
        assert result["Hello"] > 0

    def test_multiple_texts(self):
        """Test multiple texts return correct counts."""
        texts = ["Hello", "World", "Test"]
        result = _batch_encode_texts(texts)
        assert len(result) == 3
        for text in texts:
            assert text in result
            assert result[text] > 0

    def test_empty_string_skipped(self):
        """Test empty strings are skipped."""
        result = _batch_encode_texts(["Hello", "", "World"])
        assert len(result) == 2
        assert "" not in result


class TestExtractTextFields:
    """Tests for _extract_text_fields function."""

    def test_string_field(self):
        """Test extracting string field."""
        item = {"name": "test_name", "value": "test_value"}
        result = _extract_text_fields(item, "name", "value")
        assert result == ["test_name", "test_value"]

    def test_missing_field(self):
        """Test missing field is not included."""
        item = {"name": "test_name"}
        result = _extract_text_fields(item, "name", "missing")
        assert result == ["test_name"]

    def test_dict_field(self):
        """Test dict field is converted to string."""
        item = {"config": {"key": "value"}}
        result = _extract_text_fields(item, "config")
        assert len(result) == 1
        assert "config" in result[0] or "key" in result[0]


class TestExtractTextList:
    """Tests for _extract_text_list function."""

    def test_non_list_returns_empty(self):
        """Test non-list input returns empty list."""
        assert _extract_text_list("not a list") == []
        assert _extract_text_list({"not": "list"}) == []
        assert _extract_text_list(None) == []

    def test_string_items(self):
        """Test string items are extracted."""
        result = _extract_text_list(["Hello", "World"])
        assert result == ["Hello", "World"]

    def test_text_content_blocks(self):
        """Test text content blocks are extracted."""
        content = [{"type": "text", "text": "Hello"}]
        result = _extract_text_list(content)
        assert result == ["Hello"]


class TestContentTypeHandlers:
    """Tests for content type handler functions."""

    def test_handle_text_content(self):
        """Test text content handler."""
        result = _handle_text_content({"type": "text", "text": "Hello"})
        assert result == ["Hello"]

    def test_handle_text_content_empty(self):
        """Test text content handler with empty text."""
        result = _handle_text_content({"type": "text", "text": ""})
        assert result == []

    def test_handle_image_url(self):
        """Test image URL handler returns empty (tokens handled separately)."""
        result = _handle_image_url({"type": "image_url", "image_url": {"url": "http://..."}})
        assert result == []

    def test_handle_thinking(self):
        """Test thinking content handler."""
        result = _handle_thinking(
            {"type": "thinking", "thinking": "Thought", "signature": "sig123"}
        )
        assert "Thought" in result
        assert "sig123" in result

    def test_handle_thinking_empty(self):
        """Test thinking content handler with empty."""
        result = _handle_thinking({"type": "thinking"})
        assert result == []

    def test_handle_tool_use(self):
        """Test tool_use handler."""
        result = _handle_tool_use(
            {"type": "tool_use", "id": "tool_123", "name": "get_weather", "input": {"city": "SF"}}
        )
        assert "tool_123" in result
        assert "get_weather" in result

    def test_handle_tool_result_string(self):
        """Test tool_result handler with string content."""
        result = _handle_tool_result(
            {"type": "tool_result", "tool_use_id": "tool_123", "content": "Result string"}
        )
        assert "tool_123" in result
        assert "Result string" in result

    def test_handle_tool_result_list(self):
        """Test tool_result handler with list content."""
        result = _handle_tool_result(
            {
                "type": "tool_result",
                "tool_use_id": "tool_123",
                "content": [{"type": "text", "text": "Result"}],
            }
        )
        assert "tool_123" in result
        assert "Result" in result

    def test_handle_web_search_tool_result(self):
        """Test web_search_tool_result handler."""
        result = _handle_web_search_tool_result(
            {
                "type": "web_search_tool_result",
                "tool_use_id": "tool_123",
                "content": [
                    {"title": "Result 1", "url": "http://example.com/1"},
                    {"title": "Result 2", "url": "http://example.com/2"},
                ],
            }
        )
        assert "tool_123" in result
        assert "Result 1" in result
        assert "http://example.com/1" in result

    def test_handle_web_fetch_tool_result(self):
        """Test web_fetch_tool_result handler."""
        result = _handle_web_fetch_tool_result(
            {
                "type": "web_fetch_tool_result",
                "tool_use_id": "tool_123",
                "content": "Fetched content",
            }
        )
        assert "tool_123" in result
        assert "Fetched content" in result

    def test_handle_code_execution_tool_result(self):
        """Test code_execution_tool_result handler."""
        result = _handle_code_execution_tool_result(
            {
                "type": "code_execution_tool_result",
                "tool_use_id": "tool_123",
                "content": "Execution output",
            }
        )
        assert "tool_123" in result
        assert "Execution output" in result

    def test_handle_bash_code_execution_tool_result(self):
        """Test bash_code_execution_tool_result handler."""
        result = _handle_bash_code_execution_tool_result(
            {
                "type": "bash_code_execution_tool_result",
                "tool_use_id": "tool_123",
                "content": "Bash output",
            }
        )
        assert "tool_123" in result
        assert "Bash output" in result

    def test_handle_text_editor_tool_result(self):
        """Test text_editor_tool_result handler."""
        result = _handle_text_editor_tool_result(
            {
                "type": "text_editor_code_execution_tool_result",
                "tool_use_id": "tool_123",
                "content": "Editor output",
            }
        )
        assert "tool_123" in result
        assert "Editor output" in result

    def test_handle_tool_search_tool_result(self):
        """Test tool_search_tool_result handler."""
        result = _handle_tool_search_tool_result(
            {
                "type": "tool_search_tool_result",
                "tool_use_id": "tool_123",
                "content": "Search output",
            }
        )
        assert "tool_123" in result
        assert "Search output" in result

    def test_handle_search_result(self):
        """Test search_result handler."""
        result = _handle_search_result(
            {
                "type": "search_result",
                "file_id": "file_123",
                "title": "Document Title",
                "content": "Document content",
            }
        )
        assert "file_123" in result
        assert "Document Title" in result
        assert "Document content" in result

    def test_handle_container_upload(self):
        """Test container_upload handler."""
        result = _handle_container_upload(
            {
                "type": "container_upload",
                "file_id": "file_123",
                "filename": "test.txt",
                "content": "file content",
            }
        )
        assert "file_123" in result
        assert "test.txt" in result
        assert "file content" in result

    def test_handle_tool_reference(self):
        """Test tool_reference handler."""
        result = _handle_tool_reference(
            {
                "type": "tool_reference",
                "tool_id": "tool_123",
                "tool_name": "calculator",
                "tool_type": "function",
            }
        )
        assert "tool_123" in result
        assert "calculator" in result
        assert "function" in result

    def test_handle_server_tool_use(self):
        """Test server_tool_use handler."""
        result = _handle_server_tool_use(
            {
                "type": "server_tool_use",
                "id": "tool_123",
                "name": "server_func",
                "input": {"arg": "value"},
            }
        )
        assert "tool_123" in result
        assert "server_func" in result


class TestCountTextContent:
    """Tests for _count_text_content function."""

    def test_string_content(self):
        """Test string content returns positive count."""
        count = _count_text_content("Hello world")
        assert count > 0

    def test_empty_string(self):
        """Test empty string returns 0."""
        count = _count_text_content("")
        assert count == 0

    def test_list_of_strings(self):
        """Test list of strings returns correct count."""
        count = _count_text_content(["Hello", "World"])
        assert count > 0

    def test_list_with_text_blocks(self):
        """Test list with text blocks returns correct count."""
        count = _count_text_content(
            [{"type": "text", "text": "Hello"}, {"type": "text", "text": "World"}]
        )
        assert count > 0

    def test_list_with_image_url(self):
        """Test list with image_url adds fixed token count."""
        count_no_image = _count_text_content([{"type": "text", "text": "Hello"}])
        count_with_image = _count_text_content(
            [
                {"type": "text", "text": "Hello"},
                {"type": "image_url", "image_url": {"url": "http://..."}},
            ]
        )
        # Image adds 85 tokens
        assert count_with_image >= count_no_image + 85


class TestCountMessagesTokens:
    """Tests for count_messages_tokens function."""

    def test_empty_list(self):
        """Test empty list returns 0."""
        assert count_messages_tokens([]) == 0

    def test_simple_message(self):
        """Test simple message returns positive count."""
        messages = [{"role": "user", "content": "Hello"}]
        count = count_messages_tokens(messages)
        assert count > 0

    def test_multiple_messages(self):
        """Test multiple messages returns positive count."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        count = count_messages_tokens(messages)
        assert count > 0

    def test_message_with_tool_calls(self):
        """Test message with tool_calls returns positive count."""
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "function": {"name": "get_weather", "arguments": '{"city": "SF"}'},
                    }
                ],
            }
        ]
        count = count_messages_tokens(messages)
        assert count > 0

    def test_message_with_reasoning_content_string(self):
        """Test message with reasoning_content string."""
        messages = [
            {"role": "assistant", "content": "Answer", "reasoning_content": "Let me think..."}
        ]
        count = count_messages_tokens(messages)
        assert count > 0

    def test_message_with_reasoning_content_list(self):
        """Test message with reasoning_content list."""
        messages = [
            {
                "role": "assistant",
                "content": "Answer",
                "reasoning_content": [
                    {"text": "Thinking step 1"},
                    {"reasoning_content": "Thinking step 2"},
                ],
            }
        ]
        count = count_messages_tokens(messages)
        assert count > 0

    def test_message_with_reasoning_content_dict(self):
        """Test message with reasoning_content dict."""
        messages = [
            {
                "role": "assistant",
                "content": "Answer",
                "reasoning_content": {"text": "Thinking...", "signature": "sig123"},
            }
        ]
        count = count_messages_tokens(messages)
        assert count > 0

    def test_message_with_parts_gemini_format(self):
        """Test message with parts (Gemini format)."""
        messages = [{"role": "user", "parts": [{"text": "Hello"}, {"text": "World"}]}]
        count = count_messages_tokens(messages)
        assert count > 0

    def test_openresponses_input_items(self):
        """Test OpenResponses input items (input_text/output_text blocks)."""
        messages = [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello there"}],
            },
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hi!"}],
            },
        ]
        count = count_messages_tokens(messages)
        assert count > 0

    def test_message_with_function_call_gemini_format(self):
        """Test message with functionCall (Gemini format)."""
        messages = [
            {
                "role": "assistant",
                "parts": [{"functionCall": {"name": "get_weather", "args": {"city": "SF"}}}],
            }
        ]
        count = count_messages_tokens(messages)
        assert count > 0

    def test_message_with_function_response_gemini_format(self):
        """Test message with functionResponse (Gemini format)."""
        messages = [
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "get_weather",
                            "response": {"content": "Sunny, 72F"},
                        }
                    }
                ],
            }
        ]
        count = count_messages_tokens(messages)
        assert count > 0

    def test_non_dict_message_skipped(self):
        """Test non-dict messages are skipped."""
        messages = ["not a dict", 123, None]
        count = count_messages_tokens(messages)
        assert count >= 0  # Should not crash

    def test_caching_works(self):
        """Test message-level caching works."""
        messages = [{"role": "user", "content": "Test caching"}]
        # Clear cache
        msg_hash = _compute_messages_hash(messages)
        if msg_hash in _messages_token_cache:
            del _messages_token_cache[msg_hash]

        count1 = count_messages_tokens(messages)
        count2 = count_messages_tokens(messages)
        assert count1 == count2


class TestEstimateUsageFromRequest:
    """Tests for estimate_usage_from_request function."""

    def test_none_messages(self):
        """Test None messages returns zero prompt tokens."""
        result = estimate_usage_from_request(None, "completion")
        assert result["prompt_tokens"] == 0
        assert result["completion_tokens"] > 0

    def test_none_completion(self):
        """Test None completion returns zero completion tokens."""
        result = estimate_usage_from_request([{"role": "user", "content": "Hi"}], None)
        assert result["prompt_tokens"] > 0
        assert result["completion_tokens"] == 0

    def test_both_provided(self):
        """Test both messages and completion provided."""
        result = estimate_usage_from_request(
            [{"role": "user", "content": "Hello"}], "This is the completion"
        )
        assert result["prompt_tokens"] > 0
        assert result["completion_tokens"] > 0
        assert result["total_tokens"] == result["prompt_tokens"] + result["completion_tokens"]


class TestCountToolsTokens:
    """Tests for count_tools_tokens function."""

    def test_none_tools(self):
        """Test None tools returns 0."""
        assert count_tools_tokens(None) == 0

    def test_empty_list(self):
        """Test empty list returns 0."""
        assert count_tools_tokens([]) == 0

    def test_dict_format_tools(self):
        """Test dict format tools returns positive count."""
        tools = [
            {
                "name": "get_weather",
                "description": "Get the weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            }
        ]
        count = count_tools_tokens(tools)
        assert count > 0

    def test_anthropic_format_tools(self):
        """Test Anthropic format tools (input_schema) returns positive count."""
        tools = [
            {
                "name": "get_weather",
                "description": "Get the weather",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
            }
        ]
        count = count_tools_tokens(tools)
        assert count > 0

    def test_object_format_tools(self):
        """Test object format tools returns positive count."""

        class MockTool:
            name = "get_weather"
            description = "Get the weather"
            parameters = {"type": "object"}

        tools = [MockTool()]
        count = count_tools_tokens(tools)
        assert count > 0

    def test_multiple_tools(self):
        """Test multiple tools returns positive count."""
        tools = [
            {"name": "tool1", "description": "First tool", "parameters": {}},
            {"name": "tool2", "description": "Second tool", "parameters": {}},
        ]
        count = count_tools_tokens(tools)
        assert count > 0


class TestCountEmbeddingInputTokens:
    """Tests for count_embedding_input_tokens function."""

    def test_none_input(self):
        """Test None input returns 0."""
        assert count_embedding_input_tokens(None) == 0

    def test_empty_string(self):
        """Test empty string returns 0."""
        assert count_embedding_input_tokens("") == 0

    def test_string_input(self):
        """Test string input returns positive count."""
        count = count_embedding_input_tokens("Hello world")
        assert count > 0

    def test_list_of_strings(self):
        """Test list of strings returns sum of counts."""
        count = count_embedding_input_tokens(["Hello", "World"])
        assert count > 0

    def test_list_with_empty_strings(self):
        """Test list with empty strings skips them."""
        count = count_embedding_input_tokens(["Hello", "", "World"])
        assert count > 0

    def test_empty_list(self):
        """Test empty list returns 0."""
        assert count_embedding_input_tokens([]) == 0


class TestEstimateEmbeddingUsage:
    """Tests for estimate_embedding_usage function."""

    def test_none_input(self):
        """Test None input returns zero tokens."""
        result = estimate_embedding_usage(None)
        assert result["prompt_tokens"] == 0
        assert result["total_tokens"] == 0

    def test_string_input(self):
        """Test string input returns positive counts."""
        result = estimate_embedding_usage("Hello world")
        assert result["prompt_tokens"] > 0
        assert result["total_tokens"] > 0
        assert "completion_tokens" not in result or result.get("completion_tokens") == 0

    def test_list_input(self):
        """Test list input returns positive counts."""
        result = estimate_embedding_usage(["Hello", "World"])
        assert result["prompt_tokens"] > 0
        assert result["total_tokens"] == result["prompt_tokens"]


class TestCacheReadDedup:
    """Cache-read is one billable fact; two dialect expressions must not
    both reach TokenUsage (the cache-rate adjustment would apply twice)."""

    def test_flat_and_nested_cache_read_deduped(self):
        result = extract_tokens_from_usage(
            {
                "prompt_tokens": 1300,
                "completion_tokens": 50,
                "cache_read_input_tokens": 1000,
                "prompt_tokens_details": {"cached_tokens": 1000},
            }
        )
        assert result.cache_read_input_tokens == 1000
        # The duplicate nested expression is dropped — flat field wins.
        assert result.cached_prompt_tokens == 0

    def test_nested_only_still_maps(self):
        result = extract_tokens_from_usage(
            {
                "prompt_tokens": 1300,
                "completion_tokens": 50,
                "prompt_tokens_details": {"cached_tokens": 1000},
            }
        )
        assert result.cache_read_input_tokens == 0
        assert result.cached_prompt_tokens == 1000

    def test_flat_only_unchanged(self):
        result = extract_tokens_from_usage(
            {
                "input_tokens": 300,
                "output_tokens": 50,
                "cache_read_input_tokens": 1000,
            }
        )
        assert result.cache_read_input_tokens == 1000
        assert result.cached_prompt_tokens == 0


class TestCacheDoubleCountEndToEnd:
    async def test_dual_expression_charges_cache_once(self):
        """A usage dict expressing cache-read twice must be rate-adjusted
        exactly once in the final cost."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from llm_proxy.billing.cost import PricingRates, calculate_cost

        mock_model_config = MagicMock()
        mock_model_config.providers = []
        mock_config_manager = MagicMock()
        mock_config_manager.get_model_config = AsyncMock(return_value=mock_model_config)

        usage = {
            "prompt_tokens": 1300,
            "completion_tokens": 50,
            "cache_read_input_tokens": 1000,
            "prompt_tokens_details": {"cached_tokens": 1000},
        }
        rates = PricingRates(
            input_cost_per_1m=10.0,
            output_cost_per_1m=30.0,
            cached_read_cost_per_1m=1.0,
        )
        with patch("llm_proxy.billing.cost._get_provider_pricing", return_value=rates):
            breakdown = await calculate_cost(
                usage=usage, model_name="m", config_manager=mock_config_manager
            )

        # Single adjustment: 1000 tokens moved from $10/1M to $1/1M once.
        expected = (300 / 1e6) * 10.0 + (50 / 1e6) * 30.0 + (1000 / 1e6) * 1.0
        assert breakdown.cost_usd is not None
        assert abs(breakdown.cost_usd - expected) < 1e-9
