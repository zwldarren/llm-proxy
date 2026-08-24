"""Tests for SSE (Server-Sent Events) builder utilities."""

from llm_proxy.streaming.sse_builder import SSEBuilder, create_sse_error


class TestSSEBuilderInit:
    """Tests for SSEBuilder initialization."""

    def test_default_done_marker(self):
        """Default done marker should be [DONE]."""
        builder = SSEBuilder()
        assert builder.done_marker == "[DONE]"

    def test_custom_done_marker(self):
        """Custom done marker should be used."""
        builder = SSEBuilder(done_marker="[COMPLETE]")
        assert builder.done_marker == "[COMPLETE]"


class TestSerializePayload:
    """Tests for _serialize_payload method."""

    def test_serialize_dict(self):
        """Dict should be serialized to JSON."""
        builder = SSEBuilder()
        result = builder._serialize_payload({"key": "value"})
        assert result == '{"key":"value"}'

    def test_serialize_string(self):
        """String should be returned as-is."""
        builder = SSEBuilder()
        result = builder._serialize_payload("plain string")
        assert result == "plain string"

    def test_serialize_with_model_dump_json(self):
        """Object with model_dump_json should use it."""

        class MockModel:
            def model_dump_json(self):
                return '{"from_model":true}'

        builder = SSEBuilder()
        result = builder._serialize_payload(MockModel())
        assert result == '{"from_model":true}'

    def test_serialize_list(self):
        """List should be serialized to JSON."""
        builder = SSEBuilder()
        result = builder._serialize_payload([1, 2, 3])
        assert result == "[1,2,3]"


class TestDataEvent:
    """Tests for data() method."""

    def test_data_with_dict(self):
        """Dict payload should be JSON-encoded."""
        builder = SSEBuilder()
        result = builder.data({"content": "Hello"})
        assert result == 'data: {"content":"Hello"}\n\n'

    def test_data_with_string(self):
        """String payload should be returned as-is."""
        builder = SSEBuilder()
        result = builder.data("plain text")
        assert result == "data: plain text\n\n"

    def test_data_empty_dict(self):
        """Empty dict should be serialized."""
        builder = SSEBuilder()
        result = builder.data({})
        assert result == "data: {}\n\n"

    def test_data_with_nested_dict(self):
        """Nested dict should be serialized correctly."""
        builder = SSEBuilder()
        result = builder.data({"outer": {"inner": "value"}})
        assert "outer" in result
        assert "inner" in result


class TestEvent:
    """Tests for event() method."""

    def test_event_with_dict_payload(self):
        """Named event with dict payload."""
        builder = SSEBuilder()
        result = builder.event("message_start", {"type": "message_start"})
        assert result.startswith("event: message_start\n")
        assert 'data: {"type":"message_start"}' in result
        assert result.endswith("\n\n")

    def test_event_with_string_payload(self):
        """Named event with string payload."""
        builder = SSEBuilder()
        result = builder.event("error", "Something went wrong")
        assert result.startswith("event: error\n")
        assert "data: Something went wrong" in result

    def test_event_format(self):
        """Event should have correct SSE format."""
        builder = SSEBuilder()
        result = builder.event("custom", {"id": 123})
        lines = result.split("\n")
        assert lines[0] == "event: custom"
        assert lines[1].startswith("data: ")


class TestDone:
    """Tests for done() method."""

    def test_done_default_marker(self):
        """Done should use default marker."""
        builder = SSEBuilder()
        result = builder.done()
        assert result == "data: [DONE]\n\n"

    def test_done_custom_marker(self):
        """Done should use custom marker when set."""
        builder = SSEBuilder(done_marker="[COMPLETE]")
        result = builder.done()
        assert result == "data: [COMPLETE]\n\n"

    def test_done_format(self):
        """Done should have correct SSE format."""
        builder = SSEBuilder()
        result = builder.done()
        assert result.startswith("data: ")
        assert result.endswith("\n\n")


class TestError:
    """Tests for error() method."""

    def test_error_with_done(self):
        """Error should include done marker by default."""
        builder = SSEBuilder()
        error_data = {"error": {"message": "Invalid request", "type": "invalid_request_error"}}
        result = builder.error(error_data)
        assert "error" in result
        assert "data: [DONE]" in result

    def test_error_without_done(self):
        """Error without done marker."""
        builder = SSEBuilder()
        error_data = {"error": {"message": "Error"}}
        result = builder.error(error_data, include_done=False)
        assert "error" in result
        assert "[DONE]" not in result

    def test_error_format(self):
        """Error should have correct SSE data format."""
        builder = SSEBuilder()
        error_data = {"error": "test error"}
        result = builder.error(error_data)
        assert result.startswith("data: ")


class TestCreateSSEError:
    """Tests for create_sse_error convenience function."""

    def test_create_sse_error_default(self):
        """create_sse_error should use default builder."""
        error_data = {"error": {"message": "Test error"}}
        result = create_sse_error(error_data)
        assert "error" in result
        assert "[DONE]" in result

    def test_create_sse_error_without_done(self):
        """create_sse_error without done marker."""
        error_data = {"error": {"message": "Test error"}}
        result = create_sse_error(error_data, include_done=False)
        assert "error" in result
        assert "[DONE]" not in result

    def test_create_sse_error_custom_error(self):
        """create_sse_error with custom error structure."""
        error_data = {"message": "Custom error", "code": 400}
        result = create_sse_error(error_data)
        assert "Custom error" in result


class TestSSEBuilderIntegration:
    """Integration tests for SSEBuilder."""

    def test_multiple_data_events(self):
        """Multiple data events should be properly formatted."""
        builder = SSEBuilder()
        results = [
            builder.data({"chunk": 1}),
            builder.data({"chunk": 2}),
            builder.done(),
        ]
        combined = "".join(results)
        assert combined.count("data: ") == 3
        assert combined.count("\n\n") == 3

    def test_mixed_event_types(self):
        """Mixed event types should be properly formatted."""
        builder = SSEBuilder()
        results = [
            builder.event("start", {"type": "start"}),
            builder.data({"content": "hello"}),
            builder.event("end", {"type": "end"}),
            builder.done(),
        ]
        combined = "".join(results)
        assert "event: start" in combined
        assert "event: end" in combined
        assert "data: {" in combined

    def test_special_characters_in_payload(self):
        """Special characters should be properly escaped."""
        builder = SSEBuilder()
        result = builder.data({"message": "Hello\nWorld\r\nTest"})
        # orjson escapes special characters
        assert "\\n" in result or "n" in result  # Newline handling
        assert "Hello" in result
        assert "World" in result


class TestSSEBuilderEdgeCases:
    """Edge case tests."""

    def test_unicode_in_payload(self):
        """Unicode characters should be handled."""
        builder = SSEBuilder()
        result = builder.data({"text": "Hello 世界 🌍"})
        assert "Hello" in result
        assert "世界" in result

    def test_very_long_string(self):
        """Very long strings should be handled."""
        builder = SSEBuilder()
        long_string = "x" * 10000
        result = builder.data({"content": long_string})
        # Should contain the content key and the long string
        assert '"content"' in result
        assert len(result) > 10000

    def test_none_payload(self):
        """None should be serialized as null."""
        builder = SSEBuilder()
        result = builder.data({"value": None})
        assert "null" in result

    def test_numeric_payload(self):
        """Numeric values should be serialized."""
        builder = SSEBuilder()
        result = builder.data({"count": 42, "pi": 3.14})
        assert "42" in result
        assert "3.14" in result
