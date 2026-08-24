"""Tests for OpenAI Responses streaming chunk converter."""

import pytest

from llm_proxy.serialization.openai.streaming_converter import (
    OpenAIResponsesChunkConverter,
)


def _event(event_type: str, **kwargs):
    """Build a converter-compatible event dict with event_type key."""
    data = {"event_type": event_type, **kwargs}
    return data


class TestOpenAIResponsesChunkConverter:
    """Tests for OpenAIResponsesChunkConverter."""

    def test_init(self):
        """Test converter initialization."""
        converter = OpenAIResponsesChunkConverter(model="gpt-5", request_id="resp_1")
        assert converter._model == "gpt-5"
        assert converter._status == "in_progress"

    def test_handle_response_created(self):
        """Test handling response.created event."""
        converter = OpenAIResponsesChunkConverter(model="gpt-5")

        chunk = converter.convert_chunk(
            _event(
                "response.created",
                response={
                    "id": "resp_abc123",
                    "created_at": 1234567890,
                    "model": "gpt-5",
                },
            )
        )
        assert chunk is None  # No client-visible output for this event
        assert converter._response_id == "resp_abc123"
        assert converter._created_at == 1234567890

    def test_handle_text_delta(self):
        """Test handling response.output_text.delta event."""
        converter = OpenAIResponsesChunkConverter(model="gpt-5")
        converter._response_id = "resp_abc123"
        converter._created_at = 1234567890

        chunk = converter.convert_chunk(
            _event(
                "response.output_text.delta",
                delta="Hello",
                content_index=0,
                output_index=0,
            )
        )

        assert chunk is not None
        assert chunk["choices"][0]["delta"]["content"] == "Hello"
        assert chunk["object"] == "chat.completion.chunk"
        assert chunk["model"] == "gpt-5"

    def test_handle_response_completed(self):
        """Test handling response.completed event."""
        converter = OpenAIResponsesChunkConverter(model="gpt-5")
        converter._response_id = "resp_abc123"
        converter._created_at = 1234567890

        chunk = converter.convert_chunk(
            _event(
                "response.completed",
                response={
                    "id": "resp_abc123",
                    "status": "completed",
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 50,
                        "total_tokens": 150,
                    },
                },
            )
        )
        # Completed event queues the final chunk; convert_chunk returns None.
        assert chunk is None

        final = converter.finalize_chunks()
        assert len(final) == 1
        assert final[0]["choices"][0]["finish_reason"] == "stop"
        assert final[0]["usage"]["prompt_tokens"] == 100
        assert final[0]["usage"]["completion_tokens"] == 50
        assert converter._status == "completed"
        assert converter._input_tokens == 100
        assert converter._output_tokens == 50

    def test_handle_function_call(self):
        """Test handling function call events."""
        converter = OpenAIResponsesChunkConverter(model="gpt-5")
        converter._response_id = "resp_abc123"
        converter._created_at = 1234567890

        # Simulate output_item.added
        converter.convert_chunk(
            _event(
                "response.output_item.added",
                item={
                    "id": "item_1",
                    "type": "function_call",
                    "call_id": "call_123",
                    "name": "get_weather",
                },
            )
        )

        # Simulate arguments deltas
        converter.convert_chunk(
            _event(
                "response.function_call_arguments.delta",
                delta='{"loc',
            )
        )
        converter.convert_chunk(
            _event(
                "response.function_call_arguments.delta",
                delta='ation": "NYC"}',
            )
        )

        # Simulate arguments done
        chunk = converter.convert_chunk(_event("response.function_call_arguments.done"))

        assert chunk is not None
        tc = chunk["choices"][0]["delta"]["tool_calls"][0]
        assert tc["function"]["name"] == "get_weather"
        assert tc["function"]["arguments"] == '{"location": "NYC"}'
        assert len(converter._tool_calls) == 1
        assert converter._tool_calls[0]["arguments"] == '{"location": "NYC"}'
        assert converter._active_tool_calls == {}  # cleared after done

    def test_handle_parallel_function_calls(self):
        """Handle parallel tool calls with unique item_id values.

        The converter must track each tool call independently so no
        arguments are lost, even when output_index is duplicated.
        """
        converter = OpenAIResponsesChunkConverter(model="deepseek-v4-flash")
        converter._response_id = "resp_parallel"
        converter._created_at = 1234567890

        # Item 1: call_txuin6jg
        converter.convert_chunk(
            _event(
                "response.output_item.added",
                output_index=1,
                item={
                    "id": "item_a",
                    "type": "function_call",
                    "call_id": "call_txuin6jg",
                    "name": "read",
                },
            )
        )
        converter.convert_chunk(
            _event(
                "response.function_call_arguments.delta",
                output_index=1,
                delta='{"path":"/home/a.py"}',
                item_id="item_a",
            )
        )

        # Item 2: call_wg5t905z
        converter.convert_chunk(
            _event(
                "response.output_item.added",
                output_index=1,
                item={
                    "id": "item_b",
                    "type": "function_call",
                    "call_id": "call_wg5t905z",
                    "name": "read",
                },
            )
        )
        converter.convert_chunk(
            _event(
                "response.function_call_arguments.delta",
                output_index=2,
                delta='{"path":"/home/b.py"}',
                item_id="item_b",
            )
        )

        # Item 3: call_ozncdlkw
        converter.convert_chunk(
            _event(
                "response.output_item.added",
                output_index=1,
                item={
                    "id": "item_c",
                    "type": "function_call",
                    "call_id": "call_ozncdlkw",
                    "name": "read",
                },
            )
        )
        converter.convert_chunk(
            _event(
                "response.function_call_arguments.delta",
                output_index=3,
                delta='{"path":"/home/c.py"}',
                item_id="item_c",
            )
        )

        # Item 4: call_yr7g6679
        converter.convert_chunk(
            _event(
                "response.output_item.added",
                output_index=1,
                item={
                    "id": "item_d",
                    "type": "function_call",
                    "call_id": "call_yr7g6679",
                    "name": "read",
                },
            )
        )
        converter.convert_chunk(
            _event(
                "response.function_call_arguments.delta",
                output_index=4,
                delta='{"path":"/home/d.py"}',
                item_id="item_d",
            )
        )

        # Now emit done events for each.
        for oi, iid, cid, path in [
            (1, "item_a", "call_txuin6jg", "/home/a.py"),
            (2, "item_b", "call_wg5t905z", "/home/b.py"),
            (3, "item_c", "call_ozncdlkw", "/home/c.py"),
            (4, "item_d", "call_yr7g6679", "/home/d.py"),
        ]:
            chunk = converter.convert_chunk(
                _event(
                    "response.function_call_arguments.done",
                    output_index=oi,
                    item_id=iid,
                )
            )
            assert chunk is not None
            tc = chunk["choices"][0]["delta"]["tool_calls"][0]
            assert tc["id"] == cid, f"Expected {cid}, got {tc['id']}"
            assert tc["function"]["arguments"] == f'{{"path":"{path}"}}'

        # All 4 tool calls captured with correct arguments.
        assert len(converter._tool_calls) == 4
        assert converter._tool_calls[0]["arguments"] == '{"path":"/home/a.py"}'
        assert converter._tool_calls[1]["arguments"] == '{"path":"/home/b.py"}'
        assert converter._tool_calls[2]["arguments"] == '{"path":"/home/c.py"}'
        assert converter._tool_calls[3]["arguments"] == '{"path":"/home/d.py"}'
        assert converter._active_tool_calls == {}  # all cleared

    def test_handle_error(self):
        """Test handling error event."""
        converter = OpenAIResponsesChunkConverter(model="gpt-5")

        chunk = converter.convert_chunk(
            _event(
                "error",
                error={
                    "type": "invalid_request_error",
                    "code": "invalid_api_key",
                    "message": "Invalid API key",
                },
            )
        )

        assert chunk is not None
        assert "error" in chunk
        assert chunk["error"]["message"] == "Invalid API key"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
