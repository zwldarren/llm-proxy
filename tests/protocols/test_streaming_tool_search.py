"""Tests for tool_search_call streaming and namespace restoration."""

import orjson

from llm_proxy.protocols.openresponses.handler import (
    _request_context,
    clear_format_context,
)
from llm_proxy.serialization.format_context import FormatContext


class TestToolSearchCallStreaming:
    """Test that tool_search tool calls are emitted as tool_search_call items."""

    def test_tool_search_call_emitted_as_tool_search_call(self):
        """Function name 'tool_search' should emit tool_search_call items."""
        clear_format_context()
        try:
            from llm_proxy.protocols.openresponses.streaming import (
                OpenResponsesStreamingTransformer,
            )

            transformer = OpenResponsesStreamingTransformer(model="gpt-4", request_id="test-ts")

            chunks = [
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_ts_1",
                                        "type": "function",
                                        "function": {
                                            "name": "tool_search",
                                            "arguments": '{"query": "test"}',
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ]
                },
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
            ]

            events = ""
            for chunk in chunks:
                result = transformer.transform(chunk)
                if result:
                    events += result

            # Parse events and find output_item.added for tool_search_call
            added_event = None
            done_event = None
            for line in events.split("\n\n"):
                if not line.strip():
                    continue
                data_prefix = "data: "
                data_start = line.find(data_prefix)
                if data_start == -1:
                    continue
                data = orjson.loads(line[data_start + len(data_prefix) :])
                event_type = data.get("type", "")
                if (
                    event_type == "response.output_item.added"
                    and data.get("item", {}).get("type") == "tool_search_call"
                ):
                    added_event = data
                if (
                    event_type == "response.output_item.done"
                    and data.get("item", {}).get("type") == "tool_search_call"
                ):
                    done_event = data

            # Verify added event
            assert added_event is not None, "Expected tool_search_call output_item.added event"
            item = added_event["item"]
            assert item["type"] == "tool_search_call"
            assert item["call_id"] == "call_ts_1"
            assert item["status"] == "in_progress"
            assert item["execution"] == "client"
            assert item["name"] == "tool_search"

            # Verify done event
            assert done_event is not None, "Expected tool_search_call output_item.done event"
            item = done_event["item"]
            assert item["type"] == "tool_search_call"
            assert item["status"] == "completed"
            assert item["execution"] == "client"
            assert item["call_id"] == "call_ts_1"
            assert item["name"] == "tool_search"
        finally:
            clear_format_context()

    def test_tool_search_accumulation(self):
        """Accumulated tool_search calls should appear in get_accumulated_output."""
        clear_format_context()
        try:
            from llm_proxy.models.content_blocks import ToolUseBlock
            from llm_proxy.protocols.openresponses.streaming import (
                OpenResponsesStreamingTransformer,
            )

            transformer = OpenResponsesStreamingTransformer(model="gpt-4", request_id="test-acc")

            chunks = [
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_ts_acc",
                                        "type": "function",
                                        "function": {
                                            "name": "tool_search",
                                            "arguments": '{"query": "find"}',
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ]
                },
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
            ]

            for chunk in chunks:
                transformer.transform(chunk)

            accumulated = transformer.get_accumulated_output()
            assert len(accumulated) == 1
            assert isinstance(accumulated[0], ToolUseBlock)
            assert accumulated[0].id == "call_ts_acc"
            assert accumulated[0].name == "tool_search"
            assert accumulated[0].input == {"query": "find"}
        finally:
            clear_format_context()


class TestNamespaceRestoration:
    """Test namespace name restoration in streaming output."""

    def test_namespace_restored_in_done_event(self):
        """Flat name 'mcp__github__list_issues' → 'list_issues' in output_item.done."""
        # Set up FormatContext with namespace_map
        ctx = FormatContext(namespace_map={"mcp__github__list_issues": ["github", "list_issues"]})
        _request_context.set(ctx)

        try:
            from llm_proxy.protocols.openresponses.streaming import (
                OpenResponsesStreamingTransformer,
            )

            transformer = OpenResponsesStreamingTransformer(model="gpt-4", request_id="test-ns")

            chunks = [
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_mcp",
                                        "type": "function",
                                        "function": {
                                            "name": "mcp__github__list_issues",
                                            "arguments": '{"repo": "llm-proxy"}',
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ]
                },
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
            ]

            events = ""
            for chunk in chunks:
                result = transformer.transform(chunk)
                if result:
                    events += result

            # Find the output_item.added and output_item.done events
            added_event = None
            done_event = None
            for line in events.split("\n\n"):
                if not line.strip():
                    continue
                data_prefix = "data: "
                data_start = line.find(data_prefix)
                if data_start == -1:
                    continue
                data = orjson.loads(line[data_start + len(data_prefix) :])
                if (
                    data.get("type") == "response.output_item.added"
                    and data.get("item", {}).get("type") == "function_call"
                ):
                    added_event = data
                if (
                    data.get("type") == "response.output_item.done"
                    and data.get("item", {}).get("type") == "function_call"
                ):
                    done_event = data

            # Namespace restoration is applied consistently across added,
            # done, and completed events; the namespace is emitted alongside
            # the restored short name.
            assert added_event is not None, "Expected output_item.added event"
            assert added_event["item"]["name"] == "list_issues"
            assert added_event["item"]["namespace"] == "github"
            assert done_event is not None, "Expected output_item.done event"
            item = done_event["item"]
            assert item["name"] == "list_issues"
            assert item["namespace"] == "github"
            assert item["call_id"] == "call_mcp"
        finally:
            clear_format_context()

    def test_unmapped_names_passthrough(self):
        """Unmapped names should pass through unchanged."""
        ctx = FormatContext(namespace_map={"other_tool": ["other", "tool"]})
        _request_context.set(ctx)

        try:
            from llm_proxy.protocols.openresponses.streaming import (
                OpenResponsesStreamingTransformer,
            )

            transformer = OpenResponsesStreamingTransformer(model="gpt-4", request_id="test-pt")

            chunks = [
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_wx",
                                        "type": "function",
                                        "function": {
                                            "name": "get_weather",
                                            "arguments": '{"city": "SF"}',
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ]
                },
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
            ]

            events = ""
            for chunk in chunks:
                result = transformer.transform(chunk)
                if result:
                    events += result

            done_event = None
            for line in events.split("\n\n"):
                if not line.strip():
                    continue
                data_prefix = "data: "
                data_start = line.find(data_prefix)
                if data_start == -1:
                    continue
                data = orjson.loads(line[data_start + len(data_prefix) :])
                if (
                    data.get("type") == "response.output_item.done"
                    and data.get("item", {}).get("type") == "function_call"
                ):
                    done_event = data
                    break

            assert done_event is not None
            item = done_event["item"]
            assert item["name"] == "get_weather"
        finally:
            clear_format_context()
