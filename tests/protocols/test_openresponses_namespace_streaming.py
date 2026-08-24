"""Regression tests for namespace/custom tool-call streaming fixes.

Covers two bugs exposed by the Ollama provider degenerate stream (see
request_logs #13298):

1. Models echoing the short history name (``exec``) instead of the flattened
   tool-definition name (``functions__exec``) were classified as plain
   function_call items because the custom-name check used exact matching.
2. Text deltas arriving after a tool call were attributed to the open
   function_call item id, and tool-call items were closed with text events —
   both invalid event sequences.
"""

import orjson

from llm_proxy.protocols.openresponses.streaming import OpenResponsesStreamingTransformer


def _parse_events(events: str) -> list[dict]:
    parsed: list[dict] = []
    for line in events.split("\n"):
        line = line.strip()
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        if payload == "[DONE]":
            continue
        parsed.append(orjson.loads(payload))
    return parsed


_NAMESPACE_TOOLS = [
    {
        "type": "namespace",
        "name": "functions",
        "tools": [
            {"type": "custom", "name": "exec", "description": "Run JS code"},
            {
                "type": "function",
                "name": "wait",
                "description": "Wait",
                "parameters": {"type": "object", "properties": {}},
            },
        ],
    }
]

# Mirrors the mapping parse_request computes from _NAMESPACE_TOOLS and pushes
# into the FormatContext via update_format_context(namespace_map=...).
_NAMESPACE_MAP = {
    "functions__exec": ["functions", "exec"],
    "functions__wait": ["functions", "wait"],
}


def _make_transformer(**kwargs) -> OpenResponsesStreamingTransformer:
    from llm_proxy.protocols.openresponses.handler import (
        clear_format_context,
        set_format_context,
        update_format_context,
    )

    clear_format_context()
    set_format_context({"tools": _NAMESPACE_TOOLS})
    # parse_request pushes the namespace mapping after tool conversion.
    update_format_context(namespace_map=_NAMESPACE_MAP)
    transformer = OpenResponsesStreamingTransformer(
        model="deepseek-v4-flash", request_id="test-ns", **kwargs
    )
    return transformer


def _teardown():
    from llm_proxy.protocols.openresponses.handler import clear_format_context

    clear_format_context()


class TestShortNameCustomToolMatching:
    """Model echoes short history name -> still emitted as custom_tool_call."""

    def teardown_method(self):
        _teardown()

    def test_short_name_emits_custom_tool_call(self):
        transformer = _make_transformer()
        chunks = [
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "exec",
                                        "arguments": '{"content":"await tools.x()"}',
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
            events += transformer.transform(chunk) or ""
        parsed = _parse_events(events)

        added = [e for e in parsed if e["type"] == "response.output_item.added"]
        assert len(added) == 1
        assert added[0]["item"]["type"] == "custom_tool_call"
        assert added[0]["item"]["name"] == "exec"

        # Custom tool calls carry unwrapped ``input``, not JSON ``arguments``.
        done = [e for e in parsed if e["type"] == "response.output_item.done"]
        assert len(done) == 1
        assert done[0]["item"]["type"] == "custom_tool_call"
        assert done[0]["item"]["input"] == "await tools.x()"
        assert "arguments" not in done[0]["item"]

        # No function_call_arguments.* events for custom tool calls.
        types = {e["type"] for e in parsed}
        assert "response.function_call_arguments.delta" not in types
        assert "response.function_call_arguments.done" not in types

    def test_flat_name_still_restored_to_short(self):
        """Model echoes the flattened definition name -> restored for the client."""
        transformer = _make_transformer()
        chunks = [
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "functions__exec",
                                        "arguments": '{"content":"1+1"}',
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
            events += transformer.transform(chunk) or ""
        parsed = _parse_events(events)

        added = [e for e in parsed if e["type"] == "response.output_item.added"]
        assert added[0]["item"]["type"] == "custom_tool_call"
        assert added[0]["item"]["name"] == "exec"

    def test_short_name_restored_with_namespace(self):
        """Non-default namespace: short-name echo must be restored with its namespace.

        The client (Codex) looks up the tool by (namespace, name); a function_call
        carrying the bare short name (no namespace) resolves to the default
        "functions" namespace and fails to match tools in other namespaces.
        """
        from llm_proxy.protocols.openresponses.handler import (
            clear_format_context,
            set_format_context,
            update_format_context,
        )

        clear_format_context()
        set_format_context({"tools": _NAMESPACE_TOOLS})
        update_format_context(
            namespace_map={
                "functions__exec": ["functions", "exec"],
                "mcp__github__create_issue": ["mcp__github", "create_issue"],
            }
        )
        transformer = OpenResponsesStreamingTransformer(
            model="deepseek-v4-flash", request_id="test-ns"
        )
        chunks = [
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "create_issue", "arguments": "{}"},
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
            events += transformer.transform(chunk) or ""
        parsed = _parse_events(events)

        added = [e for e in parsed if e["type"] == "response.output_item.added"]
        assert added[0]["item"]["type"] == "function_call"
        assert added[0]["item"]["name"] == "create_issue"
        assert added[0]["item"]["namespace"] == "mcp__github"

        # The terminal snapshot carries the same restored name.
        completed = [e for e in parsed if e["type"] == "response.completed"]
        call = completed[0]["response"]["output"][0]
        assert call["name"] == "create_issue"
        assert call["namespace"] == "mcp__github"

    def test_regular_function_tool_unaffected(self):
        transformer = _make_transformer()
        chunks = [
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "wait", "arguments": "{}"},
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
            events += transformer.transform(chunk) or ""
        parsed = _parse_events(events)

        added = [e for e in parsed if e["type"] == "response.output_item.added"]
        assert added[0]["item"]["type"] == "function_call"
        assert added[0]["item"]["name"] == "wait"


class TestInterleavedToolCallAndText:
    """Text arriving after a tool call must not reference the tool-call item."""

    def teardown_method(self):
        _teardown()

    def _run_interleaved_stream(self) -> list[dict]:
        transformer = _make_transformer()
        chunks = [
            # Tool call header + full args (Ollama-style: one chunk per call)
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "exec",
                                        "arguments": '{"content":"cmd1"}',
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            },
            # Model then emits text (degenerate interleave)
            {"choices": [{"index": 0, "delta": {"content": "c"}, "finish_reason": None}]},
            # ... and another tool call
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 1,
                                    "id": "call_2",
                                    "type": "function",
                                    "function": {
                                        "name": "exec",
                                        "arguments": '{"content":"cmd2"}',
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
            events += transformer.transform(chunk) or ""
        return _parse_events(events)

    def test_text_delta_never_references_tool_call_item(self):
        parsed = self._run_interleaved_stream()
        tool_call_item_ids = {
            e["item"]["id"]
            for e in parsed
            if e["type"] == "response.output_item.added"
            and e["item"]["type"] in ("function_call", "custom_tool_call")
        }
        assert tool_call_item_ids, "expected at least one tool-call item"
        for e in parsed:
            if e["type"] == "response.output_text.delta":
                assert e["item_id"] not in tool_call_item_ids
            if e["type"] in ("response.content_part.added", "response.content_part.done"):
                assert e.get("item_id") not in tool_call_item_ids

    def test_interrupted_tool_call_closed_with_tool_call_events(self):
        parsed = self._run_interleaved_stream()
        types = [e["type"] for e in parsed]

        # The first tool call, interrupted by text, is closed with
        # output_item.done (custom tool calls carry input in the done item).
        tool_call_dones = [
            e
            for e in parsed
            if e["type"] == "response.output_item.done"
            and e["item"]["type"] in ("function_call", "custom_tool_call")
        ]
        assert len(tool_call_dones) == 2
        assert tool_call_dones[0]["item"]["input"] == "cmd1"
        assert tool_call_dones[1]["item"]["input"] == "cmd2"

        # output indices are unique across added items.
        added_indices = [
            e["output_index"] for e in parsed if e["type"] == "response.output_item.added"
        ]
        assert len(added_indices) == len(set(added_indices))

        # Stream terminates with response.completed.
        assert types[-1] == "response.completed"

    def test_text_after_tool_call_opens_message_item(self):
        parsed = self._run_interleaved_stream()
        text_deltas = [e for e in parsed if e["type"] == "response.output_text.delta"]
        assert text_deltas, "expected the interleaved text to be streamed"
        message_item_ids = {
            e["item"]["id"]
            for e in parsed
            if e["type"] == "response.output_item.added" and e["item"]["type"] == "message"
        }
        assert all(e["item_id"] in message_item_ids for e in text_deltas)
