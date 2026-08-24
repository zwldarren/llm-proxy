"""Regression tests for OpenResponses function_call streaming events.

Verifies that function_call output items receive their full lifecycle events
(output_item.added, function_call_arguments.delta, function_call_arguments.done,
output_item.done) before response.completed, especially when text content and
reasoning content are also present in the same response.
"""

import orjson

from llm_proxy.protocols.openresponses.streaming import OpenResponsesStreamingTransformer


def _parse_events(events: str) -> list[dict]:
    """Parse SSE events from transformer output into a list of event dicts."""
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


class TestOpenResponsesToolCallStreaming:
    """OpenResponses streaming event completeness for function calls."""

    def test_reasoning_text_and_function_call_events(self):
        """Tool call after reasoning and text must receive complete lifecycle events."""
        transformer = OpenResponsesStreamingTransformer(
            model="deepseek-v4-flash", request_id="test-789"
        )

        chunks = [
            # Reasoning content opens item at output_index 0
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": "The core package is already built."},
                        "finish_reason": None,
                    }
                ]
            },
            # Text content opens item at output_index 1
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "Good, the core package is already built."},
                        "finish_reason": None,
                    }
                ]
            },
            # Single provider chunk that carries both text delta and the full tool call.
            # This mirrors DeepSeek behavior where content and tool_calls can arrive
            # together and the first tool_call chunk already contains the full arguments.
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": " Let me run the dev server.",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_ek75ve5e",
                                    "type": "function",
                                    "function": {
                                        "name": "exec_command",
                                        "arguments": '{"cmd":"npx vite"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ]
            },
            # Finish with tool_calls
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        ]

        events = ""
        for chunk in chunks:
            events += transformer.transform(chunk) or ""

        parsed = _parse_events(events)

        # Collect event types for easier assertions
        event_types = [e["type"] for e in parsed]

        # The function_call item must be announced before any of its deltas
        assert "response.output_item.added" in event_types

        # Reasoning is streamed via the summary event family (identical names
        # in the OpenResponses spec and the OpenAI Responses API), never the
        # raw reasoning delta events.
        assert "response.reasoning_summary_part.added" in event_types
        assert "response.reasoning_summary_text.delta" in event_types
        assert "response.reasoning_summary_text.done" in event_types
        assert "response.reasoning_summary_part.done" in event_types
        assert "response.reasoning.delta" not in event_types
        assert "response.reasoning.done" not in event_types
        assert "response.reasoning_text.delta" not in event_types
        assert "response.reasoning_text.done" not in event_types

        # The function_call must be properly closed before response.completed
        assert "response.function_call_arguments.done" in event_types
        assert "response.output_item.done" in event_types
        assert "response.completed" in event_types

        # The close events for the function_call must come before response.completed
        completed_idx = event_types.index("response.completed")
        assert event_types.index("response.function_call_arguments.done") < completed_idx
        assert event_types.index("response.output_item.done") < completed_idx

        # The function_call output_item.done event must reference the function_call item id
        func_call_done_events = [
            e
            for e in parsed
            if e["type"] == "response.output_item.done" and e["item"]["type"] == "function_call"
        ]
        assert len(func_call_done_events) == 1
        func_call_done = func_call_done_events[0]
        assert func_call_done["item"]["status"] == "completed"
        assert func_call_done["item"]["name"] == "exec_command"
        assert func_call_done["item"]["arguments"] == '{"cmd":"npx vite"}'

        # Final response.completed must contain the function_call
        completed_event = next(e for e in parsed if e["type"] == "response.completed")
        output = completed_event["response"]["output"]
        func_calls = [o for o in output if o["type"] == "function_call"]
        assert len(func_calls) == 1
        assert func_calls[0]["name"] == "exec_command"
        assert func_calls[0]["call_id"] == "call_ek75ve5e"

    def test_multiple_tool_calls_split_arguments(self):
        """Multiple tool calls with split arguments must each receive complete events."""
        transformer = OpenResponsesStreamingTransformer(
            model="deepseek-v4-flash", request_id="test-multi"
        )

        chunks = [
            # Text content first
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "I'll run two commands."},
                        "finish_reason": None,
                    }
                ]
            },
            # First tool call with name but no arguments yet
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
                                    "function": {"name": "exec_command", "arguments": ""},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            },
            # First tool call arguments (split)
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [{"index": 0, "function": {"arguments": '{"cmd"'}}]
                        },
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [{"index": 0, "function": {"arguments": ': "ls"}'}}]
                        },
                        "finish_reason": None,
                    }
                ]
            },
            # Second tool call arrives after first is done
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
                                    "function": {"name": "read_file", "arguments": '{"path":"/a"}'},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            },
            # Finish with tool_calls
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        ]

        events = ""
        for chunk in chunks:
            events += transformer.transform(chunk) or ""

        parsed = _parse_events(events)
        event_types = [e["type"] for e in parsed]

        # Both tool calls must be properly closed before response.completed
        assert event_types.count("response.function_call_arguments.done") == 2
        assert event_types.count("response.output_item.done") == 3  # 1 message + 2 function_calls
        assert "response.completed" in event_types

        # Both function_calls must appear in the final response
        completed_event = next(e for e in parsed if e["type"] == "response.completed")
        output = completed_event["response"]["output"]
        func_calls = [o for o in output if o["type"] == "function_call"]
        assert len(func_calls) == 2
        names = {fc["name"] for fc in func_calls}
        assert names == {"exec_command", "read_file"}

    def test_web_search_call_client_side_emitted_as_function_call(self):
        """Client-side web_search (intercept_web_search=False) declared as a
        client function tool must be emitted as a regular ``function_call``
        item so the client can execute the search and return results — not as
        ``web_search_call``.

        Regression: Hermes Agent treats ``web_search_call`` items as
        "provider already executed this search", ignores them, and ends the
        turn (finish_reason=stop) without the search ever running — the task
        dies mid-way.
        """
        from llm_proxy.protocols.openresponses.handler import (
            clear_format_context,
            set_format_context,
        )

        # Hermes Agent declares web_search as a client-executed function tool
        set_format_context(
            {
                "tools": [
                    {
                        "type": "function",
                        "name": "web_search",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ]
            }
        )
        try:
            transformer = OpenResponsesStreamingTransformer(
                model="deepseek-v4-flash",
                request_id="test-ws-client",
                intercept_web_search=False,
            )

            chunks = [
                # Tool call header chunk (name, no arguments yet)
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_ws_1",
                                        "type": "function",
                                        "function": {
                                            "name": "web_search",
                                            "arguments": "",
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ]
                },
                # Arguments chunk (no name/id — still recognized as web_search)
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"arguments": '{"query":"hermes agent"}'},
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
        finally:
            clear_format_context()

        parsed = _parse_events(events)
        event_types = [e["type"] for e in parsed]

        # No web_search_call items at all in client-side function mode
        assert "response.output_item.added" in event_types
        assert not any(
            e.get("item", {}).get("type") == "web_search_call"
            for e in parsed
            if e["type"] in ("response.output_item.added", "response.output_item.done")
        )

        # The web_search tool call must be a function_call with full lifecycle
        fc_added = [
            e
            for e in parsed
            if e["type"] == "response.output_item.added"
            and e.get("item", {}).get("type") == "function_call"
        ]
        assert len(fc_added) == 1
        assert fc_added[0]["item"]["name"] == "web_search"
        assert fc_added[0]["item"]["call_id"] == "call_ws_1"

        fc_done = [
            e
            for e in parsed
            if e["type"] == "response.output_item.done"
            and e.get("item", {}).get("type") == "function_call"
        ]
        assert len(fc_done) == 1
        assert fc_done[0]["item"]["arguments"] == '{"query":"hermes agent"}'

        # added < arguments.delta < arguments.done < output_item.done < completed
        delta_idx = event_types.index("response.function_call_arguments.delta")
        done_idx = event_types.index("response.function_call_arguments.done")
        completed_idx = event_types.index("response.completed")
        assert parsed.index(fc_added[0]) < delta_idx < done_idx < parsed.index(fc_done[0])
        assert parsed.index(fc_done[0]) < completed_idx

        # The function_call must appear in the final response output
        completed_event = next(e for e in parsed if e["type"] == "response.completed")
        output = completed_event["response"]["output"]
        func_calls = [o for o in output if o["type"] == "function_call"]
        assert len(func_calls) == 1
        assert func_calls[0]["name"] == "web_search"
        assert func_calls[0]["call_id"] == "call_ws_1"
        assert func_calls[0]["arguments"] == '{"query":"hermes agent"}'

    def test_web_search_builtin_declaration_stays_web_search_call(self):
        """A client that declared the builtin {"type": "web_search"} tool keeps
        receiving web_search_call items (server-side execution contract)."""
        from llm_proxy.protocols.openresponses.handler import (
            clear_format_context,
            set_format_context,
        )

        set_format_context({"tools": [{"type": "web_search"}]})
        try:
            transformer = OpenResponsesStreamingTransformer(
                model="deepseek-v4-flash",
                request_id="test-ws-builtin",
                intercept_web_search=False,
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
                                        "id": "call_ws_1",
                                        "type": "function",
                                        "function": {
                                            "name": "web_search",
                                            "arguments": '{"query":"hermes agent"}',
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
        finally:
            clear_format_context()

        parsed = _parse_events(events)
        event_types = [e["type"] for e in parsed]

        # No arguments deltas for the web_search_call item
        assert "response.function_call_arguments.delta" not in event_types
        assert "response.function_call_arguments.done" not in event_types

        # web_search_call lifecycle: added before done, both before completed
        added_events = [
            e
            for e in parsed
            if e["type"] == "response.output_item.added"
            and e.get("item", {}).get("type") == "web_search_call"
        ]
        done_events = [
            e
            for e in parsed
            if e["type"] == "response.output_item.done"
            and e.get("item", {}).get("type") == "web_search_call"
        ]
        assert len(added_events) == 1
        assert len(done_events) == 1
        assert parsed.index(added_events[0]) < parsed.index(done_events[0])
        assert parsed.index(done_events[0]) < event_types.index("response.completed")

        # The action must carry the complete accumulated query
        added = added_events[0]["item"]
        expected_action = {
            "type": "search",
            "query": "hermes agent",
            "queries": ["hermes agent"],
        }
        assert added["action"] == expected_action

    def test_web_search_call_intercept_mode_not_emitted(self):
        """Server-side interception (intercept_web_search=True) must not emit
        the web_search tool call to the client at all — the proxy executes it.
        """
        transformer = OpenResponsesStreamingTransformer(
            model="deepseek-v4-flash",
            request_id="test-ws-intercept",
            intercept_web_search=True,
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
                                    "id": "call_ws_1",
                                    "type": "function",
                                    "function": {
                                        "name": "web_search",
                                        "arguments": '{"query":"hermes agent"}',
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
        # No function_call / web_search_call / arguments events for the search
        assert not any(
            e["type"].startswith("response.function_call_arguments")
            or e.get("item", {}).get("type") in ("function_call", "web_search_call")
            for e in parsed
        )
        # Intercepted searches defer completion to the continuation
        assert "response.completed" not in [e["type"] for e in parsed]

    def test_arguments_delta_never_precedes_item_added(self):
        """Invariant: any function_call_arguments.delta must reference an item_id
        that was already announced by an earlier output_item.added."""
        transformer = OpenResponsesStreamingTransformer(
            model="deepseek-v4-flash",
            request_id="test-order",
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
                                    "function": {"name": "get_weather", "arguments": '{"loc'},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": 'ation":"Paris"}'}}
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            },
            # Interleaved web_search call in the same stream
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 1,
                                    "id": "call_ws",
                                    "type": "function",
                                    "function": {
                                        "name": "web_search",
                                        "arguments": '{"query":"q"}',
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

        announced: set[str] = set()
        for event in parsed:
            etype = event["type"]
            if etype == "response.output_item.added":
                announced.add(event["item"]["id"])
            elif etype == "response.function_call_arguments.delta":
                assert event["item_id"] in announced, (
                    f"function_call_arguments.delta for unknown item "
                    f"{event['item_id']} at output_index {event['output_index']}"
                )

        # The regular function_call still gets its delta events
        delta_events = [e for e in parsed if e["type"] == "response.function_call_arguments.delta"]
        assert len(delta_events) == 2
        assert {e["output_index"] for e in delta_events} == {0}
