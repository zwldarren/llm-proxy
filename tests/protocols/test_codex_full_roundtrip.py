"""Full roundtrip integration tests for the Codex tool ecosystem.

Verifies that tool_search calls, namespace tools, and function tools all survive
the complete pipeline:

    ResponsesRequest → parse_request → InternalRequest
        → OpenAIRequestBuilder.build → Chat Completions body
        → InternalResponse → format_response → Responses output

Covers:
- ``tool_search`` in the tool list and tool_search_call/tool_search_output
  history items
- ``namespace`` tools (MCP-style) with flattened names and roundtrip
  restoration
- Plain ``function`` tools alongside namespaced tools
- ``web_search`` as a distinct tool type
"""

import orjson

from llm_proxy.models import (
    InternalResponse,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from llm_proxy.models.types import Usage
from llm_proxy.protocols.openresponses import (
    OpenResponsesProtocolSerializer,
)
from llm_proxy.serialization.context import BuildContext
from llm_proxy.serialization.format_context import FormatContext
from llm_proxy.serialization.openai.components.request_builder import OpenAIRequestBuilder

_protocol = OpenResponsesProtocolSerializer()
_builder = OpenAIRequestBuilder()


def _build_ctx(stream: bool = False) -> BuildContext:
    return BuildContext(
        provider_name="openai",
        model="gpt-5",
        stream=stream,
        target_endpoint="chat_completions",
    )


def _make_responses_request(
    *,
    model: str = "gpt-5",
    instructions: str | None = None,
    tools: list[dict] | None = None,
    input: list[dict] | None = None,
    include: list[str] | None = None,
) -> dict:
    """Build a minimal but realistic /v1/responses request dict."""
    req: dict = {
        "model": model,
        "input": input or [{"type": "message", "role": "user", "content": "Hello"}],
        "stream": False,
    }
    if instructions:
        req["instructions"] = instructions
    if tools:
        req["tools"] = tools
    if include:
        req["include"] = include
    return req


# ---------------------------------------------------------------------------
# Helper: extract items of a given type from Responses output array
# ---------------------------------------------------------------------------


def _find_output_of_type(output: list[dict], item_type: str) -> list[dict]:
    return [it for it in output if it.get("type") == item_type]


# ===================================================================
# Roundtrip 1: tool_search tool
# ===================================================================


class TestToolSearchRoundtrip:
    """Verify that tool_search survives a full request→response roundtrip."""

    def test_tool_search_in_tool_list_survives(self):
        """A tool_search in the tool list must survive parse→build→simulate→format."""

        req_data = _make_responses_request(
            tools=[{"type": "tool_search"}],
            input=[
                {"type": "message", "role": "user", "content": "Find me shell tools"},
                {
                    "type": "tool_search_call",
                    "call_id": "call_ts_1",
                    "execution": "sync",
                    "arguments": {"query": "shell"},
                },
                {
                    "type": "tool_search_output",
                    "call_id": "call_ts_1",
                    "status": "completed",
                    "execution": "sync",
                    "tools": [{"name": "exec_command", "description": "Run a command"}],
                },
                {"type": "message", "role": "user", "content": "Now run ls"},
            ],
        )

        # Step 1: parse → InternalRequest
        internal_req = _protocol.parse_request(req_data)
        assert internal_req.tools is not None
        tool_names = [t.name for t in internal_req.tools]
        assert "tool_search" in tool_names, "tool_search must survive parse_request"

        # Verify conversation: user → assistant(tool_search) → tool(result) → user
        roles = [m.role for m in internal_req.conversation.messages]
        assert roles == ["user", "assistant", "tool", "user"]
        use = internal_req.conversation.messages[1].content[0]
        assert isinstance(use, ToolUseBlock)
        assert use.name == "tool_search"
        assert use.input == {"query": "shell"}

        res_block = internal_req.conversation.messages[2].content[0]
        assert isinstance(res_block, ToolResultBlock)
        parsed = orjson.loads(res_block.content)
        assert parsed == [{"name": "exec_command", "description": "Run a command"}]

        # Step 2: build Chat Completions body
        body = _builder.build(internal_req, _build_ctx())
        assert "tools" in body
        built_tool_names = {
            t.get("name")
            or (t.get("function", {}).get("name") if isinstance(t.get("function"), dict) else None)
            or t.get("type")
            for t in body["tools"]
        }
        assert "tool_search" in built_tool_names

        # Step 3: simulate a Chat Completions response with a new tool_search call
        sim_response = InternalResponse(
            id="resp_rt_1",
            model="gpt-5",
            output=[
                ThinkingBlock(thinking="Searching for shell tools..."),
                ToolUseBlock(
                    id="call_ts_2",
                    name="tool_search",
                    input={"query": "shell"},
                ),
                TextBlock(text="Here are the matching tools."),
            ],
            usage=Usage(input_tokens=50, output_tokens=30, total_tokens=80),
            finish_reason="stop",
        )

        # Step 4: format → Responses output
        fmt_ctx = FormatContext()
        result = _protocol.format_response(sim_response, fmt_ctx)

        assert result["status"] == "completed"
        tool_search_calls = _find_output_of_type(result["output"], "tool_search_call")
        assert len(tool_search_calls) == 1
        assert tool_search_calls[0]["call_id"] == "call_ts_2"
        assert tool_search_calls[0]["arguments"] == {"query": "shell"}

        reasoning = _find_output_of_type(result["output"], "reasoning")
        assert len(reasoning) == 1


# ===================================================================
# Roundtrip 2: namespace tools (MCP-style)
# ===================================================================


class TestNamespaceRoundtrip:
    """Verify that namespace tools survive the full roundtrip with name restoration."""

    def test_namespace_tools_survive_roundtrip(self):
        """Namespace tools are flattened on parse and restored on format."""

        req_data = _make_responses_request(
            tools=[
                {
                    "type": "namespace",
                    "name": "mcp__github",
                    "tools": [
                        {
                            "type": "function",
                            "name": "list_issues",
                            "description": "List GitHub issues",
                            "parameters": {"type": "object"},
                        },
                        {
                            "type": "function",
                            "name": "create_pr",
                            "description": "Create a pull request",
                            "parameters": {"type": "object"},
                        },
                    ],
                },
            ],
            input=[
                {"type": "message", "role": "user", "content": "Show me open issues"},
                {
                    "type": "function_call",
                    "call_id": "call_gh_1",
                    "name": "list_issues",
                    "arguments": '{"repo": "my/repo"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_gh_1",
                    "output": "Issue #42: Fix login bug",
                },
            ],
        )

        # Step 1: parse → InternalRequest
        internal_req = _protocol.parse_request(req_data)

        # Tools must be flattened: mcp__github__list_issues, mcp__github__create_pr
        assert internal_req.tools is not None
        flat_names = {t.name for t in internal_req.tools}
        assert "mcp__github__list_issues" in flat_names
        assert "mcp__github__create_pr" in flat_names

        # Verify conversation: user → assistant(tool_call with flat name)
        roles = [m.role for m in internal_req.conversation.messages]
        assert roles == ["user", "assistant", "tool"]
        use = internal_req.conversation.messages[1].content[0]
        assert isinstance(use, ToolUseBlock)
        # Parser preserves original tool names in conversation history.
        # Namespace flattening only applies to the tool *definitions* array;
        # the function_call history items keep the client-supplied names.
        # The namespace_map on FormatContext handles restoration on output.
        assert use.name == "list_issues"
        assert use.input == {"repo": "my/repo"}

        res_block = internal_req.conversation.messages[2].content[0]
        assert isinstance(res_block, ToolResultBlock)
        assert res_block.content == "Issue #42: Fix login bug"

        # Step 2: build Chat Completions body
        body = _builder.build(internal_req, _build_ctx())
        assert "tools" in body
        built_tool_names = {
            t.get("name")
            or (t.get("function", {}).get("name") if isinstance(t.get("function"), dict) else None)
            or t.get("type")
            for t in body["tools"]
        }
        assert "mcp__github__list_issues" in built_tool_names
        assert "mcp__github__create_pr" in built_tool_names

        # Step 3: simulate a Chat response with namespace tool calls
        sim_response = InternalResponse(
            id="resp_ns_1",
            model="gpt-5",
            output=[
                ThinkingBlock(thinking="Let me check the issues..."),
                ToolUseBlock(
                    id="call_ns_2",
                    name="mcp__github__list_issues",
                    input={"repo": "other/repo"},
                ),
                TextBlock(text="Fetched issues from GitHub."),
            ],
            usage=Usage(input_tokens=60, output_tokens=40, total_tokens=100),
            finish_reason="stop",
        )

        # Step 4: format with namespace_map to restore original tool names
        namespace_map = {"mcp__github__list_issues": ["mcp__github", "list_issues"]}
        fmt_ctx = FormatContext(namespace_map=namespace_map)
        result = _protocol.format_response(sim_response, fmt_ctx)

        assert result["status"] == "completed"
        fn_calls = _find_output_of_type(result["output"], "function_call")
        assert len(fn_calls) == 1
        # The tool name should be restored to the original "list_issues"
        assert fn_calls[0]["name"] == "list_issues"
        assert fn_calls[0]["call_id"] == "call_ns_2"

        arguments_parsed = orjson.loads(fn_calls[0]["arguments"])
        assert arguments_parsed == {"repo": "other/repo"}

        reasoning = _find_output_of_type(result["output"], "reasoning")
        assert len(reasoning) == 1

    def test_namespace_tool_without_map_uses_flat_name(self):
        """When no namespace_map is provided, flat names are emitted as-is."""

        sim_response = InternalResponse(
            id="resp_ns_2",
            model="gpt-5",
            output=[
                ToolUseBlock(
                    id="call_ns_3",
                    name="mcp__github__list_issues",
                    input={"repo": "my/repo"},
                ),
            ],
            usage=Usage(input_tokens=10, output_tokens=10, total_tokens=20),
            finish_reason="stop",
        )

        result = _protocol.format_response(sim_response, FormatContext())
        fn_calls = _find_output_of_type(result["output"], "function_call")
        assert len(fn_calls) == 1
        # Without namespace_map, flat name is emitted as-is
        assert fn_calls[0]["name"] == "mcp__github__list_issues"


# ===================================================================
# Roundtrip 3: mixed tool types (tool_search + namespace + function)
# ===================================================================


class TestMixedToolRoundtrip:
    """Verify all tool types coexist and survive the full roundtrip together."""

    def test_mixed_tools_full_roundtrip(self):
        """tool_search + namespace + plain function all survive together."""

        req_data = _make_responses_request(
            tools=[
                {"type": "tool_search"},
                {
                    "type": "namespace",
                    "name": "mcp__filesystem",
                    "tools": [
                        {
                            "type": "function",
                            "name": "read_file",
                            "description": "Read a file",
                            "parameters": {"type": "object"},
                        },
                    ],
                },
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object"},
                },
            ],
            input=[
                {"type": "message", "role": "user", "content": "Search tools then act"},
                # tool_search call + output
                {
                    "type": "tool_search_call",
                    "call_id": "call_ts_mix",
                    "execution": "sync",
                    "arguments": {"query": "file operations"},
                },
                {
                    "type": "tool_search_output",
                    "call_id": "call_ts_mix",
                    "status": "completed",
                    "execution": "sync",
                    "tools": [
                        {"name": "read_file", "description": "Read a file"},
                    ],
                },
                # namespace function call + output
                {
                    "type": "function_call",
                    "call_id": "call_ns_mix",
                    "name": "read_file",
                    "arguments": '{"path": "/tmp/data.json"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_ns_mix",
                    "output": '{"key": "value"}',
                },
                # plain function call + output
                {
                    "type": "function_call",
                    "call_id": "call_fn_mix",
                    "name": "get_weather",
                    "arguments": '{"city": "Tokyo"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_fn_mix",
                    "output": "Sunny, 22C",
                },
                {"type": "message", "role": "user", "content": "Summarize everything"},
            ],
        )

        # Step 1: parse → InternalRequest
        internal_req = _protocol.parse_request(req_data)
        assert internal_req.tools is not None

        # All three tool types should be present
        tool_names = {t.name for t in internal_req.tools}
        assert "tool_search" in tool_names
        assert "mcp__filesystem__read_file" in tool_names
        assert "get_weather" in tool_names

        # Expected conversation structure:
        # user → assistant(tool_search_call) → tool(tool_search_output)
        # → assistant(read_file) → tool(read_file output)
        # → assistant(get_weather) → tool(get_weather output) → user
        # Each function_call_output (or tool_search_output) flushes pending
        # assistant blocks and emits a tool message, so the roles alternate.
        roles = [m.role for m in internal_req.conversation.messages]
        assert roles == [
            "user",
            "assistant",
            "tool",
            "assistant",
            "tool",
            "assistant",
            "tool",
            "user",
        ]

        # Step 2: build Chat Completions body
        body = _builder.build(internal_req, _build_ctx())
        assert "tools" in body
        built_names = {
            t.get("name")
            or (t.get("function", {}).get("name") if isinstance(t.get("function"), dict) else None)
            or t.get("type")
            for t in body["tools"]
        }
        assert "tool_search" in built_names
        assert "mcp__filesystem__read_file" in built_names
        assert "get_weather" in built_names

        # Step 3: simulate a Chat response with all tool types
        sim_response = InternalResponse(
            id="resp_mix_1",
            model="gpt-5",
            output=[
                ThinkingBlock(thinking="Processing the summary..."),
                ToolUseBlock(
                    id="call_ts_out",
                    name="tool_search",
                    input={"query": "weather data"},
                ),
                ToolUseBlock(
                    id="call_ns_out",
                    name="mcp__filesystem__read_file",
                    input={"path": "/tmp/summary.json"},
                ),
                ToolUseBlock(
                    id="call_fn_out",
                    name="get_weather",
                    input={"city": "London"},
                ),
                TextBlock(text="Here is the summary."),
            ],
            usage=Usage(input_tokens=200, output_tokens=100, total_tokens=300),
            finish_reason="stop",
        )

        # Step 4: format with namespace_map
        namespace_map = {"mcp__filesystem__read_file": ["mcp__filesystem", "read_file"]}
        fmt_ctx = FormatContext(namespace_map=namespace_map)
        result = _protocol.format_response(sim_response, fmt_ctx)

        assert result["status"] == "completed"

        # tool_search_call emitted
        ts_calls = _find_output_of_type(result["output"], "tool_search_call")
        assert len(ts_calls) == 1
        assert ts_calls[0]["call_id"] == "call_ts_out"
        assert ts_calls[0]["arguments"] == {"query": "weather data"}

        # namespace function_call — name restored
        fn_calls = _find_output_of_type(result["output"], "function_call")
        assert len(fn_calls) == 2
        fn_names = {c["name"] for c in fn_calls}
        assert "read_file" in fn_names  # restored from namespace
        assert "get_weather" in fn_names  # plain function

        # Verify a text message is also present
        messages = _find_output_of_type(result["output"], "message")
        assert len(messages) >= 1

        # Reasoning should also be present
        reasoning = _find_output_of_type(result["output"], "reasoning")
        assert len(reasoning) == 1


# ===================================================================
# Roundtrip 4: reasoning with encrypted_content
# ===================================================================


class TestReasoningRoundtrip:
    """Verify reasoning + encrypted_content survives the roundtrip."""

    def test_reasoning_with_encrypted_content_roundtrip(self):
        """Reasoning with encrypted_content must survive parse→build→simulate→format."""

        req_data = _make_responses_request(
            include=["reasoning.encrypted_content"],
            input=[
                {"type": "message", "role": "user", "content": "Help me"},
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": "Let me think..."}],
                    "encrypted_content": "OPAQUE_BLOB",
                },
                {
                    "type": "function_call",
                    "call_id": "call_r_1",
                    "name": "search",
                    "arguments": '{"q": "test"}',
                },
                {"type": "function_call_output", "call_id": "call_r_1", "output": "results"},
            ],
        )

        # Step 1: parse
        internal_req = _protocol.parse_request(req_data)
        roles = [m.role for m in internal_req.conversation.messages]
        assert roles == ["user", "assistant", "tool"]

        # Assistant must have thinking block + tool_use block
        thinking = [
            b for b in internal_req.conversation.messages[1].content if isinstance(b, ThinkingBlock)
        ]
        assert len(thinking) == 1
        assert thinking[0].thinking == "Let me think..."
        assert thinking[0].encrypted_content == "OPAQUE_BLOB"

        # Step 2: build
        body = _builder.build(internal_req, _build_ctx())
        # Reasoning content (not encrypted) should be in messages
        msgs = body.get("messages", [])
        assistant_msgs = [m for m in msgs if m.get("role") == "assistant"]
        assert assistant_msgs

        # Step 3: simulate a response with encrypted_content
        sim_response = InternalResponse(
            id="resp_reason_1",
            model="gpt-5",
            output=[
                ThinkingBlock(thinking="Let me analyze...", encrypted_content="NEW_BLOB"),
                ToolUseBlock(id="call_r_2", name="search", input={"q": "more"}),
                TextBlock(text="Final answer."),
            ],
            usage=Usage(input_tokens=30, output_tokens=20, total_tokens=50),
            finish_reason="stop",
        )

        # Step 4: format with include=["reasoning.encrypted_content"]
        fmt_ctx = FormatContext(include=["reasoning.encrypted_content"])
        result = _protocol.format_response(sim_response, fmt_ctx)

        reasoning = _find_output_of_type(result["output"], "reasoning")
        assert len(reasoning) == 1
        assert reasoning[0].get("encrypted_content") == "NEW_BLOB"

        fn_calls = _find_output_of_type(result["output"], "function_call")
        assert len(fn_calls) == 1
        assert fn_calls[0]["call_id"] == "call_r_2"

        messages = _find_output_of_type(result["output"], "message")
        assert len(messages) >= 1


# ===================================================================
# Roundtrip 5: web_search tool
# ===================================================================


class TestWebSearchRoundtrip:
    """Verify web_search tool and web_search_call history survive."""

    def test_web_search_roundtrip(self):
        """web_search tool and web_search_call history must roundtrip."""

        req_data = _make_responses_request(
            tools=[{"type": "web_search"}],
            input=[
                {"type": "message", "role": "user", "content": "What's new in tech?"},
                {
                    "type": "web_search_call",
                    "id": "ws_1",
                    "status": "completed",
                    "action": {"type": "search", "query": "tech news"},
                },
                {"type": "message", "role": "user", "content": "Summarize the results"},
            ],
        )

        # Step 1: parse → InternalRequest
        internal_req = _protocol.parse_request(req_data)

        # web_search_call becomes assistant ToolUseBlock + placeholder ToolResultBlock
        roles = [m.role for m in internal_req.conversation.messages]
        assert roles == ["user", "assistant", "tool", "user"]

        use = internal_req.conversation.messages[1].content[0]
        assert isinstance(use, ToolUseBlock)
        assert use.name == "web_search"
        assert use.input == {"query": "tech news"}

        res_block = internal_req.conversation.messages[2].content[0]
        assert isinstance(res_block, ToolResultBlock)
        assert "Web search performed" in res_block.content

        # Verify tools include web_search
        assert internal_req.tools is not None
        tool_names = {t.name for t in internal_req.tools}
        assert "web_search" in tool_names

        # Step 2: build Chat Completions body
        body = _builder.build(internal_req, _build_ctx())
        assert "tools" in body
        built_names = {
            t.get("name")
            or (t.get("function", {}).get("name") if isinstance(t.get("function"), dict) else None)
            or t.get("type")
            for t in body["tools"]
        }
        assert "web_search" in built_names

        # Step 3: simulate a Chat response with a web_search tool call
        sim_response = InternalResponse(
            id="resp_ws_1",
            model="gpt-5",
            output=[
                ToolUseBlock(
                    id="call_ws_2",
                    name="web_search",
                    input={"query": "latest tech news 2026"},
                ),
                TextBlock(text="Here are the latest tech headlines."),
            ],
            usage=Usage(input_tokens=40, output_tokens=30, total_tokens=70),
            finish_reason="stop",
        )

        # Step 4: format
        result = _protocol.format_response(sim_response, FormatContext())

        # web_search tool call emits as web_search_call item
        ws_calls = _find_output_of_type(result["output"], "web_search_call")
        assert len(ws_calls) == 1
        assert ws_calls[0]["action"]["query"] == "latest tech news 2026"

        messages = _find_output_of_type(result["output"], "message")
        assert len(messages) >= 1

    def test_web_search_function_declaration_emits_function_call(self):
        """A client that declared web_search as a client-executed function tool
        (Hermes Agent pattern) must receive a function_call item — not a
        web_search_call — so it can run the search and return results.

        Regression: Hermes Agent treats web_search_call as "provider already
        executed this", ignores it, and ends the turn (finish_reason=stop)
        without the search ever running.
        """

        req_data = _make_responses_request(
            tools=[
                {
                    "type": "function",
                    "name": "web_search",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                }
            ],
            input=[{"type": "message", "role": "user", "content": "What's new?"}],
        )
        _protocol.parse_request(req_data)

        sim_response = InternalResponse(
            id="resp_ws_2",
            model="gpt-5",
            output=[
                ToolUseBlock(
                    id="call_ws_2",
                    name="web_search",
                    input={"query": "latest tech news 2026"},
                ),
                TextBlock(text="Here are the latest tech headlines."),
            ],
            usage=Usage(input_tokens=40, output_tokens=30, total_tokens=70),
            finish_reason="stop",
        )

        # Production flow: set_format_context runs before parsing, and
        # format_response without an explicit context reads it back.
        from llm_proxy.protocols.openresponses.handler import (
            clear_format_context,
            set_format_context,
        )

        set_format_context(req_data)
        try:
            result = _protocol.format_response(sim_response)
        finally:
            clear_format_context()

        # No web_search_call; a function_call the client can execute instead
        ws_calls = _find_output_of_type(result["output"], "web_search_call")
        assert len(ws_calls) == 0

        func_calls = _find_output_of_type(result["output"], "function_call")
        assert len(func_calls) == 1
        assert func_calls[0]["name"] == "web_search"
        assert func_calls[0]["call_id"] == "call_ws_2"
        assert func_calls[0]["arguments"] == '{"query":"latest tech news 2026"}'


# ===================================================================
# Integration: full end-to-end with streaming context preservation
# ===================================================================


class TestFullEndToEnd:
    """Full end-to-end test covering parse→build→simulate→format with all features."""

    def test_full_codex_style_turn2_roundtrip(self):
        """Simulate a realistic Codex turn-2 payload and verify tool survival."""

        req_data = _make_responses_request(
            model="deepseek-v4-pro",
            instructions="You are a coding assistant.",
            include=["reasoning.encrypted_content"],
            tools=[
                {"type": "tool_search"},
                {
                    "type": "namespace",
                    "name": "mcp__github",
                    "tools": [
                        {
                            "type": "function",
                            "name": "list_issues",
                            "description": "List issues",
                            "parameters": {"type": "object"},
                        },
                    ],
                },
                {
                    "type": "function",
                    "name": "exec_command",
                    "description": "Run a shell command",
                    "parameters": {"type": "object"},
                },
            ],
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": "Find tools and check CI",
                },
                {
                    "type": "tool_search_call",
                    "call_id": "call_ts",
                    "execution": "sync",
                    "arguments": {"query": "CI"},
                },
                {
                    "type": "tool_search_output",
                    "call_id": "call_ts",
                    "status": "completed",
                    "execution": "sync",
                    "tools": [{"name": "exec_command"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_gh",
                    "name": "list_issues",
                    "arguments": '{"repo": "owner/repo"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_gh",
                    "output": "#42 Fix CI, #43 Update docs",
                },
                {
                    "type": "function_call",
                    "call_id": "call_exec",
                    "name": "exec_command",
                    "arguments": '{"cmd": "gh run list --limit 5"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_exec",
                    "output": "1 ci failure",
                },
                {"type": "message", "role": "user", "content": "What's the CI status?"},
            ],
        )

        # Step 1: parse
        internal_req = _protocol.parse_request(req_data)
        assert internal_req.tools is not None
        tool_names = {t.name for t in internal_req.tools}
        assert "tool_search" in tool_names
        assert "mcp__github__list_issues" in tool_names
        assert "exec_command" in tool_names

        # Step 2: build Chat Completions body
        body = _builder.build(internal_req, _build_ctx())
        assert "tools" in body
        built_names = {
            t.get("name")
            or (t.get("function", {}).get("name") if isinstance(t.get("function"), dict) else None)
            or t.get("type")
            for t in body["tools"]
        }
        assert "tool_search" in built_names
        assert "mcp__github__list_issues" in built_names
        assert "exec_command" in built_names

        # messages must have all the conversation turns
        msgs = body["messages"]
        # Should include system (instructions), and conversation turns
        assert len(msgs) >= 3

        # Step 3: simulate final Chat response with all tool types used
        sim_response = InternalResponse(
            id="resp_e2e_1",
            model="deepseek-v4-pro",
            output=[
                ThinkingBlock(thinking="CI is failing.", encrypted_content="BLOB_E2E"),
                ToolUseBlock(
                    id="call_ts_e2e",
                    name="tool_search",
                    input={"query": "CI fix"},
                ),
                ToolUseBlock(
                    id="call_gh_e2e",
                    name="mcp__github__list_issues",
                    input={"repo": "owner/repo", "state": "open"},
                ),
                ToolUseBlock(
                    id="call_exec_e2e",
                    name="exec_command",
                    input={"cmd": "gh run rerun 1"},
                ),
                TextBlock(text="CI run #1 failed. Retrying now."),
            ],
            usage=Usage(input_tokens=200, output_tokens=80, total_tokens=280),
            finish_reason="stop",
        )

        # Step 4: format with namespace_map
        namespace_map = {"mcp__github__list_issues": ["mcp__github", "list_issues"]}
        fmt_ctx = FormatContext(
            namespace_map=namespace_map,
            instructions="You are a coding assistant.",
            include=["reasoning.encrypted_content"],
        )
        result = _protocol.format_response(sim_response, fmt_ctx)

        assert result["status"] == "completed"
        output = result["output"]

        # tool_search_call emitted
        ts_calls = _find_output_of_type(output, "tool_search_call")
        assert len(ts_calls) == 1
        assert ts_calls[0]["call_id"] == "call_ts_e2e"
        assert ts_calls[0]["arguments"] == {"query": "CI fix"}

        # namespace function_call restored
        fn_calls = _find_output_of_type(output, "function_call")
        assert len(fn_calls) == 2
        fn_by_name = {c["name"]: c for c in fn_calls}
        assert "list_issues" in fn_by_name  # restored
        assert fn_by_name["list_issues"]["call_id"] == "call_gh_e2e"
        restored_args = orjson.loads(fn_by_name["list_issues"]["arguments"])
        assert restored_args == {"repo": "owner/repo", "state": "open"}

        assert "exec_command" in fn_by_name  # plain function
        assert fn_by_name["exec_command"]["call_id"] == "call_exec_e2e"
        exec_args = orjson.loads(fn_by_name["exec_command"]["arguments"])
        assert exec_args == {"cmd": "gh run rerun 1"}

        # reasoning with encrypted_content
        reasoning = _find_output_of_type(output, "reasoning")
        assert len(reasoning) == 1
        assert reasoning[0].get("encrypted_content") == "BLOB_E2E"

        # text message
        messages = _find_output_of_type(output, "message")
        assert len(messages) >= 1
        text_content = messages[-1].get("content", [])
        assert any(c.get("text", "") == "CI run #1 failed. Retrying now." for c in text_content)


class TestLocalShellCallOutputRoundtrip:
    """Regression: ``local_shell_call_output`` items must not be silently dropped.

    Codex's native shell tool emits ``local_shell_call`` (the call) and
    ``local_shell_call_output`` (the result) items. The call side was modeled
    and dispatched, but the output side had no schema model and no dispatch
    branch, so it fell into the forward-compatible catch-all and was skipped —
    the model saw its own tool call but never the result. This is the silent
    tool-result-loss symptom.
    """

    def test_local_shell_call_output_survives_parse(self):
        req_data = _make_responses_request(
            model="gpt-5",
            input=[
                {"type": "message", "role": "developer", "content": "dev ctx"},
                {"type": "message", "role": "user", "content": "list files"},
                {
                    "type": "local_shell_call",
                    "call_id": "call_sh",
                    "action": {"type": "exec", "command": ["ls"]},
                },
                {
                    "type": "local_shell_call_output",
                    "call_id": "call_sh",
                    "output": "file_a.txt\nfile_b.txt",
                },
                {"type": "message", "role": "user", "content": "what did you find?"},
            ],
        )

        internal_req = _protocol.parse_request(req_data)

        tool_results = [
            b
            for m in internal_req.conversation.messages
            for b in m.content
            if isinstance(b, ToolResultBlock)
        ]
        assert len(tool_results) == 1, "local_shell_call_output must produce a tool result"
        assert tool_results[0].tool_use_id == "call_sh"
        content = tool_results[0].content
        if isinstance(content, list):
            content = " ".join(getattr(b, "text", "") for b in content)
        assert "file_a.txt" in content
        assert "file_b.txt" in content

        # The assistant tool call must also survive.
        tool_uses = [
            b
            for m in internal_req.conversation.messages
            for b in m.content
            if isinstance(b, ToolUseBlock)
        ]
        assert len(tool_uses) == 1
        assert tool_uses[0].name == "local_shell"

    def test_local_shell_call_output_with_array_content(self):
        """local_shell_call_output.output may be an array of content items."""
        req_data = _make_responses_request(
            model="gpt-5",
            input=[
                {
                    "type": "local_shell_call",
                    "call_id": "call_arr",
                    "action": {"type": "exec", "command": ["echo", "hi"]},
                },
                {
                    "type": "local_shell_call_output",
                    "call_id": "call_arr",
                    "output": [
                        {"type": "input_text", "text": "stdout line"},
                    ],
                },
            ],
        )

        internal_req = _protocol.parse_request(req_data)
        tool_results = [
            b
            for m in internal_req.conversation.messages
            for b in m.content
            if isinstance(b, ToolResultBlock)
        ]
        assert len(tool_results) == 1
        content = tool_results[0].content
        if isinstance(content, list):
            content = " ".join(getattr(b, "text", "") for b in content)
        assert "stdout line" in content
