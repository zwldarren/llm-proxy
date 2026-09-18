"""Tests for making ``tool_search``-discovered tools callable.

Codex's tool-search flow: the model calls the hosted ``tool_search`` tool, the
upstream returns a ``tool_search_output`` item listing matching tool
definitions, and the model then calls one of them. Providers that bridge
``tool_search`` to a function tool (Ollama, Chat Completions, Anthropic, ...)
only ever see the bridge, so the discovered definitions must be appended to the
request's tool list — otherwise the model can never call them.

A native OpenAI Responses upstream handles ``tool_search`` server-side and must
NOT receive the definitions as regular top-level tools (it would defeat deferred
loading and namespace-typed specs are rejected at the top level).
"""

import asyncio
from types import SimpleNamespace

from llm_proxy.core.processing.stages.base import PipelineState
from llm_proxy.core.processing.stages.tool_search import ToolSearchStage
from llm_proxy.observability.event_context import EventContext
from llm_proxy.protocols.openresponses.handler import get_format_context, set_format_context
from llm_proxy.protocols.openresponses.serializer import OpenResponsesProtocolSerializer
from llm_proxy.serialization.ollama.serializer import OllamaProviderSerializer
from llm_proxy.serialization.responses_toolkit.namespace import restore_tool_name

_DISCOVERED_FUNCTION = {
    "type": "function",
    "name": "list_issues",
    "description": "List GitHub issues",
    "parameters": {"type": "object", "properties": {"repo": {"type": "string"}}},
}

_DISCOVERED_NAMESPACE = {
    "type": "namespace",
    "name": "mcp",
    "tools": [
        {
            "type": "function",
            "name": "list_issues",
            "parameters": {"type": "object"},
        }
    ],
}


def _parse(tools, input_items):
    raw = {"model": "m", "tools": tools, "input": input_items}
    set_format_context(raw)
    return raw, OpenResponsesProtocolSerializer().parse_request(raw)


def _run_stage(request, adapter):
    state = PipelineState(
        raw_data={},
        unified_request=request,
        req=None,
        strategy=None,
        trace_id="t",
        event_context=EventContext(request_id="r", trace_id="t", model="m"),
        selection=SimpleNamespace(provider_name="ollama"),
        adapter=adapter,
    )
    asyncio.run(ToolSearchStage().process(state, None))
    return state


def _ollama_body(request):
    # No explicit context: ``build_provider_request`` builds one via
    # ``BuildContext.from_request``, which carries ``request._namespace_map``
    # (mirrors the adapter's ``_build_chat_context``).
    return OllamaProviderSerializer().build_provider_request(request)


class TestParseDiscoveredTools:
    def test_discovered_tools_kept_out_of_declared_tools(self):
        _, req = _parse(
            [{"type": "tool_search"}],
            [
                {
                    "type": "tool_search_output",
                    "call_id": "ts",
                    "status": "done",
                    "execution": "c",
                    "tools": [_DISCOVERED_FUNCTION],
                },
            ],
        )
        assert [t.name for t in (req.tools or [])] == ["tool_search"]
        assert [t.name for t in (req._discovered_tools or [])] == ["list_issues"]

    def test_declared_duplicate_is_not_treated_as_discovered(self):
        _, req = _parse(
            [_DISCOVERED_FUNCTION],
            [
                {
                    "type": "tool_search_output",
                    "call_id": "ts",
                    "status": "done",
                    "execution": "c",
                    "tools": [_DISCOVERED_FUNCTION],
                },
            ],
        )
        assert [t.name for t in (req.tools or [])] == ["list_issues"]
        assert req._discovered_tools is None

    def test_namespace_mapping_covers_discovered_tools(self):
        _, req = _parse(
            [{"type": "tool_search"}],
            [
                {
                    "type": "tool_search_output",
                    "call_id": "ts",
                    "status": "done",
                    "execution": "c",
                    "tools": [_DISCOVERED_NAMESPACE],
                },
            ],
        )
        assert [t.name for t in (req._discovered_tools or [])] == ["mcp__list_issues"]
        assert req._namespace_map == {"mcp__list_issues": ["mcp", "list_issues"]}
        assert restore_tool_name(req._namespace_map, "mcp__list_issues") == ("list_issues", "mcp")

    def test_format_context_sees_discovered_tools(self):
        _parse(
            [{"type": "tool_search"}],
            [
                {
                    "type": "tool_search_output",
                    "call_id": "ts",
                    "status": "done",
                    "execution": "c",
                    "tools": [_DISCOVERED_NAMESPACE],
                },
            ],
        )
        names = {t.get("name") for t in get_format_context().tools}
        assert "mcp" in names


class TestToolSearchStage:
    def test_injects_for_non_native_provider(self):
        _, req = _parse(
            [{"type": "tool_search"}],
            [
                {
                    "type": "tool_search_output",
                    "call_id": "ts",
                    "status": "done",
                    "execution": "c",
                    "tools": [_DISCOVERED_FUNCTION],
                },
            ],
        )
        _run_stage(req, SimpleNamespace())
        assert [t.name for t in (req.tools or [])] == ["tool_search", "list_issues"]

    def test_skips_for_native_responses_upstream(self):
        _, req = _parse(
            [{"type": "tool_search"}],
            [
                {
                    "type": "tool_search_output",
                    "call_id": "ts",
                    "status": "done",
                    "execution": "c",
                    "tools": [_DISCOVERED_FUNCTION],
                },
            ],
        )
        adapter = SimpleNamespace(is_native_responses_upstream=True)
        _run_stage(req, adapter)
        assert [t.name for t in (req.tools or [])] == ["tool_search"]

    def test_stage_is_idempotent(self):
        _, req = _parse(
            [{"type": "tool_search"}],
            [
                {
                    "type": "tool_search_output",
                    "call_id": "ts",
                    "status": "done",
                    "execution": "c",
                    "tools": [_DISCOVERED_FUNCTION],
                },
            ],
        )
        adapter = SimpleNamespace()
        _run_stage(req, adapter)
        _run_stage(req, adapter)
        assert [t.name for t in (req.tools or [])] == ["tool_search", "list_issues"]

    def test_noop_without_discovered_tools(self):
        _, req = _parse([{"type": "tool_search"}], [])
        _run_stage(req, SimpleNamespace())
        assert [t.name for t in (req.tools or [])] == ["tool_search"]


class TestOllamaEndToEnd:
    def test_discovered_tool_and_history_name_reach_ollama(self):
        _, req = _parse(
            [{"type": "tool_search"}],
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hi"}],
                },
                {
                    "type": "tool_search_output",
                    "call_id": "ts",
                    "status": "done",
                    "execution": "c",
                    "tools": [_DISCOVERED_NAMESPACE],
                },
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "list_issues",
                    "arguments": "{}",
                },
                {"type": "function_call_output", "call_id": "c1", "output": "ok"},
            ],
        )
        _run_stage(req, SimpleNamespace())
        body = _ollama_body(req)
        def_names = {t["function"]["name"] for t in body.get("tools", [])}
        call_names = {
            tc["function"]["name"] for m in body["messages"] for tc in (m.get("tool_calls") or [])
        }
        result_names = {
            m["tool_name"]
            for m in body["messages"]
            if m.get("role") == "tool" and m.get("tool_name")
        }
        assert "mcp__list_issues" in def_names
        assert call_names <= def_names
        assert result_names <= def_names


class TestReplayAttachesDiscoveredTools:
    def test_replay_stored_response_attaches_discovered_tools(self):
        from llm_proxy.models import ConversationContext
        from llm_proxy.protocols.openresponses.replay import replay_stored_response

        stored = {
            "input": [],
            "output": [
                {
                    "type": "tool_search_call",
                    "call_id": "ts",
                    "status": "done",
                    "execution": "c",
                    "arguments": {"query": "gh"},
                },
                {
                    "type": "tool_search_output",
                    "call_id": "ts",
                    "status": "done",
                    "execution": "c",
                    "tools": [_DISCOVERED_FUNCTION],
                },
            ],
        }
        _, req = _parse([{"type": "tool_search"}], [])
        replay_stored_response(stored, ConversationContext(messages=[]), None, req)
        assert [t.name for t in (req._discovered_tools or [])] == ["list_issues"]

    def test_replay_does_not_duplicate_declared_tools(self):
        from llm_proxy.models import ConversationContext
        from llm_proxy.protocols.openresponses.replay import replay_stored_response

        stored = {
            "input": [],
            "output": [
                {
                    "type": "tool_search_output",
                    "call_id": "ts",
                    "status": "done",
                    "execution": "c",
                    "tools": [_DISCOVERED_FUNCTION],
                },
            ],
        }
        _, req = _parse([_DISCOVERED_FUNCTION], [])
        replay_stored_response(stored, ConversationContext(messages=[]), None, req)
        assert req._discovered_tools is None


class TestResponseSideRestoresDiscoveredNames:
    def test_format_response_restores_discovered_namespace(self):
        import orjson

        from llm_proxy.models import InternalResponse, ToolUseBlock
        from llm_proxy.models.types import Usage

        _, req = _parse(
            [{"type": "tool_search"}],
            [
                {
                    "type": "tool_search_output",
                    "call_id": "ts",
                    "status": "done",
                    "execution": "c",
                    "tools": [_DISCOVERED_NAMESPACE],
                },
            ],
        )
        _run_stage(req, SimpleNamespace())

        sim = InternalResponse(
            id="resp_1",
            model="m",
            output=[
                ToolUseBlock(
                    id="call_1",
                    name="mcp__list_issues",
                    input={"repo": "a/b"},
                )
            ],
            usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
            finish_reason="stop",
        )
        result = OpenResponsesProtocolSerializer().format_response(sim, get_format_context())
        calls = [it for it in result["output"] if it.get("type") == "function_call"]
        assert len(calls) == 1
        assert calls[0]["name"] == "list_issues"
        assert calls[0]["namespace"] == "mcp"
        assert orjson.loads(calls[0]["arguments"]) == {"repo": "a/b"}


class TestStoredRoundTripPreservesDiscovery:
    def test_stored_input_keeps_tool_search_items(self):
        from llm_proxy.protocols.openresponses.serializer import conversation_to_input_items

        _, req = _parse(
            [{"type": "tool_search"}],
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "find"}],
                },
                {
                    "type": "tool_search_call",
                    "call_id": "ts",
                    "status": "done",
                    "execution": "c",
                    "arguments": {"query": "gh"},
                },
                {
                    "type": "tool_search_output",
                    "call_id": "ts",
                    "status": "done",
                    "execution": "c",
                    "tools": [_DISCOVERED_FUNCTION],
                },
            ],
        )
        items = conversation_to_input_items(req.conversation)
        types = [it["type"] for it in items]
        assert "tool_search_call" in types
        out = next(it for it in items if it["type"] == "tool_search_output")
        assert out["tools"] == [_DISCOVERED_FUNCTION]

    def test_replay_of_stored_input_rediscovers_tools(self):
        from llm_proxy.protocols.openresponses.replay import replay_stored_response
        from llm_proxy.protocols.openresponses.serializer import conversation_to_input_items

        _, req = _parse(
            [{"type": "tool_search"}],
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "find"}],
                },
                {
                    "type": "tool_search_call",
                    "call_id": "ts",
                    "status": "done",
                    "execution": "c",
                    "arguments": {"query": "gh"},
                },
                {
                    "type": "tool_search_output",
                    "call_id": "ts",
                    "status": "done",
                    "execution": "c",
                    "tools": [_DISCOVERED_FUNCTION],
                },
            ],
        )
        stored = {"input": conversation_to_input_items(req.conversation), "output": []}

        # A continuation request that only declares tool_search, as Codex does.
        _, follow_up = _parse([{"type": "tool_search"}], [])
        replay_stored_response(stored, follow_up.conversation, None, follow_up)
        assert [t.name for t in (follow_up._discovered_tools or [])] == ["list_issues"]
