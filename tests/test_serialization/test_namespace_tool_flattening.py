"""Regression tests for namespace tool-name flattening in provider request bodies.

History tool calls carry the original short name (``exec``) while tool
definitions sent upstream are flattened (``functions__exec``). Models echo the
history name, so provider serializers rewrite history call names to the
flattened form via ``BuildContext.namespace_map``; otherwise the model sees
inconsistent naming and the response-side restore cannot classify the call.
"""

from llm_proxy.models import (
    ConversationContext,
    CustomToolUseBlock,
    Message,
    ToolResultBlock,
    ToolUseBlock,
)
from llm_proxy.serialization.context import BuildContext
from llm_proxy.serialization.ollama.serializer import OllamaProviderSerializer
from llm_proxy.serialization.openai.converter import format_conversation
from llm_proxy.serialization.responses_toolkit.namespace import (
    flatten_history_tool_name,
)

_NAMESPACE_MAP = {
    "functions__exec": ["functions", "exec"],
    "functions__wait": ["functions", "wait"],
}


def _assistant_with_calls(*blocks) -> ConversationContext:
    return ConversationContext(messages=[Message(role="assistant", content=list(blocks))])


class TestFlattenHistoryToolName:
    """Unit tests for the reverse-lookup helper."""

    def test_short_name_flattened(self):
        assert flatten_history_tool_name(_NAMESPACE_MAP, "exec") == "functions__exec"

    def test_already_flat_name_unchanged(self):
        assert flatten_history_tool_name(_NAMESPACE_MAP, "functions__exec") == "functions__exec"

    def test_unknown_name_unchanged(self):
        assert flatten_history_tool_name(_NAMESPACE_MAP, "other_tool") == "other_tool"

    def test_ambiguous_short_name_unchanged(self):
        ns_map = {
            "a__exec": ["a", "exec"],
            "b__exec": ["b", "exec"],
        }
        assert flatten_history_tool_name(ns_map, "exec") == "exec"

    def test_none_map_unchanged(self):
        assert flatten_history_tool_name(None, "exec") == "exec"

    def test_empty_name_unchanged(self):
        assert flatten_history_tool_name(_NAMESPACE_MAP, "") == ""


class TestOllamaNamespaceFlattening:
    def test_history_tool_call_names_flattened(self):
        conv = _assistant_with_calls(
            ToolUseBlock(id="call_1", name="exec", input={"content": "ls"}),
            CustomToolUseBlock(id="call_2", name="exec", input="ls -la"),
        )
        ctx = BuildContext(namespace_map=_NAMESPACE_MAP)
        result = OllamaProviderSerializer()._convert_conversation_to_ollama(conv, ctx)
        names = [tc["function"]["name"] for tc in result[0]["tool_calls"]]
        assert names == ["functions__exec", "functions__exec"]

    def test_tool_result_name_matches_flattened_call(self):
        conv = ConversationContext(
            messages=[
                Message(
                    role="assistant",
                    content=[ToolUseBlock(id="call_1", name="exec", input={"content": "ls"})],
                ),
                Message(
                    role="tool",
                    content=[ToolResultBlock(tool_use_id="call_1", content="ok")],
                ),
            ]
        )
        ctx = BuildContext(namespace_map=_NAMESPACE_MAP)
        result = OllamaProviderSerializer()._convert_conversation_to_ollama(conv, ctx)
        tool_msg = next(m for m in result if m["role"] == "tool")
        assert tool_msg["tool_name"] == "functions__exec"

    def test_no_namespace_map_keeps_original_names(self):
        conv = _assistant_with_calls(ToolUseBlock(id="call_1", name="exec", input={}))
        result = OllamaProviderSerializer()._convert_conversation_to_ollama(conv)
        assert result[0]["tool_calls"][0]["function"]["name"] == "exec"


class TestOpenAINamespaceFlattening:
    def test_chat_completions_target_flattens(self):
        conv = _assistant_with_calls(ToolUseBlock(id="call_1", name="exec", input={"a": 1}))
        ctx = BuildContext(namespace_map=_NAMESPACE_MAP, target_endpoint="chat_completions")
        result = format_conversation(conv, ctx)
        assert result[0]["tool_calls"][0]["function"]["name"] == "functions__exec"

    def test_responses_target_keeps_original(self):
        """Native Responses upstreams receive un-flattened tool definitions."""
        conv = _assistant_with_calls(ToolUseBlock(id="call_1", name="exec", input={"a": 1}))
        ctx = BuildContext(namespace_map=_NAMESPACE_MAP, target_endpoint="responses")
        result = format_conversation(conv, ctx)
        assert result[0]["tool_calls"][0]["function"]["name"] == "exec"

    def test_custom_tool_chat_target_flattens(self):
        conv = _assistant_with_calls(CustomToolUseBlock(id="call_1", name="exec", input="ls"))
        ctx = BuildContext(namespace_map=_NAMESPACE_MAP, target_endpoint="chat_completions")
        result = format_conversation(conv, ctx)
        tc = result[0]["tool_calls"][0]
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "functions__exec"

    def test_custom_tool_responses_target_keeps_original(self):
        conv = _assistant_with_calls(CustomToolUseBlock(id="call_1", name="exec", input="ls"))
        ctx = BuildContext(namespace_map=_NAMESPACE_MAP, target_endpoint="responses")
        result = format_conversation(conv, ctx)
        tc = result[0]["tool_calls"][0]
        assert tc["type"] == "custom"
        assert tc["custom"]["name"] == "exec"


class TestNamespaceMapPropagation:
    """BuildContext.from_request carries request._namespace_map."""

    def test_from_request_populates_namespace_map(self):
        from llm_proxy.models.internal import InternalRequest

        request = InternalRequest(model="m", conversation=ConversationContext())
        request._namespace_map = _NAMESPACE_MAP
        ctx = BuildContext.from_request(request)
        assert ctx.namespace_map == _NAMESPACE_MAP

    def test_from_request_defaults_to_none(self):
        from llm_proxy.models.internal import InternalRequest

        request = InternalRequest(model="m", conversation=ConversationContext())
        ctx = BuildContext.from_request(request)
        assert ctx.namespace_map is None


class TestEndToEndParseAndBuild:
    """Codex-style additional_tools request -> flattened ollama body."""

    def test_ollama_body_names_consistent(self):
        from llm_proxy.protocols.openresponses.serializer import (
            OpenResponsesProtocolSerializer,
        )
        from llm_proxy.serialization.ollama.serializer import OllamaProviderSerializer

        raw = {
            "model": "m",
            "input": [
                {
                    "role": "user",
                    "type": "message",
                    "content": [{"type": "input_text", "text": "hi"}],
                },
                {
                    "role": "developer",
                    "type": "additional_tools",
                    "tools": [
                        {
                            "type": "namespace",
                            "name": "functions",
                            "tools": [
                                {"type": "custom", "name": "exec", "description": "run js"},
                            ],
                        }
                    ],
                },
                {
                    "type": "custom_tool_call",
                    "name": "exec",
                    "namespace": "functions",
                    "call_id": "call_1",
                    "input": "await tools.exec_command({cmd: 'ls'})",
                },
                {
                    "type": "custom_tool_call_output",
                    "call_id": "call_1",
                    "output": "done",
                },
            ],
        }
        unified = OpenResponsesProtocolSerializer().parse_request(raw)
        body = OllamaProviderSerializer().build_provider_request(unified)

        def_names = {t["function"]["name"] for t in body.get("tools", [])}
        call_names = {
            tc["function"]["name"] for m in body["messages"] for tc in (m.get("tool_calls") or [])
        }
        result_names = {
            m["tool_name"]
            for m in body["messages"]
            if m.get("role") == "tool" and m.get("tool_name")
        }
        assert "functions__exec" in def_names
        assert call_names <= def_names
        assert result_names <= def_names
