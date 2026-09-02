"""Regression tests for ``/v1/responses`` ``text.format`` and tool preservation.

Two P1 bugs in the OpenResponses protocol serializer (see
``.checklist/results/sec-3.md``):

* Bug A — ``text.format``:
  1. The real OpenAI Responses API nests Structured Outputs under
     ``text.format`` (``text: {format: {type, name, schema, strict}}``). The
     ``TextFormatParam`` pydantic union flattened this to ``text: {type, ...}``
     and, because ``TextResponseFormat.type`` defaults to ``"text"``, the
     nested ``format`` object was silently mis-parsed as ``type="text"``.
  2. For the flat shape, ``JsonSchemaResponseFormat.schema_`` (alias ``schema``)
     was dumped as the ``schema_`` key by ``model_dump()`` (no alias), so the
     JSON schema was lost and every provider received ``json_schema=None``.

* Bug B — Responses built-in tools (``file_search`` / ``code_interpreter``)
  were silently dropped by ``convert_responses_tools``; ``computer_use`` /
  ``mcp`` / unknown types were dropped with a warning. They must now be
  preserved verbatim in ``InternalRequest.extra["responses_tools"]`` so a
  native Responses provider (OpenAI) can forward them, while non-native
  providers can drop them with their own warning.
"""

import logging

from llm_proxy.models import InternalRequest
from llm_proxy.protocols.openresponses import OpenResponsesProtocolSerializer
from llm_proxy.serialization import get_provider_serializer
from llm_proxy.serialization.context import BuildContext

serializer = OpenResponsesProtocolSerializer()


def _parse(raw: dict) -> InternalRequest:
    return serializer.parse_request(raw)


# ---------------------------------------------------------------------------
# Bug A: text.format parsing
# ---------------------------------------------------------------------------


class TestTextFormatParsing:
    def test_nested_json_schema_shape_is_parsed(self):
        """Real API shape ``text.format.{type,name,schema,strict}`` must parse
        into ``ResponseFormat(type="json_schema")`` with the schema preserved
        (not silently degraded to ``type="text"``)."""
        internal = _parse(
            {
                "model": "m",
                "input": "hi",
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "S",
                        "schema": {"type": "object"},
                        "strict": True,
                    }
                },
            }
        )
        rf = internal.params.response_format
        assert rf is not None
        assert rf.type == "json_schema"
        # json_schema holds the wrapper {name, schema, strict}; the schema must
        # not be lost.
        assert rf.json_schema is not None
        assert rf.json_schema.get("name") == "S"
        assert rf.json_schema.get("schema") == {"type": "object"}
        assert rf.json_schema.get("strict") is True

    def test_nested_json_object_shape_is_parsed(self):
        internal = _parse(
            {"model": "m", "input": "hi", "text": {"format": {"type": "json_object"}}}
        )
        rf = internal.params.response_format
        assert rf is not None
        assert rf.type == "json_object"
        assert rf.json_schema is None

    def test_nested_text_shape_is_parsed(self):
        internal = _parse({"model": "m", "input": "hi", "text": {"format": {"type": "text"}}})
        rf = internal.params.response_format
        assert rf is not None
        assert rf.type == "text"
        assert rf.json_schema is None

    def test_flat_json_schema_shape_keeps_schema(self):
        """Legacy flat shape ``text.{type,name,schema,strict}`` must also keep
        the schema (previously lost via the ``schema_`` alias dump bug)."""
        internal = _parse(
            {
                "model": "m",
                "input": "hi",
                "text": {
                    "type": "json_schema",
                    "name": "S",
                    "schema": {"type": "object"},
                    "strict": True,
                },
            }
        )
        rf = internal.params.response_format
        assert rf is not None
        assert rf.type == "json_schema"
        assert rf.json_schema is not None
        assert rf.json_schema.get("schema") == {"type": "object"}
        assert rf.json_schema.get("name") == "S"

    def test_flat_json_object_shape(self):
        internal = _parse({"model": "m", "input": "hi", "text": {"type": "json_object"}})
        rf = internal.params.response_format
        assert rf is not None
        assert rf.type == "json_object"

    def test_no_text_yields_no_response_format(self):
        internal = _parse({"model": "m", "input": "hi"})
        assert internal.params.response_format is None


# ---------------------------------------------------------------------------
# Bug B: Responses built-in / unknown tool preservation
# ---------------------------------------------------------------------------


class TestResponsesToolPreservation:
    def test_file_search_preserved_in_extra(self):
        """``file_search`` must not be silently dropped; it is preserved verbatim
        in ``extra["responses_tools"]`` (with its ``vector_store_ids``)."""
        internal = _parse(
            {
                "model": "m",
                "input": "hi",
                "tools": [{"type": "file_search", "vector_store_ids": ["vs1"]}],
            }
        )
        # No protocol-agnostic ToolDefinition is produced.
        assert internal.tools is None
        preserved = internal.extra.get("responses_tools")
        assert preserved == [{"type": "file_search", "vector_store_ids": ["vs1"]}]

    def test_code_interpreter_preserved_in_extra(self):
        internal = _parse(
            {
                "model": "m",
                "input": "hi",
                "tools": [{"type": "code_interpreter", "container": {"type": "auto"}}],
            }
        )
        assert internal.tools is None
        preserved = internal.extra.get("responses_tools")
        assert preserved == [{"type": "code_interpreter", "container": {"type": "auto"}}]

    def test_function_tool_still_converted(self):
        """A regular function tool must still become a ``ToolDefinition`` in
        ``internal.tools`` (not be diverted into ``extra``)."""
        internal = _parse(
            {
                "model": "m",
                "input": "hi",
                "tools": [{"type": "function", "name": "get_weather"}],
            }
        )
        assert internal.tools is not None and len(internal.tools) == 1
        assert internal.tools[0].name == "get_weather"
        assert "responses_tools" not in internal.extra

    def test_function_plus_file_search_both_survive(self):
        internal = _parse(
            {
                "model": "m",
                "input": "hi",
                "tools": [
                    {"type": "function", "name": "get_weather"},
                    {"type": "file_search", "vector_store_ids": ["vs1"]},
                ],
            }
        )
        assert internal.tools is not None and len(internal.tools) == 1
        assert internal.tools[0].name == "get_weather"
        assert internal.extra["responses_tools"] == [
            {"type": "file_search", "vector_store_ids": ["vs1"]}
        ]

    def test_computer_use_preserved_with_warning(self, caplog):
        internal = _parse(
            {
                "model": "m",
                "input": "hi",
                "tools": [{"type": "computer_use", "display_width": 1024, "display_height": 768}],
            }
        )
        assert internal.tools is None
        preserved = internal.extra.get("responses_tools")
        assert preserved == [{"type": "computer_use", "display_width": 1024, "display_height": 768}]
        assert any("computer_use" in r.message for r in caplog.records)

    def test_mcp_preserved_with_warning(self, caplog):
        with caplog.at_level(
            logging.WARNING, logger="llm_proxy.protocols.openresponses.tool_converter"
        ):
            internal = _parse(
                {
                    "model": "m",
                    "input": "hi",
                    "tools": [{"type": "mcp", "server_label": "s", "server_url": "https://x"}],
                }
            )
        assert internal.tools is None
        preserved = internal.extra.get("responses_tools")
        assert preserved == [{"type": "mcp", "server_label": "s", "server_url": "https://x"}]
        assert any("mcp" in r.message for r in caplog.records)

    def test_file_search_no_warning(self, caplog):
        """``file_search`` / ``code_interpreter`` are valid Responses built-ins;
        preserving them must not log a warning (only unknown types do)."""
        with caplog.at_level(
            logging.WARNING, logger="llm_proxy.protocols.openresponses.tool_converter"
        ):
            _parse(
                {
                    "model": "m",
                    "input": "hi",
                    "tools": [{"type": "file_search", "vector_store_ids": ["vs1"]}],
                }
            )
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


# ---------------------------------------------------------------------------
# End-to-end: the harness scenario from the task description
# ---------------------------------------------------------------------------


class TestHarnessScenario:
    def test_text_format_and_file_search_together(self):
        internal = serializer.parse_request(
            {
                "model": "m",
                "input": "hi",
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "S",
                        "schema": {"type": "object"},
                        "strict": True,
                    }
                },
                "tools": [{"type": "file_search", "vector_store_ids": ["vs1"]}],
            }
        )
        # text.format parsed correctly, schema not lost
        rf = internal.params.response_format
        assert rf.type == "json_schema"
        assert rf.json_schema is not None
        assert rf.json_schema.get("schema") == {"type": "object"}
        # file_search tool definition retained in internal.extra
        assert internal.extra.get("responses_tools") == [
            {"type": "file_search", "vector_store_ids": ["vs1"]}
        ]

        # The OpenAI provider serializer maps response_format back to
        # text.format (the other agent's territory); just assert the internal
        # state feeds it without the schema being lost.
        prov = get_provider_serializer("openai")
        ctx = BuildContext.from_request(
            internal,
            provider_name="openai",
            target_endpoint="responses",
            unknown_fields_policy="ignore",
            unsupported_block_policy="drop",
            supported_content_blocks=prov.supported_content_blocks,
        )
        body = prov.build_provider_request(internal, ctx)
        # Structured Outputs must round-trip to text.format with the schema.
        assert body.get("text", {}).get("format", {}).get("type") == "json_schema"
        assert body["text"]["format"].get("schema") == {"type": "object"}
        assert body["text"]["format"].get("name") == "S"


# ---------------------------------------------------------------------------
# Consumer behavior: each provider must either consume or drop
# extra["responses_tools"] without leaking it as a pseudo top-level key.
# ---------------------------------------------------------------------------


class TestResponsesToolsConsumerBehavior:
    """Regression tests for responses_tools handling by downstream providers."""

    @staticmethod
    def _build(provider: str, target_endpoint: str | None = None) -> dict:
        internal = serializer.parse_request(
            {
                "model": "m",
                "input": "hi",
                "tools": [
                    {"type": "file_search", "vector_store_ids": ["vs1"]},
                    {
                        "type": "function",
                        "name": "g",
                        "description": "d",
                        "parameters": {"type": "object"},
                    },
                ],
            }
        )
        prov = get_provider_serializer(provider)
        ctx = BuildContext.from_request(
            internal,
            provider_name=provider,
            target_endpoint=target_endpoint
            or ("responses" if provider == "openai" else "chat_completions"),
            unknown_fields_policy="ignore",
            unsupported_block_policy="drop",
            supported_content_blocks=prov.supported_content_blocks,
        )
        return prov.build_provider_request(internal, ctx)

    def test_openai_responses_provider_merges_responses_tools(self):
        """The native OpenAI Responses provider must merge
        extra["responses_tools"] into body["tools"] and never leak the
        internal pseudo-key."""
        body = self._build("openai", target_endpoint="responses")
        assert "responses_tools" not in body
        tools = body.get("tools", [])
        assert len(tools) == 2
        assert any(
            t.get("type") == "file_search" and t.get("vector_store_ids") == ["vs1"] for t in tools
        )
        assert any(t.get("type") == "function" and t.get("name") == "g" for t in tools)

    def test_openai_responses_provider_tools_when_only_builtin_tools(self):
        """When only Responses built-in tools are present, tools must still be
        emitted from responses_tools."""
        internal = serializer.parse_request(
            {
                "model": "m",
                "input": "hi",
                "tools": [{"type": "file_search", "vector_store_ids": ["vs1"]}],
            }
        )
        prov = get_provider_serializer("openai")
        ctx = BuildContext.from_request(
            internal,
            provider_name="openai",
            target_endpoint="responses",
            unknown_fields_policy="ignore",
            unsupported_block_policy="drop",
            supported_content_blocks=prov.supported_content_blocks,
        )
        body = prov.build_provider_request(internal, ctx)
        assert body.get("tools") == [{"type": "file_search", "vector_store_ids": ["vs1"]}]
        assert "responses_tools" not in body

    def test_deepseek_openai_chat_completions_drops_responses_tools_with_warning(self, caplog):
        """Chat-Completions-target providers must drop responses_tools (logged
        at DEBUG) and still convert the regular function tool."""
        with caplog.at_level(
            logging.DEBUG,
            logger="llm_proxy.serialization.openai.components.request_builder",
        ):
            body = self._build("deepseek")
        assert "responses_tools" not in body
        tools = body.get("tools", [])
        assert any(
            t.get("type") == "function" and t.get("function", {}).get("name") == "g" for t in tools
        )
        assert any("responses_tools" in r.message for r in caplog.records)

    def test_gemini_drops_responses_tools_with_warning(self, caplog):
        with caplog.at_level(
            logging.WARNING, logger="llm_proxy.serialization.gemini.request_builder"
        ):
            body = self._build("gemini")
        assert "responses_tools" not in body
        tools = body.get("tools", [])
        # tools is array<Tool>; the function declarations live on the entry.
        decls = tools[0].get("function_declarations", []) if tools else []
        assert any(decl.get("name") == "g" for decl in decls)
        assert any("responses_tools" in r.message for r in caplog.records)

    def test_anthropic_drops_responses_tools_with_warning(self, caplog):
        with caplog.at_level(
            logging.WARNING, logger="llm_proxy.serialization.anthropic.serializer"
        ):
            body = self._build("anthropic")
        assert "responses_tools" not in body
        tools = body.get("tools", [])
        assert any(t.get("name") == "g" for t in tools)
        assert any("responses_tools" in r.message for r in caplog.records)
