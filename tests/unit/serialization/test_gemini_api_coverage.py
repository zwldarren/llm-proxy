"""Coverage for the Gemini generateContent shapes the proxy emits and parses.

Every case here is transcribed from Google's own documentation for the Gemini
family it applies to (see ``serialization/gemini/model_family.py``):

- 2.5: ``thinkingConfig.thinkingBudget`` + ``responseSchema`` / ``parameters``.
- 3.x: ``thinkingLevel`` (+ ``includeThoughts``), plain JSON Schema through
  ``generationConfig.responseFormat`` / ``FunctionDeclaration.parametersJsonSchema``,
  ``toolConfig.includeServerSideToolInvocations`` for built-in + function
  combinations, and ``toolCall`` / ``toolResponse`` context parts.
- 4.x: ``thinkingLevel`` too, and sampling parameters are ignored.
"""

import logging

import pytest

from llm_proxy.models import (
    ConversationContext,
    GenerationParams,
    InternalRequest,
    Message,
    TextBlock,
    ThinkingConfig,
    ToolResultBlock,
    ToolUseBlock,
)
from llm_proxy.models.finish_reasons import map_finish_reason
from llm_proxy.models.tools import CodeExecutionTool, FunctionTool, HostedTool
from llm_proxy.models.types import ResponseFormat
from llm_proxy.serialization.context import BuildContext
from llm_proxy.serialization.gemini.model_family import (
    _UNPARSABLE_WARNED,
    GeminiFamilyVersion,
    gemini_model_version,
    ignores_sampling_params,
    normalize_family_overrides,
    supports_json_schema_fields,
    supports_minimal_thinking_level,
    supports_thinking_config,
    supports_thought_signatures,
    uses_thinking_budget,
    uses_thinking_level,
)
from llm_proxy.serialization.gemini.streaming_converter import GeminiStreamingTransformer
from llm_proxy.serialization.providers import get_provider_serializer
from llm_proxy.serialization.usage import parse_usage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WRAPPED_SCHEMA = {
    "name": "person",
    "description": "A person",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "role": {"oneOf": [{"const": "admin"}, {"const": "user"}]},
            "manager": {"$ref": "#/$defs/person"},
        },
        "required": ["name"],
        "$defs": {"person": {"type": "object"}},
        "additionalProperties": False,
    },
}


def _request(model: str, *, thinking: ThinkingConfig | None = None, **params):
    return InternalRequest(
        model=model,
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="hi")])]
        ),
        params=GenerationParams(thinking=thinking, **params),
    )


@pytest.fixture
def serializer():
    return get_provider_serializer("gemini")


# ---------------------------------------------------------------------------
# Model family gates
# ---------------------------------------------------------------------------


class TestModelFamily:
    def test_version_parsing(self):
        assert gemini_model_version("gemini-2.5-flash") is not None
        assert gemini_model_version("gemini-2.5-flash") == gemini_model_version("gemini-2.5-pro")
        assert gemini_model_version("gemini-4-argon") == gemini_model_version("gemini-4.0-argon")
        assert gemini_model_version("gpt-4o") is None
        assert gemini_model_version(None) is None

    def test_thinking_gates(self):
        assert supports_thinking_config("gemini-2.5-flash")
        assert supports_thinking_config("gemini-3.8-flash")
        assert supports_thinking_config("gemini-4-argon")
        assert not supports_thinking_config("gemini-2.0-flash")
        assert uses_thinking_level("gemini-3.8-flash")
        assert uses_thinking_level("gemini-4-argon")
        assert not uses_thinking_level("gemini-2.5-flash")
        assert uses_thinking_budget("gemini-2.5-flash")
        assert not uses_thinking_budget("gemini-3.8-flash")

    def test_thought_signature_gate_covers_gemini_4(self):
        """Gemini 4 enforces signatures too — it used to be skipped."""
        assert supports_thought_signatures("gemini-4-argon")
        assert supports_thought_signatures("gemini-3.8-flash")
        assert supports_thought_signatures("gemini-2.5-flash")
        assert not supports_thought_signatures("gemini-2.0-flash")

    def test_json_schema_gate(self):
        assert supports_json_schema_fields("gemini-3.8-flash")
        assert supports_json_schema_fields("gemini-4-argon")
        assert not supports_json_schema_fields("gemini-2.5-flash")

    def test_minimal_level_gate(self):
        assert supports_minimal_thinking_level("gemini-3.6-flash")
        assert supports_minimal_thinking_level("gemini-3-flash")
        # The thinking table takes minimal as the *default* on the Flash-Lite
        # Image model — it is low/medium that model rejects.
        assert supports_minimal_thinking_level("gemini-3.1-flash-lite-image")
        assert supports_minimal_thinking_level("gemini-3.1-flash-image")
        assert not supports_minimal_thinking_level("gemini-3.8-flash")
        assert not supports_minimal_thinking_level("gemini-3-pro-image")
        assert not supports_minimal_thinking_level("gemini-nano-banana-2")
        assert not supports_minimal_thinking_level("gemini-4-argon")

    def test_sampling_gate(self):
        assert ignores_sampling_params("gemini-4-argon")
        assert ignores_sampling_params("gemini-3.6-flash")
        assert ignores_sampling_params("gemini-3.8-flash")
        # The release notes start the deprecation at 3.6 Flash AND 3.5
        # Flash-Lite; 3.5 Flash proper and 3.1 Flash-Lite still accept them.
        assert ignores_sampling_params("gemini-3.5-flash-lite")
        assert not ignores_sampling_params("gemini-3.5-flash")
        assert not ignores_sampling_params("gemini-3.1-flash-lite")
        assert not ignores_sampling_params("gemini-3-flash")
        assert not ignores_sampling_params("gemini-2.5-flash")

    def test_family_override_pins_an_unparseable_id(self):
        overrides = {"gemini-flash-latest": "4.0"}
        assert gemini_model_version("gemini-flash-latest", overrides) == GeminiFamilyVersion(4, 0)
        assert uses_thinking_level("gemini-flash-latest", overrides)
        assert supports_json_schema_fields("gemini-flash-latest", overrides)
        assert supports_thought_signatures("gemini-flash-latest", overrides)
        assert ignores_sampling_params("gemini-flash-latest", overrides)

    def test_family_override_wins_over_the_regex(self):
        overrides = {"gemini-2.5-flash": "3.0"}
        assert gemini_model_version("gemini-2.5-flash", overrides) == GeminiFamilyVersion(3, 0)
        # Other ids are untouched by the pin.
        assert gemini_model_version("gemini-2.5-flash") == GeminiFamilyVersion(2, 5)

    def test_family_override_normalization_drops_invalid_entries(self, caplog):
        overrides = normalize_family_overrides(
            {"gemini-flash-latest": "4.0", "typo": "latest", "3": "3.8"}
        )
        assert overrides == {"gemini-flash-latest": "4.0", "3": "3.8"}
        assert any("typo" in record.getMessage() for record in caplog.records)
        assert normalize_family_overrides(None) == {}
        assert normalize_family_overrides(["not-a-mapping"]) == {}

    def test_unparsable_id_falls_back_legacy_and_warns_once(self, caplog):
        _UNPARSABLE_WARNED.clear()
        try:
            with caplog.at_level(
                logging.WARNING, logger="llm_proxy.serialization.gemini.model_family"
            ):
                assert gemini_model_version("gemini-flash-latest") is None
                # The legacy-safe answers: no thinkingLevel, no schema fields.
                assert not uses_thinking_level("gemini-flash-latest")
                assert not supports_json_schema_fields("gemini-flash-latest")
                gemini_model_version("gemini-flash-latest")
        finally:
            _UNPARSABLE_WARNED.clear()
        warnings = [r for r in caplog.records if "gemini-flash-latest" in r.getMessage()]
        assert len(warnings) == 1


# ---------------------------------------------------------------------------
# Structured output: responseFormat vs the deprecated responseSchema
# ---------------------------------------------------------------------------


class TestStructuredOutput:
    def test_gemini_3_uses_response_format_with_full_json_schema(self, serializer):
        request = _request(
            "gemini-3.8-flash",
            response_format=ResponseFormat(type="json_schema", json_schema=_WRAPPED_SCHEMA),
        )
        config = serializer.build_provider_request(request)["generationConfig"]

        assert "responseSchema" not in config
        assert "responseMimeType" not in config
        text_format = config["responseFormat"]["text"]
        assert text_format["mimeType"] == "application/json"
        # Full JSON Schema survives: no sanitizing on the modern shape.
        assert text_format["schema"]["properties"]["role"]["oneOf"]
        assert text_format["schema"]["properties"]["manager"]["$ref"] == "#/$defs/person"
        assert text_format["schema"]["additionalProperties"] is False

    def test_legacy_family_keeps_sanitized_response_schema(self, serializer):
        request = _request(
            "gemini-2.5-flash",
            response_format=ResponseFormat(type="json_schema", json_schema=_WRAPPED_SCHEMA),
        )
        config = serializer.build_provider_request(request)["generationConfig"]

        assert "responseFormat" not in config
        assert config["responseMimeType"] == "application/json"
        schema = config["responseSchema"]
        # The OpenAPI subset drops what it cannot express (``anyOf`` is a
        # documented Schema field and survives; ``oneOf`` / ``$ref`` /
        # ``additionalProperties`` are not).
        assert "oneOf" not in schema["properties"]["role"]
        assert "$ref" not in schema["properties"]["manager"]
        assert "additionalProperties" not in schema
        assert "$defs" not in schema

    def test_json_object_keeps_mime_type_only(self, serializer):
        request = _request(
            "gemini-3.8-flash",
            response_format=ResponseFormat(type="json_object"),
        )
        config = serializer.build_provider_request(request)["generationConfig"]
        assert config["responseFormat"] == {"text": {"mimeType": "application/json"}}

    def test_function_declaration_uses_parameters_json_schema(self, serializer):
        tool = FunctionTool(
            name="lookup",
            description="look something up",
            parameters={
                "type": "object",
                "properties": {"q": {"type": "string"}, "mode": {"anyOf": [{"const": "a"}]}},
                "required": ["q"],
                "additionalProperties": False,
            },
        )
        request = _request("gemini-3.8-flash")
        request.tools = [tool]
        declaration = serializer.build_provider_request(request)["tools"][0][
            "function_declarations"
        ][0]

        # ``parameters`` and ``parametersJsonSchema`` are mutually exclusive.
        assert "parameters" not in declaration
        assert declaration["parametersJsonSchema"]["additionalProperties"] is False
        assert declaration["parametersJsonSchema"]["properties"]["mode"]["anyOf"]

    def test_legacy_family_function_declaration_stays_sanitized(self, serializer):
        tool = FunctionTool(
            name="lookup",
            parameters={"type": "object", "properties": {"q": {"type": "string"}}},
        )
        request = _request("gemini-2.5-flash")
        request.tools = [tool]
        declaration = serializer.build_provider_request(request)["tools"][0][
            "function_declarations"
        ][0]
        assert "parametersJsonSchema" not in declaration
        assert declaration["parameters"]["properties"]["q"]["type"] == "string"


# ---------------------------------------------------------------------------
# Built-in tools and tool-context circulation
# ---------------------------------------------------------------------------


def _gemini_tooling_request(model: str, tools: list, *, tool_choice=None):
    request = _request(model)
    request.tools = tools
    request.tool_choice = tool_choice
    return request


class TestBuiltinToolCirculation:
    def test_flag_set_for_builtin_plus_function_on_gemini_3(self, serializer):
        tools = [
            HostedTool(name="google_search", type="google_search"),
            FunctionTool(name="get_weather", parameters={"type": "object", "properties": {}}),
        ]
        body = serializer.build_provider_request(_gemini_tooling_request("gemini-3.8-flash", tools))

        assert body["toolConfig"]["includeServerSideToolInvocations"] is True
        # AUTO is documented as unsupported with the flag, and VALIDATED is the
        # default there — so no functionCallingConfig is named.
        assert "functionCallingConfig" not in body["toolConfig"]
        # Both live on the same Tool entry, which the docs show as valid.
        entry = body["tools"][0]
        assert entry["google_search"] == {}
        assert entry["function_declarations"][0]["name"] == "get_weather"

    def test_explicit_tool_choice_still_emitted_with_flag(self, serializer):
        tools = [HostedTool(name="google_search", type="google_search")]
        body = serializer.build_provider_request(
            _gemini_tooling_request("gemini-3.8-flash", tools, tool_choice="required")
        )
        assert body["toolConfig"]["functionCallingConfig"]["mode"] == "ANY"
        assert body["toolConfig"]["includeServerSideToolInvocations"] is True

    def test_no_flag_without_builtin_tools(self, serializer):
        tools = [FunctionTool(name="f", parameters={"type": "object", "properties": {}})]
        body = serializer.build_provider_request(_gemini_tooling_request("gemini-3.8-flash", tools))
        assert "toolConfig" not in body

    def test_no_flag_on_legacy_family(self, serializer):
        tools = [HostedTool(name="google_search", type="google_search")]
        body = serializer.build_provider_request(_gemini_tooling_request("gemini-2.5-flash", tools))
        assert "toolConfig" not in body

    def test_code_execution_tool_maps_to_bare_key(self, serializer):
        request = _gemini_tooling_request("gemini-3.8-flash", [CodeExecutionTool()])
        body = serializer.build_provider_request(request)
        assert body["tools"] == [{"code_execution": {}}]

    def test_computer_use_carries_the_required_environment(self, serializer):
        # The API reference marks ComputerUse.environment required and every
        # documented example sets it; the proxy supplies the browser default.
        bare = HostedTool(name="computer_use", type="computer_use")
        body = serializer.build_provider_request(
            _gemini_tooling_request("gemini-3.8-flash", [bare])
        )
        assert body["tools"] == [{"computer_use": {"environment": "ENVIRONMENT_BROWSER"}}]

        named = HostedTool(
            name="computer_use", type="computer_use", params={"environment": "ENVIRONMENT_DESKTOP"}
        )
        body = serializer.build_provider_request(
            _gemini_tooling_request("gemini-3.8-flash", [named])
        )
        assert body["tools"] == [{"computer_use": {"environment": "ENVIRONMENT_DESKTOP"}}]

    def test_file_search_needs_store_names(self, serializer):
        bare = HostedTool(name="file_search", type="file_search")
        body = serializer.build_provider_request(
            _gemini_tooling_request("gemini-3.8-flash", [bare])
        )
        # A bare {"file_search": {}} is rejected by the API, so it is dropped.
        assert "tools" not in body

        configured = HostedTool(
            name="file_search",
            type="file_search",
            params={"file_search_store_names": ["stores/abc"]},
        )
        body = serializer.build_provider_request(
            _gemini_tooling_request("gemini-3.8-flash", [configured])
        )
        assert body["tools"] == [{"file_search": {"fileSearchStoreNames": ["stores/abc"]}}]


# ---------------------------------------------------------------------------
# Sampling parameters on the families that ignore them
# ---------------------------------------------------------------------------


class TestSamplingParams:
    def test_gemini_4_drops_sampling_params(self, serializer):
        request = _request("gemini-4-argon", temperature=0.2, top_p=0.9)
        config = serializer.build_provider_request(request).get("generationConfig") or {}
        assert "temperature" not in config
        assert "topP" not in config

    def test_legacy_family_keeps_sampling_params(self, serializer):
        request = _request("gemini-2.5-flash", temperature=0.2, top_p=0.9)
        config = serializer.build_provider_request(request)["generationConfig"]
        assert config["temperature"] == 0.2
        assert config["topP"] == 0.9


# ---------------------------------------------------------------------------
# Provider-pinned model families (metadata.model_family_overrides)
# ---------------------------------------------------------------------------


class TestFamilyOverrides:
    """The operator pin turns "new model, unknown id" into a config change.

    ``gemini-flash-latest`` is the shape of the problem: a real Google alias
    whose id carries no version, so every gate falls back legacy-safe. Pinning
    it to a family in the provider metadata must produce exactly the body the
    pinned family's documentation describes.
    """

    def test_unparseable_id_gets_the_legacy_safe_body(self, serializer):
        request = _request(
            "gemini-flash-latest", temperature=0.2, thinking=ThinkingConfig(type="disabled")
        )
        config = serializer.build_provider_request(request)["generationConfig"]
        assert "thinkingConfig" not in config
        assert config["temperature"] == 0.2

    def test_override_pins_the_unparseable_id_to_a_modern_family(self, serializer):
        request = _request(
            "gemini-flash-latest", temperature=0.2, thinking=ThinkingConfig(type="disabled")
        )
        context = BuildContext.from_request(
            request, model_family_overrides={"gemini-flash-latest": "4.0"}
        )
        config = serializer.build_provider_request(request, context)["generationConfig"]
        # Gemini 4 shapes: thinkingLevel floor (minimal is an error there), no
        # sampling parameters.
        assert config["thinkingConfig"] == {"thinkingLevel": "LOW", "includeThoughts": True}
        assert "temperature" not in config


# ---------------------------------------------------------------------------
# Conversation replay
# ---------------------------------------------------------------------------


class TestConversationReplay:
    def test_consecutive_same_role_messages_are_merged(self, serializer):
        request = InternalRequest(
            model="gemini-3.8-flash",
            conversation=ConversationContext(
                messages=[
                    Message(role="user", content=[TextBlock(text="first")]),
                    Message(role="user", content=[TextBlock(text="second")]),
                ]
            ),
            params=GenerationParams(),
        )
        contents = serializer.build_provider_request(request)["contents"]
        assert len(contents) == 1
        assert [part["text"] for part in contents[0]["parts"]] == ["first", "second"]

    def test_function_call_and_response_carry_the_same_id(self, serializer):
        request = InternalRequest(
            model="gemini-3.8-flash",
            conversation=ConversationContext(
                messages=[
                    Message(role="user", content=[TextBlock(text="weather?")]),
                    Message(
                        role="assistant",
                        content=[ToolUseBlock(id="fc_1", name="get_weather", input={"c": "Paris"})],
                    ),
                    Message(
                        role="tool",
                        content=[ToolResultBlock(tool_use_id="fc_1", content="sunny")],
                    ),
                ]
            ),
            params=GenerationParams(),
        )
        contents = serializer.build_provider_request(request)["contents"]

        call = contents[1]["parts"][0]["functionCall"]
        response = contents[2]["parts"][0]["functionResponse"]
        assert call["id"] == "fc_1"
        assert response["id"] == "fc_1"

    def test_dummy_signature_only_on_first_function_call_of_a_step(self, serializer):
        """Google puts the signature on the first parallel call only."""
        request = InternalRequest(
            model="gemini-3.8-flash",
            conversation=ConversationContext(
                messages=[
                    Message(role="user", content=[TextBlock(text="both?")]),
                    Message(
                        role="assistant",
                        content=[
                            ToolUseBlock(id="fc_1", name="a", input={}),
                            ToolUseBlock(id="fc_2", name="b", input={}),
                        ],
                    ),
                    Message(
                        role="tool",
                        content=[
                            ToolResultBlock(tool_use_id="fc_1", content="1"),
                            ToolResultBlock(tool_use_id="fc_2", content="2"),
                        ],
                    ),
                ]
            ),
            params=GenerationParams(),
        )
        contents = serializer.build_provider_request(request)["contents"]
        model_parts = contents[1]["parts"]
        assert model_parts[0]["thoughtSignature"]
        assert "thoughtSignature" not in model_parts[1]

    def test_tool_context_parts_are_replayed_before_the_call(self, serializer):
        """``toolCall`` / ``toolResponse`` parts ride the block's extra cache."""
        tool_call_part = {
            "toolCall": {"id": "tc_1", "toolType": "GOOGLE_SEARCH_WEB", "args": {"queries": ["q"]}},
            "thoughtSignature": "sig_tc",
        }
        request = InternalRequest(
            model="gemini-3.8-flash",
            conversation=ConversationContext(
                messages=[
                    Message(role="user", content=[TextBlock(text="weather?")]),
                    Message(
                        role="assistant",
                        content=[
                            ToolUseBlock(
                                id="fc_1",
                                name="get_weather",
                                input={},
                                extra={"gemini_tool_parts": [tool_call_part]},
                            )
                        ],
                    ),
                    Message(
                        role="tool",
                        content=[ToolResultBlock(tool_use_id="fc_1", content="sunny")],
                    ),
                ]
            ),
            params=GenerationParams(),
        )
        model_parts = serializer.build_provider_request(request)["contents"][1]["parts"]
        assert model_parts[0] == tool_call_part
        assert "functionCall" in model_parts[1]

    def test_text_part_signature_is_replayed(self, serializer):
        request = InternalRequest(
            model="gemini-3.8-flash",
            conversation=ConversationContext(
                messages=[
                    Message(role="user", content=[TextBlock(text="hi")]),
                    Message(
                        role="assistant",
                        content=[
                            TextBlock(
                                text="answer",
                                signature="sig_text",
                                signature_origin="gemini",
                            )
                        ],
                    ),
                    Message(role="user", content=[TextBlock(text="more")]),
                ]
            ),
            params=GenerationParams(),
        )
        contents = serializer.build_provider_request(request)["contents"]
        assert contents[1]["parts"][0]["thoughtSignature"] == "sig_text"

    def test_foreign_signature_is_not_replayed_on_text(self, serializer):
        request = InternalRequest(
            model="gemini-3.8-flash",
            conversation=ConversationContext(
                messages=[
                    Message(
                        role="assistant",
                        content=[
                            TextBlock(text="answer", signature="sig_x", signature_origin="openai")
                        ],
                    )
                ]
            ),
            params=GenerationParams(),
        )
        contents = serializer.build_provider_request(request)["contents"]
        assert "thoughtSignature" not in contents[0]["parts"][0]


# ---------------------------------------------------------------------------
# Response parsing (non-streaming)
# ---------------------------------------------------------------------------


_CANDIDATE_WITH_TOOL_CONTEXT = {
    "candidates": [
        {
            "content": {
                "role": "model",
                "parts": [
                    {
                        "toolCall": {
                            "id": "tc_1",
                            "toolType": "GOOGLE_SEARCH_WEB",
                            "args": {"queries": ["q"]},
                        },
                        "thoughtSignature": "sig_tc",
                    },
                    {
                        "toolResponse": {
                            "id": "tc_1",
                            "toolType": "GOOGLE_SEARCH_WEB",
                            "response": {"search_suggestions": "x"},
                        },
                    },
                    {
                        "functionCall": {
                            "name": "get_weather",
                            "args": {"city": "Paris"},
                            "id": "fc_1",
                        },
                        "thoughtSignature": "sig_fc",
                    },
                    {"text": "the answer", "thoughtSignature": "sig_text"},
                ],
            },
            "finishReason": "STOP",
        }
    ],
    "responseId": "resp_1",
    "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 4, "totalTokenCount": 14},
}


class TestResponseParsing:
    def test_builtin_tool_parts_attach_to_the_function_call(self, serializer):
        response = serializer.parse_provider_response(_CANDIDATE_WITH_TOOL_CONTEXT, model="m")
        tool_use = next(b for b in response.output if isinstance(b, ToolUseBlock))

        assert tool_use.id == "fc_1"
        assert tool_use.extra["thought_signature"] == "sig_fc"
        kinds = [next(iter(part)) for part in tool_use.extra["gemini_tool_parts"]]
        assert kinds == ["toolCall", "toolResponse"]

    def test_text_part_signature_survives(self, serializer):
        response = serializer.parse_provider_response(_CANDIDATE_WITH_TOOL_CONTEXT, model="m")
        text = next(b for b in response.output if isinstance(b, TextBlock))
        assert text.signature == "sig_text"
        assert text.signature_origin == "gemini"

    def test_continuation_token_is_kept(self, serializer):
        payload = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "partial"}]},
                    "finishReason": "CONTINUATION",
                    "continuationToken": "tok",
                }
            ],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
        }
        response = serializer.parse_provider_response(payload, model="m")
        assert response.finish_reason == "length"
        assert response.provider_info["continuationToken"] == "tok"


# ---------------------------------------------------------------------------
# Streaming converter
# ---------------------------------------------------------------------------


class TestStreamingConverter:
    def test_tool_call_keeps_the_gemini_id_and_context(self):
        converter = GeminiStreamingTransformer(model="gemini-3.8-flash", request_id="req_1")
        chunk = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"toolCall": {"id": "tc_1", "toolType": "GOOGLE_SEARCH", "args": {}}},
                            {"functionCall": {"name": "f", "args": {}, "id": "fc_1"}},
                        ]
                    }
                }
            ]
        }
        out = converter.convert_chunk(chunk)
        tool_call = out["choices"][0]["delta"]["tool_calls"][0]

        assert tool_call["id"] == "fc_1"
        assert next(iter(converter.get_tool_context()["fc_1"][0])) == "toolCall"

    def test_text_signature_lands_on_the_assembled_block(self):
        converter = GeminiStreamingTransformer(model="gemini-3.8-flash", request_id="req_1")
        converter.convert_chunk(
            {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "hi", "thoughtSignature": "sig_text"}]},
                        "finishReason": "STOP",
                    }
                ]
            }
        )
        blocks = converter.get_accumulated_output()
        text = next(b for b in blocks if isinstance(b, TextBlock))
        assert text.signature == "sig_text"
        assert text.signature_origin == "gemini"

    def test_continuation_finish_reason_is_length(self):
        converter = GeminiStreamingTransformer(model="gemini-4-argon", request_id="req_1")
        out = converter.convert_chunk(
            {
                "candidates": [
                    {"content": {"parts": [{"text": "x"}]}, "finishReason": "CONTINUATION"}
                ]
            }
        )
        assert out["choices"][0]["finish_reason"] == "length"


# ---------------------------------------------------------------------------
# Finish reasons and usage
# ---------------------------------------------------------------------------


class TestFinishReasons:
    def test_new_gemini_reasons_are_mapped(self):
        assert map_finish_reason("CONTINUATION", "gemini", "openai") == "length"
        assert map_finish_reason("CONTINUATION", "gemini", "anthropic") == "max_tokens"
        assert map_finish_reason("PUP_LIMITED_DISABLED", "gemini", "openai") == "content_filter"


class TestUsageModalityDetails:
    def test_modality_breakdown_is_folded_into_the_details_blocks(self):
        usage = parse_usage(
            "gemini",
            {
                "usageMetadata": {
                    "promptTokenCount": 30,
                    "candidatesTokenCount": 10,
                    "totalTokenCount": 40,
                    "promptTokensDetails": [
                        {"modality": "TEXT", "tokenCount": 20},
                        {"modality": "IMAGE", "tokenCount": 10},
                    ],
                    "candidatesTokensDetails": [{"modality": "AUDIO", "tokenCount": 10}],
                }
            },
        )
        assert usage is not None
        assert usage.prompt_tokens_details is not None
        assert usage.prompt_tokens_details.text_tokens == 20
        assert usage.prompt_tokens_details.image_tokens == 10
        assert usage.completion_tokens_details is not None
        assert usage.completion_tokens_details.audio_tokens == 10

    def test_no_detail_block_without_a_breakdown(self):
        usage = parse_usage(
            "gemini",
            {"usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2}},
        )
        assert usage is not None
        assert usage.prompt_tokens_details is None
        assert usage.completion_tokens_details is None


# ---------------------------------------------------------------------------
# Small conversions
# ---------------------------------------------------------------------------


class TestEmbeddingModelResourceName:
    def test_models_prefix_is_not_doubled(self, serializer):
        from llm_proxy.models.embedding import InternalEmbeddingRequest

        body = serializer.build_provider_embedding_request(
            InternalEmbeddingRequest(model="models/text-embedding-004", input="hi")
        )
        assert body["model"] == "models/text-embedding-004"

        body = serializer.build_provider_embedding_request(
            InternalEmbeddingRequest(model="text-embedding-004", input="hi")
        )
        assert body["model"] == "models/text-embedding-004"


class TestInteractionsGenerationConfigNormalization:
    """The escape hatch is written in generateContent keys, the wire wants snake."""

    def test_camel_keys_are_renamed_and_unknown_dropped(self):
        from llm_proxy.serialization.gemini_interactions.request_builder import (
            normalize_generation_config,
        )

        assert normalize_generation_config(
            {"maxOutputTokens": 5, "topP": 0.5, "includeThoughts": True, "bogus": 1}
        ) == {
            "max_output_tokens": 5,
            "top_p": 0.5,
            "thinking_summaries": "auto",
        }

    def test_include_thoughts_false_suppresses_summaries(self):
        from llm_proxy.serialization.gemini_interactions.request_builder import (
            normalize_generation_config,
        )

        assert normalize_generation_config({"includeThoughts": False}) == {
            "thinking_summaries": "none"
        }
