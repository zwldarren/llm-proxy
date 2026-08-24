# tests/unit/serialization/test_gemini_interactions_serializer.py
"""Tests for GeminiInteractionsProviderSerializer (request builder + response parser)."""

import pytest

from llm_proxy.core.exceptions import ProviderError
from llm_proxy.models import (
    AudioBlock,
    ConversationContext,
    CustomToolUseBlock,
    GeminiSpecificParams,
    GenerationParams,
    ImageBlock,
    ImageSource,
    InternalRequest,
    Message,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from llm_proxy.models.tools import (
    FunctionTool,
    ToolChoiceFunction,
    WebSearchTool,
)
from llm_proxy.serialization.gemini_interactions.serializer import (
    GeminiInteractionsProviderSerializer,
)
from llm_proxy.serialization.providers.registry import get_provider_serializer


@pytest.fixture
def serializer():
    """Create a GeminiInteractionsProviderSerializer instance directly."""
    return GeminiInteractionsProviderSerializer()


def test_registered_in_provider_registry():
    """The serializer is registered under its canonical name."""
    instance = get_provider_serializer("gemini-interactions")
    assert isinstance(instance, GeminiInteractionsProviderSerializer)


class TestRequestBuilder:
    """Request body construction for the Interactions dialect."""

    def test_basic_request(self, serializer):
        request = InternalRequest(
            model="gemini-3.7-flash",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Hello, world!")])]
            ),
            params=GenerationParams(max_tokens=1024, temperature=0.7, seed=5),
        )

        body = serializer.build_provider_request(request)

        assert body["model"] == "gemini-3.7-flash"
        assert body["input"] == [
            {"type": "user_input", "content": [{"type": "text", "text": "Hello, world!"}]}
        ]
        assert body["generation_config"] == {
            "max_output_tokens": 1024,
            "temperature": 0.7,
            "seed": 5,
        }
        # Stateless by default (privacy-friendly)
        assert body["store"] is False

    def test_store_false_default_and_extra_override(self, serializer):
        request = InternalRequest(
            model="gemini-3.7-flash",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Hi")])]
            ),
            params=GenerationParams(),
        )
        assert serializer.build_provider_request(request)["store"] is False

        request.extra = {"store": True}
        assert serializer.build_provider_request(request)["store"] is True

    def test_extra_whitelist_only(self, serializer):
        """Only whitelisted extra keys reach the body; unknown keys are dropped."""
        request = InternalRequest(
            model="gemini-3.7-flash",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Hi")])]
            ),
            params=GenerationParams(),
        )
        request.extra = {
            "store": True,
            "previous_interaction_id": "int_123",
            "background": True,
            "labels": {"env": "prod"},
            "some_unknown_key": 42,
        }

        body = serializer.build_provider_request(request)
        assert body["store"] is True
        assert body["previous_interaction_id"] == "int_123"
        assert body["background"] is True
        assert body["labels"] == {"env": "prod"}
        assert "some_unknown_key" not in body

    def test_multi_turn_stateless_replay(self, serializer):
        """Multi-turn history replays as typed steps in chronological order."""
        request = InternalRequest(
            model="gemini-3.7-flash",
            conversation=ConversationContext(
                system_messages=[SystemMessage.from_text(role="system", text="Be concise.")],
                messages=[
                    Message(role="user", content=[TextBlock(text="What's the weather?")]),
                    Message(
                        role="assistant",
                        content=[
                            ThinkingBlock(thinking="I should check the tool.", signature="sig_1"),
                            ToolUseBlock(
                                id="call_1",
                                name="get_weather",
                                input={"location": "Boston"},
                            ),
                        ],
                    ),
                    Message(
                        role="tool",
                        content=[
                            ToolResultBlock(
                                tool_use_id="call_1",
                                name="get_weather",
                                content="52F rain",
                            )
                        ],
                    ),
                    Message(role="user", content=[TextBlock(text="Thanks")]),
                ],
            ),
            params=GenerationParams(),
        )

        body = serializer.build_provider_request(request)

        assert body["system_instruction"] == "Be concise."
        assert body["input"] == [
            {"type": "user_input", "content": [{"type": "text", "text": "What's the weather?"}]},
            {
                "type": "thought",
                "signature": "sig_1",
                "summary": [{"type": "text", "text": "I should check the tool."}],
            },
            {
                "type": "function_call",
                "id": "call_1",
                "name": "get_weather",
                "arguments": {"location": "Boston"},
            },
            {
                "type": "function_result",
                "call_id": "call_1",
                "name": "get_weather",
                "result": [{"type": "text", "text": "52F rain"}],
            },
            {"type": "user_input", "content": [{"type": "text", "text": "Thanks"}]},
        ]

    def test_function_call_emits_thought_signature(self, serializer):
        """Stateless replay MUST emit the thought signature on function_call
        steps: the live Interactions API rejects function_call steps without
        it ("Function call is missing a thought_signature"). The adapter
        re-attaches cached signatures via block.extra before building."""
        request = InternalRequest(
            model="gemini-3.7-flash",
            conversation=ConversationContext(
                messages=[
                    Message(role="user", content=[TextBlock(text="Weather?")]),
                    Message(
                        role="assistant",
                        content=[
                            ToolUseBlock(
                                id="call_1",
                                name="get_weather",
                                input={"location": "Boston"},
                                extra={"thought_signature": "REAL_SIG"},
                            )
                        ],
                    ),
                ]
            ),
            params=GenerationParams(),
        )

        body = serializer.build_provider_request(request)
        fc = [s for s in body["input"] if s["type"] == "function_call"][0]
        assert fc["signature"] == "REAL_SIG"

    def test_custom_tool_call_emits_thought_signature(self, serializer):
        """Custom tools (e.g. codex's ``exec``, parsed as CustomToolUseBlock)
        must also replay the thought signature: the live API rejects
        function_call steps without it with "Request contains an invalid
        argument." Regression: the signature was only attached for
        ToolUseBlock, so custom_tool_call replays always failed."""
        request = InternalRequest(
            model="gemini-3.7-flash",
            conversation=ConversationContext(
                messages=[
                    Message(role="user", content=[TextBlock(text="Run it")]),
                    Message(
                        role="assistant",
                        content=[
                            CustomToolUseBlock(
                                id="call_1",
                                name="exec",
                                input='{"content": "ls"}',
                                extra={"thought_signature": "REAL_SIG"},
                            )
                        ],
                    ),
                ]
            ),
            params=GenerationParams(),
        )

        body = serializer.build_provider_request(request)
        fc = [s for s in body["input"] if s["type"] == "function_call"][0]
        assert fc["signature"] == "REAL_SIG"
        assert fc["name"] == "exec"
        assert fc["arguments"] == {"content": '{"content": "ls"}'}

    def test_model_output_reconstructs_thought_from_tool_signature(self, serializer):
        """A model_output step followed by a tool call must be preceded by a
        thought step carrying the same signature: the live API rejects the
        replay with "Request contains an invalid argument" otherwise. Clients
        like codex strip thoughts from history, so the converter reconstructs
        the thought from the tool call's cached signature (the signature that
        preceded the call in the original response is the same one the thought
        carried). Regression: the thought was dropped, so any turn with text
        + tool call failed on replay."""
        request = InternalRequest(
            model="gemini-3.7-flash",
            conversation=ConversationContext(
                messages=[
                    Message(role="user", content=[TextBlock(text="Run it")]),
                    Message(
                        role="assistant",
                        content=[
                            TextBlock(text="Let me run that."),
                            CustomToolUseBlock(
                                id="call_1",
                                name="exec",
                                input="ls",
                                extra={"thought_signature": "REAL_SIG"},
                            ),
                        ],
                    ),
                    Message(
                        role="tool",
                        content=[ToolResultBlock(tool_use_id="call_1", content="done")],
                    ),
                ]
            ),
            params=GenerationParams(),
        )

        body = serializer.build_provider_request(request)
        types = [s["type"] for s in body["input"]]
        assert types == [
            "user_input",
            "thought",
            "model_output",
            "function_call",
            "function_result",
        ]
        thought = body["input"][1]
        assert thought == {"type": "thought", "signature": "REAL_SIG"}
        fc = body["input"][3]
        assert fc["signature"] == "REAL_SIG"

    def test_model_output_without_tool_signature_omits_thought(self, serializer):
        """No signature available (cache miss / no thought in the original
        response): the thought step is simply not reconstructed. The trailing
        user message keeps the input ending on user_input so the unsigned
        tool turn is not degraded."""
        request = InternalRequest(
            model="gemini-3.7-flash",
            conversation=ConversationContext(
                messages=[
                    Message(role="user", content=[TextBlock(text="Run it")]),
                    Message(
                        role="assistant",
                        content=[
                            TextBlock(text="Let me run that."),
                            CustomToolUseBlock(id="call_1", name="exec", input="ls"),
                        ],
                    ),
                    Message(
                        role="tool",
                        content=[ToolResultBlock(tool_use_id="call_1", content="done")],
                    ),
                    Message(role="user", content=[TextBlock(text="Thanks")]),
                ]
            ),
            params=GenerationParams(),
        )

        body = serializer.build_provider_request(request)
        types = [s["type"] for s in body["input"]]
        assert types == [
            "user_input",
            "model_output",
            "function_call",
            "function_result",
            "user_input",
        ]
        assert all("signature" not in s for s in body["input"])

    def test_real_thought_block_not_duplicated(self, serializer):
        """When the client does send a ThinkingBlock, it is replayed as-is and
        the reconstructed thought is NOT added (no duplicate thought steps)."""
        request = InternalRequest(
            model="gemini-3.7-flash",
            conversation=ConversationContext(
                messages=[
                    Message(role="user", content=[TextBlock(text="Run it")]),
                    Message(
                        role="assistant",
                        content=[
                            ThinkingBlock(thinking="", signature="REAL_SIG"),
                            TextBlock(text="Let me run that."),
                            CustomToolUseBlock(
                                id="call_1",
                                name="exec",
                                input="ls",
                                extra={"thought_signature": "REAL_SIG"},
                            ),
                        ],
                    ),
                    Message(
                        role="tool",
                        content=[ToolResultBlock(tool_use_id="call_1", content="done")],
                    ),
                ]
            ),
            params=GenerationParams(),
        )

        body = serializer.build_provider_request(request)
        thoughts = [s for s in body["input"] if s["type"] == "thought"]
        assert len(thoughts) == 1
        assert thoughts[0] == {"type": "thought", "signature": "REAL_SIG"}

    def test_unsigned_trailing_tool_turn_degraded_to_user_input(self, serializer):
        """When the input ends with a function_result whose function_call has
        no thought signature (e.g. the client regenerated call ids after a
        session resume/rewind), the trailing tool turn is degraded to a
        user_input step: the live API rejects an unsigned trailing
        function_call with "Request contains an invalid argument."""
        request = InternalRequest(
            model="gemini-3.7-flash",
            conversation=ConversationContext(
                messages=[
                    Message(role="user", content=[TextBlock(text="Run it")]),
                    Message(
                        role="assistant",
                        content=[
                            CustomToolUseBlock(id="call_1", name="exec", input="ls"),
                        ],
                    ),
                    Message(
                        role="tool",
                        content=[ToolResultBlock(tool_use_id="call_1", content="done")],
                    ),
                ]
            ),
            params=GenerationParams(),
        )

        body = serializer.build_provider_request(request)
        types = [s["type"] for s in body["input"]]
        assert types == ["user_input", "user_input"]
        degraded = body["input"][-1]
        text = degraded["content"][0]["text"]
        assert "Tool call exec:" in text
        assert "done" in text

    def test_signed_trailing_tool_turn_not_degraded(self, serializer):
        """A signed trailing function_call keeps its structured steps."""
        request = InternalRequest(
            model="gemini-3.7-flash",
            conversation=ConversationContext(
                messages=[
                    Message(role="user", content=[TextBlock(text="Run it")]),
                    Message(
                        role="assistant",
                        content=[
                            CustomToolUseBlock(
                                id="call_1",
                                name="exec",
                                input="ls",
                                extra={"thought_signature": "REAL_SIG"},
                            ),
                        ],
                    ),
                    Message(
                        role="tool",
                        content=[ToolResultBlock(tool_use_id="call_1", content="done")],
                    ),
                ]
            ),
            params=GenerationParams(),
        )

        body = serializer.build_provider_request(request)
        types = [s["type"] for s in body["input"]]
        assert types == ["user_input", "function_call", "function_result"]
        assert body["input"][1]["signature"] == "REAL_SIG"

    def test_unsigned_middle_tool_turn_not_degraded(self, serializer):
        """An unsigned function_call in the middle (followed by a user_input)
        is accepted by the API and must NOT be degraded."""
        request = InternalRequest(
            model="gemini-3.7-flash",
            conversation=ConversationContext(
                messages=[
                    Message(role="user", content=[TextBlock(text="Run it")]),
                    Message(
                        role="assistant",
                        content=[
                            CustomToolUseBlock(id="call_1", name="exec", input="ls"),
                        ],
                    ),
                    Message(
                        role="tool",
                        content=[ToolResultBlock(tool_use_id="call_1", content="done")],
                    ),
                    Message(role="user", content=[TextBlock(text="Thanks")]),
                ]
            ),
            params=GenerationParams(),
        )

        body = serializer.build_provider_request(request)
        types = [s["type"] for s in body["input"]]
        assert types == ["user_input", "function_call", "function_result", "user_input"]

    def test_unsigned_trailing_tool_turn_not_degraded_for_gemini_25(self, serializer):
        """Gemini 2.5 treats thought signatures as optional (per the
        thought-signatures docs), so an unsigned trailing tool turn must NOT
        be degraded — the structured history is kept."""
        request = InternalRequest(
            model="gemini-2.5-pro",
            conversation=ConversationContext(
                messages=[
                    Message(role="user", content=[TextBlock(text="Run it")]),
                    Message(
                        role="assistant",
                        content=[
                            CustomToolUseBlock(id="call_1", name="exec", input="ls"),
                        ],
                    ),
                    Message(
                        role="tool",
                        content=[ToolResultBlock(tool_use_id="call_1", content="done")],
                    ),
                ]
            ),
            params=GenerationParams(),
        )

        body = serializer.build_provider_request(request)
        types = [s["type"] for s in body["input"]]
        assert types == ["user_input", "function_call", "function_result"]
        assert "signature" not in body["input"][1]

    def test_parallel_unsigned_trailing_turn_fully_degraded(self, serializer):
        """When the whole trailing parallel turn is unsigned, ALL
        function_call/function_result pairs are degraded into a single
        user_input step (not just the last pair)."""
        request = InternalRequest(
            model="gemini-3.7-flash",
            conversation=ConversationContext(
                messages=[
                    Message(role="user", content=[TextBlock(text="Both")]),
                    Message(
                        role="assistant",
                        content=[
                            CustomToolUseBlock(id="call_1", name="exec", input="ls"),
                            CustomToolUseBlock(id="call_2", name="exec", input="pwd"),
                        ],
                    ),
                    Message(
                        role="tool",
                        content=[
                            ToolResultBlock(tool_use_id="call_1", content="done"),
                            ToolResultBlock(tool_use_id="call_2", content="/home"),
                        ],
                    ),
                ]
            ),
            params=GenerationParams(),
        )

        body = serializer.build_provider_request(request)
        types = [s["type"] for s in body["input"]]
        assert types == ["user_input", "user_input"]
        text = body["input"][-1]["content"][0]["text"]
        assert "Tool call exec:" in text
        assert "done" in text
        assert "/home" in text

    def test_parallel_turn_first_signed_not_degraded(self, serializer):
        """The API only requires a signature on the FIRST function_call of
        the current turn; a signed first call with an unsigned parallel
        sibling must NOT be degraded."""
        request = InternalRequest(
            model="gemini-3.7-flash",
            conversation=ConversationContext(
                messages=[
                    Message(role="user", content=[TextBlock(text="Both")]),
                    Message(
                        role="assistant",
                        content=[
                            CustomToolUseBlock(
                                id="call_1",
                                name="exec",
                                input="ls",
                                extra={"thought_signature": "REAL_SIG"},
                            ),
                            CustomToolUseBlock(id="call_2", name="exec", input="pwd"),
                        ],
                    ),
                    Message(
                        role="tool",
                        content=[
                            ToolResultBlock(tool_use_id="call_1", content="done"),
                            ToolResultBlock(tool_use_id="call_2", content="/home"),
                        ],
                    ),
                ]
            ),
            params=GenerationParams(),
        )

        body = serializer.build_provider_request(request)
        types = [s["type"] for s in body["input"]]
        assert types == [
            "user_input",
            "function_call",
            "function_call",
            "function_result",
            "function_result",
        ]
        assert body["input"][1]["signature"] == "REAL_SIG"
        assert "signature" not in body["input"][2]

    def test_parallel_turn_first_unsigned_degraded(self, serializer):
        """The FIRST function_call of the current turn is the one the API
        validates; when it is unsigned the whole turn is degraded even if a
        later parallel call carries a signature."""
        request = InternalRequest(
            model="gemini-3.7-flash",
            conversation=ConversationContext(
                messages=[
                    Message(role="user", content=[TextBlock(text="Both")]),
                    Message(
                        role="assistant",
                        content=[
                            CustomToolUseBlock(id="call_1", name="exec", input="ls"),
                            CustomToolUseBlock(
                                id="call_2",
                                name="exec",
                                input="pwd",
                                extra={"thought_signature": "REAL_SIG"},
                            ),
                        ],
                    ),
                    Message(
                        role="tool",
                        content=[
                            ToolResultBlock(tool_use_id="call_1", content="done"),
                            ToolResultBlock(tool_use_id="call_2", content="/home"),
                        ],
                    ),
                ]
            ),
            params=GenerationParams(),
        )

        body = serializer.build_provider_request(request)
        types = [s["type"] for s in body["input"]]
        assert types == ["user_input", "user_input"]
        text = body["input"][-1]["content"][0]["text"]
        assert "done" in text
        assert "/home" in text

    def test_degradation_keeps_model_output(self, serializer):
        """Degrading the trailing tool turn keeps the model's own text as a
        model_output step (it is valid once the turn no longer ends with a
        tool call)."""
        request = InternalRequest(
            model="gemini-3.7-flash",
            conversation=ConversationContext(
                messages=[
                    Message(role="user", content=[TextBlock(text="Run it")]),
                    Message(
                        role="assistant",
                        content=[
                            TextBlock(text="Let me run that."),
                            CustomToolUseBlock(id="call_1", name="exec", input="ls"),
                        ],
                    ),
                    Message(
                        role="tool",
                        content=[ToolResultBlock(tool_use_id="call_1", content="done")],
                    ),
                ]
            ),
            params=GenerationParams(),
        )

        body = serializer.build_provider_request(request)
        types = [s["type"] for s in body["input"]]
        assert types == ["user_input", "model_output", "user_input"]
        text = body["input"][-1]["content"][0]["text"]
        assert "Tool call exec:" in text
        assert "done" in text

    def test_function_call_without_signature_omits_field(self, serializer):
        """When no signature is available (e.g. the model produced the call
        without a thought step), the field is simply omitted."""
        request = InternalRequest(
            model="gemini-3.7-flash",
            conversation=ConversationContext(
                messages=[
                    Message(role="user", content=[TextBlock(text="Weather?")]),
                    Message(
                        role="assistant",
                        content=[
                            ToolUseBlock(
                                id="call_1",
                                name="get_weather",
                                input={"location": "Boston"},
                            )
                        ],
                    ),
                ]
            ),
            params=GenerationParams(),
        )

        body = serializer.build_provider_request(request)
        fc = [s for s in body["input"] if s["type"] == "function_call"][0]
        assert "signature" not in fc
        assert "thought_signature" not in fc

    def test_tools_conversion(self, serializer):
        request = InternalRequest(
            model="gemini-3.7-flash",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Hi")])]
            ),
            params=GenerationParams(),
            tools=[
                FunctionTool(
                    name="get_weather",
                    description="Get weather",
                    parameters={
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                ),
                WebSearchTool(name="web_search", type="web_search_20250305"),
            ],
            tool_choice=ToolChoiceFunction(name="get_weather"),
        )

        body = serializer.build_provider_request(request)

        assert body["tools"] == [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
            {"type": "google_search"},
        ]
        assert body["generation_config"]["tool_choice"] == {
            "allowed_tools": {"mode": "any", "tools": ["get_weather"]}
        }

    def test_structured_output_uses_plain_json_schema(self, serializer):
        """response_format carries the JSON Schema untouched (lowercase types,
        no sanitization) per the Interactions dialect."""
        schema = {
            "type": "object",
            "properties": {
                "recipe_name": {"type": "string"},
                "ingredients": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["recipe_name", "ingredients"],
            "additionalProperties": False,
        }
        from llm_proxy.models.types import ResponseFormat

        request = InternalRequest(
            model="gemini-3.7-flash",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Recipe?")])]
            ),
            params=GenerationParams(
                response_format=ResponseFormat(type="json_schema", json_schema=schema)
            ),
        )

        body = serializer.build_provider_request(request)
        assert body["response_format"] == {
            "type": "text",
            "mime_type": "application/json",
            "schema": schema,
        }

    def test_structured_output_unwraps_openai_wrapper(self, serializer):
        """The OpenAI wrapper {name, description, schema, strict} is unwrapped
        so the Interactions schema field carries the plain JSON Schema."""
        plain_schema = {
            "type": "object",
            "properties": {"recipe_name": {"type": "string"}},
            "required": ["recipe_name"],
        }
        from llm_proxy.models.types import ResponseFormat

        request = InternalRequest(
            model="gemini-3.7-flash",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Recipe?")])]
            ),
            params=GenerationParams(
                response_format=ResponseFormat(
                    type="json_schema",
                    json_schema={
                        "name": "recipe",
                        "description": "A recipe",
                        "schema": plain_schema,
                        "strict": True,
                    },
                )
            ),
        )

        body = serializer.build_provider_request(request)
        assert body["response_format"] == {
            "type": "text",
            "mime_type": "application/json",
            "schema": plain_schema,
        }

    def test_structured_output_openai_protocol_flow(self, serializer):
        """End-to-end: a raw OpenAI chat-completions request with a wrapped
        json_schema response_format reaches the Interactions body as the
        plain JSON Schema."""
        from llm_proxy.protocols.registry import get_protocol_serializer

        plain_schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }
        raw = {
            "model": "gemini-3.7-flash",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "math_answer",
                    "description": "A math answer",
                    "schema": plain_schema,
                    "strict": True,
                },
            },
        }
        protocol = get_protocol_serializer("openai")
        request = protocol.parse_request(raw)

        body = serializer.build_provider_request(request)
        assert body["response_format"] == {
            "type": "text",
            "mime_type": "application/json",
            "schema": plain_schema,
        }

    def test_thinking_level_mapping(self, serializer):
        """Thinking effort maps to the lowercase thinking_level enum."""
        from llm_proxy.models.types import ThinkingConfig

        request = InternalRequest(
            model="gemini-3.7-flash",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Hi")])]
            ),
            params=GenerationParams(thinking=ThinkingConfig(type="enabled", budget_tokens=32000)),
        )
        body = serializer.build_provider_request(request)
        assert body["generation_config"]["thinking_level"] == "high"
        # No display request -> thinking_summaries stays omitted (API default
        # shows the final output only).
        assert "thinking_summaries" not in body["generation_config"]

        request.params.thinking = ThinkingConfig(type="enabled", effort="minimal")
        body = serializer.build_provider_request(request)
        assert body["generation_config"]["thinking_level"] == "low"

        # display=summary/full request visible thinking -> auto summaries;
        # explicit suppression (none) maps to "none".
        request.params.thinking = ThinkingConfig(type="enabled", effort="high", display="summary")
        body = serializer.build_provider_request(request)
        assert body["generation_config"]["thinking_summaries"] == "auto"

        request.params.thinking = ThinkingConfig(type="enabled", effort="high", display="none")
        body = serializer.build_provider_request(request)
        assert body["generation_config"]["thinking_summaries"] == "none"

        request.params.thinking = ThinkingConfig(type="disabled")
        body = serializer.build_provider_request(request)
        assert "thinking_level" not in body.get("generation_config", {})
        assert "thinking_summaries" not in body.get("generation_config", {})

    def test_tts_response_format_and_speech_config(self, serializer):
        request = InternalRequest(
            model="gemini-3.1-flash-tts-preview",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Say hi")])]
            ),
            params=GenerationParams(),
        )
        body = serializer.build_provider_request(request)
        assert body["response_format"] == {"type": "audio"}
        assert body["generation_config"]["speech_config"] == [{"voice": "Kore"}]

    def test_image_model_response_format(self, serializer):
        request = InternalRequest(
            model="gemini-3.1-flash-image",
            conversation=ConversationContext(
                messages=[
                    Message(
                        role="user",
                        content=[
                            TextBlock(text="![img](data:image/png;base64,iVBORw0KGgo=)"),
                        ],
                    )
                ]
            ),
            params=GenerationParams(),
        )
        body = serializer.build_provider_request(request)
        assert body["response_format"] == {"type": "image"}
        # markdown data-URI images become image content items for image models
        items = body["input"][0]["content"]
        assert items[0]["type"] == "image"
        assert items[0]["data"] == "iVBORw0KGgo="

    def test_warn_and_drop_unsupported_params(self, serializer, caplog):
        """Interactions-unsupported parameters are warned and dropped."""
        request = InternalRequest(
            model="gemini-3.7-flash",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Hi")])]
            ),
            params=GenerationParams(
                frequency_penalty=0.5,
                presence_penalty=0.5,
                n=2,
                gemini=GeminiSpecificParams(
                    top_k=5,
                    cached_content="projects/x/locations/global/cachedContents/abc",
                ),
            ),
        )
        body = serializer.build_provider_request(request)
        assert "cachedContent" not in body
        assert "generation_config" not in body or "topK" not in body["generation_config"]
        assert "candidateCount" not in body

        assert "cached_content" in caplog.text
        assert "top_k" in caplog.text
        assert "frequency_penalty" in caplog.text
        assert "n=2" in caplog.text

    def test_safety_settings_legacy_vocabulary_converted(self, serializer):
        """Legacy generateContent safety settings are converted to the
        Interactions vocabulary (category -> type, BLOCK_* -> block_*)."""
        request = InternalRequest(
            model="gemini-3.7-flash",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Hi")])]
            ),
            params=GenerationParams(
                gemini=GeminiSpecificParams(
                    safety_settings=[
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                        },
                        {
                            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                            "threshold": "BLOCK_ONLY_HIGH",
                            "method": "severity",
                        },
                    ]
                )
            ),
        )
        body = serializer.build_provider_request(request)
        assert body["safety_settings"] == [
            {"type": "harassment", "threshold": "block_medium_and_above"},
            {
                "type": "dangerous_content",
                "threshold": "block_only_high",
                "method": "severity",
            },
        ]

    def test_safety_settings_interactions_vocabulary_passthrough(self, serializer):
        """Entries already in the Interactions vocabulary pass through."""
        request = InternalRequest(
            model="gemini-3.7-flash",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Hi")])]
            ),
            params=GenerationParams(
                gemini=GeminiSpecificParams(
                    safety_settings=[
                        {"type": "jailbreak", "threshold": "off"},
                        {"type": "image_hate", "threshold": "block_none"},
                    ]
                )
            ),
        )
        body = serializer.build_provider_request(request)
        assert body["safety_settings"] == [
            {"type": "jailbreak", "threshold": "off"},
            {"type": "image_hate", "threshold": "block_none"},
        ]

    def test_safety_settings_unknown_values_dropped(self, serializer, caplog):
        """Entries with unknown categories/thresholds are warn-and-dropped."""
        request = InternalRequest(
            model="gemini-3.7-flash",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Hi")])]
            ),
            params=GenerationParams(
                gemini=GeminiSpecificParams(
                    safety_settings=[
                        {"category": "HARM_CATEGORY_UNKNOWN", "threshold": "BLOCK_NONE"},
                        {"type": "hate_speech", "threshold": "BLOCK_WEIRD"},
                        {"type": "hate_speech", "threshold": "block_none", "method": "bogus"},
                        {"type": "hate_speech", "threshold": "block_none"},
                    ]
                )
            ),
        )
        body = serializer.build_provider_request(request)
        assert body["safety_settings"] == [{"type": "hate_speech", "threshold": "block_none"}]
        assert "unsupported safety setting" in caplog.text
        assert "unsupported safety method" in caplog.text

    def test_service_tier_mapping(self, serializer):
        request = InternalRequest(
            model="gemini-3.7-flash",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Hi")])]
            ),
            params=GenerationParams(),
        )
        from llm_proxy.models.params import OpenAISpecificParams

        request.params.openai = OpenAISpecificParams(service_tier="flex")
        body = serializer.build_provider_request(request)
        assert body["service_tier"] == "flex"

    def test_media_input_items(self, serializer):
        """Multimodal user content becomes typed Content items."""
        request = InternalRequest(
            model="gemini-3.7-flash",
            conversation=ConversationContext(
                messages=[
                    Message(
                        role="user",
                        content=[
                            TextBlock(text="Describe this"),
                            ImageBlock(
                                source=ImageSource(
                                    type="base64",
                                    data="AAAA",
                                    media_type="image/png",
                                )
                            ),
                        ],
                    )
                ]
            ),
            params=GenerationParams(),
        )
        body = serializer.build_provider_request(request)
        content = body["input"][0]["content"]
        assert content == [
            {"type": "text", "text": "Describe this"},
            {"type": "image", "data": "AAAA", "mime_type": "image/png"},
        ]


class TestResponseParser:
    """steps[] timeline parsing."""

    def test_parse_text_response_with_usage(self, serializer):
        response = {
            "id": "int_1",
            "status": "completed",
            "steps": [
                {"type": "user_input", "content": [{"type": "text", "text": "Hi"}]},
                {
                    "type": "model_output",
                    "content": [{"type": "text", "text": "Hello!"}],
                },
            ],
            "usage": {
                "total_input_tokens": 7,
                "total_output_tokens": 20,
                "total_thought_tokens": 22,
                "total_cached_tokens": 3,
                "total_tool_use_tokens": 0,
                "total_tokens": 49,
            },
        }
        result = serializer.parse_provider_response(response, model="gemini-3.7-flash")

        assert result.finish_reason == "stop"
        assert len(result.output) == 1
        assert isinstance(result.output[0], TextBlock)
        assert result.output[0].text == "Hello!"
        assert result.usage is not None
        # tool use (0) folds into input; thought tokens fold into output
        assert result.usage.input_tokens == 7
        assert result.usage.output_tokens == 42
        assert result.usage.total_tokens == 49
        assert result.usage.cache_read_input_tokens == 3
        assert result.usage.reasoning_tokens == 22

    def test_parse_requires_action_tool_calls(self, serializer):
        response = {
            "id": "int_1",
            "status": "requires_action",
            "steps": [
                {
                    "type": "thought",
                    "signature": "sig_1",
                    "summary": [{"type": "text", "text": "Call it."}],
                },
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "name": "get_weather",
                    "arguments": {"location": "Boston"},
                },
            ],
        }
        result = serializer.parse_provider_response(response, model="gemini-3.7-flash")

        assert result.finish_reason == "tool_calls"
        assert len(result.output) == 2
        assert isinstance(result.output[0], ThinkingBlock)
        assert result.output[0].thinking == "Call it."
        assert result.output[0].signature == "sig_1"
        assert isinstance(result.output[1], ToolUseBlock)
        assert result.output[1].id == "fc_1"
        assert result.output[1].name == "get_weather"
        assert result.output[1].input == {"location": "Boston"}
        # The thought signature is attached to the function_call block so the
        # adapter can cache it and replay it on the next request (the live API
        # rejects function_call steps without it).
        assert result.output[1].extra == {"thought_signature": "sig_1"}

    def test_parse_function_call_without_thought_has_no_extra(self, serializer):
        response = {
            "id": "int_1",
            "status": "requires_action",
            "steps": [
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "name": "get_weather",
                    "arguments": {"location": "Boston"},
                },
            ],
        }
        result = serializer.parse_provider_response(response, model="gemini-3.7-flash")
        assert isinstance(result.output[0], ToolUseBlock)
        assert result.output[0].extra == {}

    def test_parse_incomplete_maps_to_length(self, serializer):
        response = {"id": "int_1", "status": "incomplete", "steps": []}
        result = serializer.parse_provider_response(response, model="gemini-3.7-flash")
        assert result.finish_reason == "length"

    def test_parse_budget_exceeded_maps_to_length(self, serializer):
        response = {"id": "int_1", "status": "budget_exceeded", "steps": []}
        result = serializer.parse_provider_response(response, model="gemini-3.7-flash")
        assert result.finish_reason == "length"

    def test_parse_openai_style_usage_aliases(self, serializer):
        """The migration guide streams OpenAI-style usage on completed events."""
        response = {
            "id": "int_1",
            "status": "completed",
            "steps": [{"type": "model_output", "content": [{"type": "text", "text": "Hi"}]}],
            "usage": {"prompt_tokens": 256, "completion_tokens": 128, "total_tokens": 384},
        }
        result = serializer.parse_provider_response(response, model="gemini-3.7-flash")
        assert result.usage is not None
        assert result.usage.input_tokens == 256
        assert result.usage.output_tokens == 128
        assert result.usage.total_tokens == 384

    def test_parse_failed_status_raises(self, serializer):
        response = {
            "id": "int_1",
            "status": "failed",
            "errors": [{"code": "internal", "message": "boom"}],
        }
        with pytest.raises(ProviderError) as excinfo:
            serializer.parse_provider_response(response, model="gemini-3.7-flash")
        assert "boom" in str(excinfo.value)

    def test_parse_multimodal_output(self, serializer):
        response = {
            "id": "int_1",
            "status": "completed",
            "steps": [
                {
                    "type": "model_output",
                    "content": [
                        {"type": "text", "text": "Here is your image:"},
                        {"type": "image", "data": "IMG_B64", "mime_type": "image/png"},
                        {"type": "audio", "data": "AUD_B64", "mime_type": "audio/l16"},
                    ],
                }
            ],
        }
        result = serializer.parse_provider_response(response, model="gemini-3.7-flash")
        assert len(result.output) == 3
        assert isinstance(result.output[0], TextBlock)
        assert isinstance(result.output[1], ImageBlock)
        assert result.output[1].source.data == "IMG_B64"
        assert isinstance(result.output[2], AudioBlock)
        assert result.output[2].source.data == "AUD_B64"

    def test_parse_inline_annotations(self, serializer):
        response = {
            "id": "int_1",
            "status": "completed",
            "steps": [
                {
                    "type": "model_output",
                    "content": [
                        {
                            "type": "text",
                            "text": "Spain won Euro 2024.",
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "start_index": 0,
                                    "end_index": 18,
                                    "uri": "https://example.com",
                                    "title": "Example",
                                }
                            ],
                        }
                    ],
                }
            ],
        }
        result = serializer.parse_provider_response(response, model="gemini-3.7-flash")
        text_block = result.output[0]
        assert text_block.citations is not None
        assert text_block.citations[0]["type"] == "url_citation"
        assert text_block.citations[0]["url_citation"]["url"] == "https://example.com"

    def test_parse_search_grounding_excludes_tool_use_from_input(self, serializer):
        """Search-grounded tool-use tokens are billed via the search fee."""
        response = {
            "id": "int_1",
            "status": "completed",
            "steps": [
                {"type": "google_search_call", "id": "s1"},
                {"type": "google_search_result", "call_id": "s1"},
                {"type": "model_output", "content": [{"type": "text", "text": "answer"}]},
            ],
            "usage": {
                "total_input_tokens": 100,
                "total_output_tokens": 10,
                "total_tool_use_tokens": 50,
                "total_thought_tokens": 0,
                "total_tokens": 110,
            },
        }
        result = serializer.parse_provider_response(response, model="gemini-3.7-flash")
        assert result.usage is not None
        assert result.usage.input_tokens == 100  # tool use excluded
        assert result.usage.web_search_requests == 1

    def test_parse_grounding_tool_count_usage(self, serializer):
        """grounding_tool_count also drives the web-search request count."""
        response = {
            "id": "int_1",
            "status": "completed",
            "steps": [{"type": "model_output", "content": [{"type": "text", "text": "a"}]}],
            "usage": {
                "total_input_tokens": 10,
                "total_output_tokens": 5,
                "total_tool_use_tokens": 5,
                "total_thought_tokens": 0,
                "total_tokens": 15,
                "grounding_tool_count": [{"type": "google_search", "count": 2}],
            },
        }
        result = serializer.parse_provider_response(response, model="gemini-3.7-flash")
        assert result.usage is not None
        assert result.usage.input_tokens == 10
        assert result.usage.web_search_requests == 2
