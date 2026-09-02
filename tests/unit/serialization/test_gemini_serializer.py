# tests/unit/serialization/test_gemini_serializer.py
"""Tests for GeminiProviderSerializer."""

import pytest

from llm_proxy.core.exceptions import ValidationError
from llm_proxy.models import (
    ConversationContext,
    FileBlock,
    GenerationParams,
    ImageBlock,
    ImageSource,
    InternalRequest,
    Message,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from llm_proxy.models.tools import FunctionTool
from llm_proxy.serialization.gemini.serializer import GeminiProviderSerializer


@pytest.fixture
def serializer():
    """Create a GeminiProviderSerializer instance directly."""
    return GeminiProviderSerializer()


def test_build_provider_request_basic(serializer):
    """Test building provider request with a single user message."""
    request = InternalRequest(
        model="gemini-2.0-flash",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="Hello, world!")])]
        ),
        params=GenerationParams(max_tokens=1024),
    )

    body = serializer.build_provider_request(request)

    assert body["model"] == "models/gemini-2.0-flash"
    assert "contents" in body
    assert len(body["contents"]) == 1
    assert body["contents"][0]["role"] == "user"
    assert body["contents"][0]["parts"][0]["text"] == "Hello, world!"


def test_build_provider_request_with_assistant_message(serializer):
    """Test building provider request with assistant role mapping to 'model'."""
    request = InternalRequest(
        model="gemini-2.0-flash",
        conversation=ConversationContext(
            messages=[
                Message(role="user", content=[TextBlock(text="Hello")]),
                Message(role="assistant", content=[TextBlock(text="Hi there!")]),
            ]
        ),
        params=GenerationParams(),
    )

    body = serializer.build_provider_request(request)

    assert len(body["contents"]) == 2
    assert body["contents"][0]["role"] == "user"
    assert body["contents"][1]["role"] == "model"


def test_build_provider_request_with_system_prompt(serializer):
    """Test building provider request with system instruction."""
    request = InternalRequest(
        model="gemini-2.0-flash",
        conversation=ConversationContext(
            system_messages=[
                SystemMessage.from_text(role="system", text="You are a helpful assistant.")
            ],
            messages=[Message(role="user", content=[TextBlock(text="Hello")])],
        ),
        params=GenerationParams(),
    )

    body = serializer.build_provider_request(request)

    assert "system_instruction" in body
    assert body["system_instruction"]["parts"][0]["text"] == "You are a helpful assistant."


def test_build_provider_request_with_generation_params(serializer):
    """Test building provider request with generation parameters."""
    request = InternalRequest(
        model="gemini-2.0-flash",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="Hello")])]
        ),
        params=GenerationParams(
            max_tokens=2048,
            temperature=0.7,
            top_p=0.9,
            stop=["STOP", "END"],
        ),
    )

    body = serializer.build_provider_request(request)

    assert "generationConfig" in body
    assert body["generationConfig"]["maxOutputTokens"] == 2048
    assert body["generationConfig"]["temperature"] == 0.7
    assert body["generationConfig"]["topP"] == 0.9
    assert body["generationConfig"]["stopSequences"] == ["STOP", "END"]


def test_parse_provider_response_basic(serializer):
    """Test parsing basic text response with usage."""
    provider_response = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Hello! How can I help you?"}],
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 8,
            "totalTokenCount": 18,
        },
        "modelVersion": "gemini-2.0-flash",
    }

    response = serializer.parse_provider_response(provider_response, model="gemini-2.0-flash")

    assert response.id is not None
    assert response.model == "gemini-2.0-flash"
    assert len(response.output) == 1
    assert isinstance(response.output[0], TextBlock)
    assert response.output[0].text == "Hello! How can I help you?"
    assert response.finish_reason == "stop"
    assert response.usage is not None
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 8
    assert response.usage.total_tokens == 18


def test_parse_provider_response_with_tool_call(serializer):
    """Test parsing response with function call."""
    provider_response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "functionCall": {
                                "name": "get_weather",
                                "args": {"location": "Boston", "unit": "celsius"},
                            }
                        }
                    ],
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 25,
            "candidatesTokenCount": 12,
            "totalTokenCount": 37,
        },
        "modelVersion": "gemini-2.0-flash",
    }

    response = serializer.parse_provider_response(provider_response)

    assert len(response.output) == 1
    assert isinstance(response.output[0], ToolUseBlock)
    assert response.output[0].name == "get_weather"
    assert response.output[0].input == {"location": "Boston", "unit": "celsius"}


def test_parse_provider_response_with_tool_call_thought_signature(serializer):
    """Test parsing response with functionCall containing thoughtSignature."""
    provider_response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "functionCall": {
                                "name": "get_weather",
                                "args": {"location": "Boston"},
                            },
                            "thoughtSignature": "sig_abc_123",
                        }
                    ],
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 25,
            "candidatesTokenCount": 12,
            "totalTokenCount": 37,
        },
    }

    response = serializer.parse_provider_response(provider_response)

    assert len(response.output) == 1
    assert isinstance(response.output[0], ToolUseBlock)
    assert response.output[0].extra.get("thought_signature") == "sig_abc_123"


def test_build_provider_request_with_corrupted_base64_image_raises(serializer):
    """Regression: corrupted base64 file data must raise ValidationError."""
    request = InternalRequest(
        model="gemini-2.0-flash",
        conversation=ConversationContext(
            messages=[
                Message(
                    role="user",
                    content=[
                        FileBlock(file_data="data:image/png;base64,!!!not-valid-base64!!!"),
                    ],
                )
            ]
        ),
        params=GenerationParams(),
    )

    with pytest.raises(ValidationError):
        serializer.build_provider_request(request)


def test_build_provider_request_with_tool_use_thought_signature(serializer):
    """Test that ToolUseBlock thought_signature is passed into Gemini functionCall part."""
    request = InternalRequest(
        model="gemini-2.0-flash",
        conversation=ConversationContext(
            messages=[
                Message(role="user", content=[TextBlock(text="What's the weather?")]),
                Message(
                    role="assistant",
                    content=[
                        ToolUseBlock(
                            id="call_123",
                            name="get_weather",
                            input={"location": "Boston"},
                            extra={"thought_signature": "sig_abc_123"},
                        )
                    ],
                ),
            ]
        ),
        params=GenerationParams(),
    )

    body = serializer.build_provider_request(request)

    # The assistant message becomes role "model" in Gemini
    model_content = [c for c in body["contents"] if c["role"] == "model"]
    assert len(model_content) == 1
    parts = model_content[0]["parts"]
    assert len(parts) == 1
    assert "functionCall" in parts[0]
    assert parts[0]["functionCall"]["name"] == "get_weather"
    # thoughtSignature lives at the PART level (sibling of functionCall);
    # Gemini 3.x rejects it inside functionCall as an unknown field.
    assert parts[0]["thoughtSignature"] == "sig_abc_123"


def test_build_provider_request_with_tool_call_without_thought_signature(serializer):
    """Test that functionCall part without thought_signature does not include the key."""
    request = InternalRequest(
        model="gemini-2.0-flash",
        conversation=ConversationContext(
            messages=[
                Message(role="user", content=[TextBlock(text="What's the weather?")]),
                Message(
                    role="assistant",
                    content=[
                        ToolUseBlock(
                            id="call_123",
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

    model_content = [c for c in body["contents"] if c["role"] == "model"]
    parts = model_content[0]["parts"]
    assert "thoughtSignature" not in parts[0]


def test_build_provider_request_with_tools(serializer):
    """Test building provider request with tools."""
    request = InternalRequest(
        model="gemini-2.0-flash",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="What's the weather?")])]
        ),
        params=GenerationParams(),
        tools=[
            FunctionTool(
                name="get_weather",
                description="Get current weather for a location",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                    },
                    "required": ["location"],
                },
            )
        ],
    )

    body = serializer.build_provider_request(request)

    assert "tools" in body
    # GenerateContentRequest.tools is array<Tool>.
    assert isinstance(body["tools"], list) and len(body["tools"]) == 1
    func_decl = body["tools"][0]["function_declarations"][0]
    assert func_decl["name"] == "get_weather"
    assert func_decl["description"] == "Get current weather for a location"
    assert func_decl["parameters"]["properties"]["location"]["type"] == "string"


def test_build_provider_request_with_web_search_tool(serializer):
    """Test building provider request with WebSearchTool -> google_search."""
    from llm_proxy.models.tools import WebSearchTool

    request = InternalRequest(
        model="gemini-2.0-flash",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="Search for news")])]
        ),
        params=GenerationParams(),
        tools=[
            WebSearchTool(name="web_search", type="web_search_20250305"),
        ],
    )

    body = serializer.build_provider_request(request)

    assert "tools" in body
    assert isinstance(body["tools"], list) and len(body["tools"]) == 1
    assert body["tools"][0]["google_search"] == {}
    assert "function_declarations" not in body["tools"][0]


def test_build_provider_request_with_mixed_tools(serializer):
    """Test building provider request with both FunctionTool and WebSearchTool."""
    from llm_proxy.models.tools import WebSearchTool

    request = InternalRequest(
        model="gemini-2.0-flash",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="Search and calculate")])]
        ),
        params=GenerationParams(),
        tools=[
            WebSearchTool(name="web_search", type="web_search_20250305"),
            FunctionTool(
                name="calculator",
                description="Calculate something",
                parameters={"type": "object"},
            ),
        ],
    )

    body = serializer.build_provider_request(request)

    assert "tools" in body
    assert isinstance(body["tools"], list) and len(body["tools"]) == 1
    assert body["tools"][0]["google_search"] == {}
    assert "function_declarations" in body["tools"][0]
    assert body["tools"][0]["function_declarations"][0]["name"] == "calculator"


def test_build_provider_request_with_image(serializer):
    """Test building provider request with image content."""
    request = InternalRequest(
        model="gemini-2.0-flash",
        conversation=ConversationContext(
            messages=[
                Message(
                    role="user",
                    content=[
                        TextBlock(text="What is in this image?"),
                        ImageBlock(
                            source=ImageSource(
                                type="base64",
                                data="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
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

    assert len(body["contents"]) == 1
    parts = body["contents"][0]["parts"]
    assert len(parts) == 2
    assert parts[0]["text"] == "What is in this image?"
    assert "inline_data" in parts[1]
    assert parts[1]["inline_data"]["mime_type"] == "image/png"


def test_parse_provider_response_with_multiple_text_parts(serializer):
    """Test parsing response with multiple text parts."""
    provider_response = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Part 1 "}, {"text": "Part 2"}],
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 5,
            "candidatesTokenCount": 10,
            "totalTokenCount": 15,
        },
        "modelVersion": "gemini-2.0-flash",
    }

    response = serializer.parse_provider_response(provider_response)

    assert len(response.output) == 1
    assert isinstance(response.output[0], TextBlock)
    assert response.output[0].text == "Part 1 Part 2"


def test_build_provider_request_with_tool_result(serializer):
    """Test building provider request with tool result message."""
    request = InternalRequest(
        model="gemini-2.0-flash",
        conversation=ConversationContext(
            messages=[
                Message(role="user", content=[TextBlock(text="What's the weather?")]),
                Message(
                    role="tool",
                    content=[ToolResultBlock(tool_use_id="call_abc123", content="Sunny, 72°F")],
                ),
            ]
        ),
        params=GenerationParams(),
    )

    body = serializer.build_provider_request(request)

    assert len(body["contents"]) == 2
    assert body["contents"][1]["role"] == "user"
    assert "functionResponse" in body["contents"][1]["parts"][0]
    func_resp = body["contents"][1]["parts"][0]["functionResponse"]
    assert func_resp["name"] == "call_abc123"
    assert func_resp["response"]["content"] == "Sunny, 72°F"


def test_build_provider_request_with_tool_result_no_name(serializer):
    """Test tool result without name falls back to tool_use_id."""
    request = InternalRequest(
        model="gemini-2.0-flash",
        conversation=ConversationContext(
            messages=[
                Message(role="user", content=[TextBlock(text="What's the weather?")]),
                Message(
                    role="tool",
                    content=[ToolResultBlock(tool_use_id="get_weather", content="Sunny, 72°F")],
                ),
            ]
        ),
        params=GenerationParams(),
    )

    body = serializer.build_provider_request(request)
    func_resp = body["contents"][1]["parts"][0]["functionResponse"]
    assert func_resp["name"] == "get_weather"


def test_build_provider_request_with_tool_result_json_content(serializer):
    """Test tool result with JSON content parsed into response dict."""
    request = InternalRequest(
        model="gemini-2.0-flash",
        conversation=ConversationContext(
            messages=[
                Message(role="user", content=[TextBlock(text="What's the weather?")]),
                Message(
                    role="tool",
                    content=[
                        ToolResultBlock(
                            tool_use_id="call_1",
                            content='{"temperature": 72, "unit": "F"}',
                        )
                    ],
                ),
            ]
        ),
        params=GenerationParams(),
    )

    body = serializer.build_provider_request(request)
    func_resp = body["contents"][1]["parts"][0]["functionResponse"]
    assert func_resp["name"] == "call_1"
    assert func_resp["response"]["temperature"] == 72
    assert func_resp["response"]["unit"] == "F"


class TestSanitizeGeminiSchema:
    """Tests for _sanitize_gemini_schema recursive keyword stripping."""

    def test_strips_additional_properties(self):
        result = GeminiProviderSerializer._sanitize_gemini_schema(
            {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "additionalProperties": False,
            }
        )
        assert "additionalProperties" not in result
        assert result["type"] == "object"
        assert result["properties"]["name"]["type"] == "string"

    def test_strips_definitions(self):
        result = GeminiProviderSerializer._sanitize_gemini_schema(
            {
                "type": "object",
                "definitions": {"Foo": {"type": "string"}},
                "properties": {"x": {"type": "integer"}},
            }
        )
        assert "definitions" not in result
        assert "properties" in result

    def test_strips_dollar_ref(self):
        result = GeminiProviderSerializer._sanitize_gemini_schema(
            {
                "type": "object",
                "properties": {"x": {"$ref": "#/definitions/Foo"}},
            }
        )
        assert "$ref" not in result["properties"]["x"]

    def test_strips_const(self):
        result = GeminiProviderSerializer._sanitize_gemini_schema(
            {
                "type": "object",
                "properties": {"status": {"const": "active", "type": "string"}},
            }
        )
        assert "const" not in result["properties"]["status"]
        assert result["properties"]["status"]["type"] == "string"

    def test_strips_all_of_and_one_of(self):
        """allOf and oneOf are not in Gemini's Schema definition."""
        result = GeminiProviderSerializer._sanitize_gemini_schema(
            {
                "type": "object",
                "allOf": [
                    {"$ref": "#/definitions/Base"},
                    {"properties": {"extra": {"type": "string"}}},
                ],
                "oneOf": [{"type": "string"}, {"type": "number"}],
            }
        )
        assert "allOf" not in result
        assert "oneOf" not in result

    def test_strips_encrypted_reasoning_blob_keys(self):
        """Codex embeds `encrypted`/`encrypted_content` (reasoning blobs) in tool
        parameter schemas; Gemini rejects unknown proto fields, so they must be
        stripped at every nesting level."""
        result = GeminiProviderSerializer._sanitize_gemini_schema(
            {
                "type": "object",
                "properties": {
                    "first": {
                        "type": "string",
                        "encrypted": "opaque-blob",
                        "description": "kept",
                    },
                    "nested": {
                        "type": "object",
                        "encrypted_content": "opaque-blob-2",
                        "properties": {"x": {"type": "integer", "encrypted": True}},
                    },
                },
            }
        )
        assert "encrypted" not in result["properties"]["first"]
        assert result["properties"]["first"]["description"] == "kept"
        assert "encrypted_content" not in result["properties"]["nested"]
        assert "encrypted" not in result["properties"]["nested"]["properties"]["x"]

    def test_any_of_is_preserved(self):
        """anyOf IS in Gemini's Schema definition."""
        result = GeminiProviderSerializer._sanitize_gemini_schema(
            {
                "type": "object",
                "anyOf": [{"type": "string"}, {"type": "number"}],
            }
        )
        assert "anyOf" in result
        assert len(result["anyOf"]) == 2

    def test_any_of_nested_unsupported_keywords_stripped(self):
        """Unsupported keywords inside anyOf alternatives are recursively stripped."""
        result = GeminiProviderSerializer._sanitize_gemini_schema(
            {
                "type": "object",
                "anyOf": [
                    {"type": "string", "const": "x", "minLength": 1},
                    {"type": "number", "minimum": 0, "default": 5},
                ],
            }
        )
        # minLength/minimum/default are documented Schema fields — kept.
        assert result["anyOf"][0] == {"type": "string", "minLength": 1}
        assert result["anyOf"][1] == {"type": "number", "minimum": 0, "default": 5}

    def test_keeps_documented_schema_constraint_keywords(self):
        """Constraint keywords documented in the Gemini Schema are preserved."""
        result = GeminiProviderSerializer._sanitize_gemini_schema(
            {
                "type": "object",
                "$id": "https://example.com/schema",
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "title": "My Schema",
                "default": None,
                "examples": ["x"],
                "properties": {
                    "name": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 100,
                        "pattern": "^[a-z]+$",
                    },
                    "count": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 999,
                    },
                    "tags": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 10,
                        "items": {"type": "string"},
                    },
                },
            }
        )
        # Still stripped: JSON Schema vocabulary absent from the Gemini Schema.
        assert "$id" not in result
        assert "$schema" not in result
        assert "examples" not in result
        # Documented Gemini Schema fields are preserved verbatim.
        assert result["title"] == "My Schema"
        assert result["default"] is None
        name = result["properties"]["name"]
        assert name["minLength"] == 1
        assert name["maxLength"] == 100
        assert name["pattern"] == "^[a-z]+$"
        count = result["properties"]["count"]
        assert count["minimum"] == 0
        assert count["maximum"] == 999
        tags = result["properties"]["tags"]
        assert tags["minItems"] == 1
        assert tags["maxItems"] == 10

    def test_none_schema_returns_empty_dict(self):
        """None input is handled defensively."""
        result = GeminiProviderSerializer._sanitize_gemini_schema(None)
        assert result == {}

    def test_recursive_in_nested_properties(self):
        result = GeminiProviderSerializer._sanitize_gemini_schema(
            {
                "type": "object",
                "properties": {
                    "nested": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "deep": {
                                "type": "string",
                                "const": "fixed",
                            }
                        },
                    }
                },
            }
        )
        nested = result["properties"]["nested"]
        assert "additionalProperties" not in nested
        assert "const" not in nested["properties"]["deep"]

    def test_build_request_strips_unsupported_schema_keywords(self, serializer):
        """Integration test: tool parameters with unsupported keywords are cleaned."""
        request = InternalRequest(
            model="gemini-2.0-flash",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Test")])]
            ),
            params=GenerationParams(),
            tools=[
                FunctionTool(
                    name="test_func",
                    description="A test function",
                    parameters={
                        "type": "object",
                        "additionalProperties": False,
                        "definitions": {"Foo": {"type": "string"}},
                        "properties": {
                            "url": {"type": "string", "const": "https://example.com"},
                            "nested": {
                                "type": "object",
                                "$ref": "#/definitions/Foo",
                                "additionalProperties": True,
                                "allOf": [{"$ref": "#/definitions/Foo"}],
                                "oneOf": [{"type": "string"}, {"type": "number"}],
                                "not": {"type": "null"},
                                "patternProperties": {"^S_": {"type": "string"}},
                                "propertyNames": {"maxLength": 64},
                            },
                        },
                    },
                )
            ],
        )

        body = serializer.build_provider_request(request)
        params = body["tools"][0]["function_declarations"][0]["parameters"]
        assert "additionalProperties" not in params
        assert "definitions" not in params
        nested = params["properties"]["nested"]
        assert "$ref" not in nested
        assert "allOf" not in nested
        assert "oneOf" not in nested
        assert "not" not in nested
        assert "patternProperties" not in nested
        assert "propertyNames" not in nested
        assert "const" not in params["properties"]["url"]

    def test_filters_required_to_match_properties(self):
        """required entries that do not appear in properties are removed."""
        result = GeminiProviderSerializer._sanitize_gemini_schema(
            {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name", "missing_field"],
            }
        )
        assert result["required"] == ["name"]
        assert "missing_field" not in result.get("required", [])

    def test_filters_required_in_nested_objects(self):
        """required filtering works recursively inside nested object properties."""
        result = GeminiProviderSerializer._sanitize_gemini_schema(
            {
                "type": "object",
                "properties": {
                    "outer": {
                        "type": "object",
                        "properties": {"inner": {"type": "string"}},
                        "required": ["inner", "ghost"],
                    }
                },
                "required": ["outer"],
            }
        )
        nested = result["properties"]["outer"]
        assert nested["required"] == ["inner"]
        assert "ghost" not in nested.get("required", [])

    def test_removes_empty_required_list(self):
        """If all required entries are stripped, the required key is dropped."""
        result = GeminiProviderSerializer._sanitize_gemini_schema(
            {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["unknown"],
            }
        )
        assert "required" not in result

    def test_system_role_message_converts_to_user_xml(self, serializer):
        """Message(role='system') in conversation.messages must become
        role='user' with <system-prompt> XML wrapping in Gemini format.
        (Matches Anthropic/OpenAI provider degradation logic.)"""
        request = InternalRequest(
            model="gemini-2.0-flash",
            conversation=ConversationContext(
                messages=[
                    Message(role="user", content=[TextBlock(text="Hello")]),
                    Message(
                        role="system",
                        content=[TextBlock(text="Speak only French.")],
                    ),
                    Message(role="user", content=[TextBlock(text="How are you?")]),
                ]
            ),
            params=GenerationParams(),
        )

        body = serializer.build_provider_request(request)
        contents = body["contents"]

        assert len(contents) == 3
        sys_msg = contents[1]
        assert sys_msg["role"] == "user", f"Expected 'user', got '{sys_msg['role']}'"
        assert len(sys_msg["parts"]) == 1
        assert "text" in sys_msg["parts"][0]
        assert "<system-prompt>" in sys_msg["parts"][0]["text"]
        assert "Speak only French." in sys_msg["parts"][0]["text"]
        assert "</system-prompt>" in sys_msg["parts"][0]["text"]


class TestGeminiResponseDetails:
    """Regression tests for Gemini response detail preservation."""

    def test_safety_ratings_in_provider_info(self, serializer):
        response = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "Hello"}]},
                    "finishReason": "STOP",
                    "safetyRatings": [
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "probability": "NEGLIGIBLE"}
                    ],
                }
            ],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
        }
        result = serializer.parse_provider_response(response)
        assert result.provider_info["safety_ratings"] == [
            {"category": "HARM_CATEGORY_HATE_SPEECH", "probability": "NEGLIGIBLE"}
        ]

    def test_citation_grounding_metadata_mapped_to_annotations(self, serializer):
        response = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "According to source..."}]},
                    "finishReason": "STOP",
                    "citationMetadata": {
                        "citationSources": [
                            {"startIndex": 0, "endIndex": 10, "uri": "https://example.com"}
                        ]
                    },
                    "groundingMetadata": {
                        "groundingChunks": [
                            {"web": {"uri": "https://google.com", "title": "Google"}}
                        ]
                    },
                }
            ],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
        }
        result = serializer.parse_provider_response(response)
        assert "annotations" in result.provider_info
        assert len(result.provider_info["annotations"]) == 2
        assert result.output[0].citations is result.provider_info["annotations"]

    def test_no_image_finish_reason_maps_to_content_filter(self, serializer):
        from llm_proxy.models.finish_reasons import GEMINI_TO_OPENAI

        assert GEMINI_TO_OPENAI["NO_IMAGE"] == "content_filter"
        response = {
            "candidates": [
                {
                    "content": {"parts": []},
                    "finishReason": "NO_IMAGE",
                }
            ],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 0},
        }
        result = serializer.parse_provider_response(response)
        assert result.finish_reason == "content_filter"

    def test_tool_use_prompt_token_count_mapped(self, serializer):
        response = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "Used a tool."}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "toolUsePromptTokenCount": 3,
            },
        }
        result = serializer.parse_provider_response(response)
        assert result.usage is not None
        # Tool-use prompt tokens are billed at the input rate (only Google
        # Search grounding tokens are excluded), so they fold into input.
        assert result.usage.input_tokens == 13
        assert result.usage.output_tokens == 5


class TestGeminiStreamingDetails:
    """Regression tests for Gemini streaming detail preservation."""

    def test_streaming_safety_ratings(self):
        from llm_proxy.serialization.gemini.streaming_converter import GeminiStreamingTransformer

        transformer = GeminiStreamingTransformer(model="gemini-2.0-flash")
        chunk = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "Hello"}]},
                    "finishReason": "STOP",
                    "safetyRatings": [
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "probability": "NEGLIGIBLE"}
                    ],
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "toolUsePromptTokenCount": 3,
                "totalTokenCount": 18,
            },
        }
        result = transformer.convert_chunk(chunk)
        assert result["safety_ratings"] == [
            {"category": "HARM_CATEGORY_HATE_SPEECH", "probability": "NEGLIGIBLE"}
        ]
        # prompt_tokens includes tool-use prompt tokens (input-rate billing).
        assert result["usage"]["prompt_tokens"] == 13
        usage = transformer.get_usage()
        assert usage is not None
        assert usage.input_tokens == 13

    def test_streaming_annotations_from_grounding(self):
        from llm_proxy.serialization.gemini.streaming_converter import GeminiStreamingTransformer

        transformer = GeminiStreamingTransformer(model="gemini-2.0-flash")
        chunk = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "Source says hello"}]},
                    "finishReason": "STOP",
                    "groundingMetadata": {
                        "groundingChunks": [
                            {"web": {"uri": "https://example.com", "title": "Example"}}
                        ]
                    },
                }
            ],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
        }
        result = transformer.convert_chunk(chunk)
        assert "annotations" in result["choices"][0]["delta"]
        assert len(result["choices"][0]["delta"]["annotations"]) == 1


def test_build_provider_request_gemini3_injects_dummy_signature(serializer):
    """Gemini 3 requires thoughtSignature on every functionCall part.

    History from non-Gemini providers lacks signatures; Google's recommended
    dummy (base64 of "skip_thought_signature_validator") is injected so the
    request passes validation instead of failing with "missing
    thought_signature".
    """
    request = InternalRequest(
        model="gemini-3-pro-preview",
        conversation=ConversationContext(
            messages=[
                Message(role="user", content=[TextBlock(text="What's the weather?")]),
                Message(
                    role="assistant",
                    content=[
                        ToolUseBlock(
                            id="call_123",
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
    model_content = [c for c in body["contents"] if c["role"] == "model"]
    part = model_content[0]["parts"][0]
    assert part["thoughtSignature"] == "c2tpcF90aG91Z2h0X3NpZ25hdHVyZV92YWxpZGF0b3I="


def test_build_provider_request_gemini3_preserves_real_signature(serializer):
    """Real signatures (from Gemini responses, re-attached by the adapter
    cache) are preserved verbatim; only missing ones get the dummy."""
    request = InternalRequest(
        model="gemini-3-pro-preview",
        conversation=ConversationContext(
            messages=[
                Message(role="user", content=[TextBlock(text="What's the weather?")]),
                Message(
                    role="assistant",
                    content=[
                        ToolUseBlock(
                            id="call_123",
                            name="get_weather",
                            input={"location": "Boston"},
                            extra={"thought_signature": "REAL_SIG_abc"},
                        )
                    ],
                ),
            ]
        ),
        params=GenerationParams(),
    )

    body = serializer.build_provider_request(request)
    model_content = [c for c in body["contents"] if c["role"] == "model"]
    part = model_content[0]["parts"][0]
    assert part["thoughtSignature"] == "REAL_SIG_abc"


def test_build_provider_request_gemini20_keeps_no_signature(serializer):
    """gemini-2.0 does not support thought signatures: no dummy is injected
    and the part stays signature-free (previous behavior)."""
    request = InternalRequest(
        model="gemini-2.0-flash",
        conversation=ConversationContext(
            messages=[
                Message(role="user", content=[TextBlock(text="What's the weather?")]),
                Message(
                    role="assistant",
                    content=[
                        ToolUseBlock(
                            id="call_123",
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
    model_content = [c for c in body["contents"] if c["role"] == "model"]
    part = model_content[0]["parts"][0]
    assert "thoughtSignature" not in part


def test_streaming_cached_tokens_mapped_once():
    """Regression: Gemini cached tokens must map to exactly ONE billing field.

    The streaming converter used to set both cache_read_input_tokens and
    prompt_tokens_details.cached_tokens for the same cachedContentTokenCount,
    making the cost calculator apply the cache-rate adjustment twice.
    """
    from llm_proxy.serialization.gemini.streaming_converter import GeminiStreamingTransformer

    transformer = GeminiStreamingTransformer(model="gemini-2.0-flash")
    chunk = {
        "candidates": [{"content": {"parts": [{"text": "Hello"}]}, "finishReason": "STOP"}],
        "usageMetadata": {
            "promptTokenCount": 1010,
            "candidatesTokenCount": 5,
            "cachedContentTokenCount": 1000,
            "totalTokenCount": 1015,
        },
    }
    transformer.convert_chunk(chunk)
    usage = transformer.get_usage()
    assert usage is not None
    assert usage.input_tokens == 1010
    assert usage.output_tokens == 5
    assert usage.cache_read_input_tokens == 1000
    assert usage.prompt_tokens_details is None


def test_streaming_thoughts_added_when_candidates_excludes_them():
    """When prompt + candidates + toolUse != total, candidatesTokenCount
    excludes thinking tokens, so they are added to the output side."""
    from llm_proxy.serialization.gemini.streaming_converter import GeminiStreamingTransformer

    transformer = GeminiStreamingTransformer(model="gemini-2.5-pro")
    chunk = {
        "candidates": [{"content": {"parts": [{"text": "Hello"}]}, "finishReason": "STOP"}],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "thoughtsTokenCount": 20,
            "totalTokenCount": 35,
        },
    }
    transformer.convert_chunk(chunk)
    usage = transformer.get_usage()
    assert usage is not None
    assert usage.output_tokens == 25


def test_search_grounding_excludes_tool_use_tokens_from_input():
    """Google-Search-grounded tool-use prompt tokens are billed via the
    search fee, not the input rate."""
    from llm_proxy.serialization.gemini.response_parser import (
        GeminiResponseParserMixin,
    )

    class _Parser(GeminiResponseParserMixin):
        pass

    parser = _Parser()
    response = {
        "candidates": [
            {
                "content": {"parts": [{"text": " grounded answer"}]},
                "finishReason": "STOP",
                "groundingMetadata": {"webSearchQueries": ["query-1"]},
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "toolUsePromptTokenCount": 3,
            "totalTokenCount": 18,
        },
    }
    result = parser.parse_provider_response(response, model="gemini-2.0-flash")
    assert result.usage is not None
    assert result.usage.input_tokens == 10
    assert result.usage.web_search_requests == 1
