"""Additional tests for Gemini-specific serialization and adapter features.

Covers safety settings, service tier mapping, cached content,
response modalities, image model detection, and thinking config variants.
"""

import pytest

from llm_proxy.models import (
    ConversationContext,
    GeminiSpecificParams,
    GenerationParams,
    InternalRequest,
    Message,
    OpenAISpecificParams,
    TextBlock,
    ThinkingConfig,
)
from llm_proxy.serialization.gemini.request_builder import (
    GeminiRequestBuilderMixin,
)

# ---------------------------------------------------------------------------
# Fixtures — a concrete builder for testing GeminiRequestBuilderMixin
# ---------------------------------------------------------------------------


class _ConcreteGeminiBuilder(GeminiRequestBuilderMixin):
    """Concrete implementation of the Gemini builder mixin for testing."""

    def _convert_conversation_to_gemini(self, conversation, context):
        """Minimal conversion: each message becomes a content part."""
        contents = []
        system_instruction = None
        for msg in conversation.messages:
            if msg.role == "system":
                system_instruction = (
                    msg.content[0].text
                    if msg.content and hasattr(msg.content[0], "text")
                    else str(msg.content)
                )
            else:
                text = ""
                for block in msg.content:
                    if hasattr(block, "text"):
                        text += block.text
                role = "model" if msg.role == "assistant" else "user"
                contents.append({"role": role, "parts": [{"text": text}] if text else []})
        return contents, system_instruction

    def _convert_tools_to_gemini(self, tools):
        return None

    def _build_tool_config(self, request):
        return None


@pytest.fixture
def builder():
    return _ConcreteGeminiBuilder()


@pytest.fixture
def basic_request():
    return InternalRequest(
        model="gemini-2.0-flash",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="Hello")])]
        ),
    )


# ---------------------------------------------------------------------------
# Safety settings
# ---------------------------------------------------------------------------


class TestSafetySettings:
    """Gemini safety settings are forwarded from GeminiSpecificParams."""

    def test_safety_settings_forwarded(self, builder, basic_request):
        """Safety settings appear in the request body."""
        import copy

        from llm_proxy.serialization.context import BuildContext

        request = copy.deepcopy(basic_request)
        request.params.gemini = GeminiSpecificParams(
            safety_settings=[
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                },
            ]
        )
        body = builder._build_provider_request(
            request, BuildContext.from_request(request, base_url="https://test.example.com")
        )
        assert "safetySettings" in body
        assert len(body["safetySettings"]) == 2
        assert body["safetySettings"][0]["category"] == "HARM_CATEGORY_HATE_SPEECH"
        assert body["safetySettings"][0]["threshold"] == "BLOCK_ONLY_HIGH"

    def test_no_safety_settings_when_not_configured(self, builder, basic_request):
        """When no safety settings are provided, the key is absent."""
        import copy

        from llm_proxy.serialization.context import BuildContext

        request = copy.deepcopy(basic_request)
        body = builder._build_provider_request(
            request, BuildContext.from_request(request, base_url="https://test.example.com")
        )
        assert "safetySettings" not in body


# ---------------------------------------------------------------------------
# Service tier mapping
# ---------------------------------------------------------------------------


class TestServiceTierMapping:
    """OpenAI service_tier → Gemini serviceTier mapping."""

    def test_default_maps_to_standard(self, builder, basic_request):
        import copy

        from llm_proxy.serialization.context import BuildContext

        request = copy.deepcopy(basic_request)
        request.params.openai = OpenAISpecificParams(service_tier="default")
        body = builder._build_provider_request(
            request, BuildContext.from_request(request, base_url="https://test.example.com")
        )
        assert body["serviceTier"] == "standard"

    def test_auto_is_omitted(self, builder, basic_request):
        import copy

        from llm_proxy.serialization.context import BuildContext

        request = copy.deepcopy(basic_request)
        request.params.openai = OpenAISpecificParams(service_tier="auto")
        body = builder._build_provider_request(
            request, BuildContext.from_request(request, base_url="https://test.example.com")
        )
        assert "serviceTier" not in body

    def test_flex_passthrough(self, builder, basic_request):
        import copy

        from llm_proxy.serialization.context import BuildContext

        request = copy.deepcopy(basic_request)
        request.params.openai = OpenAISpecificParams(service_tier="flex")
        body = builder._build_provider_request(
            request, BuildContext.from_request(request, base_url="https://test.example.com")
        )
        assert body["serviceTier"] == "flex"

    def test_priority_passthrough(self, builder, basic_request):
        import copy

        from llm_proxy.serialization.context import BuildContext

        request = copy.deepcopy(basic_request)
        request.params.openai = OpenAISpecificParams(service_tier="priority")
        body = builder._build_provider_request(
            request, BuildContext.from_request(request, base_url="https://test.example.com")
        )
        assert body["serviceTier"] == "priority"

    def test_no_service_tier_when_not_set(self, builder, basic_request):
        import copy

        from llm_proxy.serialization.context import BuildContext

        request = copy.deepcopy(basic_request)
        body = builder._build_provider_request(
            request, BuildContext.from_request(request, base_url="https://test.example.com")
        )
        assert "serviceTier" not in body


# ---------------------------------------------------------------------------
# Cached content
# ---------------------------------------------------------------------------


class TestCachedContent:
    """Gemini cachedContent forwarding."""

    def test_cached_content_forwarded(self, builder, basic_request):
        import copy

        from llm_proxy.serialization.context import BuildContext

        request = copy.deepcopy(basic_request)
        request.params.gemini = GeminiSpecificParams(cached_content="cached-content-name-123")
        body = builder._build_provider_request(
            request, BuildContext.from_request(request, base_url="https://test.example.com")
        )
        assert body["cachedContent"] == "cached-content-name-123"

    def test_no_cached_content_when_not_set(self, builder, basic_request):
        import copy

        from llm_proxy.serialization.context import BuildContext

        request = copy.deepcopy(basic_request)
        body = builder._build_provider_request(
            request, BuildContext.from_request(request, base_url="https://test.example.com")
        )
        assert "cachedContent" not in body


# ---------------------------------------------------------------------------
# Response modalities
# ---------------------------------------------------------------------------


class TestResponseModalities:
    """Gemini responseModalities parameter."""

    def test_response_modalities_from_gemini_params(self, builder, basic_request):
        import copy

        from llm_proxy.serialization.context import BuildContext

        request = copy.deepcopy(basic_request)
        request.params.gemini = GeminiSpecificParams(response_modalities=["TEXT", "AUDIO"])
        body = builder._build_provider_request(
            request, BuildContext.from_request(request, base_url="https://test.example.com")
        )
        assert "generationConfig" in body
        assert body["generationConfig"]["responseModalities"] == ["TEXT", "AUDIO"]

    def test_image_models_auto_inject_image_modality(self, builder):
        """Gemini image models auto-inject ``responseModalities: [IMAGE]``."""
        from llm_proxy.serialization.context import BuildContext

        request = InternalRequest(
            model="gemini-2.0-flash-exp-image",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Draw a cat")])]
            ),
        )
        body = builder._build_provider_request(
            request, BuildContext.from_request(request, base_url="https://test.example.com")
        )
        assert "generationConfig" in body
        assert body["generationConfig"]["responseModalities"] == ["IMAGE"]

    def test_image_preview_model_auto_injects(self, builder):
        """``-image-preview`` suffix models also auto-inject IMAGE modality."""
        from llm_proxy.serialization.context import BuildContext

        request = InternalRequest(
            model="gemini-2.0-flash-exp-image-preview",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Generate")])]
            ),
        )
        body = builder._build_provider_request(
            request, BuildContext.from_request(request, base_url="https://test.example.com")
        )
        assert body["generationConfig"]["responseModalities"] == ["IMAGE"]

    def test_non_image_model_no_auto_inject(self, builder, basic_request):
        """Regular text models do not auto-inject IMAGE modality."""
        import copy

        from llm_proxy.serialization.context import BuildContext

        request = copy.deepcopy(basic_request)
        request.model = "gemini-2.0-flash"
        body = builder._build_provider_request(
            request, BuildContext.from_request(request, base_url="https://test.example.com")
        )
        if "generationConfig" in body:
            assert "responseModalities" not in body["generationConfig"]


# ---------------------------------------------------------------------------
# Thinking config — Gemini 2.5 vs 3+
# ---------------------------------------------------------------------------


class TestThinkingConfig:
    """Gemini thinking config varies between Gemini 2.5 (legacy) and 3+."""

    def test_gemini_25_uses_legacy_thinking_budget(self, builder):
        """Gemini 2.5 uses legacy thinkingBudget format."""
        from llm_proxy.serialization.context import BuildContext

        request = InternalRequest(
            model="gemini-2.5-flash",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Think about this")])]
            ),
            params=GenerationParams(thinking=ThinkingConfig(type="enabled", budget_tokens=8192)),
        )
        body = builder._build_provider_request(
            request, BuildContext.from_request(request, base_url="https://test.example.com")
        )
        assert "generationConfig" in body
        tc = body["generationConfig"]["thinkingConfig"]
        assert tc["thinkingBudget"] == 8192
        assert tc["includeThoughts"]

    def test_gemini_3_uses_thinking_level(self, builder):
        """Gemini 3 uses thinkingLevel format with LOW/MEDIUM/HIGH."""
        from llm_proxy.serialization.context import BuildContext

        request = InternalRequest(
            model="gemini-3.0-pro",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Complex problem")])]
            ),
            params=GenerationParams(thinking=ThinkingConfig(type="enabled", budget_tokens=32000)),
        )
        body = builder._build_provider_request(
            request, BuildContext.from_request(request, base_url="https://test.example.com")
        )
        assert "generationConfig" in body
        tc = body["generationConfig"]["thinkingConfig"]
        assert tc["thinkingLevel"] == "HIGH"

    def test_gemini_3_disabled_thinking_no_config(self, builder):
        """Disabled thinking omits thinkingConfig entirely for Gemini 3."""
        from llm_proxy.serialization.context import BuildContext

        request = InternalRequest(
            model="gemini-3.0-pro",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Quick fact")])]
            ),
            params=GenerationParams(thinking=ThinkingConfig(type="disabled")),
        )
        body = builder._build_provider_request(
            request, BuildContext.from_request(request, base_url="https://test.example.com")
        )
        if "generationConfig" in body:
            assert "thinkingConfig" not in body["generationConfig"]

    def test_gemini_3_effort_from_openai_reasoning_effort_when_thinking_unset(self, builder):
        """Regression: effort sources other than ``thinking`` must still reach Gemini.

        Previously Gemini read only ``request.params.thinking`` and silently dropped
        effort set via ``openai.reasoning_effort`` (or ``anthropic.output_config``)
        when ``thinking`` was unset — the classic "forgotten provider" case that
        unified ``resolve_thinking`` now covers.
        """
        from llm_proxy.models.params import OpenAISpecificParams
        from llm_proxy.serialization.context import BuildContext

        request = InternalRequest(
            model="gemini-3.0-pro",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Complex problem")])]
            ),
            params=GenerationParams(openai=OpenAISpecificParams(reasoning_effort="high")),
        )
        body = builder._build_provider_request(
            request, BuildContext.from_request(request, base_url="https://test.example.com")
        )
        assert "generationConfig" in body
        assert body["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "HIGH"


# ---------------------------------------------------------------------------
# responseSchema forwarding
# ---------------------------------------------------------------------------


class TestResponseSchema:
    """Gemini responseSchema parameter."""

    def test_response_schema_from_gemini_params(self, builder, basic_request):
        import copy

        from llm_proxy.serialization.context import BuildContext

        request = copy.deepcopy(basic_request)
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        request.params.gemini = GeminiSpecificParams(response_schema=schema)
        body = builder._build_provider_request(
            request, BuildContext.from_request(request, base_url="https://test.example.com")
        )
        assert "generationConfig" in body
        assert body["generationConfig"]["responseSchema"] == schema

    def test_response_schema_unwraps_openai_wrapper(self, builder, basic_request):
        """The OpenAI wrapper {name, description, schema, strict} is unwrapped
        before sanitization so responseSchema carries the plain JSON Schema."""
        import copy

        from llm_proxy.models.types import ResponseFormat
        from llm_proxy.serialization.context import BuildContext

        request = copy.deepcopy(basic_request)
        plain_schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        request.params.response_format = ResponseFormat(
            type="json_schema",
            json_schema={
                "name": "person",
                "description": "A person",
                "schema": plain_schema,
                "strict": True,
            },
        )
        body = builder._build_provider_request(
            request, BuildContext.from_request(request, base_url="https://test.example.com")
        )
        assert body["generationConfig"]["responseMimeType"] == "application/json"
        assert body["generationConfig"]["responseSchema"] == plain_schema


# ---------------------------------------------------------------------------
# Extra field filtering
# ---------------------------------------------------------------------------


class TestExtraFieldFiltering:
    """Gemini-invalid extra keys are stripped even with passthrough policy."""

    def test_reasoning_extra_is_filtered(self, builder, basic_request):
        import copy

        from llm_proxy.serialization.context import BuildContext

        request = copy.deepcopy(basic_request)
        request.extra = {"reasoning": {"effort": "high"}, "custom_param": "value"}
        body = builder._build_provider_request(
            request, BuildContext.from_request(request, base_url="https://test.example.com")
        )
        # reasoning is blocked; custom_param passes through
        assert "reasoning" not in body
        assert body["custom_param"] == "value"

    def test_previous_response_id_is_filtered(self, builder, basic_request):
        import copy

        from llm_proxy.serialization.context import BuildContext

        request = copy.deepcopy(basic_request)
        request.extra = {"previous_response_id": "resp_abc"}
        body = builder._build_provider_request(
            request, BuildContext.from_request(request, base_url="https://test.example.com")
        )
        assert "previous_response_id" not in body


# ---------------------------------------------------------------------------
# generationConfig from gemini_params.generation_config
# ---------------------------------------------------------------------------


class TestGenerationConfigPassthrough:
    """Gemini generationConfig raw dict passthrough."""

    def test_generation_config_merged(self, builder, basic_request):
        import copy

        from llm_proxy.serialization.context import BuildContext

        request = copy.deepcopy(basic_request)
        request.params.gemini = GeminiSpecificParams(
            generation_config={"imageConfig": {"aspectRatio": "16:9", "imageSize": "2K"}}
        )
        body = builder._build_provider_request(
            request, BuildContext.from_request(request, base_url="https://test.example.com")
        )
        assert "generationConfig" in body
        assert body["generationConfig"]["imageConfig"]["aspectRatio"] == "16:9"
        assert body["generationConfig"]["imageConfig"]["imageSize"] == "2K"

    def test_generation_config_speech_config(self, builder, basic_request):
        import copy

        from llm_proxy.serialization.context import BuildContext

        request = copy.deepcopy(basic_request)
        request.params.gemini = GeminiSpecificParams(
            generation_config={
                "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Zephyr"}}}
            }
        )
        body = builder._build_provider_request(
            request, BuildContext.from_request(request, base_url="https://test.example.com")
        )
        assert "generationConfig" in body
        sc = body["generationConfig"]["speechConfig"]
        assert sc["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == "Zephyr"
