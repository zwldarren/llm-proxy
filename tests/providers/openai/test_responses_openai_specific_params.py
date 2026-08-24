"""Regression tests for OpenAI Responses API mapping of ``OpenAISpecificParams``.

The Responses-API body builder in ``providers/openai/serializer.py`` previously
read only ``temperature``/``top_p``/``max_tokens``/``stop`` plus the ``extra``
passthrough, silently dropping every field parsed into ``OpenAISpecificParams``
(``store``, ``metadata``, ``service_tier``, ``safety_identifier``,
``prompt_cache_key``, ``logprobs``/``top_logprobs``, ``max_completion_tokens``,
``reasoning_effort``) as well as ``response_format`` (Structured Outputs) and the
deprecated ``user``. These tests pin the mappings that the Responses API
actually accepts; unsupported fields remain dropped.
"""

from llm_proxy.models import ConversationContext, InternalRequest, Message, TextBlock
from llm_proxy.models.params import GenerationParams, OpenAISpecificParams
from llm_proxy.models.types import ResponseFormat
from llm_proxy.protocols.registry import get_protocol_serializer
from llm_proxy.serialization import get_provider_serializer
from llm_proxy.serialization.context import BuildContext
from llm_proxy.serialization.openai.serializer import OpenAIResponsesProviderSerializer

serializer = OpenAIResponsesProviderSerializer()


def _ctx(model: str = "gpt-5") -> BuildContext:
    return BuildContext(
        provider_name="openai",
        model=model,
        target_endpoint="responses",
        supported_content_blocks=serializer.supported_content_blocks,
    )


def _build(params: GenerationParams, *, metadata_user: str | None = None) -> dict:
    request = InternalRequest(
        model="gpt-5",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="hi")])]
        ),
        params=params,
    )
    if metadata_user is not None:
        request.metadata.user = metadata_user
    return serializer._build_provider_request(request, _ctx())


class TestResponseFormatMapping:
    def test_json_object(self):
        body = _build(GenerationParams(response_format=ResponseFormat(type="json_object")))
        assert body["text"] == {"format": {"type": "json_object"}}

    def test_json_schema_full_wrapper(self):
        js = {"name": "ans", "description": "d", "schema": {"type": "object"}, "strict": True}
        body = _build(
            GenerationParams(response_format=ResponseFormat(type="json_schema", json_schema=js))
        )
        assert body["text"]["format"] == {
            "type": "json_schema",
            "schema": {"type": "object"},
            "name": "ans",
            "description": "d",
            "strict": True,
        }

    def test_text_is_default_and_omitted(self):
        body = _build(GenerationParams(response_format=ResponseFormat(type="text")))
        assert "text" not in body

    def test_none_response_format_omitted(self):
        body = _build(GenerationParams())
        assert "text" not in body


class TestOpenAISpecificParamsMapping:
    def test_store_metadata_service_tier_safety_identifier_prompt_cache_key(self):
        body = _build(
            GenerationParams(
                openai=OpenAISpecificParams(
                    store=True,
                    metadata={"k": "v"},
                    service_tier="flex",
                    safety_identifier="sid",
                    prompt_cache_key="pck",
                )
            )
        )
        assert body["store"] is True
        assert body["metadata"] == {"k": "v"}
        assert body["service_tier"] == "flex"
        assert body["safety_identifier"] == "sid"
        assert body["prompt_cache_key"] == "pck"

    def test_top_logprobs_emits_top_logprobs_and_include(self):
        body = _build(GenerationParams(openai=OpenAISpecificParams(top_logprobs=3, logprobs=True)))
        assert body["top_logprobs"] == 3
        assert body["include"] == ["message.output_text.logprobs"]

    def test_logprobs_without_top_logprobs_only_adds_include(self):
        body = _build(GenerationParams(openai=OpenAISpecificParams(logprobs=True)))
        assert "top_logprobs" not in body
        assert body["include"] == ["message.output_text.logprobs"]

    def test_include_merged_with_existing_extra_include(self):
        request = InternalRequest(
            model="gpt-5",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            params=GenerationParams(openai=OpenAISpecificParams(top_logprobs=2, logprobs=True)),
            extra={"include": ["reasoning.encrypted_content"]},
        )
        body = serializer._build_provider_request(request, _ctx())
        assert body["include"] == [
            "reasoning.encrypted_content",
            "message.output_text.logprobs",
        ]

    def test_max_completion_tokens_falls_back_to_max_output_tokens(self):
        body = _build(GenerationParams(openai=OpenAISpecificParams(max_completion_tokens=256)))
        assert body["max_output_tokens"] == 256

    def test_max_tokens_takes_precedence_over_max_completion_tokens(self):
        body = _build(
            GenerationParams(
                max_tokens=100,
                openai=OpenAISpecificParams(max_completion_tokens=256),
            )
        )
        assert body["max_output_tokens"] == 100

    def test_reasoning_effort_fallback_when_no_thinking_config(self):
        body = _build(GenerationParams(openai=OpenAISpecificParams(reasoning_effort="high")))
        assert body["reasoning"] == {"effort": "high"}

    def test_user_mapped_from_metadata(self):
        body = _build(GenerationParams(), metadata_user="u-123")
        assert body["user"] == "u-123"


class TestUnsupportedFieldsDropped:
    """Fields the Responses API does not accept must stay dropped (not invented)."""

    def test_unsupported_dropped(self):
        body = _build(
            GenerationParams(
                seed=42,
                n=2,
                openai=OpenAISpecificParams(
                    logit_bias={"1": -1},
                    prediction={"type": "content"},
                    modalities=["text", "audio"],
                    audio={"voice": "alloy"},
                    prompt_cache_retention="30m",
                ),
            )
        )
        for absent in (
            "seed",
            "n",
            "logit_bias",
            "prediction",
            "modalities",
            "audio",
            "prompt_cache_retention",
        ):
            assert absent not in body, f"{absent} should be dropped"


def _chat_convert(protocol: str, provider: str, raw: dict) -> dict:
    """Parse a protocol request and build the provider request body."""
    proto = get_protocol_serializer(protocol)
    internal = proto.parse_request(raw)
    prov = get_provider_serializer(provider)
    target_endpoint = "responses" if provider in {"openai"} else "chat_completions"
    ctx = BuildContext.from_request(
        internal,
        provider_name=provider,
        target_endpoint=target_endpoint,
        unknown_fields_policy="ignore",
        unsupported_block_policy="drop",
        supported_content_blocks=prov.supported_content_blocks,
    )
    return prov.build_provider_request(internal, ctx)


class TestChatConvertHarness:
    """End-to-end via the chat protocol -> openai provider conversion."""

    def test_harness_body(self):
        b = _chat_convert(
            "openai",
            "openai",
            {
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "store": True,
                "metadata": {"k": "v"},
                "service_tier": "flex",
                "prompt_cache_key": "pck",
                "safety_identifier": "sid",
                "response_format": {"type": "json_object"},
                "top_logprobs": 3,
                "logprobs": True,
            },
        )
        assert b["store"] is True
        assert b["metadata"] == {"k": "v"}
        assert b["service_tier"] == "flex"
        assert b["safety_identifier"] == "sid"
        assert b["prompt_cache_key"] == "pck"
        assert b["text"] == {"format": {"type": "json_object"}}
        assert b["top_logprobs"] == 3
        assert b["include"] == ["message.output_text.logprobs"]
