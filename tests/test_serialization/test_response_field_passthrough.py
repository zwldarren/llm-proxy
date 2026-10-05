"""Upstream response-level fields survive the FULL_CONVERSION tier.

The native/wire-reuse tiers forward the raw upstream body, so fields like
``moderation`` and ``prompt_cache_diagnostics`` reach the client untouched. The
full-conversion path rebuilds the response from ``InternalResponse`` and used to
drop them; these tests pin the capture (provider serializer) and echo
(protocol formatter) on both Chat Completions and OpenResponses.
"""

from llm_proxy.models import InternalResponse, TextBlock
from llm_proxy.protocols.openai.serializer import OpenAIProtocolSerializer
from llm_proxy.protocols.registry import get_protocol_serializer
from llm_proxy.serialization.openai.components.response_parser import OpenAIResponseParser

MODERATION = {
    "input": {"type": "moderation_results", "model": "omni-moderation-latest", "results": []},
    "output": {"type": "moderation_results", "model": "omni-moderation-latest", "results": []},
}
DIAGNOSTICS = {"type": "cache_diagnostics", "reason": "cold", "cache_missed_tokens": 12}


# ---------------------------------------------------------------------------
# Chat Completions
# ---------------------------------------------------------------------------


def _chat_response_body(**extra):
    return {
        "id": "chatcmpl_1",
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
            }
        ],
        **extra,
    }


def test_chat_parser_captures_moderation():
    parsed = OpenAIResponseParser().parse(_chat_response_body(moderation=MODERATION))
    assert parsed.provider_info["moderation"] == MODERATION


def test_chat_parser_ignores_absent_moderation():
    parsed = OpenAIResponseParser().parse(_chat_response_body())
    assert "moderation" not in parsed.provider_info


def test_chat_moderation_is_known_field_not_unknown():
    # Captured explicitly, so the unknown-field extractor must not duplicate it.
    assert "moderation" in OpenAIResponseParser.known_response_fields()


def test_chat_formatter_echoes_moderation():
    response = InternalResponse(
        id="chatcmpl_1",
        model="gpt-4",
        output=[TextBlock(text="hi")],
        provider_info={"moderation": MODERATION},
    )
    result = OpenAIProtocolSerializer().format_response(response)
    assert result["moderation"] == MODERATION


def test_chat_formatter_omits_moderation_when_absent():
    response = InternalResponse(id="chatcmpl_2", model="gpt-4", output=[TextBlock(text="hi")])
    assert "moderation" not in OpenAIProtocolSerializer().format_response(response)


# ---------------------------------------------------------------------------
# OpenResponses
# ---------------------------------------------------------------------------


def test_responses_parser_captures_passthrough_fields():
    from llm_proxy.serialization.openai.serializer import OpenAIResponsesProviderSerializer

    parsed = OpenAIResponsesProviderSerializer().parse_provider_response(
        {
            "id": "resp_1",
            "model": "gpt-6-luna",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hi"}],
                }
            ],
            "moderation": MODERATION,
            "prompt_cache_diagnostics": DIAGNOSTICS,
        }
    )
    assert parsed.provider_info["moderation"] == MODERATION
    assert parsed.provider_info["prompt_cache_diagnostics"] == DIAGNOSTICS


def test_responses_formatter_echoes_provider_info_fields():
    response = InternalResponse(
        id="resp_1",
        model="gpt-6-luna",
        output=[TextBlock(text="hi")],
        provider_info={"moderation": MODERATION, "prompt_cache_diagnostics": DIAGNOSTICS},
    )
    result = get_protocol_serializer("openresponses").format_response(response)
    assert result["moderation"] == MODERATION
    assert result["prompt_cache_diagnostics"] == DIAGNOSTICS


def test_responses_formatter_omits_absent_fields():
    response = InternalResponse(id="resp_2", model="gpt-6-luna", output=[TextBlock(text="hi")])
    result = get_protocol_serializer("openresponses").format_response(response)
    assert "moderation" not in result
    assert "prompt_cache_diagnostics" not in result
