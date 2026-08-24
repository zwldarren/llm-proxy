# tests/unit/core/test_finish_reasons.py
from llm_proxy.models.finish_reasons import (
    ANTHROPIC_TO_OPENAI,
    GEMINI_TO_OPENAI,
    OPENAI_TO_ANTHROPIC,
    FinishReason,
    map_finish_reason,
)


class TestFinishReasonEnum:
    def test_finish_reason_values(self):
        assert FinishReason.STOP.value == "stop"
        assert FinishReason.LENGTH.value == "length"
        assert FinishReason.TOOL_CALLS.value == "tool_calls"
        assert FinishReason.CONTENT_FILTER.value == "content_filter"

    def test_context_length(self):
        assert FinishReason.CONTEXT_LENGTH.value == "context_length"


class TestGeminiMapping:
    def test_map_stop(self):
        assert GEMINI_TO_OPENAI["STOP"] == "stop"

    def test_map_max_tokens(self):
        assert GEMINI_TO_OPENAI["MAX_TOKENS"] == "length"

    def test_map_safety(self):
        assert GEMINI_TO_OPENAI["SAFETY"] == "content_filter"

    def test_map_recitation(self):
        assert GEMINI_TO_OPENAI["RECITATION"] == "content_filter"

    def test_map_other(self):
        assert GEMINI_TO_OPENAI["OTHER"] == "stop"


class TestAnthropicMapping:
    def test_map_end_turn(self):
        assert ANTHROPIC_TO_OPENAI["end_turn"] == "stop"

    def test_map_max_tokens(self):
        assert ANTHROPIC_TO_OPENAI["max_tokens"] == "length"

    def test_map_tool_use(self):
        assert ANTHROPIC_TO_OPENAI["tool_use"] == "tool_calls"

    def test_map_stop_sequence(self):
        # stop_sequence maps to itself so the round trip is symmetric.
        assert ANTHROPIC_TO_OPENAI["stop_sequence"] == "stop_sequence"

    def test_map_pause_turn(self):
        assert ANTHROPIC_TO_OPENAI["pause_turn"] == "stop"

    def test_map_refusal(self):
        assert ANTHROPIC_TO_OPENAI["refusal"] == "content_filter"

    def test_map_model_context_window_exceeded(self):
        assert ANTHROPIC_TO_OPENAI["model_context_window_exceeded"] == "context_length"


class TestStopSequenceSymmetry:
    """Regression: stop_sequence must survive an anthropic -> openai -> anthropic round trip."""

    def test_anthropic_to_openai_preserves_stop_sequence(self):
        assert map_finish_reason("stop_sequence", "anthropic", "openai") == "stop_sequence"

    def test_openai_to_anthropic_preserves_stop_sequence(self):
        assert map_finish_reason("stop_sequence", "openai", "anthropic") == "stop_sequence"

    def test_round_trip_stop_sequence(self):
        openai_value = map_finish_reason("stop_sequence", "anthropic", "openai")
        assert openai_value == "stop_sequence"
        anthropic_value = map_finish_reason(openai_value, "openai", "anthropic")
        assert anthropic_value == "stop_sequence"

    def test_openai_to_anthropic_mapping_has_stop_sequence(self):
        assert OPENAI_TO_ANTHROPIC["stop_sequence"] == "stop_sequence"


class TestOpenAIToAnthropicMapping:
    def test_map_stop(self):
        assert OPENAI_TO_ANTHROPIC["stop"] == "end_turn"

    def test_map_length(self):
        assert OPENAI_TO_ANTHROPIC["length"] == "max_tokens"

    def test_map_tool_calls(self):
        assert OPENAI_TO_ANTHROPIC["tool_calls"] == "tool_use"

    def test_map_content_filter(self):
        assert OPENAI_TO_ANTHROPIC["content_filter"] == "refusal"

    def test_map_pause_turn(self):
        assert OPENAI_TO_ANTHROPIC["pause_turn"] == "pause_turn"

    def test_map_refusal(self):
        assert OPENAI_TO_ANTHROPIC["refusal"] == "refusal"

    def test_map_context_length(self):
        assert OPENAI_TO_ANTHROPIC["context_length"] == "model_context_window_exceeded"


class TestMapFinishReasonFunction:
    def test_gemini_to_openai(self):
        assert map_finish_reason("STOP", "gemini", "openai") == "stop"
        assert map_finish_reason("MAX_TOKENS", "gemini", "openai") == "length"

    def test_anthropic_to_openai(self):
        assert map_finish_reason("end_turn", "anthropic", "openai") == "stop"
        assert map_finish_reason("tool_use", "anthropic", "openai") == "tool_calls"

    def test_openai_to_anthropic(self):
        assert map_finish_reason("stop", "openai", "anthropic") == "end_turn"
        assert map_finish_reason("tool_calls", "openai", "anthropic") == "tool_use"

    def test_unknown_reason_returns_none(self):
        assert map_finish_reason("unknown", "gemini", "openai") is None

    def test_none_input_returns_none(self):
        assert map_finish_reason(None, "gemini", "openai") is None
