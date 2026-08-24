"""Tests for unified thinking conversion module."""

from llm_proxy.core.thinking import (
    convert_to_anthropic,
    convert_to_gemini,
    convert_to_ollama,
    convert_to_openai,
    normalize_thinking,
    resolve_thinking,
    thinking_config_from_reasoning_effort,
    thinking_config_to_reasoning_effort,
)
from llm_proxy.models.types import ThinkingConfig


class TestNormalizeThinking:
    def test_reasoning_effort(self):
        result = normalize_thinking({"reasoning_effort": "high"})
        assert result is not None
        assert result.type == "enabled"
        assert result.budget_tokens == 32000
        assert result.effort == "high"

    def test_thinking_dict(self):
        result = normalize_thinking({"thinking": {"type": "enabled", "budget_tokens": 5000}})
        assert result is not None
        assert result.type == "enabled"
        assert result.budget_tokens == 5000
        assert result.effort is None

    def test_thinking_bool_true(self):
        result = normalize_thinking({"thinking": True})
        assert result is not None
        assert result.type == "enabled"
        assert result.budget_tokens is None

    def test_thinking_disabled(self):
        result = normalize_thinking({"thinking": {"type": "disabled"}})
        assert result is not None
        assert result.type == "disabled"

    def test_reasoning_object(self):
        result = normalize_thinking({"reasoning": {"effort": "medium"}})
        assert result is not None
        assert result.type == "enabled"
        assert result.budget_tokens == 16000
        assert result.effort == "medium"

    def test_output_config_effort(self):
        result = normalize_thinking({"output_config": {"effort": "medium"}})
        assert result is not None
        assert result.type == "enabled"
        assert result.budget_tokens == 16000
        assert result.effort == "medium"

    def test_output_config_effort_max(self):
        result = normalize_thinking({"output_config": {"effort": "max"}})
        assert result is not None
        assert result.type == "enabled"
        assert result.budget_tokens is None
        assert result.effort == "max"

    def test_output_config_dict_over_reasoning_dict(self):
        result = normalize_thinking(
            {
                "output_config": {"effort": "low"},
                "reasoning": {"effort": "high"},
            }
        )
        assert result is not None
        assert result.effort == "high"

    def test_thinking_dict_over_output_config(self):
        result = normalize_thinking(
            {
                "thinking": {"type": "enabled", "budget_tokens": 10000},
                "output_config": {"effort": "high"},
            }
        )
        assert result is not None
        assert result.type == "enabled"
        assert result.budget_tokens == 10000
        assert result.effort is None

    def test_adaptive_merges_output_config_effort(self):
        # `thinking: {type: adaptive}` + `output_config: {effort: high}` is the
        # canonical Anthropic pattern for adaptive thinking controlled by
        # effort. The effort must be preserved so downstream providers can map
        # it (e.g. to OpenAI reasoning_effort) instead of dropping it.
        result = normalize_thinking(
            {
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "high"},
            }
        )
        assert result is not None
        assert result.type == "adaptive"
        assert result.effort == "high"
        assert result.budget_tokens is None

    def test_adaptive_without_output_config_has_no_effort(self):
        result = normalize_thinking({"thinking": {"type": "adaptive"}})
        assert result is not None
        assert result.type == "adaptive"
        assert result.effort is None

    def test_no_thinking_params(self):
        result = normalize_thinking({"temperature": 0.7})
        assert result is None

    def test_reasoning_effort_precedence(self):
        result = normalize_thinking(
            {"reasoning_effort": "low", "thinking": {"type": "enabled", "budget_tokens": 10000}}
        )
        assert result is not None
        assert result.effort == "low"

    def test_thinking_dict_over_reasoning_dict(self):
        result = normalize_thinking(
            {
                "thinking": {"type": "enabled", "budget_tokens": 10000},
                "reasoning": {"effort": "high"},
            }
        )
        assert result is not None
        assert result.type == "enabled"
        assert result.budget_tokens == 10000


class TestThinkingConfigFromReasoningEffort:
    def test_none(self):
        config = thinking_config_from_reasoning_effort("none")
        assert config.type == "disabled"
        assert config.budget_tokens is None

    def test_minimal(self):
        config = thinking_config_from_reasoning_effort("minimal")
        assert config.type == "enabled"
        assert config.budget_tokens == 4000
        assert config.effort == "minimal"

    def test_low(self):
        config = thinking_config_from_reasoning_effort("low")
        assert config.type == "enabled"
        assert config.budget_tokens == 4000
        assert config.effort == "low"

    def test_medium(self):
        config = thinking_config_from_reasoning_effort("medium")
        assert config.type == "enabled"
        assert config.budget_tokens == 16000
        assert config.effort == "medium"

    def test_high(self):
        config = thinking_config_from_reasoning_effort("high")
        assert config.type == "enabled"
        assert config.budget_tokens == 32000
        assert config.effort == "high"

    def test_xhigh(self):
        config = thinking_config_from_reasoning_effort("xhigh")
        assert config.type == "enabled"
        assert config.budget_tokens == 32000
        assert config.effort == "xhigh"

    def test_case_insensitive(self):
        config = thinking_config_from_reasoning_effort("HIGH")
        assert config.type == "enabled"
        assert config.effort == "high"

    def test_unknown_defaults_to_enabled_medium(self):
        config = thinking_config_from_reasoning_effort("unknown")
        assert config.type == "enabled"
        assert config.budget_tokens == 16000
        assert config.effort == "unknown"

    def test_max(self):
        config = thinking_config_from_reasoning_effort("max")
        assert config.type == "enabled"
        assert config.budget_tokens is None
        assert config.effort == "max"


class TestThinkingConfigToReasoningEffort:
    def test_effort_field_set(self):
        config = ThinkingConfig(type="enabled", effort="high")
        assert thinking_config_to_reasoning_effort(config) == "high"

    def test_disabled(self):
        config = ThinkingConfig(type="disabled")
        assert thinking_config_to_reasoning_effort(config) == "none"

    def test_low_budget(self):
        config = ThinkingConfig(type="enabled", budget_tokens=4000)
        assert thinking_config_to_reasoning_effort(config) == "low"

    def test_medium_budget(self):
        config = ThinkingConfig(type="enabled", budget_tokens=16000)
        assert thinking_config_to_reasoning_effort(config) == "medium"

    def test_high_budget(self):
        config = ThinkingConfig(type="enabled", budget_tokens=32000)
        assert thinking_config_to_reasoning_effort(config) == "high"

    def test_no_effort_no_budget(self):
        config = ThinkingConfig(type="enabled")
        assert thinking_config_to_reasoning_effort(config) == "medium"


class TestConvertToOpenAI:
    def test_reasoning_effort_output(self):
        config = ThinkingConfig(type="enabled", effort="high")
        result = convert_to_openai(config)
        assert result == {"reasoning_effort": "high"}

    def test_thinking_dict_output(self):
        config = ThinkingConfig(type="enabled", budget_tokens=5000)
        result = convert_to_openai(config)
        assert result == {"reasoning_effort": "low"}

    def test_thinking_dict_custom_budget(self):
        config = ThinkingConfig(type="enabled", budget_tokens=7000)
        result = convert_to_openai(config)
        assert result == {"reasoning_effort": "low"}

    def test_thinking_dict_exact_medium_budget(self):
        config = ThinkingConfig(type="enabled", budget_tokens=16000)
        result = convert_to_openai(config)
        assert result == {"reasoning_effort": "medium"}

    def test_disabled(self):
        config = ThinkingConfig(type="disabled")
        result = convert_to_openai(config)
        assert result == {"reasoning_effort": "none"}

    def test_none(self):
        assert convert_to_openai(None) is None

    def test_no_type(self):
        config = ThinkingConfig(budget_tokens=5000)
        result = convert_to_openai(config)
        assert result == {"reasoning_effort": "low"}

    def test_max_effort(self):
        config = ThinkingConfig(type="enabled", effort="max")
        result = convert_to_openai(config)
        assert result == {"reasoning_effort": "max"}

    def test_adaptive_with_effort_maps_to_reasoning_effort(self):
        # adaptive + effort (merged from output_config) must surface as
        # reasoning_effort, not as an invalid `thinking: {type: adaptive}`.
        config = ThinkingConfig(type="adaptive", effort="high")
        result = convert_to_openai(config)
        assert result == {"reasoning_effort": "high"}

    def test_adaptive_without_effort_is_dropped(self):
        # `adaptive` has no OpenAI equivalent; without a derivable effort it
        # must be dropped rather than emit an unsupported thinking type that
        # OpenAI-compatible providers reject with a 4xx.
        config = ThinkingConfig(type="adaptive")
        assert convert_to_openai(config) is None


class TestConvertToAnthropic:
    def test_enabled(self):
        config = ThinkingConfig(type="enabled", budget_tokens=5000)
        result = convert_to_anthropic(config)
        assert result == {"thinking": {"type": "enabled", "budget_tokens": 5000}}

    def test_disabled(self):
        config = ThinkingConfig(type="disabled")
        result = convert_to_anthropic(config)
        assert result == {"thinking": {"type": "disabled"}}

    def test_none(self):
        assert convert_to_anthropic(None) is None

    def test_no_type(self):
        config = ThinkingConfig(budget_tokens=5000)
        assert convert_to_anthropic(config) is None

    def test_effort_outputs_output_config(self):
        config = ThinkingConfig(type="enabled", effort="medium")
        result = convert_to_anthropic(config)
        assert result is not None
        assert result["output_config"] == {"effort": "medium"}
        assert result["thinking"] == {"type": "enabled"}

    def test_effort_and_budget_tokens(self):
        config = ThinkingConfig(type="enabled", effort="high", budget_tokens=32000)
        result = convert_to_anthropic(config)
        assert result is not None
        assert result["output_config"] == {"effort": "high"}
        assert result["thinking"] == {"type": "enabled"}

    def test_budget_tokens_without_effort(self):
        config = ThinkingConfig(type="enabled", budget_tokens=5000)
        result = convert_to_anthropic(config)
        assert result is not None
        assert "output_config" not in result
        assert result["thinking"] == {"type": "enabled", "budget_tokens": 5000}

    def test_max_effort(self):
        config = ThinkingConfig(type="enabled", effort="max")
        result = convert_to_anthropic(config)
        assert result is not None
        assert result["output_config"] == {"effort": "max"}
        assert result["thinking"] == {"type": "enabled"}


class TestConvertToGemini:
    def test_low_budget(self):
        config = ThinkingConfig(type="enabled", budget_tokens=4000)
        result = convert_to_gemini(config)
        assert result == {"thinkingConfig": {"thinkingLevel": "LOW"}}

    def test_medium_budget(self):
        config = ThinkingConfig(type="enabled", budget_tokens=16000)
        result = convert_to_gemini(config)
        assert result == {"thinkingConfig": {"thinkingLevel": "MEDIUM"}}

    def test_high_budget(self):
        config = ThinkingConfig(type="enabled", budget_tokens=32000)
        result = convert_to_gemini(config)
        assert result == {"thinkingConfig": {"thinkingLevel": "HIGH"}}

    def test_from_effort_low(self):
        config = ThinkingConfig(type="enabled", effort="low")
        result = convert_to_gemini(config)
        assert result == {"thinkingConfig": {"thinkingLevel": "LOW"}}

    def test_from_effort_high(self):
        config = ThinkingConfig(type="enabled", effort="high")
        result = convert_to_gemini(config)
        assert result == {"thinkingConfig": {"thinkingLevel": "HIGH"}}

    def test_from_effort_xhigh(self):
        config = ThinkingConfig(type="enabled", effort="xhigh")
        result = convert_to_gemini(config)
        assert result == {"thinkingConfig": {"thinkingLevel": "HIGH"}}

    def test_from_effort_max(self):
        config = ThinkingConfig(type="enabled", effort="max")
        result = convert_to_gemini(config)
        assert result == {"thinkingConfig": {"thinkingLevel": "HIGH"}}

    def test_from_effort_minimal(self):
        config = ThinkingConfig(type="enabled", effort="minimal")
        result = convert_to_gemini(config)
        assert result == {"thinkingConfig": {"thinkingLevel": "LOW"}}

    def test_disabled_returns_none(self):
        config = ThinkingConfig(type="disabled")
        assert convert_to_gemini(config) is None

    def test_none(self):
        assert convert_to_gemini(None) is None

    def test_default_medium(self):
        config = ThinkingConfig(type="enabled")
        result = convert_to_gemini(config)
        assert result == {"thinkingConfig": {"thinkingLevel": "MEDIUM"}}


class TestConvertToOllama:
    def test_disabled(self):
        config = ThinkingConfig(type="disabled")
        assert convert_to_ollama(config) is False

    def test_none_effort(self):
        config = ThinkingConfig(type="enabled", effort="none")
        assert convert_to_ollama(config) is False

    def test_low_effort(self):
        config = ThinkingConfig(type="enabled", effort="low")
        assert convert_to_ollama(config) == "low"

    def test_medium_effort(self):
        config = ThinkingConfig(type="enabled", effort="medium")
        assert convert_to_ollama(config) == "medium"

    def test_high_effort(self):
        config = ThinkingConfig(type="enabled", effort="high")
        assert convert_to_ollama(config) == "high"

    def test_xhigh_effort(self):
        config = ThinkingConfig(type="enabled", effort="xhigh")
        assert convert_to_ollama(config) == "max"

    def test_max_effort(self):
        config = ThinkingConfig(type="enabled", effort="max")
        assert convert_to_ollama(config) == "max"

    def test_low_budget(self):
        config = ThinkingConfig(type="enabled", budget_tokens=4000)
        assert convert_to_ollama(config) == "low"

    def test_high_budget(self):
        config = ThinkingConfig(type="enabled", budget_tokens=32000)
        assert convert_to_ollama(config) == "high"

    def test_default_true(self):
        config = ThinkingConfig(type="enabled")
        assert convert_to_ollama(config) is True

    def test_none(self):
        assert convert_to_ollama(None) is None


class TestCrossProtocolConversion:
    """End-to-end tests simulating cross-protocol/provider routing."""

    def test_openai_reasoning_effort_to_gemini(self):
        data = {"reasoning_effort": "high"}
        config = normalize_thinking(data)
        gemini = convert_to_gemini(config)
        assert gemini == {"thinkingConfig": {"thinkingLevel": "HIGH"}}

    def test_anthropic_thinking_to_openai(self):
        data = {"thinking": {"type": "enabled", "budget_tokens": 5000}}
        config = normalize_thinking(data)
        openai = convert_to_openai(config)
        assert openai == {"reasoning_effort": "low"}

    def test_openresponses_to_ollama(self):
        data = {"reasoning": {"effort": "minimal"}}
        config = normalize_thinking(data)
        ollama = convert_to_ollama(config)
        assert ollama == "low"

    def test_openai_none_to_anthropic(self):
        data = {"reasoning_effort": "none"}
        config = normalize_thinking(data)
        anthropic = convert_to_anthropic(config)
        assert anthropic == {"thinking": {"type": "disabled"}}

    def test_anthropic_output_config_to_openai(self):
        data = {"output_config": {"effort": "medium"}}
        config = normalize_thinking(data)
        openai = convert_to_openai(config)
        assert openai == {"reasoning_effort": "medium"}

    def test_anthropic_output_config_to_anthropic(self):
        data = {"output_config": {"effort": "high"}}
        config = normalize_thinking(data)
        anthropic = convert_to_anthropic(config)
        assert anthropic == {
            "output_config": {"effort": "high"},
            "thinking": {"type": "enabled"},
        }

    def test_thinking_bool_false(self):
        result = normalize_thinking({"thinking": False})
        assert result is not None
        assert result.type == "disabled"

    def test_anthropic_adaptive_plus_output_config_to_openai(self):
        # Regression: an Anthropic client sending `thinking: {type: adaptive}`
        # together with `output_config: {effort: high}` (Claude Code's pattern)
        # must reach an OpenAI-compatible provider as `reasoning_effort: high`,
        # not as an unsupported `thinking: {type: adaptive}` that triggers a
        # 4xx "Invalid request".
        data = {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}
        config = normalize_thinking(data)
        openai = convert_to_openai(config)
        assert openai == {"reasoning_effort": "high"}

    def test_anthropic_adaptive_only_to_openai_dropped(self):
        # `adaptive` alone has no OpenAI equivalent; it must be dropped so the
        # OpenAI-compatible provider uses its default rather than rejecting an
        # unknown `thinking` type.
        data = {"thinking": {"type": "adaptive"}}
        config = normalize_thinking(data)
        assert convert_to_openai(config) is None

    def test_anthropic_adaptive_plus_output_config_roundtrip(self):
        # Anthropic adaptive + effort must survive an Anthropic -> OpenAI ->
        # Anthropic roundtrip.
        data = {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}
        config = normalize_thinking(data)
        anthropic = convert_to_anthropic(config)
        assert anthropic == {
            "output_config": {"effort": "high"},
            "thinking": {"type": "adaptive"},
        }

    def test_openai_xhigh_to_anthropic_preserves_xhigh(self):
        """OpenAI xhigh should stay as xhigh for Anthropic (now natively supported)."""
        data = {"reasoning_effort": "xhigh"}
        config = normalize_thinking(data)
        anthropic = convert_to_anthropic(config)
        assert anthropic == {
            "output_config": {"effort": "xhigh"},
            "thinking": {"type": "enabled"},
        }

    def test_openai_minimal_to_anthropic_maps_to_low(self):
        """OpenAI minimal should map to Anthropic low."""
        data = {"reasoning_effort": "minimal"}
        config = normalize_thinking(data)
        anthropic = convert_to_anthropic(config)
        assert anthropic == {
            "output_config": {"effort": "low"},
            "thinking": {"type": "enabled"},
        }

    def test_anthropic_max_to_openai_preserves_max(self):
        """Anthropic max should stay as max for OpenAI (now natively supported)."""
        data = {"output_config": {"effort": "max"}}
        config = normalize_thinking(data)
        openai = convert_to_openai(config)
        assert openai == {"reasoning_effort": "max"}

    def test_openai_xhigh_roundtrip_via_anthropic(self):
        """OpenAI xhigh → Anthropic xhigh → OpenAI xhigh.

        Uses only output_config (not thinking dict) for the reverse path,
        since Anthropic clients send one or the other, not both.
        """
        data = {"reasoning_effort": "xhigh"}
        config = normalize_thinking(data)
        anthropic = convert_to_anthropic(config)
        # Reverse path: only use output_config (the effort-bearing field)
        config2 = normalize_thinking({"output_config": anthropic["output_config"]})
        openai = convert_to_openai(config2)
        assert openai == {"reasoning_effort": "xhigh"}

    def test_anthropic_max_roundtrip_via_openai(self):
        """Anthropic max → OpenAI max → Anthropic max."""
        data = {"output_config": {"effort": "max"}}
        config = normalize_thinking(data)
        openai = convert_to_openai(config)
        # Reverse path
        config2 = normalize_thinking(openai)
        anthropic = convert_to_anthropic(config2)
        assert anthropic == {
            "output_config": {"effort": "max"},
            "thinking": {"type": "enabled"},
        }


class TestResolveThinking:
    """``resolve_thinking`` is the single source of truth provider builders use.

    It must surface effort from ``thinking`` first, then fall back to
    ``anthropic.output_config.effort`` and ``openai.reasoning_effort`` for
    directly-constructed requests that bypass the protocol parsers.
    """

    def _req(self, **params_kwargs):
        from llm_proxy.models import (
            ConversationContext,
            GenerationParams,
            InternalRequest,
            Message,
            TextBlock,
        )

        return InternalRequest(
            model="m",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            params=GenerationParams(**params_kwargs),
        )

    def test_thinking_takes_precedence(self):
        from llm_proxy.models.params import OpenAISpecificParams

        req = self._req(
            thinking=ThinkingConfig(type="enabled", effort="low"),
            openai=OpenAISpecificParams(reasoning_effort="high"),
        )
        assert resolve_thinking(req) == ThinkingConfig(type="enabled", effort="low")

    def test_falls_back_to_anthropic_output_config(self):
        from llm_proxy.models.params import AnthropicSpecificParams

        req = self._req(anthropic=AnthropicSpecificParams(output_config={"effort": "high"}))
        assert resolve_thinking(req) == ThinkingConfig(
            type="enabled", budget_tokens=32000, effort="high"
        )

    def test_falls_back_to_openai_reasoning_effort(self):
        from llm_proxy.models.params import OpenAISpecificParams

        req = self._req(openai=OpenAISpecificParams(reasoning_effort="medium"))
        assert resolve_thinking(req) == ThinkingConfig(
            type="enabled", budget_tokens=16000, effort="medium"
        )

    def test_anthropic_output_config_beats_openai_reasoning_effort(self):
        from llm_proxy.models.params import (
            AnthropicSpecificParams,
            OpenAISpecificParams,
        )

        req = self._req(
            anthropic=AnthropicSpecificParams(output_config={"effort": "low"}),
            openai=OpenAISpecificParams(reasoning_effort="high"),
        )
        assert resolve_thinking(req) == ThinkingConfig(
            type="enabled", budget_tokens=4000, effort="low"
        )

    def test_returns_none_when_no_effort_source(self):
        assert resolve_thinking(self._req()) is None

    def test_disabled_thinking_is_preserved_not_overridden(self):
        # An explicit disabled thinking must win over a set effort source;
        # provider builders rely on this to emit the disabled form.
        from llm_proxy.models.params import OpenAISpecificParams

        req = self._req(
            thinking=ThinkingConfig(type="disabled"),
            openai=OpenAISpecificParams(reasoning_effort="high"),
        )
        assert resolve_thinking(req) == ThinkingConfig(type="disabled")
