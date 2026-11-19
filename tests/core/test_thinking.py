"""Tests for unified thinking conversion module."""

from llm_proxy.core.thinking import (
    convert_to_anthropic,
    convert_to_gemini,
    convert_to_gemini_interactions,
    convert_to_gemini_legacy,
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
        assert result.type == "adaptive"
        assert result.budget_tokens is None
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

    def test_unknown_thinking_type_preserved_verbatim(self):
        result = normalize_thinking({"thinking": {"type": "future_mode", "budget_tokens": 2048}})
        assert result == ThinkingConfig(type="future_mode", budget_tokens=2048)

    def test_cohere_token_budget_and_zero_preserved(self):
        result = normalize_thinking({"thinking": {"type": "enabled", "token_budget": 4096}})
        assert result == ThinkingConfig(type="enabled", budget_tokens=4096)

        zero_result = normalize_thinking({"thinking": {"type": "enabled", "token_budget": 0}})
        assert zero_result == ThinkingConfig(type="enabled", budget_tokens=0)

    def test_glm_nested_effort_and_clear_thinking_preserved(self):
        result = normalize_thinking(
            {
                "thinking": {
                    "type": "enabled",
                    "reasoning_effort": "high",
                    "clear_thinking": False,
                }
            }
        )
        assert result == ThinkingConfig(
            type="enabled",
            effort="high",
            extra={"clear_thinking": False},
        )

    def test_reasoning_object(self):
        result = normalize_thinking({"reasoning": {"effort": "medium"}})
        assert result is not None
        assert result.type == "adaptive"
        assert result.budget_tokens is None
        assert result.effort == "medium"

    def test_reasoning_object_openrouter_fields(self):
        assert normalize_thinking({"reasoning": {"enabled": False}}) == ThinkingConfig(
            type="disabled"
        )
        assert normalize_thinking(
            {"reasoning": {"max_tokens": 5000, "exclude": True}}
        ) == ThinkingConfig(type="enabled", budget_tokens=5000, display="omitted")
        expected_enabled = ThinkingConfig(type="enabled")
        assert normalize_thinking({"reasoning": {"enabled": True}}) == expected_enabled

    def test_output_config_effort(self):
        result = normalize_thinking({"output_config": {"effort": "medium"}})
        assert result == ThinkingConfig(type=None, effort="medium")

    def test_output_config_effort_max(self):
        result = normalize_thinking({"output_config": {"effort": "max"}})
        assert result == ThinkingConfig(type=None, effort="max")

    def test_output_config_dict_over_reasoning_dict(self):
        result = normalize_thinking(
            {
                "output_config": {"effort": "low"},
                "reasoning": {"effort": "high"},
            }
        )
        assert result is not None
        assert result.effort == "high"

    def test_thinking_dict_preserves_budget_and_merges_output_config_effort(self):
        result = normalize_thinking(
            {
                "thinking": {"type": "enabled", "budget_tokens": 10000},
                "output_config": {"effort": "high"},
            }
        )
        assert result is not None
        assert result.type == "enabled"
        assert result.budget_tokens == 10000
        assert result.effort == "high"

    def test_adaptive_merges_output_config_effort(self):
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

    def test_between_tools_merges_output_config_effort(self):
        result = normalize_thinking(
            {
                "thinking": {"type": "between_tools"},
                "output_config": {"effort": "high"},
            }
        )
        assert result == ThinkingConfig(type="between_tools", effort="high")

    def test_adaptive_without_output_config_has_no_effort(self):
        result = normalize_thinking({"thinking": {"type": "adaptive"}})
        assert result is not None
        assert result.type == "adaptive"
        assert result.effort is None

    def test_enabled_without_budget_merges_output_config_effort(self):
        result = normalize_thinking(
            {
                "thinking": {"type": "enabled"},
                "output_config": {"effort": "xhigh"},
            }
        )
        assert result is not None
        assert result.type == "enabled"
        assert result.effort == "xhigh"
        assert result.budget_tokens is None

    def test_no_thinking_params(self):
        result = normalize_thinking({"temperature": 0.7})
        assert result is None

    def test_thinking_takes_precedence_over_reasoning_effort_when_disabled(self):
        result = normalize_thinking({"reasoning_effort": "high", "thinking": {"type": "disabled"}})
        assert result == ThinkingConfig(type="disabled", effort=None)

    def test_thinking_dict_merges_top_level_reasoning_effort_when_enabled(self):
        result = normalize_thinking(
            {"reasoning_effort": "low", "thinking": {"type": "enabled", "budget_tokens": 10000}}
        )
        assert result == ThinkingConfig(type="enabled", budget_tokens=10000, effort="low")

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
        assert config == ThinkingConfig(type="adaptive", effort="minimal")

    def test_low(self):
        config = thinking_config_from_reasoning_effort("low")
        assert config == ThinkingConfig(type="adaptive", effort="low")

    def test_medium(self):
        config = thinking_config_from_reasoning_effort("medium")
        assert config == ThinkingConfig(type="adaptive", effort="medium")

    def test_high(self):
        config = thinking_config_from_reasoning_effort("high")
        assert config == ThinkingConfig(type="adaptive", effort="high")

    def test_xhigh(self):
        config = thinking_config_from_reasoning_effort("xhigh")
        assert config == ThinkingConfig(type="adaptive", effort="xhigh")

    def test_case_insensitive(self):
        config = thinking_config_from_reasoning_effort("HIGH")
        assert config == ThinkingConfig(type="adaptive", effort="high")

    def test_unknown_passes_through_without_fabricated_budget(self):
        config = thinking_config_from_reasoning_effort("unknown")
        assert config == ThinkingConfig(type="adaptive", effort="unknown")

    def test_max(self):
        config = thinking_config_from_reasoning_effort("max")
        assert config == ThinkingConfig(type="adaptive", effort="max")


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

    def test_no_effort_no_budget_returns_none(self):
        config = ThinkingConfig(type="enabled")
        assert thinking_config_to_reasoning_effort(config) is None


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

    def test_bare_enabled_preserves_thinking_object(self):
        config = ThinkingConfig(type="enabled")
        assert convert_to_openai(config) == {"thinking": {"type": "enabled"}}

    def test_extra_keys_preserved_alongside_reasoning_effort(self):
        config = ThinkingConfig(
            type="enabled",
            effort="high",
            extra={"clear_thinking": False},
        )
        assert convert_to_openai(config) == {
            "reasoning_effort": "high",
            "thinking": {"type": "enabled", "clear_thinking": False},
        }

    def test_no_type(self):
        config = ThinkingConfig(budget_tokens=5000)
        result = convert_to_openai(config)
        assert result == {"reasoning_effort": "low"}

    def test_max_effort(self):
        config = ThinkingConfig(type="enabled", effort="max")
        result = convert_to_openai(config)
        assert result == {"reasoning_effort": "max"}

    def test_adaptive_with_effort_maps_to_reasoning_effort(self):
        config = ThinkingConfig(type="adaptive", effort="high")
        result = convert_to_openai(config)
        assert result == {"reasoning_effort": "high"}

    def test_adaptive_without_effort_is_dropped(self):
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

    def test_no_type_no_effort(self):
        config = ThinkingConfig(budget_tokens=5000)
        assert convert_to_anthropic(config) is None

    def test_bare_effort_omits_thinking_block(self):
        config = ThinkingConfig(type=None, effort="high")
        assert convert_to_anthropic(config) == {"output_config": {"effort": "high"}}

    def test_effort_outputs_output_config(self):
        config = ThinkingConfig(type="enabled", effort="medium")
        result = convert_to_anthropic(config)
        assert result == {"thinking": {"type": "enabled"}, "output_config": {"effort": "medium"}}

    def test_effort_and_budget_tokens_both_preserved(self):
        config = ThinkingConfig(type="enabled", effort="high", budget_tokens=32000)
        result = convert_to_anthropic(config)
        assert result == {
            "thinking": {"type": "enabled", "budget_tokens": 32000},
            "output_config": {"effort": "high"},
        }

    def test_minimal_effort_preserved_verbatim(self):
        config = ThinkingConfig(type="adaptive", effort="minimal")
        assert convert_to_anthropic(config) == {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "minimal"},
        }

    def test_budget_tokens_without_effort(self):
        config = ThinkingConfig(type="enabled", budget_tokens=5000)
        result = convert_to_anthropic(config)
        assert result is not None
        assert "output_config" not in result
        assert result["thinking"] == {"type": "enabled", "budget_tokens": 5000}

    def test_max_effort(self):
        config = ThinkingConfig(type="enabled", effort="max")
        result = convert_to_anthropic(config)
        assert result == {
            "thinking": {"type": "enabled"},
            "output_config": {"effort": "max"},
        }


class TestConvertToGemini:
    """Gemini thinking conversions across generateContent and Interactions."""

    def test_low_budget(self):
        config = ThinkingConfig(type="enabled", budget_tokens=4000)
        result = convert_to_gemini(config)
        assert result == {"thinkingConfig": {"thinkingLevel": "LOW", "includeThoughts": True}}

    def test_medium_budget(self):
        config = ThinkingConfig(type="enabled", budget_tokens=16000)
        result = convert_to_gemini(config)
        assert result == {"thinkingConfig": {"thinkingLevel": "MEDIUM", "includeThoughts": True}}

    def test_high_budget(self):
        config = ThinkingConfig(type="enabled", budget_tokens=32000)
        result = convert_to_gemini(config)
        assert result == {"thinkingConfig": {"thinkingLevel": "HIGH", "includeThoughts": True}}

    def test_from_effort_low(self):
        config = ThinkingConfig(type="enabled", effort="low")
        result = convert_to_gemini(config)
        assert result is not None
        assert result["thinkingConfig"]["thinkingLevel"] == "LOW"

    def test_from_effort_high(self):
        config = ThinkingConfig(type="enabled", effort="high")
        result = convert_to_gemini(config)
        assert result is not None
        assert result["thinkingConfig"]["thinkingLevel"] == "HIGH"

    def test_from_effort_xhigh(self):
        config = ThinkingConfig(type="enabled", effort="xhigh")
        result = convert_to_gemini(config)
        assert result == {"thinkingConfig": {"thinkingLevel": "XHIGH", "includeThoughts": True}}

    def test_from_effort_max_passes_verbatim(self):
        config = ThinkingConfig(type="enabled", effort="max")
        assert convert_to_gemini(config) == {
            "thinkingConfig": {"thinkingLevel": "MAX", "includeThoughts": True}
        }
        assert convert_to_gemini_interactions(config) == {"thinking_level": "max"}

    def test_from_effort_future_tier_passes_through(self):
        config = ThinkingConfig(type="enabled", effort="ultra")
        assert convert_to_gemini(config) == {
            "thinkingConfig": {"thinkingLevel": "ULTRA", "includeThoughts": True}
        }
        assert convert_to_gemini_interactions(config) == {"thinking_level": "ultra"}
        assert convert_to_openai(config) == {"reasoning_effort": "ultra"}
        assert convert_to_anthropic(config) == {
            "thinking": {"type": "enabled"},
            "output_config": {"effort": "ultra"},
        }
        assert convert_to_ollama(config) == "ultra"

    def test_from_effort_minimal_and_xhigh(self):
        minimal_cfg = normalize_thinking({"reasoning_effort": "minimal"})
        assert convert_to_gemini(minimal_cfg) == {
            "thinkingConfig": {"thinkingLevel": "MINIMAL", "includeThoughts": True}
        }
        xhigh_cfg = normalize_thinking({"reasoning_effort": "xhigh"})
        assert convert_to_gemini(xhigh_cfg) == {
            "thinkingConfig": {"thinkingLevel": "XHIGH", "includeThoughts": True}
        }
        assert convert_to_gemini_interactions(xhigh_cfg) == {"thinking_level": "xhigh"}

    def test_display_omitted_disables_include_thoughts_and_summaries(self):
        cfg = ThinkingConfig(type="enabled", effort="high", display="omitted")
        assert convert_to_gemini(cfg) == {
            "thinkingConfig": {"thinkingLevel": "HIGH", "includeThoughts": False}
        }
        assert convert_to_gemini_interactions(cfg) == {
            "thinking_level": "high",
            "thinking_summaries": "none",
        }
        assert convert_to_gemini_legacy(ThinkingConfig(type="enabled", display="omitted")) == {
            "thinkingConfig": {"thinkingBudget": -1, "includeThoughts": False}
        }

    def test_disabled_emits_minimal_without_forcing_thoughts(self):
        assert convert_to_gemini(ThinkingConfig(type="disabled")) == {
            "thinkingConfig": {"thinkingLevel": "MINIMAL"}
        }
        assert convert_to_gemini_interactions(ThinkingConfig(type="disabled")) == {
            "thinking_level": "minimal"
        }
        assert convert_to_gemini_legacy(ThinkingConfig(type="disabled")) == {
            "thinkingConfig": {"thinkingBudget": 0}
        }

    def test_legacy_effort_budgets_stay_within_flash_cap(self):
        cfg = thinking_config_from_reasoning_effort("high")
        assert convert_to_gemini_legacy(cfg) == {
            "thinkingConfig": {"thinkingBudget": 8192, "includeThoughts": True}
        }
        assert convert_to_gemini_legacy(ThinkingConfig(type="enabled")) == {
            "thinkingConfig": {"thinkingBudget": -1, "includeThoughts": True}
        }

    def test_none(self):
        assert convert_to_gemini(None) is None

    def test_bare_enabled_omits_thinking_level(self):
        config = ThinkingConfig(type="enabled")
        result = convert_to_gemini(config)
        assert result == {"thinkingConfig": {"includeThoughts": True}}


class TestConvertToOllama:
    def test_disabled(self):
        config = ThinkingConfig(type="disabled")
        assert convert_to_ollama(config) is False

    def test_none_effort(self):
        config = ThinkingConfig(type="enabled", effort="none")
        assert convert_to_ollama(config) is False

    def test_minimal_effort_verbatim(self):
        config = ThinkingConfig(type="enabled", effort="minimal")
        assert convert_to_ollama(config) == "minimal"

    def test_low_effort(self):
        config = ThinkingConfig(type="enabled", effort="low")
        assert convert_to_ollama(config) == "low"

    def test_medium_effort(self):
        config = ThinkingConfig(type="enabled", effort="medium")
        assert convert_to_ollama(config) == "medium"

    def test_high_effort(self):
        config = ThinkingConfig(type="enabled", effort="high")
        assert convert_to_ollama(config) == "high"

    def test_xhigh_effort_verbatim(self):
        config = ThinkingConfig(type="enabled", effort="xhigh")
        assert convert_to_ollama(config) == "xhigh"

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
        assert gemini == {"thinkingConfig": {"thinkingLevel": "HIGH", "includeThoughts": True}}

    def test_anthropic_thinking_to_openai(self):
        data = {"thinking": {"type": "enabled", "budget_tokens": 5000}}
        config = normalize_thinking(data)
        openai = convert_to_openai(config)
        assert openai == {"reasoning_effort": "low"}

    def test_openresponses_to_ollama(self):
        data = {"reasoning": {"effort": "minimal"}}
        config = normalize_thinking(data)
        ollama = convert_to_ollama(config)
        assert ollama == "minimal"

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
        assert anthropic == {"output_config": {"effort": "high"}}

    def test_thinking_bool_false(self):
        result = normalize_thinking({"thinking": False})
        assert result is not None
        assert result.type == "disabled"

    def test_anthropic_adaptive_plus_output_config_to_openai(self):
        data = {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}
        config = normalize_thinking(data)
        openai = convert_to_openai(config)
        assert openai == {"reasoning_effort": "high"}

    def test_anthropic_adaptive_only_to_openai_dropped(self):
        data = {"thinking": {"type": "adaptive"}}
        config = normalize_thinking(data)
        assert convert_to_openai(config) is None

    def test_anthropic_adaptive_plus_output_config_roundtrip(self):
        data = {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}
        config = normalize_thinking(data)
        anthropic = convert_to_anthropic(config)
        assert anthropic == {
            "output_config": {"effort": "high"},
            "thinking": {"type": "adaptive"},
        }

    def test_openai_xhigh_to_anthropic_preserves_xhigh(self):
        data = {"reasoning_effort": "xhigh"}
        config = normalize_thinking(data)
        anthropic = convert_to_anthropic(config)
        assert anthropic == {
            "output_config": {"effort": "xhigh"},
            "thinking": {"type": "adaptive"},
        }

    def test_openai_minimal_to_anthropic_preserves_minimal(self):
        data = {"reasoning_effort": "minimal"}
        config = normalize_thinking(data)
        anthropic = convert_to_anthropic(config)
        assert anthropic == {
            "output_config": {"effort": "minimal"},
            "thinking": {"type": "adaptive"},
        }

    def test_anthropic_max_to_openai_preserves_max(self):
        data = {"output_config": {"effort": "max"}}
        config = normalize_thinking(data)
        openai = convert_to_openai(config)
        assert openai == {"reasoning_effort": "max"}

    def test_openai_xhigh_roundtrip_via_anthropic(self):
        data = {"reasoning_effort": "xhigh"}
        config = normalize_thinking(data)
        anthropic = convert_to_anthropic(config)
        assert anthropic is not None
        config2 = normalize_thinking({"output_config": anthropic["output_config"]})
        openai = convert_to_openai(config2)
        assert openai == {"reasoning_effort": "xhigh"}

    def test_anthropic_max_roundtrip_via_openai(self):
        data = {"output_config": {"effort": "max"}}
        config = normalize_thinking(data)
        openai = convert_to_openai(config)
        assert openai is not None
        config2 = normalize_thinking(openai)
        anthropic = convert_to_anthropic(config2)
        assert anthropic == {
            "output_config": {"effort": "max"},
            "thinking": {"type": "adaptive"},
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
        assert resolve_thinking(req) == ThinkingConfig(type=None, effort="high")

    def test_falls_back_to_openai_reasoning_effort(self):
        from llm_proxy.models.params import OpenAISpecificParams

        req = self._req(openai=OpenAISpecificParams(reasoning_effort="medium"))
        assert resolve_thinking(req) == ThinkingConfig(type="adaptive", effort="medium")

    def test_anthropic_output_config_beats_openai_reasoning_effort(self):
        from llm_proxy.models.params import (
            AnthropicSpecificParams,
            OpenAISpecificParams,
        )

        req = self._req(
            anthropic=AnthropicSpecificParams(output_config={"effort": "low"}),
            openai=OpenAISpecificParams(reasoning_effort="high"),
        )
        assert resolve_thinking(req) == ThinkingConfig(type=None, effort="low")

    def test_returns_none_when_no_effort_source(self):
        assert resolve_thinking(self._req()) is None

    def test_disabled_thinking_is_preserved_not_overridden(self):
        from llm_proxy.models.params import OpenAISpecificParams

        req = self._req(
            thinking=ThinkingConfig(type="disabled"),
            openai=OpenAISpecificParams(reasoning_effort="high"),
        )
        assert resolve_thinking(req) == ThinkingConfig(type="disabled")


class TestBetweenToolsConversion:
    """``between_tools`` (Anthropic's Claude Sonnet 5.5 setting)."""

    def test_anthropic_receives_between_tools_verbatim(self):
        result = convert_to_anthropic(ThinkingConfig(type="between_tools"))
        assert result == {"thinking": {"type": "between_tools"}}

    def test_openai_translation_leaves_provider_default(self):
        assert convert_to_openai(ThinkingConfig(type="between_tools")) is None

    def test_gemini_returns_none_without_effort(self):
        config = ThinkingConfig(type="between_tools")
        assert convert_to_gemini(config) is None
        assert convert_to_gemini(ThinkingConfig(type="between_tools", effort="low")) == {
            "thinkingConfig": {"thinkingLevel": "LOW", "includeThoughts": True}
        }

    def test_gemini_interactions_returns_none_without_effort(self):
        assert convert_to_gemini_interactions(ThinkingConfig(type="between_tools")) is None
        assert convert_to_gemini_interactions(
            ThinkingConfig(type="between_tools", effort="low")
        ) == {"thinking_level": "low"}

    def test_gemini_legacy_uses_floor_budget(self):
        result = convert_to_gemini_legacy(ThinkingConfig(type="between_tools"))
        assert result == {"thinkingConfig": {"thinkingBudget": 1024, "includeThoughts": True}}

    def test_ollama_uses_low(self):
        assert convert_to_ollama(ThinkingConfig(type="between_tools")) == "low"
