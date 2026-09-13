"""Usage parsing for SGLang's top-level ``usage.reasoning_tokens``.

SGLang reports reasoning tokens as a top-level usage field instead of
OpenAI's nested ``completion_tokens_details.reasoning_tokens`` on both the
non-streaming response and the streaming terminal usage chunk (see
https://docs.sglang.io/docs/basic_usage/openai_api_completions). The shared
Chat Completions parser folds it into ``CompletionTokensDetails`` for the
non-streaming path, and the adapter's streaming chunk transform folds it for
the streaming path, so billing and usage records see the same reasoning split
on every tier.
"""

from llm_proxy.serialization.openai.components.response_parser import (
    OpenAIResponseParser,
    fold_top_level_reasoning_tokens,
)


class TestTopLevelReasoningTokens:
    def test_top_level_reasoning_tokens_folded(self):
        usage = OpenAIResponseParser.parse_usage(
            {
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 8,
                    "total_tokens": 18,
                    "reasoning_tokens": 5,
                }
            }
        )
        assert usage is not None
        assert usage.completion_tokens_details is not None
        assert usage.completion_tokens_details.reasoning_tokens == 5

    def test_nested_details_win_over_top_level(self):
        """The OpenAI-nested shape stays authoritative when both exist."""
        usage = OpenAIResponseParser.parse_usage(
            {
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 8,
                    "total_tokens": 18,
                    "completion_tokens_details": {"reasoning_tokens": 3},
                    "reasoning_tokens": 5,
                }
            }
        )
        assert usage is not None
        assert usage.completion_tokens_details.reasoning_tokens == 3

    def test_no_reasoning_tokens_anywhere(self):
        usage = OpenAIResponseParser.parse_usage(
            {"usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}}
        )
        assert usage is not None
        assert usage.completion_tokens_details is None

    def test_zero_top_level_reasoning_tokens_ignored(self):
        """SGLang reports ``reasoning_tokens: 0`` by default; an absent details
        block already means "no reasoning tokens"."""
        usage = OpenAIResponseParser.parse_usage(
            {
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 8,
                    "total_tokens": 18,
                    "reasoning_tokens": 0,
                }
            }
        )
        assert usage is not None
        assert usage.completion_tokens_details is None


class TestFoldTopLevelReasoningTokens:
    """The streaming counterpart: a pure usage-dict fold."""

    def test_fold_adds_nested_details(self):
        usage = {"prompt_tokens": 10, "completion_tokens": 8, "reasoning_tokens": 5}
        folded = fold_top_level_reasoning_tokens(usage)
        assert folded["completion_tokens_details"] == {"reasoning_tokens": 5}
        assert folded["reasoning_tokens"] == 5

    def test_fold_does_not_mutate_input(self):
        usage = {"completion_tokens": 8, "reasoning_tokens": 5}
        fold_top_level_reasoning_tokens(usage)
        assert "completion_tokens_details" not in usage

    def test_existing_nested_value_wins(self):
        usage = {
            "reasoning_tokens": 5,
            "completion_tokens_details": {"reasoning_tokens": 3, "audio_tokens": 1},
        }
        folded = fold_top_level_reasoning_tokens(usage)
        assert folded["completion_tokens_details"] == {"reasoning_tokens": 3, "audio_tokens": 1}

    def test_existing_nested_details_extended_when_no_reasoning(self):
        usage = {"reasoning_tokens": 5, "completion_tokens_details": {"audio_tokens": 1}}
        folded = fold_top_level_reasoning_tokens(usage)
        assert folded["completion_tokens_details"] == {"audio_tokens": 1, "reasoning_tokens": 5}

    def test_zero_and_absent_are_noops(self):
        zero = {"completion_tokens": 8, "reasoning_tokens": 0}
        assert fold_top_level_reasoning_tokens(zero) is zero
        absent = {"completion_tokens": 8}
        assert fold_top_level_reasoning_tokens(absent) is absent
