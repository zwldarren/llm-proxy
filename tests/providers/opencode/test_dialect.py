"""Dialect vocabulary and pure inference — no network, no catalog."""

import pytest

from llm_proxy.providers.opencode.dialect import (
    Dialect,
    derive_dialect_roots,
    derive_models_url,
    infer_dialect_from_id,
)

CONSOLE_BASE = "https://opencode.ai/inference/v1"
ZEN_BASE = "https://opencode.ai/zen/v1"
GO_ZEN_BASE = "https://opencode.ai/zen/go/v1"
GO_CONSOLE_BASE = "https://opencode.ai/inference/go/v1"
V2_CONSOLE_BASE = "https://opencode.ai/inference"
V2_GO_BASE = "https://opencode.ai/inference/go"


class TestInferDialectFromId:
    """The prefix heuristics, compiled from the catalog snapshot."""

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            # Anthropic family (Claude).
            ("claude-opus-4-5", Dialect.ANTHROPIC),
            ("claude-sonnet-5", Dialect.ANTHROPIC),
            ("claude-fable-5-1", Dialect.ANTHROPIC),
            # OpenAI Responses family: GPT, Grok, Muse Spark.
            ("gpt-5.5", Dialect.RESPONSES),
            ("gpt-5-codex", Dialect.RESPONSES),
            ("grok-4.7", Dialect.RESPONSES),
            ("grok-build-0.1", Dialect.RESPONSES),
            ("muse-spark-1.3", Dialect.RESPONSES),
            # Google.
            ("gemini-3.6-flash", Dialect.GOOGLE),
            ("gemini-3.1-pro", Dialect.GOOGLE),
            # System One evaluation models.
            ("jev-1.13", Dialect.SYSTEMONE),
            ("jev-1.13-free", Dialect.SYSTEMONE),
            # Everything catalog-served on Chat Completions: DeepSeek, GLM,
            # Kimi, MiniMax, Mistral, the stealth/free models.
            ("glm-5.3", Dialect.CHAT),
            ("kimi-k2.7-code", Dialect.CHAT),
            ("deepseek-v4-pro", Dialect.CHAT),
            ("minimax-m3", Dialect.CHAT),
            ("mistral-large-4", Dialect.CHAT),
            ("big-pickle", Dialect.CHAT),
            ("step-5-preview", Dialect.CHAT),
            ("qwen3.8-max", Dialect.CHAT),
            # ``grok-code`` is Chat Completions while the numbered Groks are
            # Responses — the one same-family split in the snapshot.
            ("grok-code", Dialect.CHAT),
            # Unknowable ids fall back to the universal dialect, never raise.
            ("some-future-model", Dialect.CHAT),
        ],
    )
    def test_family_mapping(self, model: str, expected: Dialect):
        assert infer_dialect_from_id(model) is expected

    def test_empty_and_none(self):
        assert infer_dialect_from_id("") is Dialect.CHAT
        assert infer_dialect_from_id(None) is Dialect.CHAT


class TestDeriveDialectRoots:
    """The v2 roots prefix every dialect (chat included); the legacy ones do not."""

    @pytest.mark.parametrize("base", [V2_CONSOLE_BASE, V2_GO_BASE])
    def test_v2_lane_roots(self, base: str):
        roots = derive_dialect_roots(base)
        assert roots[Dialect.CHAT] == f"{base}/openai/v1"
        assert roots[Dialect.RESPONSES] == f"{base}/openai/v1"
        assert roots[Dialect.ANTHROPIC] == f"{base}/anthropic/v1"
        assert roots[Dialect.GOOGLE] == f"{base}/google/v1beta"
        assert roots[Dialect.SYSTEMONE] == f"{base}/systemone/v1"

    def test_v2_lane_trailing_slash(self):
        roots = derive_dialect_roots(f"{V2_GO_BASE}/")
        assert roots[Dialect.ANTHROPIC] == f"{V2_GO_BASE}/anthropic/v1"

    def test_console_lane(self):
        roots = derive_dialect_roots(CONSOLE_BASE)
        assert roots[Dialect.CHAT] == CONSOLE_BASE
        assert roots[Dialect.RESPONSES] == "https://opencode.ai/inference/openai/v1"
        assert roots[Dialect.ANTHROPIC] == "https://opencode.ai/inference/anthropic/v1"
        assert roots[Dialect.GOOGLE] == "https://opencode.ai/inference/google/v1beta"
        assert roots[Dialect.SYSTEMONE] == "https://opencode.ai/inference/systemone/v1"

    @pytest.mark.parametrize("base", [ZEN_BASE, GO_ZEN_BASE, GO_CONSOLE_BASE])
    def test_flat_lanes(self, base: str):
        roots = derive_dialect_roots(base)
        assert set(roots.values()) == {base}

    def test_trailing_slash_still_matches_the_console_lane(self):
        roots = derive_dialect_roots(f"{CONSOLE_BASE}/")
        assert roots[Dialect.ANTHROPIC] == "https://opencode.ai/inference/anthropic/v1"

    def test_a_relay_host_is_flat(self):
        """A mirror host cannot be assumed to mirror the path layout;
        ``endpoint_base_urls`` overrides are the operator's control there."""
        roots = derive_dialect_roots("https://mirror.example.com/inference/v1")
        assert set(roots.values()) == {"https://mirror.example.com/inference/v1"}


class TestDeriveModelsUrl:
    """The model list is versioned under the v2 lane roots, flat elsewhere."""

    @pytest.mark.parametrize("base", [V2_CONSOLE_BASE, V2_GO_BASE])
    def test_v2_lanes(self, base: str):
        assert derive_models_url(base) == f"{base}/v1/models"

    def test_v2_lane_trailing_slash(self):
        assert derive_models_url(f"{V2_GO_BASE}/") == f"{V2_GO_BASE}/v1/models"

    @pytest.mark.parametrize("base", [CONSOLE_BASE, ZEN_BASE, GO_ZEN_BASE, GO_CONSOLE_BASE])
    def test_legacy_lanes(self, base: str):
        assert derive_models_url(base) == f"{base}/models"

    def test_a_relay_host_keeps_the_flat_shape(self):
        assert (
            derive_models_url("https://mirror.example.com/inference/go")
            == "https://mirror.example.com/inference/go/models"
        )
