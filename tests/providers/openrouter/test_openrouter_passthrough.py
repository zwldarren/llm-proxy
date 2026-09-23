"""OpenRouter native Responses passthrough, app attribution and header forwarding.

OpenRouter differs from a plain Chat Completions upstream in three ways this
module pins down:

* it exposes a native OpenAI Responses endpoint, so ``/v1/responses`` clients
  are forwarded verbatim instead of being flattened to Chat Completions
  (which silently dropped ``reasoning.mode``/``context``, ``provider``,
  ``plugins``, ``usage``, ...);
* it accepts the unified ``reasoning`` object on its Chat Completions wire,
  so the builder emits that rather than a bare ``reasoning_effort``;
* it tracks the calling application through attribution headers and accepts
  per-request control headers that must survive the proxy hop.
"""

import pytest

import llm_proxy.protocols.openresponses.serializer  # noqa: F401 — registration
import llm_proxy.serialization.providers.chat_completions  # noqa: F401 — registration
from llm_proxy.core.conversion import plan_conversion, prepare_native_body
from llm_proxy.models import ConversionTier
from llm_proxy.protocols.registry import get_protocol_serializer
from llm_proxy.providers.openai.client_headers import (
    capture_client_headers,
    clear_client_headers,
)
from llm_proxy.providers.openrouter.adapter import OpenRouterAdapter


@pytest.fixture
def adapter():
    return OpenRouterAdapter(api_key="sk-or-test")


@pytest.fixture(autouse=True)
def _clear_captured_headers():
    """Client headers are a contextvar; never leak them between tests."""
    yield
    clear_client_headers()


# ---------------------------------------------------------------------------
# Capability declarations
# ---------------------------------------------------------------------------


class TestNativeResponses:
    def test_declares_openresponses_native(self, adapter):
        assert adapter.native_protocols == frozenset({"openresponses"})

    def test_responses_url_hangs_off_the_chat_root(self, adapter):
        assert adapter._responses_url(model="openai/gpt-5.6") == (
            "https://openrouter.ai/api/v1/responses"
        )

    def test_responses_request_takes_the_native_tier(self, adapter):
        raw = {
            "model": "openai/gpt-5.6",
            "input": "Prove that there are infinitely many primes.",
            "reasoning": {"mode": "pro", "effort": "high", "context": "all_turns"},
            "provider": {"sort": "throughput", "zdr": True},
            "plugins": [{"id": "web"}],
            "usage": {"include": True},
            "stream": False,
        }
        request = get_protocol_serializer("openresponses").parse_request(raw)
        request.metadata.protocol_name = "openresponses"
        request._raw_protocol_data = raw

        plan = plan_conversion(adapter, request)
        assert plan.response_mode == ConversionTier.NATIVE_PASSTHROUGH
        assert plan.stream_mode == ConversionTier.NATIVE_PASSTHROUGH

        # Verbatim: nothing the Chat Completions translation used to drop is lost.
        body = prepare_native_body(adapter, request)
        assert body["reasoning"] == {"mode": "pro", "effort": "high", "context": "all_turns"}
        assert body["provider"] == {"sort": "throughput", "zdr": True}
        assert body["plugins"] == [{"id": "web"}]
        assert body["usage"] == {"include": True}
        assert body["model"] == "openai/gpt-5.6"

    def test_chat_protocol_is_not_native(self, adapter):
        """only /v1/responses is forwarded verbatim; chat keeps the translation."""
        assert not adapter.supports_native_request("openai", None)


# ---------------------------------------------------------------------------
# App attribution
# ---------------------------------------------------------------------------


class TestAppAttribution:
    def test_defaults_to_the_llm_proxy_project(self, adapter):
        headers = adapter._build_headers()
        assert headers["HTTP-Referer"] == "https://github.com/zwldarren/llm-proxy"
        assert headers["X-OpenRouter-Title"] == "LLM Proxy"

    def test_typed_config_overrides_defaults(self):
        adapter = OpenRouterAdapter(
            api_key="sk-or-test",
            app_attribution={
                "url": "https://my.app",
                "title": "My App",
                "categories": ["cli-agent", "cloud-agent"],
                "visibility": "hidden",
            },
        )
        headers = adapter._build_headers()
        assert headers["HTTP-Referer"] == "https://my.app"
        assert headers["X-OpenRouter-Title"] == "My App"
        assert headers["X-OpenRouter-Categories"] == "cli-agent,cloud-agent"
        assert headers["X-OpenRouter-App-Visibility"] == "hidden"

    def test_custom_headers_win_over_typed_config(self):
        adapter = OpenRouterAdapter(
            api_key="sk-or-test",
            custom_headers={"HTTP-Referer": "https://explicit.example"},
            app_attribution={"url": "https://my.app", "title": "My App"},
        )
        headers = adapter._build_headers()
        assert headers["HTTP-Referer"] == "https://explicit.example"
        # The title still comes from the typed config: only the collision loses.
        assert headers["X-OpenRouter-Title"] == "My App"

    def test_legacy_x_title_is_respected(self):
        adapter = OpenRouterAdapter(
            api_key="sk-or-test",
            custom_headers={"X-Title": "Legacy Name"},
        )
        headers = adapter._build_headers()
        assert headers["X-Title"] == "Legacy Name"
        assert "X-OpenRouter-Title" not in headers

    def test_visibility_ships_with_the_default_referer(self):
        """A hidden app still needs an ``HTTP-Referer`` to be attributed at all."""
        adapter = OpenRouterAdapter(
            api_key="sk-or-test",
            app_attribution={"visibility": "hidden"},
        )
        headers = adapter._build_headers()
        assert headers["HTTP-Referer"] == "https://github.com/zwldarren/llm-proxy"
        assert headers["X-OpenRouter-App-Visibility"] == "hidden"

    def test_categories_accept_a_list_or_a_string(self):
        from_list = OpenRouterAdapter(
            api_key="k", app_attribution={"categories": ["cli-agent", "cloud-agent"]}
        )._build_headers()
        from_string = OpenRouterAdapter(
            api_key="k", app_attribution={"categories": "cli-agent"}
        )._build_headers()
        assert from_list["X-OpenRouter-Categories"] == "cli-agent,cloud-agent"
        assert from_string["X-OpenRouter-Categories"] == "cli-agent"


# ---------------------------------------------------------------------------
# Client control headers
# ---------------------------------------------------------------------------


class TestClientHeaderForwarding:
    def test_openrouter_control_headers_are_forwarded(self, adapter):
        capture_client_headers(
            {
                "X-OpenRouter-Metadata": "enabled",
                "X-OpenRouter-Cache": "true",
                "X-OpenRouter-Cache-TTL": "600",
            }
        )
        headers = adapter._build_headers()
        assert headers["X-OpenRouter-Metadata"] == "enabled"
        assert headers["X-OpenRouter-Cache"] == "true"
        assert headers["X-OpenRouter-Cache-TTL"] == "600"

    def test_client_fingerprint_headers_are_not_forwarded_to_a_router(self, adapter):
        """Codex headers describe the client's original target, not OpenRouter."""
        capture_client_headers(
            {
                "originator": "codex_cli_rs",
                "X-Codex-Beta-Features": "shell_tool",
                "User-Agent": "codex/1.0",
                "X-OpenRouter-Cache": "true",
            }
        )
        headers = adapter._build_headers()
        assert headers["X-OpenRouter-Cache"] == "true"
        assert "originator" not in headers
        assert "X-Codex-Beta-Features" not in headers
        assert "User-Agent" not in headers

    def test_client_cannot_spoof_auth_or_attribution(self, adapter):
        capture_client_headers(
            {
                "Authorization": "Bearer attacker",
                "HTTP-Referer": "https://attacker.example",
                "X-OpenRouter-Title": "Attacker App",
                "X-Title": "Attacker App",
                "X-OpenRouter-Categories": "malware",
                "X-OpenRouter-App-Visibility": "hidden",
            }
        )
        headers = adapter._build_headers()
        # auth comes from the provider config, attribution from the operator.
        assert headers["Authorization"] == "Bearer sk-or-test"
        assert headers["HTTP-Referer"] == "https://github.com/zwldarren/llm-proxy"
        assert headers["X-OpenRouter-Title"] == "LLM Proxy"
        assert "X-Title" not in headers
        assert "X-OpenRouter-Categories" not in headers
        assert "X-OpenRouter-App-Visibility" not in headers


# ---------------------------------------------------------------------------
# Chat Completions extras that OpenRouter documents
# ---------------------------------------------------------------------------


class TestChatExemptExtraKeys:
    def test_documented_fields_survive_the_ignore_policy(self, adapter):
        """``unknown_fields_policy="ignore"`` must not eat OpenRouter's own fields."""
        raw = {
            "model": "openai/gpt-5.6",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning": {"mode": "pro"},
            "reasoning_effort": "high",
            "provider": {"sort": "throughput", "zdr": True},
            "models": ["a/b"],
            "plugins": [{"id": "web"}],
            "usage": {"include": True},
            "min_p": 0.05,
            "definitely_not_a_field": 1,
        }
        request = get_protocol_serializer("openai").parse_request(raw)
        request.metadata.protocol_name = "openai"

        body = adapter._build_request_body(request)

        assert body["reasoning"] == {"mode": "pro", "effort": "high"}
        assert "reasoning_effort" not in body
        assert body["provider"] == {"sort": "throughput", "zdr": True}
        assert body["models"] == ["a/b"]
        assert body["plugins"] == [{"id": "web"}]
        assert body["usage"] == {"include": True}
        assert body["min_p"] == 0.05
        # Genuinely unknown fields are still dropped by the default policy.
        assert "definitely_not_a_field" not in body
