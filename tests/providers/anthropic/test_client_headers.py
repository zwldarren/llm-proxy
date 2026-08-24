"""Tests for Claude Code client header capture and upstream forwarding."""

import pytest

from llm_proxy.providers.anthropic.adapter import AnthropicAdapter
from llm_proxy.providers.anthropic.client_headers import (
    CLAUDE_CODE_BETA,
    capture_client_headers,
    clear_client_headers,
    ensure_claude_code_beta,
    get_client_headers,
)


@pytest.fixture(autouse=True)
def _clean_contextvar():
    clear_client_headers()
    yield
    clear_client_headers()


class TestCaptureClientHeaders:
    def test_captures_whitelisted_headers(self):
        capture_client_headers(
            {
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "claude-code-20250219,oauth-2025-04-20",
                "x-app": "claude-code",
                "user-agent": "claude-cli/2.0.14 (external, cli)",
                "x-client-request-id": "req_1",
                "x-stainless-os": "macOS",
                "x-stainless-runtime": "node",
            }
        )
        captured = get_client_headers()
        assert captured["anthropic-version"] == "2023-06-01"
        assert captured["anthropic-beta"] == "claude-code-20250219,oauth-2025-04-20"
        assert captured["x-app"] == "claude-code"
        assert captured["user-agent"] == "claude-cli/2.0.14 (external, cli)"
        assert captured["x-client-request-id"] == "req_1"
        assert captured["x-stainless-os"] == "macOS"
        assert captured["x-stainless-runtime"] == "node"

    def test_drops_non_whitelisted_and_sensitive_headers(self):
        capture_client_headers(
            {
                "x-app": "claude-code",
                "authorization": "Bearer sk-client-secret",
                "x-api-key": "sk-client-secret",
                "x-goog-api-key": "sk-client-secret",
                "proxy-authorization": "Basic abc",
                "content-type": "application/json",
                "content-length": "123",
                "host": "api.anthropic.com",
                "accept": "text/event-stream",
                "accept-encoding": "gzip",
                "cookie": "session=abc",
                "x-forwarded-for": "1.2.3.4",
            }
        )
        captured = get_client_headers()
        assert captured == {"x-app": "claude-code"}

    def test_no_matching_headers_captures_empty(self):
        capture_client_headers({"x-custom-header": "value"})
        assert get_client_headers() == {}

    def test_capture_overwrites_previous_request(self):
        capture_client_headers({"x-app": "claude-code"})
        capture_client_headers({"x-client-request-id": "second"})
        assert get_client_headers() == {"x-client-request-id": "second"}


class TestEnsureClaudeCodeBeta:
    def test_none_gets_marker(self):
        assert ensure_claude_code_beta(None) == CLAUDE_CODE_BETA

    def test_keeps_client_list_when_marker_present(self):
        value = "claude-code-20250219,oauth-2025-04-20"
        assert ensure_claude_code_beta(value) == value

    def test_prepends_marker_when_missing(self):
        assert ensure_claude_code_beta("oauth-2025-04-20") == (
            "claude-code-20250219,oauth-2025-04-20"
        )


class TestAdapterHeaderMerge:
    def test_build_headers_merges_client_headers(self):
        adapter = AnthropicAdapter(api_key="sk-upstream")
        capture_client_headers(
            {
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "oauth-2025-04-20",
                "x-app": "claude-code",
                "user-agent": "claude-cli/2.0.14",
            }
        )
        headers = adapter._build_headers()
        assert headers["anthropic-version"] == "2023-06-01"
        assert headers["x-app"] == "claude-code"
        assert headers["user-agent"] == "claude-cli/2.0.14"
        # Beta is rebuilt to guarantee the claude-code marker.
        assert headers["anthropic-beta"] == "claude-code-20250219,oauth-2025-04-20"
        # Provider auth wins over anything captured.
        assert headers["x-api-key"] == "sk-upstream"

    def test_build_headers_injects_marker_when_client_sent_no_beta(self):
        adapter = AnthropicAdapter(api_key="sk-upstream")
        capture_client_headers({"x-app": "claude-code"})
        headers = adapter._build_headers()
        assert headers["anthropic-beta"] == CLAUDE_CODE_BETA

    def test_build_headers_without_capture_is_unchanged(self):
        adapter = AnthropicAdapter(api_key="sk-upstream")
        headers = adapter._build_headers()
        assert "x-app" not in headers
        assert "anthropic-beta" not in headers
        assert headers["x-api-key"] == "sk-upstream"

    def test_client_headers_never_override_provider_headers(self):
        adapter = AnthropicAdapter(api_key="sk-upstream", custom_headers={"x-app": "custom-app"})
        capture_client_headers({"x-app": "claude-code", "user-agent": "claude-cli/2.0"})
        headers = adapter._build_headers()
        # Case-insensitive collision: provider's own header is kept.
        assert headers["x-app"] == "custom-app"
        assert headers["user-agent"] == "claude-cli/2.0"
