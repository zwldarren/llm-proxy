"""Tests for Codex client header capture and upstream forwarding."""

import pytest

from llm_proxy.providers.openai.adapter import OpenAIAdapter
from llm_proxy.providers.openai.client_headers import (
    capture_client_headers,
    clear_client_headers,
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
                "originator": "codex_cli_rs",
                "openai-beta": "responses=experimental",
                "conversation_id": "conv_1",
                "session_id": "sess_1",
                "chatgpt-account-id": "acct_1",
                "x-codex-window-id": "win_1",
                "x-codex-turn-metadata": "meta",
                "user-agent": "codex_cli_rs/1.0",
                "openai-organization": "org_1",
                "openai-project": "proj_1",
                "x-client-request-id": "req_1",
                "x-stainless-os": "macOS",
                "x-stainless-runtime": "node",
            }
        )
        captured = get_client_headers()
        assert captured["originator"] == "codex_cli_rs"
        assert captured["openai-beta"] == "responses=experimental"
        assert captured["conversation_id"] == "conv_1"
        assert captured["session_id"] == "sess_1"
        assert captured["chatgpt-account-id"] == "acct_1"
        assert captured["x-codex-window-id"] == "win_1"
        assert captured["x-codex-turn-metadata"] == "meta"
        # Client fingerprint headers are forwarded verbatim.
        assert captured["user-agent"] == "codex_cli_rs/1.0"
        assert captured["openai-organization"] == "org_1"
        assert captured["openai-project"] == "proj_1"
        assert captured["x-client-request-id"] == "req_1"
        assert captured["x-stainless-os"] == "macOS"
        assert captured["x-stainless-runtime"] == "node"

    def test_drops_non_whitelisted_and_sensitive_headers(self):
        capture_client_headers(
            {
                "originator": "codex_cli_rs",
                "authorization": "Bearer sk-client-secret",
                "x-api-key": "sk-client-secret",
                "proxy-authorization": "Basic abc",
                "content-type": "application/json",
                "content-length": "123",
                "host": "api.openai.com",
                "cookie": "session=abc",
                "x-forwarded-for": "1.2.3.4",
            }
        )
        captured = get_client_headers()
        assert captured == {"originator": "codex_cli_rs"}

    def test_no_matching_headers_captures_empty(self):
        capture_client_headers({"x-custom-header": "value"})
        assert get_client_headers() == {}

    def test_capture_overwrites_previous_request(self):
        capture_client_headers({"originator": "first"})
        capture_client_headers({"session_id": "second"})
        assert get_client_headers() == {"session_id": "second"}


class TestAdapterHeaderMerge:
    def test_build_headers_merges_client_headers(self):
        adapter = OpenAIAdapter(api_key="sk-upstream")
        capture_client_headers(
            {
                "originator": "codex_cli_rs",
                "openai-beta": "responses=experimental",
                "session_id": "sess_1",
            }
        )
        headers = adapter._build_headers()
        assert headers["originator"] == "codex_cli_rs"
        assert headers["openai-beta"] == "responses=experimental"
        assert headers["session_id"] == "sess_1"
        # Provider auth wins over anything captured.
        assert headers["Authorization"] == "Bearer sk-upstream"

    def test_build_headers_without_capture_is_unchanged(self):
        adapter = OpenAIAdapter(api_key="sk-upstream")
        headers = adapter._build_headers()
        assert "originator" not in headers
        assert headers["Authorization"] == "Bearer sk-upstream"

    def test_client_headers_never_override_provider_headers(self):
        adapter = OpenAIAdapter(api_key="sk-upstream", custom_headers={"X-Custom": "provider"})
        capture_client_headers({"x-custom": "client-value", "originator": "codex"})
        headers = adapter._build_headers()
        # Case-insensitive collision: provider's own header is kept.
        assert headers["X-Custom"] == "provider"
        assert headers["originator"] == "codex"
