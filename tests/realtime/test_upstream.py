"""Unit tests for Realtime upstream URL and header derivation."""

from llm_proxy.config.types.provider import ProviderConfig
from llm_proxy.realtime.upstream import (
    build_realtime_url,
    build_upstream_headers,
)


def _provider(**overrides) -> ProviderConfig:
    """Build a ProviderConfig with realtime-test defaults."""
    defaults: dict = {"type": "openai", "api_key": "sk-upstream"}
    defaults.update(overrides)
    return ProviderConfig(**defaults)


class TestBuildRealtimeUrl:
    def test_default_openai_base_url(self):
        """No base_url configured → OpenAI default, https→wss."""
        url = build_realtime_url(_provider(), "gpt-realtime")
        assert url == "wss://api.openai.com/v1/realtime?model=gpt-realtime"

    def test_custom_base_url(self):
        """Custom base_url is used and scheme-normalized."""
        url = build_realtime_url(
            _provider(base_url="https://gateway.example.com/v1"), "gpt-realtime"
        )
        assert url == "wss://gateway.example.com/v1/realtime?model=gpt-realtime"

    def test_http_base_url_becomes_ws(self):
        url = build_realtime_url(_provider(base_url="http://localhost:8080/v1"), "gpt-realtime")
        assert url == "ws://localhost:8080/v1/realtime?model=gpt-realtime"

    def test_endpoint_base_url_full_endpoint(self):
        """A full /realtime endpoint URL is not double-suffixed."""
        url = build_realtime_url(
            _provider(endpoint_base_urls={"realtime": "https://api.openai.com/v1/realtime"}),
            "gpt-realtime",
        )
        assert url == "wss://api.openai.com/v1/realtime?model=gpt-realtime"

    def test_model_placeholder(self):
        """{model} placeholder is substituted and no model= is appended."""
        url = build_realtime_url(
            _provider(
                endpoint_base_urls={
                    "realtime": "https://rt.example.com/v1/realtime?deployment={model}"
                }
            ),
            "gpt-realtime",
        )
        assert url == "wss://rt.example.com/v1/realtime?deployment=gpt-realtime"

    def test_model_placeholder_with_query_keeps_other_params(self):
        url = build_realtime_url(
            _provider(
                endpoint_base_urls={
                    "realtime": "https://rt.example.com/v1/realtime?api-version=2025-01-01&deployment={model}"
                }
            ),
            "gpt-realtime",
        )
        assert url == (
            "wss://rt.example.com/v1/realtime?api-version=2025-01-01&deployment=gpt-realtime"
        )

    def test_model_is_url_encoded(self):
        url = build_realtime_url(_provider(), "my model/1")
        assert url == "wss://api.openai.com/v1/realtime?model=my%20model%2F1"

    def test_base_url_with_query_keeps_query_on_path(self):
        """A query-carrying base URL keeps its query and gains /realtime on the path."""
        url = build_realtime_url(
            _provider(base_url="https://rt.example.com/v1?api-version=2025-01-01"),
            "gpt-realtime",
        )
        assert url == ("wss://rt.example.com/v1/realtime?api-version=2025-01-01&model=gpt-realtime")

    def test_base_url_without_path(self):
        """A host-only base URL resolves to /realtime on the root path."""
        url = build_realtime_url(_provider(base_url="https://rt.example.com"), "gpt-realtime")
        assert url == "wss://rt.example.com/realtime?model=gpt-realtime"


class TestBuildUpstreamHeaders:
    def test_ga_model_gets_no_beta_header(self):
        """GA models must not receive the legacy beta header (GA migration)."""
        headers = build_upstream_headers(_provider(), model_name="gpt-realtime")
        assert headers["Authorization"] == "Bearer sk-upstream"
        assert "OpenAI-Beta" not in headers

    def test_legacy_preview_model_gets_beta_header(self):
        """Pre-GA realtime-preview models still require the beta header."""
        headers = build_upstream_headers(
            _provider(), model_name="gpt-4o-realtime-preview-2024-12-17"
        )
        assert headers["OpenAI-Beta"] == "realtime=v1"

    def test_mini_preview_model_gets_beta_header(self):
        headers = build_upstream_headers(
            _provider(), model_name="gpt-4o-mini-realtime-preview-2024-12-17"
        )
        assert headers["OpenAI-Beta"] == "realtime=v1"

    def test_openai_compatible_gets_no_beta_header(self):
        """The beta header is never sent to openai-compatible endpoints."""
        headers = build_upstream_headers(
            _provider(type="openai-compatible"),
            model_name="gpt-4o-realtime-preview-2024-12-17",
        )
        assert headers["Authorization"] == "Bearer sk-upstream"
        assert "OpenAI-Beta" not in headers

    def test_no_model_name_gets_no_beta_header(self):
        """Unknown model → GA behavior (no beta header)."""
        headers = build_upstream_headers(_provider())
        assert "OpenAI-Beta" not in headers

    def test_safety_identifier_forwarded(self):
        headers = build_upstream_headers(
            _provider(), model_name="gpt-realtime", safety_identifier="hashed-user-id"
        )
        assert headers["OpenAI-Safety-Identifier"] == "hashed-user-id"

    def test_custom_headers_merged(self):
        headers = build_upstream_headers(
            _provider(custom_headers={"X-Custom": "yes", "Authorization": "Bearer custom"}),
            model_name="gpt-realtime",
        )
        assert headers["X-Custom"] == "yes"
        # custom_headers win on conflicts
        assert headers["Authorization"] == "Bearer custom"

    def test_custom_headers_can_force_beta_header(self):
        """custom_headers are an escape hatch to force the beta header."""
        headers = build_upstream_headers(
            _provider(custom_headers={"OpenAI-Beta": "realtime=v1"}),
            model_name="gpt-realtime",
        )
        assert headers["OpenAI-Beta"] == "realtime=v1"
