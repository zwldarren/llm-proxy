"""Tests for the HTTP client backend resolution policy.

``HTTP_CLIENT_BACKEND=auto`` (the default) prefers the faster aiohttp backend
but falls back to httpx2 when an outbound proxy is configured, because
aiohttp's connector does not read the proxy environment variables.

The settings model reads ``HTTP_*`` variables at instantiation, so each test
constructs a fresh ``HTTPSettings`` with the backend pinned explicitly and the
proxy variables cleared first.
"""

import pytest

from llm_proxy.config.settings import HTTPSettings

_PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


@pytest.fixture(autouse=True)
def _clear_proxy_env(monkeypatch):
    for var in _PROXY_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_auto_prefers_aiohttp_without_proxy() -> None:
    settings = HTTPSettings(HTTP_CLIENT_BACKEND="auto")
    assert settings.effective_client_backend() == "aiohttp"


@pytest.mark.parametrize("var", _PROXY_ENV_VARS)
def test_auto_falls_back_to_httpx_with_proxy(monkeypatch, var: str) -> None:
    monkeypatch.setenv(var, "http://proxy.local:3128")
    settings = HTTPSettings(HTTP_CLIENT_BACKEND="auto")
    assert settings.effective_client_backend() == "httpx2"


def test_explicit_backend_ignores_proxy(monkeypatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.local:3128")
    assert HTTPSettings(HTTP_CLIENT_BACKEND="aiohttp").effective_client_backend() == "aiohttp"
    assert HTTPSettings(HTTP_CLIENT_BACKEND="httpx2").effective_client_backend() == "httpx2"


def test_invalid_backend_is_rejected() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        HTTPSettings(HTTP_CLIENT_BACKEND="requests")
