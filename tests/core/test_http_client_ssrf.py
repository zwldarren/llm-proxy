"""Tests for SSRF protections in the HTTP client download helper."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_proxy.core.exceptions import ValidationError
from llm_proxy.http.client import (
    _is_banned_host,
    download_image_as_base64,
    validate_server_url,
)


def test_is_banned_host_blocks_private_and_localhost() -> None:
    """Banned hosts include private IPs and localhost."""
    assert _is_banned_host("127.0.0.1") is True
    assert _is_banned_host("10.0.0.1") is True
    assert _is_banned_host("localhost") is True
    assert _is_banned_host("169.254.169.254") is True
    assert _is_banned_host("example.com") is False


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/",
        "http://localhost:8080/v1/chat/completions",
        "http://127.0.0.1:8080/v1",
        "http://user:pass@example.com/v1",
        "ftp://example.com/v1",
    ],
)
def test_validate_server_url_blocks_private_and_malformed(url: str) -> None:
    """Provider/tracing base URLs with private hosts or credentials are rejected."""
    with pytest.raises(ValidationError):
        validate_server_url(url, label="provider base_url")


def test_validate_server_url_allows_public_http() -> None:
    """Public http(s) URLs are accepted even if DNS cannot be resolved."""
    # This fake public-looking URL should pass because DNS resolution fails and
    # the fallback permits it (so tests work in environments without DNS).
    validate_server_url("https://api.example.com/v1", label="provider base_url")


@pytest.mark.asyncio
async def test_redirect_hook_blocks_private_target() -> None:
    """A redirect to a private address is rejected by the event hook."""
    from llm_proxy.http.client import AsyncSession

    session = AsyncSession()
    hook = session._client._event_hooks["response"][0]
    response = MagicMock()
    response.has_redirect_location = True
    response.headers = {"location": "http://169.254.169.254/"}
    response.request.url = "http://example.com/"

    with pytest.raises(ValidationError):
        await hook(response)


@pytest.mark.asyncio
async def test_redirect_hook_allows_public_target() -> None:
    """A redirect to another public address is allowed."""
    from llm_proxy.http.client import AsyncSession

    session = AsyncSession()
    hook = session._client._event_hooks["response"][0]
    response = MagicMock()
    response.has_redirect_location = True
    response.headers = {"location": "https://other.example.com/"}
    response.request.url = "http://example.com/"

    # should not raise
    await hook(response)


@pytest.mark.asyncio
async def test_redirect_hook_allows_non_redirect_response() -> None:
    """A normal response without a Location header is ignored."""
    from llm_proxy.http.client import AsyncSession

    session = AsyncSession()
    hook = session._client._event_hooks["response"][0]
    response = MagicMock()
    response.has_redirect_location = False
    response.headers = {}
    response.request.url = "http://localhost:8888/"

    # should not raise, even though the request URL points at localhost
    await hook(response)


@pytest.mark.asyncio
async def test_direct_localhost_request_is_allowed() -> None:
    """Direct requests to localhost/private hosts are allowed by the session hook.

    The redirect hook only inspects Location headers, so normal outbound
    requests to admin-configured endpoints (web search, tracing, providers)
    are not blocked.
    """
    from llm_proxy.http.client import AsyncSession

    session = AsyncSession()
    hook = session._client._event_hooks["response"][0]
    response = MagicMock()
    response.has_redirect_location = False
    response.headers = {}
    response.request.url = "http://localhost:8888/search"

    # should not raise
    await hook(response)


@pytest.mark.asyncio
async def test_download_image_as_base64_validates_url_before_request() -> None:
    """download_image_as_base64 rejects private URLs before calling the client."""
    mock_client = MagicMock()
    mock_client.get = AsyncMock()

    with pytest.raises(ValidationError):
        await download_image_as_base64(mock_client, "http://169.254.169.254/")

    mock_client.get.assert_not_awaited()


@pytest.mark.asyncio
async def test_download_image_as_base64_allows_public_url() -> None:
    """download_image_as_base64 calls the client for public URLs."""
    mock_response = MagicMock()
    mock_response.content = b"\x89PNG\r\n\x1a\n"
    mock_response.headers = {"content-type": "image/png"}
    mock_response.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    result = await download_image_as_base64(mock_client, "http://example.com/image.png")
    assert result is not None
    assert result[1] == "image/png"
    mock_client.get.assert_awaited_once()
