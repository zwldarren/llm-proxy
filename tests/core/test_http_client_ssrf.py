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
        "http://0.0.0.0:8080/v1",
        "http://[::]:8080/v1",
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
    from llm_proxy.http.client import _HttpxSession

    session = _HttpxSession()
    hook = session._client._event_hooks["response"][0]
    response = MagicMock()
    response.has_redirect_location = True
    response.headers = {"location": "http://169.254.169.254/"}
    response.request.url = "http://example.com/"

    with pytest.raises(ValidationError):
        await hook(response)


@pytest.mark.asyncio
async def test_redirect_hook_allows_public_target(public_dns) -> None:
    """A redirect to another public address is allowed."""
    from llm_proxy.http.client import _HttpxSession

    session = _HttpxSession()
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
    from llm_proxy.http.client import _HttpxSession

    session = _HttpxSession()
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
    from llm_proxy.http.client import _HttpxSession

    session = _HttpxSession()
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
async def test_download_image_as_base64_allows_public_url(mock_response_cls) -> None:
    """download_image_as_base64 calls the client for public URLs."""
    mock_client = MagicMock()
    mock_client.get = AsyncMock(
        return_value=mock_response_cls(
            stream_chunks=[b"\x89PNG\r\n\x1a\n"], headers={"content-type": "image/png"}
        )
    )

    result = await download_image_as_base64(mock_client, "http://example.com/image.png")
    assert result is not None
    assert result[1] == "image/png"
    mock_client.get.assert_awaited_once()


@pytest.fixture
def public_dns(monkeypatch):
    """Pin the DNS verdict so these tests never depend on resolving example.com."""
    monkeypatch.setattr("llm_proxy.http.client._is_banned_resolved_ip", lambda *a, **k: False)


@pytest.mark.asyncio
async def test_download_follows_a_public_redirect(public_dns, mock_response_cls) -> None:
    """A redirecting image host is a normal request, not a failed one.

    http→https upgrades, a CDN's signing hop and link shorteners all answer 3xx;
    the download follows them instead of turning a valid image into an error.
    """
    mock_client = MagicMock()
    mock_client.get = AsyncMock(
        side_effect=[
            mock_response_cls(302, headers={"location": "/final.png"}),
            mock_response_cls(
                stream_chunks=[b"\x89PNG\r\n\x1a\n"], headers={"content-type": "image/png"}
            ),
        ]
    )

    result = await download_image_as_base64(mock_client, "https://example.com/image")

    assert result is not None
    assert result[1] == "image/png"
    assert mock_client.get.await_count == 2
    # The relative Location is resolved against the URL that answered it.
    assert mock_client.get.await_args_list[1].args[0] == "https://example.com/final.png"


@pytest.mark.asyncio
async def test_download_rejects_a_redirect_to_a_private_address(mock_response_cls) -> None:
    """Every hop is validated: a redirect is as attacker-controlled as the URL."""
    redirect = mock_response_cls(302, headers={"location": "http://127.0.0.1/x.png"})
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=redirect)

    with pytest.raises(ValidationError):
        await download_image_as_base64(mock_client, "http://example.com/image.png")

    # The private target is rejected before it is fetched.
    mock_client.get.assert_awaited_once()
    assert redirect.closed is True


@pytest.mark.asyncio
async def test_download_gives_up_after_too_many_redirects(public_dns, mock_response_cls) -> None:
    """A redirect loop fails with a clear error instead of hammering a host."""
    from llm_proxy.http.client import MAX_DOWNLOAD_REDIRECTS

    responses = [
        mock_response_cls(302, headers={"location": "/loop"})
        for _ in range(MAX_DOWNLOAD_REDIRECTS + 1)
    ]
    mock_client = MagicMock()
    mock_client.get = AsyncMock(side_effect=responses)

    with pytest.raises(ValidationError, match="redirected more than"):
        await download_image_as_base64(mock_client, "https://example.com/image.png")

    assert mock_client.get.await_count == MAX_DOWNLOAD_REDIRECTS + 1
    assert all(response.closed for response in responses)


@pytest.mark.asyncio
async def test_download_rejects_a_redirect_without_a_location(
    public_dns, mock_response_cls
) -> None:
    """A redirect with nothing to follow is a broken URL, not a silent success."""
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response_cls(302))

    with pytest.raises(ValidationError, match="no Location header"):
        await download_image_as_base64(mock_client, "https://example.com/image.png")


@pytest.mark.asyncio
async def test_download_rejects_a_declared_size_over_the_cap(public_dns, mock_response_cls) -> None:
    """A Content-Length over the cap fails before the body is transferred."""
    response = mock_response_cls(
        headers={"content-type": "image/png", "content-length": str(21 * 1024 * 1024)}
    )
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=response)

    with pytest.raises(ValidationError, match="over the .*download limit"):
        await download_image_as_base64(mock_client, "https://example.com/huge.png")

    assert response.closed is True


@pytest.mark.asyncio
async def test_download_rejects_a_chunked_body_over_the_cap(public_dns, mock_response_cls) -> None:
    """A chunked response with no length header is capped while streaming."""
    response = mock_response_cls(
        stream_chunks=[b"x" * 6, b"x" * 6], headers={"content-type": "image/png"}
    )
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=response)

    with pytest.raises(ValidationError, match="exceeds the .*download limit"):
        await download_image_as_base64(mock_client, "https://example.com/huge.png", max_bytes=10)

    assert response.closed is True


@pytest.mark.asyncio
async def test_download_accepts_a_body_exactly_at_the_cap(public_dns, mock_response_cls) -> None:
    """The cap is inclusive: an image of exactly the limit still goes through."""
    response = mock_response_cls(
        stream_chunks=[b"x" * 5, b"x" * 5], headers={"content-type": "image/png"}
    )
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=response)

    result = await download_image_as_base64(
        mock_client, "https://example.com/image.png", max_bytes=10
    )

    assert result is not None
    assert result[0].endswith("eHh4eHh4eHh4eA==")  # 10 bytes of "x"
    assert response.closed is True


@pytest.mark.asyncio
async def test_download_cap_defaults_to_the_setting(
    public_dns, monkeypatch, mock_response_cls
) -> None:
    """Callers get the configured limit without passing one."""
    monkeypatch.setattr("llm_proxy.http.client.max_download_bytes", lambda: 4)
    response = mock_response_cls(stream_chunks=[b"x" * 5], headers={"content-type": "image/png"})
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=response)

    with pytest.raises(ValidationError, match="exceeds the .*download limit"):
        await download_image_as_base64(mock_client, "https://example.com/image.png")


@pytest.mark.asyncio
async def test_download_treats_an_empty_body_as_no_image(public_dns, mock_response_cls) -> None:
    """An empty download returns None so callers can report it, not a blank data URL."""
    mock_client = MagicMock()
    mock_client.get = AsyncMock(
        return_value=mock_response_cls(headers={"content-type": "image/png"})
    )

    assert await download_image_as_base64(mock_client, "https://example.com/image.png") is None


@pytest.mark.asyncio
async def test_redirect_hook_resolves_the_target_hostname(monkeypatch) -> None:
    """A public name that resolves to a private address is refused, not followed."""
    from llm_proxy.http.client import _HttpxSession

    monkeypatch.setattr(
        "llm_proxy.http.client._is_banned_resolved_ip",
        lambda host, **kwargs: host == "internal.example.com",
    )
    session = _HttpxSession()
    hook = session._client._event_hooks["response"][0]
    response = MagicMock()
    response.has_redirect_location = True
    response.headers = {"location": "http://internal.example.com/"}
    response.request.url = "https://example.com/start"

    with pytest.raises(ValidationError):
        await hook(response)
