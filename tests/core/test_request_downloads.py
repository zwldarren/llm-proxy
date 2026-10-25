"""Tests for the per-request media download budget.

``RequestDownloads`` is the chokepoint for the media a request makes the proxy
fetch: it bounds the fan-out, charges every inlined payload to a request-wide
total, and hands failures back as outcomes so each caller keeps its own rule.
"""

import asyncio

import pytest

from llm_proxy.core.exceptions import ValidationError
from llm_proxy.http.media import DEFAULT_DOWNLOAD_CONCURRENCY, RequestDownloads


def _payload(size: int, media_type: str = "image/png") -> tuple[str, str]:
    return f"data:{media_type};base64,{'x' * size}", media_type


@pytest.fixture
def fake_download(monkeypatch):
    """Install a download function and return the calls it recorded."""

    calls: list[tuple[str, dict]] = []
    responses: dict[str, tuple[str, str] | BaseException] = {}

    async def download(client, url, **kwargs):
        calls.append((url, kwargs))
        outcome = responses.get(url, _payload(10))
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr("llm_proxy.http.media.download_image_as_base64", download)
    return calls, responses


async def test_gather_downloads_each_url_once(fake_download) -> None:
    """A URL repeated across the request is fetched once and shared."""
    calls, _ = fake_download
    outcomes = await RequestDownloads().gather(
        None, ["https://a/1.png", "https://a/2.png", "https://a/1.png"]
    )

    assert [url for url, _ in calls] == ["https://a/1.png", "https://a/2.png"]
    assert list(outcomes) == ["https://a/1.png", "https://a/2.png"]
    assert all(isinstance(outcome, tuple) for outcome in outcomes.values())


async def test_total_budget_rejects_the_overflowing_image(fake_download) -> None:
    """Images are charged to the request total, not just the per-URL cap."""
    _, responses = fake_download
    responses["https://a/first.png"] = _payload(60)
    responses["https://a/second.png"] = _payload(60)
    downloads = RequestDownloads(total_bytes=100, per_download_bytes=100)

    outcomes = await downloads.gather(None, ["https://a/first.png", "https://a/second.png"])

    assert isinstance(outcomes["https://a/first.png"], tuple)
    second = outcomes["https://a/second.png"]
    assert isinstance(second, ValidationError)
    assert "HTTP_MAX_REQUEST_DOWNLOAD_BYTES" in second.message
    assert downloads.used_bytes == len(_payload(60)[0])


async def test_total_budget_accepts_payloads_that_fit(fake_download) -> None:
    """The budget is inclusive: exactly ``total_bytes`` inlined is allowed."""
    downloads = RequestDownloads(total_bytes=2 * len(_payload(10)[0]), per_download_bytes=100)

    outcomes = await downloads.gather(None, ["https://a/1.png", "https://a/2.png"])

    assert all(isinstance(outcome, tuple) for outcome in outcomes.values())
    assert downloads.used_bytes == 2 * len(_payload(10)[0])


async def test_concurrency_is_bounded(fake_download, monkeypatch) -> None:
    """A request naming many URLs runs a bounded number of fetches at once."""
    in_flight = 0
    peak = 0

    async def download(client, url, **kwargs):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0)
        in_flight -= 1
        return _payload(1)

    monkeypatch.setattr("llm_proxy.http.media.download_image_as_base64", download)
    urls = [f"https://a/{index}.png" for index in range(12)]
    downloads = RequestDownloads(total_bytes=10**6, concurrency=3)

    await downloads.gather(None, urls)

    assert peak == 3
    assert DEFAULT_DOWNLOAD_CONCURRENCY >= 1


async def test_per_download_limit_is_forwarded(fake_download) -> None:
    """The per-URL cap reaches the downloader instead of being re-derived."""
    calls, _ = fake_download

    await RequestDownloads(total_bytes=10**6, per_download_bytes=1234).gather(
        None, ["https://a/1.png"]
    )

    assert calls[0][1]["max_bytes"] == 1234


async def test_failures_are_outcomes_not_raises(fake_download) -> None:
    """One unreachable URL does not unwind the downloads that already worked."""
    _, responses = fake_download
    responses["https://a/bad.png"] = RuntimeError("blocked")

    outcomes = await RequestDownloads().gather(None, ["https://a/bad.png", "https://a/good.png"])

    assert isinstance(outcomes["https://a/bad.png"], RuntimeError)
    assert isinstance(outcomes["https://a/good.png"], tuple)


async def test_an_empty_body_is_not_charged(fake_download) -> None:
    """A download that returned nothing costs nothing against the budget."""
    _, responses = fake_download
    responses["https://a/empty.png"] = None
    downloads = RequestDownloads(total_bytes=100)

    outcomes = await downloads.gather(None, ["https://a/empty.png"])

    assert outcomes["https://a/empty.png"] is None
    assert downloads.used_bytes == 0


def test_defaults_come_from_the_http_settings() -> None:
    """Both caps default to the values documented under HTTP_*."""
    from llm_proxy.config.settings import get_settings

    downloads = RequestDownloads()
    http = get_settings().http

    assert downloads.total_bytes == http.max_request_download_bytes
    assert downloads.per_download_bytes == http.max_download_bytes
    assert downloads.total_bytes == 64 * 1024 * 1024
    assert downloads.per_download_bytes == 20 * 1024 * 1024
