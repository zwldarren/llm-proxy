"""Bounded downloads of client-supplied media, one budget per request.

Every path that turns a client-supplied URL into bytes the proxy forwards goes
through here: a chat message's ``image_url``, a document ``file_url`` (for a
provider that takes base64 only), a remote Gemini content part, and a Decisions
``input_image`` bridged onto a System One upstream. Each needs the protections
that only a shared object can give — a ceiling on how many fetches run at once,
and a ceiling on how much media one request may retain — on top of the per-URL
ones :func:`~llm_proxy.http.client.download_image_as_base64` already applies
(SSRF validation on every hop, a size cap). Without them a request that names 128
image URLs could make the proxy hold 128 × the per-download maximum in memory.

A caller decides what a missing image means. Bridging Decisions onto a System One
upstream fails the request, because dropping an image answers a question about
evidence the model never saw; chat media degrades that one part.
"""

import asyncio
from collections.abc import Iterable

from llm_proxy.config.settings import get_settings
from llm_proxy.core.exceptions import ValidationError
from llm_proxy.http.client import (
    DEFAULT_IMAGE_DOWNLOAD_TIMEOUT,
    AsyncSession,
    download_image_as_base64,
)

#: Fetches one request may have in flight at a time. The proxy pools connections
#: for many concurrent requests; bounding the fan-out of a single one keeps a
#: request that names many URLs from taking the whole pool.
DEFAULT_DOWNLOAD_CONCURRENCY = 4

#: One download's outcome: the data URL and media type, nothing (an empty body),
#: or the error that stopped it.
DownloadOutcome = tuple[str, str] | None | BaseException


class RequestDownloads:
    """The media downloads one request may run, and the bytes it may retain.

    ``per_download_bytes`` caps a single URL (``HTTP_MAX_DOWNLOAD_BYTES``);
    ``total_bytes`` caps what the request keeps inlined, measured on the inlined
    payloads (``HTTP_MAX_REQUEST_DOWNLOAD_BYTES``). Memory per request is
    therefore bounded by ``total_bytes`` plus the in-flight downloads, at most
    ``concurrency × per_download_bytes``.
    """

    def __init__(
        self,
        *,
        total_bytes: int | None = None,
        per_download_bytes: int | None = None,
        concurrency: int = DEFAULT_DOWNLOAD_CONCURRENCY,
        timeout: float = DEFAULT_IMAGE_DOWNLOAD_TIMEOUT,
    ) -> None:
        http = get_settings().http
        self.total_bytes = http.max_request_download_bytes if total_bytes is None else total_bytes
        self.per_download_bytes = (
            http.max_download_bytes if per_download_bytes is None else per_download_bytes
        )
        self._timeout = timeout
        self._slots = asyncio.Semaphore(max(1, concurrency))
        self._lock = asyncio.Lock()
        self._used = 0

    @property
    def used_bytes(self) -> int:
        """Inlined payload bytes downloaded so far."""
        return self._used

    async def fetch(self, client: AsyncSession, url: str) -> tuple[str, str] | None:
        """Download one URL, charging its payload to the request's budget.

        Raises:
            ValidationError: the URL fails its per-download checks, or inlining
                it would take the request past ``total_bytes``.
        """
        async with self._slots:
            downloaded = await download_image_as_base64(
                client, url, timeout=self._timeout, max_bytes=self.per_download_bytes
            )
            if downloaded is None:
                return None
            async with self._lock:
                if self._used + len(downloaded[0]) > self.total_bytes:
                    raise ValidationError(
                        message=(
                            f"The media in this request totals more than the "
                            f"{self.total_bytes}-byte limit "
                            f"(HTTP_MAX_REQUEST_DOWNLOAD_BYTES); '{url}' is over it"
                        )
                    )
                self._used += len(downloaded[0])
            return downloaded

    def retain(self, payload_bytes: int, *, url: str) -> None:
        """Charge copies of a payload the request keeps beyond the download.

        ``fetch`` charges one download per distinct URL. A URL that appears once
        per part is materialised once per part, so the caller charges the extra
        copies here — otherwise 128 references to one image retain 128 payloads
        while the downloader charged a single one.

        Raises:
            ValidationError: the extra copies take the request past ``total_bytes``.
        """
        if payload_bytes <= 0:
            return
        self._used += payload_bytes
        if self._used > self.total_bytes:
            raise ValidationError(
                message=(
                    f"The media in this request totals more than the "
                    f"{self.total_bytes}-byte limit "
                    f"(HTTP_MAX_REQUEST_DOWNLOAD_BYTES); '{url}' is repeated"
                )
            )

    async def gather(self, client: AsyncSession, urls: Iterable[str]) -> dict[str, DownloadOutcome]:
        """Download each distinct URL once, returning one outcome per URL.

        Errors are outcomes rather than exceptions so every caller can apply its
        own rule (fail the request, or drop that part) without unwinding the
        downloads that already succeeded.
        """
        unique = list(dict.fromkeys(urls))
        outcomes = await asyncio.gather(
            *(self.fetch(client, url) for url in unique), return_exceptions=True
        )
        return dict(zip(unique, outcomes, strict=True))


__all__ = [
    "DEFAULT_DOWNLOAD_CONCURRENCY",
    "DownloadOutcome",
    "RequestDownloads",
]
