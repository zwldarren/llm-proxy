"""Errors the outbound HTTP layer raises on its own.

Kept apart from :mod:`llm_proxy.http.client` so the aiohttp backend raises the
same types without importing the httpx2 session module (which imports it back,
lazily). Both derive from ``httpx2.RequestError`` so every existing handler keeps
treating them as client-side request failures rather than network errors: the
same bytes would come back on a retry, so retrying the provider cannot help.
"""

import httpx2


class UpstreamResponseTooLarge(httpx2.RequestError):
    """An upstream body — or one line of a stream — exceeded the configured cap.

    The proxy reads upstream bodies as a stream and stops here instead of
    buffering all of what a broken or hostile endpoint sent, so a single request
    cannot exhaust a worker's memory.
    """
