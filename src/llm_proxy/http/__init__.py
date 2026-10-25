"""HTTP client package with connection pool management and request utilities.

The public names are PEP 810 lazy imports: importing this package costs nothing
until a symbol is actually used, so packages that merely sit on the import graph
of `llm_proxy.http` never pay for `httpx2`/`aiohttp`.
"""

lazy from llm_proxy.http.client import (
    DEFAULT_IMAGE_DOWNLOAD_TIMEOUT,
    HTTPClient,
    ProviderHTTPClientManager,
    download_image_as_base64,
    fetch_json,
)
lazy from llm_proxy.http.media import DEFAULT_DOWNLOAD_CONCURRENCY, RequestDownloads

__all__ = [
    "DEFAULT_DOWNLOAD_CONCURRENCY",
    "DEFAULT_IMAGE_DOWNLOAD_TIMEOUT",
    "HTTPClient",
    "ProviderHTTPClientManager",
    "RequestDownloads",
    "download_image_as_base64",
    "fetch_json",
]
