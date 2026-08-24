"""HTTP connection lifecycle for provider adapters."""

import asyncio
import inspect
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import orjson

from llm_proxy.http.client import AsyncSession
from llm_proxy.observability.logger import get_logger

if TYPE_CHECKING:
    from llm_proxy.core.exceptions import ProviderError

logger = get_logger(__name__)


class HttpTransport:
    """HTTP connection management for provider adapters.

    Manages client lifecycle (creation, pooling, cleanup), streaming POST
    connections, and HTTP response status checking.
    """

    def __init__(
        self,
        provider_name: str | None = None,
        connect_timeout: float = 10.0,
        read_timeout: float = 600.0,
        http_client: AsyncSession | None = None,
        disable_http2: bool = True,
        max_connections: int = 200,
        max_keepalive: int = 200,
    ):
        self._provider_name = provider_name
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout
        self._http_client = http_client
        self._owns_http_client = http_client is None
        self._client_lock = asyncio.Lock()
        self._disable_http2 = disable_http2
        self._max_connections = max_connections
        self._max_keepalive = max_keepalive

    async def get_client(self) -> AsyncSession:
        if self._http_client is not None:
            return self._http_client

        async with self._client_lock:
            if self._http_client is not None:
                return self._http_client

            logger.warning(
                f"Creating isolated HTTP client for {self._provider_name}. "
                "Consider injecting a shared client for connection pool reuse."
            )
            timeout = (self._connect_timeout, self._read_timeout)
            self._http_client = AsyncSession(
                timeout=timeout,
                disable_http2=self._disable_http2,
                max_connections=self._max_connections,
                max_keepalive_connections=self._max_keepalive,
            )
            self._owns_http_client = True
            return self._http_client

    async def close(self) -> None:
        if self._owns_http_client and self._http_client is not None:
            await self._http_client.close()
            self._http_client = None

    @asynccontextmanager
    async def streaming_post(
        self,
        client: AsyncSession,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        files: Any = None,
        timeout: tuple[float, float],
    ) -> AsyncIterator[Any]:
        """Open a streaming POST response and always close it when iteration stops."""
        post_kwargs: dict[str, Any] = {
            "headers": headers,
            "timeout": timeout,
            "stream": True,
        }
        if json is not None:
            post_kwargs["json"] = json
        if data is not None:
            post_kwargs["data"] = data
        if files is not None:
            post_kwargs["files"] = files

        response = await client.post(url, **post_kwargs)

        if hasattr(response, "__aenter__"):
            async with response:
                yield response
        else:
            try:
                yield response
            finally:
                close_fn = getattr(response, "aclose", None)
                if callable(close_fn):
                    result = close_fn()
                    if inspect.isawaitable(result):
                        await result

    async def check_response_status(
        self,
        response: Any,
        provider_name: str,
        parse_error_fn: Callable[[int, dict[str, Any]], ProviderError],
    ) -> None:
        """Check response status and raise ProviderError for error responses."""
        assert response.status_code is not None
        if response.status_code >= 400:
            error_body: dict[str, Any] = {}
            raw_text = ""
            try:
                raw_text = response.text
                error_body = orjson.loads(raw_text)
            except Exception:
                if raw_text:
                    logger.warning(
                        f"Provider returned non-JSON error body: {raw_text[:1000]}",
                    )
                    msg = f"HTTP {response.status_code}: {raw_text[:500]}"
                else:
                    msg = f"HTTP {response.status_code}: empty or unreadable response body"
                error_body = {"error": {"message": msg}}
            raise parse_error_fn(response.status_code, error_body)


__all__ = ["HttpTransport"]
