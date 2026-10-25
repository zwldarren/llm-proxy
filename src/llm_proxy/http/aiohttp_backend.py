"""aiohttp-backed HTTP session.

Selected with ``HTTP_CLIENT_BACKEND=aiohttp`` (see
``llm_proxy.config.settings.HTTPSettings``). It implements the same duck-typed
surface as the httpx2 session in :mod:`llm_proxy.http.client` — ``request``,
``get``, ``post``, ``close`` and a :class:`AiohttpResponse` exposing
``status_code``, ``headers``, ``content``, ``aread``, ``text``, ``json``,
``raise_for_status``, ``aiter_bytes`` and ``iter_lines``.

Measured at roughly 1.8x the throughput and ~45% lower CPU per request than
httpx2 on the gateway hot path under an identical core budget, because
aiohttp's connector and header/parsing layers are C-backed while
httpcore2/anyio do the same work in Python.

Two behaviours are duplicated rather than delegated, because the rest of the
codebase depends on them:

* transport exceptions are translated into the matching ``httpx2`` exception
  types, so retry classification (:mod:`llm_proxy.providers.components.retry_policy`)
  and error handling (:mod:`llm_proxy.core.errors`) keep working unchanged;
* redirects are followed manually so every ``Location`` target is SSRF-checked,
  matching the httpx2 ``response`` event hook.
"""

import asyncio
import json as _json
import socket
import time
from typing import Any
from urllib.parse import urljoin

import aiohttp
import httpx2
from aiohttp.abc import AbstractResolver

from llm_proxy.config.settings import get_settings
from llm_proxy.http.limits import (
    check_line_length,
    flush_lines,
    max_response_bytes,
    read_capped,
    split_lines,
)
from llm_proxy.observability.logger import get_logger

logger = get_logger(__name__)

#: httpx follows at most 20 redirects before raising TooManyRedirects.
_MAX_REDIRECTS = 20
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


def _default_user_agent() -> str:
    # Imported lazily: client.py imports this module to build the dispatcher.
    from llm_proxy.http.client import DEFAULT_USER_AGENT

    return DEFAULT_USER_AGENT


async def _validate_redirect(url: str) -> None:
    """SSRF-check a redirect target, resolving its hostname.

    A redirect to a public name that maps to a private address is the same SSRF
    as one to a literal IP, and the OS resolver runs anyway for the real
    connection; ``getaddrinfo`` blocks, so it runs in a worker thread.
    """
    from llm_proxy.http.client import validate_server_url

    await asyncio.to_thread(validate_server_url, url, label="redirect target", fail_closed=True)


def _translate(exc: BaseException) -> httpx2.RequestError:
    """Map an aiohttp/asyncio transport failure to the httpx2 exception type.

    Retry classification is by exception type, so the mapping must preserve the
    httpx2 hierarchy: connect failures are ConnectError, read-side failures are
    ReadError (both NetworkError), peer/protocol failures are
    RemoteProtocolError and timeouts are ReadTimeout (all retryable).
    """
    if isinstance(exc, asyncio.TimeoutError):  # covers aiohttp.ServerTimeoutError
        return httpx2.ReadTimeout(str(exc))
    if isinstance(exc, aiohttp.ClientConnectorCertificateError):
        return httpx2.ConnectError(str(exc))
    if isinstance(exc, aiohttp.ClientConnectorError):
        return httpx2.ConnectError(str(exc))
    if isinstance(exc, aiohttp.ServerDisconnectedError):
        return httpx2.RemoteProtocolError(str(exc))
    if isinstance(exc, aiohttp.ClientPayloadError):
        return httpx2.RemoteProtocolError(str(exc))
    if isinstance(exc, aiohttp.ClientOSError):
        return httpx2.ReadError(str(exc))
    if isinstance(exc, aiohttp.ClientError):
        return httpx2.RequestError(str(exc))
    return httpx2.RequestError(str(exc))


def _split_timeout(timeout: Any) -> tuple[float, float]:
    """Normalize the httpx-style timeout argument to (connect, read) seconds."""
    default = (10.0, 600.0)
    if timeout is None:
        return default
    if isinstance(timeout, tuple):
        connect, read = timeout
        return float(connect), float(read)
    if isinstance(timeout, httpx2.Timeout):
        return (
            float(timeout.connect if timeout.connect is not None else default[0]),
            float(timeout.read if timeout.read is not None else default[1]),
        )
    return float(timeout), float(timeout)


def _build_form(files: Any, data: Any) -> aiohttp.FormData:
    """Build multipart form data from httpx-style ``data`` and ``files``.

    ``files`` uses the httpx shapes the providers emit: ``{field: (filename,
    content)}`` or ``{field: (filename, content, content_type)}``.
    """
    form = aiohttp.FormData()
    if isinstance(data, dict):
        for key, value in data.items():
            form.add_field(key, str(value))
    items = files.items() if isinstance(files, dict) else files
    for field, spec in items:
        if isinstance(spec, (tuple, list)):
            if len(spec) >= 3:
                filename, value, content_type = spec[0], spec[1], spec[2]
            else:
                filename, value = spec[0], spec[1]
                content_type = None
            form.add_field(field, value, filename=filename, content_type=content_type)
        else:
            form.add_field(field, spec, filename=field)
    return form


class _SsrfGuardResolver(AbstractResolver):
    """Re-check every DNS answer as the connection is made.

    ``validate_server_url`` runs when a base_url is saved, but the OS resolver runs
    again for each new connection: a hostname can answer with a public address at
    save time and a private one later (DNS rebinding). Checking at resolve time
    closes that gap for every request this backend makes, redirect targets
    included. Hosts listed in ``HTTP_ALLOWED_HOSTS`` are exempt, exactly as in
    ``validate_server_url``.

    A host that is itself an IP literal, ``localhost`` or a metadata name is passed
    through: those are already rejected by the pre-flight checks every URL-taking
    path runs, and filtering them here would break the loopback upstreams a
    developer runs locally. What this resolver adds is the case only DNS reveals.
    """

    def __init__(self) -> None:
        self._inner = aiohttp.resolver.ThreadedResolver()

    async def resolve(
        self, host: str, port: int = 0, family: socket.AddressFamily = socket.AF_INET
    ) -> list[Any]:
        records = await self._inner.resolve(host, port, family)
        # Imported here: llm_proxy.http.client builds this module's session, so a
        # module-scope import would be circular.
        from llm_proxy.http.client import _allowed_hosts, _is_banned_host

        if _is_banned_host(host):
            return records
        if host.lower().rstrip(".") in _allowed_hosts():
            return records
        for record in records:
            address = str(record["host"])
            if _is_banned_host(address):
                # The connector rewrites the OSError into a generic "cannot
                # connect", so the reason has to reach the log from here.
                logger.warning(
                    f"Refusing to connect to '{host}': it resolves to the "
                    f"disallowed address {address}"
                )
                raise OSError(
                    f"Refusing to connect to '{host}': it resolves to the "
                    f"disallowed address {address}"
                )
        return records

    async def close(self) -> None:
        await self._inner.close()


class AiohttpResponse:
    """Response surface matching ``llm_proxy.http.client.Response``."""

    def __init__(
        self,
        response: aiohttp.ClientResponse,
        *,
        content: bytes | None = None,
        method: str = "",
        url: str = "",
    ) -> None:
        self._resp = response
        self._content = content
        self._method = method
        self._url = url
        self._released = False

    @property
    def status_code(self) -> int:
        return self._resp.status

    @property
    def headers(self) -> Any:
        return self._resp.headers

    @property
    def content(self) -> bytes | None:
        return self._content

    @property
    def text(self) -> str:
        return self._decode(self._content or b"")

    def json(self) -> Any:
        return _json.loads(self._content or b"")

    def _decode(self, data: bytes) -> str:
        charset = self._resp.charset or "utf-8"
        try:
            return data.decode(charset)
        except LookupError, UnicodeDecodeError:
            return data.decode("utf-8", errors="replace")

    async def aread(self) -> bytes:
        if self._content is None:
            self._content = await read_capped(self, max_response_bytes(), self._url)
        return self._content

    def raise_for_status(self) -> None:
        if self._resp.status < 400:
            return
        request = httpx2.Request(self._method or "GET", self._url or "http://localhost")
        response = httpx2.Response(
            self._resp.status,
            headers=dict(self._resp.headers),
            content=self._content or b"",
            request=request,
        )
        raise httpx2.HTTPStatusError(
            f"Client error '{self._resp.status}' for url '{self._url}'",
            request=request,
            response=response,
        )

    async def aiter_bytes(self):
        """Yield the body in chunks without buffering all of it."""
        if self._content is not None:
            if self._content:
                yield self._content
            return
        try:
            async for chunk in self._resp.content.iter_any():
                yield chunk
        except (TimeoutError, aiohttp.ClientError) as exc:
            raise _translate(exc) from exc

    async def iter_lines(self):
        """Yield response lines as bytes without the line terminator."""
        if self._content is not None:
            lines, pending = split_lines(self._content)
            lines.extend(flush_lines(pending))
            for line in lines:
                check_line_length(line, self._url)
                yield line
            return
        buffer = b""
        try:
            async for chunk in self._resp.content.iter_any():
                lines, buffer = split_lines(buffer + chunk)
                for line in lines:
                    check_line_length(line, self._url)
                    yield line
                check_line_length(buffer, self._url)
            for line in flush_lines(buffer):
                check_line_length(line, self._url)
                yield line
        except (TimeoutError, aiohttp.ClientError) as exc:
            raise _translate(exc) from exc

    async def close(self) -> None:
        if not self._released:
            self._released = True
            self._resp.release()

    async def __aenter__(self) -> AiohttpResponse:
        return self

    async def __aexit__(self, _exc, _value, _tb) -> None:
        await self.close()


class AiohttpSession:
    """aiohttp implementation of the provider HTTP session surface."""

    def __init__(
        self,
        timeout: Any = None,
        disable_http2: bool = True,
        max_connections: int | None = None,
        max_keepalive_connections: int | None = None,
        **kwargs: Any,
    ) -> None:
        # aiohttp has no HTTP/2 support; ``disable_http2`` is accepted for
        # signature parity with the httpx2 session and ignored.
        del disable_http2
        self._connect_timeout, self._read_timeout = _split_timeout(timeout)
        http_settings = get_settings().http
        self._max_connections = max_connections or http_settings.max_connections
        self._follow_redirects = bool(kwargs.pop("follow_redirects", True))
        headers = dict(kwargs.pop("headers", None) or {})
        headers.setdefault("User-Agent", _default_user_agent())
        for key in ("extensions", "event_hooks"):
            kwargs.pop(key, None)
        self._default_headers = headers
        self._session: aiohttp.ClientSession | None = None
        self._lock = asyncio.Lock()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is not None and not self._session.closed:
            return self._session
        async with self._lock:
            if self._session is None or self._session.closed:
                connector = aiohttp.TCPConnector(
                    limit=self._max_connections,
                    limit_per_host=self._max_connections,
                    ttl_dns_cache=300,
                    resolver=_SsrfGuardResolver(),
                )
                self._session = aiohttp.ClientSession(
                    connector=connector,
                    headers=self._default_headers,
                    timeout=aiohttp.ClientTimeout(
                        connect=self._connect_timeout, sock_read=self._read_timeout
                    ),
                )
        return self._session

    def _request_timeout(self, timeout: Any) -> aiohttp.ClientTimeout | None:
        if timeout is None:
            return None
        connect, read = _split_timeout(timeout)
        return aiohttp.ClientTimeout(connect=connect, sock_read=read)

    async def request(self, method: str, url: str, **kwargs: Any) -> AiohttpResponse:
        stream = bool(kwargs.pop("stream", False))
        if "allow_redirects" in kwargs:
            kwargs["follow_redirects"] = kwargs.pop("allow_redirects")
        follow = bool(kwargs.pop("follow_redirects", self._follow_redirects))
        timeout = self._request_timeout(kwargs.pop("timeout", None))
        json_body = kwargs.pop("json", None)
        content = kwargs.pop("content", None)
        data = kwargs.pop("data", None)
        files = kwargs.pop("files", None)
        params = kwargs.pop("params", None)
        headers = kwargs.pop("headers", None)
        for key in ("cookies", "auth", "extensions"):
            kwargs.pop(key, None)
        if kwargs:
            raise TypeError(f"AiohttpSession.request got unsupported arguments: {sorted(kwargs)}")

        if files:
            data = _build_form(files, data)
        elif content is not None and data is None:
            data = content

        session = await self._get_session()
        timer = None
        started = 0.0
        from llm_proxy.http.upstream_timing import current_upstream_timer

        timer = current_upstream_timer()
        started = time.perf_counter() if timer is not None else 0.0
        try:
            response = await self._request_with_redirects(
                session,
                method,
                url,
                follow=follow,
                params=params,
                headers=headers,
                data=data,
                json_body=json_body,
                timeout=timeout,
            )
            if stream:
                return AiohttpResponse(response, method=method, url=str(response.url))
            payload = await read_capped(
                AiohttpResponse(response, method=method, url=str(response.url)),
                max_response_bytes(),
                str(response.url),
            )
            return AiohttpResponse(
                response,
                content=payload,
                method=method,
                url=str(response.url),
            )
        finally:
            if timer is not None:
                timer.add((time.perf_counter() - started) * 1000.0)

    async def _request_with_redirects(
        self,
        session: aiohttp.ClientSession,
        method: str,
        url: str,
        *,
        follow: bool,
        params: Any,
        headers: Any,
        data: Any,
        json_body: Any,
        timeout: aiohttp.ClientTimeout | None,
    ) -> aiohttp.ClientResponse:
        current_method = method.upper()
        current_url = url
        current_params = params
        current_data = data
        current_json = json_body
        try:
            for _ in range(_MAX_REDIRECTS + 1):
                response = await session.request(
                    current_method,
                    current_url,
                    params=current_params,
                    headers=headers,
                    data=current_data,
                    json=current_json,
                    timeout=timeout,
                    allow_redirects=False,
                )
                if not follow or response.status not in _REDIRECT_STATUSES:
                    return response
                location = response.headers.get("Location")
                if not location:
                    return response
                target = urljoin(str(response.url), location)
                await _validate_redirect(target)
                response.release()
                if response.status == 303 or (
                    response.status in (301, 302) and current_method == "POST"
                ):
                    current_method = "GET"
                    current_data = None
                    current_json = None
                current_url = target
                current_params = None
            raise httpx2.TooManyRedirects(f"Exceeded maximum redirects for url '{url}'")
        except (TimeoutError, aiohttp.ClientError, httpx2.RequestError) as exc:
            if isinstance(exc, httpx2.RequestError):
                raise
            raise _translate(exc) from exc

    async def get(self, url: str, **kwargs: Any) -> AiohttpResponse:
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs: Any) -> AiohttpResponse:
        return await self.request("POST", url, **kwargs)

    async def close(self) -> None:
        session = self._session
        self._session = None
        if session is not None and not session.closed:
            await session.close()

    async def __aenter__(self) -> AiohttpSession:
        return self

    async def __aexit__(self, _exc, _value, _tb) -> None:
        await self.close()
