"""HTTP client wrapping httpx2 with connection pool management."""

import asyncio
import base64
import ipaddress
from importlib.metadata import version as _distribution_version
from types import MappingProxyType
from typing import Any, Final
from urllib.parse import urljoin, urlparse

import httpx2

from llm_proxy.config.settings import get_settings
from llm_proxy.core.exceptions import ValidationError
from llm_proxy.core.utils import quiet_aclose
from llm_proxy.observability.logger import get_logger

logger = get_logger(__name__)

# Identify ourselves to upstreams instead of leaking the httpx default UA.
# Request-level headers (e.g. a client fingerprint UA merged on the native
# passthrough path) override this client-level default per httpx semantics.
try:
    _LLM_PROXY_VERSION = _distribution_version("llm-proxy")
except Exception:
    _LLM_PROXY_VERSION = "dev"

DEFAULT_USER_AGENT: Final[str] = f"llm-proxy/{_LLM_PROXY_VERSION}"


# RFC 1918 + loopback + link-local + multicast + reserved blocks
_BANNED_NETWORKS: Final[tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]] = (
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("100.64.0.0/10"),  # CGNAT / shared address space
    ipaddress.ip_network("192.0.0.0/24"),
    ipaddress.ip_network("192.0.2.0/24"),  # TEST-NET-1
    ipaddress.ip_network("198.18.0.0/15"),  # benchmark/testing
    ipaddress.ip_network("198.51.100.0/24"),  # TEST-NET-2
    ipaddress.ip_network("203.0.113.0/24"),  # TEST-NET-3
    ipaddress.ip_network("224.0.0.0/4"),  # multicast
    ipaddress.ip_network("240.0.0.0/4"),  # reserved
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),  # unique local
    ipaddress.ip_network("fe80::/10"),  # link-local
    ipaddress.ip_network("::ffff:127.0.0.0/104"),  # IPv4-mapped loopback
    ipaddress.ip_network("::ffff:10.0.0.0/104"),  # IPv4-mapped private
    ipaddress.ip_network("::ffff:172.16.0.0/108"),  # IPv4-mapped private
    ipaddress.ip_network("::ffff:192.168.0.0/112"),  # IPv4-mapped private
    ipaddress.ip_network("::ffff:169.254.0.0/112"),  # IPv4-mapped link-local
    ipaddress.ip_network("::ffff:100.64.0.0/106"),  # IPv4-mapped CGNAT
    ipaddress.ip_network("::ffff:192.0.0.0/120"),  # IPv4-mapped IETF
    ipaddress.ip_network("::ffff:192.0.2.0/120"),  # IPv4-mapped TEST-NET-1
    ipaddress.ip_network("::ffff:198.18.0.0/111"),  # IPv4-mapped benchmark
    ipaddress.ip_network("::ffff:198.51.100.0/120"),  # IPv4-mapped TEST-NET-2
    ipaddress.ip_network("::ffff:203.0.113.0/120"),  # IPv4-mapped TEST-NET-3
    ipaddress.ip_network("::ffff:224.0.0.0/100"),  # IPv4-mapped multicast
    ipaddress.ip_network("::ffff:240.0.0.0/100"),  # IPv4-mapped reserved
)

# Hosts that should never be fetched by the server, regardless of DNS.
_BANNED_HOSTS: Final[frozenset[str]] = frozenset(
    {
        "localhost",
        "metadata.google.internal",
        "metadata.platform.internal",
        "metadata.internal",
        "metadata.kubernetes.internal",
        "host.docker.internal",
        "169.254.169.254",
    }
)


def _is_banned_host(host: str) -> bool:
    """Check whether a URL host is banned from server-side fetching.

    Bans include loopback/private/link-local IPs, test networks, common
    cloud metadata endpoints, and the literal host ``localhost``.
    """
    if not host:
        return True
    host_lower = host.lower().rstrip(".")
    if host_lower in _BANNED_HOSTS:
        return True
    try:
        addr = ipaddress.ip_address(host_lower)
        for network in _BANNED_NETWORKS:
            if addr in network:
                return True
    except ValueError:
        # Not an IP address; hostname. We do not resolve DNS here, so hostnames
        # that map to private IPs can only be blocked by an allowlist. For now,
        # permit public hostnames and let the application's egress firewall
        # enforce network-level restrictions.
        pass

    return False


def validate_url_format(url: str, label: str = "URL") -> None:
    """Validate that a URL has an allowed scheme, a host, and no credentials.

    This is intentionally lighter than the full SSRF check used for
    user-supplied download URLs. It is suitable for admin-controlled
    outbound endpoints such as tracing provider base URLs, where private or
    self-hosted hosts are valid.
    """
    if not url or not isinstance(url, str):
        raise ValidationError(message=f"Invalid {label}: empty or non-string value")

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValidationError(
            message=f"Invalid {label} scheme '{parsed.scheme}': only http/https are allowed"
        )

    if parsed.username is not None:
        raise ValidationError(
            message=f"{label.capitalize()} containing credentials are not allowed"
        )

    if parsed.hostname is None:
        raise ValidationError(message=f"Invalid {label}: missing host")


def _validate_url_common(url: str, label: str = "URL") -> tuple[str, str]:
    """Common URL validation: parse, scheme, credentials, literal host ban.

    Returns the normalized URL and the hostname for further checks.
    """
    validate_url_format(url, label=label)

    parsed = urlparse(url)
    host = parsed.hostname
    if host is None or _is_banned_host(host):
        raise ValidationError(message=f"{label.capitalize()} host is not allowed: {host}")

    return url, host


def _is_banned_resolved_ip(host: str) -> bool:
    """Resolve a hostname and check whether any resolved IP is banned.

    This is synchronous DNS resolution intended for infrequent configuration
    validation, not for the hot request path. Returns True if any A/AAAA record
    falls into a banned network. Returns False when resolution fails, logging a
    warning so that the URL is not rejected merely because DNS is unavailable.
    """
    import socket

    try:
        addrinfo = socket.getaddrinfo(host, None)
    except OSError as e:
        logger.warning(f"Could not resolve host '{host}' for SSRF validation; allowing URL: {e}")
        return False

    seen_ips: set[str] = set()
    for _, _, _, _, sockaddr in addrinfo:
        ip = str(sockaddr[0])
        if ip in seen_ips:
            continue
        seen_ips.add(ip)
        if _is_banned_host(ip):
            return True
    return False


def validate_server_url(url: str, *, label: str = "URL", resolve_dns: bool = True) -> None:
    """Validate a server-side outbound URL before storing or fetching.

    Rejects non-http(s) schemes, URLs containing credentials, literal banned
    hosts/IPs, and (by default) hostnames that resolve to banned networks. This
    prevents SSRF via base_url configurations such as cloud metadata endpoints or
    private services.

    Args:
        url: The URL to validate.
        label: Human-readable label for error messages.
        resolve_dns: If True, resolve the hostname and reject private/internal
            resolved IPs. Disable only when DNS is unavailable or the URL is
            already known to be safe.
    """
    _, host = _validate_url_common(url, label=label)
    if resolve_dns and _is_banned_resolved_ip(host):
        raise ValidationError(
            message=f"{label.capitalize()} '{url}' resolves to a private or internal address"
        )


_IMAGE_EXTENSION_TO_MIME: Final[MappingProxyType[str, str]] = MappingProxyType(
    {
        # Image formats
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp",
        "svg": "image/svg+xml",
        "bmp": "image/bmp",
        "tiff": "image/tiff",
        "tif": "image/tiff",
        # Audio formats
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "ogg": "audio/ogg",
        "flac": "audio/flac",
        "aac": "audio/aac",
        "m4a": "audio/mp4",
        "opus": "audio/opus",
        "webm": "audio/webm",
        # Document formats
        "pdf": "application/pdf",
        "txt": "text/plain",
        "html": "text/html",
        "csv": "text/csv",
        "json": "application/json",
        "xml": "text/xml",
        "md": "text/markdown",
        # Video formats
        "mp4": "video/mp4",
        "avi": "video/x-msvideo",
        "mov": "video/quicktime",
        "wmv": "video/x-ms-wmv",
        "mkv": "video/x-matroska",
    }
)


class Response:
    def __init__(self, httpx2_response, context_manager=None):
        self._resp = httpx2_response
        self._ctx = context_manager
        self._content_read = False
        self._content = b""

    @property
    def status_code(self):
        return self._resp.status_code

    @property
    def headers(self):
        return self._resp.headers

    @property
    def content(self):
        return self._resp.content

    async def aread(self):
        if not self._content_read:
            self._content = await self._resp.aread()
            self._content_read = True
        return self._content

    @property
    def text(self):
        return self._resp.text

    def json(self):
        return self._resp.json()

    def raise_for_status(self):
        self._resp.raise_for_status()

    async def iter_lines(self):
        lines = self._resp.aiter_lines()
        try:
            async for line in lines:
                if isinstance(line, str):
                    yield line.encode("utf-8")
                else:
                    yield line
        finally:
            # Close the inner iterator explicitly instead of abandoning it to
            # asyncio's async-gen GC finalizer (see quiet_aclose).
            await quiet_aclose(lines)

    async def close(self):
        if self._ctx is not None:
            await self._ctx.__aexit__(None, None, None)
        else:
            await self._resp.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc, value, _tb):
        await self.close()


async def _on_redirect(response: httpx2.Response) -> None:
    """Validate redirect targets to prevent SSRF via Location headers."""
    if not response.has_redirect_location:
        return
    location = response.headers.get("location")
    if not location:
        return
    target = urljoin(str(response.request.url), location)
    validate_server_url(target, label="redirect target", resolve_dns=False)


class AsyncSession:
    """HTTP client wrapping httpx2.AsyncClient with connection pool management."""

    def __init__(
        self,
        timeout=None,
        disable_http2=True,
        max_connections: int | None = None,
        max_keepalive_connections: int | None = None,
        **kwargs,
    ):
        if isinstance(timeout, tuple):
            connect_timeout, read_timeout = timeout
            timeout = httpx2.Timeout(
                connect=connect_timeout, read=read_timeout, write=60.0, pool=60.0
            )
        http2 = not disable_http2
        limits = httpx2.Limits(
            max_connections=max_connections or get_settings().http.max_connections,
            max_keepalive_connections=(
                max_keepalive_connections or get_settings().http.max_keepalive
            ),
        )

        headers = dict(kwargs.pop("headers", None) or {})
        headers.setdefault("User-Agent", DEFAULT_USER_AGENT)

        self._client = httpx2.AsyncClient(
            timeout=timeout,
            http2=http2,
            limits=limits,
            follow_redirects=True,
            event_hooks={"response": [_on_redirect]},
            headers=headers,
            **kwargs,
        )

    async def request(self, method, url, **kwargs):
        stream = kwargs.pop("stream", False)
        if "allow_redirects" in kwargs:
            kwargs["follow_redirects"] = kwargs.pop("allow_redirects")
        timeout = kwargs.pop("timeout", None)
        if isinstance(timeout, tuple):
            connect, read = timeout
            timeout = httpx2.Timeout(connect=connect, read=read, write=60.0, pool=60.0)

        if stream:
            ctx = self._client.stream(method, url, timeout=timeout, **kwargs)
            resp = await ctx.__aenter__()
            return Response(resp, context_manager=ctx)
        else:
            resp = await self._client.request(method, url, timeout=timeout, **kwargs)
            return Response(resp)

    async def get(self, url, **kwargs):
        return await self.request("GET", url, **kwargs)

    async def post(self, url, **kwargs):
        return await self.request("POST", url, **kwargs)

    async def close(self):
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc, value, _tb):
        await self.close()


class HTTPClient:
    """HTTP client with connection pool management."""

    def __init__(
        self,
        max_keepalive_connections: int | None = None,
        max_connections: int | None = None,
        connect_timeout: float = 10.0,
        read_timeout: float = 600.0,
        disable_http2: bool = True,
    ):
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout
        self._disable_http2 = disable_http2
        self._max_keepalive = max_keepalive_connections
        self._max_connections = max_connections

        self._client: AsyncSession | None = None
        self._lock = asyncio.Lock()

    def _create_client(self) -> AsyncSession:
        return AsyncSession(
            timeout=(self._connect_timeout, self._read_timeout),
            disable_http2=self._disable_http2,
            max_connections=self._max_connections,
            max_keepalive_connections=self._max_keepalive,
        )

    async def start(self) -> None:
        async with self._lock:
            if self._client is not None:
                return
            self._client = self._create_client()

    async def close(self) -> None:
        async with self._lock:
            if self._client:
                await self._client.close()
                self._client = None

    @property
    def client(self) -> AsyncSession:
        if self._client is None:
            from llm_proxy.core.exceptions import ConfigurationError

            raise ConfigurationError("HTTPClient has not been started. Call start() first.")
        return self._client

    async def __aenter__(self) -> HTTPClient:
        await self.start()
        return self

    async def __aexit__(self, exc_type, _exc_val, _exc_tb) -> None:
        await self.close()


class _ProviderClientSession:
    """Internal wrapper that tracks health for a single provider's HTTP session."""

    def __init__(
        self,
        provider_name: str,
        session: AsyncSession,
    ):
        self.provider_name = provider_name
        self._session = session

    async def get_session(self) -> AsyncSession:
        assert self._session is not None
        return self._session

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None


class ProviderHTTPClientManager:
    """Manages HTTP clients per provider with automatic recovery."""

    def __init__(
        self,
        max_keepalive_connections: int | None = None,
        max_connections: int | None = None,
        connect_timeout: float = 10.0,
        read_timeout: float = 600.0,
        disable_http2: bool = True,
    ):
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout
        self._disable_http2 = disable_http2
        self._max_keepalive = max_keepalive_connections
        self._max_connections = max_connections
        self._sessions: dict[str, _ProviderClientSession] = {}
        self._lock = asyncio.Lock()

    def _create_session(self) -> AsyncSession:
        return AsyncSession(
            timeout=(self._connect_timeout, self._read_timeout),
            disable_http2=self._disable_http2,
            max_connections=self._max_connections,
            max_keepalive_connections=self._max_keepalive,
        )

    async def get_client(self, provider_name: str) -> AsyncSession:
        wrapper = self._sessions.get(provider_name)
        if wrapper is not None:
            return await wrapper.get_session()
        async with self._lock:
            wrapper = self._sessions.get(provider_name)
            if wrapper is None:
                session = self._create_session()
                wrapper = _ProviderClientSession(
                    provider_name=provider_name,
                    session=session,
                )
                self._sessions[provider_name] = wrapper
        return await wrapper.get_session()

    async def close(self) -> None:
        for wrapper in list(self._sessions.values()):
            await wrapper.close()
        self._sessions.clear()

    async def __aenter__(self) -> ProviderHTTPClientManager:
        return self

    async def __aexit__(self, exc_type, _exc_val, _exc_tb) -> None:
        await self.close()


async def fetch_json(
    client: AsyncSession,
    url: str,
    method: str = "GET",
    **kwargs: Any,
) -> dict[str, Any]:
    response = await client.request(method, url, **kwargs)
    response.raise_for_status()
    return response.json()


DEFAULT_IMAGE_DOWNLOAD_TIMEOUT = 30.0


async def download_image_as_base64(
    client: AsyncSession,
    url: str,
    timeout: float = DEFAULT_IMAGE_DOWNLOAD_TIMEOUT,
) -> tuple[str, str] | None:
    """Download a file from ``url`` and return it as a base64 data URL.

    The URL is validated before the request to prevent server-side request
    forgery (SSRF) via user-supplied image/file URLs.
    """
    validate_server_url(url, label="download URL")

    response = await client.get(url, timeout=timeout, follow_redirects=False)
    response.raise_for_status()

    image_data = response.content
    if image_data is None:
        return None

    content_type = response.headers.get("content-type", "")
    # Strip any charset suffix (e.g. "application/pdf; charset=utf-8")
    if ";" in content_type:
        content_type = content_type.split(";")[0].strip()
    if not content_type or "/" not in content_type:
        path = url.split("?")[0].split("#")[0]
        extension = path.rsplit(".", 1)[-1].lower() if "." in path else ""
        content_type = _IMAGE_EXTENSION_TO_MIME.get(extension) or "image/jpeg"

    base64_data = base64.b64encode(image_data).decode("utf-8")
    data_url = f"data:{content_type};base64,{base64_data}"

    return data_url, content_type
