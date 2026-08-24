"""Upstream WebSocket connection for the Realtime API relay.

Builds the upstream ``wss://`` URL and authentication headers from a resolved
provider selection, and opens the connection with the ``websockets`` client
library. The relay itself lives in :mod:`llm_proxy.realtime.relay`.
"""

from collections.abc import Sequence
from urllib.parse import quote, urlsplit, urlunsplit

from websockets.asyncio.client import ClientConnection, connect
from websockets.typing import Subprotocol

from llm_proxy.config.types.provider import ProviderConfig
from llm_proxy.core.ws_common import WS_MAX_MESSAGE_BYTES
from llm_proxy.observability.logger import get_logger

logger = get_logger(__name__)

# OpenAI's default Realtime endpoint base (same host as the REST API).
DEFAULT_BASE_URL = "https://api.openai.com/v1"

# Endpoint-type key in ProviderConfig.endpoint_base_urls for per-provider
# Realtime URL overrides (mirrors the adapter convention, e.g. "responses").
REALTIME_ENDPOINT_KEY = "realtime"

# The Realtime protocol subprotocol. OpenAI's server selects it when offered;
# openai-compatible servers that do not negotiate subprotocols still accept
# the connection (the negotiated subprotocol is simply None).
REALTIME_SUBPROTOCOL = "realtime"

# Legacy beta header value. Pre-GA Realtime models
# (gpt-4o-realtime-preview-*, gpt-4o-mini-realtime-preview-*) required it;
# the official Beta→GA migration guidance says to remove it for GA models
# (gpt-realtime*), so it is sent only to OpenAI itself and only for legacy
# preview model names.
OPENAI_BETA_HEADER_VALUE = "realtime=v1"

# Handshake budget. Provider timeouts default to 600s which is far too long
# to wait for a WebSocket upgrade; cap the connect timeout at 60s.
MAX_CONNECT_TIMEOUT = 60.0

# Upstream messages can carry large conversation items (full audio in
# conversation.item.created); match the client-side 64 MiB cap.
MAX_UPSTREAM_MESSAGE_BYTES = WS_MAX_MESSAGE_BYTES


def build_realtime_url(
    provider_config: ProviderConfig,
    provider_model_name: str,
) -> str:
    """Build the upstream Realtime WebSocket URL for a provider selection.

    Resolution order mirrors the adapter convention
    (``_resolve_endpoint_url``): ``endpoint_base_urls["realtime"]`` first,
    then the provider ``base_url``, then OpenAI's default. The URL may carry
    a ``{model}`` placeholder (e.g. Azure-style ``?deployment={model}``),
    which is substituted with the upstream model name. The scheme is
    normalized to ``wss://``/``ws://`` and ``/realtime`` is appended unless
    the path already ends with it. A ``model`` query parameter is appended
    only when neither ``model`` nor ``deployment`` is already present.

    Args:
        provider_config: The selected provider configuration
        provider_model_name: The upstream model name to connect with

    Returns:
        The upstream WebSocket URL
    """
    base = (
        provider_config.endpoint_base_urls.get(REALTIME_ENDPOINT_KEY)
        or provider_config.base_url
        or DEFAULT_BASE_URL
    ).rstrip("/")

    if "{model}" in base:
        base = base.replace("{model}", quote(provider_model_name, safe=""))

    # Split into components so ``/realtime`` is appended to the path only:
    # appending to the full URL string would corrupt base URLs that carry a
    # query string (e.g. ``https://x/v1?api-version=…``).
    parts = urlsplit(base)
    scheme = {"https": "wss", "http": "ws"}.get(parts.scheme, parts.scheme)
    path = parts.path
    if not path.endswith("/realtime"):
        path = f"{path.rstrip('/')}/realtime" if path else "/realtime"

    query = parts.query
    if "model=" not in query and "deployment=" not in query:
        model_param = f"model={quote(provider_model_name, safe='')}"
        query = f"{query}&{model_param}" if query else model_param

    return urlunsplit((scheme, parts.netloc, path, query, ""))


def _is_legacy_preview_model(model_name: str) -> bool:
    """True for pre-GA Realtime models that still require the beta header.

    The GA interface (``gpt-realtime``, ``gpt-realtime-2.1``, ...) must not
    receive ``OpenAI-Beta: realtime=v1`` per the official Beta→GA migration
    guidance; only legacy preview models (``gpt-4o-realtime-preview-*``,
    ``gpt-4o-mini-realtime-preview-*``) required it.
    """
    return "realtime-preview" in model_name


def build_upstream_headers(
    provider_config: ProviderConfig,
    *,
    model_name: str | None = None,
    safety_identifier: str | None = None,
) -> dict[str, str]:
    """Build the upstream WebSocket handshake headers.

    The provider API key is injected as ``Authorization: Bearer``; provider
    ``custom_headers`` are merged on top (they win on conflicts). The legacy
    ``OpenAI-Beta: realtime=v1`` header is sent only to OpenAI itself, and
    only for legacy preview model names (never to GA models or
    openai-compatible endpoints).

    Args:
        provider_config: The selected provider configuration
        model_name: Upstream model name; gates the legacy ``OpenAI-Beta``
            header (sent only for pre-GA ``*-realtime-preview`` models)
        safety_identifier: Optional end-user safety identifier forwarded as
            ``OpenAI-Safety-Identifier`` (official docs recommend passing it
            on the connection request)

    Returns:
        Headers for the upstream WebSocket handshake
    """
    headers: dict[str, str] = {
        "Authorization": f"Bearer {provider_config.get_api_key()}",
    }
    if (
        provider_config.type == "openai"
        and model_name is not None
        and _is_legacy_preview_model(model_name)
    ):
        headers["OpenAI-Beta"] = OPENAI_BETA_HEADER_VALUE
    if safety_identifier:
        headers["OpenAI-Safety-Identifier"] = safety_identifier
    headers.update(provider_config.custom_headers or {})
    return headers


async def connect_upstream(
    url: str,
    headers: dict[str, str],
    *,
    timeout: float = 30.0,
    subprotocols: Sequence[Subprotocol] | None = None,
) -> ClientConnection:
    """Open the upstream Realtime WebSocket connection.

    Args:
        url: The upstream WebSocket URL
        headers: Handshake headers (see :func:`build_upstream_headers`)
        timeout: Connect (handshake) timeout in seconds
        subprotocols: Subprotocols to offer; defaults to ``["realtime"]``

    Returns:
        The connected WebSocket client connection

    Raises:
        websockets exceptions (InvalidStatus, InvalidHandshake, ...) on
        connection failure; the caller maps them to a Realtime error event.
    """
    if subprotocols is None:
        subprotocols = [Subprotocol(REALTIME_SUBPROTOCOL)]
    logger.debug(f"Connecting upstream Realtime WebSocket: {url}")
    return await connect(
        url,
        additional_headers=headers,
        subprotocols=subprotocols,
        open_timeout=min(timeout, MAX_CONNECT_TIMEOUT),
        max_size=MAX_UPSTREAM_MESSAGE_BYTES,
        # Keep the connection alive through NATs and idle periods; the
        # upstream closes the session when it is done regardless.
        ping_interval=20.0,
        ping_timeout=20.0,
    )


__all__ = [
    "DEFAULT_BASE_URL",
    "MAX_CONNECT_TIMEOUT",
    "MAX_UPSTREAM_MESSAGE_BYTES",
    "OPENAI_BETA_HEADER_VALUE",
    "REALTIME_ENDPOINT_KEY",
    "REALTIME_SUBPROTOCOL",
    "build_realtime_url",
    "build_upstream_headers",
    "connect_upstream",
]
