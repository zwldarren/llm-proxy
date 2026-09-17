"""Core request utilities that don't depend on the api layer.

Moved from api.middleware.logging.filters and api.utils to eliminate
reverse dependencies (core importing from api).
"""

import ipaddress
import logging
from functools import lru_cache

from fastapi import Request

from llm_proxy.config.settings import get_settings

_logger = logging.getLogger(__name__)

# Per-request cache keys (live in the ASGI scope state).
_CLIENT_IP_KEY = "_resolved_client_ip"
_PEER_TRUSTED_KEY = "_peer_is_trusted_proxy"

# Limit the number of X-Forwarded-For hops parsed to avoid O(n*m) DoS.
MAX_XFF_HOPS = 50


def _parse_trusted_networks(
    trusted_proxies: list[str],
) -> list[ipaddress.IPv4Network | ipaddress.IPv6Network]:
    """Pre-parse trusted proxy network strings into ip_network objects.

    Memoised on the configured value: this runs from per-request helpers
    (:func:`get_client_ip`, :func:`peer_is_trusted_proxy`), and constructing
    ``ip_network`` objects is not cheap (regex parse + integer arithmetic) for
    a value that only changes when settings are reloaded.
    """
    return list(_parse_trusted_networks_cached(tuple(trusted_proxies)))


@lru_cache(maxsize=8)
def _parse_trusted_networks_cached(
    trusted_proxies: tuple[str, ...],
) -> tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]:
    networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
    for network in trusted_proxies:
        try:
            networks.append(ipaddress.ip_network(network, strict=False))
        except ValueError:
            _logger.warning("Invalid trusted proxy network configured: %s", network)
    return tuple(networks)


def _is_trusted_proxy(
    peer_ip: str,
    trusted_networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network],
) -> bool:
    """Check whether peer_ip belongs to any configured trusted proxy network."""
    if not trusted_networks:
        return False
    try:
        peer_addr = ipaddress.ip_address(peer_ip)
    except ValueError:
        return False
    return any(peer_addr in network for network in trusted_networks)


def _rightmost_untrusted_ip(
    forwarded: str,
    trusted_networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network],
) -> str | None:
    """Walk the X-Forwarded-For chain from right to left.

    Return the IP address of the first untrusted hop (the closest untrusted
    client). If every hop is trusted, return the leftmost valid IP. If no hop
    is a valid IP address, return None so the caller can fall back to the peer
    IP.
    """
    if not forwarded:
        return None
    hops = [h.strip() for h in forwarded.split(",") if h.strip()][:MAX_XFF_HOPS]
    if not hops:
        return None

    last_valid_hop: str | None = None
    for hop in reversed(hops):
        try:
            ipaddress.ip_address(hop)
        except ValueError:
            continue
        last_valid_hop = hop
        if not _is_trusted_proxy(hop, trusted_networks):
            return hop

    # All valid hops are trusted; return the leftmost valid IP (original client).
    return last_valid_hop


def _scope_state(request: Request) -> dict[str, object] | None:
    """Return the per-request ASGI scope state dict, or None when unavailable.

    Production requests always carry a mutable ``scope`` dict, so IP/trust
    resolution computed once can be reused by every later call in the same
    request (auth, rate limiting, audit). Test doubles may not expose
    ``scope``; those fall back to recomputing on every call.
    """
    scope = getattr(request, "scope", None)
    if not isinstance(scope, dict):
        return None
    return scope.setdefault("state", {})


def peer_is_trusted_proxy(request: Request) -> bool:
    """Check whether the immediate TCP peer is in the configured TRUSTED_PROXIES.

    Returns False when the request has no client info or the peer IP is not a
    member of any configured trusted-proxy network. Use this to decide whether
    session/identity headers forwarded by a proxy may be trusted.

    Memoised on the request scope: this is called several times per request
    (context assembly, IP resolution) for a value that cannot change.
    """
    state = _scope_state(request)
    if state is not None:
        cached = state.get(_PEER_TRUSTED_KEY)
        if isinstance(cached, bool):
            return cached

    if not request.client:
        result = False
    else:
        peer_ip = request.client.host
        trusted_networks = _parse_trusted_networks(get_settings().security.trusted_proxies)
        result = _is_trusted_proxy(peer_ip, trusted_networks)

    if state is not None:
        state[_PEER_TRUSTED_KEY] = result
    return result


def get_client_ip(request: Request) -> str:
    """Get the real client IP, considering proxy headers only from trusted proxies.

    By default the immediate TCP peer is checked against the private network
    ranges (RFC 1918), loopback, and link-local — the common Docker / traefik /
    same-host reverse-proxy topology. If the peer is one of those, X-Forwarded-For
    is parsed right-to-left and the first untrusted hop is used as the real
    client IP; X-Real-IP is honoured only when X-Forwarded-For is absent.

    If the peer is a public address (not in the configured ``TRUSTED_PROXIES``),
    forwarded headers are ignored and the peer IP is returned. This prevents
    arbitrary clients from spoofing their IP for rate-limiting, lockout, and
    audit attribution.

    Set ``TRUSTED_PROXIES=`` (empty) to trust nobody and always use the peer IP
    — useful when the service is directly exposed to the public internet.
    """
    # Memoised on the request scope: auth, rate limiting and audit each
    # resolve the client IP, and the value is constant for the request.
    state = _scope_state(request)
    if state is not None:
        cached = state.get(_CLIENT_IP_KEY)
        if isinstance(cached, str):
            return cached

    client_ip = _compute_client_ip(request)

    if state is not None:
        state[_CLIENT_IP_KEY] = client_ip
    return client_ip


def _compute_client_ip(request: Request) -> str:
    if not request.client:
        _logger.warning("Request has no client info; cannot determine peer IP")
        return "unknown"
    peer_ip = request.client.host
    trusted_networks = _parse_trusted_networks(get_settings().security.trusted_proxies)

    if not _is_trusted_proxy(peer_ip, trusted_networks):
        return peer_ip

    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        extracted = _rightmost_untrusted_ip(forwarded_for, trusted_networks)
        if extracted:
            return extracted

    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        real_ip = real_ip.strip()
        try:
            ipaddress.ip_address(real_ip)
            return real_ip
        except ValueError:
            _logger.warning("Invalid IP in X-Real-IP header: %s", real_ip)

    return peer_ip


__all__ = ["get_client_ip", "peer_is_trusted_proxy"]
