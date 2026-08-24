"""Core request utilities that don't depend on the api layer.

Moved from api.middleware.logging.filters and api.utils to eliminate
reverse dependencies (core importing from api).
"""

import ipaddress
import logging

from fastapi import Request

from llm_proxy.config.settings import get_settings

_logger = logging.getLogger(__name__)

# Limit the number of X-Forwarded-For hops parsed to avoid O(n*m) DoS.
MAX_XFF_HOPS = 50


def _parse_trusted_networks(
    trusted_proxies: list[str],
) -> list[ipaddress.IPv4Network | ipaddress.IPv6Network]:
    """Pre-parse trusted proxy network strings into ip_network objects."""
    networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
    for network in trusted_proxies:
        try:
            networks.append(ipaddress.ip_network(network, strict=False))
        except ValueError:
            _logger.warning("Invalid trusted proxy network configured: %s", network)
    return networks


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


def peer_is_trusted_proxy(request: Request) -> bool:
    """Check whether the immediate TCP peer is in the configured TRUSTED_PROXIES.

    Returns False when the request has no client info or the peer IP is not a
    member of any configured trusted-proxy network. Use this to decide whether
    session/identity headers forwarded by a proxy may be trusted.
    """
    if not request.client:
        return False
    peer_ip = request.client.host
    trusted_networks = _parse_trusted_networks(get_settings().security.trusted_proxies)
    return _is_trusted_proxy(peer_ip, trusted_networks)


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
