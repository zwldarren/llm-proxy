"""Tests for core request utilities, especially IP extraction."""

from unittest.mock import MagicMock

import pytest

from llm_proxy.config.settings import (
    DEFAULT_TRUSTED_PROXIES,
    SecuritySettings,
    Settings,
    set_settings,
)
from llm_proxy.core.request_utils import get_client_ip


@pytest.fixture
def reset_settings():
    """Reset global settings after each test."""
    from llm_proxy.config.settings import reset_settings as _reset

    yield
    _reset()


def _make_request(*, peer_ip: str, headers: dict[str, str] | None = None) -> MagicMock:
    """Build a minimal FastAPI-style request mock."""
    request = MagicMock(spec=["client", "headers"])
    request.client = MagicMock(host=peer_ip, port=0)
    request.headers = headers or {}
    return request


def test_get_client_ip_ignores_headers_without_trusted_proxy(reset_settings) -> None:
    """Without TRUSTED_PROXIES, spoofed X-Forwarded-For is ignored."""
    set_settings(Settings(security=SecuritySettings(**{"TRUSTED_PROXIES": ""})))
    request = _make_request(peer_ip="203.0.113.5", headers={"X-Forwarded-For": "1.2.3.4"})
    assert get_client_ip(request) == "203.0.113.5"


def test_get_client_ip_ignores_headers_from_untrusted_peer(reset_settings) -> None:
    """When TRUSTED_PROXIES is configured but peer is NOT in the trusted network,
    X-Forwarded-For is still ignored (untrusted source).
    """
    set_settings(Settings(security=SecuritySettings(**{"TRUSTED_PROXIES": "10.0.0.0/8"})))
    request = _make_request(
        peer_ip="203.0.113.5",
        headers={"X-Forwarded-For": "1.2.3.4"},
    )
    assert get_client_ip(request) == "203.0.113.5"


def test_get_client_ip_honors_x_forwarded_for_from_trusted_proxy(reset_settings) -> None:
    """When peer is a trusted proxy, X-Forwarded-For is parsed right-to-left."""
    set_settings(Settings(security=SecuritySettings(**{"TRUSTED_PROXIES": "10.0.0.0/8"})))
    request = _make_request(
        peer_ip="10.0.0.1",
        headers={"X-Forwarded-For": "1.2.3.4, 10.1.1.1, 10.0.0.2"},
    )
    # 10.0.0.2 and 10.1.1.1 are trusted; 1.2.3.4 is the first untrusted hop
    assert get_client_ip(request) == "1.2.3.4"


def test_get_client_ip_returns_leftmost_when_all_trusted(reset_settings) -> None:
    """If every hop is trusted, return the leftmost (original client)."""
    set_settings(Settings(security=SecuritySettings(**{"TRUSTED_PROXIES": "0.0.0.0/0"})))
    request = _make_request(
        peer_ip="10.0.0.1",
        headers={"X-Forwarded-For": "1.2.3.4, 10.1.1.1"},
    )
    assert get_client_ip(request) == "1.2.3.4"


def test_get_client_ip_uses_x_real_ip_from_trusted_proxy(reset_settings) -> None:
    """X-Real-IP is used when peer is trusted and X-Forwarded-For is absent."""
    set_settings(Settings(security=SecuritySettings(**{"TRUSTED_PROXIES": "10.0.0.0/8"})))
    request = _make_request(
        peer_ip="10.0.0.1",
        headers={"X-Real-IP": "1.2.3.4"},
    )
    assert get_client_ip(request) == "1.2.3.4"


def test_get_client_ip_ignores_x_real_ip_without_trusted_proxy(reset_settings) -> None:
    """Without a trusted proxy, X-Real-IP is ignored."""
    set_settings(Settings(security=SecuritySettings(**{"TRUSTED_PROXIES": ""})))
    request = _make_request(
        peer_ip="203.0.113.5",
        headers={"X-Real-IP": "1.2.3.4"},
    )
    assert get_client_ip(request) == "203.0.113.5"


def test_get_client_ip_prefers_x_forwarded_for_over_x_real_ip(reset_settings) -> None:
    """X-Forwarded-For takes precedence over X-Real-IP."""
    set_settings(Settings(security=SecuritySettings(**{"TRUSTED_PROXIES": "10.0.0.0/8"})))
    request = _make_request(
        peer_ip="10.0.0.1",
        headers={
            "X-Forwarded-For": "1.2.3.4",
            "X-Real-IP": "5.6.7.8",
        },
    )
    assert get_client_ip(request) == "1.2.3.4"


def test_get_client_ip_falls_back_to_peer_on_invalid_header(reset_settings) -> None:
    """Malformed X-Forwarded-For from a trusted proxy falls back to peer IP."""
    set_settings(Settings(security=SecuritySettings(**{"TRUSTED_PROXIES": "10.0.0.0/8"})))
    request = _make_request(
        peer_ip="10.0.0.1",
        headers={"X-Forwarded-For": "not-an-ip"},
    )
    assert get_client_ip(request) == "10.0.0.1"


def test_get_client_ip_handles_empty_x_forwarded_for(reset_settings) -> None:
    """Empty X-Forwarded-For from a trusted proxy falls back to peer IP."""
    set_settings(Settings(security=SecuritySettings(**{"TRUSTED_PROXIES": "10.0.0.0/8"})))
    request = _make_request(
        peer_ip="10.0.0.1",
        headers={"X-Forwarded-For": ""},
    )
    assert get_client_ip(request) == "10.0.0.1"


def test_get_client_ip_handles_whitespace_only_entries(reset_settings) -> None:
    """Whitespace-only entries in X-Forwarded-For are skipped gracefully."""
    set_settings(Settings(security=SecuritySettings(**{"TRUSTED_PROXIES": "10.0.0.0/8"})))
    request = _make_request(
        peer_ip="10.0.0.1",
        headers={"X-Forwarded-For": "1.2.3.4,  , 10.0.0.2"},
    )
    assert get_client_ip(request) == "1.2.3.4"


def test_get_client_ip_supports_ipv6_trusted_proxy(reset_settings) -> None:
    """IPv6 trusted proxy networks are accepted."""
    set_settings(Settings(security=SecuritySettings(**{"TRUSTED_PROXIES": "::1/128"})))
    request = _make_request(
        peer_ip="::1",
        headers={"X-Forwarded-For": "2001:db8::1"},
    )
    assert get_client_ip(request) == "2001:db8::1"


def test_trusted_proxies_validation_accepts_cidr(reset_settings) -> None:
    """CIDR notation is accepted for trusted proxy configuration."""
    settings = Settings(security=SecuritySettings(**{"TRUSTED_PROXIES": "10.0.0.0/8"}))
    assert settings.security.trusted_proxies == ["10.0.0.0/8"]


def test_trusted_proxies_validation_accepts_comma_string(reset_settings) -> None:
    """Comma-separated string is parsed into a list."""
    settings = Settings(
        security=SecuritySettings(**{"TRUSTED_PROXIES": "10.0.0.0/8, 192.168.0.0/16"})
    )
    assert settings.security.trusted_proxies == ["10.0.0.0/8", "192.168.0.0/16"]


def test_trusted_proxies_validation_rejects_invalid_network(reset_settings) -> None:
    """Invalid CIDR raises a validation error."""
    with pytest.raises(ValueError):
        Settings(security=SecuritySettings(**{"TRUSTED_PROXIES": "not-a-network"}))


def test_trusted_proxies_default_trusts_private_ranges(reset_settings) -> None:
    """Out of the box, common private/loopback/link-local networks are trusted.

    This mirrors mature reverse-proxy-aware servers (e.g. Tomcat RemoteIpValve)
    so a typical Docker / traefik deployment resolves the real client IP without
    any extra configuration.
    """
    settings = Settings(security=SecuritySettings())
    assert settings.security.trusted_proxies == DEFAULT_TRUSTED_PROXIES
    # Spot-check representative ranges from each family.
    parsed = {n for n in settings.security.trusted_proxies}
    assert "10.0.0.0/8" in parsed
    assert "172.16.0.0/12" in parsed
    assert "192.168.0.0/16" in parsed
    assert "127.0.0.0/8" in parsed
    assert "::1/128" in parsed


def test_trusted_proxies_empty_string_disables_trust(reset_settings) -> None:
    """Setting TRUSTED_PROXIES= (empty) opts out: nobody is trusted.

    This is the escape hatch for direct public exposure where forwarded
    headers must never be honoured.
    """
    settings = Settings(security=SecuritySettings(**{"TRUSTED_PROXIES": ""}))
    assert settings.security.trusted_proxies == []


def test_get_client_ip_resolves_real_ip_behind_lan_proxy_by_default(reset_settings) -> None:
    """The user's traefik scenario: with default settings, a LAN proxy peer
    (e.g. 10.0.1.12) forwards X-Forwarded-For and the real public client IP is
    used — so per-IP lockout/rate-limiting attributes failures to the real
    client instead of the shared proxy IP.
    """
    set_settings(Settings(security=SecuritySettings()))  # default trusted proxies
    request = _make_request(
        peer_ip="10.0.1.12",  # traefik on the LAN
        headers={"X-Forwarded-For": "203.0.113.42"},  # real public client
    )
    assert get_client_ip(request) == "203.0.113.42"


def test_get_client_ip_public_peer_ignores_forwarded_headers_by_default(reset_settings) -> None:
    """With default settings, a public peer is NOT trusted, so a spoofed
    X-Forwarded-For is ignored and the peer IP is returned.
    """
    set_settings(Settings(security=SecuritySettings()))  # default trusted proxies
    request = _make_request(
        peer_ip="203.0.113.5",  # public, not in any private range
        headers={"X-Forwarded-For": "1.2.3.4"},
    )
    assert get_client_ip(request) == "203.0.113.5"
