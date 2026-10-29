"""Public console URL helpers shared by WebAuthn and OAuth."""

from urllib.parse import ParseResult, urlparse

from llm_proxy.config.types.console_auth import ConsoleAuthSettings

#: Loopback hosts where plain HTTP is still a secure context.
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def public_origin(settings: ConsoleAuthSettings) -> str | None:
    """Return the canonical scheme://host[:port], or None when unset/invalid.

    WebAuthn needs a secure context and a stable RP ID; OAuth needs an exact
    redirect URI. Accept HTTPS everywhere and HTTP only for localhost
    development, matching the browser security model.
    """
    parsed = _parse_base_url(settings.public_base_url)
    if parsed is None:
        return None
    return f"{parsed.scheme}://{parsed.netloc}"


def _parse_base_url(raw: str) -> ParseResult | None:
    """Parse a configured base URL, or None when unset/invalid."""
    raw = raw.strip().rstrip("/")
    if not raw:
        return None
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    if parsed.scheme == "http" and parsed.hostname not in _LOOPBACK_HOSTS:
        return None
    return parsed


def is_secure_issuer(issuer: str) -> bool:
    """Whether an issuer URL is HTTPS, or plain HTTP on loopback."""
    parsed = urlparse(issuer)
    if not parsed.hostname:
        return False
    return parsed.scheme == "https" or (
        parsed.scheme == "http" and parsed.hostname in _LOOPBACK_HOSTS
    )


def rp_id(settings: ConsoleAuthSettings) -> str | None:
    """WebAuthn relying-party id (the hostname, never a path)."""
    parsed = _parse_base_url(settings.public_base_url)
    if parsed is None:
        return None
    return parsed.hostname
