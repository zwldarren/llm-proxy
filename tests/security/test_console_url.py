"""Public console URL gating for WebAuthn and OIDC."""

import pytest

from llm_proxy.config.types.console_auth import ConsoleAuthSettings
from llm_proxy.security.console_url import public_origin, rp_id


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://proxy.example.com", "https://proxy.example.com"),
        ("https://proxy.example.com/", "https://proxy.example.com"),
        ("http://localhost:5173", "http://localhost:5173"),
        ("http://127.0.0.1:8080", "http://127.0.0.1:8080"),
        ("http://proxy.example.com", None),
        ("http://192.168.1.20:8080", None),
        ("ftp://proxy.example.com", None),
        ("not a url", None),
        ("", None),
    ],
)
def test_public_origin_allows_https_and_loopback_only(url: str, expected: str | None) -> None:
    settings = ConsoleAuthSettings(public_base_url=url)
    assert public_origin(settings) == expected


def test_rp_id_is_the_hostname_without_the_path() -> None:
    settings = ConsoleAuthSettings(public_base_url="https://auth.example.com/proxy")
    assert rp_id(settings) == "auth.example.com"


def test_rp_id_is_none_without_a_secure_origin() -> None:
    assert rp_id(ConsoleAuthSettings(public_base_url="http://example.com")) is None
