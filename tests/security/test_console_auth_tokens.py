"""Tests for invitation tokens, recovery codes and TOTP."""

from llm_proxy.config.types.auth import ProxyAuthConfig
from llm_proxy.security.jwt import JWTManager
from llm_proxy.security.tokens import (
    generate_invite_token,
    generate_recovery_codes,
    hash_recovery_code,
    hash_token,
    normalize_recovery_code,
)
from llm_proxy.security.totp import (
    generate_totp_secret,
    provisioning_uri,
    totp_code,
    verify_totp_code,
)


def test_invite_tokens_are_random():
    first = generate_invite_token()
    second = generate_invite_token()
    assert first != second
    assert hash_token(first) != hash_token(second)


def test_recovery_codes_are_unique_and_normalized():
    codes = generate_recovery_codes(10)
    assert len(set(codes)) == 10
    assert all(len(code) == 11 and code[5] == "-" for code in codes)
    assert normalize_recovery_code(codes[0].lower()) == codes[0].replace("-", "")
    assert hash_recovery_code(codes[0]) == hash_recovery_code(codes[0].lower())


def test_totp_accepts_current_and_neighbouring_windows():
    secret = generate_totp_secret()
    now = 1_760_000_000.0
    code = totp_code(secret, for_time=now)
    assert len(code) == 6 and code.isdigit()
    assert verify_totp_code(secret, code, for_time=now)
    assert verify_totp_code(secret, code, for_time=now + 30, window=1)
    assert not verify_totp_code(secret, code, for_time=now + 120, window=1)
    assert not verify_totp_code(secret, "12345", for_time=now)


def test_provisioning_uri_contains_issuer_and_secret():
    secret = generate_totp_secret()
    uri = provisioning_uri(secret, account_name="alice")
    assert uri.startswith("otpauth://totp/")
    assert "issuer=LLM%20Proxy" in uri
    assert f"secret={secret}" in uri


def test_scoped_tokens_reject_the_wrong_purpose():
    manager = JWTManager(ProxyAuthConfig(jwt_secret="x" * 32))
    token = manager.create_scoped_token("mfa_login", {"uid": 1}, ttl_seconds=60)
    assert manager.verify_scoped_token(token, "mfa_login")["uid"] == 1
    try:
        manager.verify_scoped_token(token, "oauth_state")
    except Exception as exc:
        assert "purpose" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("a scoped token must not be accepted for another purpose")
