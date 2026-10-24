"""Redis connection-log redaction."""

from llm_proxy.database.redis_client import _redact_redis_url


def test_redacts_credentials_and_query_string() -> None:
    assert (
        _redact_redis_url("rediss://user:secret@redis.example:6379/0?ssl_cert=secret")
        == "rediss://redis.example:6379/0"
    )


def test_preserves_ipv6_brackets() -> None:
    assert (
        _redact_redis_url("rediss://user:secret@[2001:db8::1]:6379/0")
        == "rediss://[2001:db8::1]:6379/0"
    )
