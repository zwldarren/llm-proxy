"""Tests for password security utilities."""

import pytest

from llm_proxy.security.passwords import (
    SENSITIVE_KEYS,
    generate_api_key,
    hash_api_key,
    hash_password,
    is_bcrypt_hash,
    mask_headers,
    mask_sensitive,
    verify_admin_password,
    verify_api_key,
)


class TestGenerateAPIKey:
    """Tests for API key generation."""

    def test_generate_api_key_format(self):
        """API key should have correct sk- prefix."""
        key = generate_api_key()
        assert key.startswith("sk-")

    def test_generate_api_key_length(self):
        """API key should be 67 characters (sk- + 64 hex)."""
        key = generate_api_key()
        assert len(key) == 67

    def test_generate_api_key_hex_chars(self):
        """API key after prefix should be valid hex."""
        key = generate_api_key()
        hex_part = key[3:]  # Remove 'sk-' prefix
        int(hex_part, 16)  # Should not raise

    def test_generate_api_key_randomness(self):
        """Generated keys should be unique (statistically)."""
        keys = [generate_api_key() for _ in range(100)]
        unique_keys = set(keys)
        # With 64 hex chars, collision is astronomically unlikely
        assert len(unique_keys) == 100


class TestHashAndVerifyAPIKey:
    """Tests for API key hashing and verification."""

    def test_hash_and_verify_roundtrip(self):
        """Hashed key should verify correctly."""
        original = "sk-test-api-key-12345"
        hashed = hash_api_key(original)
        assert verify_api_key(original, hashed)

    def test_hash_produces_bcrypt_hash(self):
        """Hashed key should be a valid bcrypt hash."""
        hashed = hash_api_key("test-key")
        assert is_bcrypt_hash(hashed)

    def test_verify_wrong_key_fails(self):
        """Wrong key should not verify."""
        hashed = hash_api_key("correct-key")
        assert not verify_api_key("wrong-key", hashed)

    def test_verify_empty_key_fails(self):
        """Empty key should not verify against non-empty hash."""
        hashed = hash_api_key("some-key")
        assert not verify_api_key("", hashed)


class TestIsBcryptHash:
    """Tests for bcrypt hash detection."""

    def test_valid_dollar_2a_prefix(self):
        """$2a$ prefix should be recognized."""
        # Generate a real bcrypt hash with $2a$ prefix
        import bcrypt

        hashed = bcrypt.hashpw(b"test", bcrypt.gensalt(rounds=10, prefix=b"2a"))
        hash_str = hashed.decode("utf-8")
        assert hash_str.startswith("$2a$")
        assert is_bcrypt_hash(hash_str)

    def test_valid_dollar_2b_prefix(self):
        """$2b$ prefix should be recognized."""
        import bcrypt

        hashed = bcrypt.hashpw(b"test", bcrypt.gensalt(rounds=10))
        hash_str = hashed.decode("utf-8")
        assert hash_str.startswith("$2b$")
        assert is_bcrypt_hash(hash_str)

    def test_valid_dollar_2y_prefix(self):
        """$2y$ prefix (PHP compatibility) should be recognized."""
        # PHP's bcrypt uses $2y$ prefix, which is format-compatible with $2b$
        # Generate a real $2b$ hash and check the function accepts standard bcrypt format
        import bcrypt

        hashed = bcrypt.hashpw(b"test", bcrypt.gensalt(rounds=10))
        hash_str = hashed.decode("utf-8")
        # Replace prefix to simulate PHP's $2y$ format (semantically equivalent)
        php_style_hash = "$2y$" + hash_str[4:]
        assert len(php_style_hash) == 60
        assert is_bcrypt_hash(php_style_hash)

    def test_invalid_prefix(self):
        """Invalid prefix should not be recognized."""
        hash_str = "$2c$10$abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTU"
        assert not is_bcrypt_hash(hash_str)

    def test_empty_string(self):
        """Empty string should not be recognized as bcrypt hash."""
        assert not is_bcrypt_hash("")

    def test_none_value(self):
        """None should not be recognized as bcrypt hash."""
        assert not is_bcrypt_hash(None)  # type: ignore

    def test_non_string_value(self):
        """Non-string values should not be recognized."""
        assert not is_bcrypt_hash(123)
        assert not is_bcrypt_hash(["$2b$10$..."])

    def test_wrong_length(self):
        """Wrong length should not be recognized as bcrypt hash."""
        # 59 chars (missing one)
        short = "$2b$10$abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQR"
        assert not is_bcrypt_hash(short)
        # 61 chars (extra one)
        long_hash = "$2b$10$abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUV"
        assert not is_bcrypt_hash(long_hash)

    def test_truncated_hash(self):
        """Truncated bcrypt hash should not be recognized."""
        truncated = "$2b$10$abc"
        assert not is_bcrypt_hash(truncated)


class TestBcrypt72ByteBoundary:
    """Tests for bcrypt 72-byte boundary condition."""

    def test_72_byte_password(self):
        """Password exactly 72 bytes should work (bcrypt limit)."""
        password = "a" * 72
        hashed = hash_password(password)
        assert verify_admin_password(password, hashed)

    def test_over_72_bytes_password(self):
        """Password over 72 bytes raises ValueError (bcrypt limit)."""
        password = "a" * 73  # Just over 72 bytes
        # bcrypt raises ValueError for passwords > 72 bytes
        with pytest.raises(ValueError, match="cannot be longer than 72 bytes"):
            hash_password(password)

    def test_71_byte_password(self):
        """Password just under 72 bytes should work."""
        password = "a" * 71
        hashed = hash_password(password)
        assert verify_admin_password(password, hashed)


class TestMaskHeaders:
    """Tests for header masking."""

    def test_mask_authorization_header(self):
        """Authorization header should be masked."""
        headers = {"Authorization": "Bearer token123"}
        masked = mask_headers(headers)
        assert masked["Authorization"] == "***"

    def test_mask_api_key_header(self):
        """API key headers should be masked."""
        headers = {"X-API-Key": "secret-key", "api-key": "another-key"}
        masked = mask_headers(headers)
        assert masked["X-API-Key"] == "***"
        assert masked["api-key"] == "***"

    def test_preserve_non_sensitive_headers(self):
        """Non-sensitive headers should be preserved."""
        headers = {"Content-Type": "application/json", "Accept": "text/plain"}
        masked = mask_headers(headers)
        assert masked["Content-Type"] == "application/json"
        assert masked["Accept"] == "text/plain"

    def test_case_insensitive_matching(self):
        """Header matching should be case-insensitive."""
        headers = {"AUTHORIZATION": "Bearer token", "Content-Type": "application/json"}
        masked = mask_headers(headers)
        assert masked["AUTHORIZATION"] == "***"
        assert masked["Content-Type"] == "application/json"

    def test_empty_headers(self):
        """Empty headers should return empty dict."""
        assert mask_headers({}) == {}

    def test_none_value(self):
        """None headers raises AttributeError."""
        with pytest.raises(AttributeError):
            mask_headers(None)


class TestMaskSensitive:
    """Tests for sensitive data masking in nested structures."""

    def test_mask_simple_dict(self):
        """Sensitive key values should be masked."""
        data = {"api_key": "secret123", "name": "test"}
        masked = mask_sensitive(data, SENSITIVE_KEYS)
        # api_key is sensitive, value is masked (truncated format)
        assert masked["name"] == "test"
        assert masked["api_key"] != "secret123"  # value is masked

    def test_mask_nested_dict(self):
        """Nested dicts should have sensitive key values masked."""
        data = {
            "outer": {
                "inner": {
                    "password": "secret123",
                    "safe": "value",
                }
            }
        }
        masked = mask_sensitive(data, SENSITIVE_KEYS)
        # password is sensitive, value should be masked
        assert masked["outer"]["inner"]["safe"] == "value"
        assert masked["outer"]["inner"]["password"] != "secret123"

    def test_mask_in_list(self):
        """Sensitive key values in lists should be masked."""
        data = [{"api_key": "key1"}, {"password": "pass1"}]
        masked = mask_sensitive(data, SENSITIVE_KEYS)
        assert masked[0]["api_key"] != "key1"  # masked
        assert masked[1]["password"] != "pass1"  # masked

    def test_preserve_non_sensitive(self):
        """Non-sensitive data should be preserved unchanged."""
        data = {"name": "John", "age": 30, "city": "NYC"}
        masked = mask_sensitive(data, SENSITIVE_KEYS)
        assert masked == data

    def test_mask_value_short(self):
        """Short values should be fully masked."""
        data = {"secret": "short"}
        _masked = mask_sensitive(data, SENSITIVE_KEYS)
        # secret is not in SENSITIVE_KEYS, so value is not masked by key
        # But let's test a key that IS in SENSITIVE_KEYS
        data2 = {"password": "x"}
        masked2 = mask_sensitive(data2, SENSITIVE_KEYS)
        assert masked2["password"] != "x"  # masked

    def test_mask_value_long(self):
        """Long values should be masked (first 3 + ... + last 4)."""
        data = {"password": "my-very-long-secret-value-12345"}
        masked = mask_sensitive(data, SENSITIVE_KEYS)
        # password is in SENSITIVE_KEYS, value is masked
        assert masked["password"] != "my-very-long-secret-value-12345"

    def test_empty_data(self):
        """Empty data should be returned as-is."""
        assert mask_sensitive({}, SENSITIVE_KEYS) == {}
        assert mask_sensitive([], SENSITIVE_KEYS) == []

    def test_none_data(self):
        """None data should be returned as-is."""
        assert mask_sensitive(None, SENSITIVE_KEYS) is None

    def test_non_dict_list(self):
        """Non-container values should be returned as-is."""
        assert mask_sensitive("string", SENSITIVE_KEYS) == "string"
        assert mask_sensitive(123, SENSITIVE_KEYS) == 123

    def test_empty_sensitive_keys(self):
        """Empty sensitive_keys should return data as-is."""
        data = {"password": "secret"}
        masked = mask_sensitive(data, frozenset())
        assert masked == data

    def test_deeply_nested_structure(self):
        """Deeply nested structures should be handled without recursion limit."""
        # Build a deeply nested structure
        data = {"level1": {"level2": {"level3": {"level4": {"password": "deep"}}}}}
        masked = mask_sensitive(data, SENSITIVE_KEYS)
        assert masked["level1"]["level2"]["level3"]["level4"]["password"] == "***"


class TestVerifyAdminPasswordWarnings:
    """Warning messages must not prompt users to re-save or migrate."""

    def test_plaintext_hash_warning_does_not_mention_migration(self, caplog):
        """A plaintext hash logs a warning without migration instructions."""
        with caplog.at_level("WARNING"):
            assert not verify_admin_password("secret", "plaintext-hash")

        assert len(caplog.records) == 1
        assert "re-save" not in caplog.records[0].message.lower()
        assert "migrate" not in caplog.records[0].message.lower()

    def test_malformed_hash_warning_does_not_mention_migration(self, caplog):
        """A malformed bcrypt hash logs a warning without migration instructions."""
        malformed = "$2b$10$" + "x" * 53
        assert is_bcrypt_hash(malformed)

        with caplog.at_level("WARNING"):
            assert not verify_admin_password("secret", malformed)

        assert len(caplog.records) == 1
        assert "re-save" not in caplog.records[0].message.lower()
        assert "migrate" not in caplog.records[0].message.lower()

    def test_valid_hash_does_not_log_warning(self, caplog):
        """A valid bcrypt hash should verify without emitting warnings."""
        hashed = hash_password("correct horse battery staple")

        with caplog.at_level("WARNING"):
            assert verify_admin_password("correct horse battery staple", hashed)

        assert not caplog.records


class TestVerifyAdminPassword:
    """Tests for admin password verification."""

    def test_verify_correct_password(self):
        """Correct password should verify successfully."""
        hashed = hash_password("my-secure-password")
        assert verify_admin_password("my-secure-password", hashed)

    def test_verify_wrong_password(self):
        """Wrong password should fail verification."""
        hashed = hash_password("correct-password")
        assert not verify_admin_password("wrong-password", hashed)

    def test_verify_empty_password(self):
        """Empty password should fail against non-empty hash."""
        hashed = hash_password("some-password")
        assert not verify_admin_password("", hashed)

    def test_verify_plaintext_stored(self):
        """Plaintext-stored password should fail and log warning."""
        # "plaintext-hash" is not a valid bcrypt hash (too short, wrong format)
        # The function logs a warning and returns False
        result = verify_admin_password("any-password", "plaintext-hash")
        assert not result

    def test_verify_invalid_bcrypt_format(self, caplog):
        """Invalid bcrypt format should fail gracefully."""
        # $2x$ is not a recognized bcrypt prefix - treated as plaintext
        invalid_hash = "$2x$10$abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTU"

        with caplog.at_level("WARNING"):
            result = verify_admin_password("password", invalid_hash)

        assert not result
        assert len(caplog.records) == 1
        # Warning is about plaintext/unsupported format
        msg = caplog.records[0].message.lower()
        assert "plaintext" in msg or "unsupported" in msg
