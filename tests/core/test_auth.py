"""Tests for JWTManager and authentication utilities."""

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import jwt
import pytest

from llm_proxy.config.types.auth import ProxyAuthConfig
from llm_proxy.core.exceptions import ConfigurationError
from llm_proxy.security.jwt import JWTManager
from llm_proxy.security.passwords import (
    hash_password,
    is_bcrypt_hash,
    verify_admin_password,
)

_TEST_SECRET = "this_is_a_test_secret_key_that_is_32bytes_long_for_jwt"


@pytest.fixture
def auth_config():
    """Create a test auth config with a valid JWT secret."""
    return ProxyAuthConfig(jwt_secret=_TEST_SECRET)


@pytest.fixture
def jwt_manager(auth_config):
    """Create a JWTManager instance."""
    return JWTManager(auth_config)


class TestJWTManager:
    """Test suite for JWTManager (token signing/verification only)."""

    def test_init(self, jwt_manager, auth_config):
        """Test JWTManager initialization."""
        assert jwt_manager.auth_config == auth_config
        assert jwt_manager.secret == _TEST_SECRET
        assert jwt_manager._ALGORITHM == "HS256"
        assert timedelta(hours=24) == jwt_manager._EXPIRES_IN

    def test_create_token_success(self, jwt_manager):
        """Test successful token creation."""
        token = jwt_manager.create_token("testuser")

        assert isinstance(token, str)
        assert len(token) > 0

        # Verify token can be decoded
        payload = jwt.decode(token, _TEST_SECRET, algorithms=["HS256"])
        assert payload["sub"] == "testuser"
        assert payload["type"] == "admin"
        assert "exp" in payload
        assert "iat" in payload

    def test_create_token_no_secret(self):
        """Token creation without secret raises error at config level."""
        with pytest.raises(ValueError, match="jwt_secret must be set"):
            ProxyAuthConfig(jwt_secret="")

    def test_create_token_expiration(self, jwt_manager):
        """Test that created token has correct expiration."""
        now = datetime.now(tz=UTC)

        with patch("llm_proxy.security.jwt.datetime") as mock_datetime:
            mock_datetime.now.return_value = now
            mock_datetime.UTC = UTC

            token = jwt_manager.create_token("testuser")

        payload = jwt.decode(token, _TEST_SECRET, algorithms=["HS256"])
        expected_exp = now + timedelta(hours=24)
        assert payload["exp"] == int(expected_exp.timestamp())

    def test_verify_token_success(self, jwt_manager):
        """Test successful token verification."""
        token = jwt_manager.create_token("testuser")

        payload = jwt_manager.verify_token(token)

        assert payload["sub"] == "testuser"
        assert payload["type"] == "admin"

    def test_verify_token_no_secret(self):
        """Token verification without secret raises error at config level."""
        with pytest.raises(ValueError, match="jwt_secret must be set"):
            ProxyAuthConfig(jwt_secret="")

    def test_verify_token_invalid_signature(self, jwt_manager):
        """Test verification with invalid signature."""
        token = jwt_manager.create_token("testuser")

        # Tamper with the token
        tampered_token = token[:-5] + "xxxxx"

        with pytest.raises(ConfigurationError, match="Invalid token"):
            jwt_manager.verify_token(tampered_token)

    def test_verify_token_expired(self, jwt_manager):
        """Test verification of expired token."""
        # Create a token that expired 1 hour ago
        past_time = datetime.now(tz=UTC) - timedelta(hours=1)
        payload = {
            "sub": "testuser",
            "exp": past_time,
            "iat": past_time - timedelta(hours=1),
            "type": "admin",
        }
        expired_token = jwt.encode(payload, _TEST_SECRET, algorithm="HS256")

        with pytest.raises(ConfigurationError, match="Invalid token"):
            jwt_manager.verify_token(expired_token)

    def test_verify_token_malformed(self, jwt_manager):
        """Test verification of malformed token."""
        with pytest.raises(ConfigurationError, match="Invalid token"):
            jwt_manager.verify_token("not_a_valid_token")

    def test_full_token_lifecycle(self, jwt_manager):
        """Test complete token lifecycle: create -> verify -> use."""
        token = jwt_manager.create_token("admin_user")

        payload = jwt_manager.verify_token(token)
        assert payload["sub"] == "admin_user"
        assert "exp" in payload
        assert "iat" in payload
        assert payload["type"] == "admin"


class TestAdminPasswordHashing:
    """Tests for admin password hashing/verification (used by the users table)."""

    def test_hash_password_produces_bcrypt(self):
        """hash_password produces a valid bcrypt hash."""
        hashed = hash_password("secret")
        assert is_bcrypt_hash(hashed)

    def test_hash_password_is_salting(self):
        """Hashing the same password twice yields different hashes."""
        assert hash_password("secret") != hash_password("secret")

    def test_verify_admin_password_success(self):
        """A password verifies against its bcrypt hash."""
        hashed = hash_password("correct horse battery staple")
        assert verify_admin_password("correct horse battery staple", hashed) is True

    def test_verify_admin_password_wrong(self):
        """A wrong password does not verify against the hash."""
        hashed = hash_password("correct horse battery staple")
        assert verify_admin_password("wrong password", hashed) is False

    def test_is_bcrypt_hash_rejects_non_bcrypt_strings(self):
        """is_bcrypt_hash returns False for plaintext and malformed values."""
        assert is_bcrypt_hash("") is False
        assert is_bcrypt_hash("plaintext-password") is False
        assert is_bcrypt_hash("$2b$") is False
        assert is_bcrypt_hash("$2b$10$short") is False
        assert is_bcrypt_hash("not-a-hash") is False

    def test_verify_admin_password_rejects_malformed_bcrypt_hash(self):
        """A malformed bcrypt-looking hash should be rejected, not fall back to plaintext.

        A malformed hash indicates data corruption or tampering and should
        never result in a successful authentication.
        """
        malformed = "$2b$10$" + "x" * 53
        assert is_bcrypt_hash(malformed)
        # The function should return False for a malformed hash, not fall back
        # to plaintext comparison.
        assert verify_admin_password("any-password", malformed) is False
        assert verify_admin_password(malformed, malformed) is False

    def test_hash_password_empty_string(self):
        """Hashing an empty string should produce a valid bcrypt hash."""
        hashed = hash_password("")
        assert is_bcrypt_hash(hashed)

    def test_hash_password_long_string(self):
        """Hashing a very long password should raise ValueError.

        bcrypt has a 72-byte limit and modern bindings reject longer inputs
        explicitly rather than silently truncating.
        """
        long_password = "x" * 1000
        with pytest.raises(ValueError, match="cannot be longer than 72 bytes"):
            hash_password(long_password)
