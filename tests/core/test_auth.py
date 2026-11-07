"""Tests for JWTManager and authentication utilities."""

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import jwt
import pytest

from llm_proxy.config.types.auth import ProxyAuthConfig
from llm_proxy.core.exceptions import ConfigurationError
from llm_proxy.database.tables import UserRecord
from llm_proxy.security.jwt import JWTManager
from llm_proxy.security.passwords import (
    has_password_hash,
    hash_password,
    is_bcrypt_hash,
    password_needs_rehash,
    verify_password,
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
    """Tests for password hashing/verification (used by the users table)."""

    def test_hash_password_produces_argon2id(self):
        """hash_password produces a valid Argon2id hash."""
        hashed = hash_password("secret")
        assert hashed.startswith("$argon2id$")
        assert has_password_hash(hashed)
        assert not is_bcrypt_hash(hashed)

    def test_user_has_password_uses_the_supported_hash_predicate(self):
        user = UserRecord(username="alice", password_hash=hash_password("secret"), role="admin")

        assert user.has_password is True
        user.password_hash = "$2b$12$" + "a" * 53
        assert user.has_password is True
        user.password_hash = ""
        assert user.has_password is False

    def test_hash_password_is_salting(self):
        """Hashing the same password twice yields different hashes."""
        assert hash_password("secret") != hash_password("secret")

    def test_verify_password_success(self):
        """A password verifies against its hash."""
        hashed = hash_password("correct horse battery staple")
        assert verify_password("correct horse battery staple", hashed) is True

    def test_verify_password_wrong(self):
        """A wrong password does not verify against the hash."""
        hashed = hash_password("correct horse battery staple")
        assert verify_password("wrong password", hashed) is False

    def test_verify_password_accepts_legacy_bcrypt_hash(self):
        """Hashes written before the Argon2id migration still verify."""
        import bcrypt

        legacy = bcrypt.hashpw(b"correct horse battery staple", bcrypt.gensalt(rounds=4))
        legacy_hash = legacy.decode("utf-8")

        assert verify_password("correct horse battery staple", legacy_hash) is True
        assert verify_password("wrong password", legacy_hash) is False

    def test_password_needs_rehash_flags_bcrypt_only(self):
        """Only legacy bcrypt hashes ask to be replaced."""
        import bcrypt

        legacy = bcrypt.hashpw(b"secret", bcrypt.gensalt(rounds=4)).decode("utf-8")

        assert password_needs_rehash(legacy) is True
        assert password_needs_rehash(hash_password("secret")) is False
        assert password_needs_rehash("") is False
        assert password_needs_rehash("plaintext") is False

    def test_is_bcrypt_hash_rejects_non_bcrypt_strings(self):
        """is_bcrypt_hash returns False for plaintext and malformed values."""
        assert is_bcrypt_hash("") is False
        assert is_bcrypt_hash("plaintext-password") is False
        assert is_bcrypt_hash("$2b$") is False
        assert is_bcrypt_hash("$2b$10$short") is False
        assert is_bcrypt_hash("not-a-hash") is False

    def test_verify_password_rejects_malformed_bcrypt_hash(self):
        """A malformed bcrypt-looking hash should be rejected, not fall back to plaintext.

        A malformed hash indicates data corruption or tampering and should
        never result in a successful authentication.
        """
        malformed = "$2b$10$" + "x" * 53
        assert is_bcrypt_hash(malformed)
        # The function should return False for a malformed hash, not fall back
        # to plaintext comparison.
        assert verify_password("any-password", malformed) is False
        assert verify_password(malformed, malformed) is False

    def test_hash_password_empty_string(self):
        """Hashing an empty string should produce a valid hash."""
        hashed = hash_password("")
        assert has_password_hash(hashed)

    def test_hash_password_long_string(self):
        """Argon2id has no truncation limit, so a long secret still hashes.

        The strength policy caps what a user may *set* at 72 UTF-8 bytes (the
        legacy bcrypt limit); the hashing primitive itself is unbounded.
        """
        long_password = "x" * 1000
        hashed = hash_password(long_password)
        assert verify_password(long_password, hashed) is True
