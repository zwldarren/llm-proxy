"""Tests for API key encryption utilities."""

import pytest
from cryptography.fernet import Fernet

from llm_proxy.security.encryption import init_encryption


class TestEncryptDecrypt:
    """Test encryption and decryption round-trip."""

    def test_encrypt_decrypt_roundtrip(self):
        """Encrypted key should decrypt to original."""
        key = Fernet.generate_key().decode()
        init_encryption(key)

        import llm_proxy.security.encryption as enc_module

        original = "sk-test-api-key-12345"
        encrypted = enc_module.encrypt_api_key(original)
        assert encrypted is not None
        assert encrypted.startswith("enc:")
        decrypted = enc_module.decrypt_api_key(encrypted)
        assert decrypted == original

    def test_encrypt_empty_returns_empty(self):
        """Empty string should return empty."""
        key = Fernet.generate_key().decode()
        init_encryption(key)

        import llm_proxy.security.encryption as enc_module

        assert enc_module.encrypt_api_key("") == ""
        assert enc_module.encrypt_api_key(None) is None

    def test_decrypt_unencrypted_returns_as_is(self):
        """Non-encrypted keys should return unchanged."""
        key = Fernet.generate_key().decode()
        init_encryption(key)

        import llm_proxy.security.encryption as enc_module

        plain = "sk-plain-key"
        assert enc_module.decrypt_api_key(plain) == plain

    def test_decrypt_batch(self):
        """Batch decryption should work for mixed keys."""
        key = Fernet.generate_key().decode()
        init_encryption(key)

        import llm_proxy.security.encryption as enc_module

        keys = ["sk-plain-1", "sk-plain-2"]
        encrypted_keys = [enc_module.encrypt_api_key(k) for k in keys]
        decrypted = enc_module.decrypt_api_keys(encrypted_keys)
        assert decrypted == keys


class TestDecryptInvalidToken:
    """Test InvalidToken handling - should raise error, not return encrypted key."""

    def test_decrypt_invalid_token_raises_error(self):
        """Decrypt should raise EncryptionError for InvalidToken, not return encrypted key."""
        from llm_proxy.core.exceptions import EncryptionError

        key1 = Fernet.generate_key().decode()
        key2 = Fernet.generate_key().decode()

        init_encryption(key1)
        import llm_proxy.security.encryption as enc_module

        original = "sk-test-api-key"
        encrypted = enc_module.encrypt_api_key(original)

        init_encryption(key2)

        with pytest.raises(EncryptionError) as exc_info:
            enc_module.decrypt_api_key(encrypted)

        assert "invalid token" in str(exc_info.value).lower()

    def test_decrypt_invalid_token_does_not_return_encrypted_key(self):
        """Decrypt with wrong key must not silently return encrypted key."""

        key1 = Fernet.generate_key().decode()

        init_encryption(key1)
        import llm_proxy.security.encryption as enc_module

        original = "sk-test-api-key"
        encrypted = enc_module.decrypt_api_key(original)

        assert not encrypted.startswith("enc:")


class TestEncryptionDisabled:
    """Tests for when encryption is disabled."""

    def test_init_with_none_key(self):
        """None key should disable encryption."""
        init_encryption(None)
        import llm_proxy.security.encryption as enc_module

        # When encryption disabled, returns plaintext
        assert enc_module.encrypt_api_key("my-key") == "my-key"

    def test_init_with_empty_key(self):
        """Empty key should disable encryption."""
        init_encryption("")
        import llm_proxy.security.encryption as enc_module

        assert enc_module.encrypt_api_key("my-key") == "my-key"

    def test_init_with_short_key(self):
        """Key shorter than 32 chars should disable encryption."""
        init_encryption("short-key")  # Less than 32 chars
        import llm_proxy.security.encryption as enc_module

        assert enc_module.encrypt_api_key("my-key") == "my-key"


class TestEncryptionEdgeCases:
    """Edge case tests for encryption."""

    def test_already_encrypted_key(self):
        """Already encrypted key should not be double-encrypted."""
        key = Fernet.generate_key().decode()
        init_encryption(key)
        import llm_proxy.security.encryption as enc_module

        original = "sk-test-key"
        encrypted = enc_module.encrypt_api_key(original)

        # Encrypting an already encrypted key should return it unchanged
        double_encrypted = enc_module.encrypt_api_key(encrypted)
        assert double_encrypted == encrypted

    def test_encrypt_none_returns_none(self):
        """None input should return None."""
        key = Fernet.generate_key().decode()
        init_encryption(key)
        import llm_proxy.security.encryption as enc_module

        assert enc_module.encrypt_api_key(None) is None

    def test_decrypt_none_returns_none(self):
        """None input should return None."""
        key = Fernet.generate_key().decode()
        init_encryption(key)
        import llm_proxy.security.encryption as enc_module

        assert enc_module.decrypt_api_key(None) is None

    def test_decrypt_disabled_raises_for_encrypted_value(self):
        """When encryption is disabled, decrypting an 'enc:' value must raise."""
        from llm_proxy.core.exceptions import EncryptionError

        init_encryption(None)
        import llm_proxy.security.encryption as enc_module

        # Fail closed: ciphertext cannot be decrypted without the key.
        with pytest.raises(EncryptionError):
            enc_module.decrypt_api_key("enc:something")

    def test_encrypt_very_long_key(self):
        """Very long API keys should be encryptable."""
        key = Fernet.generate_key().decode()
        init_encryption(key)
        import llm_proxy.security.encryption as enc_module

        long_key = "sk-" + "x" * 10000
        encrypted = enc_module.encrypt_api_key(long_key)
        decrypted = enc_module.decrypt_api_key(encrypted)
        assert decrypted == long_key

    def test_batch_with_none_values(self):
        """Batch decrypt should handle None values."""
        key = Fernet.generate_key().decode()
        init_encryption(key)
        import llm_proxy.security.encryption as enc_module

        result = enc_module.decrypt_api_keys([None, "sk-key", None])
        assert result == [None, "sk-key", None]

    def test_batch_empty_list(self):
        """Batch decrypt empty list should return empty list."""
        key = Fernet.generate_key().decode()
        init_encryption(key)
        import llm_proxy.security.encryption as enc_module

        result = enc_module.decrypt_api_keys([])
        assert result == []
