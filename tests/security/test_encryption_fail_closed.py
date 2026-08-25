"""Fail-closed tests for API key encryption.

Encryption failures must raise instead of silently storing plaintext, and
decryption must never silently return ciphertext.
"""

from unittest.mock import AsyncMock

import pytest
from cryptography.fernet import Fernet
from sqlalchemy.ext.asyncio import AsyncSession

from llm_proxy.core.exceptions import EncryptionError
from llm_proxy.security.encryption import init_encryption


class TestEncryptFailClosed:
    """Encryption failures must raise, never fall back to plaintext."""

    def test_encrypt_failure_raises_and_does_not_leak_plaintext(self, monkeypatch):
        """A failing Fernet.encrypt must raise and never yield the plaintext key."""
        key = Fernet.generate_key().decode()
        init_encryption(key)

        import llm_proxy.security.encryption as enc_module

        class FailingFernet:
            """Fernet stand-in whose encrypt primitive always fails."""

            def encrypt(self, data):
                raise RuntimeError("crypto backend unavailable")

        monkeypatch.setattr(enc_module, "_get_fernet", lambda: FailingFernet())

        plaintext = "sk-secret-key"
        with pytest.raises(EncryptionError) as exc_info:
            enc_module.encrypt_api_key(plaintext)

        assert "encrypt" in str(exc_info.value).lower()
        # The exception must not leak the plaintext key either.
        assert plaintext not in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_create_provider_persists_nothing_on_encryption_failure(self, monkeypatch):
        """A failing encrypt must abort provider creation before any DB write."""
        from llm_proxy.database.repositories.config_providers import ProviderRepository

        session = AsyncMock(spec=AsyncSession)
        repo = ProviderRepository(session)

        def failing_encrypt(*args, **kwargs):
            raise EncryptionError("Failed to encrypt API key: boom")

        monkeypatch.setattr(
            "llm_proxy.database.repositories.config_providers.encrypt_api_key",
            failing_encrypt,
        )

        with pytest.raises(EncryptionError):
            await repo.create_provider(name="p", type="openai", api_key="sk-secret-key")

        # Nothing was added to the session, so no plaintext can be persisted.
        session.add.assert_not_called()
        session.flush.assert_not_called()


class TestDecryptFailClosed:
    """Decryption must never silently return ciphertext."""

    def test_decrypt_enc_prefix_without_key_raises(self):
        """Decrypting an 'enc:' value while encryption is disabled must raise."""
        init_encryption(None)

        import llm_proxy.security.encryption as enc_module

        with pytest.raises(EncryptionError):
            enc_module.decrypt_api_key("enc:something")

    def test_decrypt_legacy_plaintext_still_readable(self):
        """Legacy plaintext rows (no 'enc:' prefix) must still read as-is."""
        init_encryption(None)

        import llm_proxy.security.encryption as enc_module

        assert enc_module.decrypt_api_key("sk-legacy-plain") == "sk-legacy-plain"
