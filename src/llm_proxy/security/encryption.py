"""API key encryption utilities for secure storage.

This module provides encryption/decryption functionality for API keys
stored in the database using Fernet symmetric encryption.

SECURITY NOTE: The encryption key is derived from the ENCRYPTION_KEY environment
variable. This key must be kept secret and should be at least 32 characters.

Usage:
    # At startup:
    from llm_proxy.security.encryption import init_encryption
    init_encryption(encryption_key)

    # Then use the encryption functions:
    encrypted = encrypt_api_key(api_key)
    decrypted = decrypt_api_key(encrypted)
"""

import base64
import hashlib

from cryptography.fernet import Fernet, InvalidToken

from llm_proxy.core.exceptions import EncryptionError
from llm_proxy.observability.logger import get_logger

logger = get_logger(__name__)

# Minimum key length required
_MIN_KEY_LENGTH = 32

# Cache the Fernet instance
_fernet_instance: Fernet | None = None
_encryption_enabled: bool | None = None


def _derive_key(encryption_key: str) -> bytes:
    """Derive a valid Fernet key from the given encryption key."""
    derived_key = hashlib.sha256(encryption_key.encode()).digest()
    return base64.urlsafe_b64encode(derived_key)


def init_encryption(encryption_key: str | None = None) -> None:
    """Initialize the encryption module with the given key.

    Must be called at startup before any encryption/decryption operations.
    If encryption_key is None or empty, encryption is disabled and API keys
    will be stored in plaintext.

    Args:
        encryption_key: The encryption key (at least 32 characters).
    """
    global _fernet_instance, _encryption_enabled

    _fernet_instance = None
    _encryption_enabled = False

    if not encryption_key:
        logger.warning(
            "SECURITY WARNING: API key encryption is DISABLED. "
            "API keys will be stored in PLAINTEXT in the database. "
            "Set ENCRYPTION_KEY environment variable (at least 32 characters) "
            "to enable encryption. "
            "Generate a secure key with: "
            'python -c "import secrets; print(secrets.token_urlsafe(32))"'
        )
        return

    if len(encryption_key) < _MIN_KEY_LENGTH:
        logger.warning(
            f"ENCRYPTION_KEY is less than {_MIN_KEY_LENGTH} characters. "
            f"API key encryption is disabled for security."
        )
        return

    key_bytes = _derive_key(encryption_key)
    _fernet_instance = Fernet(key_bytes)
    _encryption_enabled = True


def _get_fernet() -> Fernet | None:
    """Get the Fernet instance for encryption/decryption.

    Requires init_encryption() to have been called at startup.
    """
    return _fernet_instance if _encryption_enabled else None


def encrypt_api_key(api_key: str | None) -> str | None:
    """Encrypt an API key for secure storage.

    Args:
        api_key: The plaintext API key to encrypt.

    Returns:
        The encrypted API key as a base64 string, prefixed with 'enc:'.
        If encryption is not enabled, returns the original key.
        Returns None if api_key is None.
    """
    if not api_key:
        return api_key

    # Already encrypted
    if api_key.startswith("enc:"):
        return api_key

    fernet = _get_fernet()
    if fernet is None:
        logger.debug("Encryption not enabled, storing API key in plaintext")
        return api_key

    try:
        encrypted = fernet.encrypt(api_key.encode())
        return f"enc:{encrypted.decode()}"
    except Exception as e:
        logger.error(f"Failed to encrypt API key: {e}")
        # Fall back to plaintext if encryption fails
        return api_key


def decrypt_api_key(encrypted_key: str | None) -> str | None:
    """Decrypt an encrypted API key.

    Args:
        encrypted_key: The encrypted API key (prefixed with 'enc:').

    Returns:
        The decrypted plaintext API key.
        If the key is not encrypted (no 'enc:' prefix), returns as-is.
        Returns None if encrypted_key is None.
    """
    if not encrypted_key:
        return encrypted_key

    # Not encrypted
    if not encrypted_key.startswith("enc:"):
        return encrypted_key

    fernet = _get_fernet()
    if fernet is None:
        logger.warning(
            "Cannot decrypt API key: encryption not enabled. "
            "Set ENCRYPTION_KEY environment variable."
        )
        # Return the encrypted value (will likely fail when used)
        return encrypted_key

    try:
        encrypted_data = encrypted_key[4:]  # Remove 'enc:' prefix
        decrypted = fernet.decrypt(encrypted_data.encode())
        return decrypted.decode()
    except InvalidToken as e:
        logger.error(
            "Failed to decrypt API key: invalid token. The ENCRYPTION_KEY may have changed."
        )
        raise EncryptionError("Failed to decrypt API key: invalid token") from e
    except Exception as e:
        logger.error(f"Failed to decrypt API key: {e}")
        raise EncryptionError(f"Failed to decrypt API key: {e}") from e


def decrypt_api_keys(encrypted_keys: list[str | None]) -> list[str | None]:
    """Batch decrypt a list of encrypted API keys.

    This is more efficient than calling decrypt_api_key() multiple times
    as it retrieves the Fernet instance once.

    Args:
        encrypted_keys: List of encrypted API keys.

    Returns:
        List of decrypted plaintext API keys in the same order.
    """
    if not encrypted_keys:
        return []
    return [decrypt_api_key(key) for key in encrypted_keys]
