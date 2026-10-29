"""Base64url helpers shared by WebAuthn challenges and OAuth state encoding."""

import base64


def b64url_encode(data: bytes) -> str:
    """Encode bytes as unpadded base64url."""
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def b64url_decode(value: str) -> bytes:
    """Decode unpadded base64url text, tolerating missing padding."""
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
