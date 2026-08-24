"""JWT authentication utilities."""

from datetime import UTC, datetime, timedelta
from typing import Any

import jwt

from llm_proxy.config.types.auth import ProxyAuthConfig
from llm_proxy.core.exceptions import ConfigurationError


class JWTManager:
    """JWT token management for admin authentication.

    The JWTManager only signs and verifies tokens using the configured JWT
    secret. Admin credential verification is performed by the login endpoint
    against the ``users`` table (see ``UserRepository`` and
    ``verify_admin_password``), not here.
    """

    _ALGORITHM = "HS256"
    _EXPIRES_IN = timedelta(hours=24)

    def __init__(self, auth_config: ProxyAuthConfig):
        self.auth_config = auth_config
        self.secret = auth_config.jwt_secret

    _ALLOWED_ROLES: frozenset[str] = frozenset({"admin", "viewer"})

    def create_token(self, username: str, role: str = "admin", token_version: int = 0) -> str:
        """Create a JWT token for the given username.

        Validates that the role is one of the allowed roles (admin, member).
        The token embeds the user's current ``token_version`` ("tv" claim);
        bumping the version server-side (e.g. on password change) immediately
        invalidates all previously issued tokens.
        """
        if role not in self._ALLOWED_ROLES:
            raise ConfigurationError(
                f"Invalid role '{role}'. Allowed roles: {sorted(self._ALLOWED_ROLES)}"
            )
        if not self.secret:
            raise ConfigurationError("JWT secret not configured")
        now = datetime.now(tz=UTC)
        payload = {
            "sub": username,
            "role": role,
            "exp": now + self._EXPIRES_IN,
            "iat": now,
            "type": role,  # reflects the actual role, not hardcoded "admin"
            "tv": token_version,
        }
        return jwt.encode(payload, self.secret, algorithm=self._ALGORITHM)

    def verify_token(self, token: str) -> dict[str, Any]:
        """Verify JWT token and return payload."""
        if not self.secret:
            raise ConfigurationError("JWT secret not configured")
        try:
            return jwt.decode(
                token,
                self.secret,
                algorithms=[self._ALGORITHM],
                options={"verify_exp": True},
            )
        except jwt.InvalidTokenError as e:
            raise ConfigurationError(f"Invalid token: {e}") from e
