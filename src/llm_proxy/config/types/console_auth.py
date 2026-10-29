"""Console authentication settings stored in ``server_config``.

The JWT signing secret stays in the secrets holder (``ProxyAuthConfig``).
This model owns the operator-facing settings that the admin UI manages:
the public base URL used for WebAuthn and OAuth callbacks, optional SMTP
delivery, and optional OIDC providers.

Secrets (SMTP password, OIDC client secrets) are encrypted at rest by the
server-settings write transform and masked in the admin response. Runtime
reads decrypt them for the auth services.
"""

from pydantic import BaseModel, Field, field_validator


class SmtpSettings(BaseModel):
    """Optional SMTP delivery for invitations, email verification and password resets."""

    enabled: bool = Field(
        default=False, description="Send invitation, verification and password-reset email"
    )
    host: str = Field(default="", max_length=255)
    port: int = Field(default=587, ge=1, le=65535)
    username: str = Field(default="", max_length=255)
    password: str = Field(default="", description="SMTP password (write-only; masked on read)")
    from_address: str = Field(default="", max_length=320)
    starttls: bool = Field(default=True)
    use_ssl: bool = Field(default=False)

    @field_validator("host", "username", "from_address")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()


class OidcProviderSettings(BaseModel):
    """One configurable OpenID Connect provider."""

    id: str = Field(..., min_length=1, max_length=50, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    name: str = Field(..., min_length=1, max_length=100)
    issuer: str = Field(..., min_length=1, max_length=500)
    client_id: str = Field(..., min_length=1, max_length=255)
    client_secret: str = Field(default="", description="Client secret (write-only; masked on read)")
    scopes: str = Field(default="openid email profile", max_length=255)
    enabled: bool = Field(default=True)

    @field_validator("issuer")
    @classmethod
    def _strip_issuer(cls, value: str) -> str:
        return value.rstrip("/")


class ConsoleAuthSettings(BaseModel):
    """Operator-facing console auth configuration (server_config ``console_auth``)."""

    public_base_url: str = Field(
        default="",
        description=(
            "Canonical external URL, e.g. https://proxy.example.com. Used for the "
            "WebAuthn RP ID/origin and OAuth redirect URIs. Must be HTTPS (or "
            "http://localhost for development)."
        ),
    )
    require_2fa_for_admins: bool = Field(
        default=False,
        description="Require every admin account to enroll TOTP before using the console",
    )
    smtp: SmtpSettings = Field(default_factory=SmtpSettings)
    oidc_providers: list[OidcProviderSettings] = Field(default_factory=list)

    @field_validator("public_base_url")
    @classmethod
    def _strip_base_url(cls, value: str) -> str:
        return value.strip().rstrip("/")
