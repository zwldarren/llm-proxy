"""Console authentication settings stored in ``server_config``.

The JWT signing secret stays in the secrets holder (``ProxyAuthConfig``).
This model owns the operator-facing settings that the admin UI manages:
the public base URL used for WebAuthn and OAuth callbacks, optional email
delivery (SMTP or Resend), and optional OIDC providers.

Secrets (SMTP password, Resend API key, OIDC client secrets) are encrypted
at rest by the server-settings write transform and masked in the admin
response. Runtime reads decrypt them for the auth services.
"""

from typing import ClassVar, Literal

from pydantic import BaseModel, Field, field_validator


class SmtpSettings(BaseModel):
    """Optional SMTP delivery for invitations, email verification and password resets."""

    #: Name this transport goes by in logs, the test-email response and
    #: ``users.email_verified_by``. Each block answers for itself so no consumer
    #: has to re-derive the kind from the class.
    kind: ClassVar[Literal["smtp"]] = "smtp"

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

    def is_deliverable(self) -> bool:
        """Whether this block carries every detail a delivery attempt needs."""
        return bool(self.enabled and self.host and self.from_address)


class ResendSettings(BaseModel):
    """Optional Resend (https://resend.com) delivery for the same messages."""

    kind: ClassVar[Literal["resend"]] = "resend"

    enabled: bool = Field(
        default=False,
        description="Send invitation, verification and password-reset email via Resend",
    )
    api_key: str = Field(default="", description="Resend API key (write-only; masked on read)")
    from_address: str = Field(default="", max_length=320)

    @field_validator("api_key", "from_address")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()

    def is_deliverable(self) -> bool:
        """Whether this block carries every detail a delivery attempt needs."""
        return bool(self.enabled and self.api_key and self.from_address)


#: One of the two blocks ``ConsoleAuthSettings`` stores; only one is ever used.
type EmailTransport = SmtpSettings | ResendSettings


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
    require_2fa_after_passkey: bool = Field(
        default=False,
        description=(
            "Also ask for a TOTP or recovery code after a passkey sign-in. A passkey "
            "assertion already proves possession plus user verification, so this is off "
            "by default; turn it on only when a compliance regime demands a second "
            "prompt regardless of the primary factor"
        ),
    )
    smtp: SmtpSettings = Field(default_factory=SmtpSettings)
    resend: ResendSettings = Field(default_factory=ResendSettings)
    oidc_providers: list[OidcProviderSettings] = Field(default_factory=list)

    @field_validator("public_base_url")
    @classmethod
    def _strip_base_url(cls, value: str) -> str:
        return value.strip().rstrip("/")

    def email_transport(self) -> EmailTransport | None:
        """The one transport outbound mail goes through, or ``None`` when unconfigured.

        Both blocks stay stored and independently editable, so an operator can
        switch between them without retyping the other block's credentials, but
        only one of them is ever used. An enabled block is the operator's
        choice: when Resend is enabled it is the transport, and if it is
        incomplete the answer is ``None`` rather than SMTP. Falling back would
        hide a broken Resend setup behind mail that quietly left by another
        route, and the operator would only learn about it from a delivery
        report for a transport they had turned off.
        """
        if self.resend.enabled:
            return self.resend if self.resend.is_deliverable() else None
        return self.smtp if self.smtp.is_deliverable() else None

    @property
    def email_enabled(self) -> bool:
        """Whether a transport is configured well enough to deliver mail."""
        return self.email_transport() is not None
