"""Schemas for the authentication endpoints (``api/routers/auth.py``).

Login, logout, the first-run setup wizard and its status probe.
"""

from pydantic import BaseModel, Field, field_validator

from llm_proxy.api.schemas.common import validate_username_format
from llm_proxy.security.passwords import validate_password_strength


class LoginRequest(BaseModel):
    """Schema for admin login request."""

    username: str = Field(
        ...,
        min_length=1,
        max_length=320,
        description="Username or verified email address",
    )
    password: str = Field(..., description="Admin password")


class LoginResponse(BaseModel):
    """Schema for admin login response."""

    access_token: str | None = Field(default=None, description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    session_api_key: str = Field(default="", description="Session API key for /v1/* access")
    username: str | None = Field(
        default=None, description="Resolved account name; null on a 2FA challenge"
    )
    role: str | None = Field(default=None, description="Account role; null on a 2FA challenge")
    mfa_required: bool = Field(
        default=False,
        description="When true, complete the login with /api/auth/2fa/verify or /recover",
    )
    challenge_token: str | None = Field(
        default=None,
        description="Short-lived token to carry the first-factor result into the 2FA step",
    )
    must_change_password: bool = Field(
        default=False,
        description="When true, the user must set a new password before any other API access",
    )


class SetupRequest(BaseModel):
    """Schema for first-run admin account creation."""

    username: str = Field(
        ..., min_length=1, max_length=64, description="Username for the admin account"
    )
    password: str = Field(
        ...,
        min_length=8,
        max_length=72,
        description="Password for the admin account (8-72 characters)",
    )

    _validate_username = field_validator("username")(validate_username_format)
    _validate_password = field_validator("password")(validate_password_strength)


class SessionStatusResponse(BaseModel):
    """Who, if anyone, the browser's console cookie identifies."""

    authenticated: bool = False
    username: str | None = None
    role: str | None = None
    must_change_password: bool = False


class SetupStatusResponse(BaseModel):
    """Schema indicating whether first-run setup is required."""

    needs_setup: bool = Field(
        ..., description="True if no admin account exists yet and setup is required"
    )


class MfaVerifyRequest(BaseModel):
    """Complete a 2FA-gated login with a TOTP code."""

    challenge_token: str = Field(..., min_length=1)
    code: str = Field(..., min_length=6, max_length=10)


class MfaRecoverRequest(BaseModel):
    """Complete a 2FA-gated login with a one-time recovery code."""

    challenge_token: str = Field(..., min_length=1)
    recovery_code: str = Field(..., min_length=1, max_length=64)


class StepUpRequest(BaseModel):
    """Re-authenticate the current console session for a sensitive action."""

    password: str | None = Field(default=None, max_length=72)


class StepUpResponse(BaseModel):
    """Step-up result; a passkey may be required when the account has no password."""

    verified: bool = False
    passkey_required: bool = False


class EmailVerifyRequest(BaseModel):
    """Finish an email-address change from the link sent by SMTP."""

    token: str = Field(..., min_length=1)


class AuthCapabilities(BaseModel):
    """What this deployment's sign-in screen may offer (public, no credentials)."""

    password_reset: bool = Field(
        default=False,
        description="Self-service password reset works here (SMTP and public base URL set)",
    )


class PasswordResetRequest(BaseModel):
    """Start a self-service password reset for a username or verified email."""

    identifier: str = Field(
        ...,
        min_length=1,
        max_length=320,
        description="Username or verified email address",
    )


class PasswordResetRequested(BaseModel):
    """Acknowledgement; identical whether or not an email was actually sent."""

    requested: bool = Field(default=True, description="The request was accepted")


class PasswordResetConfirm(BaseModel):
    """Finish a password reset from the emailed link."""

    token: str = Field(..., min_length=1)
    password: str = Field(..., min_length=8, max_length=72)

    _validate_password = field_validator("password")(validate_password_strength)
