"""Schemas for the authentication endpoints (``api/routers/auth.py``).

Login, logout, the first-run setup wizard and its status probe.
"""

from pydantic import BaseModel, Field, field_validator

from llm_proxy.security.passwords import validate_password_strength


class LoginRequest(BaseModel):
    """Schema for admin login request."""

    username: str = Field(..., description="Admin username")
    password: str = Field(..., description="Admin password")


class LoginResponse(BaseModel):
    """Schema for admin login response."""

    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    session_api_key: str = Field(default="", description="Session API key for /v1/* access")
    must_change_password: bool = Field(
        default=False,
        description="When true, the user must set a new password before any other API access",
    )


class SetupRequest(BaseModel):
    """Schema for first-run admin account creation."""

    username: str = Field(
        ..., min_length=1, max_length=100, description="Username for the admin account"
    )
    password: str = Field(
        ...,
        min_length=8,
        max_length=72,
        description="Password for the admin account (8-72 characters)",
    )

    _validate_password = field_validator("password")(validate_password_strength)


class SetupStatusResponse(BaseModel):
    """Schema indicating whether first-run setup is required."""

    needs_setup: bool = Field(
        ..., description="True if no admin account exists yet and setup is required"
    )
