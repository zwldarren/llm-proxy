"""Schemas for TOTP enrollment, recovery codes and 2FA-gated login."""

from pydantic import BaseModel, Field


class TwoFactorStatus(BaseModel):
    enabled: bool = False
    recovery_codes_remaining: int = 0
    required_for_admins: bool = False


class TwoFactorBeginResponse(BaseModel):
    secret: str
    otpauth_uri: str


class TwoFactorCodeRequest(BaseModel):
    code: str = Field(..., min_length=6, max_length=10)


class TwoFactorRecoveryCodesResponse(BaseModel):
    recovery_codes: list[str]


class TwoFactorDisableRequest(BaseModel):
    code: str = Field(..., min_length=6, max_length=10)


class TwoFactorRecoveryRegenerateRequest(BaseModel):
    code: str = Field(..., min_length=6, max_length=10)
