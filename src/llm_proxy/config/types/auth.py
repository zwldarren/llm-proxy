"""Authentication configuration types."""

from pydantic import BaseModel, Field, model_validator


class ProxyAuthConfig(BaseModel):
    """Authentication configuration for the proxy."""

    jwt_secret: str = Field(default="", description="JWT secret for signing admin tokens")

    @model_validator(mode="after")
    def _require_jwt_secret(self):
        if not self.jwt_secret:
            raise ValueError("jwt_secret must be set (authentication is always enabled)")
        return self

    model_config = {"extra": "forbid"}
