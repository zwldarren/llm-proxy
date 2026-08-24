"""Logging configuration types."""

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from llm_proxy.observability.types import LogType


class LoggingConfig(BaseModel):
    enable_database_logging: bool = Field(
        default=True, description="Persist request logs to the database"
    )
    retention_days: int = Field(
        default=30,
        description="How many days to retain request logs",
    )
    mask_sensitive_data: bool = Field(default=True, description="Mask sensitive fields in logs")
    log_level: str = Field(default="INFO", description="Logging verbosity")
    sampling_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Rate at which to log full request/response bodies",
    )
    audit_sampling_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Sampling rate for audit logs",
    )
    audit_retention_days: int | None = Field(
        default=None,
        ge=0,
        description="Retention days for audit logs",
    )
    sensitive_keys: list[str] = Field(
        default_factory=lambda: [
            "authorization",
            "api_key",
            "apikey",
            "password",
            "passwd",
            "token",
            "access_token",
            "refresh_token",
            "jwt_secret",
        ],
        description="List of key names to mask in logs",
    )

    verbose_routing_logs: bool = Field(
        default=False,
        description="Include detailed per-candidate routing scorecards in request log metadata",
    )

    def get_sampling_rate(self, log_type: LogType | str | None = None) -> float:
        if log_type is not None:
            value = getattr(log_type, "value", log_type)
            if value == "audit" and self.audit_sampling_rate is not None:
                return self.audit_sampling_rate
        return self.sampling_rate

    def get_retention_days(self, log_type: LogType | str | None = None) -> int:
        if log_type is not None:
            value = getattr(log_type, "value", log_type)
            if value == "audit" and self.audit_retention_days is not None:
                return self.audit_retention_days
        return self.retention_days
