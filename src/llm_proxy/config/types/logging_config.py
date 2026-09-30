"""Logging configuration types."""

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from llm_proxy.observability.types import LogType


#: Default retention window (days) for request logs, for audit logs that inherit
#: it, and for usage records. ``0`` keeps rows indefinitely.
#:
#: Keep this at or above the longest budget window (monthly budgets aggregate
#: ``usage_records`` from the 1st of the month, i.e. up to 31 days back):
#: a shorter window silently under-counts spend and stops enforcing budgets.
DEFAULT_RETENTION_DAYS = 30

#: Default cap (bytes) on the serialized size of a stored request/response body.
#: Larger bodies are replaced by a ``{"_truncated": true, "size": N}`` marker so
#: one oversized payload (a huge context, a base64 image) cannot dominate the
#: log store. ``0`` disables the cap. Matches ``max_response_body_bytes``, the
#: existing 1 MiB ceiling on buffered raw SSE.
DEFAULT_MAX_LOGGED_BODY_BYTES = 1024 * 1024


class LoggingConfig(BaseModel):
    log_input_output: bool = Field(
        default=False,
        description=(
            "Log request/response bodies. Off by default: rows still persist with "
            "bodies scrubbed to a marker — metadata, tokens, cost, routing and audit "
            "fields are unaffected. This is the master switch, so ``x-log-full: true`` "
            "cannot re-enable bodies for a single request"
        ),
    )
    log_raw_stream: bool = Field(
        default=False,
        description=(
            "Store the raw SSE text of streaming responses. Off by default: the "
            "reassembled non-streaming response body is stored instead — an order "
            "of magnitude smaller (the SSE envelope repeats on every delta) and "
            "renderable without client-side SSE parsing"
        ),
    )
    retention_days: int = Field(
        default=DEFAULT_RETENTION_DAYS,
        description="How many days to retain request logs",
    )
    max_logged_body_bytes: int = Field(
        default=DEFAULT_MAX_LOGGED_BODY_BYTES,
        ge=0,
        description=(
            "Cap on the serialized size of a stored request/response body in bytes; "
            "larger bodies are stored as a truncation marker. 0 disables the cap"
        ),
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
