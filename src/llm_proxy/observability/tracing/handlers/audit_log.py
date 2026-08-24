"""Audit log handler for database logging via the unified capture layer.

This handler replaces the HTTP logging middleware's database writing logic
by consuming events from the TracingRegistry with EventContext.
"""

import socket
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from llm_proxy.core.utils import safe_int
from llm_proxy.observability.audit_helpers import (
    determine_action_category,
    determine_event_type,
    determine_outcome,
    determine_resource_id,
    determine_resource_type,
)
from llm_proxy.observability.cost import calculate_event_cost
from llm_proxy.observability.logger import get_logger
from llm_proxy.observability.service import RequestLogCreate, UsageRecordCreate, UsageService
from llm_proxy.observability.tracing.handlers.base import TracingHandler
from llm_proxy.observability.types import LogType
from llm_proxy.security.passwords import mask_headers, mask_sensitive

if TYPE_CHECKING:
    from llm_proxy.config.manager import DatabaseConfigManager
    from llm_proxy.config.types.logging_config import LoggingConfig
    from llm_proxy.models import InternalRequest, InternalResponse
    from llm_proxy.observability.event_context import EventContext

logger = get_logger(__name__)


def _get_server_hostname() -> str:
    """Get the server hostname for audit logs."""
    try:
        return socket.gethostname()
    except Exception:
        logger.debug("Failed to get server hostname", exc_info=True)
        return "unknown"


_VERBOSE_ROUTING_KEYS: frozenset[str] = frozenset(
    {
        "candidate_scorecards",
        "weights_used",
        "guardrail_notes",
        "signal_votes",
    }
)


@dataclass
class TokenMetadata:
    """Token and cost metadata extracted from EventContext."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    cached_prompt_tokens: int | None = None
    cache_savings_usd: float | None = None
    audio_input_tokens: int | None = None
    audio_output_tokens: int | None = None


def _extract_token_metadata(context: EventContext) -> TokenMetadata:
    """Extract token and cost metadata from EventContext."""
    return TokenMetadata(
        prompt_tokens=context.prompt_tokens,
        completion_tokens=context.completion_tokens,
        total_tokens=context.total_tokens,
        cost_usd=context.cost_usd,
        cache_creation_input_tokens=context.cache_creation_input_tokens,
        cache_read_input_tokens=context.cache_read_input_tokens,
        cached_prompt_tokens=context.cached_prompt_tokens,
        cache_savings_usd=context.cache_savings_usd,
        audio_input_tokens=context.audio_input_tokens,
        audio_output_tokens=context.audio_output_tokens,
    )


class AuditLogHandler(TracingHandler):
    """Handler that writes request logs to database.

    Replaces the logging middleware's database writing logic.
    Respects sampling decisions made in EventContext.
    UsageRecord is always created independently of sampling.

    This handler implements the new *_with_context methods to receive
    unified EventContext with pre-computed token/cost data.
    """

    provider_name = "audit_log"

    def __init__(
        self,
        enabled: bool = True,
        config: LoggingConfig | None = None,
        config_manager: DatabaseConfigManager | None = None,
    ) -> None:
        super().__init__(enabled=enabled)
        self._config = config
        self._config_manager = config_manager
        self._usage_service = UsageService(retention_days=365)
        # Note: RequestLogService needs to be created lazily when config is available

    def _get_request_log_service(self):
        """Get or create RequestLogService.

        Refreshes the logging config from the config manager's cached
        ProxyConfig on every call so UI-managed changes (retention, masking,
        sampling) take effect without a restart; falls back to the startup
        config / code defaults when no manager is available.
        """
        if self._config_manager is not None:
            from llm_proxy.config.manager import resolve_logging_config

            self._config = resolve_logging_config(self._config_manager)
        if self._config is None:
            from llm_proxy.config.manager import load_logging_config

            self._config = load_logging_config()

        from llm_proxy.observability.service import RequestLogService

        return RequestLogService(self._config)

    async def on_request_start(
        self,
        request: InternalRequest,
        context: EventContext,
    ) -> None:
        """Store context reference for later use.

        Args:
            request: The unified request
            context: Event context with sampling decision
        """
        if not self._enabled:
            return

    async def on_request_end(
        self,
        request: InternalRequest,
        response: InternalResponse,
        context: EventContext,
    ) -> None:
        """Write request log to database.

        Args:
            request: The unified request
            response: The unified response
            context: Event context with accumulated data
        """
        if not self._enabled:
            return

        # Update context with response data (use getattr for response types
        # like InternalSpeechResponse that don't have usage)
        response_usage = getattr(response, "usage", None)
        if response_usage is not None:
            context.update_usage(response_usage)
        # Respect any status already set (e.g. by the processing pipeline) and
        # only default to 200 when unset.
        if context.response_status_code is None:
            context.response_status_code = 200

        # Calculate cost if not already done
        if context.cost_usd is None and context.has_billable_data() and self._config_manager:
            await self._calculate_cost(context)

        # Build and write log
        log_data = self._build_log_create(request, response, context)
        self._get_request_log_service().create_log_background(log_data)

        # Always create usage record (independent of sampling)
        usage_record = self._build_usage_record(context)
        self._usage_service.create_usage_background(usage_record)

    async def on_error(
        self,
        request: InternalRequest,
        error: Exception,
        context: EventContext,
    ) -> None:
        """Write error log to database.

        For streaming requests, on_stream_end will write the log with
        captured streaming data, so we skip writing here.

        Args:
            request: The unified request
            error: The exception that occurred
            context: Event context with error details
        """
        if not self._enabled:
            return

        from llm_proxy.core.errors.utils import extract_error_details
        from llm_proxy.core.exceptions import ProviderError
        from llm_proxy.observability.service import format_exception_stacktrace

        # Populate error information in context
        context.error_message = str(error)
        if isinstance(error, ProviderError):
            context.response_status_code = error.status_code
        else:
            context.response_status_code = getattr(error, "status_code", None) or 500
        context.error_stack_trace = format_exception_stacktrace(error)
        context.error_details = extract_error_details(error)

        # Skip writing log for streaming - on_stream_end will handle it with captured data
        if context.is_streaming:
            return

        # Build and write error log
        log_data = self._build_error_log_create(request, error, context)
        self._get_request_log_service().create_log_background(log_data)

        # Always create usage record on error
        usage_record = self._build_usage_record(context)
        self._usage_service.create_usage_background(usage_record)

    async def on_stream_start(
        self,
        request: InternalRequest,
        context: EventContext,
    ) -> None:
        """Initialize streaming capture.

        Args:
            request: The unified request
            context: Event context for streaming state
        """
        if not self._enabled:
            return
        context.is_streaming = True

    async def on_stream_chunk(
        self,
        request: InternalRequest,
        chunk: str,
        context: EventContext,
    ) -> None:
        """Capture streaming chunk for logging.

        Args:
            request: The unified request
            chunk: The SSE chunk data
            context: Event context for streaming capture
        """
        if not self._enabled:
            return

        # Capture chunk if sampling allows
        context.capture_streaming_chunk(chunk)

    async def on_stream_end(
        self,
        request: InternalRequest,
        context: EventContext,
        error: Exception | None = None,
    ) -> None:
        """Write streaming log to database.

        Args:
            request: The unified request
            context: Event context with streaming data
            error: Exception if stream ended with error
        """
        if not self._enabled:
            return

        if error is not None:
            from llm_proxy.core.errors.utils import extract_error_details
            from llm_proxy.core.exceptions import ProviderError
            from llm_proxy.observability.service import format_exception_stacktrace

            context.error_message = str(error)
            if isinstance(error, ProviderError):
                context.response_status_code = error.status_code
            else:
                context.response_status_code = getattr(error, "status_code", None) or 500
            context.error_stack_trace = format_exception_stacktrace(error)
            context.error_details = extract_error_details(error)
        else:
            context.response_status_code = 200

        # Extract usage from transformer if available
        if context.transformer and hasattr(context.transformer, "get_usage"):
            usage = context.transformer.get_usage()
            if usage:
                context.update_usage(usage)

        # Calculate cost if needed
        if context.cost_usd is None and context.has_billable_data() and self._config_manager:
            await self._calculate_cost(context)

        # Set response body from captured streaming data
        if context.should_capture_full_body:
            streaming_body = context.get_streaming_body()
            # Convert bytes to string for JSON serialization
            if streaming_body:
                try:
                    # Try to decode as UTF-8 text (SSE chunks are text)
                    context.response_body = streaming_body.decode("utf-8")
                except UnicodeDecodeError:
                    # If not valid UTF-8, store as base64
                    import base64

                    context.response_body = {
                        "encoding": "base64",
                        "data": base64.b64encode(streaming_body).decode("ascii"),
                    }
            else:
                context.response_body = None

        # Build and write log
        log_data = self._build_streaming_log_create(request, context)
        self._get_request_log_service().create_log_background(log_data)

        # Always create usage record
        usage_record = self._build_usage_record(context)
        self._usage_service.create_usage_background(usage_record)

    async def _calculate_cost(self, context: EventContext) -> None:
        """Calculate cost for the request if not already populated.

        Delegates to the shared ``calculate_event_cost`` helper so that the
        precedence rules (provider-reported cost > pricing DB) stay in one
        place. The processing pipeline also calls this helper *before*
        dispatching end events, so by the time this runs the cost is usually
        already set and this is a no-op.

        Args:
            context: Event context with token data
        """
        await calculate_event_cost(context, self._config_manager)

    def _mask_request_data(
        self,
        context: EventContext,
    ) -> tuple[dict[str, Any], dict[str, Any] | str]:
        """Mask request headers and body based on config.

        Args:
            context: Event context with request data

        Returns:
            Tuple of (masked_headers, masked_body)
        """
        request_headers = context.request_headers
        request_body = context.request_body

        if self._config and self._config.mask_sensitive_data:
            sensitive_keys = frozenset(k.lower() for k in self._config.sensitive_keys)
            request_headers = mask_headers(request_headers)
            if context.should_capture_full_body and isinstance(request_body, dict):
                request_body = mask_sensitive(request_body, sensitive_keys)

        if not context.should_capture_full_body:
            return {}, {"_sampled_out": True}

        return request_headers, request_body

    def _mask_response_data(
        self,
        context: EventContext,
    ) -> tuple[dict[str, Any], dict[str, Any] | str]:
        """Mask response headers and body based on config.

        Args:
            context: Event context with response data

        Returns:
            Tuple of (masked_headers, masked_body)
        """
        response_headers = context.response_headers
        response_body = context.response_body

        if self._config and self._config.mask_sensitive_data:
            sensitive_keys = frozenset(k.lower() for k in self._config.sensitive_keys)
            response_headers = mask_headers(response_headers)
            if context.should_capture_full_body and isinstance(response_body, dict):
                response_body = mask_sensitive(response_body, sensitive_keys)

        return response_headers, response_body

    def _build_log_base(
        self,
        context: EventContext,
        default_status_code: int = 200,
    ) -> dict[str, Any]:
        """Build base fields for RequestLogCreate.

        Args:
            context: Event context with all data
            default_status_code: Default status code if none set

        Returns:
            Dict with common fields for RequestLogCreate
        """
        token_meta = _extract_token_metadata(context)
        log_type = (
            LogType(context.log_type) if isinstance(context.log_type, str) else context.log_type
        )

        endpoint = context.metadata.get("endpoint", "")
        method = context.metadata.get("method", "POST")

        return {
            "request_id": context.request_id,
            "timestamp": context.start_timestamp,
            "endpoint": endpoint,
            "log_type": log_type,
            "method": method,
            "status_code": context.response_status_code or default_status_code,
            "response_time_ms": safe_int(context.latency_ms),
            "user_identity": context.user_id,
            "user_id": context.auth_user_id,
            "model": context.model,
            "provider": context.provider,
            "error_message": context.error_message,
            "error_stack_trace": context.error_stack_trace,
            "prompt_tokens": token_meta.prompt_tokens,
            "completion_tokens": token_meta.completion_tokens,
            "total_tokens": token_meta.total_tokens,
            "cache_creation_input_tokens": token_meta.cache_creation_input_tokens,
            "cache_read_input_tokens": token_meta.cache_read_input_tokens,
            "cached_prompt_tokens": token_meta.cached_prompt_tokens,
            "cost_usd": token_meta.cost_usd,
            "cache_savings_usd": token_meta.cache_savings_usd,
            "ttft_ms": safe_int(context.ttft_ms) if context.ttft_ms is not None else None,
            "api_key_name": context.api_key_name,
            "client_ip": context.client_ip,
            "user_agent": context.user_agent,
            "session_id": context.session_id,
            "auth_method": context.auth_method,
            "server_hostname": _get_server_hostname(),
            "service_name": "llm-proxy",
            "event_type": determine_event_type(endpoint),
            "action_category": determine_action_category(method),
            "resource_type": determine_resource_type(endpoint),
            "resource_id": determine_resource_id(endpoint, context.request_body),
            "outcome": determine_outcome(context.response_status_code, context.error_message),
        }

    def _extract_routing_metadata(self, context: EventContext) -> dict[str, Any]:
        """Extract routing analytics from EventContext metadata.

        Args:
            context: Event context with routing metadata

        Returns:
            Dict with routing analytics or empty dict if not available
        """
        routing = context.metadata.get("routing", {})
        if not routing:
            return {}
        result: dict[str, Any] = {
            "routing_complexity": routing.get("complexity"),
            "routing_confidence": routing.get("confidence"),
            "routing_reasoning": routing.get("reasoning"),
            "routing_cost_estimate": routing.get("cost_estimate"),
            "routing_savings": routing.get("savings"),
            "routing_tier": routing.get("tier"),
        }
        return result

    def _redact_verbose_routing(self, log_metadata: dict[str, Any]) -> dict[str, Any]:
        """Strip verbose nested routing keys from persisted log metadata.

        ``_build_log_create`` / ``_build_streaming_log_create`` spread
        ``**context.metadata`` into ``log_metadata``. The verbose nested
        ``routing`` keys (candidate_scorecards, weights_used, guardrail_notes,
        signal_votes) are already omitted upstream when the toggle is disabled,
        but this handler acts as defense-in-depth: it replaces the nested
        ``routing`` dict with a shallow copy that omits the verbose keys when
        ``verbose_routing_logs`` is off.

        A fresh dict is built from the nested ``routing`` mapping rather than
        mutating it in place, so the original ``context.metadata`` object (and
        its nested ``routing`` dict) is never modified.

        The compact flat ``routing_*`` keys emitted by
        ``_extract_routing_metadata`` are always included; only the verbose
        nested dict is guarded here.
        """
        verbose_enabled = self._config and getattr(self._config, "verbose_routing_logs", False)
        if not verbose_enabled:
            routing = log_metadata.get("routing")
            if isinstance(routing, dict):
                log_metadata["routing"] = {
                    key: value for key, value in routing.items() if key not in _VERBOSE_ROUTING_KEYS
                }
        return log_metadata

    def _add_fallback_metadata(
        self,
        log_metadata: dict[str, Any],
        context: EventContext,
    ) -> dict[str, Any]:
        """Add fallback attempts to log metadata if present.

        Args:
            log_metadata: Existing log metadata
            context: Event context with fallback attempts

        Returns:
            Updated log metadata with fallback information
        """
        if context.fallback_attempts:
            log_metadata["fallback_attempts"] = context.fallback_attempts
            log_metadata["fallback_count"] = len(context.fallback_attempts)
            failed_providers = [a["provider"] for a in context.fallback_attempts]
            log_metadata["fallback_providers"] = failed_providers
        return log_metadata

    def _add_retry_metadata(
        self,
        log_metadata: dict[str, Any],
        context: EventContext,
    ) -> dict[str, Any]:
        """Add same-provider retry attempts to log metadata if present.

        Each entry records a failed same-provider attempt (transient / server
        error) from RetryPolicy. ``retry_count`` is the number of attempts that
        were actually retried (``retried=True``); the full list is kept for the
        frontend timeline, including the terminal exhausted attempt.
        """
        if context.retry_attempts:
            log_metadata["retry_attempts"] = context.retry_attempts
            log_metadata["retry_count"] = sum(1 for a in context.retry_attempts if a.get("retried"))
        return log_metadata

    def _build_log_metadata(
        self,
        context: EventContext,
        *,
        extra: dict[str, Any] | None = None,
        include_metadata: bool = True,
        include_ttft: bool = True,
    ) -> dict[str, Any]:
        """Build log_metadata dict with common fields and optional extras.

        Args:
            context: Event context with all data.
            extra: Optional extra fields to include. These take precedence over
                   context.metadata when both contain the same key.
            include_metadata: If True, include context.metadata in the result.
            include_ttft: If True, include ttft_ms when available.

        Returns:
            Dict with log metadata fields.
        """
        log_metadata: dict[str, Any] = {
            "is_api_endpoint": context.is_api_endpoint,
            "request_type": context.request_type_display,
        }
        # extra takes precedence over context.metadata so that callers can
        # override or add fields without mutating the original context.
        if include_metadata:
            log_metadata.update(context.metadata)
        if extra:
            log_metadata.update(extra)
        if include_ttft and context.ttft_ms is not None:
            log_metadata["ttft_ms"] = context.ttft_ms
        log_metadata = self._redact_verbose_routing(log_metadata)
        log_metadata = self._add_fallback_metadata(log_metadata, context)
        log_metadata = self._add_retry_metadata(log_metadata, context)
        routing_meta = self._extract_routing_metadata(context)
        log_metadata.update(routing_meta)
        if context.provider_model_name:
            log_metadata["provider_model_name"] = context.provider_model_name
        return log_metadata

    def _build_log_create(
        self,
        request: InternalRequest,  # noqa: ARG002
        response: InternalResponse,  # noqa: ARG002
        context: EventContext,
    ) -> RequestLogCreate:
        """Build RequestLogCreate from EventContext."""
        base = self._build_log_base(context)

        request_headers, request_body = self._mask_request_data(context)
        response_headers, response_body = self._mask_response_data(context)

        if not context.should_capture_full_body:
            response_headers = {}
            response_body = {"_sampled_out": True}
        elif not context.should_log_input_output:
            request_body = {"_sampled_out": True}
            response_body = {"_sampled_out": True}

        log_metadata = self._build_log_metadata(context, extra={"streaming": False})

        return RequestLogCreate(
            **base,
            request_headers=request_headers,
            request_body=request_body,
            response_headers=response_headers,
            response_body=response_body,
            log_metadata=log_metadata,
        )

    def _build_error_log_create(
        self,
        request: InternalRequest,  # noqa: ARG002
        error: Exception,
        context: EventContext,
    ) -> RequestLogCreate:
        """Build RequestLogCreate for error case."""
        base = self._build_log_base(context, default_status_code=500)
        request_headers, request_body = self._mask_request_data(context)

        log_metadata = self._build_log_metadata(
            context, extra={"error_details": context.error_details}
        )

        return RequestLogCreate(
            **base,
            request_headers=request_headers,
            request_body=request_body,
            response_headers={},
            response_body={"error": True, "message": "Request failed"},
            log_metadata=log_metadata,
        )

    def _build_streaming_log_create(
        self,
        request: InternalRequest,  # noqa: ARG002
        context: EventContext,
    ) -> RequestLogCreate:
        """Build RequestLogCreate for streaming case."""
        base = self._build_log_base(context)

        request_headers, request_body = self._mask_request_data(context)
        response_headers, response_body = self._mask_response_data(context)

        if not context.should_capture_full_body:
            response_body = {"streaming": True, "truncated": context.streaming_truncated}

        log_metadata = self._build_log_metadata(
            context,
            extra={
                "streaming": True,
                "response_body_truncated": context.streaming_truncated,
            },
        )

        return RequestLogCreate(
            **base,
            request_headers=request_headers,
            request_body=request_body,
            response_headers=response_headers,
            response_body=response_body,
            log_metadata=log_metadata,
        )

    def _build_usage_record(self, context: EventContext) -> UsageRecordCreate:
        """Build UsageRecordCreate from EventContext.

        UsageRecord is always created independently of sampling.

        Args:
            context: Event context with token/cost data

        Returns:
            UsageRecordCreate ready for database insertion
        """
        return UsageRecordCreate(
            timestamp=context.start_timestamp,
            request_id=context.request_id,
            model=context.internal_model or context.model,
            provider=context.provider,
            user_id=context.auth_user_id,
            prompt_tokens=context.prompt_tokens,
            completion_tokens=context.completion_tokens,
            total_tokens=context.total_tokens,
            cost_usd=context.cost_usd,
            cache_creation_input_tokens=context.cache_creation_input_tokens,
            cache_read_input_tokens=context.cache_read_input_tokens,
            cached_prompt_tokens=context.cached_prompt_tokens,
            cache_savings_usd=context.cache_savings_usd,
            audio_input_tokens=context.audio_input_tokens,
            audio_output_tokens=context.audio_output_tokens,
            response_time_ms=safe_int(context.latency_ms),
            status_code=context.response_status_code,
            user_identity=context.user_id,
            api_key_name=context.api_key_name,
            is_streaming=context.is_streaming,
            ttft_ms=safe_int(context.ttft_ms) if context.ttft_ms is not None else None,
            log_type=context.log_type,
        )
