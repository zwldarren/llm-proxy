"""Request-log sink: the observation projected into a ``request_logs`` row.

This is the database-facing sink. It owns what belongs to *this* store and to no
other: audit classification (``event_type`` …), the body markers that explain why
a payload is absent, and the routing of the row into the shared assembler.
Everything it needs beyond that — the facts, the masking, the size cap — arrives
already decided on the observation.

The assembly goes through :func:`llm_proxy.observability.store.assemble_row` and
its metadata builder, so every situation — the endpoint lifecycle and the
middleware/tool verbs alike — produces one row shape. The endpoint lifecycle *is*
this sink now. See ADR-0044.

Size: the store applies the deployment's body cap with the *live* config (see
``service._request_log_from_create``), which is why this sink registers a policy
with ``max_bytes=0`` — a static policy cannot follow a settings change, and
capping twice would only move the marker earlier. A sink that is not backed by
that store supplies its own cap through its ``SinkPolicy``.
"""

from collections.abc import Callable
from typing import Any

from llm_proxy.billing.tokens import ProxyBillableDimensions, token_usage_from
from llm_proxy.config.types.logging_config import LoggingConfig
from llm_proxy.core.utils import safe_int
from llm_proxy.observability.classification import (
    determine_action_category,
    determine_event_type,
    determine_outcome,
    determine_resource_type,
)
from llm_proxy.observability.observation.model import (
    ContentFacts,
    ObservationKind,
    RequestObservation,
)
from llm_proxy.observability.redaction import body_marker, scrub_log_metadata
from llm_proxy.observability.service import RequestLogCreate
from llm_proxy.observability.store import (
    assemble_row,
    build_log_metadata,
    dispatch_row,
    logging_config,
    stream_body,
    token_column,
)
from llm_proxy.observability.types import LogType


def _response_body(content: ContentFacts) -> Any:
    """The body to store for a finished stream, mirroring the lifecycle's rule.

    ``content.response_body`` is deliberately not consulted: for a stream the
    buffered raw frames or the reassembled body are the row's body (the pre-seam
    lifecycle overwrote ``response_body`` with exactly that), so a stale value
    there is not the stream's body.
    """
    return stream_body(content)


class DatabaseLogSink:
    """Write one ``request_logs`` row per request observation."""

    name = "database_log"
    kinds = frozenset({ObservationKind.REQUEST})

    def __init__(
        self,
        *,
        config_provider: Callable[[], LoggingConfig] | None = None,
        dispatch: Callable[[RequestLogCreate], bool] | None = None,
    ) -> None:
        self._config_provider = config_provider or logging_config
        self._dispatch = dispatch or (lambda row: dispatch_row(row, None))

    def build_row(self, observation: RequestObservation) -> RequestLogCreate:
        """Project the observation into the store's row shape."""
        config = self._config_provider()
        content = observation.content
        mode = observation.policy.mode
        streaming = bool(content.streaming) if content is not None else False
        failed = observation.outcome.failed

        # Bodies: the sink explains *why* a payload is absent. A withheld
        # projection and an unsampled capture are different facts on the row.
        marker = body_marker(bodies_enabled=mode != "none")
        bodies_visible = content is not None and content.captured and not content.withheld
        request_headers: dict[str, Any] = dict(content.request_headers) if content else {}
        response_headers: dict[str, Any] = dict(content.response_headers) if content else {}
        request_body: Any = marker
        response_body: Any = dict(marker)

        if bodies_visible:
            assert content is not None  # narrowed by ``bodies_visible``
            request_body = content.request_body
            if failed:
                # The upstream error body is content; only the outcome is stored.
                response_headers = {}
                response_body = {"error": True, "message": "Request failed"}
            elif streaming:
                response_body = _response_body(content)
            else:
                response_body = content.response_body
        elif failed:
            # An error row keeps its constant outcome body even when payloads are
            # off: it is a classification fact, not upstream content.
            response_headers = {}
            response_body = {"error": True, "message": "Request failed"}

        if failed:
            # The error row explains the failure and deliberately stores no
            # upstream response body (only the outcome), matching the lifecycle.
            extra: dict[str, Any] = {"error_details": dict(observation.outcome.error_details)}
        else:
            extra = {"streaming": streaming}
            if streaming:
                extra["response_body_truncated"] = bool(
                    content is not None and content.raw_stream_truncated
                )

        log_metadata = build_log_metadata(
            metadata=observation.attributes,
            is_api_endpoint=observation.core.is_api_endpoint,
            request_type=observation.core.request_type or "chat",
            config=config,
            ttft_ms=observation.timing.ttft_ms,
            fallback_attempts=observation.routing.fallbacks,
            retry_attempts=observation.routing.retries,
            provider_model_name=observation.routing.provider_model_name,
            extra=extra,
        )
        if failed and mode == "none":
            log_metadata = scrub_log_metadata(log_metadata)

        usage = observation.usage.usage
        tokens = token_usage_from(
            usage,
            ProxyBillableDimensions(
                images_generated=observation.usage.images_generated,
                tts_characters=observation.usage.tts_characters,
                web_search_requests=observation.usage.web_search_requests,
            ),
        )
        log_type = LogType(observation.core.log_type)

        return assemble_row(
            log_type=log_type,
            request_id=observation.core.request_id,
            trace_id=observation.core.trace_id,
            timestamp=observation.core.started_at,
            endpoint=observation.core.endpoint,
            method=observation.core.method,
            status_code=observation.outcome.status_code or (500 if failed else 200),
            response_time_ms=safe_int(observation.timing.latency_ms),
            event_type=determine_event_type(observation.core.endpoint),
            action_category=determine_action_category(observation.core.method),
            resource_type=determine_resource_type(observation.core.endpoint),
            # Derived at capture from the raw body; a withheld projection must
            # not change the row's classification.
            resource_id=observation.core.resource_id,
            outcome=determine_outcome(observation.outcome.status_code),
            user_identity=observation.identity.user,
            user_id=observation.identity.user_id,
            api_key_name=observation.identity.api_key_name,
            auth_method=observation.identity.auth_method,
            client_ip=observation.identity.client_ip,
            user_agent=observation.identity.user_agent,
            session_id=observation.identity.session_id,
            model=observation.routing.model,
            provider=observation.routing.provider,
            prompt_tokens=token_column(tokens.prompt_tokens, usage),
            completion_tokens=token_column(tokens.completion_tokens, usage),
            total_tokens=token_column(tokens.total_tokens, usage),
            cache_creation_input_tokens=token_column(tokens.cache_creation_input_tokens, usage),
            cache_read_input_tokens=token_column(tokens.cache_read_input_tokens, usage),
            cached_prompt_tokens=token_column(tokens.cached_prompt_tokens, usage),
            cost_usd=observation.usage.cost_usd,
            cache_savings_usd=observation.usage.cache_savings_usd,
            ttft_ms=(
                safe_int(observation.timing.ttft_ms)
                if observation.timing.ttft_ms is not None
                else None
            ),
            fallback_count=len(observation.routing.fallbacks),
            error_message=observation.outcome.error_message,
            error_stack_trace=observation.outcome.error_stack_trace,
            request_headers=request_headers,
            request_body=request_body,
            response_headers=response_headers,
            response_body=response_body,
            log_metadata=log_metadata,
        )

    async def emit(self, observation: RequestObservation) -> None:
        self._dispatch(self.build_row(observation))

    async def flush(self) -> None:
        """Nothing to flush: the log store's background writer owns buffering."""

    async def close(self) -> None:
        """Nothing to close: the writer is process-lifetime (see the service)."""


__all__ = ["DatabaseLogSink"]
