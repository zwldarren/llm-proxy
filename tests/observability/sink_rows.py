"""Build a stored row/record through the live observation path, for tests.

The endpoint lifecycle's row builders moved behind the observation/sink seam
(ADR-0044). Tests that used to call them directly go through this helper so they
exercise the production path: capture → policy projection → sink row.

``legacy_row_reference`` is the frozen pre-seam copy, used only by the parity
tests; everything else should use these.
"""

from typing import Any

from llm_proxy.config.types.logging_config import LoggingConfig
from llm_proxy.observability import store
from llm_proxy.observability.event_context import EventContext
from llm_proxy.observability.observation.collector import collect
from llm_proxy.observability.observation.policy import FACTS_ONLY, SinkPolicy
from llm_proxy.observability.observation.projection import apply_sink_policy
from llm_proxy.observability.sinks.database_log import DatabaseLogSink
from llm_proxy.observability.sinks.database_usage import DatabaseUsageSink


def sink_row(
    context: EventContext,
    config: LoggingConfig,
    *,
    error: Exception | None = None,
) -> Any:
    """The request-log row the live path stores for this captured request.

    The capture must already hold the request's *final* state, as the live path
    leaves it: ``record_request_end``/``record_request_error``/``record_stream_end``
    fold the end state into the capture through ``capture_request_end`` (usage,
    status default, error message/stack/details) before the bus emits. Passing
    ``error=`` here only marks the outcome as failed — it does not apply the
    error's own status or format its stack, so a test that pins those fields must
    prepare the context the way the verbs do.
    """
    projection = apply_sink_policy(collect(context, error=error), SinkPolicy(content="full"))
    assert projection is not None
    sink = DatabaseLogSink(config_provider=lambda: config, dispatch=lambda row: True)
    return sink.build_row(projection)


def sink_usage_record(context: EventContext) -> Any:
    """The usage record the live path stores for this captured request.

    Accounting is never sampled and never carries content, so this is the whole
    live path: capture → ``FACTS_ONLY`` projection → the sink's record.
    """
    projection = apply_sink_policy(collect(context), FACTS_ONLY)
    assert projection is not None
    return DatabaseUsageSink(dispatch=lambda record: None).build_record(projection)


def sink_stream_body(context: EventContext) -> Any:
    """The stream-body *rule* applied to this capture's raw snapshot.

    This is not the sink's final row body in every case: the sink derives it from
    the projected content and only when the content was captured and not withheld
    (a body-less or sampled-out stream stores a marker instead). Use it to pin the
    rule, not to predict the row.
    """
    content = collect(context).content
    assert content is not None
    return store.stream_body(content)


def context_log_metadata(
    context: EventContext,
    config: LoggingConfig,
    *,
    extra: dict[str, Any] | None = None,
) -> Any:
    """The metadata the log sink derives from this context (the live primitive)."""
    return store.build_log_metadata(
        metadata=context.metadata,
        is_api_endpoint=context.is_api_endpoint,
        request_type=context.request_type_display,
        config=config,
        ttft_ms=context.ttft_ms,
        fallback_attempts=context.fallback_attempts,
        retry_attempts=context.retry_attempts,
        provider_model_name=context.provider_model_name,
        extra=extra,
    )


__all__ = ["context_log_metadata", "sink_row", "sink_stream_body", "sink_usage_record"]
