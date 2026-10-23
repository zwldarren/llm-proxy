"""Usage sink: the observation projected into a ``usage_records`` row.

Usage is accounting, not a log: it is written for every billable request,
independently of body sampling and of whether the request log was kept. That is
exactly what a sink policy says — ``SinkPolicy(record=True, content="none",
sample_record=1.0)`` — so the rule lives in one declaration instead of a guard in
the writer.

See ADR-0044.
"""

from collections.abc import Callable

from llm_proxy.billing.tokens import ProxyBillableDimensions
from llm_proxy.core.utils import safe_int
from llm_proxy.observability.observation.model import ObservationKind, RequestObservation
from llm_proxy.observability.service import UsageRecordCreate
from llm_proxy.observability.store import build_usage_record, usage_service


class DatabaseUsageSink:
    """Write one ``usage_records`` row per request observation."""

    name = "database_usage"
    kinds = frozenset({ObservationKind.REQUEST})

    def __init__(
        self,
        *,
        dispatch: Callable[[UsageRecordCreate], None] | None = None,
    ) -> None:
        self._dispatch = dispatch or (
            lambda record: usage_service().create_usage_background(record)
        )

    def build_record(self, observation: RequestObservation) -> UsageRecordCreate:
        """Project the observation into the store's usage-row shape."""
        content = observation.content
        return build_usage_record(
            timestamp=observation.core.started_at,
            request_id=observation.core.request_id,
            model=observation.routing.internal_model or observation.routing.model,
            provider=observation.routing.provider,
            user_id=observation.identity.user_id,
            usage=observation.usage.usage,
            dimensions=ProxyBillableDimensions(
                images_generated=observation.usage.images_generated,
                tts_characters=observation.usage.tts_characters,
                web_search_requests=observation.usage.web_search_requests,
            ),
            cost_usd=observation.usage.cost_usd,
            cache_savings_usd=observation.usage.cache_savings_usd,
            response_time_ms=safe_int(observation.timing.latency_ms),
            status_code=observation.outcome.status_code,
            user_identity=observation.identity.user,
            api_key_name=observation.identity.api_key_name,
            is_streaming=bool(content.streaming) if content is not None else False,
            ttft_ms=(
                safe_int(observation.timing.ttft_ms)
                if observation.timing.ttft_ms is not None
                else None
            ),
            log_type=observation.core.log_type,
        )

    async def emit(self, observation: RequestObservation) -> None:
        self._dispatch(self.build_record(observation))

    async def flush(self) -> None:
        """Nothing to flush: the usage store's background writer owns buffering."""

    async def close(self) -> None:
        """Nothing to close: the writer is process-lifetime (see the service)."""


__all__ = ["DatabaseUsageSink"]
