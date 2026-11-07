"""Batch spend: the one ``usage_records`` row a finished batch writes.

A batch has no request row of its own — it outlives the request that created it,
and its results are read on a later poll by a different request — so its spend
cannot ride the ordinary per-request usage write. Instead the poll that
materializes the results prices every line the upstream reported and enqueues a
single aggregate row (``log_type='batch'``, ADR-0058).

Why one row per batch rather than one per line:

- the client made one call, and one row keeps the accounting honest about that
  (request *counts* stay endpoint-only, as for the routing judge's own rows);
- a batch is one model by construction, so the aggregate still prices exactly
  (pricing is linear in tokens) and per-model breakdowns stay correct.

The row is stamped with the batch's **creation** instant, not the poll's: that is
the instant the work was submitted, which is what peak/off-peak windows price by,
and it keeps the spend in the period the client asked for. Cost is summed over
the lines, each priced through the same :func:`calculate_cost` the synchronous
path uses, so a per-line context tier is applied to the line that reached it
instead of to the batch's total.

Lines that reported no usage contribute no tokens (a failure line, a provider
that omits the counter). A batch whose lines all stayed silent records no row:
there is nothing to bill and a $0 row would misreport the spend as known.
"""

from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from typing import Any

from llm_proxy.batch.types import BatchOwner
from llm_proxy.billing.cost import calculate_cost
from llm_proxy.billing.tokens import ProxyBillableDimensions
from llm_proxy.database.tables import BatchJobRecord
from llm_proxy.models.types import Usage
from llm_proxy.observability.logger import get_logger
from llm_proxy.observability.service import UsageRecordCreate
from llm_proxy.observability.store import build_usage_record, usage_service
from llm_proxy.observability.types import LogType
from llm_proxy.serialization.usage import UsageDialect, add_usage, parse_usage

logger = get_logger(__name__)

#: Every materialized result line is in OpenAI's chat-completion wire shape,
#: whichever upstream produced it (the service formats translated results through
#: the OpenAI protocol serializer), so one usage dialect reads them all.
RESULT_USAGE_DIALECT: UsageDialect = "chat"


def usage_owner_names(owner: str) -> tuple[str | None, str | None]:
    """Split a storage scope into a usage row's ``(api_key_name, user_identity)``.

    ``owner_scope`` builds the scope as an API key's name or ``user:<name>``;
    reading it back is what lets the aggregate row be attributed to the same key
    or user the synchronous rows carry.
    """
    parsed = BatchOwner.from_scope(owner)
    return parsed.api_key_name, parsed.user


async def price_batch_results(
    job: BatchJobRecord,
    bodies: Iterable[Mapping[str, Any]],
    *,
    config_manager: Any,
) -> UsageRecordCreate | None:
    """Price a batch's result bodies into its single usage row, or ``None``.

    ``bodies`` are the materialized success bodies (already in OpenAI wire
    shape). Pricing every line separately keeps the context-tier overlay of a
    long prompt on that prompt alone; the tokens are then summed for the row.
    """
    total = Usage()
    cost_usd = 0.0
    priced = False
    priced_lines = 0
    at = datetime.fromtimestamp(job.created_at, tz=UTC)

    for body in bodies:
        usage = parse_usage(RESULT_USAGE_DIALECT, body)
        if usage is None:
            continue
        priced_lines += 1
        add_usage(total, usage)
        breakdown = await calculate_cost(
            usage,
            job.model,
            config_manager,
            provider_name=job.provider,
            at=at,
        )
        if breakdown.cost_usd is not None:
            cost_usd += breakdown.cost_usd
            priced = True

    if not priced_lines:
        return None

    api_key_name, user_identity = usage_owner_names(job.owner)
    return build_usage_record(
        timestamp=float(job.created_at),
        request_id=job.id,
        model=job.model,
        provider=job.provider,
        user_id=job.user_id,
        usage=total,
        dimensions=ProxyBillableDimensions(),
        cost_usd=cost_usd if priced else None,
        # Cache savings are the request path's own comparison (what the cached
        # tokens would have cost). A batch line reports the cached count, not a
        # saving, so the column stays null rather than carrying a guess.
        cache_savings_usd=None,
        # Latency belongs to the request path: a batch's "response time" is its
        # completion window, which would poison every latency average it landed in.
        response_time_ms=None,
        status_code=200,
        user_identity=user_identity,
        api_key_name=api_key_name,
        is_streaming=False,
        ttft_ms=None,
        log_type=LogType.BATCH.value,
    )


async def record_batch_usage(
    job: BatchJobRecord,
    bodies: Iterable[Mapping[str, Any]],
    *,
    config_manager: Any,
    dispatch: Callable[[UsageRecordCreate], None] | None = None,
) -> UsageRecordCreate | None:
    """Price the batch and hand its usage row to the store's background writer.

    ``dispatch`` is the test seam the usage sink uses too: production leaves it
    unset and the row goes to the process-lifetime usage writer.
    """
    record = await price_batch_results(job, bodies, config_manager=config_manager)
    if record is None:
        return None
    (dispatch or usage_service().create_usage_background)(record)
    logger.info(
        f"Batch {job.id}: recorded spend of {record.cost_usd} USD over "
        f"{record.total_tokens} tokens for {job.model}"
    )
    return record


__all__ = ["RESULT_USAGE_DIALECT", "price_batch_results", "record_batch_usage", "usage_owner_names"]
