"""Gemini batch dialect: a long-running operation over inline requests.

Gemini's batch endpoint is a Google long-running operation, not OpenAI's file
job, and it is the only one of the three that needs real translation: each batch
entry is a ``GenerateContentRequest``, and each result is a
``GenerateContentResponse``. The proxy therefore takes the translating side of the
seam — the service parses the client's chat line into an ``InternalRequest``, this
adapter rebuilds it through the Gemini serializer, and results are parsed back to
``InternalResponse`` for the OpenAI protocol formatter.

Wire notes that matter:

- Create: ``POST {base}/models/{model}:batchGenerateContent`` with
  ``{batch: {displayName, inputConfig: {requests: {requests: [{request,
  metadata}]}}}}``. The inline payload must stay under 20 MB, so the built
  request is measured before it is sent.
- The response is an ``Operation`` named ``batches/{id}``; state lives in
  ``metadata.state`` (``BATCH_STATE_*``) and progress in ``metadata.batchStats``
  (counts are JSON strings).
- Finished responses arrive at ``response.inlinedResponses.inlinedResponses[]``
  (nested envelope), in input order. The per-request ``metadata`` echo is not
  reliable, so the proxy maps results back to ``custom_id`` by position.
- Cancel: ``POST {base}/batches/{id}:cancel``.
"""

from datetime import datetime
from typing import Any

import orjson

from llm_proxy.batch.types import (
    BATCH_TIMESTAMP_FIELDS,
    DEFAULT_COMPLETION_WINDOW,
    TERMINAL_BATCH_STATUSES,
    BatchRequestItem,
    BatchResultItem,
    BatchSnapshot,
    BatchSubmission,
)
from llm_proxy.core.exceptions import ProviderError, ValidationError
from llm_proxy.core.utils import optional_int
from llm_proxy.observability.logger import get_logger
from llm_proxy.providers.capabilities.batch import BatchCapabilityMixin
from llm_proxy.providers.capabilities.host import BatchSelf

logger = get_logger(__name__)

#: Gemini rejects an inline batch whose *total request* exceeds 20 MB; the
#: proxy measures the built payload (which adds the batch envelope) and refuses
#: before spending an upstream round trip.
GEMINI_INLINE_LIMIT_BYTES: int = 20_000_000

#: ``BatchState`` → OpenAI batch status. ``PENDING`` is "the service is
#: preparing to run the batch", which is what OpenAI calls ``validating``.
_STATE_MAP: frozendict[str, str] = frozendict(
    {
        "BATCH_STATE_UNSPECIFIED": "validating",
        "BATCH_STATE_PENDING": "validating",
        "BATCH_STATE_RUNNING": "in_progress",
        "BATCH_STATE_SUCCEEDED": "completed",
        "BATCH_STATE_FAILED": "failed",
        "BATCH_STATE_CANCELLED": "cancelled",
        "BATCH_STATE_EXPIRED": "expired",
    }
)

#: Where a terminal state's timestamp lands. Gemini reports one ``endTime``.
_END_TIME_FIELD: frozendict[str, str] = frozendict(
    {
        "completed": "completed_at",
        "failed": "failed_at",
        "cancelled": "cancelled_at",
        "expired": "expired_at",
    }
)


def _counts(raw: Any) -> dict[str, int]:
    """Read Gemini's string-valued ``batchStats`` into integer counts."""
    if not isinstance(raw, dict):
        return {"total": 0, "completed": 0, "failed": 0}
    return {
        "total": optional_int(raw.get("requestCount")) or 0,
        "completed": optional_int(raw.get("successfulRequestCount")) or 0,
        "failed": optional_int(raw.get("failedRequestCount")) or 0,
    }


def _rfc3339_to_unix(value: Any) -> int | None:
    """Convert a Google ``createTime``/``endTime`` timestamp to unix seconds."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return None


def validate_inline_size(size: int) -> None:
    """Refuse an inline payload that is not strictly below Gemini's 20 MB cap."""
    if size >= GEMINI_INLINE_LIMIT_BYTES:
        raise ValidationError(
            message=(
                f"Batch payload is {size} bytes, at or above Gemini's "
                f"{GEMINI_INLINE_LIMIT_BYTES}-byte inline limit; split the batch."
            ),
            code="batch_too_large",
            status_code=413,
        )


def _batch_metadata(operation: dict[str, Any]) -> dict[str, Any]:
    metadata = operation.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _inlined_response_entries(operation: dict[str, Any]) -> list[dict[str, Any]]:
    """Find the inline response entries wherever this operation put them.

    A finished operation carries them at
    ``response.inlinedResponses.inlinedResponses``; some responses also surface
    the output block under ``metadata.output``. Both are tried so a shape change
    on either side degrades to "no results" instead of an exception.
    """
    candidates: list[Any] = []
    response = operation.get("response")
    if isinstance(response, dict):
        candidates.append(response)
    metadata = _batch_metadata(operation)
    candidates.append(metadata.get("output"))
    candidates.append(metadata)
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        block = candidate.get("inlinedResponses")
        # Both `{inlinedResponses: [...]}` and the nested
        # `{inlinedResponses: {inlinedResponses: [...]}}` envelope appear.
        if isinstance(block, dict):
            block = block.get("inlinedResponses")
        if isinstance(block, list):
            return [entry for entry in block if isinstance(entry, dict)]
    return []


def results_from_operation(operation: dict[str, Any]) -> list[BatchResultItem] | None:
    """Normalize a finished operation's inline responses into result items."""
    entries = _inlined_response_entries(operation)
    if not entries:
        return None
    items: list[BatchResultItem] = []
    for index, entry in enumerate(entries):
        metadata = entry.get("metadata")
        custom_id = metadata.get("custom_id") if isinstance(metadata, dict) else None
        body = entry.get("response")
        error = entry.get("error")
        if isinstance(body, dict):
            items.append(
                BatchResultItem(
                    index=index,
                    custom_id=custom_id if isinstance(custom_id, str) else None,
                    body=body,
                )
            )
            continue
        if isinstance(error, dict):
            items.append(
                BatchResultItem(
                    index=index,
                    custom_id=custom_id if isinstance(custom_id, str) else None,
                    error={
                        "code": error.get("status") or error.get("code") or "batch_request_failed",
                        "message": error.get("message") or "Batch request failed",
                    },
                )
            )
    return items or None


def snapshot_from_operation(operation: dict[str, Any]) -> BatchSnapshot:
    """Normalize a Gemini operation into a :class:`BatchSnapshot`."""
    metadata = _batch_metadata(operation)
    state = str(metadata.get("state") or "BATCH_STATE_UNSPECIFIED")
    status = _STATE_MAP.get(state, "validating")
    stats = metadata.get("batchStats")
    counts = _counts(stats)
    timestamps: dict[str, int | None] = {field: None for field in BATCH_TIMESTAMP_FIELDS}
    start_time = _rfc3339_to_unix(metadata.get("startTime"))
    if start_time is not None:
        timestamps["in_progress_at"] = start_time
    end_field = _END_TIME_FIELD.get(status)
    if end_field is not None:
        timestamps[end_field] = _rfc3339_to_unix(metadata.get("endTime"))

    errors: list[dict[str, Any]] | None = None
    operation_error = operation.get("error")
    if isinstance(operation_error, dict):
        errors = [
            {
                "code": operation_error.get("status") or operation_error.get("code") or "failed",
                "message": operation_error.get("message") or "Batch failed",
            }
        ]
    return BatchSnapshot(
        status=status,
        request_counts=counts,
        timestamps=timestamps,
        errors=errors,
        results=results_from_operation(operation) if status in TERMINAL_BATCH_STATUSES else None,
    )


def _batch_id(operation_name: str) -> str:
    """``batches/123`` → ``123`` (the id used in the REST path)."""
    return operation_name.removeprefix("batches/")


class GeminiBatchMixin(BatchCapabilityMixin):
    """Batch support for the Gemini adapter (``batchGenerateContent``)."""

    SUPPORTS_BATCH = True
    BATCH_FORWARDS_CHAT_BODY = False
    BATCH_CREATE_ENDPOINT = "/models/{model}:batchGenerateContent"

    async def submit_batch(
        self: BatchSelf,
        *,
        model: str,
        requests: list[BatchRequestItem],
        display_name: str,
        completion_window: str = DEFAULT_COMPLETION_WINDOW,
    ) -> BatchSubmission:
        if getattr(self, "_is_interactions_variant", lambda: False)():
            raise ValidationError(
                message=(
                    "Batch mode is not available for the Gemini Interactions API variant; "
                    "set the provider's api_variant to 'generate_content'."
                ),
                code="batch_unsupported",
                status_code=400,
            )

        model_name = model.removeprefix("models/")
        payload = {
            "batch": {
                "displayName": display_name,
                "inputConfig": {
                    "requests": {
                        "requests": [
                            {"request": item.body, "metadata": {"custom_id": item.custom_id}}
                            for item in requests
                        ]
                    }
                },
            }
        }
        validate_inline_size(len(orjson.dumps(payload)))

        raw = await self._batch_json(
            "POST",
            self._batch_create_url(model_name),
            headers=self._build_headers(),
            json_body=payload,
        )
        upstream_id = raw.get("name")
        if not isinstance(upstream_id, str) or not upstream_id:
            raise ProviderError(
                message="Gemini batch create returned no operation name",
                error_type="api_error",
                status_code=502,
                provider_name=self.provider_name,
            )
        return BatchSubmission(upstream_id=upstream_id, snapshot=snapshot_from_operation(raw))

    async def fetch_batch(self: BatchSelf, upstream_id: str) -> BatchSnapshot:
        raw = await self._batch_json(
            "GET",
            self._batches_url(f"/{_batch_id(upstream_id)}"),
            headers=self._build_headers(),
        )
        return snapshot_from_operation(raw)

    async def cancel_batch(self: BatchSelf, upstream_id: str) -> BatchSnapshot | None:
        batch_id = _batch_id(upstream_id)
        await self._batch_request(
            "POST",
            self._batches_url(f"/{batch_id}:cancel"),
            headers=self._build_headers(),
        )
        snapshot = await self.fetch_batch(upstream_id)
        # Google's cancel is asynchronous: the operation leaves PENDING/RUNNING
        # for CANCELLED at its own pace, so report the in-between state.
        if snapshot.status in {"validating", "in_progress"}:
            timestamps = dict(snapshot.timestamps)
            timestamps["cancelling_at"] = int(datetime.now().timestamp())
            return BatchSnapshot(
                status="cancelling",
                request_counts=snapshot.request_counts,
                timestamps=timestamps,
                errors=snapshot.errors,
                results=None,
            )
        return snapshot


__all__ = [
    "GeminiBatchMixin",
    "results_from_operation",
    "snapshot_from_operation",
    "validate_inline_size",
]
