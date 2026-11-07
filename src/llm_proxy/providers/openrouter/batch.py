"""OpenRouter batch dialect: inline requests, inline results.

OpenRouter's batch API is the odd one out in a way that simplifies the seam: it
takes the requests inline (no file), binds the batch to one ``model``, and hands
the results back **inside the status object** — one entry per request, matched on
``custom_id``, and not in input order. There is no results file and no cancel
endpoint (a finished batch can be deleted, which the proxy does not expose).

Wire notes that matter:

- ``POST /v1/batches`` requires ``endpoint``, ``model`` and a non-empty
  ``requests`` array of ``{custom_id, body}``; a body that names a model must
  match the batch model, which is why the proxy enforces one model per batch.
- It answers ``202 Accepted`` with the batch object, status ``validating``.
- Final statuses are ``completed``, ``failed``, ``expired`` and ``cancelled``;
  ``results`` is populated only once the status is ``completed``.
- ``request_counts`` and ``results`` share OpenAI's shape, so both pass through
  with their line bodies untouched.
"""

from typing import Any

from llm_proxy.batch.types import (
    BATCH_TIMESTAMP_FIELDS,
    DEFAULT_COMPLETION_WINDOW,
    SUPPORTED_BATCH_ENDPOINT,
    BatchRequestItem,
    BatchResultItem,
    BatchSnapshot,
    BatchSubmission,
)
from llm_proxy.core.exceptions import ValidationError
from llm_proxy.core.utils import optional_int
from llm_proxy.observability.logger import get_logger
from llm_proxy.providers.capabilities.batch import BatchCapabilityMixin, request_counts
from llm_proxy.providers.capabilities.host import BatchSelf

logger = get_logger(__name__)

BATCHES_ENDPOINT = "/batches"

#: ``finalized_at`` is the only terminal timestamp OpenRouter reports; it is
#: mapped onto the field that matches the terminal status it arrived with.
_FINALIZED_AT_FIELD: frozendict[str, str] = frozendict(
    {
        "completed": "completed_at",
        "failed": "failed_at",
        "expired": "expired_at",
        "cancelled": "cancelled_at",
    }
)


def snapshot_from_batch(raw: dict[str, Any]) -> BatchSnapshot:
    """Normalize an OpenRouter batch object into a :class:`BatchSnapshot`."""
    status = str(raw.get("status") or "validating")
    timestamps: dict[str, int | None] = {field: None for field in BATCH_TIMESTAMP_FIELDS}
    finalized_at = optional_int(raw.get("finalized_at"))
    finalized_field = _FINALIZED_AT_FIELD.get(status)
    if finalized_at is not None and finalized_field is not None:
        timestamps[finalized_field] = finalized_at
    counts = request_counts(raw.get("request_counts"))
    raw_error = raw.get("error")
    errors: list[dict[str, Any]] | None = None
    if isinstance(raw_error, dict):
        message = raw_error.get("message")
        if isinstance(message, str) and message:
            errors = [{"code": "batch_failed", "message": message}]
        elif raw_error:
            errors = [raw_error]
    return BatchSnapshot(
        status=status,
        request_counts=counts,
        timestamps=timestamps,
        errors=errors,
        results=_results_from(raw.get("results")),
        expires_at=optional_int(raw.get("expires_at")),
        usage=raw.get("usage") if isinstance(raw.get("usage"), dict) else None,
    )


def _results_from(raw: Any) -> list[BatchResultItem] | None:
    """Convert OpenRouter's inline ``results`` into normalized result items."""
    if not isinstance(raw, list) or not raw:
        return None
    items: list[BatchResultItem] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            continue
        response = entry.get("response")
        error = entry.get("error")
        if isinstance(response, dict):
            body = response.get("body")
            items.append(
                BatchResultItem(
                    index=index,
                    custom_id=entry.get("custom_id"),
                    body=body if isinstance(body, dict) else None,
                    status_code=optional_int(response.get("status_code")) or 200,
                    request_id=response.get("request_id"),
                )
            )
            continue
        items.append(
            BatchResultItem(
                index=index,
                custom_id=entry.get("custom_id"),
                error=error if isinstance(error, dict) else {"message": "Batch request failed"},
            )
        )
    return items or None


class OpenRouterBatchMixin(BatchCapabilityMixin):
    """Batch support for the OpenRouter adapter (inline Batches API)."""

    SUPPORTS_BATCH = True
    BATCH_FORWARDS_CHAT_BODY = True
    BATCHES_ENDPOINT = BATCHES_ENDPOINT

    async def submit_batch(
        self: BatchSelf,
        *,
        model: str,
        requests: list[BatchRequestItem],
        display_name: str,
        completion_window: str = DEFAULT_COMPLETION_WINDOW,
    ) -> BatchSubmission:
        raw = await self._batch_json(
            "POST",
            self._batches_url(),
            headers=self._build_headers(),
            json_body={
                "endpoint": SUPPORTED_BATCH_ENDPOINT,
                "model": model,
                "completion_window": completion_window,
                "requests": [{"custom_id": item.custom_id, "body": item.body} for item in requests],
            },
        )
        upstream_id = raw.get("id")
        if not isinstance(upstream_id, str) or not upstream_id:
            raise ValidationError(
                message="OpenRouter batch create returned no batch id",
                code="upstream_batch_error",
                status_code=502,
            )
        return BatchSubmission(upstream_id=upstream_id, snapshot=snapshot_from_batch(raw))

    async def fetch_batch(self: BatchSelf, upstream_id: str) -> BatchSnapshot:
        raw = await self._batch_json(
            "GET", self._batches_url(f"/{upstream_id}"), headers=self._build_headers()
        )
        return snapshot_from_batch(raw)

    async def cancel_batch(self: BatchSelf, upstream_id: str) -> BatchSnapshot | None:
        # OpenRouter documents "There is no cancel endpoint yet"; returning None
        # lets the service refuse with one message instead of a 405 from here.
        return None


__all__ = ["OpenRouterBatchMixin", "snapshot_from_batch"]
