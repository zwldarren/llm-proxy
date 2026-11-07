"""OpenAI batch dialect: a JSONL file plus a batch job, both upstream-side.

OpenAI is the reference batch API and the proxy's client surface, so this
dialect is the one with no translation: the client's chat-completions lines are
written into a JSONL file (only ``model`` rewritten), the proxy uploads that file
through OpenAI's Files API, and the finished job's ``output_file_id`` /
``error_file_id`` are downloaded and handed to the batch service.

Wire notes that matter:

- The batch create body names the endpoint the lines target,
  ``/v1/chat/completions``, and ``completion_window`` (``24h`` is the only
  accepted value).
- ``request_counts`` is already OpenAI's shape, so it is copied through.
- ``errors`` is a list envelope (``{"object": "list", "data": [...]}``) or null;
  the proxy stores it as a plain list and re-wraps it for the client.
- Result lines are ``{id, custom_id, response: {status_code, request_id, body},
  error}``. The error file carries the same shape with ``response: null``.
"""

from collections.abc import Iterable
from typing import Any

import orjson

from llm_proxy.batch.types import (
    BATCH_TIMESTAMP_FIELDS,
    DEFAULT_COMPLETION_WINDOW,
    SUPPORTED_BATCH_ENDPOINT,
    BatchRequestItem,
    BatchResultItem,
    BatchSnapshot,
    BatchSubmission,
)
from llm_proxy.core.exceptions import ProviderError
from llm_proxy.core.utils import optional_int
from llm_proxy.observability.logger import get_logger
from llm_proxy.providers.capabilities.batch import BatchCapabilityMixin, request_counts
from llm_proxy.providers.capabilities.host import BatchSelf

logger = get_logger(__name__)

#: The upstream's own path prefixes, appended to the provider base URL (which
#: already carries ``/v1``). Both are overridable through the provider's
#: ``endpoint_base_urls`` metadata under the keys ``files`` and ``batch``.
FILES_ENDPOINT = "/files"
BATCHES_ENDPOINT = "/batches"


def batch_jsonl_bytes(items: Iterable[dict[str, Any]]) -> bytes:
    """Serialize batch lines as JSONL (one object per line, trailing newline)."""
    return b"".join(orjson.dumps(item) + b"\n" for item in items)


def parse_batch_jsonl(content: bytes) -> list[dict[str, Any]]:
    """Parse a batch result file into dicts, skipping blank lines."""
    lines: list[dict[str, Any]] = []
    for raw in content.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            parsed = orjson.loads(raw)
        except orjson.JSONDecodeError:
            logger.warning("Skipping unparsable batch result line")
            continue
        if isinstance(parsed, dict):
            lines.append(parsed)
    return lines


def result_items_from_lines(
    lines: Iterable[dict[str, Any]], *, start_index: int = 0
) -> list[BatchResultItem]:
    """Convert OpenAI-shaped result lines into normalized result items."""
    items: list[BatchResultItem] = []
    for offset, line in enumerate(lines):
        response = line.get("response")
        error = line.get("error")
        if isinstance(response, dict):
            body = response.get("body")
            items.append(
                BatchResultItem(
                    index=start_index + offset,
                    custom_id=line.get("custom_id"),
                    body=body if isinstance(body, dict) else None,
                    error=None,
                    status_code=optional_int(response.get("status_code")) or 200,
                    request_id=response.get("request_id"),
                )
            )
            continue
        items.append(
            BatchResultItem(
                index=start_index + offset,
                custom_id=line.get("custom_id"),
                error=error if isinstance(error, dict) else {"message": "Batch request failed"},
            )
        )
    return items


def snapshot_from_batch(raw: dict[str, Any]) -> BatchSnapshot:
    """Normalize an OpenAI batch object into a :class:`BatchSnapshot`."""
    timestamps: dict[str, int | None] = {field: None for field in BATCH_TIMESTAMP_FIELDS}
    for field in BATCH_TIMESTAMP_FIELDS:
        timestamps[field] = optional_int(raw.get(field))
    counts = request_counts(raw.get("request_counts"))
    errors = raw.get("errors")
    error_list: list[dict[str, Any]] | None = None
    if isinstance(errors, dict):
        data = errors.get("data")
        if isinstance(data, list):
            error_list = [entry for entry in data if isinstance(entry, dict)] or None
    elif isinstance(errors, list):
        error_list = [entry for entry in errors if isinstance(entry, dict)] or None
    return BatchSnapshot(
        status=str(raw.get("status") or "validating"),
        request_counts=counts,
        timestamps=timestamps,
        errors=error_list,
        output_file_id=raw.get("output_file_id"),
        error_file_id=raw.get("error_file_id"),
        expires_at=optional_int(raw.get("expires_at")),
    )


class OpenAIBatchMixin(BatchCapabilityMixin):
    """Batch support for the OpenAI adapter (native Files + Batches API)."""

    SUPPORTS_BATCH = True
    BATCH_FORWARDS_CHAT_BODY = True
    FILES_ENDPOINT = FILES_ENDPOINT
    BATCHES_ENDPOINT = BATCHES_ENDPOINT

    async def submit_batch(
        self: BatchSelf,
        *,
        model: str,
        requests: list[BatchRequestItem],
        display_name: str,
        completion_window: str = DEFAULT_COMPLETION_WINDOW,
    ) -> BatchSubmission:
        payload = batch_jsonl_bytes(
            {
                "custom_id": item.custom_id,
                "method": "POST",
                "url": SUPPORTED_BATCH_ENDPOINT,
                "body": item.body,
            }
            for item in requests
        )
        upload = await self._batch_json(
            "POST",
            self._files_url(),
            headers=self._upload_headers(),
            files={"file": (f"{display_name}.jsonl", payload, "application/jsonl")},
            data={"purpose": "batch"},
        )
        upstream_file_id = upload.get("id")
        if not isinstance(upstream_file_id, str) or not upstream_file_id:
            raise ProviderError(
                message="OpenAI file upload returned no file id",
                error_type="api_error",
                status_code=502,
                provider_name=self.provider_name,
            )

        raw = await self._batch_json(
            "POST",
            self._batches_url(),
            headers=self._build_headers(),
            json_body={
                "input_file_id": upstream_file_id,
                "endpoint": SUPPORTED_BATCH_ENDPOINT,
                "completion_window": completion_window,
            },
        )
        upstream_id = raw.get("id")
        if not isinstance(upstream_id, str) or not upstream_id:
            raise ProviderError(
                message="OpenAI batch create returned no batch id",
                error_type="api_error",
                status_code=502,
                provider_name=self.provider_name,
            )
        return BatchSubmission(
            upstream_id=upstream_id,
            snapshot=snapshot_from_batch(raw),
            upstream_file_id=upstream_file_id,
        )

    async def fetch_batch(self: BatchSelf, upstream_id: str) -> BatchSnapshot:
        raw = await self._batch_json(
            "GET", self._batches_url(f"/{upstream_id}"), headers=self._build_headers()
        )
        return snapshot_from_batch(raw)

    async def fetch_batch_results(
        self: BatchSelf, snapshot: BatchSnapshot
    ) -> list[BatchResultItem] | None:
        """Download the output and error files and normalize their lines.

        Errors come last so the file keeps the order OpenAI's own output does:
        successes in the output file, failures in the error file.
        """
        items: list[BatchResultItem] = []
        if snapshot.output_file_id:
            content = await self._batch_bytes(
                "GET",
                self._files_url(f"/{snapshot.output_file_id}/content"),
                headers=self._build_headers(),
            )
            items.extend(result_items_from_lines(parse_batch_jsonl(content)))
        if snapshot.error_file_id:
            content = await self._batch_bytes(
                "GET",
                self._files_url(f"/{snapshot.error_file_id}/content"),
                headers=self._build_headers(),
            )
            items.extend(
                result_items_from_lines(parse_batch_jsonl(content), start_index=len(items))
            )
        return items or None

    async def cancel_batch(self: BatchSelf, upstream_id: str) -> BatchSnapshot | None:
        raw = await self._batch_json(
            "POST", self._batches_url(f"/{upstream_id}/cancel"), headers=self._build_headers()
        )
        return snapshot_from_batch(raw)


__all__ = [
    "OpenAIBatchMixin",
    "batch_jsonl_bytes",
    "parse_batch_jsonl",
    "result_items_from_lines",
    "snapshot_from_batch",
]
