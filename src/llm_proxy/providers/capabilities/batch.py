"""Batch capability mixin.

Batch is the one capability whose *client* surface is not the request/response
pipeline: a job outlives the request that created it, and status is pulled from
the upstream rather than produced by a call. So the seam is deliberately narrow
and state-free — submit, fetch, fetch results, cancel — and every method returns
the proxy-normalized types from :mod:`llm_proxy.batch.types`. An adapter owns
its own dialect and nothing else: upload format, status names, result shapes. The
batch service (``llm_proxy.batch.service``) owns persistence, translation of
client lines, and the OpenAI wire objects.

Adapters that include this mixin declare ``SUPPORTS_BATCH = True`` in their
concrete mixin (``providers/openai/batch.py`` etc.); providers with no batch API
simply do not include it and are refused by name at batch creation.

Two sides of the seam:

- ``BATCH_FORWARDS_CHAT_BODY = True`` — the upstream's batch entry *is* an
  OpenAI chat-completions body, so the client's line reaches the adapter
  verbatim (only ``model`` rewritten) and results come back already in the
  output-file shape.
- ``BATCH_FORWARDS_CHAT_BODY = False`` — the line is parsed into an
  ``InternalRequest`` and the adapter builds the provider body through
  ``build_batch_request``; results are parsed back with ``parse_batch_result``.

URL shapes differ only in their paths, so the path fragments are class
attributes here and the dialect modules set them; the helpers below resolve them
against the provider base URL (including ``endpoint_base_urls`` overrides).
"""

from typing import Any, ClassVar

from llm_proxy.batch.types import (
    DEFAULT_COMPLETION_WINDOW,
    BatchRequestItem,
    BatchResultItem,
    BatchSnapshot,
    BatchSubmission,
)
from llm_proxy.core.exceptions import ProviderError
from llm_proxy.core.utils import optional_int
from llm_proxy.providers.capabilities.host import BatchSelf


def request_counts(raw: Any) -> dict[str, int]:
    """Normalize OpenAI-shaped ``request_counts`` into integer counters."""
    if not isinstance(raw, dict):
        return {"total": 0, "completed": 0, "failed": 0}
    return {
        "total": optional_int(raw.get("total")) or 0,
        "completed": optional_int(raw.get("completed")) or 0,
        "failed": optional_int(raw.get("failed")) or 0,
    }


class BatchCapabilityMixin:
    """Mixin for provider adapters that can run a batch.

    Implementations live next to their adapters (``providers/*/batch.py``) so
    each dialect's wire quirks stay in one file; this class fixes the contract,
    the URL shape and the shared HTTP helpers all three use.
    """

    #: Whether this adapter can run a batch at all. Concrete mixins set it.
    SUPPORTS_BATCH: ClassVar[bool] = False

    #: Whether the upstream accepts OpenAI chat-completions bodies as batch
    #: entries unchanged (OpenAI, OpenRouter) or needs translation (Gemini).
    BATCH_FORWARDS_CHAT_BODY: ClassVar[bool] = True

    #: Provider paths, appended to the base URL (or to the operator's
    #: ``endpoint_base_urls`` override for the ``files`` / ``batch`` endpoints).
    FILES_ENDPOINT: ClassVar[str] = ""
    BATCHES_ENDPOINT: ClassVar[str] = "/batches"
    #: Create path. Empty means "POST to :attr:`BATCHES_ENDPOINT`" (OpenAI,
    #: OpenRouter); Gemini's is a per-model custom method with ``{model}``.
    BATCH_CREATE_ENDPOINT: ClassVar[str] = ""

    async def submit_batch(
        self: BatchSelf,
        *,
        model: str,
        requests: list[BatchRequestItem],
        display_name: str,
        completion_window: str = DEFAULT_COMPLETION_WINDOW,
    ) -> BatchSubmission:
        """Submit the batch upstream and return its id plus first snapshot."""
        raise NotImplementedError

    async def fetch_batch(self: BatchSelf, upstream_id: str) -> BatchSnapshot:
        """Read the batch's current state from the upstream."""
        raise NotImplementedError

    async def fetch_batch_results(
        self: BatchSelf, snapshot: BatchSnapshot
    ) -> list[BatchResultItem] | None:
        """Read the finished results, or ``None`` when the upstream has none.

        The default is right for upstreams that return results with the status
        (OpenRouter, Gemini); OpenAI overrides it to download result files.
        """
        return snapshot.results

    async def cancel_batch(self: BatchSelf, upstream_id: str) -> BatchSnapshot | None:
        """Ask the upstream to cancel. ``None`` means the upstream has no cancel."""
        raise NotImplementedError

    def build_batch_request(self: BatchSelf, request: Any) -> dict[str, Any]:
        """Build one provider request body from a parsed ``InternalRequest``.

        Only used on the translating side of the seam (``BATCH_FORWARDS_CHAT_BODY
        = False``); the verbatim side never calls it.
        """
        return self._get_serializer().build_provider_request(request)

    def parse_batch_result(self: BatchSelf, payload: dict[str, Any], model: str) -> Any:
        """Parse one provider result body into an ``InternalResponse``.

        Only used on the translating side; the verbatim side's results are already
        chat-completion bodies.
        """
        return self._get_serializer().parse_provider_response(payload, model=model)

    # ── URLs ──────────────────────────────────────────────────────────────

    def _files_url(self: BatchSelf, suffix: str = "") -> str:
        return self._resolve_endpoint_url("files", self.FILES_ENDPOINT) + suffix

    def _batches_url(self: BatchSelf, suffix: str = "") -> str:
        return self._resolve_endpoint_url("batch", self.BATCHES_ENDPOINT) + suffix

    def _batch_create_url(self: BatchSelf, model: str) -> str:
        if not self.BATCH_CREATE_ENDPOINT:
            return self._batches_url()
        return self._resolve_endpoint_url("batch", self.BATCH_CREATE_ENDPOINT, model=model)

    def _upload_headers(self: BatchSelf) -> dict[str, str]:
        """Headers for a multipart upload.

        The shared header builder sets ``Content-Type: application/json``, which
        would make the upstream parse the multipart body as JSON; the client sets
        the boundary-bearing content type itself when ``files=`` is passed.
        """
        headers = self._build_headers()
        headers.pop("Content-Type", None)
        return headers

    # ── Shared HTTP helpers ───────────────────────────────────────────────

    async def _batch_request(
        self: BatchSelf,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
        files: Any = None,
        data: dict[str, Any] | None = None,
    ) -> Any:
        """Issue one batch HTTP request through the adapter's retry policy."""

        async def _make_request() -> Any:
            client = await self._get_client()
            kwargs: dict[str, Any] = {}
            if headers is not None:
                kwargs["headers"] = headers
            if json_body is not None:
                kwargs["json"] = json_body
            if files is not None:
                kwargs["files"] = files
            if data is not None:
                kwargs["data"] = data
            response = await client.request(method, url, **kwargs)
            await self._check_response_status(response)
            return response

        return await self._with_retry(_make_request)

    async def _batch_json(
        self: BatchSelf,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
        files: Any = None,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Issue a request and read a JSON object back."""
        response = await self._batch_request(
            method, url, headers=headers, json_body=json_body, files=files, data=data
        )
        payload = response.json()
        if not isinstance(payload, dict):
            raise ProviderError(
                message=(
                    f"{self.provider_name} batch endpoint {method} {url} "
                    "returned a non-object JSON payload"
                ),
                error_type="api_error",
                status_code=502,
                provider_name=self.provider_name,
                original_error={"payload_type": type(payload).__name__},
            )
        return payload

    async def _batch_bytes(
        self: BatchSelf,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> bytes:
        """Issue a request and read the raw body (result files are JSONL)."""
        response = await self._batch_request(method, url, headers=headers)
        content = response.content
        return content if content is not None else b""


__all__ = ["BatchCapabilityMixin"]
