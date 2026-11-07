"""The batch service: OpenAI's batch surface over three upstream dialects.

What lives here is everything that is *the proxy's* rather than a provider's:

- file storage and ownership,
- JSONL parsing and the one-model-per-batch rule,
- provider resolution (the same highest-priority provider the synchronous path
  would pick, with the same upstream model-name mapping),
- line translation on the side of the seam that needs it,
- normalizing an upstream snapshot into the stored job,
- materializing results into a proxy-hosted output file, and
- rendering OpenAI wire objects.

Upstream wire knowledge stays in ``providers/*/batch.py``; this module never
inspects a provider payload. See ADR-0058.
"""

import hashlib
import secrets
import time
from dataclasses import dataclass
from typing import Any, Protocol, cast

import orjson

from llm_proxy.batch.accounting import record_batch_usage
from llm_proxy.batch.types import (
    BATCH_STATUSES,
    BATCH_TIMESTAMP_FIELDS,
    DEFAULT_COMPLETION_WINDOW,
    SUPPORTED_BATCH_ENDPOINT,
    TERMINAL_BATCH_STATUSES,
    BatchRequestItem,
    BatchSnapshot,
    BatchSubmission,
    BatchTarget,
)
from llm_proxy.core.exceptions import (
    ConfigurationError,
    ForbiddenError,
    LLMProxyError,
    ModelNotFoundError,
    NotFoundError,
    ValidationError,
)
from llm_proxy.database.repositories.batches import BatchRepository
from llm_proxy.database.tables import BatchFileRecord, BatchJobRecord
from llm_proxy.http.client import ProviderHTTPClientManager
from llm_proxy.observability.logger import get_logger
from llm_proxy.protocols.registry import get_protocol_serializer
from llm_proxy.routing.config import routing_mode_from_model
from llm_proxy.serialization.format_context import FormatContext
from llm_proxy.services import RuntimeServices

logger = get_logger(__name__)

#: Purpose an uploaded batch input file carries, and the purpose the proxy puts
#: on the result files it materializes. Both are OpenAI's own vocabulary.
FILE_PURPOSE_BATCH = "batch"
FILE_PURPOSE_BATCH_OUTPUT = "batch_output"
SUPPORTED_FILE_PURPOSES: frozenset[str] = frozenset({FILE_PURPOSE_BATCH, FILE_PURPOSE_BATCH_OUTPUT})

#: Largest input JSONL the proxy accepts. Gemini's inline batch caps the whole
#: request at 20 MB and the proxy wraps every line, so a 16 MiB upload leaves
#: room for the envelope while staying a plausible batch size (thousands of
#: chat requests); bigger batches get a 413 naming the limit.
MAX_BATCH_FILE_BYTES: int = 16 * 1024 * 1024

#: OpenAI's per-batch request ceiling.
MAX_BATCH_LINES: int = 50_000

DEFAULT_LIST_LIMIT: int = 20
MAX_LIST_LIMIT: int = 100


class BatchAdapter(Protocol):
    """The provider-side contract the service drives (``BatchCapabilityMixin``).

    Declared here so the service depends on the methods it calls rather than on
    ``BaseAdapter``. ``SUPPORTS_BATCH`` is optional at runtime because adapters
    without the mixin never declare it; ``_adapter_supports_batch`` owns that
    absence check.
    """

    provider_name: str
    SUPPORTS_BATCH: bool
    BATCH_FORWARDS_CHAT_BODY: bool

    async def submit_batch(
        self,
        *,
        model: str,
        requests: list[BatchRequestItem],
        display_name: str,
        completion_window: str = ...,
    ) -> BatchSubmission: ...

    async def fetch_batch(self, upstream_id: str) -> BatchSnapshot: ...

    async def fetch_batch_results(self, snapshot: BatchSnapshot) -> list[Any] | None: ...

    async def cancel_batch(self, upstream_id: str) -> BatchSnapshot | None: ...

    def build_batch_request(self, request: Any) -> dict[str, Any]: ...

    def parse_batch_result(self, payload: dict[str, Any], model: str) -> Any: ...

    async def close(self) -> None: ...


def _adapter_supports_batch(adapter: BatchAdapter) -> bool:
    """Whether an adapter exposes the batch capability.

    Only adapters including ``BatchCapabilityMixin`` carry ``SUPPORTS_BATCH``;
    ordinary providers must be refused by name rather than crashing on the
    missing attribute.
    """
    return bool(getattr(adapter, "SUPPORTS_BATCH", False))


@dataclass(frozen=True, slots=True)
class ParsedBatchLine:
    """One validated JSONL line, still in the client's dialect."""

    lineno: int
    custom_id: str
    model: str
    body: dict[str, Any]


def _is_chat_endpoint(url: str) -> bool:
    """Whether a client-supplied path names the chat-completions endpoint."""
    return "/" + url.strip().strip("/") in ("/chat/completions", "/v1/chat/completions")


def _normalize_batch_endpoint(endpoint: Any) -> str:
    """Validate the batch endpoint and return its canonical spelling.

    Clients built on the OpenAI SDK send ``/v1/chat/completions``; a client with
    a base URL ending at ``/v1`` sometimes sends ``/chat/completions``. Both name
    the same endpoint, and the stored value is always the canonical one.
    """
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValidationError(
            message="'endpoint' is required and must be '/v1/chat/completions'",
            code="invalid_request_error",
            status_code=400,
        )
    if _is_chat_endpoint(endpoint):
        return SUPPORTED_BATCH_ENDPOINT
    raise ValidationError(
        message=(
            f"Unsupported batch endpoint '{endpoint}'. Batch mode supports "
            f"'{SUPPORTED_BATCH_ENDPOINT}' only."
        ),
        code="batch_unsupported_endpoint",
        status_code=400,
    )


def _line_error(lineno: int, message: str, *, code: str = "invalid_batch_line") -> ValidationError:
    return ValidationError(message=f"Batch line {lineno}: {message}", code=code, status_code=400)


def parse_batch_lines(content: str) -> list[ParsedBatchLine]:
    """Parse and validate a JSONL batch input file.

    Every rule the upstreams share is enforced here, before any provider call,
    so a bad line fails at creation with its line number instead of as an opaque
    upstream rejection an hour later: one JSON object per line, a unique
    ``custom_id``, a ``POST /v1/chat/completions`` entry, a non-empty ``model``,
    and no ``stream`` (a batch request has no stream to write to).
    """
    lines: list[ParsedBatchLine] = []
    seen_ids: set[str] = set()
    for lineno, raw in enumerate(content.splitlines(), start=1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            parsed = orjson.loads(raw)
        except orjson.JSONDecodeError as exc:
            raise _line_error(lineno, f"not valid JSON ({exc})") from exc
        if not isinstance(parsed, dict):
            raise _line_error(lineno, "each line must be a JSON object")

        custom_id = parsed.get("custom_id")
        if not isinstance(custom_id, str) or not custom_id.strip():
            raise _line_error(lineno, "'custom_id' is required and must be a non-empty string")
        if custom_id in seen_ids:
            raise _line_error(lineno, f"duplicate custom_id '{custom_id}'")
        seen_ids.add(custom_id)

        method = parsed.get("method", "POST")
        if not isinstance(method, str) or method.upper() != "POST":
            raise _line_error(lineno, "'method' must be 'POST'")

        url = parsed.get("url")
        if not isinstance(url, str) or not _is_chat_endpoint(url):
            raise _line_error(lineno, f"'url' must be '{SUPPORTED_BATCH_ENDPOINT}'")

        body = parsed.get("body")
        if not isinstance(body, dict):
            raise _line_error(lineno, "'body' is required and must be a JSON object")
        if body.get("stream"):
            raise _line_error(lineno, "'stream' is not supported in a batch request")

        model = body.get("model")
        if not isinstance(model, str) or not model.strip():
            raise _line_error(lineno, "'body.model' is required and must be a non-empty string")

        lines.append(
            ParsedBatchLine(lineno=lineno, custom_id=custom_id, model=model.strip(), body=body)
        )

    if not lines:
        raise ValidationError(
            message="Batch input file contains no requests",
            code="empty_batch",
            status_code=400,
        )
    if len(lines) > MAX_BATCH_LINES:
        raise ValidationError(
            message=(
                f"Batch input file has {len(lines)} requests, above the {MAX_BATCH_LINES} "
                "requests per batch limit"
            ),
            code="batch_too_many_requests",
            status_code=413,
        )
    models = {line.model for line in lines}
    if len(models) > 1:
        raise ValidationError(
            message=(
                "Every request in a batch must use the same model; this file mixes "
                f"{len(models)} ({', '.join(sorted(models))}). OpenRouter and Gemini "
                "bind a batch to one model, so split the file into one batch per model."
            ),
            code="mixed_batch_models",
            status_code=400,
        )
    return lines


def _result_line_id(batch_id: str, custom_id: str) -> str:
    """A stable id for one result line, matching OpenAI's ``batch_req_…`` shape."""
    digest = hashlib.sha256(f"{batch_id}:{custom_id}".encode()).hexdigest()[:24]
    return f"batch_req_{digest}"


@dataclass(frozen=True, slots=True)
class RenderedResults:
    """The two result files plus the bodies the spend accounting prices.

    The bodies are the ones actually written, after translation and the client's
    model name was restored, so the row that bills a batch and the output file a
    client downloads describe the same lines.
    """

    successes: bytes
    errors: bytes
    bodies: list[dict[str, Any]]


class BatchService:
    """Files and batch jobs for one request's database session.

    ``adapter_factory`` is the test seam: production builds the provider adapter
    from the live configuration, tests hand in a fake that speaks the same
    capability contract. ``usage_dispatch`` is the spend seam, defaulting to the
    process-lifetime usage writer.
    """

    def __init__(
        self,
        session: Any,
        services: RuntimeServices,
        *,
        adapter_factory: Any = None,
        usage_dispatch: Any = None,
    ) -> None:
        self._repo = BatchRepository(session)
        self._services = services
        self._adapter_factory = adapter_factory
        self._usage_dispatch = usage_dispatch

    # ── Files ─────────────────────────────────────────────────────────────

    async def create_file(
        self,
        *,
        owner: str,
        filename: str,
        purpose: str,
        content: bytes,
    ) -> BatchFileRecord:
        """Store an uploaded file after validating purpose, size and encoding."""
        purpose = (purpose or FILE_PURPOSE_BATCH).strip()
        if purpose not in SUPPORTED_FILE_PURPOSES:
            raise ValidationError(
                message=(
                    f"Unsupported file purpose '{purpose}'; supported purposes are "
                    f"{', '.join(sorted(SUPPORTED_FILE_PURPOSES))}"
                ),
                code="invalid_file_purpose",
                status_code=400,
            )
        if not content.strip():
            raise ValidationError(
                message="Uploaded file is empty",
                code="empty_file",
                status_code=400,
            )
        if len(content) > MAX_BATCH_FILE_BYTES:
            raise ValidationError(
                message=(
                    f"Uploaded file is {len(content)} bytes, above the "
                    f"{MAX_BATCH_FILE_BYTES}-byte batch limit"
                ),
                code="file_too_large",
                status_code=413,
            )
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValidationError(
                message="Batch files must be UTF-8 encoded JSONL",
                code="invalid_file_encoding",
                status_code=400,
            ) from exc

        return await self._repo.create_file(
            file_id=f"file-{secrets.token_hex(12)}",
            owner=owner,
            filename=filename or "batch.jsonl",
            purpose=purpose,
            content=text,
            created_at=int(time.time()),
        )

    async def get_file(self, *, owner: str, file_id: str) -> BatchFileRecord:
        record = await self._repo.get_file(file_id, owner)
        if record is None:
            raise NotFoundError(
                message=f"No such file: '{file_id}'",
                code="file_not_found",
                status_code=404,
            )
        return record

    async def get_file_content(self, *, owner: str, file_id: str) -> tuple[str, bytes]:
        """Return ``(filename, content_bytes)`` for the download endpoint."""
        record = await self.get_file(owner=owner, file_id=file_id)
        return record.filename, record.content.encode("utf-8")

    async def list_files(
        self,
        *,
        owner: str,
        limit: int = DEFAULT_LIST_LIMIT,
        after: str | None = None,
        purpose: str | None = None,
    ) -> tuple[list[BatchFileRecord], bool]:
        limit = _clamp_limit(limit)
        records = await self._repo.list_files(owner, limit=limit + 1, after=after, purpose=purpose)
        return records[:limit], len(records) > limit

    async def delete_file(self, *, owner: str, file_id: str) -> bool:
        """Delete a stored file, unless a running batch reads it as input.

        Deleting the requests out from under a job would leave its
        ``input_file_id`` dangling and the job unpollable, so that one case is
        refused while the job is not terminal.
        """
        if await self._repo.input_file_in_use(file_id, owner):
            raise ValidationError(
                message=(
                    f"File '{file_id}' is the input of a batch that has not finished; "
                    "cancel or wait for the batch before deleting it."
                ),
                code="file_in_use",
                status_code=400,
            )
        return await self._repo.delete_file(file_id, owner)

    # ── Batch jobs ────────────────────────────────────────────────────────

    async def create_batch(
        self,
        *,
        owner: str,
        input_file_id: str,
        endpoint: str,
        completion_window: str,
        metadata: dict[str, str] | None = None,
        allowed_models: list[str] | None = None,
        user_id: int | None = None,
    ) -> BatchJobRecord:
        """Translate a stored input file and submit it to the resolved upstream.

        ``allowed_models`` is the calling key's allowlist, read from the request
        facts: the model restriction middleware cannot see a batch's model (it
        lives in the uploaded file, not in the create body), so the per-key
        authorization check happens here instead.

        ``user_id`` is the authenticated user the job bills to. Creation is the
        only moment it is known — a batch outlives its request, and results are
        priced on a later poll by a different request — so it is stored on the
        job and copied onto the usage row at materialization.
        """
        endpoint = _normalize_batch_endpoint(endpoint)
        if completion_window != DEFAULT_COMPLETION_WINDOW:
            raise ValidationError(
                message=(
                    f"Unsupported completion_window '{completion_window}'; "
                    f"'{DEFAULT_COMPLETION_WINDOW}' is the only accepted value"
                ),
                code="invalid_completion_window",
                status_code=400,
            )

        source = await self.get_file(owner=owner, file_id=input_file_id)
        if source.purpose != FILE_PURPOSE_BATCH:
            raise ValidationError(
                message=(
                    f"File '{input_file_id}' has purpose '{source.purpose}'; a batch input "
                    f"file must have purpose '{FILE_PURPOSE_BATCH}'"
                ),
                code="invalid_input_file",
                status_code=400,
            )

        lines = parse_batch_lines(source.content)
        requested_model = lines[0].model
        if allowed_models is not None and requested_model not in allowed_models:
            raise ForbiddenError(
                message=(f"This API key is not authorized to access model '{requested_model}'."),
                code="model_not_allowed",
            )
        target = await self._resolve_target(requested_model)

        adapter = await self._adapter_for(target.provider_name)
        try:
            if not _adapter_supports_batch(adapter):
                raise ValidationError(
                    message=(
                        f"Provider '{target.provider_name}' ({target.provider_type}) has no "
                        f"batch API, so model '{requested_model}' cannot be batched. Configure "
                        "a provider with batch support (openai, openrouter, gemini) for this model."
                    ),
                    code="batch_unsupported",
                    status_code=400,
                )
            requests = [
                BatchRequestItem(
                    custom_id=line.custom_id,
                    body=self._build_line(adapter, line, target.upstream_model),
                )
                for line in lines
            ]
            batch_id = f"batch_{secrets.token_hex(12)}"
            submission = await adapter.submit_batch(
                model=target.upstream_model,
                requests=requests,
                display_name=batch_id,
                completion_window=completion_window,
            )
        finally:
            await adapter.close()

        snapshot = submission.snapshot
        logger.info(
            f"Batch {batch_id} submitted to {target.provider_name} ({target.provider_type}) as "
            f"{submission.upstream_id} for {len(lines)} requests"
            + (f", input file {submission.upstream_file_id}" if submission.upstream_file_id else "")
        )
        counts = snapshot.request_counts or {
            "total": len(lines),
            "completed": 0,
            "failed": 0,
        }
        if not counts.get("total"):
            counts = {**counts, "total": len(lines)}
        return await self._repo.create_job(
            id=batch_id,
            owner=owner,
            user_id=user_id,
            provider=target.provider_name,
            provider_type=target.provider_type,
            upstream_id=submission.upstream_id,
            upstream_file_id=submission.upstream_file_id,
            model=requested_model,
            upstream_model=target.upstream_model,
            endpoint=endpoint,
            completion_window=completion_window,
            status=snapshot.status,
            input_file_id=input_file_id,
            request_counts=counts,
            batch_metadata=metadata or {},
            custom_ids=[line.custom_id for line in lines],
            created_at=int(time.time()),
            expires_at=snapshot.expires_at,
            **{
                field: value
                for field in BATCH_TIMESTAMP_FIELDS
                if (value := snapshot.timestamps.get(field)) is not None
            },
        )

    async def get_batch(self, *, owner: str, batch_id: str) -> BatchJobRecord:
        """Return a job, refreshing it from the upstream while it is unfinished.

        A finished job whose results were already pulled is served from the row.
        A terminal job that never got a results pull (the download failed on an
        earlier read, or a cancel landed on a final status) retries that pull
        here — that is the only way results appear without a background poller.
        """
        job = await self._repo.get_job(batch_id, owner)
        if job is None:
            raise NotFoundError(
                message=f"No such batch: '{batch_id}'",
                code="batch_not_found",
                status_code=404,
            )
        if job.results_materialized_at is not None:
            return job

        if job.status in TERMINAL_BATCH_STATUSES:
            # Already final in the row: a retried results pull must not turn a
            # status read into a 5xx just because the upstream was unavailable.
            try:
                snapshot = await self._fetch_snapshot(job)
            except LLMProxyError as exc:
                logger.warning(
                    f"Batch {job.id} is {job.status} but its results could not be read yet: {exc}"
                )
                return job
        else:
            snapshot = await self._fetch_snapshot(job)

        await self._apply_snapshot(job, snapshot)
        if snapshot.terminal:
            await self._materialize(job, snapshot)
        return job

    async def list_batches(
        self,
        *,
        owner: str,
        limit: int = DEFAULT_LIST_LIMIT,
        after: str | None = None,
    ) -> tuple[list[BatchJobRecord], bool]:
        """List stored jobs, newest first. Does not poll upstreams.

        Listing is a single query on purpose: refreshing N jobs would turn one
        list call into N provider calls, and clients that need the current state
        poll the job id they care about.
        """
        limit = _clamp_limit(limit)
        jobs = await self._repo.list_jobs(owner, limit=limit + 1, after=after)
        return jobs[:limit], len(jobs) > limit

    async def cancel_batch(self, *, owner: str, batch_id: str) -> BatchJobRecord:
        job = await self._repo.get_job(batch_id, owner)
        if job is None:
            raise NotFoundError(
                message=f"No such batch: '{batch_id}'",
                code="batch_not_found",
                status_code=404,
            )
        if job.status in TERMINAL_BATCH_STATUSES:
            return job

        adapter = await self._adapter_for(job.provider)
        try:
            snapshot = await adapter.cancel_batch(job.upstream_id)
        finally:
            await adapter.close()
        if snapshot is None:
            raise ValidationError(
                message=(
                    f"Provider '{job.provider}' ({job.provider_type}) has no batch cancel "
                    "endpoint, so this batch cannot be cancelled upstream."
                ),
                code="batch_cancel_unsupported",
                status_code=400,
            )
        await self._apply_snapshot(job, snapshot)
        if snapshot.terminal:
            # A cancel can land on a final status directly (Gemini's operation
            # is cancelled at once; OpenAI may already have finished), and no
            # later read would retry the pull, so take it here.
            await self._materialize(job, snapshot)
        logger.info(f"Batch {job.id}: cancel answered by {job.provider} as '{job.status}'")
        return job

    # ── Translation ───────────────────────────────────────────────────────

    def _build_line(
        self, adapter: BatchAdapter, line: ParsedBatchLine, upstream_model: str
    ) -> dict[str, Any]:
        """Turn one client line into the provider's batch entry body.

        The verbatim side only rewrites the model. The translating side parses
        with the OpenAI protocol serializer and rebuilds through the provider's
        own serializer — the same canonical path a synchronous request takes.
        """
        body = dict(line.body)
        if adapter.BATCH_FORWARDS_CHAT_BODY:
            body["model"] = upstream_model
            return body

        try:
            unified = get_protocol_serializer("openai").parse_request(body)
        except LLMProxyError as exc:
            raise _line_error(
                line.lineno, f"cannot be translated for {adapter.provider_name}: {exc.message}"
            ) from exc
        unified.model = upstream_model
        unified.metadata.protocol_name = "openai"
        unified._raw_protocol_data = body
        return adapter.build_batch_request(unified)

    def _format_translated_result(
        self, adapter: BatchAdapter, payload: dict[str, Any], job: BatchJobRecord
    ) -> dict[str, Any]:
        """Convert one provider result payload into an OpenAI chat completion."""
        internal = adapter.parse_batch_result(payload, job.model)
        formatted = get_protocol_serializer("openai").format_response(internal, FormatContext())
        formatted["model"] = job.model
        return formatted

    # ── Provider resolution and adapters ──────────────────────────────────

    async def _resolve_target(self, model: str) -> BatchTarget:
        """Resolve the provider and upstream model name for a model.

        Takes the highest-priority enabled provider configured for the model and
        maps the name the same way the synchronous path does (the mapping's
        ``provider_model_name``, else the model's ``model_name``). A provider
        without a batch API is not skipped — the caller explains that failure by
        name, instead of silently spending on a different provider. Selection
        strategies, circuit breakers and provider stats are deliberately not
        consulted: a batch is submitted once, so the operator's static priority
        order is what decides.
        """
        if routing_mode_from_model(model) is not None:
            raise ValidationError(
                message=(
                    f"'{model}' is a virtual model. Batch mode needs a concrete model: "
                    "routing picks a model per request, and a batch is submitted once."
                ),
                code="batch_requires_concrete_model",
                status_code=400,
            )

        config = await self._services.config_manager().get_config()
        model_config = config.models.get(model)
        if model_config is None:
            raise ModelNotFoundError(model)

        for entry in model_config.get_providers_by_priority():
            provider_config = config.provider_configs.get(entry.provider)
            if provider_config is None or not provider_config.enabled:
                continue
            upstream_model = entry.provider_model_name or model_config.model_name or model
            return BatchTarget(
                provider_name=entry.provider,
                provider_type=provider_config.type,
                upstream_model=upstream_model,
            )

        raise ValidationError(
            message=f"No enabled provider is configured for model '{model}'",
            code="provider_not_available",
            status_code=400,
        )

    async def _adapter_for(self, provider_name: str) -> BatchAdapter:
        if self._adapter_factory is not None:
            return cast(BatchAdapter, await self._adapter_factory(provider_name))

        from llm_proxy.core.internal_call import build_provider_adapter

        manager = self._services.config_manager()
        config = await manager.get_config()
        provider_config = await manager.get_provider_config(provider_name)
        http_manager = self._services.http_client()
        http_client = await http_manager.get_client(provider_name)
        return cast(
            BatchAdapter,
            build_provider_adapter(
                provider_name,
                provider_config,
                http_client=http_client,
                unknown_fields_policy=config.server_params.unknown_fields_policy,
                unsupported_block_policy=config.server_params.unsupported_block_policy,
                http_client_manager=(
                    http_manager if isinstance(http_manager, ProviderHTTPClientManager) else None
                ),
            ),
        )

    # ── Snapshot handling ─────────────────────────────────────────────────

    async def _fetch_snapshot(self, job: BatchJobRecord) -> BatchSnapshot:
        adapter = await self._adapter_for(job.provider)
        try:
            if not _adapter_supports_batch(adapter):
                raise ConfigurationError(
                    message=(
                        f"Provider '{job.provider}' ({job.provider_type}) no longer supports "
                        f"batch; batch '{job.id}' cannot be polled."
                    )
                )
            return await adapter.fetch_batch(job.upstream_id)
        finally:
            await adapter.close()

    async def _apply_snapshot(self, job: BatchJobRecord, snapshot: BatchSnapshot) -> None:
        fields: dict[str, Any] = {}
        if snapshot.status in BATCH_STATUSES:
            fields["status"] = snapshot.status
        if snapshot.request_counts:
            fields["request_counts"] = snapshot.request_counts
        if snapshot.expires_at is not None:
            fields["expires_at"] = snapshot.expires_at
        if snapshot.errors is not None:
            fields["errors"] = snapshot.errors
        if snapshot.usage is not None:
            fields["usage"] = snapshot.usage
        for field in BATCH_TIMESTAMP_FIELDS:
            value = snapshot.timestamps.get(field)
            if value is not None:
                fields[field] = value
        if fields:
            await self._repo.update_job(job, **fields)

    async def _materialize(self, job: BatchJobRecord, snapshot: BatchSnapshot) -> None:
        """Pull results once and store them as the job's output/error files.

        Successes go to ``output_file_id`` and failures to ``error_file_id``,
        matching OpenAI's split. The upstream pull happens before the claim, so a
        failed download leaves no marker behind and the next status read retries
        — a completed batch must not turn into a 5xx just because the download
        was momentarily refused.

        The claim is taken only once the pull succeeded, which keeps it a short
        write and makes it one-shot: a second concurrent poll may download the
        same bytes, but only one of them stores files. A batch that finished with
        no results keeps the claim and both file ids null, so polling it again
        does not re-download anything.
        """
        if job.output_file_id or job.error_file_id:
            return
        adapter = await self._adapter_for(job.provider)
        try:
            items = await adapter.fetch_batch_results(snapshot)
            forwards_body = bool(adapter.BATCH_FORWARDS_CHAT_BODY)
            rendered = self._result_lines(
                job, items or [], adapter=adapter, forwards_body=forwards_body
            )
        except Exception as exc:  # noqa: BLE001 - a status read must not fail on results
            logger.warning(
                f"Batch results for {job.id} are not available yet: {exc}",
            )
            return
        finally:
            await adapter.close()

        if not await self._repo.claim_materialization(job, now=int(time.time())):
            # Another read got there first; reload so this response carries the
            # files it stored instead of the ids this request downloaded.
            await self._repo.refresh_job(job)
            return

        fields: dict[str, Any] = {}
        if rendered.successes:
            record = await self._store_result_file(
                owner=job.owner,
                filename=f"{job.id}_output.jsonl",
                purpose=FILE_PURPOSE_BATCH_OUTPUT,
                content=rendered.successes,
            )
            fields["output_file_id"] = record.id
        if rendered.errors:
            record = await self._store_result_file(
                owner=job.owner,
                filename=f"{job.id}_errors.jsonl",
                purpose=FILE_PURPOSE_BATCH_OUTPUT,
                content=rendered.errors,
            )
            fields["error_file_id"] = record.id
        if fields:
            await self._repo.update_job(job, **fields)
            logger.info(f"Batch {job.id}: results stored as {fields}")

        await self._record_usage(job, rendered.bodies)

    async def _record_usage(self, job: BatchJobRecord, bodies: list[dict[str, Any]]) -> None:
        """Price the batch's lines into its one usage row (ADR-0058).

        A status read is not the place to fail on accounting: the results are
        already stored and claimed, so a pricing error is logged and the batch
        keeps its (now terminal, results-materialized) state rather than turning
        every later poll into a 5xx.
        """
        if not bodies:
            return
        try:
            await record_batch_usage(
                job,
                bodies,
                config_manager=self._services.config_manager(),
                dispatch=self._usage_dispatch,
            )
        except Exception as exc:  # noqa: BLE001 - accounting must not fail the poll
            logger.warning(f"Batch {job.id}: spend could not be recorded: {exc}")

    def _result_lines(
        self,
        job: BatchJobRecord,
        items: list[Any],
        *,
        adapter: BatchAdapter,
        forwards_body: bool,
    ) -> RenderedResults:
        """Render result items as the two JSONL files, in OpenAI's line shape."""
        custom_ids = list(job.custom_ids or [])
        success: list[bytes] = []
        failed: list[bytes] = []
        bodies: list[dict[str, Any]] = []
        for item in items:
            custom_id = item.custom_id or (
                custom_ids[item.index] if 0 <= item.index < len(custom_ids) else None
            )
            if not custom_id:
                logger.warning(f"Batch {job.id}: dropping a result with no custom_id")
                continue
            body = item.body
            if body is not None and not forwards_body:
                try:
                    body = self._format_translated_result(adapter, body, job)
                except Exception as exc:  # noqa: BLE001 - one bad line must not drop the file
                    logger.warning(
                        f"Batch {job.id}: result for '{custom_id}' could not be parsed: {exc}"
                    )
                    body = None
            if body is not None:
                # The client asked for its own model name; a result that echoed the
                # upstream's name would make batch output disagree with the
                # synchronous response for the same model mapping.
                body = {**body, "model": job.model}
                bodies.append(body)
                success.append(
                    orjson.dumps(
                        {
                            "id": _result_line_id(job.id, custom_id),
                            "custom_id": custom_id,
                            "response": {
                                "status_code": item.status_code or 200,
                                "request_id": item.request_id,
                                "body": body,
                            },
                            "error": None,
                        }
                    )
                )
                continue
            failed.append(
                orjson.dumps(
                    {
                        "id": _result_line_id(job.id, custom_id),
                        "custom_id": custom_id,
                        "response": None,
                        "error": item.error
                        or {"code": "batch_request_failed", "message": "Batch request failed"},
                    }
                )
            )
        return RenderedResults(successes=_jsonl(success), errors=_jsonl(failed), bodies=bodies)

    async def _store_result_file(
        self, *, owner: str, filename: str, purpose: str, content: bytes
    ) -> BatchFileRecord:
        """Store a materialized result file; no input caps apply to results."""
        return await self._repo.create_file(
            file_id=f"file-{secrets.token_hex(12)}",
            owner=owner,
            filename=filename,
            purpose=purpose,
            content=content.decode("utf-8"),
            created_at=int(time.time()),
        )


def _jsonl(lines: list[bytes]) -> bytes:
    return b"".join(line + b"\n" for line in lines)


def _clamp_limit(limit: int) -> int:
    if limit <= 0:
        return DEFAULT_LIST_LIMIT
    return min(limit, MAX_LIST_LIMIT)


__all__ = [
    "FILE_PURPOSE_BATCH",
    "FILE_PURPOSE_BATCH_OUTPUT",
    "MAX_BATCH_FILE_BYTES",
    "MAX_BATCH_LINES",
    "BatchService",
    "ParsedBatchLine",
    "parse_batch_lines",
]
