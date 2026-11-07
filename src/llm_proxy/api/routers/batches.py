"""OpenAI-compatible batch surface: ``/v1/files`` and ``/v1/batches``.

This router is not a ``ProtocolEndpoint``: the protocol factory exists to run
chat-shaped requests through ``UnifiedProcessor``, while a batch is a job that
outlives the request that created it. Authentication is the same as every other
``/v1/*`` path — ``ApiKeyAuthMiddleware`` gates the transport (key verification,
rate limit, budget) and the router re-checks with ``require_any_auth`` so the
handlers are safe under a bare test app.

Ownership: every file and job is scoped to the key that created it, so ids are
not guessable across keys. A console session (admin JWT) gets its own scope.
"""

from typing import Annotated, Any
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from llm_proxy.api.dependencies import get_request_identity, require_any_auth
from llm_proxy.api.schemas.batch import (
    BatchCreateRequest,
    BatchList,
    BatchObject,
    DeletedFileObject,
    FileList,
    FileObject,
)
from llm_proxy.batch.service import DEFAULT_LIST_LIMIT, MAX_LIST_LIMIT, BatchService
from llm_proxy.batch.types import BatchOwner
from llm_proxy.batch.wire import batch_object, deleted_file_object, file_object, list_object
from llm_proxy.core.exceptions import AuthenticationFailedError
from llm_proxy.core.request_facts import facts_for
from llm_proxy.database.connection import get_async_session
from llm_proxy.services import runtime_services


def owner_scope(request: Request) -> str:
    """The storage scope a file or job belongs to.

    An API key's name is the natural owner for ``/v1/*`` traffic; the console
    (JWT) gets a user-scoped key so an operator can debug without borrowing a
    key name, and the two can never collide because of the ``user:`` prefix.

    An unauthenticated request raises instead of falling back to a shared scope:
    the middleware and ``require_any_auth`` both mean this cannot happen, and a
    fallback would silently let anonymous requests read whatever previous
    anonymous requests stored.
    """
    identity = get_request_identity(request)
    if identity.api_key_name:
        return BatchOwner.for_api_key(identity.api_key_name).scope
    if identity.user:
        return BatchOwner.for_user(identity.user).scope
    raise AuthenticationFailedError(
        message="Batch files and jobs require an authenticated caller.",
        code="unauthorized",
        status_code=401,
    )


def _service(request: Request, session: AsyncSession) -> BatchService:
    return BatchService(session, runtime_services(request))


def _content_disposition(filename: str) -> str:
    """Build a download header that survives a non-ASCII filename.

    HTTP headers are latin-1, so a Chinese/JA filename would raise on encode if it
    went into ``filename=`` directly. The quoted ``filename*`` form carries the
    real name (RFC 5987) and the plain ``filename`` is the ASCII fallback every
    client understands.
    """
    cleaned = filename.replace('"', "").replace("\n", "").replace("\r", "") or "batch.jsonl"
    fallback = cleaned if cleaned.isascii() else "batch.jsonl"
    return f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{quote(cleaned)}"


def _path_variants(path: str) -> list[str]:
    """``/v1/files`` and ``/v1/files/``.

    The SPA catch-all (``GET /{full_path:path}``) can partial-match ahead of
    Starlette's ``redirect_slashes``, so a client whose base URL ends in ``/``
    would otherwise miss the route. Same reason the protocol router factory
    registers trailing-slash aliases.
    """
    if path.endswith("/"):
        return [path]
    return [path, f"{path}/"]


router = APIRouter(tags=["batch"], dependencies=[Depends(require_any_auth)])


# ── Files ─────────────────────────────────────────────────────────────────


async def upload_file(
    request: Request,
    file: Annotated[UploadFile, File(description="JSONL batch input file")],
    purpose: Annotated[str, Form()] = "batch",
    session: AsyncSession = Depends(get_async_session),
) -> Any:
    """Upload a JSONL file the batch surface stores on the client's behalf."""
    content = await file.read()
    record = await _service(request, session).create_file(
        owner=owner_scope(request),
        filename=file.filename or "batch.jsonl",
        purpose=purpose,
        content=content,
    )
    return file_object(record)


async def list_files(
    request: Request,
    limit: int = Query(DEFAULT_LIST_LIMIT, ge=1, le=MAX_LIST_LIMIT),
    after: str | None = Query(None),
    purpose: str | None = Query(None),
    session: AsyncSession = Depends(get_async_session),
) -> Any:
    """List stored files, newest first."""
    records, has_more = await _service(request, session).list_files(
        owner=owner_scope(request), limit=limit, after=after, purpose=purpose
    )
    return list_object([file_object(record) for record in records], has_more=has_more)


async def retrieve_file(
    request: Request,
    file_id: str,
    session: AsyncSession = Depends(get_async_session),
) -> Any:
    """Read one stored file's metadata."""
    record = await _service(request, session).get_file(owner=owner_scope(request), file_id=file_id)
    return file_object(record)


async def download_file(
    request: Request,
    file_id: str,
    session: AsyncSession = Depends(get_async_session),
) -> Response:
    """Download a file's content (an input JSONL or a materialized result)."""
    filename, content = await _service(request, session).get_file_content(
        owner=owner_scope(request), file_id=file_id
    )
    return Response(
        content=content,
        media_type="application/jsonl",
        headers={"Content-Disposition": _content_disposition(filename)},
    )


async def delete_file(
    request: Request,
    file_id: str,
    session: AsyncSession = Depends(get_async_session),
) -> Any:
    """Delete a stored file. Answers 404 for an id this key does not own."""
    service = _service(request, session)
    deleted = await service.delete_file(owner=owner_scope(request), file_id=file_id)
    if not deleted:
        # Raise the same 404 the retrieve path does rather than invent a second shape.
        await service.get_file(owner=owner_scope(request), file_id=file_id)
    return deleted_file_object(file_id)


# ── Batches ───────────────────────────────────────────────────────────────


async def create_batch(
    request: Request,
    body: BatchCreateRequest,
    session: AsyncSession = Depends(get_async_session),
) -> Any:
    """Translate a stored input file and submit it to the resolved upstream."""
    job = await _service(request, session).create_batch(
        owner=owner_scope(request),
        input_file_id=body.input_file_id,
        endpoint=body.endpoint,
        completion_window=body.completion_window,
        metadata=body.metadata,
        allowed_models=facts_for(request).allowed_models,
        user_id=get_request_identity(request).user_id,
    )
    return batch_object(job)


async def list_batches(
    request: Request,
    limit: int = Query(DEFAULT_LIST_LIMIT, ge=1, le=MAX_LIST_LIMIT),
    after: str | None = Query(None),
    session: AsyncSession = Depends(get_async_session),
) -> Any:
    """List batches, newest first. Reports each job's last observed state."""
    jobs, has_more = await _service(request, session).list_batches(
        owner=owner_scope(request), limit=limit, after=after
    )
    return list_object([batch_object(job) for job in jobs], has_more=has_more)


async def retrieve_batch(
    request: Request,
    batch_id: str,
    session: AsyncSession = Depends(get_async_session),
) -> Any:
    """Read a batch, refreshing it from the upstream while it is unfinished."""
    job = await _service(request, session).get_batch(owner=owner_scope(request), batch_id=batch_id)
    return batch_object(job)


async def cancel_batch(
    request: Request,
    batch_id: str,
    session: AsyncSession = Depends(get_async_session),
) -> Any:
    """Ask the upstream to cancel a batch."""
    job = await _service(request, session).cancel_batch(
        owner=owner_scope(request), batch_id=batch_id
    )
    return batch_object(job)


# ── Registration ──────────────────────────────────────────────────────────
# Paths are registered explicitly (instead of with decorators) so each route also
# gets its trailing-slash variant, with only the canonical path in the schema.


def _register(
    path: str,
    handler: Any,
    methods: list[str],
    *,
    route_name: str,
    response_model: Any = None,
) -> None:
    for variant in _path_variants(path):
        router.add_api_route(
            variant,
            handler,
            methods=methods,
            response_model=response_model,
            name=f"{route_name}{'_slash' if variant != path else ''}",
            include_in_schema=variant == path,
        )


_register("/v1/files", upload_file, ["POST"], route_name="upload_file", response_model=FileObject)
_register("/v1/files", list_files, ["GET"], route_name="list_files", response_model=FileList)
_register(
    "/v1/files/{file_id}",
    retrieve_file,
    ["GET"],
    route_name="retrieve_file",
    response_model=FileObject,
)
_register(
    "/v1/files/{file_id}",
    delete_file,
    ["DELETE"],
    route_name="delete_file",
    response_model=DeletedFileObject,
)
_register("/v1/files/{file_id}/content", download_file, ["GET"], route_name="download_file")

_register(
    "/v1/batches", create_batch, ["POST"], route_name="create_batch", response_model=BatchObject
)
_register("/v1/batches", list_batches, ["GET"], route_name="list_batches", response_model=BatchList)
_register(
    "/v1/batches/{batch_id}",
    retrieve_batch,
    ["GET"],
    route_name="retrieve_batch",
    response_model=BatchObject,
)
_register(
    "/v1/batches/{batch_id}/cancel",
    cancel_batch,
    ["POST"],
    route_name="cancel_batch",
    response_model=BatchObject,
)


__all__ = ["owner_scope", "require_any_auth", "router"]
