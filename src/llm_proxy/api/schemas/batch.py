"""Schemas for the batch surface (``api/routers/batches.py``).

These are OpenAI wire shapes, not admin write schemas: they exist so the OpenAPI
document is accurate and FastAPI validates what a batch client sends. Response
models are read shapes only — the objects they describe are rendered by
``llm_proxy.batch.wire``.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

from llm_proxy.batch.types import BatchStatus


class FileObject(BaseModel):
    """OpenAI's FileObject."""

    id: str
    object: Literal["file"] = "file"
    bytes: int
    created_at: int
    filename: str
    purpose: str


class DeletedFileObject(BaseModel):
    """OpenAI's file deletion acknowledgement."""

    id: str
    object: Literal["file"] = "file"
    deleted: Literal[True] = True


class FileList(BaseModel):
    """OpenAI's list envelope for files."""

    object: Literal["list"] = "list"
    data: list[FileObject]
    first_id: str | None = None
    last_id: str | None = None
    has_more: bool = False


class BatchError(BaseModel):
    """One entry OpenAI's batch ``errors`` list carries."""

    code: str | None = None
    message: str | None = None
    param: str | None = None
    line: int | None = None


class BatchErrorList(BaseModel):
    """OpenAI's list envelope for a batch's own errors."""

    object: Literal["list"] = "list"
    data: list[BatchError]


class RequestCounts(BaseModel):
    """Per-batch progress counters."""

    total: int = 0
    completed: int = 0
    failed: int = 0


class BatchObject(BaseModel):
    """OpenAI's Batch object."""

    id: str
    object: Literal["batch"] = "batch"
    endpoint: str
    errors: BatchErrorList | None = None
    input_file_id: str
    completion_window: str
    status: BatchStatus
    output_file_id: str | None = None
    error_file_id: str | None = None
    created_at: int
    in_progress_at: int | None = None
    expires_at: int | None = None
    finalizing_at: int | None = None
    completed_at: int | None = None
    failed_at: int | None = None
    expired_at: int | None = None
    cancelling_at: int | None = None
    cancelled_at: int | None = None
    request_counts: RequestCounts = Field(default_factory=RequestCounts)
    metadata: dict[str, Any] | None = None


class BatchList(BaseModel):
    """OpenAI's list envelope for batches."""

    object: Literal["list"] = "list"
    data: list[BatchObject]
    first_id: str | None = None
    last_id: str | None = None
    has_more: bool = False


class BatchCreateRequest(BaseModel):
    """OpenAI's batch create body.

    ``endpoint`` and ``completion_window`` are plain strings here: the service
    rejects unsupported values with an OpenAI-shaped error that names the
    supported set, which a ``Literal`` schema cannot do (it would answer 422).
    """

    input_file_id: str = Field(..., description="Id of an uploaded JSONL file (purpose=batch)")
    endpoint: str = Field(..., description="The endpoint every line targets")
    completion_window: str = Field(default="24h", description="Only '24h' is accepted")
    metadata: dict[str, str] | None = Field(default=None, description="Client metadata")


__all__ = [
    "BatchCreateRequest",
    "BatchError",
    "BatchErrorList",
    "BatchList",
    "BatchObject",
    "DeletedFileObject",
    "FileList",
    "FileObject",
    "RequestCounts",
]
