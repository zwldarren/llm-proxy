"""OpenAI wire objects for the batch surface.

The client dialect is OpenAI's, so the rendering is a straight projection of the
stored rows — no protocol serializer is involved: a batch object and a file
object are bookkeeping shapes, not model output. Field names, null-ness and the
``object`` discriminator follow OpenAI's API reference so the official SDKs and
``openai``-compatible clients parse them unchanged.
"""

from typing import Any

from llm_proxy.batch.types import BATCH_TIMESTAMP_FIELDS
from llm_proxy.database.tables import BatchFileRecord, BatchJobRecord


def file_object(record: BatchFileRecord) -> dict[str, Any]:
    """Render a stored file as an OpenAI FileObject."""
    return {
        "id": record.id,
        "object": "file",
        "bytes": record.bytes,
        "created_at": record.created_at,
        "filename": record.filename,
        "purpose": record.purpose,
    }


def deleted_file_object(file_id: str) -> dict[str, Any]:
    """Render a deleted file as OpenAI's deletion acknowledgement."""
    return {"id": file_id, "object": "file", "deleted": True}


def batch_object(job: BatchJobRecord) -> dict[str, Any]:
    """Render a stored job as an OpenAI Batch object.

    ``errors`` keeps OpenAI's envelope (``{object: "list", data: [...]}``) rather
    than a bare array, because that is what the schema declares and what SDK
    clients type against.
    """
    counts = job.request_counts or {}
    body: dict[str, Any] = {
        "id": job.id,
        "object": "batch",
        "endpoint": job.endpoint,
        "errors": ({"object": "list", "data": job.errors} if job.errors else None),
        "input_file_id": job.input_file_id,
        "completion_window": job.completion_window,
        "status": job.status,
        "output_file_id": job.output_file_id,
        "error_file_id": job.error_file_id,
        "created_at": job.created_at,
        "request_counts": {
            "total": int(counts.get("total") or 0),
            "completed": int(counts.get("completed") or 0),
            "failed": int(counts.get("failed") or 0),
        },
        "metadata": job.batch_metadata or {},
    }
    for field in BATCH_TIMESTAMP_FIELDS:
        body[field] = getattr(job, field, None)
    body["expires_at"] = job.expires_at
    return body


def list_object(items: list[dict[str, Any]], *, has_more: bool = False) -> dict[str, Any]:
    """Render a page as OpenAI's list envelope (newest first, id cursor)."""
    return {
        "object": "list",
        "data": items,
        "first_id": items[0]["id"] if items else None,
        "last_id": items[-1]["id"] if items else None,
        "has_more": has_more,
    }


__all__ = ["batch_object", "deleted_file_object", "file_object", "list_object"]
