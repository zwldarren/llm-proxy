"""Conversation compaction for the OpenResponses /v1/responses/compact endpoint.

The compaction endpoint returns a compacted conversation state object that can
be used to preserve long-running context without asserting provider-specific
compression behavior (spec 2026-04-24).

The proxy compacts by serializing the full item list (previous input + previous
output + new input) into an opaque blob carried by a ``compaction`` item. The
blob is prefixed with a marker so the proxy can recognize and rehydrate its own
compaction items on a follow-up request, while foreign (e.g. Codex) compaction
blobs keep passing through as opaque encrypted content.
"""

import json
import secrets
import time
from typing import Any

_COMPACTION_MARKER = "llm-proxy-compaction:v1:"


def encode_compaction_blob(items: list[dict[str, Any]]) -> str:
    """Encode a list of conversation items into an opaque compaction blob."""
    payload = {"items": items}
    return _COMPACTION_MARKER + json.dumps(payload, separators=(",", ":"))


def decode_compaction_blob(blob: str) -> list[dict[str, Any]] | None:
    """Decode a proxy-produced compaction blob back into conversation items.

    Returns None when the blob is not a proxy-produced compaction blob (e.g. a
    Codex encrypted compaction payload), so callers can fall back to treating
    it as opaque encrypted content.
    """
    if not isinstance(blob, str) or not blob.startswith(_COMPACTION_MARKER):
        return None
    try:
        payload = json.loads(blob[len(_COMPACTION_MARKER) :])
        items = payload.get("items")
        return items if isinstance(items, list) else None
    except ValueError, TypeError:
        return None


def build_compaction_response(
    model: str,
    items: list[dict[str, Any]],
    *,
    response_id: str | None = None,
    created_at: int | None = None,
) -> dict[str, Any]:
    """Build a ``response.compaction`` response body.

    Args:
        model: The model the compaction is for.
        items: The full conversation item list to compact.
        response_id: Optional response id (generated when omitted).
        created_at: Optional Unix timestamp (generated when omitted).

    Returns:
        The response.compaction object per the OpenResponses reference.
    """
    compaction_item: dict[str, Any] = {
        "type": "compaction",
        "id": f"comp_{secrets.token_hex(12)}",
        "encrypted_content": encode_compaction_blob(items),
        "created_by": "llm-proxy",
    }
    return {
        "id": response_id or f"compaction_{secrets.token_hex(12)}",
        "object": "response.compaction",
        "output": [compaction_item],
        "created_at": created_at or int(time.time()),
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


__all__ = [
    "encode_compaction_blob",
    "decode_compaction_blob",
    "build_compaction_response",
]
