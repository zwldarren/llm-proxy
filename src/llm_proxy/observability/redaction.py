"""Single choke point for scrubbing content out of stored request logs.

The ``logging.log_input_output`` setting is the master switch for body
logging. It used to be enforced by each writer individually, which is exactly
the kind of invariant that rots: the endpoint/audit handler scrubbed its
``request_body``/``response_body``, but upstream error bodies, MCP tool
arguments/results and web-search payloads kept landing in ``log_metadata``,
and a writer added later could forget the switch entirely.

Instead, the log store (:func:`llm_proxy.observability.service
._request_log_from_create`) calls :func:`scrub_log_metadata` for every row it
builds. A writer may put whatever it wants on ``RequestLogCreate``; the store
decides what is persisted, so no writer can bypass the switch.

Two markers are produced so the Logs UI can explain *why* a body is absent:

- ``_bodies_disabled`` — the operator turned body logging off.
- ``_sampled_out`` — body logging is on, but this request lost the
  ``sampling_rate`` draw (or, for streaming, was not captured).

Scope note: the switch covers bodies and content embedded in ``log_metadata``.
Masked request/response headers are deliberately retained — they are diagnostic
metadata, not free-form content. The scalar ``error_message`` /
``error_stack_trace`` columns are kept on purpose — they are the diagnostic
reason (rate limit, budget, upstream status) that rejection rows and operators
rely on; the content-bearing upstream error text lives in
``log_metadata["error_details"]`` and is scrubbed.
"""

from typing import Any

#: Stored in place of a body when the master switch is off. Distinct from the
#: sampling sentinel so the UI never claims a body was "sampled out" when the
#: operator simply disabled body logging.
BODIES_DISABLED_MARKER: dict[str, Any] = {"_bodies_disabled": True}

#: Stored in place of a body that lost the ``sampling_rate`` draw.
SAMPLED_OUT_MARKER: dict[str, Any] = {"_sampled_out": True}

#: ``log_metadata`` keys whose values carry user or upstream content. They are
#: stripped whenever body logging is off: MCP tool arguments and results,
#: web-search queries and results. Metadata that is not content (server name,
#: operation, counts, statuses, hashes) is left alone.
CONTENT_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "mcp_arguments",
        "mcp_result_summary",
        "web_search_query",
        "web_search_results",
    }
)

#: Keys inside ``log_metadata["error_details"]`` that carry the upstream
#: response payload or its message. Classification fields (error type, status
#: code, url, method, …) are kept so the row stays diagnosable.
_ERROR_DETAILS_CONTENT_KEYS: frozenset[str] = frozenset(
    {
        "response_body",
        "original_error",
        # ``HTTPException.detail`` is a free-form message and can echo the
        # offending input; treat it as content like the upstream body.
        "detail",
    }
)

#: Keys inside each entry of ``log_metadata["fallback_attempts"]`` /
#: ``log_metadata["retry_attempts"]`` that carry the upstream error text. The
#: classification fields (provider, attempt, error_type, status_code, retried)
#: are kept so the retry/fallback timeline stays readable.
_ATTEMPT_ENTRY_CONTENT_KEYS: frozenset[str] = frozenset({"error_message"})

#: Attempt lists written by the audit handler (per failed fallback provider and
#: per same-provider retry).
_ATTEMPT_LIST_KEYS: frozenset[str] = frozenset({"fallback_attempts", "retry_attempts"})


def _scrub_attempt_entries(entries: Any) -> Any:
    """Replace the error text inside recorded fallback/retry attempts."""
    if not isinstance(entries, list):
        return entries
    scrubbed: list[Any] = []
    for entry in entries:
        if not isinstance(entry, dict):
            scrubbed.append(entry)
            continue
        scrubbed.append(
            {
                key: dict(BODIES_DISABLED_MARKER) if key in _ATTEMPT_ENTRY_CONTENT_KEYS else value
                for key, value in entry.items()
            }
        )
    return scrubbed


def body_marker(*, bodies_enabled: bool) -> dict[str, Any]:
    """Return the marker stored in place of an unavailable body.

    Args:
        bodies_enabled: Whether the ``log_input_output`` master switch is on.

    Returns:
        ``{"_bodies_disabled": True}`` when the switch is off, otherwise the
        sampling sentinel ``{"_sampled_out": True}``. A fresh dict each call so
        callers can never mutate a shared constant.
    """
    return dict(SAMPLED_OUT_MARKER if bodies_enabled else BODIES_DISABLED_MARKER)


def scrub_log_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Strip content-bearing values from ``log_metadata``.

    Returns a dict with content values replaced; the input (often the live
    ``EventContext.metadata``) is never mutated. Content keys are replaced by
    :data:`BODIES_DISABLED_MARKER` so the row keeps a visible, explained
    placeholder instead of silently losing the key. Covered: MCP
    arguments/results, web-search queries/results, ``error_details`` (upstream
    body, original error, message), and the ``error_message`` recorded inside
    each fallback/retry attempt.

    Rows with nothing to scrub return the input unchanged (no copy): this runs
    on every row while body logging is off, and most rows carry no content.
    """
    if not metadata:
        return {}

    content_keys = CONTENT_METADATA_KEYS.intersection(metadata)
    attempt_keys = _ATTEMPT_LIST_KEYS.intersection(metadata)
    error_details = metadata.get("error_details")
    error_detail_keys = (
        _ERROR_DETAILS_CONTENT_KEYS.intersection(error_details)
        if isinstance(error_details, dict)
        else frozenset()
    )
    if not content_keys and not attempt_keys and not error_detail_keys:
        return metadata

    scrubbed = dict(metadata)

    for key in content_keys:
        scrubbed[key] = dict(BODIES_DISABLED_MARKER)

    for key in attempt_keys:
        scrubbed[key] = _scrub_attempt_entries(scrubbed[key])

    if error_detail_keys and isinstance(error_details, dict):
        scrubbed["error_details"] = {
            **error_details,
            **{key: dict(BODIES_DISABLED_MARKER) for key in error_detail_keys},
        }

    return scrubbed


__all__ = [
    "BODIES_DISABLED_MARKER",
    "CONTENT_METADATA_KEYS",
    "SAMPLED_OUT_MARKER",
    "body_marker",
    "scrub_log_metadata",
]
