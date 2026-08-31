"""Shared outbound-header merge helpers for provider adapters."""

from collections.abc import Mapping


def merge_passthrough_headers(headers: dict[str, str], client_headers: Mapping[str, str]) -> None:
    """Merge captured client headers into outbound headers, in place.

    Existing keys are never overridden (the adapter's own defaults win);
    matching is case-insensitive, mirroring HTTP header semantics. Callers
    with explicit override semantics (e.g. ``anthropic-version``) re-apply
    them after this merge.
    """
    existing = {k.lower() for k in headers}
    for key, value in client_headers.items():
        if key.lower() not in existing:
            headers[key] = value
