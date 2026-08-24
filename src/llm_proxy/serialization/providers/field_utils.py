"""Shared field extraction utilities for provider serializers.

Consolidates the ``_known_*_fields`` / ``_extract_extra_fields`` pattern that was
duplicated across mixins and serializers into a single place.
"""

from typing import Any


def extract_extra_fields(data: dict[str, Any], known_fields: set[str]) -> dict[str, Any]:
    """Return fields from *data* that are not in *known_fields*.

    Fields not explicitly handled by a serializer are candidates for passthrough
    and are stored in ``InternalRequest.extra``.
    """
    return {k: v for k, v in data.items() if k not in known_fields}


def extract_unknown_response_fields(
    response: dict[str, Any], known_fields: set[str]
) -> dict[str, Any]:
    """Return fields from *response* that are not in *known_fields*.

    Unknown fields are preserved in ``InternalResponse.provider_info`` so
    provider-specific metadata (e.g. OpenRouter *id_provider*, *cost*) is not
    silently dropped.
    """
    if not known_fields:
        return {}
    return {k: v for k, v in response.items() if k not in known_fields}


__all__ = [
    "extract_extra_fields",
    "extract_unknown_response_fields",
]
