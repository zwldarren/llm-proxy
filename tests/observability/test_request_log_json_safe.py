"""Tests for JSON-safe sanitization of request log payloads.

Multipart upload endpoints (``/v1/images/edits``, ``/v1/audio/*``) capture
raw file bytes in the request body. Bytes are not JSON serializable, so an
unsanitized log row makes the background writer fail with
``TypeError: Object of type bytes is not JSON serializable`` and eventually
trip its circuit breaker. These tests pin the sanitization at the DB
boundary (``_request_log_from_create``).
"""

import json

import pytest

from llm_proxy.observability.service import (
    RequestLogCreate,
    _json_safe,
    _request_log_from_create,
)
from llm_proxy.observability.types import LogType


class TestJsonSafe:
    """Tests for the recursive JSON-safe sanitizer."""

    def test_bytes_replaced_with_placeholder(self):
        """Top-level bytes become a size-carrying placeholder."""
        assert _json_safe(b"\x89PNG\r\n") == {"$binary": True, "size": 6}
        assert _json_safe(bytearray(b"abc")) == {"$binary": True, "size": 3}

    def test_nested_bytes_and_bytearrays(self):
        value = {
            "images": [
                {"file": b"\x00\x01", "filename": "a.png", "content_type": "image/png"},
                {"file": bytearray(b"\x02")},
            ],
            "mask": {"file": b"\xff", "filename": "m.png"},
            "model": "gpt-4o",
            "nested": {"list": [1, b"raw", "text"]},
            "plain": "kept",
        }
        result = _json_safe(value)
        assert result["images"][0]["file"] == {"$binary": True, "size": 2}
        assert result["images"][0]["filename"] == "a.png"
        assert result["images"][1]["file"] == {"$binary": True, "size": 1}
        assert result["mask"]["file"] == {"$binary": True, "size": 1}
        assert result["nested"]["list"][1] == {"$binary": True, "size": 3}
        assert result["plain"] == "kept"
        # The whole result must round-trip through the JSON serializer.
        json.dumps(result)

    def test_non_binary_values_untouched(self):
        value = {"a": 1, "b": "text", "c": None, "d": [True, 1.5, {"e": "f"}]}
        assert _json_safe(value) == value
        assert _json_safe("string") == "string"
        assert _json_safe(None) is None
        assert _json_safe(42) == 42

    @pytest.mark.parametrize(
        "value",
        [
            b"",
            b"\x00\x01\x02",
            {"file": b"x" * 1000},
            [b"a", {"nested": [bytearray(b"b")]}],
        ],
    )
    def test_always_json_serializable(self, value):
        json.dumps(_json_safe(value))


class TestRequestLogFromCreate:
    """The DB boundary must never receive bytes in JSON columns."""

    def _create(self, **overrides) -> RequestLogCreate:
        fields: dict = {
            "request_id": "req-1",
            "timestamp": 1.0,
            "endpoint": "/v1/images/edits",
            "method": "POST",
            "status_code": 200,
            "response_time_ms": 10,
            "log_type": LogType.ENDPOINT,
        }
        fields.update(overrides)
        return RequestLogCreate(**fields)

    def test_binary_request_body_is_sanitized(self):
        data = self._create(
            request_headers={"content-type": "multipart/form-data"},
            request_body={
                "model": "gpt-4o",
                "images": [{"file": b"\x89PNG", "filename": "a.png"}],
                "mask": {"file": b"\x00\x01", "filename": "mask.png"},
            },
            response_body={"data": []},
            log_metadata={"streaming": False},
        )
        log = _request_log_from_create(data)
        json.dumps(log.request_body)  # must not raise
        assert log.request_body["images"][0]["file"] == {"$binary": True, "size": 4}
        assert log.request_body["mask"]["file"] == {"$binary": True, "size": 2}
        assert log.request_body["images"][0]["filename"] == "a.png"

    def test_binary_response_body_is_sanitized(self):
        data = self._create(
            request_body={"model": "gpt-4o"},
            response_body={"image": b"\x89PNG" * 10},
        )
        log = _request_log_from_create(data)
        json.dumps(log.response_body)  # must not raise
        assert log.response_body["image"] == {"$binary": True, "size": 40}

    def test_all_json_columns_serializable(self):
        data = self._create(
            request_headers={"X-Bin": b"\x01\x02"},
            request_body={"file": b"\x01"},
            response_headers={"X-Other": b"\x03"},
            response_body={"data": bytearray(b"\x04")},
            log_metadata={"tool_args": {"blob": b"\x05"}},
        )
        log = _request_log_from_create(data)
        for column in (
            log.request_headers,
            log.request_body,
            log.response_headers,
            log.response_body,
            log.log_metadata,
        ):
            json.dumps(column)  # must not raise
