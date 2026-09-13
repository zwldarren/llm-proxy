"""Tests for core utility helpers (api/routers/logs.py and billing rely on both)."""

import pytest

from llm_proxy.core.utils import coerce_float, safe_float


class TestCoerceFloat:
    """coerce_float distinguishes "unset" from a real zero."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (1.5, 1.5),
            (2, 2.0),
            ("3.25", 3.25),
            (0, 0.0),
            (None, None),
            ("", None),
            ("not-a-number", None),
            ([], None),
        ],
    )
    def test_coerces_or_returns_none(self, value, expected):
        assert coerce_float(value) == expected


class TestSafeFloat:
    """safe_float keeps its fallback contract on top of coerce_float."""

    @pytest.mark.parametrize(
        ("value", "default", "expected"),
        [
            (1.5, 0.0, 1.5),
            ("2.5", 0.0, 2.5),
            (0, 0.0, 0.0),
            (None, 0.0, 0.0),
            (None, 7.5, 7.5),
            ("junk", 0.0, 0.0),
            ("junk", 7.5, 7.5),
            ({"a": 1}, 3.0, 3.0),
        ],
    )
    def test_falls_back_to_default(self, value, default, expected):
        assert safe_float(value, default) == expected
