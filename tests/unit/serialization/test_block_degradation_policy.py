"""Tests for should_degrade_block with UnsupportedBlockPolicy values.

Verifies that should_degrade_block correctly handles the three block-policy
values: 'drop' (skip), 'degrade' (to text), 'error' (raise).
"""

import pytest

from llm_proxy.core.exceptions import ProviderError
from llm_proxy.models import RedactedThinkingBlock
from llm_proxy.serialization._shared_degradation import should_degrade_block

_RTB = RedactedThinkingBlock(data="redacted")
_EMPTY = frozenset()
_SUPPORTED = frozenset({RedactedThinkingBlock})


class TestDropPolicy:
    """'drop' policy: unsupported blocks are silently skipped."""

    def test_drop_skips_unsupported_block(self):
        assert should_degrade_block("drop", _RTB, "p", supported_blocks=_EMPTY) is False

    def test_drop_keeps_supported_block(self):
        assert should_degrade_block("drop", _RTB, "p", supported_blocks=_SUPPORTED) is False


class TestDegradePolicy:
    """'degrade' policy: unsupported blocks are degraded to text."""

    def test_degrade_degrades_unsupported(self):
        assert should_degrade_block("degrade", _RTB, "p", supported_blocks=_EMPTY) is True

    def test_degrade_keeps_supported(self):
        assert should_degrade_block("degrade", _RTB, "p", supported_blocks=_SUPPORTED) is False


class TestErrorPolicy:
    """'error' policy: unsupported blocks raise ProviderError."""

    def test_error_raises_for_unsupported(self):
        with pytest.raises(ProviderError, match="does not support content block type"):
            should_degrade_block("error", _RTB, "p", supported_blocks=_EMPTY)

    def test_error_keeps_supported(self):
        assert should_degrade_block("error", _RTB, "p", supported_blocks=_SUPPORTED) is False


class TestBackwardCompatibility:
    """Old callers pass 'ignore' (from unknown_fields_policy) — must still work."""

    def test_ignore_skips_block(self):
        assert should_degrade_block("ignore", _RTB, "p", supported_blocks=_EMPTY) is False
