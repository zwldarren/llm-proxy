"""Tests for context variables."""

from unittest.mock import MagicMock

from llm_proxy.core.context import (
    get_config_manager,
    reset_context,
    set_config_manager,
)


class TestContextVariables:
    """Test suite for context variables."""

    def setup_method(self):
        """Reset context before each test."""
        reset_context()

    def teardown_method(self):
        """Reset context after each test."""
        reset_context()

    def test_get_config_manager_default(self):
        """Test getting config manager when not set."""
        result = get_config_manager()

        assert result is None

    def test_set_and_get_config_manager(self):
        """Test setting and getting config manager."""
        mock_manager = MagicMock()

        set_config_manager(mock_manager)
        result = get_config_manager()

        assert result is mock_manager

    def test_reset_context_clears(self):
        """Test that reset_context clears all variables."""
        mock_manager = MagicMock()
        set_config_manager(mock_manager)

        reset_context()

        assert get_config_manager() is None

    def test_context_overwrite(self):
        """Test overwriting context values."""
        mock_manager1 = MagicMock()
        mock_manager2 = MagicMock()

        set_config_manager(mock_manager1)
        assert get_config_manager() is mock_manager1

        set_config_manager(mock_manager2)
        assert get_config_manager() is mock_manager2
