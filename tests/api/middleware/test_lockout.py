"""Tests for the in-memory lockout managers' enable/disable gate.

Login lockout is off by default (``security.login_lockout_enabled``); the
gating lives on :class:`BaseLockoutManager` so no call site can forget it.
"""

from llm_proxy.api.middleware.security import BaseLockoutManager


def test_lockout_enabled_by_default():
    """A manager without an ``enabled_getter`` keeps locking out."""
    manager = BaseLockoutManager(max_attempts=2, lockout_duration=60)

    assert manager.enabled is True
    manager.record_failed_attempt("alice")
    manager.record_failed_attempt("alice")
    assert manager.is_locked_out("alice") is True


def test_disabled_lockout_records_nothing_and_never_locks():
    manager = BaseLockoutManager(max_attempts=2, lockout_duration=60, enabled_getter=lambda: False)

    for _ in range(10):
        manager.record_failed_attempt("alice")

    assert manager.is_locked_out("alice") is False
    assert manager.get_lockout_remaining("alice") == 0


def test_lockout_toggle_is_read_per_call():
    """Flipping the getter takes effect immediately (hot-reloadable)."""
    enabled = {"value": True}
    manager = BaseLockoutManager(
        max_attempts=2, lockout_duration=60, enabled_getter=lambda: enabled["value"]
    )

    manager.record_failed_attempt("bob")
    manager.record_failed_attempt("bob")
    assert manager.is_locked_out("bob") is True

    enabled["value"] = False
    assert manager.is_locked_out("bob") is False
    assert manager.get_lockout_remaining("bob") == 0
    # Disabling must not erase state: re-enabling resumes the existing lockout.
    enabled["value"] = True
    assert manager.is_locked_out("bob") is True
