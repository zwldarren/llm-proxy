"""Invitation lifecycle decisions that do not need a database."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from llm_proxy.database.repositories.invitations import InvitationRepository


def _invitation(**overrides):
    now = datetime.now(UTC)
    values = {
        "accepted_at": None,
        "revoked_at": None,
        "expires_at": now + timedelta(days=1),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_pending_before_expiry() -> None:
    assert InvitationRepository.status(_invitation()) == "pending"


def test_expired_after_expiry() -> None:
    record = _invitation(expires_at=datetime.now(UTC) - timedelta(seconds=1))
    assert InvitationRepository.status(record) == "expired"


def test_acceptance_and_revocation_take_precedence() -> None:
    accepted = _invitation(accepted_at=datetime.now(UTC))
    revoked = _invitation(revoked_at=datetime.now(UTC))
    assert InvitationRepository.status(accepted) == "accepted"
    assert InvitationRepository.status(revoked) == "revoked"


def test_naive_expiry_is_treated_as_utc() -> None:
    record = _invitation(expires_at=datetime.now() + timedelta(days=1))
    assert InvitationRepository.status(record) == "pending"
