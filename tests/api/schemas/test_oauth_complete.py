"""Validation for the OAuth invitation completion payload."""

import pytest
from pydantic import ValidationError

from llm_proxy.api.schemas.oauth import OAuthCompleteRequest


def test_allows_no_username_or_password() -> None:
    request = OAuthCompleteRequest(completion_token="token")
    assert request.username is None
    assert request.password is None


def test_normalizes_username_and_rejects_invalid_characters() -> None:
    assert OAuthCompleteRequest(completion_token="token", username="Alice_1").username == "Alice_1"

    with pytest.raises(ValidationError):
        OAuthCompleteRequest(completion_token="token", username="alice@example.com")


def test_rejects_weak_optional_password() -> None:
    with pytest.raises(ValidationError):
        OAuthCompleteRequest(completion_token="token", password="alllowercase1!")
