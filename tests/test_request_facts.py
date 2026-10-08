"""The request-facts record: one slot, typed fields, created on first use.

These tests pin the carrier's contract, not any one consumer: the record is a
single ``request.state`` slot that every reader reaches through ``facts_for``,
it is created by whoever touches it first (the logging middleware in
production, the first reader in a test), and the string keys it replaced are
inert.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from starlette.requests import Request

from llm_proxy.core.request_facts import (
    RequestFacts,
    RequestIdentity,
    facts_for,
    facts_from_scope,
    get_request_identity,
    mint_request_id,
    set_request_identity,
)


def _receive():
    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    return receive


def _request() -> Request:
    return Request({"type": "http", "method": "GET", "path": "/v1/models", "state": {}}, _receive())


class TestTheCarrier:
    """One slot, one record, whoever asks."""

    def test_one_record_per_request(self):
        request = _request()

        assert facts_for(request) is facts_for(request)
        assert request.state.request_facts is facts_for(request)

    def test_a_scope_written_record_is_the_one_a_request_reads(self):
        """A pure-ASGI middleware and request-bound code share the record."""
        scope = {"type": "http", "method": "GET", "path": "/v1/models", "state": {}}
        from_scope = facts_from_scope(scope)
        from_scope.request_id = "req-1"

        request = Request(scope, _receive())

        assert facts_for(request) is from_scope
        assert facts_for(request).request_id == "req-1"

    def test_two_requests_get_two_records(self):
        assert facts_for(_request()).request_id != facts_for(_request()).request_id

    def test_a_raw_state_attribute_is_not_the_carrier(self):
        """The point of the record: no second, silently-ignored channel."""
        request = _request()
        request.state.request_id = "hand-written"

        assert facts_for(request).request_id != "hand-written"

    def test_a_test_double_still_gets_a_record(self):
        """Doubles answer with mocks for every name, so the record is created
        explicitly instead of being read back as a mock."""
        for request in (MagicMock(), SimpleNamespace(state=SimpleNamespace())):
            facts = facts_for(request)
            assert isinstance(facts, RequestFacts)
            assert facts_for(request) is facts

    def test_a_bare_scope_gets_its_state_dict(self):
        scope: dict = {"type": "http"}

        assert isinstance(facts_from_scope(scope), RequestFacts)
        assert scope["state"]["request_facts"] is facts_from_scope(scope)


class TestRequestId:
    """The id is minted once, here, and never re-derived."""

    def test_the_id_is_minted_once_and_kept(self):
        request = _request()
        request_id = facts_for(request).request_id

        assert request_id
        assert facts_for(request).request_id == request_id

    def test_mint_request_id_is_opaque_and_unique(self):
        assert len(mint_request_id()) == 32  # uuid4 hex
        assert mint_request_id("ws_").startswith("ws_")
        assert mint_request_id() != mint_request_id()


class TestIdentity:
    """The identity is a field of the record, reached through its own accessor."""

    def test_defaults_to_unauthenticated(self):
        identity = get_request_identity(_request())

        assert isinstance(identity, RequestIdentity)
        assert identity.is_authenticated is False
        assert identity.display_name is None

    def test_round_trips_through_the_accessor(self):
        request = _request()
        identity = RequestIdentity(user="alice", user_id=7, auth_method="jwt")

        set_request_identity(request, identity)

        assert get_request_identity(request) is identity
        assert facts_for(request).identity is identity
        assert identity.is_authenticated is True
        assert identity.display_name == "alice"

    def test_an_api_key_identity_displays_the_key_name(self):
        identity = RequestIdentity(api_key_name="sk-1", auth_method="api_key")

        assert identity.is_authenticated is True
        assert identity.display_name == "sk-1"


class TestDefaults:
    """Absent facts are typed defaults, not missing attributes."""

    def test_the_latches_and_allowlist_start_empty(self):
        facts = RequestFacts()

        assert facts.allowed_models is None  # None = unrestricted, [] = deny-all
        assert facts.client_disconnected is False
        assert facts.audit_log_written is False
        assert facts.model is None
        assert facts.provider is None
        assert facts.session_id is None
        assert facts.parsed_request_body is None
        assert facts.request_body is None
        assert facts.response_body is None

    def test_mutable_defaults_are_per_record(self):
        first, second = RequestFacts(), RequestFacts()

        first.request_headers["x-foo"] = "bar"
        first.response_headers["x-bar"] = "foo"

        assert second.request_headers == {}
        assert second.response_headers == {}
