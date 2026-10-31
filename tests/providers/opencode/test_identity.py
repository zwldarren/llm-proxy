"""The client-identity headers the edge expects — id format fidelity."""

from llm_proxy.providers.opencode import identity


def test_static_identity_headers():
    headers = identity.identity_headers()

    assert headers["User-Agent"] == identity.OPENCODE_USER_AGENT
    assert headers["User-Agent"].startswith("opencode/")
    assert headers["x-opencode-client"] == "cli"
    assert headers["x-opencode-project"] == "global"


def test_id_shapes():
    headers = identity.identity_headers()

    session, request = headers["x-opencode-session"], headers["x-opencode-request"]
    # {prefix}_{12 hex}{14 base62}
    assert session.startswith("ses_") and len(session) == 4 + 12 + 14
    assert request.startswith("msg_") and len(request) == 4 + 12 + 14
    assert all(c in identity._ID_CHARS for c in session[4:])
    assert all(c in identity._ID_CHARS for c in request[4:])


def test_session_time_is_descending_and_lags():
    """Sessions sort newest-first and are minted ~3s in the past.

    Only the slot's low 48 bits are encoded, so the timestamp is recovered
    modulo 2^36 (bits above the 12-bit counter, within the hex field).
    """
    headers = identity.identity_headers()

    def recover(hex12: str, *, descending: bool) -> int:
        value = int(hex12, 16)
        if descending:
            value = ~value & identity._TIME_MASK
        return (value >> identity._COUNTER_BITS) & ((1 << 36) - 1)

    session_time = recover(headers["x-opencode-session"][4:16], descending=True)
    request_time = recover(headers["x-opencode-request"][4:16], descending=False)
    assert (request_time - session_time) % (1 << 36) == identity._SESSION_LAG_MS


def test_ids_are_unique_across_calls():
    first = identity.identity_headers()
    second = identity.identity_headers()

    assert first["x-opencode-session"] != second["x-opencode-session"]
    assert first["x-opencode-request"] != second["x-opencode-request"]


def test_resolved_session_is_preserved():
    headers = identity.identity_headers(session_id="ses_conversation")

    assert headers["x-opencode-session"] == "ses_conversation"
    # The request id is always per request; a client's value is not reused.
    assert headers["x-opencode-request"].startswith("msg_")
    assert headers["x-opencode-request"] != "ses_conversation"


def test_user_agent_is_overridable():
    headers = identity.identity_headers(user_agent="llm-proxy/0.5.2")

    assert headers["User-Agent"] == "llm-proxy/0.5.2"


def test_llm_proxy_user_agent_identifies_the_proxy():
    user_agent = identity.llm_proxy_user_agent()

    assert user_agent.startswith("llm-proxy/")
    assert not user_agent.startswith("opencode/")


def test_seeded_session_is_deterministic_and_shaped():
    first = identity.session_id_from_seed("conversation")
    second = identity.session_id_from_seed("conversation")

    assert first == second
    assert first.startswith("ses_")
    assert len(first) == 4 + 12 + 14
    assert all(c in identity._ID_CHARS for c in first[4:])
    assert identity.session_id_from_seed("another") != first


def test_counter_splits_per_millisecond():
    """Two ids minted in the same millisecond get consecutive counters."""
    now = 1_700_000_000_000
    a = identity._id("msg", now, descending=False)
    b = identity._id("msg", now, descending=False)

    assert int(a[4:16], 16) + 1 == int(b[4:16], 16)
