"""Error envelopes follow the client protocol — including base_url aliases.

Errors on the Anthropic Messages path (canonical or alias) must use the
Anthropic envelope so Claude Code parses them natively; every other API path
uses the OpenAI envelope.
"""

import pytest
from starlette.requests import Request

from llm_proxy.api.middleware.exceptions import protocol_for_request


def _make_request(path: str) -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "query_string": b"",
        "headers": [],
        "scheme": "http",
        "server": ("testserver", 80),
    }
    return Request(scope)


@pytest.fixture(autouse=True, scope="module")
def _registered_protocols() -> None:
    """Populate the protocol registry so alias paths resolve to a protocol."""
    from llm_proxy.api.routers.protocol import import_registered_protocol_modules

    import_registered_protocol_modules()


@pytest.mark.parametrize(
    "path",
    [
        "/v1/messages",
        "/messages",
        "/v1/v1/messages",
        "/v1/messages/count_tokens",
        "/messages/count_tokens",
        "/v1/v1/messages/count_tokens",
    ],
)
def test_anthropic_paths_use_anthropic_envelope(path: str) -> None:
    assert protocol_for_request(_make_request(path)) == "anthropic"


@pytest.mark.parametrize(
    "path",
    [
        "/v1/chat/completions",
        "/chat/completions",
        "/v1/v1/chat/completions",
        "/v1/responses",
        "/responses",
    ],
)
def test_other_api_paths_use_openai_envelope(path: str) -> None:
    assert protocol_for_request(_make_request(path)) == "openai"
