"""Shared fixtures for middleware tests.

Several middleware behaviours resolve the client protocol from the request
path (error envelope selection, model restriction, alias-path auth), which
requires the protocol modules to be imported before any path assertion runs.
"""

import pytest


@pytest.fixture(autouse=True, scope="module")
def _registered_protocols() -> None:
    """Populate the protocol registry so path-based protocol lookups resolve."""
    from llm_proxy.api.routers.protocol import import_registered_protocol_modules

    import_registered_protocol_modules()
