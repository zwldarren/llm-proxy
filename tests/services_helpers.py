"""The in-memory test adapter for :mod:`llm_proxy.services`.

``RuntimeServices`` reads the process-lifetime services off an
``app.state``-shaped object by name. A test that exercises one consumer usually
means to provide exactly one or two of those services and nothing else, so this
module builds a view over a plain :class:`types.SimpleNamespace`: no FastAPI app,
no lifespan, no state attribute leaking in from a fixture.

    services = services_for(config_manager=manager)
    assert services.circuit_breaker() is None  # present-but-absent, not a Mock

Prefer this over handing a ``MagicMock`` state to ``RuntimeServices``: a mock
answers every attribute with another mock, which silently erases the
present/absent distinction these accessors exist to express.
"""

from types import SimpleNamespace
from typing import Any

from llm_proxy.services import RuntimeServices


def services_for(**slots: Any) -> RuntimeServices:
    """Return a services view over a fresh state carrying ``slots``."""
    return RuntimeServices(SimpleNamespace(**slots))
