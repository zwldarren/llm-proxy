"""The process-lifetime services seam: what absence means, and where it reads.

Pins the interface of :mod:`llm_proxy.services` (ADR-0021): which services are
required (absent ⇒ ``ConfigurationError`` naming the startup step), which are
optional (absent ⇒ ``None``, a documented state the caller degrades from), how a
services view resolves from an app, request, websocket or state, and that the
installers keep writing the ``app.state`` attribute names that test doubles and
older readers still use.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from services_helpers import services_for
from starlette.datastructures import State
from starlette.requests import Request
from starlette.websockets import WebSocket

from llm_proxy.core.exceptions import ConfigurationError
from llm_proxy.services import RuntimeServices, runtime_services


def _request(app: FastAPI, path: str = "/v1/chat/completions") -> Request:
    """A real Request whose ``app`` is ``app``."""
    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "query_string": b"",
        "headers": [],
        "scheme": "http",
        "server": ("testserver", 80),
        "client": ("127.0.0.1", 1234),
        "app": app,
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(scope, receive)


class TestRequiredServices:
    """A required service's absence is a startup bug, and it says so."""

    def test_a_missing_config_manager_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="Config manager not initialized"):
            services_for().config_manager()

    def test_a_missing_http_client_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="HTTP client not initialized"):
            services_for().http_client()

    def test_present_services_are_returned(self) -> None:
        manager = MagicMock()
        client = MagicMock()
        services = services_for(config_manager=manager, http_client=client)

        assert services.config_manager() is manager
        assert services.http_client() is client

    def test_the_tolerant_readers_return_none_instead(self) -> None:
        """``*_or_none`` is the documented-defaults path, not a silent fallback."""
        services = services_for()

        assert services.config_manager_or_none() is None
        assert services.http_client_or_none() is None


class TestOptionalServices:
    """An optional service's absence is a legal state: ``None``, never a raise."""

    def test_absent_services_are_none(self) -> None:
        services = services_for()

        assert services.redis() is None
        assert services.redis_client() is None
        assert services.circuit_breaker() is None
        assert services.provider_stats() is None
        assert services.mcp_manager() is None
        assert services.web_search_interceptor() is None
        assert services.protocol_processor("openai") is None

    def test_the_redis_reads_split_the_wrapper_from_the_client(self) -> None:
        """Health needs the connection wrapper; stores need the raw client."""
        raw = MagicMock()
        wrapper = MagicMock(client=raw)
        services = services_for(redis_client=wrapper)

        assert services.redis() is wrapper
        assert services.redis_client() is raw

    def test_an_installed_protocol_processor_is_returned_by_name(self) -> None:
        processor = MagicMock()
        services = services_for()

        services.install_protocol_processor("openai", processor)

        assert services.protocol_processor("openai") is processor
        assert services.protocol_processor("anthropic") is None


class TestOwnedSlots:
    """The mutable slots have one owner: this module."""

    def test_background_tasks_are_created_once_and_shared(self) -> None:
        services = services_for()

        first = services.background_tasks()
        first.add(AsyncMock())

        assert services.background_tasks() is first

    def test_the_judge_warmup_task_round_trips(self) -> None:
        services = services_for()
        task = MagicMock()

        assert services.judge_warmup_task() is None
        services.set_judge_warmup_task(task)
        assert services.judge_warmup_task() is task


class TestInstallers:
    """Installers are the only writers, and they keep the storage names stable."""

    def test_installers_write_the_documented_state_attributes(self) -> None:
        state = SimpleNamespace()
        services = RuntimeServices(state)
        config_manager = MagicMock()
        http_client = MagicMock()

        services.install_config_manager(config_manager)
        services.install_http_client(http_client)
        services.install_redis(None)
        services.install_circuit_breaker(MagicMock())
        services.install_provider_stats(MagicMock())
        services.install_mcp_manager(MagicMock())
        services.install_protocol_processor("openai", MagicMock())
        services.install_web_search_interceptor(None)

        # The names are the contract older readers and test doubles write directly.
        assert state.config_manager is config_manager
        assert state.http_client is http_client
        assert state.redis_client is None
        assert state.circuit_breaker is not None
        assert state.provider_stats is not None
        assert state.mcp_manager is not None
        assert state.protocol_processors["openai"] is not None
        assert state.web_search_interceptor is None

    def test_installing_redis_records_a_disabled_feature_as_none(self) -> None:
        """Installing ``None`` is how the lifespan records "Redis is disabled"."""
        services = services_for()

        services.install_redis(None)

        assert services.redis() is None
        assert services.redis_client() is None


class TestResolution:
    """One view, four carriers: the app is always the way in."""

    def test_resolves_a_request_app_state(self) -> None:
        app = FastAPI()
        manager = MagicMock()
        app.state.config_manager = manager

        assert runtime_services(_request(app)).config_manager() is manager

    def test_resolves_a_websocket_app_state(self) -> None:
        app = FastAPI()
        processor = MagicMock()
        app.state.protocol_processors = {"openresponses": processor}
        websocket = WebSocket(
            {
                "type": "websocket",
                "path": "/v1/responses",
                "query_string": b"",
                "headers": [],
                "scheme": "ws",
                "server": ("testserver", 80),
                "client": ("127.0.0.1", 1234),
                "app": app,
            },
            AsyncMock(),
            AsyncMock(),
        )

        assert runtime_services(websocket).protocol_processor("openresponses") is processor

    def test_resolves_an_app(self) -> None:
        app = FastAPI()
        app.state.provider_stats = MagicMock()

        assert runtime_services(app).provider_stats() is app.state.provider_stats

    def test_resolves_a_state_object(self) -> None:
        state = State()
        state.mcp_manager = MagicMock()

        assert runtime_services(state).mcp_manager() is state.mcp_manager

    def test_a_request_double_needs_an_explicit_app(self) -> None:
        """A request double says it is a request, and its request-scoped state is ignored.

        ``RuntimeServices`` reads ``.app`` first: a request carries the app, an app
        carries the state. A bare ``MagicMock`` answers every name with a mock of its
        own, so a request double has to assign ``.app`` explicitly — otherwise the
        view reads a service nobody installed. The request-scoped ``request.state``
        is deliberately not a services carrier (ADR-0021).
        """
        app = FastAPI()
        manager = MagicMock()
        app.state.config_manager = manager

        double = MagicMock()
        double.app = app
        double.state = SimpleNamespace(config_manager=MagicMock())

        assert runtime_services(double).config_manager() is manager
