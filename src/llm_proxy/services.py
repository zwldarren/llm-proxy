"""The process-lifetime services, behind one typed interface.

Every subsystem that outlives a request is constructed once in
:mod:`llm_proxy.api.lifecycle` and stored on FastAPI's ``app.state`` bag: the
config manager, the pooled HTTP client, Redis, the circuit-breaker store, the
provider-stats store, the MCP manager, one processor per protocol, and the
web-search runtime. Consumers read them with
``getattr(request.app.state, "name", None)`` — 43 such probes across 17 modules —
which has three costs:

* **Every site re-derives the None rule.** ``None`` means three different things
  depending on the name: "the lifespan never ran" (a startup bug),
  "the feature is disabled" (a legal state), and "not built yet in this worker"
  (a lazy slot). A reader cannot tell them apart, and neither can a typo: a
  misspelled name is silently ``None`` forever.
* **Required services degrade silently.** The authentication middlewares used to
  dispatch a request unauthenticated when the config manager was missing, rather
  than refusing to serve.
* **Startup order has no locality.** Which service may be read by which layer,
  and which may still be absent when, is knowledge that lived in the reader.

This module is the single place that knows the attribute names and, for each
service, whether absence is a legal state:

* **Required** — absence means the lifespan did not run or its startup step
  failed. The accessor raises :class:`~llm_proxy.core.exceptions.ConfigurationError`
  with the step that is missing. Never a silent no-op.
* **Optional** — absence is a documented state (feature disabled, or a test
  adapter). The accessor returns ``None``, and the name says so where the
  distinction is not obvious.

``app.state`` stays the carrier: FastAPI's lifespan and existing test doubles
already write it, so this is a typed *view* over that bag rather than a
container of its own. ``RuntimeServices(state)`` accepts any object with
attributes, which is also the in-memory test adapter
(``RuntimeServices(State())``); the production adapter is the FastAPI app.

Two attributes are deliberately *not* owned here:

* ``app.state.limiter`` — slowapi's own contract; the library looks the limiter
  up by that exact name.
* ``web_search_interceptor`` — written by :mod:`llm_proxy.web_search.runtime`,
  which owns the rebuild-on-config-change invariant (and its
  "never built in this worker" sentinel). This module owns the read and the
  install; the snapshot the runtime pairs with it stays module-owned.
"""

import asyncio
from typing import TYPE_CHECKING, Any

from starlette.datastructures import State

from llm_proxy.core.exceptions import ConfigurationError

if TYPE_CHECKING:
    from fastapi import FastAPI, Request, WebSocket
    from redis.asyncio import Redis

    from llm_proxy.config.manager import DatabaseConfigManager
    from llm_proxy.core.circuit_breaker import CircuitBreakerStore
    from llm_proxy.core.processing.unified import UnifiedProcessor
    from llm_proxy.core.provider_stats import ProviderStatsStore
    from llm_proxy.database.redis_client import RedisClient
    from llm_proxy.http.client import ProviderHTTPClientManager
    from llm_proxy.mcp.manager import MCPProxyManager
    from llm_proxy.web_search.interceptor import WebSearchInterceptor


class RuntimeServices:
    """Typed view over the process-lifetime services stored on ``app.state``.

    Construct one per call site through :func:`runtime_services`; instances are
    cheap (one reference) and stateless. Accessors that install are called only
    by :mod:`llm_proxy.api.lifecycle` at startup.
    """

    __slots__ = ("_state",)

    def __init__(self, state: Any) -> None:
        """Wrap an ``app.state``-shaped object (production: ``app.state``)."""
        self._state = state

    # ── Required services ─────────────────────────────────────────────────
    # Absence is a startup bug, not a configuration: the lifespan installs
    # these before the app can serve a request. Read them without a None check.

    def config_manager(self) -> DatabaseConfigManager:
        """The live configuration manager, or raise if the lifespan did not run.

        The manager is the entry point for every UI-managed setting and for
        virtual-model resolution, so a missing one cannot be defaulted: the
        proxy would serve with code defaults while the operator's settings
        silently do not apply.
        """
        manager = self.config_manager_or_none()
        if manager is None:
            raise ConfigurationError(
                "Config manager not initialized. Ensure lifespan is properly configured."
            )
        return manager

    def http_client_or_none(self) -> ProviderHTTPClientManager | None:
        """Like :meth:`http_client`, for the callers that may run without the pool.

        The routing judge reads the pool defensively: a judge call is internal
        and optional, and the background warm-up also runs where the shared pool
        is not set up (early startup, tests). It degrades to opening its own
        client rather than refusing to consult the judge.
        """
        return self._read("http_client")

    def http_client(self) -> ProviderHTTPClientManager:
        """The pooled provider HTTP client manager, or raise if absent.

        Required: adapters built without it open an unpooled client per call,
        which is correct but defeats every connection limit the operator set.
        """
        manager = self.http_client_or_none()
        if manager is None:
            raise ConfigurationError(
                "HTTP client not initialized. Ensure lifespan is properly configured."
            )
        return manager

    # ── Optional services ─────────────────────────────────────────────────
    # Absence is a legal state: the feature is off (Redis disabled) or not yet
    # built in this worker (cold stats store). Callers degrade deliberately.

    def config_manager_or_none(self) -> DatabaseConfigManager | None:
        """Like :meth:`config_manager`, for the tolerant ``resolve_*`` readers.

        ``resolve_logging_config`` / ``resolve_security_params`` /
        ``resolve_keepalive_params`` fall back to code defaults when the manager
        is missing (early startup, tests); this accessor is for them. Anything
        that *acts* on a setting should use :meth:`config_manager`.
        """
        return self._read("config_manager")

    def redis(self) -> RedisClient | None:
        """The Redis connection manager, or ``None`` when Redis is disabled.

        This is the wrapper (``is_connected``, ``health_check``, ``client``);
        most callers want :meth:`redis_client` instead.
        """
        return self._read("redis_client")

    def redis_client(self) -> Redis | None:
        """The raw async Redis client, or ``None`` when Redis is disabled.

        ``None`` is a legal state that means "this feature needs Redis and does
        not have it" — callers either degrade or fail with a specific message.
        """
        wrapper = self.redis()
        return wrapper.client if wrapper is not None else None

    def circuit_breaker(self) -> CircuitBreakerStore | None:
        """The per-worker circuit-breaker store, or ``None`` when unavailable.

        The store itself carries the enabled flag, so a caller that finds it
        simply records no failures; tests and early startup legitimately have
        none.
        """
        return self._read("circuit_breaker")

    def provider_stats(self) -> ProviderStatsStore | None:
        """The per-worker provider latency store, or ``None`` when unavailable.

        Process-local and lost on restart; the ``balanced`` selection strategy
        degrades to cost ordering while the store is cold.
        """
        return self._read("provider_stats")

    def mcp_manager(self) -> MCPProxyManager | None:
        """The per-worker MCP server manager, or ``None`` when it was not built."""
        return self._read("mcp_manager")

    def web_search_interceptor(self) -> WebSearchInterceptor | None:
        """The live web-search interceptor, or ``None`` when search is off.

        Read-only: the interceptor is *rebuilt* from the request path by
        :func:`llm_proxy.web_search.runtime.ensure_web_search_interceptor`, which
        owns the config snapshot it was built from.
        """
        return self._read("web_search_interceptor")

    def protocol_processor(self, protocol: str) -> UnifiedProcessor | None:
        """The unified processor installed for one protocol, or ``None``.

        The caller owns the policy for absence: the HTTP protocol router refuses
        the request, the OpenResponses websocket closes with a 503 frame before
        reading a turn.
        """
        processors: dict[str, UnifiedProcessor] = self._read("protocol_processors") or {}
        return processors.get(protocol)

    # ── Mutable slots: exactly one owner, here ────────────────────────────

    def background_tasks(self) -> set[asyncio.Task[Any]]:
        """The strong-reference set for fire-and-forget tasks, created on demand.

        The event loop only holds weak references to tasks, so every detached
        task needs a strong reference or it may be garbage collected mid-run.
        One set is shared by every producer (the lifespan's judge warm-up, the
        background OpenResponses responses) and drained at shutdown, so this
        accessor — not its callers — owns creating it.
        """
        tasks: set[asyncio.Task[Any]] | None = self._read("background_tasks")
        if tasks is None:
            tasks = set()
            self._state.background_tasks = tasks
        return tasks

    def judge_warmup_task(self) -> asyncio.Task[Any] | None:
        """The in-flight judge warm-up task, or ``None`` when none is running."""
        return self._read("judge_warmup_task")

    def set_judge_warmup_task(self, task: asyncio.Task[Any]) -> None:
        """Record the in-flight judge warm-up task (at most one at a time)."""
        self._state.judge_warmup_task = task

    # ── Installation: the only writers of these slots ─────────────────────
    # Called by the lifespan, except ``install_web_search_interceptor``, which the
    # web-search runtime calls from the request path when it swaps a rebuilt
    # interceptor (it owns that rebuild invariant).

    def install_config_manager(self, manager: DatabaseConfigManager) -> None:
        """Install the config manager (``startup_config``)."""
        self._state.config_manager = manager

    def install_http_client(self, manager: ProviderHTTPClientManager) -> None:
        """Install the pooled HTTP client manager (``startup_http_client``)."""
        self._state.http_client = manager

    def install_redis(self, connection: RedisClient | None) -> None:
        """Install Redis, or record that Redis is disabled (``startup_redis``)."""
        self._state.redis_client = connection

    def install_circuit_breaker(self, store: CircuitBreakerStore) -> None:
        """Install the circuit-breaker store (``startup_circuit_breaker``)."""
        self._state.circuit_breaker = store

    def install_provider_stats(self, store: ProviderStatsStore) -> None:
        """Install the provider latency store (``startup_provider_stats``)."""
        self._state.provider_stats = store

    def install_mcp_manager(self, manager: MCPProxyManager) -> None:
        """Install the MCP server manager (``startup_mcp_servers``)."""
        self._state.mcp_manager = manager

    def install_protocol_processor(self, protocol: str, processor: UnifiedProcessor) -> None:
        """Install the processor for one protocol (``startup_protocols``).

        Keyed by protocol name in one mapping rather than one ``app.state``
        attribute per protocol (``"{name}_processor"``): the name is data, so a
        lookup is a mapping miss the type checker can see instead of an
        attribute that only some other module's string formatting can hit.
        """
        processors: dict[str, UnifiedProcessor] = self._read("protocol_processors")
        if processors is None:
            processors = {}
            self._state.protocol_processors = processors
        processors[protocol] = processor

    def install_web_search_interceptor(self, interceptor: WebSearchInterceptor | None) -> None:
        """Install the live web-search interceptor (written by the web-search runtime).

        Called from the request path by
        :func:`llm_proxy.web_search.runtime.ensure_web_search_interceptor`, which
        owns the rebuild invariant and pairs this slot with the config snapshot
        it was built from. It installs the interceptor *before* that snapshot so
        a reader that sees the new snapshot sees the new interceptor.
        """
        self._state.web_search_interceptor = interceptor

    # ── Internals ────────────────────────────────────────────────────────

    def _read(self, name: str) -> Any:
        """Read one owned slot without a default the type checker cannot see."""
        return getattr(self._state, name, None)


def runtime_services(target: FastAPI | Request | WebSocket | State) -> RuntimeServices:
    """Return the services view for an app, request, websocket or state object.

    ``app.state`` is the carrier for all four: a request and a websocket reach it
    through ``app``, an app holds it directly, and a state object *is* it.

    A test double has to be unambiguous about which of the four it stands in for:
    a request double needs an explicit ``app`` (``double.app.state.…``), and an app
    double needs a real ``state`` — pass a ``FastAPI()`` app rather than a bare
    ``MagicMock``, whose ``.app``, ``.state`` and every other name answers with a
    mock of its own, so the view would read a service nobody installed.
    """
    if isinstance(target, State):
        return RuntimeServices(target)
    app = getattr(target, "app", None)
    if app is None or not hasattr(app, "state"):
        app = target
    return RuntimeServices(app.state)


__all__ = ["RuntimeServices", "runtime_services"]
