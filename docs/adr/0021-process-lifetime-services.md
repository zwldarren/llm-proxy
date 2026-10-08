# One seam owns the process-lifetime services

## Status

Accepted (2026-10).

## Context

Everything that outlives a request is constructed once in
`llm_proxy.api.lifecycle` and stored on FastAPI's `app.state` bag: the config
manager, the pooled HTTP client, Redis, the circuit-breaker store, the
provider-stats store, the MCP manager, one `UnifiedProcessor` per protocol, and
the web-search runtime (which builds and swaps itself from the request path).
Consumers read them with `getattr(request.app.state, "name", None)` — **43 such
probes across 17 modules** — and each probe re-derived three things on its own:

- **Whether absence is legal.** `None` meant "the lifespan never ran" (a startup
  bug), "the feature is disabled" (a legal state), and "not built yet in this
  worker" (a lazy slot). Nothing in the call site distinguished them.
- **The attribute name.** A typo is a permanent `None`, and never an error. The
  worst case was `setattr(app.state, f"{protocol_name}_processor", …)`: the name
  of a service is *data*, so no checker could see a reader that asked for a
  protocol the writer never installed.
- **The failure mode degraded instead of failing.** The two authentication
  middlewares read the config manager defensively — `if config_manager is None:
  return None` — so a lifespan that never installed one made the gate silently do
  nothing, leaving the decision to whatever dependency each route happened to
  declare (the protocol routes' `require_any_auth`, the protocol fast path's
  re-check, the admin routers' `require_admin_role`). Nothing is exploitable today
  — those dependencies are there — but the gate that owns the decision is no longer
  the one making it, and a route added without one, or a path the fast path
  bypasses, is where that becomes a hole.

Ownership was also split in the wrong direction. `app.state.background_tasks` —
the strong-reference set that keeps fire-and-forget tasks from being garbage
collected mid-run — was created by *two* producers (`api/lifecycle.py` and
`api/routers/protocol.py`), each with its own copy of the same check-then-create.
`api/lifecycle.py` read nine of its own services back through the same probes it
had just written.

## Decision

One leaf module — `llm_proxy.services` — owns the attribute names, the type of
each service, and whether absence is a legal state. `RuntimeServices` is a typed
*view* over the state object, obtained with `runtime_services(app | request |
websocket)` or `RuntimeServices(state)`:

- **Required services raise.** `config_manager()` and `http_client()` raise
  `ConfigurationError` naming the missing startup step. The two authentication
  middlewares call `config_manager()` for exactly this reason: a proxy whose
  lifespan did not run now refuses the request at the gate that owns the decision,
  instead of degrading into whatever the route below remembers to check.
- **Optional services return `None`, and the name says so.** `redis()`,
  `redis_client()`, `circuit_breaker()`, `provider_stats()`, `mcp_manager()`,
  `web_search_interceptor()` and `protocol_processor(name)`. Where a service has
  *both* policies the pair is explicit: `config_manager_or_none()` and
  `http_client_or_none()` are for the readers that legitimately run without it
  (the `resolve_*` configuration helpers with documented defaults, the judge
  warm-up during startup).
- **One place installs.** `install_*` methods are the only writers of the
  lifecycle-owned slots, so `app.state` attribute names appear in one module.
  The per-protocol processors become one mapping (`install_protocol_processor` /
  `protocol_processor`), which turns "a protocol the writer never installed"
  from a silently missing attribute into a mapping miss.
- **The shared mutable slots have one owner.** `background_tasks()` creates the
  set on first use, so both producers get the same one and shutdown drains it.

`app.state` stays the carrier rather than becoming a container: FastAPI's
lifespan and the existing test doubles already write it, so the change is a typed
view, not a second place where state can live. `RuntimeServices` accepts any
object with attributes, which is also the in-memory adapter that tests use
(`tests/services_helpers.py`: `services_for(config_manager=manager)`), against
the production adapter (`app.state`).

Two attributes are deliberately *not* owned here:

- `app.state.limiter` — slowapi resolves the limiter by that exact name; it is
  the library's contract, not ours.
- `web_search_config_snapshot` — the web-search runtime owns the
  rebuild-on-config-change invariant (and its "never built in this worker"
  sentinel), and installs the interceptor itself through
  `install_web_search_interceptor`, interceptor first so a reader that sees the
  new snapshot always sees the new interceptor.

The routing layer stops passing an untyped `app_state: object` around:
`judge_call_kwargs`, `warm_judge`, `orchestrate_smart_routing` and
`resolve_virtual_model` take `services: RuntimeServices`, which is where the
judge's three defensive reads (`http_client_or_none`, `circuit_breaker`,
`provider_stats`) now live. `get_embedding_signal` loses its unused `app_state`
parameter — it was a module-level cache all along.

## Considered Options

- **A container on `app.state` (`app.state.services = RuntimeServices(...)`).**
  Rejected, but only on cost, not on the interface: it makes the state object
  invisible everywhere, yet it also breaks every existing writer — the lifespan,
  the web-search runtime, and 40+ test setups that assign
  `app.state.config_manager` directly — for no interface difference at the call
  sites. The view gives the same typed accessors while leaving test doubles that
  write the bag working, and the in-memory adapter is one line.
- **One accessor per protocol as a property (`app.state.openai_processor`) kept
  as the storage, with `protocol_processor(name)` reading it.** Rejected: the
  name stays data, so the writer and the reader agree only by string formatting
  — which is the failure this candidate exists to remove.
- **Making every service required (no `_or_none` pairs).** Rejected: Redis is
  off in plenty of correct deployments, the stats store is legitimately cold, and
  the `resolve_*` helpers document a defaults path for early startup and tests.
  Encoding "optional" as `None` is honest; the fix is that the *name* says so and
  the required ones cannot be forgotten.
- **Keeping the probes but adding a typed helper only for the config manager.**
  Rejected: it would leave the background-task double-create, the dynamic
  protocol attribute, and the untyped `app_state` parameter in routing — three of
  the five failure modes — untouched.

## Consequences

- A misspelled service is now a type error at the accessor, and a missing
  required service is a `500` with the startup step named, not a silently
  disabled feature. The two authentication gates are the sharpest case: they no
  longer decide nothing and leave the decision to the route below. The two silently
  disabled *features* (keepalive, security-param defaults, CORS origin list) keep
  their documented defaults, which is why their readers are named
  `config_manager_or_none()`.
- Installing a protocol's processor and reading it are one module's concern, so
  "which protocols exist" is answerable from the mapping instead of by probing
  attribute names.
- `app.state` remains the carrier, so the invariant is not enforced by the type
  system: a module can still write a raw attribute. The rule is that it does not,
  and `rg 'app\.state' src/` is now a short, reviewable list (the lifespan's
  installs, slowapi's limiter, the web-search snapshot).
- Routing no longer accepts `Any` application state, so the judge's dependencies
  are visible in its signature.
- Tests gain the in-memory adapter (`tests/services_helpers.py`) and lose the
  habit of handing a `MagicMock` in as application state, which answered every
  attribute with a mock and erased the present/absent distinction the accessors
  exist to express.
