# Lazy imports bind at the third-party SDK boundary

## Status

Accepted (2026-10).

## Context

Python 3.15 adds PEP 810: `lazy import x` / `lazy from x import y` at module
scope bind a proxy and load the real module the first time the name is used.
`import llm_proxy.api` — what every uvicorn worker pays before it can accept a
connection, and what every test import pays — measured **~1.1–1.25 s** (min/median
of seven cold `-X importtime` runs). **Eleven** trees belonging to opt-in subsystems
were inside that process before any request existed.

Most of those trees belong to subsystems an operator may never turn on. Attributing
each tree to the first `llm_proxy` edge that pulled it in (`-X importtime` plus an
`__import__` tracer) gave a short, unambiguous list:

Costs are the module's own `-X importtime` cumulative figure on the pre-migration
tree; "pulled by" is the first `llm_proxy` edge on its parent chain:

| tree | cost | pulled by |
| --- | --- | --- |
| `mcp` | 351 ms | `api/lifecycle.py` → `llm_proxy.mcp` → `mcp.backend` |
| ├ `uvicorn` | 20 ms | reached only through the `mcp` tree |
| └ `sse_starlette` | 21 ms | reached only through the `mcp` tree |
| `langfuse` (+ OpenTelemetry SDK/exporters) | 88 ms | `tracing/handlers/__init__.py` registering `LangfuseTracingHandler` |
| `redis` | 54 ms | `database/redis_client.py`, `cache/redis_cache.py`, `openresponses/store.py` |
| `jwt` | 17 ms | `security/jwt.py` |
| `websockets` (client stack) | 40 ms | `realtime/{relay,upstream}.py` |
| `cryptography`, `bcrypt`, `tiktoken` | ~19 ms | `security/encryption.py`, `security/passwords.py`, `billing/tokens.py` |

None of them is needed to serve a chat completion, and three of the six are
opt-in-by-configuration. The codebase already had the pre-PEP-810 shape of the
answer in three places — `http/`, `observability/` and `streaming/` carried a
hand-written module `__getattr__` + `importlib.import_module` façade — and 405
function-level imports scattered through `src/`, which is the pattern PEP 810
explicitly exists to replace.

## Decision

**Mark the import edge that crosses into a third-party SDK `lazy`, and leave the
application's own import graph alone.**

- The three `__getattr__` façades (`http/`, `streaming/`, `observability/`) and the
  `mcp/` façade became PEP 810 `lazy from …` re-export blocks. `__all__` and the
  public surface are unchanged; `AttributeError` for an unknown name, `dir()` and
  direct submodule imports all still behave as before, and the proxy resolves once
  and caches (the hand-written ones re-ran a dict lookup per attribute access).
- SDK edges became `lazy` at their point of use: `mcp` / `mcp.client.*` /
  `mcp.server.*` / `mcp.types`, `langfuse` + `opentelemetry.sdk`, `redis.asyncio` +
  `redis.exceptions`, `jwt`, `bcrypt`, `cryptography.fernet`, `tiktoken`,
  `websockets`. The classes that own those SDKs keep their existing construction
  points, so `MCPProxyManager()` and handler registration stay cheap instead of
  dragging the SDK in.
- Where a deferred name appeared in a module-level *value* position, that position
  moved rather than the import: `cache/redis_cache.py`'s `_CONNECTION_ERRORS` tuple
  became a `@cache`-d `_connection_errors()` accessor, because a module-level tuple
  literal evaluates its members and would reify the proxies at import.
- `lazy` on an edge that was already a module-scope import is a pure timing change
  (bind-once before, bind-once at first use now). `api/lifecycle.py` and
  `__main__.py` therefore only converted their own module-scope edges.
- The CLI's server stack is `lazy` in `__main__.py`, so argument parsing and
  `--help` do not build the app.

**Rejected: `-X lazy_imports=all` / `PYTHON_LAZY_IMPORTS` / `__lazy_modules__`.**
A global switch makes the *deployment* responsible for a property the source should
state, applies to third-party internals we do not control, and cannot be reviewed
per call site. The explicit keyword is greppable, type-checkable, and revertible
one edge at a time.

## Consequences

- `import llm_proxy.api`: **1089 ms → 740 ms** at the minimum, **1250 ms → 813 ms**
  at the median (seven cold runs each side, baseline measured from a worktree of the
  pre-migration commit) — about **-32%**. The eleven optional trees
  (`mcp`, `mcp_types`, `langfuse`, `redis`, `tiktoken`, `jwt`, `bcrypt`,
  `cryptography`, `websockets`, `sse_starlette`, `uvicorn`) went **11 → 0**: none of
  them is imported unless its feature is used. What remains is what every request
  needs (`fastapi`, `pydantic`, `sqlalchemy`, `httpx2`, `orjson`, `slowapi`/`limits`,
  `platformdirs`, `dotenv`) plus `opentelemetry`, which FastAPI's own telemetry module
  imports — not ours to defer.
- `llm-proxy --help`: **559 ms → 109 ms** (min of seven runs).
- The 25 function-level imports in `api/lifecycle.py` stay function-level (see the
  late-binding rule above), so the lifespan's own dependency list is still spread
  across the hooks that use it. That is the accepted cost of not weakening patchability.
- Related mechanical axes landed with this one and are governed by
  `AGENTS.md → Python 3.15 features`: PEP 814 `frozendict` on immutable module-level
  tables (49 in the tree), PEP 661 `sentinel` for unset markers, PEP 800
  `disjoint_base` on the two mutually exclusive Decisions/System One bridge mixins,
  `re.prefixmatch`, and removal of the now-redundant `from __future__ import
  annotations` (PEP 649 makes annotations lazy by default). PEP 798 had exactly two
  multi-clause comprehensions in `src/` and both carry a filter or an element
  transform, so neither qualifies; PEP 747 `TypeForm` and PEP 728 `extra_items` have
  no call site here (every `type[X]` parameter is used through `X.model_fields`,
  `create_model` or `isinstance`, which need a real class, and there are no
  `TypedDict` classes in `src/`).
- The generated OpenAPI document is **byte-identical** across the whole migration
  (`cmp` of `llm-proxy-openapi` output from the pre-migration worktree and from this
  tree), which is the check that keeps the payload-shaped-table rule below honest.
- A subsystem's import cost now lands on the first operation that uses it: the
  first MCP server start, the first token count, the first JWT signature. Each is
  one-time and off the response path's critical timing; the trade is deliberate
  against "every worker pays it at boot".
- A missing optional dependency now fails at first use instead of at import, with a
  traceback naming both the access site and the original import. `uvloop`
  (`__main__.py`) and `brotli`/`zstd` (`content_encoding.py`) keep their
  function-level guarded imports: `lazy` inside a `try` is a `SyntaxError`, so the
  graceful-absence contract wins over uniformity.
- **Function-level imports are not to be hoisted into `lazy` ones.** A function-level
  import re-reads the source module's attribute on every call; a module-scope `lazy`
  import resolves once and caches. Hoisting `api/lifecycle.py`'s 25 function-level
  imports was tried and reverted: tests patch the *source* module (e.g.
  `patch("llm_proxy.config.get_encryption_key", …)`) and depend on the per-call
  lookup. Deferral is already what those imports buy, so the hoist traded a real
  semantic for no measurement.
- `@dataclass` / pydantic / `TypedDict` field annotations are evaluated when the
  class is created, so a `lazy` name used there reifies immediately (measured: 305 ms,
  zero saving). Function-, module- and plain-class annotations are lazy in 3.15 and
  are safe. Where a dataclass field needed an SDK type (`_session: ClientSession |
  None` in `mcp/backend.py`), the annotation points at a `TYPE_CHECKING`-only alias
  and the runtime value comes from the `lazy` import.
- PEP 810 needs no tooling work: ruff 0.17 parses and formats `lazy`, its isort
  rules sort lazy blocks, and `ty` type-checks through the proxy.
