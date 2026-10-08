# Server settings registry: one descriptor per UI-managed config key

## Status

Accepted (2026-10).

## Context

Every section of the settings UI is one row of the `server_config` table —
`logging`, `web_search_config`, `smart_routing`, `provider_selection`,
`request_policy`, `resilience`, `security`, `keepalive`, `rate_limits`,
`cors_origins`, `mcp_security_policy` — and each key had its shape declared
three times. The admin router (`api/routers/config/server.py`, 646 lines)
copied the same GET/PUT pair ten times, with six near-identical
`_build_*_response` helpers; `DatabaseConfigManager.load` hand-parsed the same
keys again with ten ad-hoc strategies; and the request/response models existed
in two parallel hierarchies — `api/schemas/server_config.py` (14 classes) and
`config/types/*` (the runtime truth) — that had already drifted (`SearXNG`
`max_results` validated `le=10` on the wire but `le=20` at runtime;
`sensitive_keys` a CSV string on the wire, `list[str]` at runtime) and were
policed by a dedicated drift-guard test file.

The drift guard was the tell: it existed because the two hierarchies *could*
disagree, and its docstring cites a shipped bug (keepalive defaults) as the
reason. The ten endpoint copies were the second tell: adding a settings
section meant writing a handler pair, a response builder, a parse block, a
schema class, and a test file — the key's shape was the only new information,
and it had no single home.

This is the report card's first candidate from the 2026-10 architecture
review: the keyed resource was shallow everywhere — an interface as complex
as its implementation, copied per key.

## Decision

One registry, `llm_proxy.config.server_settings`, holds one `ServerSetting`
descriptor per key:

- the `server_config` row key, the route path, and the row description;
- the **canonical model** — the runtime truth, a Pydantic model living in
  `config/types/` (`SecurityParams`, `KeepaliveParams`, `RequestPolicyConfig`,
  `ResilienceConfig`, `RateLimitsConfig`, `CorsConfig`,
  `McpSecurityPolicyConfig`, `SmartRoutingConfig`,
  `ProviderSelectionConfig`, `WebSearchConfig`, and `LoggingWireConfig`, the
  wire dialect of `LoggingConfig`);
- the **write mode** — one of six, covering only the variance that exists
  today: full replace (request_policy, resilience, security, keepalive,
  mcp_security), replace-dropping-unset-sub-configs (web_search), replace
  with the wrapper's inner dict (rate_limits), replace with a stripped bare
  list (cors), partial field merge (smart_routing, provider_selection), and
  partial merge over wire defaults in wire field names (logging);
- the **read mode** — model-dump, wire-defaults-merged-raw (logging, so
  legacy rows round-trip unvalidated), raw-or-default (web_search),
  limits-merged (rate_limits), list-wrap (cors);
- optional declared extras: the write validator (rate_limits' unknown-bucket
  and bad-spec checks), the update-field narrowing
  (provider_selection carries only `strategy`), the DELETE verb and message
  (web_search only).

Three consumers are generated from the table:

- the admin endpoints — `api/routers/config/server.py` registers its routes
  from the registry in a loop, one handler per verb;
- the runtime read — `load_server_settings(rows)` returns the typed
  `ServerSettingsRuntime` snapshot that `DatabaseConfigManager.load`
  assembles `ServerParams`/`ProxyConfig` from, preserving each key's
  tolerant-parse semantics (warn and fall back; logging's retention/body caps
  fail loud);
- the contract test — `tests/api/test_server_settings_contract.py` walks the
  table key by key: default responses, PUT persistence per write mode,
  merge-vs-replace semantics, DELETE, and validation errors.

Wire compatibility is a hard constraint: every path, verb, payload, response
document, and 422 message is byte-identical to the hand-written endpoints it
replaces (the frontend's `config.ts` calls all eleven paths unchanged). The
merge/replace divergence the old endpoints had (partial vs full, re-read vs
echo) is preserved as declared per-key variance, not unified. Reload
semantics are untouched: every PUT still commits then runs the full
`commit_and_reload`, and the reload listeners in `lifecycle.py` are as they
were.

The rate-limit bucket grammar (`DEFAULT_RATE_LIMITS`, "N/period" parsing,
override validation) moved from `api/middleware/rate_limiting.py` into
`config/types/rate_limits.py`; the middleware imports it. Neither the API
layer nor the middleware owns another module's data anymore.
`McpSecurityPolicy` subclasses `McpSecurityPolicyConfig`, so the UI-managed
policy fields have one source of defaults and the DB-override field list is
derived, not hand-maintained.

`api/schemas/server_config.py` is **deleted** — this amends ADR-0023 for the
server-config resource: its schemas are not hand-written per resource, they
*are* the canonical runtime models (or, for logging, their declared wire
dialect), generated into the admin API by the registry. The one-module-per-
resource rule stands for every other resource.

## Considered Options

- **Keep the two model hierarchies and police them with the drift guard.**
  Rejected: the guard's existence is the admission that the duplication will
  drift again; it caught defaults but not validation ranges or type shape
  (the `sensitive_keys` str/list mismatch survived it). One canonical model
  makes drift impossible rather than detectable, and deletes 381 lines of
  mirror plus the guard file itself.
- **Generate the endpoints as one generic `/server/{key}` route.** Rejected:
  it changes the wire paths the frontend calls today and loses per-key
  OpenAPI operation identity; the descriptor generates the existing per-key
  paths instead, so the refactor is invisible at the API surface.
- **One canonical model with no wire dialect (rename `log_retention_days` /
  make `sensitive_keys` a list on the wire).** Rejected: wire compatibility
  was the constraint; the dialect is real (the wire names predate the
  UI-managed key), so it is declared — `LoggingWireConfig` plus the merge
  mode — instead of being silently entangled in one model's validators.
- **Hand-written thin delegates calling a shared service.** Rejected: keeps
  the copy-paste shape alive at the call sites; generation from the table is
  what makes a new setting a one-row change.

## Consequences

- The 646-line router becomes a route-registration loop; the ten parse
  blocks in `manager.load` become one snapshot call; `server_config.py`'s
  14 mirror classes, the six `_build_*_response` helpers, and the drift-guard
  test file are deleted. The registry module costs ~600 lines including the
  runtime loader, so the net deletion is roughly 700 lines — and what
  shrinks is the part that had to be kept in sync.
- Adding a UI-managed setting is adding a descriptor row: endpoint, runtime
  read, and contract-test coverage come with it. This is the leverage the
  report card asked for.
- The contract test's default-response assertions are the drift guard now:
  the wire defaults cannot diverge from the runtime model because both are
  the same class's `model_dump()`.
- Five keys that had no endpoint-level tests (keepalive, resilience,
  rate_limits, cors, web_search) are covered by the parametrized contract
  walk.
- The read/write mode enums are deliberately closed: a key with genuinely new
  semantics is a new enum member with one implementation, reviewed in one
  place — not a fourth copy of the handler pair.
- Regression coverage: `tests/api/test_server_settings_contract.py` (40
  tests) pins the wire contract per key; `tests/test_config/test_manager.py`
  keeps covering the runtime read against the registry loader.

Amends ADR-0023 (server-config schemas are generated from canonical models,
not hand-written); extends the ADR-0021 direction (the read side of the same
data now has one owner too).
