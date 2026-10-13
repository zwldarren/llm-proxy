# One key gate owns the auth sequence

## Status

Accepted (2026-10).

## Context

Every protected transport — the `/v1/*` middleware, the `/servers/*` MCP
middleware, the Realtime and OpenResponses WebSocket handshakes — resolves a
client API key through the same sequence: extract the key, verify it (bcrypt
against the cached key snapshot, with stale-entry revalidation and a
session-key fallback), enforce the per-key rate limit and the key/user budget,
account for client-IP lockout, and claim the throttled `last_used_at`
persistence. Until now that knowledge lived in a module named for one of the
transports:

- The gate verbs (`verify_api_key_for_mcp`, `check_key_rate_limit`,
  `check_key_budget`, `budget_status_rejection`, …) were defined in
  `api/middleware/mcp_proxy.py` — the module named for, and documented as, the
  `/servers/*` transport — and imported back out by the other two transports,
  including a deferred import from `core/ws_common` across the layering.
- The sequence itself was hand-orchestrated twice, nearly step for step, in
  `api_key_auth._dispatch` and `MCPProxyMiddleware.__call__`, and a third,
  thinner time in `authenticate_ws`. The copies had drifted into shipped
  behaviour differences: the `/v1` gate wrote audit rows for its rejections
  while `/servers` and the WS handshakes only `logger.info`'d; the lockout
  rejection paid the auth-failure delay on one path and not the other; the
  claim/clear order was inverted between the two HTTP gates; and a bad
  WebSocket key never touched lockout accounting at all.
- The verified principal travelled as `scope["llm_proxy_auth"]`, an untyped
  dict written by three middlewares in three different shapes (4 keys, 2 keys,
  9 keys) and read by exactly one consumer for exactly one key — the same
  typo-becomes-permanent-None failure mode ADR-0022 created request facts to
  kill, one channel over.
- `MCPProxyMiddleware` probed
  `getattr(getattr(self.main_app, "state", None), "config_manager", None)` —
  the raw `app.state` access ADR-0021 banned — because its short-circuit sits
  outside the FastAPI pipeline and that probe was its only chance to adopt a
  peer worker's configuration generation.

## Decision

One module, `llm_proxy.api.gate`, owns the whole sequence behind one verb:

```
gate_request(request, policy) -> GateRejection | None
```

- **Policy is data, declared where the wire is spoken.** Each transport
  declares a frozen `GatePolicy` row in its own module (`V1_GATE_POLICY`,
  `MCP_GATE_POLICY`, `REALTIME_GATE_POLICY`, `OPENRESPONSES_WS_POLICY`):
  which checks run (lockout, auth-failure penalty, rate limit, budget,
  last-used claim) and how strictly a key is extracted (`KeyExtraction`).
  The three transports legitimately disagree on what a non-Bearer
  `Authorization` header means, and only the WS transports accept subprotocol
  and query-param key sources — those are declarations, not accidents.
- **Rendering is the transport's; logging is the gate's.** The gate returns a
  structured `GateRejection` (kind, status, protocol-shaped body, message,
  headers) and the transport renders it in its own wire form — `JSONResponse`,
  raw ASGI with security headers, or a Realtime/OpenResponses error event plus
  a close code. The gate itself writes every rejection row
  (`record_failed_auth` for auth failures, `record_rejection` for quota), so
  "a rejection leaves a row" is an invariant of the module, not a courtesy of
  whichever transport remembered.
- **The verified principal is a fact of the request.** Identity is stamped
  into the request facts' `RequestIdentity` before the quota checks (so
  rejection rows are attributable), and the allowlists travel as facts too —
  the record gains `allowed_mcp_servers` beside `allowed_models`. There is no
  principal carrier and no scope dict; `api/routers/mcp.py` reads the facts
  off the shared scope state.
- **Drift is unified to one behaviour, not preserved per transport.** A
  lockout rejection is an audit row plus `Retry-After` with no extra failure
  delay (the lockout is the brake); the success path claims the last-used slot
  and then clears failed attempts; the WS handshakes log their rejections like
  every other transport; and the Realtime handshake now claims the last-used
  slot on connect and renders `Invalid API key` for a bad key (its pre-gate
  path did neither). Behaviour-changing unifications are listed here so
  they read as decisions, not accidents.
- **The MCP middleware keeps its short-circuit refresh, minus the probe.** It
  sets `scope["app"]` before the gate and adopts the peer configuration
  generation through `runtime_services(request).config_manager_or_none()` —
  ADR-0021 compliance without losing the only refresh point an MCP-only worker
  has. Its rejection renderer keeps applying the security-header set, because
  it short-circuits before `SecurityHeadersMiddleware`.
- **The verification verb is `verify_api_key`.** The verbatim relocation
  briefly kept the old `verify_api_key_for_mcp` name; the rename is part of
  this ADR — the gate is no longer named after one transport, and the bcrypt
  helper of the same name in `security.passwords` is imported as
  `key_matches_hash` to keep the two distinct. The verified record the verb
  returns is `VerifiedKeyAuth`, a frozen dataclass the gate's quota checks
  read by attribute (session keys carry an `is_session` flag, so nothing
  sniffs the `session:` prefix of a principal id).

The budget/spend-cache mechanics, the verified-key cache
(`api_key_cache.py`), and the lockout manager stay as they are: the gate
consumes them, and none of them is shallow.

## Alternatives rejected

- **Folding JWT verification into the gate.** JWT is a different credential
  family (signature verification against a token, not a bcrypt lookup against
  a key snapshot) and only serves `/api/*`. Only its identity stamping is
  shared: it writes the same typed `RequestIdentity`, and its `llm_proxy_auth`
  scope write — which had no reader — is deleted.
- **A separate budget/limits module.** The gate is the only consumer; a seam
  with one adapter is hypothetical. If a second consumer of budget enforcement
  appears, splitting it out costs the same then as it would now.
- **Returning a `VerifiedPrincipal` carrier from the gate.** It would sit
  beside the facts' identity as a second identity channel — the shape ADR-0022
  exists to prevent.

## Consequences

- Adding a protected transport means declaring a `GatePolicy` row and a
  rejection renderer — not re-orchestrating the sequence.
- The transport × step matrix is pinned by
  `tests/api/middleware/test_key_gate_contract.py`: every policy row is held
  against the gate's actual behaviour, step by step.
- `api/middleware/mcp_proxy.py` is again what its name says: the `/servers/*`
  transport. The gate's interface, not its home, is what the transports
  import.
