# One log-intake module owns every request-log row

## Status

Accepted (2026-10).

## Context

`RequestLogCreate` is a ~44-field DTO that mirrors the `request_logs` table. It
was constructed at **13 sites across 8 modules** — the endpoint audit handler
(3 builders), the exception middleware, the admin-API logging middleware, the
audit classification helpers (3 writers), the tool log service (2), the internal
call log service, the realtime usage observer, and the auth router. Every site
re-derived the same knowledge by hand: classification (`determine_event_type` /
`action_category` / `resource_type` / `resource_id` / `outcome`), attribution
(`get_request_identity`), masking (`mask_headers` / `mask_sensitive` /
`scrub_log_metadata`), the server/service stamps, the background-writer dispatch
rules (AUDIT rows take the dedicated hash-chain writer), and which records a
situation produces (an early failure writes a *usage* row too).

Two failure modes followed from that spread:

- **The dedup flag was negotiated, not owned.** `request.state.audit_log_written`
  was written at **7 sites in 5 modules** and read at 2 more, purely so the
  pipeline, the exception handler and the admin middleware would not double-write
  one request. Correct "check then set" ordering had no locality.
- **One situation had two implementations.** `api/middleware/exceptions.py`
  re-implemented the audit handler's early-failure row (plus its own body/header
  backfill and byte-stripping) instead of sharing a builder.

## Decision

The rows move behind one deep module: `llm_proxy.observability.log_intake`, with
**one verb per situation**. `configure()` installs the config manager at startup
(the live UI-managed `LoggingConfig` is re-resolved per write, as the writers
always did). The verbs own classification, masking, attribution, the
server/service stamps, the record fan-out, dispatch to the background writers,
and the idempotent `audit_log_written` guard.

- **Endpoint lifecycle** (from `EventContext`): `record_request_end`,
  `record_request_error`, `record_stream_end` (row + usage record).
- **Pre-pipeline failures**: `record_early_failure` (row + usage record,
  backfills and masks headers/body from `request.state`).
- **Admin/auth**: `record_admin_request` (the ASGI middleware's captured
  bodies), and the explicit-action family `record_member_action`,
  `record_key_reveal`, `record_auth_event`, `record_failed_auth`, each owning
  its own classification.
- **Rejections**: `record_rejection`, which also owns the per-(key, status)
  retry-storm dedupe window (moved with it).
- **Auxiliary calls**: `record_mcp_call`, `record_web_search`,
  `record_internal_call`, `record_realtime_turn`.

`RequestLogCreate` becomes the module's assembly detail — no call site builds
one. `observability/audit_helpers.py`, `observability/tool_logging.py` and
`observability/internal_call_logging.py` are deleted (hard cut, no
compatibility shims); the tool/internal entry records move to the intake module,
`RealtimeSessionContext` stays in `realtime/usage.py` and satisfies the verb's
structural interface, and `CONTENT_HASH_VERSION` moves to `log_repository.py`
(its only consumer, and the module that owns the hash chain) to keep the
dependency direction one-way.

## Considered Options

- **Situation factories returning a `RequestLogCreate`** (call sites keep
  dispatching): rejected — the interface would shrink, but "AUDIT goes to the
  dedicated writer", "an early failure also writes usage" and the dedup ordering
  would stay re-derivable at 13 sites, which is exactly the knowledge that had no
  locality. The deletion test settles it: deleting the verbs relocates assembly
  *and* policy across every caller.
- **Coexistence (intake lands beside the sites, migrate incrementally)**:
  rejected — two paths building rows is the split-brain failure this change
  exists to close, and the flag would stay half-owned until the last site moved.
- **One generic classified-row verb instead of the explicit-action family**:
  rejected — it would push `event_type`/`resource_type`/`outcome` back out to
  callers, re-leaking classification. Four narrow verbs keep the situation's
  classification inside the module.
- **Keep the entry dataclasses in their old modules and have the verbs import
  them**: rejected — it creates a cycle (`tool_logging` → intake for dispatch,
  intake → `tool_logging` for the type). The tool/internal records move to the
  intake module (they are verb parameters); `RealtimeSessionContext` is read
  structurally for the same reason.

## Consequences

- Adding a log fact (a new column, a new masking rule, a new record fan-out)
  touches **one** module and one test surface instead of every caller.
- `AuditLogHandler` is now a thin tracing handler: it only arms the stream
  capture and forwards finished situations to the verbs. `api/middleware/*`,
  the admin routers, `mcp/*`, `web_search/interceptor.py`, `realtime/usage.py`
  and `api/context.py` call verbs.
- Tests split along the seam: behavior through the verb interface lives in
  `tests/observability/test_log_intake_verbs.py`; the classification table and
  internal row-shape rules (verbose-routing redaction, retry metadata, stored
  stream body) are internal-seam tests in `tests/observability/test_log_intake.py`.
  The old per-site construction tests were replaced, not layered.
- Behavior is preserved deliberately, including two non-obvious rules now pinned
  by tests: tool/internal rows keep the binary success/failure outcome the old
  writers used (not the 3-way audit outcome), and the error path does not force a
  cost calculation.
- The module is configured at startup in `startup_tracing`; `RequestLogService` /
  `UsageService` remain the dispatch machinery (and the test adapter seam), no
  longer a call-site concern.
