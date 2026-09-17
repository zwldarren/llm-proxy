# Log store follow-ups: retention, dead columns, log-type classification

## Status

Accepted (2026-09).

## Context

ADR-0015 removed the raw-SSE request-log body and listed, as out of scope, the smaller
defects the same investigation found around it. Three of them were still open, plus one
gap the ADR itself recorded as a known loss:

- `usage_records` was pruned on a hardcoded 365-day window while `request_logs` followed
  the UI's retention setting, so the dashboard kept history the operator had already
  asked the logs to drop (and the table is the bigger of the two writers).
- Six `request_logs` columns existed for a body-compression feature that was never
  wired up (`request_body_compressed`, `response_body_compressed`,
  `request_body_compression`, `response_body_compression`,
  `request_body_original_size`, `response_body_original_size`).
- `GET /v1/models` was the one `/v1/` path classified `LogType.AUDIT`, so a catalog
  read — polled routinely by client UIs — was pushed through the audit hash chain and
  pruned on audit retention.
- Beta terminal extras (`stop_sequence`, `stop_details`, `container`,
  `diagnostics`) were absent from every reassembled streaming body, because
  `format_response` reads them from `provider_info` and reassembly had no provider
  response to fill it from.

## Decision

**One retention window for the log store.** The background usage writer is started with
`LoggingConfig.retention_days` (`api/lifecycle.py`), the same UI-managed value the log
writer gets. `UsageService` no longer holds a retention of its own: retention is a
property of the writer, which startup configures once. `0` still means keep
indefinitely, now for both tables.

**Drop the dead compression columns.** All six are removed by a migration
(`a1f4c7d92b30`); `request_body`/`response_body` stay as the JSON columns they are.

**`GET /v1/models` is an ENDPOINT.** `determine_log_type` returns ENDPOINT for every
`/v1/` path; non-`/v1/` (console/API) paths stay AUDIT.

**Terminal provider extras reach the log.** `StreamingTransformer` gains a
`get_terminal_provider_info()` verb and `assemble_stream_response_body` a
`provider_info` argument. The lifecycle reads the verb *before* `finalize()` (which
clears the pending terminal state) and passes the result into
`InternalResponse.provider_info`, exactly as the non-streaming provider parser does.
Anthropic implements it for both tiers: the converted path from its pending
`stop_sequence`/`stop_details`/`container` and the chunk-level `diagnostics`, the
native path from the frames' own `message_delta`/`message_start` payloads. An explicit
`null` diagnostics is preserved, since it means "no cache divergence".

## Considered Options

- **A separate `usage_retention_days` UI field**: rejected — usage records are the
  dashboard's numbers, not a compliance artifact, and a second knob invites a dashboard
  that silently disagrees with the logs it summarizes. One window is the behavior
  operators expect from "retention".
- **Implement body compression instead of dropping the columns** (ADR-0015's
  alternative): rejected — reassembly already made the stored body an order of magnitude
  smaller, and compression keeps every other raw-body defect (unmaskable, unreadable,
  shape-divergent) while adding a decompress step to every read.
- **Exclude `/v1/models` from logging entirely**: rejected — it is a real request with a
  real status code; ENDPOINT classification gives it the sampling and retention the
  operator already controls, without inventing an unlogged path.

## Consequences

- Usage records and logs share one window: shortening log retention now also shortens
  dashboard history. On an existing install the next usage sweep prunes rows older than
  the configured window (default 30 days) — this is the row reduction the change is for.
- `/v1/models` rows move out of the audit table and off the hash chain, and follow
  endpoint sampling and retention.
- Streaming and non-streaming Anthropic logs now carry the same terminal extras; the
  other protocols are unchanged until they model extras worth logging.
- Downgrading past the drop migration re-creates the compression columns empty; the
  feature remains unimplemented.
