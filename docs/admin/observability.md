# Logs, Usage & Tracing

Every request through the proxy produces a log row, a usage record, and — when it is
an administrative or security event — a tamper-evident audit entry.

![Request logs](../screenshots/logs.png)

## Proxy Logs

**Logs → Proxy Logs** lists one row per request with timestamp, key/user, model,
provider, status, TTFT, duration, tokens, and cost. Open a row for the full request
and response bodies, headers, retry/fallback details, and (for smart-routed requests)
the routing decision.

| Field | Notes |
| --- | --- |
| `status_code` | Upstream or proxy status. `499` means the client disconnected mid-request and the upstream call was cancelled |
| `ttft_ms` | Time to first token (streaming) |
| `response_time_ms` | Total duration |
| Token counts | Prompt, completion, total; cache creation/read; cached prompt; audio in/out |
| `cost_usd` / `cache_savings_usd` | Billed cost and the estimated saving from cache reads |
| `log_metadata` | `fallback_attempts`/`fallback_count`/`fallback_providers`, `retry_attempts`/`retry_count`, routing fields, `provider_model_name` |
| `client_ip`, `user_agent`, `auth_method` | Attribution, computed through trusted-proxy resolution |
| Bodies | JSON, or `{"$binary": true, "size": N}` for binary payloads |

**Filtering** (`GET /api/logs`): page or cursor pagination (max 200 per page), free-text
`search`, date range, exact status or status range, model, provider, user, API key,
endpoint, and log type. Non-admins are automatically scoped to their own rows.

### Privacy controls

Configured in [Settings → Log Management](settings.md#log-management):

- **Sampling** (`sampling_rate`, default 1.0) limits **body capture**: sampled-out
  bodies are stored as `{"_sampled_out": true}` while metadata is still recorded.
  `x-log-full: true` forces full capture for one request.
- **Masking** (`mask_sensitive_data`, default on) replaces credentials in headers,
  bodies, and MCP/tool arguments. Values ≤ 8 chars become `***`; longer values keep
  their first 3 and last 4 characters. Add your own field names via **Extra Sensitive Keys**.
- **Retention** (`log_retention_days`, default 30; `0` = forever) is enforced by a
  sweep per log type that runs once a day. Audit retention can be set separately.

### Durability

Logs and usage are written through in-process queues with background batch writers.
A full queue **drops** records (audit writes log an error instead); write failures
retry with backoff and open a breaker after 5 consecutive failures. On shutdown the
queues are drained (bounded at 5 s). Audit rows use a dedicated writer so endpoint
traffic cannot crowd them out.

## Usage analytics

The **Usage** dashboard and `GET /api/logs/usage-stats` aggregate:

- Summary: total cost, requests, input/output tokens, success rate, average response
  time, average TTFT, average tokens/second, cache creation/read tokens, cache savings.
- Breakdowns by provider and by model, plus a daily series.

Metrics come from the dedicated `usage_records` table (kept 365 days), which survives
log deletion; request counts and success rate come from `request_logs` when available.
Averages are the only latency/throughput statistics reported — no percentiles. Cache
savings require a `cached_read_cost_per_1m` price on the model; without it the metric
is skipped with a warning.

## Audit Logs

Administrative and security events (logins, member and key management, provider key
reveals, and every non-`/v1/` admin operation) are written to a **hash-chained** audit
log — each row's hash includes the previous row's, so removing or editing history is
detectable.

- **Chain**: `content_hash` = SHA-256 over a canonical JSON dump of the event,
  outcome, resource, model/provider, cost, tokens, and bodies. The stored link hash is
  SHA-256 of `"{sequence}:{content_hash}:{previous_hash}"`, with `GENESIS` as the seed
  of the chain.
- **Verify**: Logs → Audit Logs → **Verify Integrity**
  (`GET /api/logs/audit/verify-integrity`, admin only, optional sequence range)
  recomputes every hash and checks the linkage, returning `valid`, `verified_count`,
  and any errors.

Audit rows are admin-only in the console and API.

## MCP Calls and Web Search tabs

| Tab | Contents |
| --- | --- |
| **MCP Calls** | One row per MCP operation: server, operation, resource type/name, arguments and result summary (masked), status, duration |
| **Web Search** | One row per intercepted search: query, provider, status, result count, max uses vs. current use |

## Tracing

Tracing is **per user** (Settings → General → Tracing) and currently supports
**Langfuse** plus the always-on internal audit handler. There is no global tracing
configuration and no OTLP exporter in v0.2.3.

When enabled, each request exports one generation observation:

- Name and model from the request, with model parameters.
- `input` = messages (or `{tools, messages}` when tools are present); `output` = the
  assistant message with tool calls separated out.
- `usage_details` (input/output/cache/audio tokens) and `cost_details.total`.
- Metadata: request id, trace id, user id, session id.

Correlate a trace with a request by sending `x-langfuse-trace-id` (or `x-trace-id`);
the proxy echoes a trace id header on responses and both headers are CORS-exposed.

## Related

- [Settings](settings.md#log-management) — retention, masking, sampling
- [Cost Control](../guides/cost-control.md) — turning usage data into budgets
