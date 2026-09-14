# Monitoring & Alerting

Everything the proxy knows about its own behavior is queryable through the API, so
you can build dashboards and alerts without touching the internals.

## Health

| Endpoint | Use |
| --- | --- |
| `GET /api/health` | Full snapshot: database and Redis status, always HTTP 200 |
| `GET /api/health/ready` | **503** when not ready — use for load balancer/orchestrator readiness |
| `GET /api/health/live` | Always 200 — use for liveness probes |

```bash
curl -s http://localhost:8080/api/health/ready | jq .
```

## Traffic and reliability

| Signal | Where |
| --- | --- |
| Requests, cost, tokens, success rate | `GET /api/logs/usage-stats` or the **Usage** dashboard |
| Average response time, average TTFT, tokens/second, cache savings | Same response (`summary`) |
| Provider/model breakdown, daily series | `by_provider`, `by_model`, `daily_usage` |
| Error rate by status | `GET /api/logs?status_code=…` / `status_code_from`/`status_code_to` |
| Retries and fallbacks | Log metadata: `retry_count`, `fallback_count`, `fallback_providers` |
| Circuit breaker state | `GET /api/config/circuit-breaker` (admin), with reset endpoints |
| Latest log timestamp + total count (lightweight, built for polling) | `GET /api/logs/stats` |
| Client disconnects | Logs with status **499** |

::: note Averages, not percentiles
Latency and throughput reporting is average-based; there are no p50/p95/p99
endpoints in v0.2.3. For percentiles, export request logs
(`GET /api/logs`, cursor pagination, up to 200 per page) into your own store.
:::

## Things worth alerting on

| Condition | Why |
| --- | --- |
| `/api/health/ready` != 200 | Database down, or Redis enabled but unhealthy |
| Status 5xx trending up | Provider failures, timeouts, configuration errors (502 = upstream/parse, 504 = timeout) |
| `fallback_count > 0` repeatedly | A preferred provider is sick |
| Circuit breaker trips | Provider outage; the proxy is already avoiding it |
| 499s trending up | Clients/CDNs giving up — check keepalive settings and CDN timeouts |
| 429 `budget_exceeded` | A client hit its cap (intended or a runaway loop) |
| 429 `too_many_auth_failures` | Wrong key deployed, or someone probing |
| Audit chain verification fails | Tampering or corruption — investigate immediately |

```bash
# 5xx in the last hour
curl -s "http://localhost:8080/api/logs?status_code_from=500&status_code_to=599&page_size=200" \
  -H "Authorization: Bearer $ADMIN_JWT" | jq '.items | length'

# Audit integrity (admin)
curl -s "http://localhost:8080/api/logs/audit/verify-integrity" \
  -H "Authorization: Bearer $ADMIN_JWT" | jq '{valid, verified_count, errors}'
```

## External systems

| System | Integration |
| --- | --- |
| **Langfuse** | Per-user tracing (Settings → General → Tracing). One generation per request: messages in, assistant message out, usage and cost details, user/session ids |
| **Log aggregation** | The proxy does not push logs anywhere (Redis log shipping is not implemented in v0.2.3). Pull from `GET /api/logs`, or scrape the application log output |
| **Application logs** | `LOG_LEVEL` (startup), `--log-file`, container stdout |
| **OTLP/OpenTelemetry** | Not implemented in v0.2.3 (only `langfuse` and the internal `audit_log` handlers exist) |

Trace correlation: send `x-langfuse-trace-id` (or `x-trace-id`) on a request and it
becomes the trace id in Langfuse; responses carry a trace id header, and both headers
are CORS-exposed for browser clients.

## Operational hygiene

- **Retention**: logs default to 30 days, audit inherits that, usage records 365 days.
  Adjust in [Settings → Log Management](../admin/settings.md#log-management), or delete
  manually with `DELETE /api/logs/cleanup?older_than_days=N` (admin-only; without the
  parameter it applies the configured retention window).
- **Sampling**: keep `sampling_rate` at 1.0 while investigating; lower it to cut
  storage without losing metadata.
- **Backups**: usage and logs live in the database — see
  [Upgrades & Backups](../deployment/upgrades.md#backups).
- **Queue drops**: if you see dropped-record warnings, increase
  `LOG_WRITE_QUEUE_SIZE` / `USAGE_WRITE_QUEUE_SIZE` (restart required).
