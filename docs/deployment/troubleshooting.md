# Troubleshooting

Start with the health endpoints, then work outward.

## Health endpoints

All three are unauthenticated:

| Endpoint | Returns |
| --- | --- |
| `GET /api/health` | Always 200. `status` plus `services.database` (a `SELECT 1`) and `services.redis` (`status`, `healthy`, `enabled`; `disabled` when Redis is off) |
| `GET /api/health/ready` | 200 `{"status":"ready"}` when healthy, otherwise **503** — use this for orchestrator readiness |
| `GET /api/health/live` | Always 200 `{"status":"alive"}` — liveness only, no dependency checks |

## Startup failures

The process fails loudly rather than starting half-configured. Find the error in the
container/console logs:

| Error | Cause / fix |
| --- | --- |
| Migration error on boot | Database unreachable, credentials wrong, or a failed migration. The port is never bound; fix the database and restart |
| `Could not find alembic.ini` | Running from an installation that lost `alembic.ini` (it must be in the CWD or repo root) |
| Redis `ConfigurationError` at boot | `REDIS_ENABLED=true` but the server is unreachable — there is no silent degradation at startup |
| Invalid `TRUSTED_PROXIES` CIDR | Fix the value; startup validates it |
| Settings validation error | e.g. `DB_POOL_SIZE < 1`, `REDIS_TIMEOUT <= 0` |
| Config validation error | A stored provider/model/user record is invalid — the message names the record |

## Symptom → cause → fix

| Symptom | Cause | Fix |
| --- | --- | --- |
| `401 authorization header missing` / `invalid authorization header format` | No key sent, or `Authorization` does not use `Bearer` | Send `Authorization: Bearer sk-…` or `x-api-key: sk-…`. Console JWTs do **not** work on `/v1/*` |
| `429 too_many_auth_failures` with `Retry-After` | Too many invalid keys from this IP (default: 10 in 300 s) | Wait out `Retry-After`; check the key |
| `403 model_not_allowed` | Key or user allowlist excludes the model | Adjust the allowlist, or use an allowed model. Remember: empty allowlist = deny all |
| `404 model_not_found` | Model name is not in the model mappings | Add the mapping (or fix the name). Virtual models need smart routing enabled |
| `500 configuration_error` mentioning a virtual model | `auto`/`fast`/`best` used while smart routing is **disabled**, or on a non-chat endpoint | Enable smart routing, or use a concrete model name |
| `429 rate_limit_exceeded` | Per-key RPM cap or an auth bucket | Raise `rate_limit_rpm`, or wait for the fixed 60 s window |
| `429 budget_exceeded` / `user_budget_exceeded` | Key or account budget exhausted | Raise the budget or reset it (account budgets only by an admin) |
| `413 body_size_exceeded` | Body above `max_request_body_size_bytes` (64 MiB default) | Raise the limit or shrink the request |
| `499 client_disconnected` in logs | Client/CDN gave up mid-request | Usually a CDN timeout — keep keepalive enabled, or stream instead of waiting on a long non-streaming call |
| Response arrives as **200 with an error JSON body** | Keepalive heartbeat mode had already committed 200 before the failure | Expected tradeoff; check the body, and the real error is in the logs. See [Reverse Proxy & TLS](../deployment/reverse-proxy.md) |
| Browser reports a CORS failure instead of a real status | Auth/body/rate-limit middleware run **outside** CORS | Reproduce with `curl` to see the true status; then fix the underlying error |
| Streaming request stalls with no data | Upstream silence; the proxy is sending `: keep-alive` comment frames | This is normal. If your reverse proxy buffers responses, disable buffering |
| Admin UI missing at `/` (API works) | `frontend/dist` did not exist at startup | Run `uv run llm-proxy --build-frontend` and restart |
| Logs screen shows rows but bodies are `{"_sampled_out": true}` | `log_input_output` is off (bodies scrubbed, metadata rows still written), or `sampling_rate < 1.0` | Re-enable in Settings → Log Management, or send `x-log-full: true` on a request to force full capture for it |
| Usage stats empty but logs exist | Usage and logs are written separately | Check the date range; usage records are pruned on the log retention window |
| `503 redis_not_available` on Responses endpoints | Redis is not enabled, but stored responses need it | Enable Redis, or avoid the response-storage endpoints |
| MCP server refuses to start | Deny-by-default policy: its command or env var is not allowlisted | Allowlist the exact command in Settings → Advanced → MCP Security (see [MCP Servers](../admin/mcp.md)) |
| MCP `403 Access denied to MCP server` | Key allowlist excludes the server | Update the key's MCP allowlist |
| MCP returns 404 for a server that exists | Server is not running (stopped/disabled, or crash at startup) | Check status in the console; disable + re-enable to restart |
| MCP changes don't take effect | A running server is not restarted by an update | Disable, then re-enable the server |
| `web_search` tool calls return `web_search_tool_result_error` | Search backend unreachable / rate-limited / bad key | The model still answers; check the Web Search log tab and the backend config |
| Smart routing picks unexpected models | Classifier signals or mode weights | Review Settings → Advanced → Smart Routing; use routing diagnostics (`verbose_routing_logs`) and per-request feedback |
| Claude Code warns the session "isn't eligible" for no-charge auto mode | The upstream cannot run server-side auto-mode checks | See [Claude Code auto mode](#claude-code-auto-mode) below |

## Claude Code auto mode

Claude Code (v2.1.278+) with auto mode asks the upstream to run its safety checks by
adding a `safeguards` request field and the `dangerous-tool-use-2026-09-03`
`anthropic-beta` value; the upstream must answer with a `safeguard_results` response
field (streamed: inside the terminal `message_delta` delta), keyed by tool-use id.
When neither half comes back, Claude Code falls back to its own classifier — billed
as token usage — and shows a notice the first time it would have checked an action.

Nothing breaks when this happens; auto mode keeps working.

- **Non-Anthropic upstreams cannot serve this at all** — Ollama, vLLM, SGLang, and the
  third-party Claude-compatible providers (z.ai, Kimi, MiniMax, Moonshot, DeepSeek,
  Qwen) have no server-side check. The notice is expected on those routes.
- **An Anthropic-format upstream** works when the fields survive end to end. This proxy's
  native Anthropic passthrough forwards `safeguards`, the full `anthropic-beta` value,
  and every response key verbatim; the converted path keeps `safeguard_results` too.
- **Silence the notice** by setting `CLAUDE_CODE_AUTO_MODE_SERVER=0` in the environment
  Claude Code starts from. It then never asks for server-side checks and the notice
  disappears.

To confirm which state a session is in, run `/status` in Claude Code and check the
**Auto mode server** row: `Enabled` means the server is performing the checks,
`Disabled` means the session has fallen back.

## Getting more detail

- **Request id** — every response carries `X-Request-Id`; search the Logs screen for
  it. Note that an inbound `x-request-id` is used as the tracing/routing id, while the
  response header is always the server-generated id.
- **Application logs** — `LOG_LEVEL=DEBUG` (restart required) or `--log-file`.
- **Audit trail** — admin operations and security events are in Logs → Audit Logs, with
  **Verify Integrity** to prove the chain is intact.
- **Provider failures** — errors surface with the upstream status and message; 502
  means provider/network/parse failure, 504 a timeout (upstream calls use a fixed
  600 s read timeout with a 10 s connect; the provider `timeout` field is not applied).
