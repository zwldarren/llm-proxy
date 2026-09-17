# Server Settings

**Settings** holds everything that is not a provider, model, key, or user: logging,
features, resilience, and security policy. Every section here is written to the
database and **hot-reloaded** — no restart, ever.

The screen has two tabs: **General** and **Advanced**. Preference and Tracing are
available to all users; everything else is admin-only.

## General tab

### Preference

Theme (light/dark/system) and language (English / 中文). Stored in the browser, not on
the server.

### Log Management

| Field | Default | Effect |
| --- | --- | --- |
| **Log Input/Output** | on | Master switch for persisting bodies. Turning it off keeps the log rows (status, tokens, cost, routing, audit metadata) but stores bodies as `{"_sampled_out": true}` |
| **Log Raw Stream** | off | Store the raw SSE text of streaming responses. Off, a streamed response is reassembled into the same JSON a non-streaming call would return — smaller, masked like any other body, and rendered without SSE parsing. The native-passthrough tiers are reassembled too (Chat Completions, Anthropic and Responses frames are each rebuilt into their non-streaming shape); generic image streams keep their raw frames, since they have no non-streaming shape to rebuild. Turn on (or send `x-log-full: true` for one request) to inspect the exact wire frames |
| **Log Retention Period** | `30` | Rows older than this are deleted by a sweep per log type that runs once a day; `0` keeps logs forever. Usage records (the numbers behind the dashboard) share this window |
| **Audit Retention Days** | inherits | Separate retention for audit rows |
| **Body Sampling Rate** | `1.0` | Fraction of requests whose bodies are stored. Sampling gates **body capture only** — metadata is always recorded. A client can force full capture for one request with `x-log-full: true` |
| **Audit Sampling Rate** | inherits | Same, for audit rows |
| **Mask Sensitive Data** | on | Masks credentials in headers/bodies before storage |
| **Extra Sensitive Keys** | empty | Extra comma-separated field names to mask, on top of the built-in list (`authorization`, `api_key`, `password`, `token`, `access_token`, `refresh_token`, `jwt_secret`, …) |
| **Manual Cleanup** | — | The **Cleanup Logs** action permanently deletes rows older than the chosen age. Irreversible |

Usage records (the numbers behind the dashboard) are stored separately but pruned on the
same window as the logs; set **Log Retention Period** to `0` to keep both forever.

### Web Search

| Field | Notes |
| --- | --- |
| **Enabled** | Off by default. When on, the proxy intercepts `web_search` tool calls for providers that lack native search |
| **Provider** | `searxng` or `ollama` |
| **SearXNG** | `url` (required), optional `api_key` or basic-auth username/password, `engines` list, `timeout` (default 30 s), `max_results` (default 10, max 20) |
| **Ollama** | `api_key` (required), `base_url` (default `https://ollama.com`), `timeout`, `max_results` (max 10) |

Details and client-visible behavior: [Tools, Reasoning & Web Search](../api/tools.md#web-search).

### Tracing

Per-user Langfuse configuration: enable tracing and add one or more backends. Settings
apply **only to your own requests** — there is no global tracing switch.

| Field | Notes |
| --- | --- |
| **Provider** | `langfuse` (and the always-on internal `audit_log` handler) |
| **Public key / Secret key** | Required; secrets are masked in API responses |
| **Base URL** | Default `https://cloud.langfuse.com`; self-hosted instances are allowed |
| **Sample rate** | 0.0–1.0 |
| **Timeout, version** | Optional tuning |

Correlate traces with requests using the `x-langfuse-trace-id` request header; the
proxy also returns a trace id header. Up to 10 handlers per user, deduplicated.

### About

Running version and update check against GitHub tags (cached 6 hours). Disable the
outbound check with `UPDATE_CHECK__ENABLED=false`.

## Advanced tab (admin only)

### Request Policy

How the proxy treats fields it does not understand:

| Field | Options | Default |
| --- | --- | --- |
| **Unknown fields policy** | `ignore` · `passthrough` · `error` | `ignore` — unknown top-level request fields are dropped |
| **Unsupported block policy** | `drop` · `degrade` · `error` | `drop` — unsupported content blocks are dropped |

Requests that take the native-passthrough tier (Anthropic and Responses clients on
supporting providers) bypass the unknown-fields policy — their bodies are forwarded
verbatim.

### Smart Routing

| Field | Default | Effect |
| --- | --- | --- |
| **Enabled** | off | Enables the virtual models `auto`, `fast`, `best` |
| **Mode weights** | `fast 0.35`, `auto 0.65`, `best 1.0` | Quality-vs-cost bias per virtual model |
| **Routing Diagnostics** | off | Adds the full routing decision payload to smart-routing log metadata (diagnostics) |

The classifier runs in-process — no extra LLM call. With the `smart-routing` extra
installed, an embedding signal sharpens the prediction; without it, structural signals
are used. Per-request feedback can be submitted from the log detail view.

See [Virtual Models & Routing](../api/routing.md).

### Provider Selection

How providers of the same priority are ordered when several can serve a model:

| Strategy | Behavior |
| --- | --- |
| `random` (default) | Random pick within the priority group |
| `session_sticky` | Reuses the provider chosen earlier in the same conversation when Redis is available; without Redis (or without a conversation key) it degrades to deterministic rendezvous hashing |
| `cost_optimized` | Cheapest usable provider first |
| `balanced` | Blends cost and observed latency/error statistics |

Higher-priority groups are always tried before lower ones, regardless of strategy.

### Retry & Fallback

| Field | Default | Effect |
| --- | --- | --- |
| **Max retries** | `3` | Retries against the same provider (per-model override available on the model) |
| **Max fallback attempts** | `10` | Maximum provider hops for one request |

### Circuit Breaker

| Field | Default | Effect |
| --- | --- | --- |
| **Enabled** | on | Skip providers that are failing |
| **Failure threshold** | `5` | Consecutive failures before a provider is tripped |
| **Cooldown** | `60` s | Time before the provider is retried |

The tab lists live breaker state per provider and offers **Reset all** / per-provider
reset. Counters are per process and reset on restart.

### Security & Rate Limiting

| Field | Default | Effect |
| --- | --- | --- |
| Login lockout enabled | off | Lock an account after repeated failed logins (keyed by username). Off by default — a hard per-account lockout lets anyone who knows the username lock the account out on purpose |
| Max failed login attempts / lockout duration | `5` / `900` s | Used only when login lockout is enabled |
| Max failed API-key attempts / lockout duration | `10` / `300` s | Key lockout, keyed by client IP (always on) |
| Auth failure delay | `100` ms | Artificial delay on failed auth (±10% jitter) |
| Rate limiting disabled | off | Master switch. The UI warns loudly when disabled — the proxy becomes brute-force/DoS-exposed |
| Redis fail-closed | on | With Redis rate limiting, block requests when Redis errors (instead of allowing them) |
| HSTS enabled / max-age | on / `31536000` | `Strict-Transport-Security` header. Disable only for local HTTP development |
| Max request body size | `67108864` (64 MiB) | `0` disables. A declared `Content-Length` is checked up front; chunked or undeclared bodies are measured as they stream and rejected only if they exceed the limit |

Lockout counters live in-process; a restart clears them. Rate-limit windows are
in-memory unless Redis-backed rate limiting is enabled.

### Response Keepalive

| Field | Runtime default | Effect |
| --- | --- | --- |
| **Enabled** | on | Start heartbeat mode for slow non-streaming responses |
| **Grace period** | `60` s | How long the real response may take before heartbeats begin |
| **Interval** | `15` s | Heartbeat cadence (also used for SSE `: keep-alive` comments) |

Read the caveats before changing anything here:
[Reverse Proxy & TLS](../deployment/reverse-proxy.md#cloudflare-and-other-impatient-cdns).

### Rate Limits

Per-bucket overrides for the built-in IP limits, in `N/period` form
(`5/minute`, `100/hour`). Editable buckets:

| Bucket | Default |
| --- | --- |
| `auth.login` | `5/minute` |
| `auth.setup` | `5/minute` |
| `auth.setup_status` | `10/minute` |

Unknown bucket names or unparseable values are rejected with 422. Per-API-key limits
(`rate_limit_rpm`) are set on the key itself.

### CORS Origins

List of origins allowed to call the API from a browser. Empty (default) disables CORS
entirely — correct for same-origin deployments. See
[Reverse Proxy & TLS](../deployment/reverse-proxy.md#cors) for the interaction with
auth failures.

### MCP Security

Deny-by-default policy for MCP servers: allowed/blocked commands, env filtering, and
URL/IP blocks. Full reference: [MCP Servers](mcp.md#security-policy).

## Related

- [Logs, Usage & Tracing](observability.md)
- [Environment Variables](../reference/environment.md) — the startup-only settings
