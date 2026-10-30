# Server Settings

**Settings** holds everything that is not a provider, model, key, or user: logging,
features, resilience, and security policy. Every section here is written to the
database and **hot-reloaded** — no restart, ever.

Hot reload covers every worker process, not just the one that served the request:
a peer picks up the shared revision on its next request (checked at most once per
second) and re-applies the setting, including the state that is not part of the
config row itself — the web search interceptor, circuit-breaker thresholds, MCP
server processes, and personal tracing registries. API-key and team-role edits
propagate the same way, so a revoked key or a demoted role takes effect on every
worker in about a second instead of waiting out the in-process cache TTL. This
needs Redis (`REDIS_ENABLED=true`), which the shipped compose sets. Without Redis
a multi-worker deployment only hot-reloads the worker that handled the change; the
provider/model lookup cache is a separate, optional switch (`REDIS_CACHE_ENABLED`).

The screen has two tabs: **General** and **Advanced**, and it is admin-only: every
setting here is instance-wide. Theme, language and per-user tracing are personal
settings and live in **Account → Preferences** instead. The **Sections** menu beside
the tabs jumps to a section and names the one you are currently reading, so you can
tell where you are without opening it.

## General tab

### Log Management

The console groups these controls into **Capture**, **Retention**, and **Audit** so
body capture, lifecycle policy, and audit overrides can be scanned independently.
Grouping does not change the underlying hot-reloaded settings.

| Field | Default | Effect |
| --- | --- | --- |
| **Log Input/Output** | off | Master switch for persisting bodies. Off, the log rows are still written (status, tokens, cost, routing, audit metadata) with every content payload stored as `{"_bodies_disabled": true}` — request/response bodies, upstream error bodies and messages, and MCP/web-search arguments and results. On, bodies are stored — bounded by **Max Logged Body Bytes** and filtered by **Body Sampling Rate**. The switch wins over `x-log-full: true`: bodies cannot be re-enabled per request |
| **Log Raw Stream** | off | Store the raw SSE text of streaming responses. Off, a streamed response is reassembled into the same JSON a non-streaming call would return — smaller, masked like any other body, and rendered without SSE parsing. The native-passthrough tiers are reassembled too (Chat Completions, Anthropic and Responses frames are each rebuilt into their non-streaming shape); generic image streams keep their raw frames, since they have no non-streaming shape to rebuild. Turn on (or send `x-log-full: true` for one request) to inspect the exact wire frames |
| **Log Retention Period** | `30` | Rows older than this are deleted by a sweep per log type that runs once a day, deleting in bounded batches (looped until exhausted) instead of one unbounded one; `0` keeps logs forever. Usage records (the numbers behind the dashboard) share this window. Keep this at ≥ 31 days while monthly or lifetime budgets are in use: budget enforcement sums spend from `usage_records` over the whole calendar budget window (a monthly budget can look back up to 31 days), so a shorter window silently under-counts spend and budgets stop being enforced reliably. Without a monthly/lifetime budget, shorter values are fine |
| **Audit Retention Days** | inherits | Separate retention for audit rows |
| **Body Sampling Rate** | `1.0` | Fraction of requests whose bodies are stored, once **Log Input/Output** is on. Sampling gates **body capture only** — metadata is always recorded, and a sampled-out body is stored as `{"_sampled_out": true}` (distinct from the disabled marker). A client can force full capture for one request with `x-log-full: true` |
| **Max Logged Body Bytes** | `1048576` (1 MiB) | Cap on stored request/response bodies. A body bigger than the cap is stored as `{"_truncated": true, "size": N}` (the original serialized size in bytes) instead of the payload. Enforced at write time, so it also applies to requests that forced capture with `x-log-full: true`. `0` = unlimited |
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
| **SearXNG** | `url` (required), optional `api_key` or basic-auth username/password, `engines` list, `timeout` (default 30 s), `max_results` (default 10, max 10) |
| **Ollama** | `api_key` (required), `base_url` (default `https://ollama.com`), `timeout`, `max_results` (max 10) |

Search credentials are write-only in the API: they are encrypted at rest (when
`ENCRYPTION_KEY` is configured) and reads return `****`. Submitting the mask or a
blank field keeps the stored secret; to remove one, clear the whole provider
block. A value that looks like the mask but is not exactly `****` is rejected
with `422` instead of silently replacing the stored credential — clear the field
and enter the full new secret to rotate it.

Details and client-visible behavior: [Tools, Reasoning & Web Search](../api/tools.md#web-search).

### MCP Security

Deny-by-default policy for MCP servers: allowed/blocked commands, env filtering, and
URL/IP blocks. Full reference: [MCP Servers](mcp.md#security-policy).

### About

Running version and update check against GitHub tags (cached 6 hours). Disable the
outbound check with `UPDATE_CHECK__ENABLED=false`.

## Advanced tab (admin only)

### Request Policy

How the proxy treats fields and content blocks it does not understand. The setting is
global, but **Provider default** leaves the decision to each provider's adapter (most
drop unknown fields; vLLM/SGLang pass them through so engine-specific options such as
`top_k`/`guided_json`/`chat_template_kwargs` survive). A concrete value overrides every
provider's own default.

| Field | Options | Default |
| --- | --- | --- |
| **Unknown fields policy** | `provider default` · `ignore` · `passthrough` · `error` | **Provider default** — per adapter (usually `ignore`) |
| **Unsupported block policy** | `provider default` · `drop` · `degrade` · `error` | **Provider default** — per adapter (currently `drop`) |

Unknown-fields policy applies to **top-level** request fields the internal model does
not recognize; unknown fields nested inside messages are handled by each serializer.
Requests that take the native-passthrough tier (Anthropic and Responses clients on
supporting providers) bypass it — their bodies are forwarded verbatim.

For content blocks, lossless conversion runs first: a block the provider can represent
(after conversion) is never affected by the policy. The policy governs the residual
blocks with no representation — `drop` removes them, `degrade` keeps a text placeholder
(`[Image: …]`), and `error` rejects the request. `degrade` never silently falls back to
dropping: a block with no specific placeholder gets a generic `[<TypeName> block]`.

### Smart Routing

| Field | Default | Effect |
| --- | --- | --- |
| **Enabled** | off | Enables the virtual models `auto`, `fast`, `best` |
| **Mode weights** | `fast 0.35`, `auto 0.65`, `best 1.0` | Quality-vs-cost bias per virtual model |
| **Routing Judge** | off | Consults a System One model on ambiguous first turns; see [Routing Judge](../api/routing.md#routing-judge) |
| **Routing Diagnostics** | off | Adds the full routing decision payload to smart-routing log metadata (diagnostics) |
| **Graceful Degradation** | off | When routing cannot satisfy a preference (mode assignment, output half of the context window), serve degraded instead of refusing; see [Virtual Models & Routing](../api/routing.md) |

The signal ensemble runs in-process (metadata + structural signals; the embedding
signal was removed with the heavy ML stack that backed it). The optional routing
judge is the only part of smart routing that makes an extra
upstream call — one short System One request per judged turn, bounded by its gate
and deadline, with the verdict and its cost recorded in the request's routing
metadata. Each such call is also a row of its own in Logs → **Judge Calls**
(log type `judge`), attributed to the request that triggered it; that row is what
spend and budget aggregates count. Per-request feedback can be submitted from the log
detail view.

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

### Instance Sign-in

| Field | Default | Effect |
| --- | --- | --- |
| Public base URL | empty | Canonical external URL (`https://proxy.example.com`). WebAuthn RP ID and OAuth redirect URIs derive from it. Must be HTTPS, or `http://localhost` for development. Changing it invalidates registered passkeys |
| Require admins to use 2FA | off | Blocks admins that have not confirmed TOTP; the console funnels them to **Account → Sign-in & security** |
| SMTP enabled | off | Send invitation, email-verification and password-reset messages. Without it, invitations are copyable links and a forgotten password goes to an administrator |
| SMTP host / port / username / password / from address | empty / `587` / empty / empty / empty | Standard SMTP delivery. The password is encrypted at rest and masked in the API response |
| STARTTLS / TLS (SMTPS) | on / off | Transport security for the SMTP connection |
| OIDC providers | empty | Named OpenID Connect providers (`id`, display name, issuer, client ID/secret, scopes). OAuth buttons appear only for enabled providers. Secrets are encrypted at rest and masked on read |

SMTP details appear after delivery is enabled, and saved OIDC providers start
collapsed. The form validates URLs, required fields and the SMTP port before
saving; **Discard changes** restores the last saved snapshot, and leaving with
unsaved edits requires confirmation.

The SMTP password and OIDC client secrets are write-only: reads return `****`.
Submitting that exact mask or a blank value keeps the stored secret. Editing the
mask in place is rejected rather than overwriting the real credential; clear the
field and enter the full new secret to rotate it.

An OIDC provider's **id** is permanent once saved: the stored client secret is
keyed by it, so the console locks the field for an existing provider. Each
**issuer** must be an HTTPS URL (or `http://localhost` for development); the
proxy refuses to save a plain-HTTP issuer, because the client secret and the
authorization code travel to it.

Passkey and OAuth endpoints are unavailable when no HTTPS public base URL is
configured. This is deliberate: WebAuthn cannot be made secure over plain HTTP or a
LAN IP.

### Security & Rate Limiting

| Field | Default | Effect |
| --- | --- | --- |
| Login lockout enabled | off | Lock an account after repeated failed logins (keyed by username). Off by default — a hard per-account lockout lets anyone who knows the username lock the account out on purpose |
| API-key IP lockout enabled | off | Lock a client IP after repeated failed API key authentications. Off by default — the lockout is keyed by IP, so one client with bad credentials locks out everyone sharing that address (NAT, corporate egress, or every local test request arriving as `127.0.0.1`). Turn it on for instances reachable from untrusted networks |
| Max failed login attempts / lockout duration | `5` / `900` s | Used only when login lockout is enabled |
| Max failed API-key attempts / lockout duration | `10` / `300` s | Used only when API-key IP lockout is enabled |
| Auth failure delay | `100` ms | Artificial delay on failed auth (±10% jitter) |
| Rate limiting disabled | off | Master throttle switch: disables the rate-limit buckets, per-key requests/minute, both failure lockouts and the auth failure delay. Budget caps and the body-size limit are unaffected. The UI warns loudly when on — the proxy becomes brute-force/DoS-exposed |
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
(`5/minute`, `100/hour`). Editable buckets and their defaults:

| Bucket | Default |
| --- | --- |
| `auth.login` | `5/minute` |
| `auth.setup` | `5/minute` |
| `auth.setup_status` | `10/minute` |
| `auth.capabilities` | `20/minute` |
| `auth.mfa` | `10/minute` |
| `auth.step_up` | `10/minute` |
| `auth.invite_accept` | `10/minute` |
| `auth.invite_status` | `20/minute` |
| `auth.passkey` | `20/minute` |
| `auth.oauth` | `20/minute` |
| `auth.email_verify` | `20/minute` |
| `auth.password_reset` | `5/minute` |
| `auth.password_reset_confirm` | `10/minute` |
| `account.sensitive` | `10/minute` |

Unknown bucket names or unparseable values are rejected with 422, and the count
must be positive (`0/minute` is rejected rather than silently disabling the
bucket). Per-API-key limits (`rate_limit_rpm`) are set on the key itself.

### CORS Origins

List of origins allowed to call the API from a browser. Empty (default) disables CORS
entirely — correct for same-origin deployments. See
[Reverse Proxy & TLS](../deployment/reverse-proxy.md#cors) for the interaction with
auth failures.

## Related

- [Logs, Usage & Tracing](observability.md)
- [Environment Variables](../reference/environment.md) — the startup-only settings
