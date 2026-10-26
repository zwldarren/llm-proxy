# Security Hardening

LLM Proxy sits between your applications and paid upstream APIs, holds provider
credentials, and serves an admin console. This page is the deployment checklist worth
reviewing before exposing it to the internet.

## Credential model in one table

| Credential | Storage | Notes |
| --- | --- | --- |
| API keys (`sk-…`) | bcrypt hash | Plaintext shown once; used on `/v1/*` and `/servers/*` only |
| Console sessions | HS256 JWT, 24 h, no refresh | `/api/*` only; invalidated by password/role/deactivation changes |
| User passwords | bcrypt, policy 8–72 chars with mixed classes | Forced change for admin-created accounts |
| Provider API keys | Fernet-encrypted at rest (`enc:` prefix) | Reveal is admin-only and audited; rotating `ENCRYPTION_KEY` breaks them |
| JWT / encryption secrets | Auto-generated, stored in the database | Keep the database; a backup contains both |

## Edge and transport

| Item | Default | Recommendation |
| --- | --- | --- |
| TLS | Terminates at your reverse proxy | Always HTTPS in production; see [Reverse Proxy & TLS](../deployment/reverse-proxy.md) |
| HSTS | **on**, `max-age=31536000; includeSubDomains` | Keep on; disable only for local HTTP development |
| Security headers | `nosniff`, `DENY`, `Referrer-Policy`, `Permissions-Policy`, CSP `default-src 'self'` (script/style allow `unsafe-inline`) | Emitted by the app; no edge duplication needed |
| `TRUSTED_PROXIES` | RFC1918 + loopback + link-local | Replace with your actual proxy networks — it **replaces**, not extends. Spoofed `X-Forwarded-For` from untrusted peers is ignored by design |
| Request body limit | 64 MiB, enforced while streaming (chunked measured, not rejected) | Keep in sync with the reverse proxy's own limit |
| CORS | Disabled (empty origin list) | Add only origins that genuinely need browser access |

## Abuse protection

| Item | Default | Notes |
| --- | --- | --- |
| Rate limiting | Enabled (in-memory) | `rate_limit_disabled` is the master throttle switch: rate-limit buckets, per-key requests/minute, both failure lockouts, and the auth failure delay. It does **not** lift budget caps or the request size limit. The UI warns loudly when enabled |
| Shared rate limits | Off until `REDIS_RATE_LIMIT_ENABLED=true` | Required for multi-worker/multi-replica deployments: the memory backend keeps one window **per process**, so N workers enforce N times the configured limit. `GET /api/health` → `services.rate_limiting.backend` says which store is live |
| Redis fail mode | `redis_rate_limit_fail_closed=true` | Requests blocked when Redis errors — safer for public deployments |
| Login lockout | **Off by default.** When enabled: 5 failures / 15 min, keyed by username | Off because a hard per-account lockout lets anyone who knows the username lock the account out. The per-IP `auth.login` limit and the auth failure delay apply regardless |
| API-key IP lockout | **Off by default.** When enabled: 10 failures / 5 min, keyed by **client IP** | Off because the lockout is keyed by IP: one client with bad credentials locks out everyone sharing that address (NAT, corporate egress — and in development, every request is `127.0.0.1`). Turn it on for instances reachable from untrusted networks. Applies to `/v1/*` and `/servers/*`; the auth failure delay applies regardless |
| Model allowlist rejection | 403, not counted as an auth failure | The key is already authenticated, so requesting a disallowed model neither feeds the IP lockout nor pays the failure delay |
| WebSocket quotas | Budget checked before each billable turn | `/v1/responses` re-runs the key gate before every `response.create`; `/v1/realtime` checks before forwarding each `response.create` and again after a completed turn. Enforcement is best-effort: spend is cached for 15 s and usage rows are written asynchronously, so a burst or an in-flight turn can overshoot the cap |
| Auth failure delay | 100 ms ±10% jitter | Slows brute force; keep enabled on public instances |
| Console auth buckets | `auth.login` 5/min, `auth.setup` 5/min, `auth.setup_status` 10/min | Tunable per bucket |

## Keys, roles, and tenants

- **Least privilege**: `viewer` accounts cannot reach provider/model/team/MCP/settings
  APIs at all. Reserve `admin` for operators.
- **Key hygiene**: set expiry, `rate_limit_rpm`, model allowlists, and budgets on every
  production key. Remember that an explicitly empty allowlist denies everything.
- **Account budgets** cap the total across a member's keys and can only be reset by an
  admin.
- **Isolation guarantees**: viewers see only their own logs/usage; API keys are
  owner-scoped even for admins; stored Responses are namespaced per key so tenants
  cannot read or delete each other's.
- **Personal tracing**: a non-admin's `base_url` must be a public host; internal
  collectors must be listed in `HTTP_ALLOWED_HOSTS`. Admins keep the permissive
  self-hosted check. Demoting an admin to viewer clears their personal tracing
  configuration; they can re-add a public collector afterwards.

## Outbound media downloads

A client-supplied image URL reaches the network in two places: a chat message's
`image_url` on a provider that fetches it (Ollama downloads it into base64), and a
Decisions request's `input_image` when it is bridged onto a System One upstream
(none of them fetch a remote image, so the proxy does). Both use the same
SSRF-validated fetch:

- Loopback, RFC1918, link-local, CGNAT, multicast, the cloud metadata hosts, and
  URLs carrying credentials are refused outright.
- A hostname is resolved and every returned address is checked, so a public name
  pointing at a private IP is refused too.
- Redirects are followed (at most five), and **each hop is validated like the
  first** — an http→https upgrade or a CDN's signing hop still works, a redirect to
  `127.0.0.1` or to a name that resolves there does not.
- The body is read as a stream and capped at `HTTP_MAX_DOWNLOAD_BYTES` (20 MiB by
  default), so a URL pointing at a multi-gigabyte file costs one rejected request
  instead of the worker's memory. 20 MiB is the per-image ceiling of the largest
  mainstream vision APIs, so the cap never cuts an image an upstream would take.
- Downloads for one request are also bounded as a group: at most four run at once,
  and the inlined payloads may total `HTTP_MAX_REQUEST_DOWNLOAD_BYTES` (64 MiB by
  default, the same figure the proxy accepts in a request body). A request naming
  many URLs therefore cannot make the proxy hold hundreds of megabytes; peak
  memory is the total plus the in-flight downloads.
- On the bridged Decisions path, a refused or failed download is a `400` naming
  the URL: the request would otherwise be answered about evidence the model never
  saw. Chat media takes the other rule from ADR-0046 — the part is dropped and the
  request goes through.

The false positives this can produce are internal image hosts and hosts whose DNS
resolves into a private range (split-horizon DNS). List those hosts in
`HTTP_ALLOWED_HOSTS` to exempt them from the ban — the scheme and credential rules
still apply.

## Outbound requests

Every URL the proxy itself fetches — provider `base_url`, a personal tracing
collector, an MCP remote server, a web-search endpoint, a client's media URL — is
SSRF-checked when it is configured. Requests made through the proxy's shared
outbound session are checked again **when the connection is made**: the default
backend's DNS resolver refuses an answer in a banned network, so a hostname that
points at a public address when the settings row is saved and a private one later
(DNS rebinding) cannot be reached. Redirect targets are resolved and checked in both
backends. Literal addresses and `localhost` are only checked up front;
`HTTP_ALLOWED_HOSTS` exempts a host in both places. Forcing
`HTTP_CLIENT_BACKEND=httpx2` keeps the up-front and redirect checks but not the
connect-time resolver, and the MCP server client, the smart-routing judge/flywheel
clients and the Langfuse and OTLP/LangSmith exporters keep their own HTTP clients, so
they are covered by the configured-URL check only. See ADR-0047.

What the proxy reads back is bounded too:

- A buffered provider response is read until `HTTP_MAX_RESPONSE_BYTES` (64 MiB by
  default) and then refused with a `502`, so a broken or hostile endpoint cannot
  exhaust a worker's memory by answering with a huge body.
- One line of a streamed response is capped at 8 MiB. An endpoint that never sends
  a newline would otherwise grow the SSE read buffer without limit.
- An MCP tool call, prompt or resource is capped at 8 MiB across its content
  blocks (`MCP_MAX_RESULT_BYTES`); a cut payload is marked so the model can tell a
  truncated answer from a complete one.

## MCP

MCP servers run commands and reach the network, so the policy is deny-by-default:

- Allowlist the exact **commands** and **environment variables** each server needs.
  Shell metacharacters in arguments are rejected outright.
- SSRF protection resolves hostnames and rejects blocked IPs (loopback, RFC1918,
  link-local, cloud metadata, CGNAT) when a server is created or updated. The MCP
  SDK owns its HTTP client, so this check runs at configuration time rather than at
  connect time — see [Outbound requests](#outbound-requests).
- Keep `require_key_mcp_permissions=true` so every MCP server is explicitly granted per
  API key.
- Session slots are bounded per configured MCP server per worker (1000 sessions,
  10 min idle) so a key holder cannot exhaust an endpoint with an `initialize` loop.
- Remote (`streamableHttp`) servers cannot be given custom auth headers in v0.2.3 —
  prefer stdio, or a network path you control.
- Every MCP call is logged with masked arguments in Logs → MCP Calls.

See [MCP Servers](../admin/mcp.md#security-policy).

## Logging and privacy

- **Bodies are off by default** (`log_input_output`): every row still records status,
  tokens, cost, routing and audit metadata, but every content payload is stored as
  `{"_bodies_disabled": true}` — request/response bodies, upstream error bodies and
  messages, and MCP/web-search arguments and results. The row's free-text error reason
  and stack trace are kept (they are classification, not payload; rejection rows depend on
  them); the upstream error payload lives in `log_metadata.error_details` and is scrubbed.
  Turn it on deliberately — and back off afterwards when you need payload-level
  debugging. It is a master switch: `x-log-full: true` cannot re-enable bodies
  for a single request.
- **Masking** is on by default; add your own field names under **Sensitive keys**.
  Masked values keep their first 3 and last 4 characters (≤ 8 chars → `***`). The built-in
  set covers credentials in `Authorization`, `x-api-key`, and `Cookie`/`Set-Cookie`
  headers, plus bodies and MCP/tool arguments.
- **Sampling** reduces body capture (once bodies are enabled) without losing metadata;
  sampled-out bodies are stored as `{"_sampled_out": true}` (distinct from the
  disabled marker) and `x-log-full: true` forces full capture for a single request when
  debugging.
- **Retention** defaults to 30 days for logs, audit and usage records alike. Shorten it
  for sensitive workloads.
- **Audit chain**: verify integrity from Logs → Audit Logs → **Verify Integrity** on a
  schedule, and store the result. A failing chain means the audit history was altered.
- **No export endpoint exists** — treat database access as access to all logs, and
  restrict it accordingly.

## Secrets and rotation

| Action | Consequence | Procedure |
| --- | --- | --- |
| Rotate an API key | None beyond the key | Create replacement, switch clients, delete old |
| Reset a user password | Sessions revoked, forced change at next sign-in | Admin: Team → Reset password |
| Rotate provider API keys | None | Edit the provider (reveal is audited) |

Rotating `JWT_SECRET` or `ENCRYPTION_KEY` reaches beyond the console —
`ENCRYPTION_KEY` in particular makes stored provider keys undecryptable until they
are re-entered. Both are auto-generated on first run, and the escrow/migration SQL
for reading them back out of the database lives in
[Upgrades & Backups → Secrets: what rotation breaks](../deployment/upgrades.md#secrets-what-rotation-breaks).

## Air-gapped or privacy-strict deployments

- Set `UPDATE_CHECK__ENABLED=false` to stop the version check calling GitHub.
- Tracing is per user and opt-in; leaving it disabled keeps request content inside
  your infrastructure.
- Web search and MCP are disabled/deny-by-default until explicitly configured.

## Related

- [Server Settings](../admin/settings.md#security-rate-limiting) — every threshold above
- [Logs, Usage & Tracing](../admin/observability.md) — audit verification
- [Upgrades & Backups](../deployment/upgrades.md#secrets-what-rotation-breaks) — secret persistence
