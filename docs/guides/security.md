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
| Rate limiting | Enabled (in-memory) | Setting `rate_limit_disabled` warns loudly and exposes brute-force/DoS |
| Shared rate limits | Off until `REDIS_RATE_LIMIT_ENABLED=true` | Required for multi-worker/multi-replica deployments |
| Redis fail mode | `redis_rate_limit_fail_closed=true` | Requests blocked when Redis errors — safer for public deployments |
| Login lockout | **Off by default.** When enabled: 5 failures / 15 min, keyed by username | Off because a hard per-account lockout lets anyone who knows the username lock the account out. The per-IP `auth.login` limit and the auth failure delay apply regardless |
| API-key lockout | 10 failures / 5 min, keyed by **client IP** | Applies to `/v1/*` and `/servers/*` |
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

## MCP

MCP servers run commands and reach the network, so the policy is deny-by-default:

- Allowlist the exact **commands** and **environment variables** each server needs.
  Shell metacharacters in arguments are rejected outright.
- SSRF protection resolves hostnames and rejects blocked IPs (loopback, RFC1918,
  link-local, cloud metadata, CGNAT) — including DNS-rebinding tricks.
- Keep `require_key_mcp_permissions=true` so every MCP server is explicitly granted per
  API key.
- Remote (`streamableHttp`) servers cannot be given custom auth headers in v0.2.3 —
  prefer stdio, or a network path you control.
- Every MCP call is logged with masked arguments in Logs → MCP Calls.

See [MCP Servers](../admin/mcp.md#security-policy).

## Logging and privacy

- **Masking** is on by default; add your own field names under **Sensitive keys**.
  Masked values keep their first 3 and last 4 characters (≤ 8 chars → `***`).
- **Sampling** reduces body capture without losing metadata; `x-log-full: true` forces
  full capture for a single request when debugging.
- **Retention** defaults to 30 days (logs and audit); usage records are kept 365 days.
  Shorten both for sensitive workloads.
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
