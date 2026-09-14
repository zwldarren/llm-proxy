---
pageClass: api-reference kicker-overview
aside: false
---

# Authentication

Two credential systems exist, and they never overlap:

| Credential | Valid on | Purpose |
| --- | --- | --- |
| **API key** (`sk-…`) | `/v1/*`, `/servers/*` | Your applications and MCP clients |
| **Console JWT** (from sign-in) | `/api/*` | The admin UI and admin API |

A valid console JWT is **rejected** on `/v1/*` and `/servers/*`, and an API key is
rejected on `/api/*`.

## API keys

Accepted forms:

```http
Authorization: Bearer sk-your-proxy-key
```

```http
x-api-key: sk-your-proxy-key
```

- Format: `sk-` followed by 64 hex characters. Keys are stored bcrypt-hashed; the
  plaintext is shown once at creation.
- Session keys created by the console for the playgrounds (`sk-ui-…`) are also accepted
  on proxy routes and inherit their owner's restrictions.
- A non-`Bearer` `Authorization` value → `401 Invalid authorization header format`;
  no header at all → `401 Authorization header missing`.

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "x-api-key: $LLM_PROXY_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "hi"}]}'
```

### WebSockets

`/v1/responses` and `/v1/realtime` accept the key via `Authorization`, `x-api-key`, or
an `api_key` query parameter (Realtime additionally supports the
`openai-insecure-api-key.<key>` subprotocol). Prefer headers — query strings may be
captured by reverse-proxy access logs.

## Console JWT

Obtained from `POST /api/auth/login` (or `POST /api/auth/setup` on first run) and sent
as `Authorization: Bearer <jwt>` to `/api/*`. Tokens are HS256, valid 24 hours, with no
refresh endpoint. Changing your password, changing your role, or being deactivated
invalidates all existing tokens server-side.

## Public endpoints

No credential is required for:

- `GET /api/health`, `/api/health/ready`, `/api/health/live`
- `POST /api/auth/setup`, `GET /api/auth/setup-status`, `POST /api/auth/login`
- The admin UI's static files (the SPA itself)

Everything else under `/api/*` needs a JWT; everything under `/v1/*` and `/servers/*`
needs an API key.

## Failure handling

| Situation | Response |
| --- | --- |
| Missing/unknown key | 401 with the protocol's error envelope |
| Repeated invalid keys from one IP | **429** `too_many_auth_failures` (default: 10 failures → 300 s lockout; counters are per process and reset on restart). The `/v1/*` 429 carries `Retry-After`; the `/servers/*` 429 does not |
| Model not in the key's allowlist | 403 `model_not_allowed` |
| MCP server not in the key's allowlist | 403 `Access denied to MCP server '<name>'` |
| Budget exhausted | 429 `budget_exceeded` (or `user_budget_exceeded` for account budgets) |
| Failed auth | Responses are delayed by `auth_failure_delay_ms` (default 100 ms, ±10% jitter) to slow brute force |

Failed authentication attempts, lockouts, and logins are recorded in the audit log.

## Related

- [API Keys](../admin/api-keys.md) — creating keys with budgets and allowlists
- [Errors & Rate Limits](errors.md) — exact response bodies
