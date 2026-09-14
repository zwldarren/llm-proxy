---
pageClass: api-reference kicker-overview
aside: false
---

# Errors & Rate Limits

## Error envelopes

The proxy answers in the protocol the client spoke.

### OpenAI

```json
{
  "error": {
    "message": "Model 'foo' not found in configuration",
    "type": "not_found_error",
    "code": "model_not_found",
    "param": null,
    "error_id": "9f2c…"
  }
}
```

`code` and `param` are omitted when unset — the example shows the richest shape.

### Anthropic

```json
{
  "type": "error",
  "error": { "type": "not_found_error", "message": "…" }
}
```

Proxy-generated Anthropic errors carry no `request_id`; correlate them via the
`X-Request-Id` response header instead. A `request_id` appears only when the upstream
error body is already provider-shaped and is passed through verbatim, preserving the
provider's own id.

### Responses API

Same shape as OpenAI, with `type` remapped to spec codes:
`invalid_request_error → invalid_request`, `authentication_error →
authentication_failed`, `permission_error → permission_denied`, `not_found_error →
not_found`, `rate_limit_error → too_many_requests`, `api_error`/`provider_error` →
`server_error`.

Admin API (`/api/*`) rejections from the JWT middleware (missing/invalid token,
revoked token, disabled account) and request-validation failures use FastAPI's
`{"detail": …}` shape. Exceptions raised inside admin routers go through the app-wide
handlers and use the same protocol envelope as everything else.

`error_id` is the request id — search it in **Logs** to see the full request.
Unhandled 5xx messages are sanitized to `Internal server error`; details stay in the
server logs.

## Status codes

| Status | Typical codes | Meaning |
| --- | --- | --- |
| 400 | `ValidationError`, `invalid_content_length`, `payload_too_deeply_nested` | Malformed request or body |
| 401 | `authentication_error` | Missing/unknown API key |
| 403 | `model_not_allowed`, `forbidden`, `password_change_required` | Not permitted by allowlist/role/account state; deactivated accounts get `{"detail": "Account is disabled"}` on `/api/*` |
| 404 | `model_not_found`, `not_found`, `mcp_server_not_found`, `provider_not_configured` | Unknown model/response/server; stored responses expire after 24 h |
| 409 | `conflict` | Duplicate feedback, cancel of a non-cancellable response |
| 413 | `body_size_exceeded` | Over `max_request_body_size_bytes` (default 10 MiB), or chunked transfer while the limit is active |
| 429 | `rate_limit_exceeded`, `too_many_auth_failures`, `budget_exceeded`, `user_budget_exceeded` | Rate limit, lockout, or budget cap. Rate-limit, per-key RPM, and `/v1/*` lockout 429s carry `Retry-After`; the `/servers/*` MCP lockout 429 and budget rejections do not |
| 499 | `client_disconnected` | Client/CDN gave up; upstream cancelled (logged as failure) |
| 500 | `configuration_error`, `internal_error` | Bad configuration (e.g. smart routing disabled, missing model), routing constraints unsatisfiable, or upstream responses that fail to parse (500 `api_error`) |
| 502 | `provider_error`, `network_error` | Upstream failed |
| 503 | `redis_not_available`, `budget_check_unavailable`, `web_search_error`, readiness | Dependency missing/unhealthy; budget checks fail closed |
| 504 | `timeout_error` | Upstream read timeout (600 s) |

## Headers

| Header | When |
| --- | --- |
| `X-Request-Id` | Always (main middleware stack) — the id shown in the Logs screen and in `error_id` |
| `Retry-After` | On rate-limit, `/v1/*` lockout, and forwarded upstream 429s (the `/servers/*` MCP lockout 429 and budget rejections do not set it) |
| Upstream `x-ratelimit-*` / `anthropic-ratelimit-*` / `retry-after` | Forwarded on streaming responses |

There are no proxy-generated `X-RateLimit-*` headers. Note that an inbound
`x-request-id` is used as the tracing/routing id, while the response header is always
the server-generated id.

## Rate limits

| Limit | Scope | Default |
| --- | --- | --- |
| Per-key RPM | One API key, sliding 60-second window | None (unlimited) unless set on the key |
| `auth.login` | Client IP | 5/minute |
| `auth.setup` | Client IP | 5/minute |
| `auth.setup_status` | Client IP | 10/minute |
| Failed API-key attempts | Client IP | 10 failures → 300 s lockout (`too_many_auth_failures`) |
| Failed logins | Username | 5 failures → 900 s lockout (`account_locked`) |

- The auth buckets are configurable per bucket in
  [Settings → Advanced → Rate Limits](../admin/settings.md#rate-limits) using `N/period`
  (`5/minute`, `100/hour`).
- Per-key limits (`rate_limit_rpm`) are set when creating/editing a key and are
  admin-only.
- When Redis-backed rate limiting is enabled, windows are shared across workers and
  the fail mode is controlled by `redis_rate_limit_fail_closed` (default: block on
  Redis errors).

## Budgets

Budget rejections are 429s with distinct codes:

| Code | Budget |
| --- | --- |
| `budget_exceeded` | The API key's budget |
| `user_budget_exceeded` | The owner's account budget (admin-managed) |

Budget checks fail **closed**: if spend cannot be read, the request is rejected
(`503 budget_check_unavailable`).

## Retries and fallback

Client-visible failures happen only after the proxy's own retry/fallback chain is
exhausted:

- Retries within a provider for `408/429/500/502/503/504` and network/timeout errors
  (`max_retries`, default 3).
- Fallback across providers up to `max_fallback_attempts` (default 10); `4xx` client
  errors skip retries and move to the next provider immediately.

If everything fails, the last upstream error (or the routing constraint error) is
what the client receives. The log entry records all attempts
(`fallback_count`, `retry_count`).
