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

Obtained from `POST /api/auth/login` (or `POST /api/auth/setup` on first run). Tokens
are HS256, valid 24 hours, with no refresh endpoint. Changing your password, changing
your role, or being deactivated invalidates all existing tokens server-side.

The browser console does **not** keep the JWT in JavaScript-accessible storage. Login
sets two `HttpOnly`, `SameSite=Lax` cookies (`Secure` on HTTPS):

| Cookie | Path | Contents |
| --- | --- | --- |
| `llm_proxy_console` | `/api` | Console JWT |
| `llm_proxy_session_key` | `/v1` | Short-lived session API key used by the playgrounds |

Unsafe cookie-authenticated requests must also send `X-CSRF-Token` matching the
readable `llm_proxy_csrf` cookie (double-submit CSRF). `GET /api/auth/session` reports
the cookie's current identity without failing on a stale cookie: a token whose
session was rotated away or revoked answers `authenticated: false` with no username
rather than naming the account it used to belong to. Signing out clears all three
cookies and revokes **its own** server-side session — a browser whose session another
sign-in already rotated away cancels nothing, so the newer session stays live.

Scripted clients may continue to send the returned `access_token` as
`Authorization: Bearer <jwt>`; an explicit `Authorization` or `x-api-key` header always
takes precedence over a cookie. On `/v1/*`, the session-key cookie is accepted only
when neither header is present and the request carries a valid CSRF token for unsafe
methods.

The login identifier is either a username or a verified email alias. If the account
has TOTP enabled, the response is `{ "mfa_required": true, "challenge_token": "…" }`
instead of an access token; complete it with `POST /api/auth/2fa/verify` (or
`/api/auth/2fa/recover` for a one-time recovery code). Passkey and OIDC logins use the
same second step.

Only one console session is active per user. The JWT carries a `sid` claim; signing in
on another device rotates the session and immediately invalidates the previous JWT.
Sensitive operations additionally require a step-up within 15 minutes
(`POST /api/auth/step-up` with the current password, or the passkey step-up endpoints).

Console auth failures use these stable codes so the UI can react without parsing prose:

| Code | Status | Meaning |
| --- | --- | --- |
| `step_up_required` | 403 | The session's last strong authentication is older than 15 minutes; call `POST /api/auth/step-up` (or the passkey step-up flow) and retry the action |
| `session_revoked` | 401 | A newer sign-in rotated the single active session; sign in again |
| `password_change_required` | 403 | An admin-created or admin-reset password must be replaced before other API access |
| `mfa_enrollment_required` | 403 | The instance requires admins to enroll TOTP; only the enrollment flow, the security page's reads, step-up, profile and logout routes are reachable |
| `mfa_challenge_expired` | 401 | The short-lived 2FA challenge expired; restart sign-in |
| `invalid_current_password` | 400 | A self-service password or username change supplied the wrong current password; the console session remains valid |
| `invalid_mfa_code` | 400 | The submitted TOTP or recovery code is not valid |
| `last_factor` | 400 | The change would remove the account's last usable sign-in factor |
| `oauth_onboarding_only` | 400 | A username/password may only be chosen while completing a *new* invited OAuth account |
| `username_required` | 400 | A new invited OAuth account must choose a username before the first console session |
| `oidc_id_token_invalid` | 401 | The provider's ID token failed signature, issuer, audience or nonce validation |
| `email_verification_superseded` | 401 | The verification link is older than the account's current address (already used, or the address changed since it was sent) |
| `password_reset_invalid` | 401 | The password reset link is malformed, expired, or belongs to an account no reset link may act on (deactivated, passwordless, unverified email) |
| `password_reset_superseded` | 401 | The reset link was already used, or the password changed after it was sent; request a new one |
| `oauth_link_expired` | 401 | The provider link hand-off token expired; start the link again |
| `oauth_link_mismatch` | 401 | The link hand-off token belongs to a different account |
| `oauth_link_session_changed` | 401 | The console session that started the link is no longer the active one |

## Public endpoints

No credential is required for:

- `GET /api/health`, `/api/health/ready`, `/api/health/live`
- `POST /api/auth/setup`, `GET /api/auth/setup-status`, `POST /api/auth/login`
- `GET /api/auth/capabilities` — what the sign-in screen may offer here
  (`password_reset` is true only when SMTP delivery and a public base URL are set)
- `POST /api/auth/password/forgot`, `POST /api/auth/password/reset` — the reset
  request always answers `202 {"requested": true}` and never discloses whether an
  account exists; the reset link is the credential on the confirm step
- `GET /api/auth/session` (optional credential: returns `{ "authenticated": false }`
  instead of a 401 when no valid cookie is present)
- `POST /api/auth/2fa/verify`, `POST /api/auth/2fa/recover`
- `POST /api/auth/email/verify`
- `GET /api/auth/invitations/{token}`, `POST /api/auth/invitations/{token}/accept`
- `GET /api/auth/oauth/providers`, the OAuth begin/callback/complete routes
  (the **callback** only verifies the provider's answer; finishing an account
  **link** requires `POST /api/me/oauth/link/complete` with the console session
  that started it. The PKCE verifier is handed to the callback in an HttpOnly
  cookie, never through `state`.)
- `POST /api/auth/passkeys/login/begin`, `POST /api/auth/passkeys/login/finish`
- The admin UI's static files (the SPA itself)

Everything else under `/api/*` needs a JWT; everything under `/v1/*` and `/servers/*`
needs an API key.

## Failure handling

| Situation | Response |
| --- | --- |
| Missing/unknown key | 401 with the protocol's error envelope |
| Repeated invalid keys from one IP | **429** `too_many_auth_failures` when API-key IP lockout is enabled (off by default; 10 failures → 300 s, counters per process and reset on restart). The `/v1/*` 429 carries `Retry-After`; the `/servers/*` 429 does not |
| Model not in the key's allowlist | 403 `model_not_allowed`. Not an auth failure: the key is already authenticated, so this never feeds the IP lockout and adds no delay |
| MCP server not in the key's allowlist | 403 `Access denied to MCP server '<name>'` |
| Budget exhausted | 429 `budget_exceeded` (or `user_budget_exceeded` for account budgets) |
| Failed auth | Responses are delayed by `auth_failure_delay_ms` (default 100 ms, ±10% jitter) to slow brute force |

Failed authentication attempts, lockouts, and logins are recorded in the audit log.

## Related

- [API Keys](../admin/api-keys.md) — creating keys with budgets and allowlists
- [Errors & Rate Limits](errors.md) — exact response bodies
