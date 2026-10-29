# 0050 — Browser console credentials live in HttpOnly cookies

## Status

Accepted (2026-10).

## Context

The browser console previously stored its JWT and the `/v1` session API key in
`localStorage`. Any XSS could read both and keep a working credential after the
tab closed, and because the tokens were attached by JavaScript on every request,
the frontend had to parse the JWT to learn the username and role.

The console API and proxy API deliberately use different credentials: a console
JWT is valid only on `/api/*`, while `/v1/*` accepts API keys. Scripted clients
must keep using explicit headers.

## Decision

- **Transport.** The browser receives two `HttpOnly`, `SameSite=Lax` cookies
  (`Secure` on HTTPS): the console JWT on `/api` and the short-lived session API
  key on `/v1`. Both expire with the 24-hour server session. The frontend never
  reads or persists either value.
- **CSRF.** Unsafe requests authenticated by a cookie must send
  `X-CSRF-Token` equal to the readable `llm_proxy_csrf` cookie (double-submit).
  The CSRF token is rotated on login and cleared on logout. `SameSite=Lax` is a
  second, independent barrier against cross-site POSTs.
- **Precedence.** An explicit `Authorization` or `x-api-key` header always wins
  over a cookie, so SDKs, tests and the documented admin API contract are
  unchanged; the cookie is a browser transport, not a new authorization scheme.
- **Identity.** `GET /api/auth/session` reports the cookie's account, role and
  forced-change flag without failing on a stale cookie; the frontend uses that
  response instead of decoding a JWT, and a logout or `401` invalidates the
  memoized probe.
- **Logout.** The response clears all three cookies and revokes the session the
  token's `sid` names — its own session only (see ADR-0051). The token version is
  not bumped: another device's session must survive a stale tab signing out.
  Password changes revoke every session; stale cookies then resolve to an
  unauthenticated probe on the login screen.

## Alternatives considered

- **Keep the JWT in memory only.** A reload would sign the operator out, and the
  `/v1` session key would still need a persistent home for the playgrounds.
- **HttpOnly JWT only, keep the `/v1` key in storage.** Leaves the credential
  that can spend money readable by XSS.
- **Authorization header for every same-origin call.** This is the status quo
  that requires JavaScript-readable credentials and manual header plumbing.
- **Strict SameSite alone.** Blocks cross-site POSTs but does not protect against
  same-site attackers or browser quirks; the explicit CSRF token keeps the
  defense visible and testable.

## Consequences

The browser exposes no bearer token to page scripts, and the
`/api/auth/session` probe is the single source of UI identity. A reverse
proxy must preserve cookies and the `X-CSRF-Token` header. Cross-origin console
deployments are not supported by `SameSite=Lax` cookies; the console is served
from the same origin as its API. The login response keeps returning an
`access_token` for scripted clients; the browser ignores it.
