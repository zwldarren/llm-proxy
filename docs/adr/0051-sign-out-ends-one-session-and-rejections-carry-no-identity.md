# 0051 — Sign-out ends one session, and a rejected credential carries no identity

## Status

Accepted (2026-10). Amends ADR-0050's logout rule.

## Context

The console keeps one active session per user (ADR-0049) and binds every JWT to
it with a `sid` claim, so a newer sign-in rotates the previous session away. Two
invariants follow from that model, and both must hold everywhere:

- **Sign-out must be scoped to the caller's own session.** With one session per
  user, revoking by user is indistinguishable from a correct sign-out — until a
  second tab or device signs in. Signing in on device B rotates device A's
  session away; a "sign out" that revokes by user would then end B's live
  session and every JWT device B held. A browser that no longer holds a session
  must not be able to end the one that does.
- **A rejected credential must not be an identity.** `GET /api/auth/session` and
  `POST /api/auth/logout` are optional-auth paths: they must stay reachable
  without a live session (a stale cookie has to be clearable, and the console has
  to learn it is signed out). If the gate attributes the request to the token's
  named account before validating subject, token version and bound session —
  and returns "allowed" on these paths without undoing the attribution — a
  rotated or revoked session answers `authenticated: true` with its old username,
  the console renders signed-in chrome for a session every other endpoint
  rejects, and that stale identity is exactly what lets the sign-out above act
  as the account.

Both are the same principle in two places: "this request may proceed without a
session" is not "this request may be attributed to the session it presented".

## Decision

- **Sign-out is scoped to the caller's own session.** `POST /api/auth/logout`
  revokes the session its token's `sid` names, through
  `UserSessionRepository.deactivate_session(session_id, user_id)`, and touches
  nothing else: no `token_version` bump, no other session. Ending every session
  stays the account-wide revoke (`deactivate_user_sessions`) that password
  changes, admin resets, membership changes and the CLI use.
- **A token with no `sid`** has no single session to end, so sign-out keeps the
  account-wide fallback for that case only. No current login path mints such a
  token and they expire within the 24-hour JWT lifetime; the fallback exists so
  "sign out" is never a no-op.
- **A rejected credential leaves an anonymous request.** On optional-auth paths,
  every credential failure the gate can detect — unverifiable token, wrong
  token type, missing/inactive subject, stale `token_version`, revoked or
  expired session, or a database outage during subject validation — resets the
  identity to unauthenticated before the request proceeds. `GET /api/auth/session`
  then answers `authenticated: false` with no username, and `logout` acts as an
  anonymous request that only clears the cookies it was given.
- **Optional paths keep answering 200.** A stale cookie stays clearable and
  logout during a database outage still clears the browser's cookies; leniency
  just never means "authenticated as whoever the token named".

## Alternatives considered

- **Revoke by session but bump `token_version` too.** Bumping it invalidates
  every JWT the account holds, which defeats the point: another device's token
  dies even though its session was never touched.
- **Keep the stale identity and let each route decide.** Two routes would have
  to remember to re-validate, and every future optional-auth route would have to
  remember again. The gate owns the decision, so the gate clears the identity.
- **Reject on optional-auth paths (401 instead of anonymous).** It would make a
  stale cookie unremovable and make the console's own session probe fail, so a
  reload could not discover it is signed out — the same class of dead end the
  forced-change and MFA-enrollment allowlists exist to avoid.
- **Require `sid` on every token and drop the legacy grace.** No live token
  lacks `sid` after 24 hours, so the fallback is transitional by construction;
  removing it is a separate decision about the migration window, not part of
  these two rules.

## Consequences

- Signing out of one browser never signs the account out everywhere. The
  account-wide revoke remains available to the flows that mean it (password
  change, admin reset, membership change), and any future "sign out everywhere"
  button must call `deactivate_user_sessions` explicitly rather than relying on
  `logout`.
- `GET /api/auth/session` is a truthful probe: `authenticated: false` always
  means "nothing in this cookie can act", which is what the frontend's memoized
  session probe and its 401 handling assume.
- The gate's optional-auth branch has one exit shape — anonymous 200 — instead
  of "anonymous when no credential, named identity when validation failed".
