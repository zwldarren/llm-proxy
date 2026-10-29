# 0049 — Console auth: invites, passkeys, TOTP and OIDC

## Status

Accepted (2026-10).

## Context

Before this decision the console's only credential path was username + bcrypt
password with a 24-hour JWT, and new members received an admin-chosen temporary
password. Invitations, OAuth, passkeys and two-factor authentication must be added
while the `/v1/*` API-key surface stays unchanged.

The four features sit on different layers: invitations are account lifecycle,
passkeys and TOTP are authentication factors, and OIDC is external trust. Adding
them independently would produce conflicting recovery and session semantics.

## Decision

- **Identity.** `username` remains the canonical identifier. A nullable, unique,
  lowercased email is a login alias. OIDC identities bind by a verified email only
  through an invitation or an authenticated link; there is no silent merge.
- **Account creation.** Invite-only. Invitations are single-use, 7-day, hashed at
  rest, revocable, and can be re-sent (which rotates the token). A copyable link is
  the delivery mechanism; SMTP is optional. Without SMTP, the admin attests the
  address by delivering the link. Administrators may still create an account
  directly with a temporary password (the documented escape hatch for deployments
  without SMTP); that password must be changed on first sign-in.
- **Passkeys.** First-class primary factor with discoverable credentials and required
  user verification. Multiple named credentials per user; synced passkeys allowed; no
  attestation requirement. The last passkey cannot be removed without another usable
  primary factor (a password or a linked OIDC provider).
- **TOTP.** Optional per user, enforceable for admins. It applies after every primary
  factor, so a passkey cannot bypass it. Ten hashed, one-time recovery codes are
  generated on enrollment.
- **OIDC.** Generic, configurable providers (issuer, client ID/secret, scopes). Login
  requires either a matching invitation or an authenticated link; conflicts are
  rejected. OAuth additionally depends on external provider configuration and a
  usable public URL.
- **Sessions.** One active console session per user, bound to the JWT `sid` claim.
  A new login revokes the previous session immediately. No refresh token.
- **Step-up.** State-changing team/API-key operations and security-factor changes
  require a session verified within 15 minutes. Legacy tokens without `sid` remain
  valid until their 24-hour expiry.
- **Rate limits.** The new endpoints each get a per-IP bucket: `auth.mfa`,
  `auth.step_up` and `auth.invite_accept` at 10/minute; `auth.invite_status`,
  `auth.passkey`, `auth.oauth`, `auth.email_verify` and `auth.capabilities` at
  20/minute; `auth.password_reset` at 5/minute and `auth.password_reset_confirm`
  at 10/minute.
- **Password reset by mail.** `POST /api/auth/password/forgot` takes a username or
  verified email and always answers `202 {"requested": true}` — same status, same
  body, and mail queued after the response — whether the account exists, is
  deactivated, is passwordless, has an unverified email, or SMTP is disabled. The
  endpoint must not become an account oracle. Eligibility is narrow: a link is
  mailed only to a **verified** address on an account that already has a bcrypt
  password; admin-attested addresses do not qualify (a typo there would hand out
  the account). The link is a scoped JWT, 30 minutes, single use, carrying `uid`
  and a SHA-256 fingerprint of the stored password hash; applying it changes the
  password, so the fingerprint no longer matches and a replay is refused
  (`password_reset_superseded`) — no token table is needed. `POST
  /api/auth/password/reset` sets the password, clears `must_change_password` (the
  user chose it), bumps `token_version` and deactivates every console session: a
  reset is how a locked-out or compromised account gets back in, so older
  credentials must die. No SMTP means no link; `GET /api/auth/capabilities`
  reports `password_reset` from `smtp.enabled` plus a usable `public_base_url`,
  and the sign-in page renders a link when it is true and an "ask an
  administrator" line when it is false. Excluded by design: email OTP, refresh
  tokens, account merging. Recovery is passkeys, the permanent password fallback,
  the self-service reset link and administrator reset; TOTP recovery codes remain
  a second-factor fallback only.
- **Deployment gate.** Passkeys and OIDC require `public_base_url` over HTTPS (or
  localhost). Plain HTTP/LAN-IP deployments get invitations and TOTP only.

## Consequences

Operators must configure a stable HTTPS URL before passkeys/OIDC; changing that
URL invalidates registered passkeys. The `/v1/*` API-key contract is untouched.
Second-factor and token material is not cheap to brute-force: recovery codes use
a transcription-safe alphabet (no `0`/`O`/`1`/`I`/`L`) and are hashed with an
iterated KDF — a bare digest of a ~50-bit code is enumerable offline. Email
verification links are pinned to the address at send time, which makes them
single-use without server-side state.

## Security invariants

- **Account linking is two-phase.** The provider callback is unauthenticated, so it
  only verifies *which* identity the IdP vouched for and hands back a 10-minute
  `oauth_link` token; the binding itself requires
  `POST /api/me/oauth/link/complete` with the session that started the link (`sid`
  pinned in the state claims). A leaked state never binds an identity by itself.
- **PKCE lives in an HttpOnly cookie.** The `code_verifier` is never part of `state`:
  `state` travels through the provider, browser history and IdP logs, so a verifier
  inside it protects nothing against the interception it exists to stop.
- **Invitation rotation reaches in-flight OAuth flows.** The state pins the invitation
  `token_hash`; a flow started from a link that has since been re-sent is refused.
- **Single use is atomic.** Invitation acceptance and recovery-code consumption are
  conditional UPDATEs, and the last-factor guards (passkey delete, provider unlink,
  TOTP state changes) run under a row lock on the account, so concurrent requests
  cannot each observe a spare factor.
- **Revocation is not the only audit signal.** Failed TOTP, failed passkey login and
  rejected OIDC callbacks are recorded as `Outcome.FAILURE`, not just successes.
- **Step-up covers every write that weakens a control**, including the `logging` and
  `rate_limits` settings (full prompt logging; relaxing the login lockout bucket).

