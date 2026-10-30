# Users & Roles

LLM Proxy is multi-tenant: every person gets their own account, their own API keys,
their own logs, and optionally their own model allowlist and spending cap.

**Team** (admin only) is where accounts are created and managed.

## Roles

| | `admin` | `viewer` |
| --- | --- | --- |
| Usage & logs | All users | Own usage and logs only |
| Audit log | View + verify integrity | No access |
| Providers / Models management / MCP Servers / Team / server settings | Full | No access (403) |
| API keys | Own keys; may set MCP allowlist and per-key rate limits | Own keys; may set the MCP allowlist; no per-key rate limit |
| Account rate limit | Set any member's cap in Team | Read-only (enforced on every key) |
| Playgrounds | Yes | Yes |

Promote a member by changing their role in **Team**. A role change revokes all of
that member's active sessions — they must sign in again.

## Inviting a member

**Team → Invite Member** creates a single-use, 7-day link for an email address.
The link is shown once and can be copied; if SMTP is configured the proxy sends it
automatically. Re-sending rotates the token and invalidates the previous link.

The invitee opens the link, chooses a username and password, and is signed in. The
inviter's role, model allowlist and account budget are applied at acceptance. No
administrator-chosen temporary password is ever transmitted.

The invitation strip above the member table keeps every invitation, not just the
open ones: pending rows offer **Resend** and **Revoke**, while accepted, revoked and
expired rows move into **Past invitations** with the timestamp that settled them.

If SMTP is not configured, the copyable link is the delivery mechanism: possession
of the link is the credential, so share it only with the invitee. With SMTP, the
email address is verified before the invite is accepted. OAuth invitations follow
the same rule and only create an account when the provider reports a verified email
matching the invitation.

Existing users add or change their login email from **Account → Email**. The
proxy sends a 24-hour verification link; only the link's recipient can complete the
change. The link is single-use: it is pinned to the address the account had when it
was sent, so it stops working once the address changes (and cannot be replayed to
resurrect a superseded one). Without SMTP, an administrator can set the email in
**Team** and the address is marked as admin-attested.

An OAuth invitation asks the invitee to choose a username before the first console
session is created; the account is not silently named after the email local part.

## Creating a member manually

**Team → Add Member**:

| Field | Notes |
| --- | --- |
| **Username** | Up to 64 characters; letters, digits, `_` and `-` |
| **Password** | 8–72 characters with uppercase, lowercase, digit, and special character |
| **Role** | `viewer` (default) or `admin` |
| **Allowed models** | No selection = all models; an explicit empty selection = deny all |

New members (and any member whose password an admin resets) must change their password
at first sign-in. Until they do, every endpoint except profile/password/logout is
blocked with `403 password_change_required`, and the console funnels them to the
**Set a New Password** screen.

::: warning Empty allowlist means no access
Clearing the model selection is not the same as "unrestricted" — it denies every
model. Click **Clear selection** only if you intend to lock the member out of
model access entirely.
:::

Password changes bump the account's token version, invalidating all previously issued
JWTs, and deactivate all session API keys — the user must sign in again.

### Forgetting your password

**Forgot password?** on the sign-in page emails a single-use link to the account's
**verified** email address. The link is valid for 30 minutes; choosing a new password
there ends every existing session on the account, so the old password stops working
everywhere at once.

The entry is only a link when the instance can actually deliver mail — SMTP enabled
and a public base URL set in **Instance sign-in**. Without them it reads "ask an
administrator"; the **Team** member password reset (see [Member lifecycle](#member-lifecycle))
is the recovery path there. Accounts with no password (passkey-only or OIDC-only),
no email address, or an unverified (admin-attested) address are never reset by mail:
they recover with a passkey, a TOTP recovery code, or an administrator reset. The
endpoint answers the same way for every address, so the page cannot be used to find
out which accounts exist.

## Passkeys and two-factor authentication

Users manage their password, passkeys and TOTP in the **Sign-in & security** tab of
**Account**. First-run setup lands on this page (with a passkey prompt) after the
admin account is created, so the first session does not end with a password as the
only way back in.

- **Password** is optional. A passwordless account (passkey or OIDC only) can set
  one from the same page; the proxy asks for a fresh re-authentication instead of
  a current password.

- **Passkeys** are a primary login factor. The console supports discoverable
  credentials (no username needed), multiple named passkeys per user, and synced
  passkeys. User verification (biometric/PIN) is required. Passkeys have an inline
  rename action, since the name is how you tell a laptop from a phone. A
  passwordless account cannot remove its last passkey unless it also has a password
  or a linked OAuth provider; losing the last usable factor requires an
  administrator reset.
- **Two-factor authentication** is optional per user. Enrollment shows a QR code for
  the authenticator app, with the setup secret and `otpauth://` URI as the manual
  fallback. When enabled, a TOTP code is required after *every* primary factor —
  password, passkey or OIDC — otherwise the strongest factor would bypass 2FA. Ten
  one-time recovery codes are shown once, with copy and download actions; the panel
  keeps reminding you that this is the only time they are visible until you confirm
  you stored them.
- Administrators can require TOTP for every admin in **Instance sign-in**. An admin
  who has not enrolled yet is redirected to **Account → Sign-in & security** and
  blocked from the rest of the console until they do; the panel explains why. The
  same policy makes "disable 2FA" unavailable to those admins.
- Changing a password, passkey, 2FA state or linked provider requires a recent
  re-authentication (15-minute step-up window).
- From **Team**, an administrator can reset a member's TOTP/recovery codes or revoke
  all of their passkeys. Revoking passkeys is refused with `last_factor` when the
  member has no password and no linked OIDC provider, because that would lock them out.

Passkeys and OIDC require a canonical HTTPS `public_base_url`. A plain-HTTP or
LAN-IP deployment can still use invitations and TOTP; the passkey/OAuth endpoints
return `passkey_unavailable` / `oauth_unavailable` until TLS and a hostname are set,
and the sign-in page omits the passkey button entirely when the browser has no
WebAuthn support or the page is not a secure context.
Changing the public URL invalidates existing passkeys because the WebAuthn relying
party ID changes.

When configuring an OIDC provider in **Instance sign-in**, the page shows the exact
callback URL to register at the identity provider
(`<public_base_url>/api/auth/oauth/<provider id>/callback`). The provider ID is
immutable once saved, so that URL is stable for as long as the public base URL is.

## Model allowlists

Each member can be restricted to a set of model names.

- Applied to the Models screen (read-only catalog for viewers), to `/v1/models`, and
  to every inference request.
- A viewer's API keys can only narrow this list further — the intersection is enforced
  when the key is created or updated.
- `null`/empty selection at the account level means unrestricted; `[]` means deny all.

## Account budgets

Beyond per-key budgets (see [API Keys](api-keys.md#budgets)), an admin can set an
**account budget** for a member. It caps the combined spend of *all* of that member's
keys in the same UTC calendar windows.

- Requests over the account budget fail with **429** and code `user_budget_exceeded`.
- Only an admin can reset it: **Team → member → Reset budget**.
- The member can see their own status at `GET /api/me/budget`.

## Account rate limits

Per-key rate limits are owner-managed and admin-only to set, so an admin cannot
apply one to a member's key. The member-wide control is an **account rate limit**,
set from **Team → member → Manage rate limit** (or
`PUT /api/team/members/{id}/rate-limit`).

- It is a sliding 60-second window across **all** of the member's keys, including
  the UI session key; `null` means unlimited.
- It is enforced independently of any per-key limit: either cap can reject with
  **429** and code `rate_limit_exceeded`.
- Key-level limits remain available on the admin's own keys in **API Keys**.

## Member lifecycle

| Action | Effect |
| --- | --- |
| **Edit username / password** | Username change is immediate; password reset forces a change at next sign-in and revokes sessions |
| **Change role** | Sessions revoked, token version bumped |
| **Deactivate** | Signs the member out and stops their keys; keys and usage history are preserved |
| **Reactivate** | Restores sign-in and key usage |
| **Delete** | Revokes active sessions, permanently removes the member **and their API keys**; usage history remains |

Safety rules enforced by the proxy:

- You cannot delete, deactivate, or demote yourself.
- You cannot delete, demote, or deactivate the **last active admin**.
- Deactivating or deleting a member is irreversible from their side — the console
  asks for confirmation, with a stricter prompt when the member is an admin.

## Sign-in protection

- `POST /api/auth/login` is rate-limited per IP (default 5 requests/minute).
- Account lockout is **off by default**. When enabled, failed logins are counted
  **per username**: after 5 failures the account is locked for 15 minutes, and
  rotating source IPs does not bypass the lockout. It is off by default because a
  hard per-account lockout is itself a denial-of-service vector — anyone who knows
  the username can lock the account out. The per-IP limit above and the auth failure
  delay apply either way.
- Usernames that do not exist still pay the same bcrypt cost, so timing does not
  reveal whether an account exists.
- Successful sign-in clears the failure counter and is recorded in the audit log.

Both thresholds are configurable in [Server Settings](settings.md#security-rate-limiting).

## Self-service account endpoints

The Account workspace exposes these under three tabs: **Profile** (identity, budget,
current session), **Preferences** (theme and language, stored in this browser, plus
your own tracing backends) and **Sign-in & security**.

Any signed-in user can:

| Endpoint | Purpose |
| --- | --- |
| `GET /api/me/profile` | Profile and role |
| `GET /api/me/budget` | Account budget status |
| `PUT /api/me/password` | Change own password |
| `PUT /api/me/username` | Change own username (requires current password) |
| `GET /api/me/2fa` | 2FA status and remaining recovery codes |
| `POST /api/me/2fa/begin` / `confirm` / `disable` | Enroll or remove TOTP |
| `POST /api/me/email` | Send a verification link to a new login email |
| `GET /api/me/passkeys` | List registered passkeys |
| `POST /api/me/passkeys/register/begin` / `finish` | Register a passkey |
| `PATCH/DELETE /api/me/passkeys/{id}` | Rename or delete a passkey |
| `GET /api/me/oauth/identities` | List linked OIDC identities |
| `DELETE /api/me/oauth/identities/{id}` | Unlink a provider identity |
| `POST /api/me/oauth/link/complete` | Finish linking a provider identity (the callback only verifies it; binding needs the signed-in session) |
| `POST /api/me/2fa/recovery-codes` | Regenerate the 2FA recovery codes |
| `POST /api/auth/email/verify` | Complete an email verification link |
| `GET/PUT /api/me/tracing/` | Personal tracing backend configuration |
| `POST /api/me/feedback` | Rate a smart-routing decision for a request |
| `POST /api/auth/logout` | Sign out of this browser (revokes only the console session its own token's `sid` names; a legacy token without `sid` falls back to the account-wide revoke) |
