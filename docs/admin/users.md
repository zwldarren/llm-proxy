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
| API keys | Own keys; may set MCP allowlist and rate limits | Own keys; MCP allowlist forced to "all", no rate limit |
| Playgrounds | Yes | Yes |

Promote a member by changing their role in **Team**. A role change revokes all of
that member's active sessions — they must sign in again.

## Creating a member

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

## Member lifecycle

| Action | Effect |
| --- | --- |
| **Edit username / password** | Username change is immediate; password reset forces a change at next sign-in and revokes sessions |
| **Change role** | Sessions revoked, token version bumped |
| **Deactivate** | Signs the member out and stops their keys; keys and usage history are preserved |
| **Reactivate** | Restores sign-in and key usage |
| **Delete** | Permanently removes the member **and their API keys**; usage history remains |

Safety rules enforced by the proxy:

- You cannot delete, deactivate, or demote yourself.
- You cannot delete, demote, or deactivate the **last active admin**.
- Deactivating or deleting a member is irreversible from their side — the console
  asks for confirmation, with a stricter prompt when the member is an admin.

## Sign-in protection

- `POST /api/auth/login` is rate-limited per IP (default 5 requests/minute).
- Failed logins are counted **per username**: after 5 failures the account is locked
  for 15 minutes, and rotating source IPs does not bypass the lockout.
- Usernames that do not exist still pay the same bcrypt cost, so timing does not
  reveal whether an account exists.
- Successful sign-in clears the failure counter and is recorded in the audit log.

Both thresholds are configurable in [Server Settings](settings.md#security-rate-limiting).

## Self-service account endpoints

Any signed-in user can:

| Endpoint | Purpose |
| --- | --- |
| `GET /api/me/profile` | Profile and role |
| `GET /api/me/budget` | Account budget status |
| `PUT /api/me/password` | Change own password |
| `PUT /api/me/username` | Change own username (requires current password) |
| `GET/PUT /api/me/tracing` | Personal Langfuse tracing configuration |
| `POST /api/me/feedback` | Rate a smart-routing decision for a request |
| `POST /api/auth/logout` | Sign out (deactivates the session API keys) |
