# First Setup

The first time you open the proxy there are no accounts, so the console shows a
**Create Admin Account** screen at `/setup`.

## Create the admin account

1. Open `http://localhost:8080` (or your deployment URL).
2. Enter a username and password. The password must be **8–72 characters** with at
   least one uppercase letter, one lowercase letter, one digit, and one special
   character.
3. Submit — the account is created with the `admin` role and you are signed in
   immediately.

Notes:

- Setup is rate-limited (5 requests/minute by IP) and can only run once: if an admin
  already exists, it returns `400 setup_complete`.
- You chose the password yourself, so there is no forced password change. Accounts an
  admin creates later **must** change their password at first sign-in.

## Then do these three things

| Step | Where | Why |
| --- | --- | --- |
| 1. Add a provider | [Providers](../admin/providers.md) | No models exist until a provider and its models are configured |
| 2. Check models and pricing | [Models & Pricing](../admin/models.md) | Model names are what clients send; pricing drives cost tracking and budgets |
| 3. Create an API key | [API Keys](../admin/api-keys.md) | Clients authenticate with API keys — console logins do not work on `/v1/*` |

Optional next steps: enable [smart routing](../api/routing.md) for the virtual models
`auto`/`fast`/`best`, wire up [MCP servers](../admin/mcp.md), or configure
[Langfuse tracing](../admin/settings.md#tracing) for your account.

## Sessions and security

- Console sessions are JWTs valid for **24 hours**; there is no refresh token — sign in
  again when one expires. `POST /api/auth/logout` ends the session and deactivates its
  playground keys.
- Failed sign-ins lock the **username** after 5 attempts for 15 minutes (configurable
  defaults; see [Server Settings](../admin/settings.md#security-rate-limiting)).
- The JWT secret and the provider-key encryption key are generated on first run and
  stored in the database. Keep the database: losing it means losing both. See
  [Upgrades & Backups](../deployment/upgrades.md#secrets-what-rotation-breaks).
