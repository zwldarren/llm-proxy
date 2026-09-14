# FAQ

## Can I use my console login as an API key?

No. Console JWTs are accepted only on `/api/*`. Clients must use an API key
(`sk-…`) on `/v1/*` and `/servers/*`. The reverse is also true: API keys are rejected
on `/api/*`.

## My key is lost — can I recover it?

No. Keys are bcrypt-hashed and shown once at creation. Create a replacement, switch
clients, delete the old key. There is no rotate endpoint, but rotation without downtime
is simply: create new → switch → delete old.

## Can one model name map to several providers?

Yes — that is the point of provider mappings. Give each mapping a `priority`; higher
priority is preferred, and failures fall back down the chain. See
[Models & Pricing](../admin/models.md#provider-priority-and-fallback).

## Why does `cost_usd` show as null?

Null means *unknown*, not free: no rate is configured for the model/mapping, or the
request had nothing billable. Configure pricing (manually or via the models.dev sync)
to get real numbers. See [Cost Control](../guides/cost-control.md#1-prices).

## Why did I get HTTP 200 with an error JSON body?

The non-streaming [keepalive](../deployment/reverse-proxy.md#cloudflare-and-other-impatient-cdns)
had already committed a 200 before the failure surfaced. The real error is in the
logs; the body contains the error JSON. This only happens for non-streaming requests
that outlive the grace period (60 s by default).

## A request returned 403 `model_not_allowed`. Where do I fix it?

Two allowlists exist: the **user's** and the **key's**, and the effective access is
their intersection. Check [Users & Roles](../admin/users.md#model-allowlists) and
[API Keys](../admin/api-keys.md). An explicitly empty allowlist denies all models.

## Does the proxy support vision / tools / reasoning / JSON mode?

Yes for all four, provided the selected upstream supports them — the proxy translates
between protocols. Capability flags on the model record drive routing and the catalog;
they do not block non-chat endpoints. See
[Create chat completion](../api/chat/create-completion.md) and [Tools & Reasoning](../api/tools.md).

## Can I export logs to my own system?

There is no export endpoint in v0.2.3; use `GET /api/logs` with cursor pagination
(max 200 per page) or scrape container logs. Langfuse tracing (per user) covers
LLM-level observability.

## Is there an OpenAPI/Swagger page?

No — the FastAPI docs (`/docs`, `/redoc`, `/openapi.json`) are disabled. This
documentation and the [Endpoint Index](../api/endpoints.md) are the reference.

## Can I run multiple replicas?

Yes, with PostgreSQL and `REDIS_ENABLED=true` + `REDIS_RATE_LIMIT_ENABLED=true`.
Without Redis, rate limits, circuit breaker state, and provider statistics are
per-process. Migrations run at startup per process — migrate once before scaling.
See [PostgreSQL & Redis](../deployment/databases.md).

## The keepalive section shows "disabled", but keepalive seems active. Which is true?

Both are: with no stored configuration the runtime uses **enabled / 60 s grace /
15 s interval**, while the Settings form shows its own defaults. Saving the section
applies exactly what the form shows. Review the values before saving.

## Does the proxy inject MCP tools into model requests?

No. MCP servers are re-exposed as MCP endpoints at `/servers/<name>/mcp` for MCP
clients; they are not spliced into `/v1/*` model requests. See
[MCP Servers](../admin/mcp.md).

## What happens to a request when the client disconnects?

The upstream call is cancelled and the request is logged as **499** — visible in the
logs instead of silently succeeding. Budgets count what was actually used at that
point.

## I'm locked out of the console. How do I get back in?

- Another admin can reset your password from **Team** (which forces a change at next
  sign-in).
- If you are the **last admin** and locked out (lockout expires by default after
  15 minutes; a forgotten password is worse): there is no CLI recovery command —
  edit the database directly. Passwords are stored as bcrypt hashes in the
  `users.password_hash` column, so replacing the hash is the supported-by-hand route.
- Lockout counters live in process memory, so restarting the proxy also clears a
  lockout.

## Which languages does the console support?

English and 中文, switchable in Settings → General → Preference (browser-local,
per user).

## Where do feature requests and bugs go?

[GitHub issues](https://github.com/zwldarren/llm-proxy/issues) — see the triage labels
in the repository for how issues are handled.
