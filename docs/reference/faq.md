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

## How are files and media mapped when I route to another provider?

The proxy renders each part in whatever form the upstream accepts, so the same
`input_image` / `input_file` / `input_audio` part may arrive as inline bytes, a remote
URI, or an uploaded-file reference. Worth knowing:

- **`file_id` is passed through as-is.** There is no `/v1/files` store, so an id must
  have been issued by the upstream it is sent to. An OpenAI Files-API id sent to Gemini
  (or the reverse) fails upstream.
- **HTTP(S) image, audio and document URIs are downloaded and inlined** before the
  request is forwarded, so any publicly reachable URL works. SSRF protection applies.
- **Video URIs are not downloaded.** Gemini fetches video itself and reliably accepts
  only YouTube URLs (or File API / `gs://` URIs); an arbitrary `https://…/v.mp4` is
  rejected upstream.
- **Multimodal tool results** stay structured where the upstream supports it (Responses
  `function_call_output.output`, Gemini 3 `functionResponse.parts`); elsewhere the media
  is degraded to a text placeholder rather than dropped.
- A block the target provider cannot represent at all follows the
  `unsupported_block_policy`; the default **Provider default** resolves to `drop` for
  every provider today, while `degrade` keeps a text placeholder (`[Image: …]`,
  `[File: …]`) and never silently drops — a block with no specific placeholder gets a
  generic `[<TypeName> block]` marker.

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

## Is response keepalive on by default?

Yes. With no stored configuration the runtime uses **enabled / 60 s grace /
15 s interval**, and the Settings form shows those same values, so the panel
reflects what is actually in effect. Turn it off there if a client cannot
tolerate the whitespace heartbeats.

## Does the proxy inject MCP tools into model requests?

No. MCP servers are re-exposed as MCP endpoints at `/servers/<name>/mcp` for MCP
clients; they are not spliced into `/v1/*` model requests. See
[MCP Servers](../admin/mcp.md).

## What happens to a request when the client disconnects?

The upstream call is cancelled and the request is logged as **499** — visible in the
logs instead of silently succeeding. Budgets count what was actually used at that
point.

## I'm locked out of the console. How do I get back in?

- Login lockout is **off by default**, so a wrong password normally just returns 401.
  If you enabled it and are locked out, wait out the 15-minute window — or restart
  the proxy, since the lockout counters live in process memory.
- Another admin can reset your password from **Team** (which forces a change at next
  sign-in).
- If you are the **last admin** with a forgotten password: there is no CLI recovery
  command — edit the database directly. Passwords are stored as bcrypt hashes in the
  `users.password_hash` column, so replacing the hash is the supported-by-hand route.

## Which languages does the console support?

English and 中文, switchable in Settings → General → Preference (browser-local,
per user).

## Where do feature requests and bugs go?

[GitHub issues](https://github.com/zwldarren/llm-proxy/issues) — see the triage labels
in the repository for how issues are handled.
