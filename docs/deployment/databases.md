# PostgreSQL & Redis

## Database

LLM Proxy stores everything — providers, models, API keys, users, logs, usage, and
generated secrets — in one SQL database. SQLite is the default; PostgreSQL is required
for any multi-worker or multi-replica deployment.

| Mode | When | Configuration |
| --- | --- | --- |
| **SQLite** | Default, single process | `LLM_PROXY_DB_PATH` or `--config` (default: `~/.local/share/llm-proxy/config.db` on Linux) |
| **PostgreSQL** | Production, scaling, compose | `DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/db` |

- Bare `postgresql://` / `postgres://` URLs are rewritten to the asyncpg driver
  automatically.
- SQLite runs with `journal_mode=WAL`, `synchronous=NORMAL`, `foreign_keys=ON`.
  The file's `-wal`/`-shm` siblings are part of the database.
- PostgreSQL connections get `pool_pre_ping` plus `DB_POOL_SIZE`, `DB_MAX_OVERFLOW`,
  `DB_POOL_RECYCLE_SECONDS`, and `DB_POOL_TIMEOUT_SECONDS`
  (see [Environment Variables](../reference/environment.md)).
- SQLite has a single writer — do not use it with `--workers > 1` or multiple
  replicas.

### Migrations

Migrations are the **only** schema source. There is no `create_all` fallback.

- At startup the proxy runs `alembic upgrade head` (once per launch, in the top-level
  process) before binding the port. If the database is unreachable or a migration fails, the process exits —
  loudly, with the migration error.
- `alembic.ini` must exist: it is looked up in the current directory, then at the repo
  root. A wheel install without it fails at startup with
  `Could not find alembic.ini`.
- Manual operations (point the env vars at the target database first):

```bash
uv run alembic upgrade head   # apply migrations
uv run alembic current        # show the current revision
```

::: info Migrations run once, in the launching process
With `--workers N`, the top-level process runs `alembic upgrade head` before
uvicorn spawns the workers, and the spawned workers skip it (the completion is
propagated via the internal `LLM_PROXY_MIGRATIONS_DONE` flag). Migrations
run exactly once per launch. With several replicas, still migrate once before
scaling out — each replica is its own launcher.
:::

## Redis

Redis is optional but unlocks features that cannot work per-process or without shared
storage.

| Feature | Requires |
| --- | --- |
| Shared rate limiting across workers/replicas | `REDIS_ENABLED=true` **and** `REDIS_RATE_LIMIT_ENABLED=true` |
| Config cache (provider/model lookups) | `REDIS_ENABLED=true` **and** `REDIS_CACHE_ENABLED=true` |
| Sticky provider routing | `REDIS_ENABLED=true` **and** provider selection strategy `session_sticky` (global default is `random`) — other strategies never read or write the sticky key |
| Last-model continuity (smart routing `auto`/`fast`/`best`) | Redis connected (`REDIS_ENABLED=true`) and smart routing enabled |
| OpenResponses response store: `GET`/`DELETE`/cancel/`input_items`, and `background: true` | Redis connected |
| Log shipping | Not implemented in v0.2.3 — logs are always stored in the SQL database |

Behavior details:

- **Enabled but unreachable at startup** → `ConfigurationError`, startup fails. There
  is no silent degradation at boot. At runtime, cache lookups degrade gracefully with
  warnings.
- **Rate-limit fail mode**: `redis_rate_limit_fail_closed` (default true) blocks
  requests when Redis errors; set it false to fail open.
- Rate-limit keys use the prefix `REDIS_RATE_LIMIT_PREFIX` (default `rate_limit:`);
  cache keys use `REDIS_CACHE_PREFIX` (default `cache:`).
- Sticky routing applies only with the `session_sticky` provider-selection strategy
  (the global default is `random`); keys look like
  `routing:conv:{conversation}:provider:{model}`. Last-model continuity is written only
  by smart routing (`auto`/`fast`/`best`) and its key lives for 30 minutes.
- Stored Responses expire after **24 hours** (Redis TTL 86400 s), namespaced per API
  key so tenants cannot read each other's responses.
- Redis holds only reconstructible state: safe to flush, though doing so drops sticky
  routing state and stored Responses.

## Compose defaults

With the shipped compose file, `REDIS_ENABLED=true` and `REDIS_URL=redis://redis:6379`
are already set. To also share rate limiting, add to `.env`:

```bash
REDIS_RATE_LIMIT_ENABLED=true
# Optional: cache provider/model lookups in Redis
REDIS_CACHE_ENABLED=true
```

## Related

- [Environment Variables](../reference/environment.md) — every database and Redis knob
- [Upgrades & Backups](upgrades.md) — dump/restore procedures
