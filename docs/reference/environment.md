# Environment Variables

Environment variables are read **once at startup** from the process environment plus a
`.env` file in the current working directory. Changing them requires a restart — all
runtime behavior (providers, models, limits, logging policy, security thresholds) is
configured in the [admin console](../admin/overview.md) and hot-reloaded instead.

## User-facing

| Variable | Default | Effect |
| --- | --- | --- |
| `DATABASE_URL` | unset | Full database URL. Wins over `LLM_PROXY_DB_PATH`. `postgresql://` is rewritten to `postgresql+asyncpg://` |
| `LLM_PROXY_DB_PATH` | platform data dir + `config.db` | SQLite file path (used only when `DATABASE_URL` is unset) |
| `JWT_SECRET` | auto-generated | JWT signing secret override; ignored unless ≥ 32 chars. Changing it invalidates all sessions |
| `ENCRYPTION_KEY` | auto-generated | Provider-key encryption override; ignored unless ≥ 32 chars. Changing it makes existing stored provider keys undecryptable |
| `LOG_LEVEL` | `INFO` | Application log level; `--log-level` overrides |
| `TRUSTED_PROXIES` | RFC1918 + loopback + link-local ranges | CIDR list allowed to set `X-Forwarded-For`/`X-Real-IP`. **Replaces** the default list; empty = trust nobody; invalid CIDR fails startup |
| `REDIS_ENABLED` | `false` | Master Redis switch: memory rate limiting and no config cache when false |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL |
| `REDIS_RATE_LIMIT_ENABLED` | `false` | Redis-backed sliding-window rate limiting (requires `REDIS_ENABLED`) |
| `REDIS_CACHE_ENABLED` | `false` | Redis cache for provider/model config lookups |
| `REDIS_LOGGING_ENABLED` | `false` | Reserved; no effect in v0.2.3 — logs are always stored in the SQL database |
| `UVICORN_WORKERS` | auto | Uvicorn worker processes. Unset = CPU budget (cgroup-aware) capped at 16 on PostgreSQL; always 1 on SQLite. `--workers` overrides |
| `UVICORN_TIMEOUT_KEEPALIVE` | `600` | Keep-alive timeout in seconds |
| `UPDATE_CHECK__ENABLED` | `true` | Double underscore. Whether `/api/system/info` checks GitHub for a newer version |

## Advanced

Pool sizes, timeouts, and background writer tuning. Change only if you know why.

| Variable | Default | Effect |
| --- | --- | --- |
| `HTTP_MAX_CONNECTIONS` | `200` | httpx pool size for provider calls |
| `HTTP_MAX_KEEPALIVE` | `200` | httpx keep-alive pool size |
| `HTTP_DISABLE_HTTP2` | `true` | HTTP/2 is disabled by default |
| `HTTP_CLIENT_BACKEND` | `auto` | Outbound HTTP backend: `auto`, `httpx2` or `aiohttp`. `auto` selects `aiohttp` (measured ~1.6x the throughput at ~45% lower CPU per request on the gateway hot path) unless an outbound proxy is configured, in which case it keeps `httpx2` because aiohttp does not honour `HTTP_PROXY`/`HTTPS_PROXY`/`ALL_PROXY` |
| `DB_POOL_SIZE` | `min((cpu_count * 2 + 1) / workers, 80 / (2 * workers))`, min 1 | PostgreSQL pool size per worker process (ignored for SQLite). The second bound caps the total at ~80 connections (overflow defaults to the pool size) so N workers don't overrun `max_connections`. The `DB_POOL_SIZE=10` / `DB_MAX_OVERFLOW=10` lines in `.env.example` are examples, not the defaults |
| `DB_MAX_OVERFLOW` | same as pool size | PostgreSQL overflow |
| `DB_POOL_RECYCLE_SECONDS` | `3600` (min 60) | PostgreSQL connection recycle |
| `DB_POOL_TIMEOUT_SECONDS` | `30` (min 1) | PostgreSQL pool checkout timeout |
| `REDIS_POOL_SIZE` | `10` | Redis connection pool |
| `REDIS_TIMEOUT` | `5.0` | Redis socket and connect timeout (seconds) |
| `REDIS_RATE_LIMIT_PREFIX` | `rate_limit:` | Rate-limit key prefix |
| `REDIS_CACHE_PREFIX` | `cache:` | Config-cache key prefix |
| `REDIS_CACHE_TTL_PROVIDER_CONFIG` | `300` | Provider-config cache TTL (seconds) |
| `REDIS_CACHE_TTL_MODEL_MAPPING` | `300` | Model-mapping cache TTL (seconds) |
| `REDIS_LOGGING_PREFIX` / `_TTL_DAYS` / `_BATCH_SIZE` / `_FLUSH_INTERVAL_MS` | `logs:` / `30` / `50` / `250` | Reserved; no effect in v0.2.3 |
| `LOG_WRITE_QUEUE_SIZE` | `2000` | Request-log writer queue depth; full queue drops records |
| `LOG_WRITE_BATCH_SIZE` | `100` | Request-log batch size |
| `LOG_WRITE_FLUSH_INTERVAL_MS` | `500` | Request-log flush interval |
| `USAGE_WRITE_QUEUE_SIZE` | `2000` | Usage writer queue depth |
| `USAGE_WRITE_BATCH_SIZE` | `200` | Usage batch size |
| `USAGE_WRITE_FLUSH_INTERVAL_MS` | `500` | Usage flush interval |

## Not configurable here

These live in the admin console (database-backed, hot-reloaded) — deliberately, so
they can change without a restart:

- **Logging policy**: retention, body logging, masking, sampling, sensitive keys
- **Security**: lockout thresholds, HSTS, request body size, auth failure delay,
  rate-limit buckets
- **Network behavior**: CORS origins, response keepalive
- **Features**: web search, smart routing, provider selection strategy, retry/fallback,
  circuit breaker, MCP security policy
- **Providers, models, pricing, API keys, users, MCP servers**

## `.env.example`

The repository ships a fully commented `.env.example` documenting every variable above,
including ready-made `TRUSTED_PROXIES` examples for Docker/Traefik and Cloudflare.

## Internal (do not set)

`LLM_PROXY_LOG_LEVEL`, `LLM_PROXY_LOG_FILE` and `LLM_PROXY_WORKER_COUNT` are written by the server process itself
before it starts uvicorn, so worker processes inherit the resolved level, log file and worker count.
Set them via `LOG_LEVEL` / `--log-level` / `--log-file` / `--workers` instead — values you put in the
environment are overwritten at startup.
