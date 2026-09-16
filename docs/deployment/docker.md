# Docker Compose

The shipped `docker-compose.yaml` is the fastest production-ish deployment: the proxy
plus PostgreSQL and Redis, with persistent volumes and health-gated startup.

## Layout

```yaml
services:
  llm-proxy:   # ghcr.io/zwldarren/llm-proxy:stable   → 8080:8080
  db:          # postgres:18                          → volume postgres_data
  redis:       # redis:8-alpine                       → volume redis_data
```

## Before you start

Nothing. `docker compose up -d` works as-is: `POSTGRES_PASSWORD` defaults to
`llmproxy` and the compose file treats `.env` as optional. Copy the template only
when you want to change something:

```bash
cp .env.example .env
```

`.env.example` documents every supported variable; uncomment the ones you need.

::: warning Set a real password before the first start
Postgres applies `POSTGRES_PASSWORD` when the `postgres_data` volume is first
initialized. The default `llmproxy` is fine for a local trial, but for anything
reachable beyond localhost set your own value in `.env` **before** the first
`docker compose up` — afterwards the only way to change it is recreating the
volume.
:::

## Key details

- **Database URL is wired for you**: `postgresql+asyncpg://llmproxy:<POSTGRES_PASSWORD>@db:5432/llmproxy`.
- **`POSTGRES_PASSWORD` defaults to `llmproxy`** — compose interpolates it into the
  proxy's `DATABASE_URL` and the database's bootstrap. Override it in `.env`.
- **`.env` is optional** (`env_file` with `required: false`): without it, the
  service starts on the built-in defaults.
- **Startup order** is enforced with healthchecks: the proxy starts only after
  `pg_isready` succeeds and `redis-cli ping` answers.
- **Redis is enabled** (`REDIS_ENABLED=true`, `REDIS_URL=redis://redis:6379`) with
  shared rate limiting and the config cache already on
  (`REDIS_RATE_LIMIT_ENABLED=true`, `REDIS_CACHE_ENABLED=true`) — the
  prerequisite for the auto-selected multi-worker default. See
  [PostgreSQL & Redis](databases.md).
- **Redis eviction** is `allkeys-lru` with AOF persistence — safe for the
  reconstructible data the proxy stores there.

## Environment

Compose reads `.env` for interpolation **and** passes the same file to the container
via `env_file` (when it exists). So `.env` is the single place for `POSTGRES_*`,
`JWT_SECRET`, `ENCRYPTION_KEY`, `TRUSTED_PROXIES`, pool tuning, and the `REDIS_*`
switches.
Full list: [Environment Variables](../reference/environment.md).

## Image tags

Published to GHCR by CI:

| Tag | Meaning |
| --- | --- |
| `stable` | Newest tagged release — what compose pins. Recommended for production |
| `latest` | Newest build from `main` |
| `0.2.3` (pep440) | Exact release, best for pinning |
| `main` | Branch build |
| `pr-<n>` | Pull-request builds (build-only, not pushed) |

Images are signed with cosign on release builds (verify by digest if your registry
policy requires it).

## Ports and exposure

Only `llm-proxy` publishes a port (`8080`). `db` and `redis` stay on the internal
network. If you expose the proxy to the internet, put a reverse proxy with TLS in
front of it — see [Reverse Proxy & TLS](reverse-proxy.md).

## Healthchecks and restarts

`db` and `redis` have healthchecks; the `llm-proxy` service does **not** define one
in the shipped compose file, even though the app exposes
[health endpoints](../deployment/troubleshooting.md#health-endpoints). The compose
file also sets **no `restart:` policy on any service** — after a host reboot the
stack stays down until something starts it. Add a healthcheck and
`restart: unless-stopped` yourself:

```yaml
healthcheck:
  test: ["CMD", "python", "-c", "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8080/api/health/ready')"]
  interval: 30s
  timeout: 5s
  retries: 3
restart: unless-stopped
```

## Updating

```bash
docker compose pull
docker compose up -d
```

Schema migrations run automatically at startup; see [Upgrades & Backups](upgrades.md)
for what to check first.

## Where data lives

| Data | Location |
| --- | --- |
| Providers, models, keys, users, logs, usage, secrets | PostgreSQL volume `postgres_data` (mount point `/var/lib/postgresql`) |
| Routing/session state, response store, AOF | Redis volume `redis_data` |

The proxy container itself is stateless — it has no volume, so `docker compose down`
loses nothing as long as the named volumes stay.
