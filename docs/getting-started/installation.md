# Installation

Two supported ways to run LLM Proxy: the prebuilt Docker image (recommended) or from
source. Both end at the same place: a first-run setup screen at
`http://localhost:8080`.

## Requirements

| Path | Needs |
| --- | --- |
| Docker Compose | Docker Engine 24+ with Compose v2 |
| From source | Python **3.14+**, [uv](https://docs.astral.sh/uv/), [bun](https://bun.sh) (to build the admin UI, and required by the dev servers) |

## Option A — Docker Compose (recommended)

```bash
git clone https://github.com/zwldarren/llm-proxy.git
cd llm-proxy
docker compose up -d
```

Open **http://localhost:8080**.

Compose starts the proxy together with PostgreSQL and Redis. The database password
defaults to `llmproxy`, so a local trial needs no configuration at all. Change it
in a `.env` file **before the first start** — Postgres applies the password when the
volume is initialized, so changing it later means recreating the volume:

```bash
echo "POSTGRES_PASSWORD=$(openssl rand -hex 16)" > .env
docker compose up -d
```

`.env` is optional and also carries `POSTGRES_USER`, `POSTGRES_DB`, `JWT_SECRET`,
`ENCRYPTION_KEY`, and the `REDIS_*` switches; see `.env.example` for the full list.

The image ships the built admin UI and the full smart-routing stack, and bundles
`bun`/`bunx` and `uv`/`uvx` so stdio MCP servers can run `bunx`- or `uvx`-style
commands inside the container. (`npx` is not included.)

## Option B — From source

```bash
git clone https://github.com/zwldarren/llm-proxy.git
cd llm-proxy
uv sync                       # core proxy, no ML dependencies

# Optional: embedding-based smart-routing signal (adds the ML stack)
uv sync --extra smart-routing

uv run llm-proxy --build-frontend
```

`--build-frontend` runs `bun install` (if needed) and `bun run build`, producing
`frontend/dist`. The server serves that directory at `/` — if `dist` is missing at
startup, the API works but the admin UI is not mounted until you rebuild and restart.

Without the `smart-routing` extra, routing falls back to the built-in structural
signals; everything else is identical.

## Server options

```bash
uv run llm-proxy --help
```

| Flag | Default | Effect |
| --- | --- | --- |
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8080` | Bind port |
| `--log-level` | `INFO` | Overrides `LOG_LEVEL` |
| `--log-file` | — | Also write logs to this file |
| `--config` | — | Path to a SQLite database file (sets `LLM_PROXY_DB_PATH`; ignored when `DATABASE_URL` is set) |
| `--workers` | `1` | Uvicorn worker processes. Ignored with a warning when combined with `--reload` |
| `--reload` | off | Dev auto-reload, watches the package directory |
| `--build-frontend` | off | Build `frontend/dist` before starting (requires bun) |

::: warning One process is the default for a reason
Rate-limit windows, circuit-breaker state, and provider statistics are
per-process. Run multiple workers or replicas only with
`REDIS_ENABLED=true` **and** `REDIS_RATE_LIMIT_ENABLED=true`, and use
PostgreSQL instead of SQLite. See [PostgreSQL & Redis](../deployment/databases.md).
:::

## Development servers

```bash
uv run llm-proxy-dev [--host H] [--port N]    # defaults: localhost / 9911
```

Starts the backend with `--reload` on **port 9911** (host `localhost`) plus a Vite dev
server on `0.0.0.0:5173` that proxies API calls to the backend. Frontend changes are
hot-reloaded. `Ctrl-C` stops both.

The command takes `--host` (default `localhost`) and `--port` (default `9911`); any
remaining arguments are passed through to the backend server. `uv run python -m
llm_proxy` starts the backend alone (same as `uv run llm-proxy`) without the dev
frontend.

The dev frontend proxies API calls to the backend via `VITE_API_BASE_URL`. When unset,
Vite falls back to `http://localhost:8000`; the dev wrapper sets it to the backend
host/port it started, so the two stay in sync automatically.

## Next steps

1. [First Setup](first-setup.md) — create the admin account.
2. [Providers](../admin/providers.md) — add your first upstream.
3. [Connect a Client](connect-a-client.md) — point an SDK at the proxy.
4. [Connect an AI Agent](connect-an-agent.md) — Claude Code, Codex, OpenCode, Pi, OMP, Hermes.
