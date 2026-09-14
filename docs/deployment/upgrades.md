# Upgrades & Backups

## Upgrading

### Docker Compose

```bash
docker compose pull
docker compose up -d
```

- `stable` follows the newest tagged release; pin an exact tag
  (`ghcr.io/zwldarren/llm-proxy:0.2.3`) if you want upgrades to be a conscious act.
- **Migrations run automatically at startup** and move the schema to `head`. If a
  migration fails, the container exits and stays down — read the logs, fix the
  database, then start again.
- There is **no downgrade path**: schema migrations are forward-only.

### From source

```bash
git pull
uv sync
uv run llm-proxy --build-frontend
```

### Before you upgrade

1. **Back up the database** (below). This is the only irreversible part.
2. Read the [CHANGELOG](https://github.com/zwldarren/llm-proxy/blob/main/CHANGELOG.md)
   for breaking changes in the target version.
3. If you pin tags, note the current one so you can roll back the *image* — the
   schema will not roll back with it.

### Version awareness

Admins can check for updates from **Settings → General → About**
(`GET /api/system/info`): the running version is compared against the newest GitHub
tag, cached for 6 hours (`?force=true` bypasses the cache at most once per minute).
Set `UPDATE_CHECK__ENABLED=false` to disable the outbound check entirely.

## Backups

### PostgreSQL

With the shipped compose file the database is the `postgres_data` volume:

```bash
docker compose exec -T db pg_dump -U llmproxy llmproxy > llm-proxy-$(date +%F).sql
```

Restore into a fresh volume:

```bash
docker compose down
docker compose up -d db
docker compose exec -T db psql -U llmproxy -d llmproxy < llm-proxy-2026-01-01.sql
docker compose up -d
```

### SQLite

The database is a single file plus its `-wal`/`-shm` siblings. With the app stopped,
copy all three (or use `sqlite3 config.db ".backup out.db"`). Default location:
`~/.local/share/llm-proxy/config.db`.

### Redis

Redis holds only reconstructible state (rate-limit windows, sticky routing, stored
Responses) and is persisted with AOF. Backing it up is optional.

### What a backup contains

Everything the app knows: providers, models, pricing, API keys (bcrypt hashes),
users, logs, usage records — **and the two auto-generated secrets**
(`jwt_secret`, `encryption_key_store` rows in `server_config`). Restoring the backup
restores working sessions and decryptable provider keys.

## Secrets: what rotation breaks

Both secrets are auto-generated on first run and stored in the database. Environment
overrides (`JWT_SECRET`, `ENCRYPTION_KEY`) must be **≥ 32 characters** or they are
silently ignored.

| Action | Consequence |
| --- | --- |
| Rotate `JWT_SECRET` | Every console session is invalidated; users sign in again |
| Rotate `ENCRYPTION_KEY` | **Stored provider API keys can no longer be decrypted** — logs warn "The ENCRYPTION_KEY may have changed." Re-enter every provider key after rotating |
| Lose the database | Sessions and decryptable provider keys are gone; logs, usage, and configuration too |

To read the secrets out of the database (for migration or escrow):

PostgreSQL:

```sql
SELECT key, value->>'key' AS secret FROM server_config WHERE key IN ('jwt_secret','encryption_key_store');
```

SQLite:

```sql
SELECT key, json_extract(value, '$.key') AS secret FROM server_config WHERE key IN ('jwt_secret','encryption_key_store');
```

The `server_config.value` column is JSON: the secret is wrapped in an envelope and
lives in its `key` field, so extracting the raw `value` yields the wrapper, not the
secret.

## Scaling notes

| Topology | Requirements |
| --- | --- |
| One process | SQLite is fine; in-memory rate limits and circuit breaker apply |
| Multiple workers/replicas | PostgreSQL + `REDIS_ENABLED=true` + `REDIS_RATE_LIMIT_ENABLED=true`; avoid concurrent automatic migration (migrate once, then scale) |
| Behind a CDN | Keep keepalive enabled (default) to avoid 524s — see [Reverse Proxy & TLS](reverse-proxy.md) |
