# 升级与备份

## 升级

### Docker Compose

```bash
docker compose pull
docker compose up -d
```

- `stable` 跟随最新的带标签发布版；如果希望升级是件有意识的事，请钉到具体标签
  （`ghcr.io/zwldarren/llm-proxy:0.5.3`）。
- 迁移在启动时自动执行，把数据库结构推进到 `head`。迁移失败时容器会退出并保持
  停止——看日志、修好数据库，再重新启动。
- 没有降级路径：结构迁移只向前。

### 从源码运行

```bash
git pull
uv sync
uv run llm-proxy --build-frontend
```

### 升级之前

1. 备份数据库（见下文）。这是唯一不可逆的部分。
2. 读 [CHANGELOG](https://github.com/zwldarren/llm-proxy/blob/main/CHANGELOG.md)，
   确认目标版本里的破坏性变更。
3. 如果你钉了标签，记下当前标签，以便回滚**镜像**：数据库结构不会跟着回滚。

### 版本感知

管理员可以在 **设置 → 常规 → 关于**（`GET /api/system/info`）检查更新：运行中的
版本会与 GitHub 上最新的标签比较，结果缓存 6 小时（`?force=true` 可绕过缓存，最多
每分钟一次）。把 `UPDATE_CHECK__ENABLED=false` 即可完全关闭外发检查。

## 备份 {#backups}

### PostgreSQL

使用自带的 compose 文件时，数据库就是 `postgres_data` 卷：

```bash
docker compose exec -T db pg_dump -U llmproxy llmproxy > llm-proxy-$(date +%F).sql
```

恢复到全新的卷：

```bash
docker compose down
docker compose up -d db
docker compose exec -T db psql -U llmproxy -d llmproxy < llm-proxy-2026-01-01.sql
docker compose up -d
```

### SQLite

数据库就是一个文件，外加同目录下的 `-wal`/`-shm` 文件。停掉应用后把三个文件都复制走
（或者用 `sqlite3 config.db ".backup out.db"`）。默认位置：
`~/.local/share/llm-proxy/config.db`。

### Redis

Redis 里只有可重建的状态（速率限制窗口、粘性路由、已存储的 Responses），并用 AOF
持久化。备份它是可选项。

### 备份里有什么

应用知道的一切：提供商、模型、定价、API 密钥（SHA-256 摘要）、用户、日志、用量
记录——**以及两个自动生成的机密**（`server_config` 里的 `jwt_secret`、
`encryption_key_store` 行）。恢复备份也就恢复了可用的会话和可解密的提供商密钥。

## 机密：轮换会破坏什么 {#secrets-what-rotation-breaks}

两个机密都在首次运行时自动生成并存进数据库。环境变量覆盖（`JWT_SECRET`、
`ENCRYPTION_KEY`）必须 **≥ 32 个字符**，否则会被静默忽略。

| 操作 | 后果 |
| --- | --- |
| 轮换 `JWT_SECRET` | 所有控制台会话失效，用户需要重新登录 |
| 轮换 `ENCRYPTION_KEY` | **已存的提供商 API 密钥无法再解密**——日志会警告 "The ENCRYPTION_KEY may have changed."。轮换后需重新录入每个提供商的密钥 |
| 丢失数据库 | 会话和可解密的提供商密钥都没了，日志、用量和配置也一样 |

把机密从数据库里读出来（用于迁移或托管）：

PostgreSQL：

```sql
SELECT key, value->>'key' AS secret FROM server_config WHERE key IN ('jwt_secret','encryption_key_store');
```

SQLite：

```sql
SELECT key, json_extract(value, '$.key') AS secret FROM server_config WHERE key IN ('jwt_secret','encryption_key_store');
```

`server_config.value` 列是 JSON：机密被包在一个信封里，实际值在它的 `key` 字段中，
所以直接取 `value` 拿到的是外层包装，不是机密本身。

## 扩展说明

| 拓扑 | 要求 |
| --- | --- |
| 单进程 | SQLite 就够；速率限制与熔断器在内存中生效 |
| 多工作进程/多副本 | PostgreSQL + `REDIS_ENABLED=true` + `REDIS_RATE_LIMIT_ENABLED=true`（自带的 compose 两个都设了）；在 PostgreSQL 上工作进程会自动扩展。迁移每次启动只跑一次——加副本之前先迁移 |
| 位于 CDN 之后 | 保持保活开启（默认），避免 524——见 [反向代理与 TLS](/zh/deployment/reverse-proxy) |
