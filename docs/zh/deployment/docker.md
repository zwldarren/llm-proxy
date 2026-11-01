# Docker Compose

仓库自带的 `docker-compose.yaml` 是最省事的准生产部署方式：代理加 PostgreSQL 和
Redis，带持久化卷，并按健康检查决定启动顺序。

## 组成

```yaml
services:
  llm-proxy:   # ghcr.io/zwldarren/llm-proxy:stable   → 8080:8080
  db:          # postgres:18                          → volume postgres_data
  redis:       # redis:8-alpine                       → volume redis_data
```

## 开始之前

什么都不用做，直接 `docker compose up -d` 就能跑：`POSTGRES_PASSWORD` 默认是
`llmproxy`，compose 文件把 `.env` 当作可选。只有想改动什么时才需要复制模板：

```bash
cp .env.example .env
```

`.env.example` 里写明了所有支持的变量，按需取消注释即可。

::: warning 首次启动前改掉默认密码
Postgres 只在 `postgres_data` 卷**第一次**初始化时应用 `POSTGRES_PASSWORD`。本地
试用用默认的 `llmproxy` 没问题，但只要服务会被 localhost 之外访问，就请在第一次
`docker compose up` **之前**在 `.env` 里设好自己的值——之后再改就只能重建卷了。
:::

## 关键细节

- 数据库 URL 已经接好：`postgresql+asyncpg://llmproxy:<POSTGRES_PASSWORD>@db:5432/llmproxy`。
- `POSTGRES_PASSWORD` 默认是 `llmproxy`：compose 会把它插值进代理的
  `DATABASE_URL` 和数据库的初始化流程。请在 `.env` 里覆盖。
- `.env` 是可选文件（`env_file` 标了 `required: false`）：没有它，服务按内置
  默认值启动。
- 启动顺序由健康检查保证：代理要等 `pg_isready` 成功、`redis-cli ping` 有回应
  之后才启动。
- Redis 已启用（`REDIS_ENABLED=true`、`REDIS_URL=redis://redis:6379`），共享
  速率限制和配置缓存也都开着（`REDIS_RATE_LIMIT_ENABLED=true`、
  `REDIS_CACHE_ENABLED=true`）——这是自动选择多工作进程默认值的前提。见
  [PostgreSQL 与 Redis](/zh/deployment/databases)。
- Redis 淘汰策略是 `allkeys-lru`，并开启 AOF 持久化：代理放在 Redis 里的都是
  可重建数据，这样配置是安全的。

## 环境变量

compose 会读 `.env` 做变量插值，**同时**通过 `env_file` 把同一个文件传给容器
（如果存在）。因此 `POSTGRES_*`、`JWT_SECRET`、`ENCRYPTION_KEY`、`TRUSTED_PROXIES`、
连接池调优和 `REDIS_*` 开关都只需写在 `.env` 一处。
完整列表：[环境变量](/zh/reference/environment)。

## 镜像标签

由 CI 发布到 GHCR：

| 标签 | 含义 |
| --- | --- |
| `stable` | 最新的带标签发布版 —— compose 默认钉的就是它，生产环境推荐 |
| `latest` | `main` 分支的最新构建 |
| `0.5.3`（pep440） | 精确的发布版本，最适合钉版本 |
| `main` | 分支构建 |
| `pr-<n>` | Pull Request 构建（只构建，不推送） |

发布构建的镜像带 cosign 签名（如果你们的镜像仓库策略有要求，可以按 digest 校验）。

## 端口与暴露

只有 `llm-proxy` 发布端口（`8080`）。`db` 和 `redis` 留在内部网络。如果要把代理
暴露到公网，请在前面加一层带 TLS 的反向代理——见
[反向代理与 TLS](/zh/deployment/reverse-proxy)。

## 健康检查与重启

`db` 和 `redis` 有健康检查；`llm-proxy` 服务在自带的 compose 文件里**没有**定义，
尽管应用本身提供了[健康检查端点](/zh/deployment/troubleshooting#health-endpoints)。
compose 文件也**没有为任何服务设置 `restart:` 策略**——主机重启后整个栈会一直停着，
直到有人把它拉起来。请自行补上健康检查和 `restart: unless-stopped`：

```yaml
healthcheck:
  test: ["CMD", "python", "-c", "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8080/api/health/ready')"]
  interval: 30s
  timeout: 5s
  retries: 3
restart: unless-stopped
```

## 升级

```bash
docker compose pull
docker compose up -d
```

数据库结构迁移会在启动时自动执行；升级前该检查什么，见
[升级与备份](/zh/deployment/upgrades)。

## 数据存放在哪

| 数据 | 位置 |
| --- | --- |
| 提供商、模型、密钥、用户、日志、用量、机密 | PostgreSQL 卷 `postgres_data`（挂载点 `/var/lib/postgresql`） |
| 路由/会话状态、响应存储、AOF | Redis 卷 `redis_data` |

代理容器本身是无状态的——它没有挂载卷，所以只要具名卷还在，`docker compose down`
不会丢任何东西。
