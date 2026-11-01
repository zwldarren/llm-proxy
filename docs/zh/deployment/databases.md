# PostgreSQL 与 Redis

## 数据库

LLM Proxy 把所有东西都放在一个 SQL 数据库里：提供商、模型、API 密钥、用户、日志、
用量，以及自动生成的机密。默认是 SQLite；只要涉及多工作进程或多副本部署，就必须用
PostgreSQL。

| 模式 | 适用场景 | 配置 |
| --- | --- | --- |
| **SQLite** | 默认，单进程 | `LLM_PROXY_DB_PATH` 或 `--config`（Linux 上默认 `~/.local/share/llm-proxy/config.db`） |
| **PostgreSQL** | 生产、扩展、compose 部署 | `DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/db` |

- 只写 `postgresql://` / `postgres://` 的 URL 会自动改写成 asyncpg 驱动。
- SQLite 启用 `journal_mode=WAL`、`synchronous=NORMAL`、`foreign_keys=ON` 和
  `PRAGMA auto_vacuum=INCREMENTAL`。`auto_vacuum` 在文件创建时就定死了：只有新建的
  数据库会带上它，已有文件保持原来的模式。
  普通的 `DELETE` 不会把空间还给文件系统——文件大小不会缩小。因此每次按保留策略
  删除行之后，代理都会执行一次有上限的 `PRAGMA incremental_vacuum`，接着执行
  `PRAGMA wal_checkpoint(TRUNCATE)`（仅对启用了增量自动回收的数据库）。如果要从一个
  已经丢失空间的老数据库里把空间拿回来，可以在停止服务后做一次 `VACUUM`——或者用
  `.backup` 导出到新文件再替换掉它：

  ```bash
  systemctl stop llm-proxy            # or however you stop the service
  sqlite3 ~/.local/share/llm-proxy/config.db "VACUUM;"
  systemctl start llm-proxy
  ```

  操作对象是数据库文件本身：同目录下的 `-wal`/`-shm` 文件也是数据库的一部分（由 SQLite
  管理，`VACUUM` 或 `.backup` 会自动处理它们）。
- PostgreSQL 连接默认带 `pool_pre_ping`，并支持 `DB_POOL_SIZE`、`DB_MAX_OVERFLOW`、
  `DB_POOL_RECYCLE_SECONDS`、`DB_POOL_TIMEOUT_SECONDS`
  （见 [环境变量](/zh/reference/environment)）。
- SQLite 同一时间只允许一个写者：不要配 `--workers > 1`，也不要跑多个副本。

### 迁移

迁移是数据库结构的**唯一**来源，没有 `create_all` 兜底。

- 启动时，代理在绑定端口前执行 `alembic upgrade head`（每次启动一次，在顶层进程里
  执行）。如果数据库连不上或迁移失败，进程会直接退出——并且把迁移错误明确打出来。
- `alembic.ini` 必须存在：代理先在当前目录找，再找仓库根目录。用 wheel 安装但缺少
  它时，启动会失败并报 `Could not find alembic.ini`。
- 手动操作（先把环境变量指向目标数据库）：

```bash
uv run alembic upgrade head   # apply migrations
uv run alembic current        # show the current revision
```

::: info 迁移只在启动进程里跑一次
用 `--workers N` 时，顶层进程会在 uvicorn 派生工作进程之前执行
`alembic upgrade head`，派生出的工作进程会跳过（完成状态通过内部标志
`LLM_PROXY_MIGRATIONS_DONE` 传递）。每次启动恰好迁移一次。如果部署多个副本，仍应
在扩容前先迁移一次——每个副本都是自己的启动者。
:::

## Redis

Redis 是可选的，但它能解锁一批无法按进程工作、或必须有共享存储才能实现的功能。

| 功能 | 前提 |
| --- | --- |
| 跨工作进程/副本的共享速率限制 | `REDIS_ENABLED=true` **且** `REDIS_RATE_LIMIT_ENABLED=true` |
| 配置缓存（提供商/模型查询） | `REDIS_ENABLED=true` **且** `REDIS_CACHE_ENABLED=true` |
| 设置在所有工作进程/副本上热重载（网络搜索、熔断器、MCP 服务器、MCP 安全策略） | `REDIS_ENABLED=true` |
| API 密钥/用户角色变更跨工作进程/副本生效（吊销、角色、启用状态、预算、模型白名单） | `REDIS_ENABLED=true` |
| 个人追踪配置在所有工作进程/副本上变更 | `REDIS_ENABLED=true` |
| 粘性提供商路由 | `REDIS_ENABLED=true` **且**提供商选择策略为 `session_sticky`（全局默认）——其他策略从不读写粘性键 |
| 上一次模型延续（智能路由 `auto`/`fast`/`best`） | Redis 已连接（`REDIS_ENABLED=true`）且开启智能路由 |
| OpenResponses 响应存储：`GET`/`DELETE`/取消/`input_items`，以及 `background: true` | Redis 已连接 |
| 日志外发 | 未实现——日志始终存在 SQL 数据库里 |

行为细节：

- 已启用但启动时连不上 → 抛 `ConfigurationError`，启动失败。启动阶段不会静默
  降级。运行期缓存查询会降级并打警告。
- 速率限制的失败模式：`redis_rate_limit_fail_closed`（默认 true）在 Redis 出错时
  拒绝请求；设为 false 则放行。
- 速率限制键的前缀是 `REDIS_RATE_LIMIT_PREFIX`（默认 `rate_limit:`）；缓存键的前缀
  是 `REDIS_CACHE_PREFIX`（默认 `cache:`）。
- 粘性路由只在提供商选择策略为 `session_sticky`（全局默认）时生效；键形如
  `routing:conv:{conversation}:provider:{model}`。没有 Redis 时该策略依然能把会话
  固定住，用的是基于映射键的无状态 rendezvous 哈希。
  上一次模型延续只由智能路由（`auto`/`fast`/`best`）写入，其键存活 30 分钟。
- 已存储的 Responses 在 **24 小时**后过期（Redis TTL 86400 秒），并按 API 密钥分
  命名空间，租户之间无法读到彼此的响应。
- Redis 里只有可重建的状态：清空是安全的，代价是丢掉粘性路由状态和已存储的
  Responses。

## Compose 默认值

使用自带的 compose 文件时，`REDIS_ENABLED=true` 和 `REDIS_URL=redis://redis:6379`
已经设好。如果还想共享速率限制，在 `.env` 里加上：

```bash
REDIS_RATE_LIMIT_ENABLED=true
# Optional: cache provider/model lookups in Redis
REDIS_CACHE_ENABLED=true
```

## 相关页面

- [环境变量](/zh/reference/environment)：所有数据库与 Redis 开关
- [升级与备份](/zh/deployment/upgrades)：导出与恢复流程
