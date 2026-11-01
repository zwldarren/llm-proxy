# 环境变量

环境变量在**启动时**从进程环境以及当前工作目录下的 `.env` 文件读取一次。修改它们需要
重启；所有运行时行为（提供商、模型、限制、日志策略、安全阈值）都在
[管理控制台](/zh/admin/overview)里配置并热重载。

## 常用变量

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `DATABASE_URL` | 未设置 | 完整的数据库 URL。优先于 `LLM_PROXY_DB_PATH`。`postgresql://` 会被改写成 `postgresql+asyncpg://` |
| `LLM_PROXY_DB_PATH` | 平台数据目录 + `config.db` | SQLite 文件路径（仅在 `DATABASE_URL` 未设置时使用） |
| `JWT_SECRET` | 自动生成 | JWT 签名密钥覆盖值；少于 32 字符会被忽略。修改它会使所有会话失效 |
| `ENCRYPTION_KEY` | 自动生成 | 提供商密钥加密覆盖值；少于 32 字符会被忽略。修改它会让已存储的提供商密钥无法解密 |
| `LOG_LEVEL` | `INFO` | 应用日志级别；`--log-level` 优先 |
| `TRUSTED_PROXIES` | RFC1918 + 回环 + 链路本地网段 | 允许设置 `X-Forwarded-For`/`X-Real-IP` 的 CIDR 列表。**替换**默认列表；留空 = 谁都不信；CIDR 非法会导致启动失败 |
| `REDIS_ENABLED` | `false` | Redis 总开关：为 false 时使用内存速率限制，且没有配置缓存 |
| `REDIS_URL` | `redis://localhost:6379` | Redis 连接 URL |
| `REDIS_RATE_LIMIT_ENABLED` | `false` | 基于 Redis 的滑动窗口速率限制（需要 `REDIS_ENABLED`） |
| `REDIS_CACHE_ENABLED` | `false` | 用于提供商/模型配置查询的 Redis 缓存 |
| `REDIS_LOGGING_ENABLED` | `false` | 保留项，无效果——日志始终存储在 SQL 数据库中 |
| `UVICORN_WORKERS` | 自动 | Uvicorn 工作进程数。未设置 = 按 CPU 预算（感知 cgroup），PostgreSQL 下最多 16，SQLite 下始终为 1。`--workers` 优先 |
| `UVICORN_TIMEOUT_KEEPALIVE` | `600` | Keep-alive 超时（秒） |
| `UPDATE_CHECK__ENABLED` | `true` | 注意是双下划线。是否让 `/api/system/info` 向 GitHub 查询新版本 |

## 高级

连接池大小、超时和后台写入器调优。不清楚原因就不要改。

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `HTTP_MAX_CONNECTIONS` | `200` | 提供商调用使用的 httpx 连接池大小 |
| `HTTP_MAX_KEEPALIVE` | `200` | httpx keep-alive 连接池大小 |
| `HTTP_DISABLE_HTTP2` | `true` | HTTP/2 默认禁用 |
| `HTTP_CLIENT_BACKEND` | `auto` | 外发 HTTP 后端：`auto`、`httpx2` 或 `aiohttp`。`auto` 选择 `aiohttp`（在网关热路径上实测吞吐约为 1.6 倍，每请求 CPU 低约 45%），除非配置了外发代理，此时沿用 `httpx2`，因为 aiohttp 不支持 `HTTP_PROXY`/`HTTPS_PROXY`/`ALL_PROXY` |
| `HTTP_ALLOWED_HOSTS` | *空* | 逗号分隔的主机列表，豁免 SSRF 私有地址封锁，适用于私有网络中的自建提供商和内部图像主机。每一条都会削弱 SSRF 防护 |
| `HTTP_MAX_DOWNLOAD_BYTES` | `20971520`（20 MiB） | 代理为客户端提供的图像/文件 URL 缓冲的最大正文，流式读取中即生效。20 MiB 是主流视觉 API 单图的上限（OpenAI 20 MB，Gemini 内联请求 20 MB；Anthropic 为 5 MB），因此该上限不会拒绝任何上游本来能接受的图片 |
| `HTTP_MAX_REQUEST_DOWNLOAD_BYTES` | `67108864`（64 MiB） | 单个请求在其列出的所有 URL 上可保留的内联媒体总量；与请求体上限相同。每请求内存峰值为该总量加上最多四个进行中的下载 |
| `HTTP_MAX_RESPONSE_BYTES` | `67108864`（64 MiB） | 代理缓冲的上游响应上限。超过上限的响应直接以 `502` 拒绝，而不是读进内存 |
| `DB_POOL_SIZE` | `min((cpu_count * 2 + 1) / workers, 80 / (2 * workers))`，最小为 1 | 每个工作进程的 PostgreSQL 连接池大小（SQLite 下忽略）。第二个上界把总连接数限制在约 80（溢出默认等于池大小），避免 N 个工作进程打爆 `max_connections`。`.env.example` 中的 `DB_POOL_SIZE=10` / `DB_MAX_OVERFLOW=10` 是示例，不是默认值 |
| `DB_MAX_OVERFLOW` | 与连接池大小相同 | PostgreSQL 溢出连接数 |
| `DB_POOL_RECYCLE_SECONDS` | `3600`（最小 60） | PostgreSQL 连接回收时间 |
| `DB_POOL_TIMEOUT_SECONDS` | `30`（最小 1） | PostgreSQL 连接池获取超时 |
| `REDIS_POOL_SIZE` | `10` | Redis 连接池大小 |
| `REDIS_TIMEOUT` | `5.0` | Redis socket 与连接超时（秒） |
| `REDIS_RATE_LIMIT_PREFIX` | `rate_limit:` | 速率限制键前缀 |
| `REDIS_CACHE_PREFIX` | `cache:` | 配置缓存键前缀 |
| `REDIS_CACHE_TTL_PROVIDER_CONFIG` | `300` | 提供商配置缓存 TTL（秒） |
| `REDIS_CACHE_TTL_MODEL_MAPPING` | `300` | 模型映射缓存 TTL（秒） |
| `REDIS_LOGGING_PREFIX` / `_TTL_DAYS` / `_BATCH_SIZE` / `_FLUSH_INTERVAL_MS` | `logs:` / `30` / `50` / `250` | 保留项，无效果 |
| `LOG_WRITE_QUEUE_SIZE` | `2000` | 请求日志写入器队列深度；队列写满会丢弃记录 |
| `LOG_WRITE_BATCH_SIZE` | `100` | 请求日志批大小 |
| `LOG_WRITE_FLUSH_INTERVAL_MS` | `500` | 请求日志刷写间隔 |
| `USAGE_WRITE_QUEUE_SIZE` | `2000` | 用量写入器队列深度 |
| `USAGE_WRITE_BATCH_SIZE` | `200` | 用量批大小 |
| `USAGE_WRITE_FLUSH_INTERVAL_MS` | `500` | 用量刷写间隔 |

## 不能用环境变量配置

下面这些都在管理控制台里（存数据库、热重载）——刻意如此，这样修改时不用重启：

- 日志策略：保留期、正文记录、原始流捕获、脱敏、采样、敏感字段
- 安全：锁定阈值、HSTS、请求体大小、认证失败延迟、速率限制桶
- 网络行为：CORS 来源、响应保活
- 功能：网络搜索、智能路由、提供商选择策略、重试/回退、熔断器、MCP 安全策略
- 提供商、模型、定价、API 密钥、用户、MCP 服务器

## `.env.example`

仓库附带一份注释完整的 `.env.example`，记录了上面每一个变量，还包括可直接使用的
`TRUSTED_PROXIES` 示例（Docker/Traefik 与 Cloudflare）。

## 内部使用（不要手动设置）

`LLM_PROXY_LOG_LEVEL`、`LLM_PROXY_LOG_FILE` 和 `LLM_PROXY_WORKER_COUNT` 由服务器
进程在启动 uvicorn 之前自行写入，好让各工作进程继承解析后的日志级别、日志文件和
工作进程数。请改用 `LOG_LEVEL` / `--log-level` / `--log-file` / `--workers` 来设置
——你放进环境里的值会在启动时被覆盖。
