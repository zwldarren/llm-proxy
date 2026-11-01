# 故障排查

先从健康检查端点入手，再往外查。

## 健康检查端点 {#health-endpoints}

三个端点都不需要鉴权：

| 端点 | 返回 |
| --- | --- |
| `GET /api/health` | 始终 200。包含 `status`，以及 `services.database`（执行一次 `SELECT 1`）和 `services.redis`（`status`、`healthy`、`enabled`；Redis 关闭时为 `disabled`）。还有 `services.rate_limiting`——仅作参考，永远不会拉低 `status`：`backend`（`redis` 表示整个部署共用一个窗口，`memory` 表示**每个进程**各一个窗口，因此 N 个工作进程实际会执行 N 倍于配置的限额）、`shared` 和 `disabled`（对应 `rate_limit_disabled` 开关） |
| `GET /api/health/ready` | 健康时返回 200 `{"status":"ready"}`，否则 **503**——编排系统的就绪探针用这个 |
| `GET /api/health/live` | 始终 200 `{"status":"alive"}`——只表示存活，不检查依赖 |

## 启动失败

进程宁可明确失败，也不会带着半套配置启动。到容器/控制台日志里找错误：

| 错误 | 原因 / 处理 |
| --- | --- |
| 启动时报迁移错误 | 数据库连不上、凭据错误，或迁移失败。端口不会被绑定；修好数据库后重启 |
| `Could not find alembic.ini` | 运行的安装里丢了 `alembic.ini`（它必须在当前工作目录或仓库根目录） |
| 启动时 Redis 抛 `ConfigurationError` | `REDIS_ENABLED=true` 但服务器连不上——启动阶段不会静默降级 |
| `TRUSTED_PROXIES` 的 CIDR 非法 | 修正该值；启动时会校验 |
| 设置校验错误 | 例如 `DB_POOL_SIZE < 1`、`REDIS_TIMEOUT <= 0` |
| 配置校验错误 | 已存的提供商/模型/用户记录不合法——错误信息会指出是哪条记录 |

## 现象 → 原因 → 处理

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `401 authorization header missing` / `invalid authorization header format` | 没带密钥，或 `Authorization` 不是 `Bearer` 形式 | 发送 `Authorization: Bearer sk-…` 或 `x-api-key: sk-…`。控制台 JWT **不能**用在 `/v1/*` 上 |
| `429 too_many_auth_failures`，带 `Retry-After` | 启用了 API 密钥的 IP 锁定（默认关闭），且该地址用无效密钥达到了 `max_failed_api_key_attempts` 次 | 等过 `Retry-After`，并修正密钥。由于锁定以 IP 为键，一个坏客户端会把共用该地址的所有人一起挡住——在开发机上，就是从 `127.0.0.1` 来的所有请求。可在 设置 → 高级 → 安全与速率限制 里关掉，或打开 **禁用限流**（计数器在进程内，重启也会清零） |
| `403 model_not_allowed` | 密钥或用户白名单不包含该模型 | 调整白名单，或改用允许的模型。记住：白名单为空等于全部拒绝 |
| `404 model_not_found` | 模型名不在模型映射里 | 添加映射（或修正名称）。虚拟模型需要先开启智能路由 |
| `500 configuration_error` 并提到虚拟模型 | 在智能路由**关闭**时使用了 `auto`/`fast`/`best`，或在非聊天端点上使用 | 开启智能路由，或改用具体模型名 |
| `429 rate_limit_exceeded` | 单密钥 RPM 上限或鉴权桶 | 调高 `rate_limit_rpm`，或等固定的 60 秒窗口过去 |
| `429 budget_exceeded` / `user_budget_exceeded` | 密钥或账户预算用完 | 调高预算或重置（账户预算只有管理员能重置） |
| `413 body_size_exceeded` | 请求体超过 `max_request_body_size_bytes`（默认 64 MiB） | 调高上限或缩小请求 |
| 日志里出现 `499 client_disconnected` | 响应完成前套接字已关闭：SDK/浏览器的读超时、你自己反向代理里的 `proxy_read_timeout`，或 CDN 524（信息里提到 Cloudflare 只是举了最典型的例子——任何中间层看起来都一样） | 调高客户端/代理的读超时，保持保活开启，或者改用流式而不是干等一个非流式长请求。最后一帧之后的正常关闭会记为成功，不是 499 |
| 响应是 **200 但正文是错误 JSON** | 失败发生前保活心跳模式已经提交了 200 | 这是预期的取舍；检查响应体，实际错误在日志里。见 [反向代理与 TLS](/zh/deployment/reverse-proxy) |
| 浏览器报 CORS 失败，而不是真实状态码 | 鉴权/请求体/速率限制中间件运行在 CORS **外层** | 用 `curl` 复现拿到真实状态码，再修根因 |
| 流式请求卡住没有数据 | 上游长时间无输出；代理正在发送 `: keep-alive` 注释帧 | 这是正常的。如果反向代理会缓冲响应，请关掉缓冲 |
| `/` 下没有管理界面（API 正常） | 启动时 `frontend/dist` 不存在 | 运行 `uv run llm-proxy --build-frontend` 后重启 |
| 日志页面有记录，但正文是 `{"_bodies_disabled": true}` | 预期默认值：**记录输入/输出** 关闭，因此所有内容负载（请求/响应体、上游错误体、MCP/网络搜索的参数与结果）都被抹掉，元数据行仍然照写 | 在 设置 → 日志管理 里打开 **记录输入/输出** 才会存正文。总开关关闭时，`x-log-full: true` 也无法重新开启 |
| 正文日志已开启，但日志页面某些行是 `{"_sampled_out": true}` | **正文采样率** 低于 1.0，该请求没被抽中。特意与「已禁用」标记区分开 | 把 **正文采样率** 调到 1.0，或在需要抓取的请求上发送 `x-log-full: true` |
| 用量统计为空，但有日志 | 用量和日志是分开写入的 | 检查日期范围；用量记录会按日志保留窗口被清理 |
| Responses 端点返回 `503 redis_not_available` | 未启用 Redis，但存储响应需要它 | 启用 Redis，或避开响应存储相关端点 |
| MCP 服务器拒绝启动 | 默认拒绝策略：它的命令或环境变量不在白名单里 | 在 设置 → 常规 → MCP 安全 里把确切命令加入白名单（见 [MCP 服务器](/zh/admin/mcp)） |
| MCP `403 Access denied to MCP server` | 密钥白名单不包含该服务器 | 更新该密钥的 MCP 白名单 |
| MCP 对存在的服务器返回 404 | 服务器没在运行（已停止/已禁用，或启动时崩溃） | 在控制台查看状态；先禁用再启用即可重启 |
| MCP 改动不生效 | 更新不会重启正在运行的服务器 | 先禁用该服务器，再重新启用 |
| `web_search` 工具调用返回 `web_search_tool_result_error` | 搜索后端连不上 / 被限流 / 密钥有误 | 模型仍会作答；查看网络搜索日志标签页和后端配置 |
| 智能路由选中的模型出乎意料 | 分类器信号或模式权重 | 检查 设置 → 高级 → 智能路由；使用路由诊断（`verbose_routing_logs`）和逐请求反馈 |
| Claude Code 提示会话「isn't eligible」使用免费自动模式 | 上游无法执行服务端自动模式检查 | 见下文 [Claude Code 自动模式](#claude-code-auto-mode) |

## Claude Code 自动模式 {#claude-code-auto-mode}

Claude Code（v2.1.278+）的自动模式会要求上游做安全检查：在请求里加上 `safeguards`
字段和 `dangerous-tool-use-2026-09-03` 这个 `anthropic-beta` 值；上游必须在响应里
返回 `safeguard_results` 字段（流式时位于末尾的 `message_delta` delta 内），并以
工具调用 id 为键。两者都没回来时，Claude Code 会回退到自己的分类器——按 Token 用量
计费——并在第一次本应检查某个操作时显示一条提示。

这不会弄坏任何东西，自动模式照常工作。

- 非 Anthropic 上游完全无法提供这项能力：Ollama、vLLM、SGLang，以及第三方
  Claude 兼容提供商（z.ai、Kimi、MiniMax、Moonshot、DeepSeek、Qwen）都没有服务端
  检查。走这些路由时出现该提示是正常的。
- Anthropic 格式的上游只要这些字段能端到端保留就能工作。本代理的原生 Anthropic
  透传会原样转发 `safeguards`、完整的 `anthropic-beta` 值以及所有响应键；转换路径也
  会保留 `safeguard_results`。
- 想消掉这条提示，在启动 Claude Code 的环境里设置
  `CLAUDE_CODE_AUTO_MODE_SERVER=0`。这样它就不再请求服务端检查，提示也会消失。

想确认某个会话处于哪种状态，在 Claude Code 里运行 `/status`，看 **Auto mode
server** 这一行：`Enabled` 表示由服务端执行检查，`Disabled` 表示会话已经回退了。

## 获取更多细节

- 请求 ID：每个响应都带 `X-Request-Id`，用它在日志页面搜索。注意：入站的
  `x-request-id` 会被用作追踪/路由 id，而响应头里始终是服务端生成的 id。
- 应用日志：`LOG_LEVEL=DEBUG`（需要重启）或 `--log-file`。
- 审计轨迹：管理操作和安全事件在 日志 → 审计日志，可用
  **校验完整性** 证明哈希链完好。
- 提供商失败：错误会带上上游状态码和消息；502 表示提供商/网络/解析失败，
  504 表示超时（上游调用固定 600 秒读超时、10 秒连接超时；提供商配置里的 `timeout`
  字段不会生效）。
