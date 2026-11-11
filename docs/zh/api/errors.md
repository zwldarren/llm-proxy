---
pageClass: api-reference kicker-overview
aside: false
---

# 错误与速率限制

## 错误信封

代理用客户端所用的协议返回错误。

### OpenAI

```json
{
  "error": {
    "message": "Model 'foo' not found in configuration",
    "type": "not_found_error",
    "code": "model_not_found",
    "param": null,
    "error_id": "9f2c…"
  }
}
```

`code` 和 `param` 没有值时会省略——示例展示的是字段最全的形态。

### Anthropic

```json
{
  "type": "error",
  "error": { "type": "not_found_error", "message": "…" }
}
```

代理自己生成的 Anthropic 错误不带 `request_id`；改用响应头 `X-Request-Id` 关联。
只有当上游错误体本身就是提供商格式并被原样透传时，才会出现 `request_id`，此时保留的
是提供商自己的 id。

### Responses API

结构与 OpenAI 相同，只是 `type` 被映射成规范里的码：
`invalid_request_error → invalid_request`、`authentication_error →
authentication_failed`、`permission_error → permission_denied`、`not_found_error →
not_found`、`rate_limit_error → too_many_requests`、`api_error`/`provider_error` →
`server_error`。

管理 API（`/api/*`）里由 JWT 中间件拒绝的请求（令牌缺失或无效、令牌被撤销、账号被
停用）以及请求校验失败，使用 FastAPI 的 `{"detail": …}` 结构。管理路由内部抛出的
异常会交给全局异常处理器，输出与其他地方相同的协议信封。

`error_id` 就是请求 id——在**日志**里搜索它即可看到完整请求。未处理的 5xx 消息会被
清洗为 `Internal server error`，详情只留在服务器日志中。

## 状态码

| 状态码 | 常见错误码 | 含义 |
| --- | --- | --- |
| 400 | `ValidationError`、`invalid_content_length`、`payload_too_deeply_nested`、`no_available_models`、`routing_constraints_unmet` | 请求或请求体格式错误；被路由的请求没有模型能承接（图像能力、上下文长度） |
| 401 | `authentication_error` | API 密钥缺失或未知 |
| 403 | `model_not_allowed`、`forbidden`、`allowlist_exhausted`、`password_change_required` | 被白名单/角色/账号状态拒绝；`allowlist_exhausted` 表示路由池已耗尽密钥的模型/提供商限制；被停用的账号访问 `/api/*` 会得到 `{"detail": "Account is disabled"}` |
| 404 | `model_not_found`、`not_found`、`mcp_server_not_found`、`provider_not_configured` | 模型/响应/服务器不存在；已存储的响应 24 小时后过期 |
| 409 | `conflict` | 重复提交反馈，或取消一个不可取消的响应 |
| 413 | `body_size_exceeded` | 超出 `max_request_body_size_bytes`（默认 64 MiB） |
| 429 | `rate_limit_exceeded`、`too_many_auth_failures`、`budget_exceeded`、`user_budget_exceeded` | 速率限制、锁定或预算上限。速率限制、单密钥 RPM 以及 `/v1/*` 的锁定 429 带 `Retry-After`；`/servers/*` 的 MCP 锁定 429、路由时的 `max_cost` 上限以及预算拒绝都不带 |
| 499 | `client_disconnected` | 客户端或中间环节在响应中途放弃；上游调用被取消（按失败记录）。SDK、浏览器、反向代理、Cloudflare 的各种读超时，从这里看都一样；最后一个分片之后的关闭则记为成功 |
| 500 | `configuration_error`、`internal_error` | 配置有问题（如智能路由未开启、模型缺失），或上游响应解析失败（500 `api_error`） |
| 502 | `provider_error`、`network_error` | 上游失败 |
| 503 | `redis_not_available`、`budget_check_unavailable`、`web_search_error`、就绪检查 | 依赖缺失或不健康；预算检查按失败关闭处理 |
| 504 | `timeout_error` | 上游读超时（600 秒） |

## 响应头

| 响应头 | 出现时机 |
| --- | --- |
| `X-Request-Id` | 始终（主中间件链）——日志页面和 `error_id` 里显示的就是这个 id |
| `Retry-After` | 速率限制、`/v1/*` 锁定，以及转发的上游 429（`/servers/*` 的 MCP 锁定 429 和预算拒绝不设置） |
| 上游的 `x-ratelimit-*` / `anthropic-ratelimit-*` / `retry-after` | 在流式响应上转发 |

代理不会自行生成 `X-RateLimit-*` 响应头。注意入站的 `x-request-id` 会用作用于
追踪/路由 id，而响应头里始终是服务器生成的 id。

## 速率限制

| 限制 | 范围 | 默认值 |
| --- | --- | --- |
| 单密钥 RPM | 单个 API 密钥，60 秒滑动窗口 | 密钥上未设置时无限制 |
| `auth.login` | 客户端 IP | 5 次/分钟 |
| `auth.setup` | 客户端 IP | 5 次/分钟 |
| `auth.setup_status` | 客户端 IP | 10 次/分钟 |
| 其他控制台鉴权桶：`auth.capabilities`、`auth.mfa`、`auth.step_up`、`auth.invite_accept`、`auth.invite_status`、`auth.passkey`、`auth.oauth`、`auth.email_verify`、`auth.password_reset`（5 次/分钟）、`auth.password_reset_confirm`（10 次/分钟）、`account.sensitive` | 客户端 IP | 见[服务器设置 → 速率限制](/zh/admin/settings#rate-limits)中的默认值 |
| `config.test_email` | 客户端 IP | 5 次/分钟 |
| API 密钥失败尝试 | 客户端 IP | **默认关闭。** 开启 `api_key_lockout_enabled` 后：10 次失败 → 锁定 300 秒（`too_many_auth_failures`） |
| 登录失败 | 用户名 | 默认关闭；开启后 5 次失败 → 锁定 900 秒（`account_locked`） |

以上每一行都会被 `rate_limit_disabled`（设置 → 高级 → 安全与速率限制）绕过，该开关同时跳过鉴权失败
延迟。预算上限不受影响。

每个速率限制项都按客户端 IP 各自维护滑动窗口：轮询 `auth.setup_status` 不会消耗
`auth.login` 的配额，单密钥 RPM 上限也是跟着密钥走，而不是跟着请求来源地址。

- 各鉴权桶可以在[设置 → 高级 → 速率限制](/zh/admin/settings#rate-limits)里按桶
  配置，写法为 `N/period`（`5/minute`、`100/hour`）。
- 单密钥限制（`rate_limit_rpm`）在创建/编辑密钥时设置，仅管理员可改。
- 启用 Redis 速率限制后，窗口在工作进程之间共享，失败模式由
  `redis_rate_limit_fail_closed` 控制（默认：Redis 报错时阻断）。

## 预算

预算拒绝是 429，错误码不同：

| 错误码 | 预算 |
| --- | --- |
| `budget_exceeded` | API 密钥自身的预算 |
| `user_budget_exceeded` | 所属用户的账户预算（由管理员管理） |

预算检查按**失败关闭**处理：读不到花费时直接拒绝请求
（`503 budget_check_unavailable`）。

## 重试与回退

客户端能看到的失败，都是代理自己的重试/回退链已经耗尽之后的结果：

- 在同一提供商内对 `408/429/500/502/503/504` 以及网络/超时错误重试
  （`max_retries`，默认 3）。
- 跨提供商回退，最多 `max_fallback_attempts` 次（默认 10）。只要还有其他提供商，任何
  错误都可以触发回退——从不依据上游状态码或错误类型来阻止回退，因为提供商经常误报
  （上下文窗口溢出常常就是个普通的 `400`）。

如果全都失败，客户端收到的是最后一个上游错误（或路由约束错误）。日志条目会记录所有
尝试（`fallback_count`、`retry_count`）。
