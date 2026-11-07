---
pageClass: api-reference kicker-overview
aside: false
---

# 认证

代理有两套凭据体系，二者互不通用：

| 凭据 | 适用路径 | 用途 |
| --- | --- | --- |
| **API 密钥**（`sk-…`） | `/v1/*`、`/servers/*` | 你的应用和 MCP 客户端 |
| **控制台 JWT**（登录后获得） | `/api/*` | 管理界面和管理 API |

有效的控制台 JWT 在 `/v1/*` 和 `/servers/*` 上会被**拒绝**，API 密钥在 `/api/*` 上
同样会被拒绝。

## API 密钥

支持两种写法：

```http
Authorization: Bearer sk-your-proxy-key
```

```http
x-api-key: sk-your-proxy-key
```

- 格式：`sk-` 加 64 个十六进制字符。密钥以 SHA-256 摘要存储，明文只在创建时显示一次。
- 控制台为游乐场创建的会话密钥（`sk-ui-…`）在代理路由上同样有效，并继承其所属
  用户的限制。
- `Authorization` 的值不是 `Bearer` 形式 → `401 Invalid authorization header format`；
  完全没有该请求头 → `401 Authorization header missing`。

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "x-api-key: $LLM_PROXY_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "hi"}]}'
```

### WebSocket

`/v1/responses` 和 `/v1/realtime` 可以通过 `Authorization`、`x-api-key` 或
`api_key` 查询参数传递密钥（Realtime 还支持
`openai-insecure-api-key.<key>` 子协议）。优先用请求头——查询字符串可能被反向代理的
访问日志记录下来。

## 控制台 JWT

由 `POST /api/auth/login` 获取（首次启动时用 `POST /api/auth/setup`）。令牌是
HS256，有效期 24 小时，没有刷新端点。修改密码、修改角色或被停用，都会在服务端
使所有已签发的令牌失效。

浏览器控制台**不会**把 JWT 放进 JavaScript 可访问的存储。登录时设置两个
`HttpOnly`、`SameSite=Lax` 的 Cookie（HTTPS 下为 `Secure`）：

| Cookie | 路径 | 内容 |
| --- | --- | --- |
| `llm_proxy_console` | `/api` | 控制台 JWT |
| `llm_proxy_session_key` | `/v1` | 供游乐场使用的短期会话 API 密钥 |

以 Cookie 鉴权的不安全请求还必须带上 `X-CSRF-Token`，其值与可读的
`llm_proxy_csrf` Cookie 一致（双重提交 CSRF 校验）。`GET /api/auth/session` 会
报告该 Cookie 当前对应的身份，但不会因 Cookie 过期而失败：如果令牌所属的会话已被
轮换或撤销，接口返回 `authenticated: false` 且不带用户名，而不是报出它过去所属的
账号。退出登录会清除这三个 Cookie，并撤销**自己**在服务端的会话——如果某个浏览器
的会话已被另一次登录轮换掉，它什么也取消不了，较新的会话继续有效。

脚本客户端仍可以把返回的 `access_token` 作为 `Authorization: Bearer <jwt>` 发送；
显式的 `Authorization` 或 `x-api-key` 请求头总是优先于 Cookie。在 `/v1/*` 上，只有
两个请求头都不存在、且请求对不安全方法带有有效 CSRF 令牌时，才会接受会话密钥
Cookie。

登录标识可以是用户名，也可以是已验证的邮箱别名。如果账号启用了 TOTP，用密码或 OIDC
登录时响应是 `{ "mfa_required": true, "challenge_token": "…" }` 而不是访问令牌；接着用
`POST /api/auth/2fa/verify` 完成验证（一次性恢复码用 `/api/auth/2fa/recover`）。
通行密钥登录不在此列：断言本身就要求用户验证，单次登录已经同时证明了设备与持机人，
因此直接返回访问令牌。如果部署方要求每种主因素之后都输一次验证码，可以设置
`console_auth.require_2fa_after_passkey`（`PUT /api/config/server/console-auth`），
把通行密钥登录重新纳入第二步。

每个用户同时只有一个活动的控制台会话。JWT 里带 `sid` 声明；在另一台设备登录会轮换
会话，并立刻使之前的 JWT 失效。敏感操作额外要求在 15 分钟内完成 step-up
（用当前密码调用 `POST /api/auth/step-up`，或走通行密钥的 step-up 端点）。

控制台鉴权失败使用下列稳定错误码，界面无需解析文案即可响应：

| 错误码 | 状态码 | 含义 |
| --- | --- | --- |
| `step_up_required` | 403 | 会话最近一次强认证已超过 15 分钟；调用 `POST /api/auth/step-up`（或通行密钥 step-up 流程）后重试 |
| `session_revoked` | 401 | 较新的登录轮换了唯一的活动会话；请重新登录 |
| `password_change_required` | 403 | 管理员创建或重置的密码必须先更换，才能访问其他 API |
| `mfa_enrollment_required` | 403 | 该实例要求管理员必须登记 TOTP；此时只有登记流程、安全页面的读取操作、step-up、个人资料和退出登录路由可达 |
| `mfa_challenge_expired` | 401 | 短期双因素认证挑战已过期；请重新登录 |
| `invalid_current_password` | 400 | 自助修改密码或用户名时提供的当前密码错误；控制台会话仍然有效 |
| `invalid_mfa_code` | 400 | 提交的 TOTP 或恢复码无效 |
| `last_factor` | 400 | 该操作会移除账号最后一个可用的登录因素 |
| `oauth_onboarding_only` | 400 | 只有在完成一个**新的**受邀 OAuth 账号时才能选择用户名/密码 |
| `username_required` | 400 | 新的受邀 OAuth 账号必须先选择用户名，才能建立第一个控制台会话 |
| `oidc_id_token_invalid` | 401 | 提供商的 ID 令牌未通过签名、签发方、受众或 nonce 校验 |
| `email_verification_superseded` | 401 | 验证链接早于账号当前地址（已被使用，或发送后地址已变更） |
| `password_reset_invalid` | 401 | 密码重置链接格式错误、已过期，或所属账号不允许重置（已停用、无密码、邮箱未验证） |
| `password_reset_superseded` | 401 | 重置链接已被使用，或发送后密码已变更；请重新申请 |
| `oauth_link_expired` | 401 | 提供商的关联交接令牌已过期；请重新发起关联 |
| `oauth_link_mismatch` | 401 | 关联交接令牌属于另一个账号 |
| `oauth_link_session_changed` | 401 | 发起关联的控制台会话已不是当前活动会话 |

## 公开端点

以下端点无需凭据：

- `GET /api/health`、`/api/health/ready`、`/api/health/live`
- `POST /api/auth/setup`、`GET /api/auth/setup-status`、`POST /api/auth/login`
- `GET /api/auth/capabilities` — 登录页在当前环境下可以提供的功能
  （只有配置了 SMTP 投递和公开 base URL 时 `password_reset` 才为 true）
- `POST /api/auth/password/forgot`、`POST /api/auth/password/reset` — 重置请求
  始终返回 `202 {"requested": true}`，绝不透露账号是否存在；确认步骤的凭据就是
  重置链接本身
- `GET /api/auth/session`（凭据可选：没有有效 Cookie 时返回
  `{ "authenticated": false }` 而不是 401）
- `POST /api/auth/2fa/verify`、`POST /api/auth/2fa/recover`
- `POST /api/auth/email/verify`
- `GET /api/auth/invitations/{token}`、`POST /api/auth/invitations/{token}/accept`
- `GET /api/auth/oauth/providers`，以及 OAuth 的 begin/callback/complete 路由
  （**callback** 只校验提供商的应答；完成账号**关联**需要带着发起关联的那个控制台
  会话调用 `POST /api/me/oauth/link/complete`。PKCE verifier 通过 HttpOnly Cookie
  交给 callback，绝不经过 `state`。）
- `POST /api/auth/passkeys/login/begin`、`POST /api/auth/passkeys/login/finish`
- 管理界面的静态文件（SPA 本身）

`/api/*` 下的其他所有路径都需要 JWT；`/v1/*` 和 `/servers/*` 下的所有路径都需要
API 密钥。

## 失败处理

| 情况 | 响应 |
| --- | --- |
| 密钥缺失或未知 | 401，使用当前协议的错误信封 |
| 同一 IP 反复提交无效密钥 | 启用 API 密钥 IP 锁定后返回 **429** `too_many_auth_failures`（默认关闭；10 次失败 → 300 秒，计数器按进程计并在重启时清零）。`/v1/*` 的 429 带 `Retry-After`，`/servers/*` 的 429 不带 |
| 模型不在密钥白名单内 | 403 `model_not_allowed`。这不属于鉴权失败：密钥已通过认证，因此不会触发 IP 锁定，也不会增加延迟 |
| MCP 服务器不在密钥白名单内 | 403 `Access denied to MCP server '<name>'` |
| 预算耗尽 | 429 `budget_exceeded`（账户预算为 `user_budget_exceeded`） |
| 鉴权失败 | 响应会被延迟 `auth_failure_delay_ms`（默认 100 毫秒，±10% 抖动）以减缓暴力破解 |

鉴权失败、锁定和登录都会记入审计日志。

## 相关

- [API 密钥](/zh/admin/api-keys) — 创建带预算和白名单的密钥
- [错误与速率限制](/zh/api/errors) — 具体的响应体
