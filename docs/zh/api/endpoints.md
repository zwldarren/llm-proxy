---
pageClass: api-reference kicker-overview
aside: false
---

# 端点索引

代理提供的全部路由。鉴权列里的 `X` 表示 API 密钥
（`Authorization: Bearer sk-…` 或 `x-api-key: sk-…`）；`J` 表示控制台 JWT
（`Authorization: Bearer <jwt>`）；`—` 表示公开访问。

## LLM 端点

| 方法 | 路径 | 鉴权 | 协议 | 说明 |
| --- | --- | --- | --- | --- |
| POST | `/v1/chat/completions` | X | OpenAI | Chat Completions |
| POST | `/v1/responses` | X | Responses | JSON 或 SSE；`background: true` 需要 Redis |
| POST | `/v1/responses/compact` | X | Responses | 优先原生透传，回退到本地无损打包 |
| POST | `/v1/responses/input_tokens` | X | Responses | 不调用模型即可估算 prompt 大小；上游支持时用上游的原生计数，否则用本地 o200k_base 估算 |
| GET | `/v1/responses/{response_id}` | X | Responses | 读取已存储的响应（TTL 24 小时） |
| DELETE | `/v1/responses/{response_id}` | X | Responses | 删除已存储的响应 |
| POST | `/v1/responses/{id}/cancel` | X | Responses | 取消后台响应（不可取消时返回 409） |
| GET | `/v1/responses/{id}/input_items` | X | Responses | `limit` 取 1–100（默认 20），`order` 为 asc/desc，`after` 指定游标 |
| POST | `/v1/messages` | X | Anthropic | Anthropic Messages |
| POST | `/v1/messages/count_tokens` | X | Anthropic | 上游支持时用上游的原生计数，否则用本地 o200k_base 估算 |
| POST | `/v1/embeddings` | X | OpenAI | Embeddings |
| POST | `/v1/rerank` | X | Cohere 兼容 | 按查询语句重排文档（`/v2/rerank` 是给 Cohere SDK 用的别名）。指定的模型必须带 `supports_rerank` 标志，否则返回 `400 invalid_request_error` |
| POST | `/v1/systemone` | X | System One | TypeSafe Jev 评估（也可经 OpenRouter 和本地 Ollama 模型）。指定的模型必须带 `supports_systemone` 标志，否则返回 `400 invalid_request_error` |
| POST | `/v1/decisions` | X | Decisions | OpenAI 的 Decisions 端点（`gpt-6-luna`），System One 模型也可通过代理的桥接提供。受同一个 `supports_systemone` 标志约束 |
| POST | `/v1/images/generations` | X | OpenAI | JSON 请求体 |
| POST | `/v1/images/edits` | X | OpenAI | JSON 或 multipart（`image`、`image[]`、`mask`） |
| POST | `/v1/audio/speech` | X | OpenAI | 返回二进制音频 |
| POST | `/v1/audio/transcriptions` | X | OpenAI | multipart；`model` 和 `file` 必填 |
| POST | `/v1/audio/translations` | X | OpenAI | multipart；`model` 和 `file` 必填 |
| GET | `/v1/models` | X | — | 模型列表，按密钥的白名单过滤 |
| GET | `/v1/protocols` | X | — | 已注册的协议及其基础路径 |

::: note 路径别名
协议路由同时接受末尾带斜杠的形式，以及不同 SDK 版本使用的别名——例如
`POST /chat/completions` 和 `POST /v1/v1/chat/completions` 与标准路径进入同一个
处理器。
:::

## 能力矩阵 {#capability-matrix}

各提供商支持的能力并不一致，下表列出谁能提供什么。

| 能力 | 提供商 |
| --- | --- |
| Chat | 所有类型 |
| Embeddings | `openai`、`openai-compatible` 系列（含 DeepSeek、Kimi、MiniMax、Moonshot、Qwen、xAI、vLLM、SGLang、Chutes、Mistral、NanoGPT、OpenRouter）、`gemini`、`ollama`、`cohere` |
| Rerank | `cohere`（原生 `/v2/rerank`）、`openrouter`（`/api/v1/rerank`，例如 `cohere/rerank-v3.5`）、上游提供 `/rerank` 的 `openai-compatible` 系列（vLLM、SGLang、Jina）。受 `supports_rerank` 标志约束 |
| 图像（生成 + 编辑） | `openai`、`openai-compatible` 系列、`gemini`、`qwen` |
| 音频（语音、STT） | `openai`、`openai-compatible` 系列**但 `qwen`/`qwen-intl` 除外**（DashScope 只在原生端点上提供语音——见下文）、`gemini`（原生 TTS/STT）、`ollama`（仅语音，v0.40.8+——WAV/PCM）、`openrouter`（STT）、`cohere`（仅 STT——v2 没有语音合成） |
| 音频（翻译） | `openai`、`openai-compatible` 系列**但 `qwen`/`qwen-intl` 除外**、`gemini`。`openrouter`、`cohere` 以及 `qwen`/`qwen-intl` 没有对应的上游端点，会以 `400` 拒绝请求 |
| System One（评估） | `typesafe`（直接对接 Jev）、`openrouter`（转售的 TypeSafe Jev，以及它的其他决策模型）、`ollama`（本地和云端模型，v0.35+）、`opencode-zen`/`opencode-go` 网关线路（按模型区分方言）。对 `ollama` 的 Chat 和 embeddings 仍走原生 `/api/*` 接口 |
| Decisions（评估） | `openai`（原生 `/v1/decisions`），以及上面所有 System One 提供商——代理会把它们桥接到 System One 路由。图像按上游文档说明的渠道传递（`assets`、`state` 部分或 `images`）；托管的图片 URL 会先下载并内联 |

专用的 `anthropic` 提供商类型只提供 chat。`typesafe` 提供商类型只提供 System One
——把它分配给承接 `/v1/systemone` 的模型，并注意对这类模型的 chat 请求会以 `400`
拒绝。

`qwen`/`qwen-intl` 不提供任何 `/audio/*` 路由：DashScope 没有 OpenAI 兼容的音频
接口。语音合成（Qwen-TTS、CosyVoice）走原生
`/api/v1/services/aigc/multimodal-generation/generation` 端点，OpenAI 兼容的语音识别
（`qwen3-asr-flash`）则通过 `/chat/completions` 加 `input_audio` 内容片段调用。对
qwen 模型的音频请求会被拒绝，返回 `400` `invalid_request_error` 并说明这一限制。

能力标志是叠加的：`supports_rerank` 把模型标记为重排模型——没有这个标志时，
`/v1/rerank` 指定该模型会被拒绝并返回 `400 invalid_request_error`；但该标志不会
限制模型的 chat 能力。`supports_systemone` 把模型标记为决策模型，同时约束
`/v1/systemone` 和 `/v1/decisions`——它们只是同一套评估原语上的两种信封，由提供商
在两者之间转换（见 [Decisions](/zh/api/decisions#one-endpoint-two-envelopes)）。凡是
提供 `/v1/systemone` 的提供商也提供 `/v1/decisions`。该标志不影响模型的 chat 能力，
因此当模型背后的提供商两者都支持时，它可以同时提供 chat 和评估。向任一评估端点
请求一个没有该标志的模型，会在调用任何提供商之前被拒绝，返回
`400 invalid_request_error`。该标志还决定模型能否被选为
[路由判定器](/zh/api/routing#routing-judge)。

## WebSocket 端点

| 路径 | 鉴权 | 说明 |
| --- | --- | --- |
| `WS /v1/responses` | X | 与 SSE 流相同的事件；客户端发送 `{"type":"response.create", …}` |
| `WS /v1/realtime` | X | OpenAI Realtime 中继；`?model=` 必须解析到 `openai`/`openai-compatible` 提供商上带 `supports_realtime` 的模型 |

## MCP

| 方法 | 路径 | 鉴权 | 说明 |
| --- | --- | --- | --- |
| POST | `/servers/{server_name}/mcp` | X | 针对某个已配置服务器的 Streamable-HTTP MCP 端点（支持所有 MCP 方法） |

## 健康检查

| 路径 | 鉴权 | 说明 |
| --- | --- | --- |
| GET | `/api/health` | — | 数据库与 Redis 快照，外加速率限制状态，始终返回 200 |
| GET | `/api/health/ready` | — | 依赖未就绪时返回 503 |
| GET | `/api/health/live` | — | 仅表示存活 |

## 管理 API

全部位于 `/api` 下，除特别说明外都需要 JWT。分组如下：

| 前缀 | 内容 |
| --- | --- |
| `/api/auth/*` | `setup-status`、`capabilities`、`session`、`setup`、`login`、双因素验证/恢复、step-up、通行密钥登录、OAuth、接受邀请、密码重置（forgot/confirm）、`logout` |
| `/api/api-keys*` | 密钥增删改查、预算重置、单密钥用量、花费汇总 |
| `/api/config/providers*` | 提供商增删改查、提供商类型、密钥查看、上游模型列表 |
| `/api/config/models*` | 模型增删改查、定价/元数据同步端点、模型名 |
| `/api/config/server/*` | 所有服务器设置分区（日志、网络搜索、智能路由、提供商选择、请求策略、韧性、安全、保活心跳、速率限制、CORS、MCP 安全） |
| `/api/config/circuit-breaker*` | 熔断器状态与重置 |
| `/api/logs*` | 日志查询/详情/统计、用量统计、审计校验、清理 |
| `/api/routing/*` | 训练数据导出：反馈标签和判定器结论，输出为 JSONL（`/api/routing/export/dataset`） |
| `/api/mcp/*` | MCP 服务器增删改查、状态、能力 |
| `/api/team/members*` | 成员增删改查、角色、模型白名单、账户预算、账户速率限制、密码重置、邮箱 |
| `/api/team/invitations*` | 创建/列出/撤销/重发一次性邮件邀请 |
| `/api/me/*` | 个人资料、预算、密码、用户名、邮箱验证、双因素认证、通行密钥、已关联的 OIDC 身份、追踪、反馈 |
| `/api/catalog/models` | 展示用目录（任何已登录用户） |
| `/api/system/info` | 版本与更新检查（管理员） |

## 静态前端

当 `frontend/dist` 存在时，`/assets/*` 提供编译后的资源，其他任何不属于
`/api/*`、`/v1/*`、`/servers/*` 的 GET 路径都回退到 `index.html`（SPA 路由）。
没有构建产物时，这些路径返回 404，只有 API 可用。

## 不可用的端点

- 运行时的 OpenAPI / Swagger / ReDoc：未注册（`docs_url=None`、
  `redoc_url=None`、`openapi_url=None`），因此 `/docs`、`/redoc`、`/openapi.json`
  永远不会挂载。对这些路径发 GET 会走
  [静态前端](#static-frontend) 一节的 SPA 回退：`frontend/dist` 存在时返回
  `index.html`（200），否则 404。生成的 `/v1` 契约以
  [OpenAPI 规范](/zh/api/openapi) 的形式发布在本套文档里。
- 日志导出端点：改为分页查询 `GET /api/logs`。

## 相关

- [认证](/zh/api/authentication) — 请求头与凭据
- [错误与速率限制](/zh/api/errors) — 各类失败响应
