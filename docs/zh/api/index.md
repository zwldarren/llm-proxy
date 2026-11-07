---
pageClass: api-reference
aside: false
---

# API 参考

Base URL：`http://localhost:8080`（你的代理地址）。所有 LLM 端点都支持三种客户端
协议：OpenAI Chat Completions、OpenAI Responses 和 Anthropic Messages。网关负责在
这些协议与已配置的上游之间做翻译。

所有 `/v1/*` 路由用同一个 API 密钥鉴权：`Authorization: Bearer sk-…`，详见
[认证](/zh/api/authentication)。

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $LLM_PROXY_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## 聊天

覆盖三种协议的对话补全，以及响应存储与 Token 计数。

- [创建聊天补全](/zh/api/chat/create-completion) — `POST /v1/chat/completions`
- [创建响应](/zh/api/chat/create-response) — `POST /v1/responses`（SSE + WebSocket）
- [检索响应](/zh/api/chat/retrieve-response) — `GET /v1/responses/{response_id}`
- [删除响应](/zh/api/chat/delete-response) — `DELETE /v1/responses/{response_id}`
- [取消响应](/zh/api/chat/cancel-response) — `POST /v1/responses/{id}/cancel`
- [列出输入条目](/zh/api/chat/list-input-items) — `GET /v1/responses/{id}/input_items`
- [压缩响应](/zh/api/chat/compact-response) — `POST /v1/responses/compact`
- [Anthropic Messages](/zh/api/chat/messages) — `POST /v1/messages`
- [Token 计数](/zh/api/chat/count-tokens) — `POST /v1/messages/count_tokens`

## 图像

- [生成图像](/zh/api/images/create) — `POST /v1/images/generations`
- [编辑图像](/zh/api/images/edit) — `POST /v1/images/edits`

## 音频

- [生成语音](/zh/api/audio/speech) — `POST /v1/audio/speech`
- [语音转写](/zh/api/audio/transcription) — `POST /v1/audio/transcriptions`
- [语音翻译](/zh/api/audio/translation) — `POST /v1/audio/translations`

## Embeddings

- [创建 Embeddings](/zh/api/embeddings) — `POST /v1/embeddings`

## Rerank

- [重排文档](/zh/api/rerank) — `POST /v1/rerank`

## 批量

- [批量模式](/zh/api/batch) — `POST /v1/batches` 配合 `/v1/files`，转发到 OpenAI、OpenRouter 或 Gemini

## 模型

- [列出模型](/zh/api/models) — `GET /v1/models`

## 进阶与通用主题

- [认证](/zh/api/authentication) — API 密钥与控制台 JWT
- [OpenAPI 规范](/zh/api/openapi) — 由代码生成的 `/v1` 契约
- [端点索引](/zh/api/endpoints) — 每个路由，包括 WebSocket、MCP、健康检查与管理 API
- [流式响应](/zh/api/streaming) — 各协议的 SSE 格式与 WebSocket 传输
- [错误与速率限制](/zh/api/errors) — 错误结构、状态码、预算
- [工具、推理与网络搜索](/zh/api/tools) — 工具调用、思考、网络搜索拦截
- [虚拟模型与路由](/zh/api/routing) — `auto` / `fast` / `best`

## 约定

- `model` 始终是代理里配置的名称，包括智能路由开启时的虚拟模型 `auto`、`fast`、
  `best`。
- 响应里的 `model` 字段回显客户端请求的别名，不是解析后的上游模型。
- 每个响应都带 `X-Request-Id`；反馈问题时请附上它。
- 上游的速率限制响应头（`x-ratelimit-*`、`retry-after`）会原样转发。
