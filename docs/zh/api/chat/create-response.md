---
pageClass: api-reference kicker-chat
aside: false
---

# 创建响应

OpenAI Responses API，包含响应存储、后台模式和 WebSocket 传输。

::: endpoint POST /v1/responses
同时也提供 `/responses` 和 `/v1/v1/responses` 两个别名。
:::

::: endpoint WS /v1/responses
事件对象与 SSE 流相同；客户端发送 `{"type": "response.create", …}`。详见
[WebSocket 传输](#websocket-transport)。
:::

```bash [请求]
curl http://localhost:8080/v1/responses \
  -H "Authorization: Bearer $LLM_PROXY_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "input": "Hello!"}'
```

```python [请求 — OpenAI SDK]
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-your-proxy-key")
resp = client.responses.create(model="auto", input="Hello!")
print(resp.output_text)
```

```json [响应]
{
  "id": "resp_9f2c…",
  "object": "response",
  "created_at": 1718047200,
  "status": "completed",
  "model": "auto",
  "output": [
    {
      "id": "msg_9f2c…",
      "type": "message",
      "role": "assistant",
      "status": "completed",
      "content": [
        {"type": "output_text", "text": "Hello! How can I help?", "annotations": []}
      ]
    }
  ],
  "usage": {
    "input_tokens": 25,
    "output_tokens": 9,
    "total_tokens": 34,
    "input_tokens_details": {"cached_tokens": 0},
    "output_tokens_details": {"reasoning_tokens": 0}
  }
}
```

## 请求

必填：`model`、`input`（字符串或条目列表）。

输入条目：`message`（角色可为 user/system/developer/assistant）、`function_call`、
`function_call_output`、`reasoning`、`item_reference`、`custom_tool_call(_output)`、
`web_search_call`、`image_generation_call`、`local_shell_call(_output)`、
`tool_search_call` / `tool_search_output`、`agent_message`、`additional_tools`
（携带工具的条目——Codex 就是这样发送工具的，其中的 `tools` 会合并进实际生效的
工具集）、压缩条目，另有兜底类型，保证无法识别的条目永远不会校验失败。

内容片段：`input_text`、`input_image`（`image_url`、`detail`）、`input_file`
（`file_url` / `file_data` / `file_id`、`filename`）、`input_video`、`input_audio`
（`audio_data` / `audio_url`、`format`）、`encrypted_content`。

`input_file.file_url` 在 Responses 协议中仍以 URL 形式保留。当请求被路由到协议里没有
URL 文件输入的提供商时，URL 会放进该提供商原生的 URL 字段（Anthropic 的
`document` / `image` URL source、Gemini 的 `file_data.file_uri`、OpenRouter 的
`file.file_data`、智谱 / Z.AI 的 `file.file_url`、Mistral 的 `document_url`）；
如果协议没有 URL 文件输入但接受内联字节，则下载该 URL 并内联（Gemini，以及
OpenAI / 通用 `openai-compatible` 的 Chat Completions `file` 片段）；两者都不支持时
降级为文本占位符。详见
[FAQ](/zh/reference/faq#how-are-files-and-media-mapped-when-i-route-to-another-provider)。

关键参数：

| 参数 | 说明 |
| --- | --- |
| `instructions` | 系统级指令 |
| `store` | **默认为 true**——响应会被持久化，以便之后检索 |
| `background` | 强制 `store=true, stream=false`；需要 Redis（见下文） |
| `previous_response_id` | 续接上一个响应；先从本地存储解析，解析不到再向上游查询 |
| `truncation` | `auto` / `disabled` |
| `max_output_tokens` | 输出 Token 上限；必须 ≥ 1，并在响应中回显 |
| `top_logprobs` | 返回的 top logprobs 数量，0–20，并在响应中回显 |
| `reasoning` | `effort`（`none` … `max`）、`mode`、`context`、`summary` |
| `text` | `{format: text/json_object/json_schema, verbosity}` |
| `tools` | `function`、`web_search`/`web_search_preview`（可带过滤条件、上下文大小、用户位置）、`custom`、`tool_search`、`namespace`（子工具会被展平后还原），以及为原生上游保留的 `file_search`/`code_interpreter` |
| `tool_choice` | `auto` / `required` / `none`、`{type:"function",name}`、`{type:"allowed_tools",…}` |
| `metadata` | 最多 16 对（键 ≤ 64 字符，值 ≤ 512 字符） |
| `service_tier`、`safety_identifier`、`prompt_cache_key`、`max_tool_calls` | 在支持时转发 |
| `stream_options` | 只转发 `include_obfuscation`——Responses API 会拒绝 `include_usage` |

`conversation` 和 `prompt` 只会转发给原生 Responses 提供商，其他提供商会直接丢弃。
未知的 `include` 取值会被接受并转发。

当客户端请求 `include: ["reasoning.encrypted_content"]`，**或**响应是无状态的
（`store: false`）时，推理条目会带上 `encrypted_content`。按目前 OpenAI 的推理文档，
无状态模式默认返回加密的推理内容，`include` 则作为旧的显式请求方式保留。

来自 Anthropic 的推理条目还会把校验载荷（`thinking.signature` /
`redacted_thinking.data`）放进 `encrypted_content`，无论 `include` 或 `store` 怎么
设置——Responses 协议里没有别的字段能承载它，丢掉就会导致下一轮的扩展思考出问题。

::: tip 零数据保留（ZDR）
ZDR 组织在上游是无状态的，但请求省略 `store` 时代理看不出这一点，会按已存储处理，
从而把加密推理内容本身挡在 `include` 后面。要拿到它，请显式传 `store: false` 或
`include: ["reasoning.encrypted_content"]`。
:::

## 响应

响应对象覆盖规范里的全部表面：`id`、`status`（`queued`、`in_progress`、
`completed`、`failed`、`incomplete`、`cancelled`）、`output`、`usage`、`error`、
`incomplete_details`，以及请求参数的回显（`temperature`、`top_p`、`tool_choice`、
`truncation` 等）。

- `usage`：`input_tokens`、`output_tokens`、`total_tokens`、
  `input_tokens_details.cached_tokens`（上游报告时还有 `cache_write_tokens`）、
  `output_tokens_details.reasoning_tokens`。
- 状态推导：finish reason 为 `length`/`content_filter` → `incomplete`；出错 →
  `failed`，并带一个 `error` 对象。
- 输出条目：`message`（含 `output_text` 和 annotations）、`function_call`、
  `custom_tool_call`、`web_search_call`、推理条目，以及内部没有对应物的上游条目
  （按原位置重新插入）。
- 输出为空时仍会返回一个 message 条目，其 `output_text` 为空。

无法解析的 `previous_response_id` 会以 **400** `previous_response_not_found` 失败，
而不是悄悄续接。

## 响应存储

由于 `store` 默认为 **true**，每个响应都会被持久化：存储由 Redis 支撑，按 API 密钥
分区，**24 小时**后过期（固定值）。之后可以用下列子资源端点管理已存储的响应：

| 端点 | 页面 |
| --- | --- |
| `GET /v1/responses/{id}` | [检索响应](/zh/api/chat/retrieve-response) |
| `DELETE /v1/responses/{id}` | [删除响应](/zh/api/chat/delete-response) |
| `POST /v1/responses/{id}/cancel` | [取消响应](/zh/api/chat/cancel-response) |
| `GET /v1/responses/{id}/input_items` | [列出输入条目](/zh/api/chat/list-input-items) |
| `POST /v1/responses/compact` | [压缩响应](/zh/api/chat/compact-response) |

`store: false` 的响应无法通过 `GET` 检索（没带 API 密钥的流式响应同样不会持久化）。

### 后台模式

`background: true`：

- 需要 Redis，**并且**请求必须通过 API 密钥鉴权：否则返回 503
  `redis_not_available`（快速失败，而不是给出一个注定查不到的轮询 id）。
- 立即返回 **HTTP 200**，`status: "in_progress"`，id 已预先生成，`output: []`。
- 完成后用最终结果覆盖存储的响应体；失败时覆盖为 `status: "failed"` 并带一个
  `error` 对象。
- 用 `GET /v1/responses/{id}` 轮询，用 `/cancel` 停止。

## 流式

返回完整的事件集合，每个事件都带 `sequence_number`，`event:` 与响应体的 `type`
一致；以 `data: [DONE]` 结束。失败时依次发出 `error`、`response.failed`，然后
`[DONE]`。详见[流式](/zh/api/streaming#openai-responses-api)。

## 输入 Token 估算

`POST /v1/responses/input_tokens` 估算一个请求的 prompt 大小，不会调用模型。

```bash
curl https://your-proxy/v1/responses/input_tokens \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-5", "input": "Summarize the release notes."}'
```

```json
{"object": "response.input_tokens", "input_tokens": 12}
```

请求体只包含与计数相关的字段（`model`、`input`、`instructions`、`tools`、
`tool_choice`、`text`、`reasoning`）。如果选中的提供商提供自己的计数器（MiniMax
即是），就返回上游的计数；否则用本地 `o200k_base` 估算。

## WebSocket 传输 {#websocket-transport}

`WS /v1/responses` 用 JSON 文本帧承载与 SSE 流相同的事件。

- 发送 `{"type": "response.create", …}`，字段与 HTTP 请求相同（`type`、`stream`、
  `stream_options`、`background` 会被去掉，流式强制开启）。
- 其他任何消息类型 → `400 invalid_request`。
- 轮次是**串行**的：每条连接同时只有一个进行中的响应。
- `previous_response_id` 先在连接本地状态里解析，再到 Redis（`store=true`）里找。
  `store=false` 的续接在同一条 socket 内可用。
- 上限：每条消息 64 MiB，连接最长 60 分钟。鉴权失败以 `4401` 关闭 socket；被禁止的
  模型和上游失败则以 `403`/`500` 错误帧返回，而不是关闭码。

## 相关内容

- [流式](/zh/api/streaming#openai-responses-api) — 事件参考
- [错误与速率限制](/zh/api/errors)
