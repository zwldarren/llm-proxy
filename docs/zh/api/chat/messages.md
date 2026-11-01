---
pageClass: api-reference kicker-chat
aside: false
---

# Anthropic Messages

Anthropic 兼容端点，供 `anthropic` SDK、Claude Code 以及任何使用 Messages API 的
客户端调用。

::: endpoint POST /v1/messages
同时也提供 `/messages` 和 `/v1/v1/messages` 两个别名。
:::

```bash [请求]
curl http://localhost:8080/v1/messages \
  -H "Authorization: Bearer $LLM_PROXY_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

```python [请求 — Anthropic SDK]
from anthropic import Anthropic

client = Anthropic(base_url="http://localhost:8080", api_key="sk-your-proxy-key")
message = client.messages.create(
    model="auto",
    max_tokens=4096,
    thinking={"type": "enabled", "budget_tokens": 2048},
    messages=[{"role": "user", "content": "Hello!"}],
)
print(message.content[0].text)
```

```json [响应]
{
  "id": "msg_9f2c…",
  "type": "message",
  "role": "assistant",
  "model": "auto",
  "content": [
    {"type": "text", "text": "Hello! How can I help you today?"}
  ],
  "stop_reason": "end_turn",
  "usage": {
    "input_tokens": 25,
    "output_tokens": 9
  }
}
```

## 支持的请求字段

| 字段 | 说明 |
| --- | --- |
| `model` | 必填；任意已配置的名称或虚拟模型 |
| `max_tokens` | 省略时默认为 16384；允许传 `0`（用于预热缓存） |
| `messages` | `user` / `assistant`；`tool` 和 `function` 角色会降级为 `tool_result` 的 user 轮次；`developer` 消息会合并进 `system` |
| `system` | 字符串，或块列表。使用块列表时，非文本块会被丢弃，除非某个文本块带 `cache_control`/`citations`——此时列表原样转发 |
| `stop_sequences` | 原样转发 |
| `temperature`、`top_p`、`top_k` | 目标是 Claude 且开启思考时省略；当强制的 `tool_choice`（`any`/`tool`）关闭思考时恢复发送 |
| `tools`、`tool_choice` | `tool_choice.type`：`auto` / `any` / `tool` / `none`；`disable_parallel_tool_use` 可以放在顶层、`tool_choice` 内部，或通过 OpenAI 的 `parallel_tool_calls: false` 表达——最终一律输出在 API 定义的位置，即 `tool_choice` 内部 |
| `thinking` | `{"type":"enabled","budget_tokens":N}` 或 effort 形式。目标是 Claude 时，`budget_tokens` 必须 ≥ 1024 且小于 `max_tokens`，否则返回 400 `invalid_parameter` |
| `cache_control` | 支持顶层和每个内容块（`{"type":"ephemeral","ttl":…}`）；可用于 text、image、audio、file/document、tool_use、server_tool_use、tool_result 等块类型 |
| `metadata` | 转发；`user_id` 也会从 OpenAI 风格的 metadata 或 `user` 填充 |
| `container`、`inference_geo`、`service_tier`、`output_config` | 转发；旧的 `output_format` 会作为别名写入 `output_config.format`。`service_tier` 接受 Anthropic 的 `auto`/`standard_only`，也接受 Anthropic 兼容上游（MiniMax）文档中的 `standard`/`priority` 名称；当上游就是 Anthropic 本身时，后者会转换为最接近的 Anthropic 档位 |
| `context_management` | 接受并转发 |
| `betas` | 合并进 `anthropic-beta` 请求头上下文，并从请求体中移除 |
| `workspace_id`、`user_profile_id` | 在官方 API 中属于请求头信息：提升为 `anthropic-workspace-id` / `anthropic-user-profile-id` 并从请求体中移除（Messages API 会把它们当作未知顶层字段而拒绝） |

未知顶层字段会被**接受**（绝不会返回 422），并转发给上游，除非它们是已知的跨协议
附加项（`reasoning`、`previous_response_id`、`truncation`、`include`、`background`、
`max_tool_calls`、`parallel_tool_calls` 等），这些会被剔除。走
[原生透传层级](/zh/getting-started/concepts#protocol-in-protocol-out) 的请求会
原样转发，包括客户端指纹请求头。

Claude 相关细节：

- 工具定义按类型识别：`function`、`web_search_*`、`code_execution_*`、`bash_*`、
  `text_editor_*`、`memory_*`、`web_fetch_*`、`tool_search_*`；未知的工具对象原样
  透传。
- `anthropic-version` 默认为 `2023-06-01`（客户端传的值优先），`anthropic-beta`
  会透传给原生 Anthropic 上游。Claude Code 客户端（`x-app: claude-code` 或
  `claude-cli/…` user agent）缺少时会补上 `claude-code-20250219`。
- 请求被路由到**非 Anthropic** 提供商时，系统文本里内嵌的 Claude Code 计费头行会被
  剔除。

## 响应

标准 Messages 结构：`id`、`type: "message"`、`role: "assistant"`、`content`、
`model`、`stop_reason`（默认 `end_turn`）、`usage`。

- `usage.input_tokens` **不含**缓存 Token，即
  `input_tokens − cache_read_input_tokens − cache_creation_input_tokens`。
  上游报告时会出现 `cache_read_input_tokens` / `cache_creation_input_tokens`。
- 提供商提供时，附加字段会透传：`usage.cache_creation`、`usage.inference_geo`、
  `usage.server_tool_use`、`usage.output_tokens_details`、`usage.service_tier`、
  `usage.speed`、`usage.iterations`，以及顶层的 `stop_sequence`、`stop_details`、
  `container`、`diagnostics`。

双向处理的内容块类型：`text`、`image`（base64 / url / file_id）、
`document`/`file`（base64 / url / file_id，可带 citations 和 context）、
`audio`（base64/url；`file_id` 音频会降级为 `[Audio: file_id=…]` 文本占位符，因为
Anthropic API 没有音频 file_id）、`tool_use`、`tool_result`、`thinking`、
`redacted_thinking`、`server_tool_use`、`web_search_tool_result`、
`web_fetch_tool_result`、`search_result`、`container_upload`、`tool_reference`、
`mid_conv_system`，以及代码执行结果。未知块类型在 Anthropic 路径上原样回放。

当还存在其他内容时，空文本块会被移除；整个响应都为空时，则变成一个空文本块。

## 流式

只使用具名 SSE 事件，没有 `[DONE]`：`message_start`、`content_block_start` /
`content_block_delta` / `content_block_stop`、`message_delta`（停止原因 + 用量）、
`message_stop`；失败时以 `event: error` 到达。详见[流式](/zh/api/streaming)。

## 相关内容

- [Token 计数](/zh/api/chat/count-tokens)
- [工具、推理与联网搜索](/zh/api/tools)
- [流式](/zh/api/streaming) · [错误与速率限制](/zh/api/errors)
