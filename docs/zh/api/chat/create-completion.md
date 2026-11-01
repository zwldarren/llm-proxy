---
pageClass: api-reference kicker-chat
aside: false
---

# 创建聊天补全

最主要的 OpenAI 兼容端点。任何 OpenAI SDK、LangChain、LlamaIndex 或原生 HTTP
客户端都能直接使用。

::: endpoint POST /v1/chat/completions
同时也提供 `/chat/completions` 和 `/v1/v1/chat/completions` 两个别名，方便 base URL
会自行增删 `/v1` 前缀的客户端。
:::

```bash [请求]
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $LLM_PROXY_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

```python [请求 — OpenAI SDK]
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-your-proxy-key")
resp = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.choices[0].message.content)
```

```json [响应]
{
  "id": "chatcmpl-9f2c…",
  "object": "chat.completion",
  "created": 1718047200,
  "model": "auto",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 9,
    "total_tokens": 34
  }
}
```

## 请求字段

顶层字段的接受范围很宽松：未知字段一律不会被拒绝，其中大部分会转发给上游（`previous_response_id`、
`background`、`truncation`、`include`、`conversation`、`max_tool_calls`、`text`
这类只属于 Responses 的键，在发往 Chat Completions 上游时会被剔除）。

支持的字段：`temperature`、`top_p`、`max_tokens`、`max_completion_tokens`、`n`、
`stream`、`stream_options`、`stop`、`presence_penalty`、`frequency_penalty`、
`logit_bias`、`logprobs`、`top_logprobs`、`user`、`response_format`、`seed`、`tools`、
`tool_choice`、`parallel_tool_calls`、`thinking`、`audio`、`modalities`、
`reasoning_effort`、`prediction`、`web_search_options`、`service_tier`、`verbosity`、
`store`、`metadata`、`prompt_cache_key`、`prompt_cache_retention`、
`safety_identifier`。

需要留意的转换：

| 输入 | 行为 |
| --- | --- |
| 只传 `max_completion_tokens` | 映射为 `max_tokens`；两者都设置时，`max_tokens` 优先 |
| `response_format: {"type":"json_schema", …}` | 原样保留，并在 Chat Completions 协议上重新输出 |
| `tool_choice: {"type":"allowed_tools", …}` | 仅适用于 Responses 风格：折叠为其模式（`auto`/`required`）并过滤工具列表。Chat Completions 请求不会构造这种形式，该字典会作为嵌套请求体转发给上游 |
| `parallel_tool_calls: false` | 转发给上游，并在内部关闭并行工具调用 |
| 未知工具类型 | 校验时接受，但从上游工具列表中移除 |

OpenAI 自家上游有一处特殊情况（这个提供商类型使用 Responses 方言）：`seed`、`n`、
`logit_bias`、各种 penalty、`prediction`、`modalities`、`audio` 和
`prompt_cache_retention` 会被有意丢弃，`response_format: json_schema` 则改映射为
`text.format`。

### 消息

- 角色：`system`、`developer`（按原样保留）、`user`、`assistant`、`tool`
  （通过 `tool_call_id` 与待处理的工具调用匹配），以及原样透传的未知角色。
- 内容片段：`text`、`image_url`（远程 URL 或 `data:` base64）、`input_audio`
  （base64 + `format`）、`file`（`file_data`/`file_id`/`filename`）、`video_url`、
  `refusal`。无法识别的片段会降级为文本，而不是直接报错。
- assistant 轮次可以包含 `tool_calls`（函数或自定义；保留 `thought_signature`）、
  `reasoning_content`（附带 `reasoning_signature`、`reasoning_is_redacted`、
  `encrypted_content`）、`reasoning_segments`、`refusal` 和 `audio`。
- 输入时，推理内容从 `reasoning_content` 或 OpenRouter/NanoGPT 的 `reasoning`
  写法读取（两者都存在时以 `reasoning_content` 为准）。被遮蔽的推理内容从
  `reasoning_is_redacted` + `encrypted_content` 读取，不透明载荷本身在后者里，
  `reasoning_content` 只是 `[redacted]` 显示占位符。
- 交错的 assistant 轮次（`thinking → tool_call → thinking`）无法用单个
  `reasoning_content` 字段表达。这时代理会输出 `reasoning_segments` 数组，每项形如
  `{type, text?, signature?, encrypted_content?, after_tool_calls?}`（被遮蔽时为
  `{type: "redacted", data}`），其中 `after_tool_calls` 表示其前面有多少个工具调用。
  把它原样回传即可保持工具调用的位置；丢弃这个未知字段的客户端会回退到拼接后的
  `reasoning_content`。

## 响应

`{"id", "object":"chat.completion", "created", "model", "choices":[…], "usage", …}`

- `model` 回显的是**客户端请求的别名**（`auto` 或你定义的模型名），绝不是解析后的
  上游模型。
- 每个 choice：`index`、`message`、`finish_reason`（`stop`、`length`、`tool_calls`、
  `content_filter`、`context_length`）。
- message：`content`（文本拼接结果；为空或只有 refusal 时为 `null`）、`arguments`
  为 JSON 字符串的 `tool_calls`，以及 `reasoning_content`——代理输出的是
  **`reasoning_content`，永远不是 `reasoning`**——还有 `reasoning_signature`、
  `encrypted_content`（被遮蔽推理或桥接提供商签名的封装载荷）、
  `reasoning_segments`（仅交错轮次，见上文）、`refusal`、`audio`，适用时还有
  `annotations`。生成的图像以 Markdown 形式出现在 `content` 里。
- `usage`：`prompt_tokens`、`completion_tokens`、`total_tokens`，以及明细
  （`prompt_tokens_details.cached_tokens`/`audio_tokens`、
  `completion_tokens_details.reasoning_tokens`/`audio_tokens`/…）。`n > 1` 时产生
  `n` 个 choice；如果提供商返回的不同输出不足，最后一个会被复制，并在日志里留下
  警告。
- 提供商提供时，`logprobs` 与 `system_fingerprint`/`service_tier` 会透传。

上游的限流响应头（`x-ratelimit-*`、`retry-after`、`openai-processing-ms` 等）会随
响应一并转发，代理自己的 `X-Request-Id` 也在其中。

## 流式

`stream: true` 会返回 `chat.completion.chunk` 帧，然后是一个只含用量的分块，最后是
`data: [DONE]`。只要上游报告了用量，就会发出这个用量分块——转换后的上游请求一律
要求 `include_usage: true`，与客户端的 `stream_options` 无关。流中途出错时先发一个
错误帧，再发 `[DONE]`。详见[流式](/zh/api/streaming)。

## 校验错误

存在两种错误结构，与上游行为对应：

- 由端点模型校验的请求体（chat、embeddings、speech）返回 FastAPI 默认的 **422**
  `{"detail": [...]}`。
- 手工解析的请求体（multipart 音频、图像）返回 **400**，响应体是 OpenAI 风格的
  `{"error": {…}}`。

## 相关内容

- [工具、推理与联网搜索](/zh/api/tools)
- [流式](/zh/api/streaming) · [错误与速率限制](/zh/api/errors)
- [虚拟模型与路由](/zh/api/routing)
