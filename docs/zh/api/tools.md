---
pageClass: api-reference kicker-advanced
aside: false
---

# 工具、推理与网络搜索

## 工具调用

工具调用可以跨协议使用：OpenAI 客户端能调用 Anthropic 上游提供的工具，反过来也一样。

| 方面 | 行为 |
| --- | --- |
| 工具类型 | `function`、`custom`（语法/格式），以及内置的 `web_search`/`web_search_preview` 定义 |
| `tool_choice` | `auto` / `none` / `required`，或 `{"type":"function","name":…}`；Responses 风格的 `allowed_tools` 会折算成对应模式并过滤工具列表 |
| 并行调用 | 支持；`parallel_tool_calls: false` 关闭并行工具调用 |
| 未知工具类型 | 校验阶段接受，构建上游请求时丢弃 |
| 结果回传 | 照常发送 `tool` 消息（Chat）/ `tool_result` 块（Anthropic）/ `function_call_output` 条目（Responses） |
| 自定义工具调用 | 返回时重新包装为 `type: "function"`，`arguments = {"content": <input>}` |

上游支持时，代理会保留协议特有的附加信息，例如工具的 `strict` 标志、语法定义和
思考签名。

## 推理

各协议的思考控制参数都会被接受并归一化：

| 客户端 | 输入字段 |
| --- | --- |
| OpenAI Chat | `reasoning_effort`（`none`…`max`）、`thinking`（布尔值或 `{type, budget_tokens, effort, display}`），以及 Responses 形态的 `reasoning: {effort}` / `output_config: {effort}` |
| Anthropic | `thinking: {"type":"enabled","budget_tokens":N}` 或 effort 形式 |
| Responses | `reasoning: {effort, mode, context, summary}` |

几个提供商特有的行为：

- 目标是 Claude 时，`budget_tokens` 必须 ≥ 1024 且小于 `max_tokens`（否则返回 400
  `invalid_parameter`），并且开启思考期间会省略 `temperature`/`top_p`——若强制
  `tool_choice`（`any`/`tool`），思考会被关闭、这两个参数重新生效。
- 出站时，能推导出 `reasoning_effort` 就发它，否则发 `thinking` 对象。

推理内容如何返回：

| 协议 | 字段 |
| --- | --- |
| Chat Completions | `reasoning_content`（不是 `reasoning`），外加 `reasoning_signature`；被隐去的块 → `reasoning_content: "[redacted]"` + `reasoning_is_redacted: true` + `encrypted_content: <不透明数据>`；交错的多轮还会带 `reasoning_segments` |
| Anthropic | 带 `signature` 的 `thinking` 块，以及 `redacted_thinking` 块 |
| Responses | 推理输出条目，流式期间还有 `response.reasoning_summary_text.delta` 事件；Token 计数在 `output_tokens_details.reasoning_tokens` |

多轮推理的回传：把你收到的原样传回来。Chat Completions 输入侧，代理会读取
`reasoning_content`，或 OpenRouter/NanoGPT 的 `reasoning` 写法（两者同时出现时以
`reasoning_content` 为准），以及 `reasoning_signature`、`reasoning_is_redacted`、
`encrypted_content` 和 `reasoning_segments`。

::: warning 混合提供商的推理负载
`encrypted_content` 一字段多用：代理为回传而桥接 Anthropic 思考签名时，装的是签名；
其他情况下则是 OpenAI 自身的加密推理数据。代理直接从 OpenAI 上游解析出来的加密数据会
带标记，因此绝不会被当作签名回放给 Anthropic；但已经送到你客户端再回来的数据没有来源
标记——把这段历史回放给 Anthropic 可能被拒绝。尽量让特定提供商的推理留在同一个提供商
内部。
:::

## 网络搜索 {#web-search}

代理可以自己处理 `web_search` 工具调用，所以配合你自己的搜索提供商，**每个**模型都能
联网。

### 启用

1. 设置 → 常规 → 网络搜索：打开拦截并配置搜索提供商：
   - SearXNG：`url` 必填；可选的 API 密钥或 basic auth；`engines` 列表；
     `timeout`（默认 30 秒）；`max_results`（默认 10，最多 10）。超过 4000 字符的
     查询会被拒绝。
   - Ollama：`api_key` 必填；`base_url` 默认 `https://ollama.com`；`timeout`；
     `max_results`（最多 10）。
2. 保存设置。改动会热加载。

### 拦截的工作方式

- 只拦截**没有原生网络搜索**的提供商（由提供商的*原生网络搜索*标志决定）。有原生支持
  时，你的工具会原样传下去。
- 在 Chat Completions 中，`web_search_options` 字段（或 `web_search` 工具定义）会
  触发拦截；代理把该工具替换成一个名为 `web_search` 的普通函数，只接受一个 `query`
  字符串，然后拿模型的调用去执行你的搜索提供商。
- 代理带着搜索结果重新调用模型，直到模型不再搜索，再把**最终答案**返回给客户端：
  非流式路径不会暴露中间的调用杂音；多次调用的用量会累加。
- 工具定义上的 `max_uses` 按请求生效。
- 走原生透传层的请求会在该请求上关闭拦截。

拦截器还会读取工具的其他选项，但只有一部分会传给搜索提供商：

| 选项 | 行为 |
| --- | --- |
| `allowed_domains`、`blocked_domains` | SearXNG 会把结果限定在这些域名内/外；Ollama 忽略 |
| `user_location` | SearXNG：作为 `location`（城市/地区/国家）传入，`language` 由 `timezone` 推导；Ollama 忽略 |
| `search_context_size`、`external_web_access`、`return_token_budget`、`search_content_types` | 从工具定义中读取，但没有任何搜索提供商执行 |

### 客户端看到什么

- Anthropic 协议的客户端会收到原生的 `server_tool_use` + `web_search_tool_result`
  块，以及计费计数 `provider_info.server_tool_use.web_search_requests`。
- 其他协议收到结果块：`{"type":"web_search_result", "url", "title",
  "encoded_content": <base64 片段>, "encoded_index": <base64 sha256>, "page_age"?}`。
  注意这是 base64 编码，**不是** Anthropic 的加密。
- 搜索失败**不会**让 HTTP 请求失败：模型会收到带错误码的
  `web_search_tool_result_error`——
  `too_many_requests`（429/超时）、`invalid_input`（400）、`query_too_long`、
  `unavailable`（5xx）、`invalid_api_key`（搜索提供商返回 401）、
  `max_uses_exceeded`。
  同一个错误码和搜索提供商的错误消息会体现在客户端可见的
  `web_search_call.action` 上（`error_code` / `error_message`，条目的 `status` 为
  `failed`）；Anthropic 协议的客户端则在 `web_search_tool_result_error` 对象里拿到
  错误码，该对象没有消息字段。

### 可观测性

每次被拦截的搜索都会记录（日志 → **网络搜索**）：查询语句、提供商、状态、结果数量，
以及当前使用次数与上限。查询语句按原文存储，属于内容字段，因此**日志输入/输出**总
开关会连同其他请求体一起把它擦除。`provider` 列里是搜索提供商的协议名称
（`searxng` / `ollama`）。

## MCP 是另一回事

MCP 服务器不会被注入模型请求——它们以 `/servers/<name>/mcp` 的形式重新暴露给 MCP
客户端，沿用同一套 API 密钥、预算和日志。参见 [MCP 服务器](/zh/admin/mcp)。
