---
pageClass: api-reference kicker-chat
aside: false
---

# Token 计数

统计一个 Messages 请求会消耗多少 Token——选中的适配器支持时转发给上游原生的
`count_tokens`，否则用 o200k_base 分词器在本地估算。

::: endpoint POST /v1/messages/count_tokens
:::

```bash [请求]
curl http://localhost:8080/v1/messages/count_tokens \
  -H "Authorization: Bearer $LLM_PROXY_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "system": "You are a helpful assistant.",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

```python [请求 — Anthropic SDK]
from anthropic import Anthropic

client = Anthropic(base_url="http://localhost:8080", api_key="sk-your-proxy-key")
result = client.messages.count_tokens(
    model="auto",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(result.input_tokens)
```

```json [响应]
{"input_tokens": 24}
```

## 行为

- 选中的适配器支持时（原生 Anthropic 提供商），代理会把请求转发给上游原生的
  `count_tokens`。转发的请求体会过滤为 `model`、`messages`、`system`、`tools`、
  `tool_choice`、`thinking`、`cache_control`、`output_config`。
- 否则用 o200k_base 分词器在本地估算（系统文本、消息和工具 schema 都计入）。
- 鉴权失败和未知模型按原样返回（401/404）；其他上游失败则回退到本地估算，并记录
  一条警告。

## 相关内容

- [Anthropic Messages](/zh/api/chat/messages)
