---
pageClass: api-reference kicker-chat
aside: false
---

# Count tokens

Count the tokens a Messages request would consume — forwards to the upstream's
native `count_tokens` when the selected adapter supports it, otherwise estimates
locally with the o200k_base tokenizer.

::: endpoint POST /v1/messages/count_tokens
:::

```bash [Request]
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

```python [Request — Anthropic SDK]
from anthropic import Anthropic

client = Anthropic(base_url="http://localhost:8080", api_key="sk-your-proxy-key")
result = client.messages.count_tokens(
    model="auto",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(result.input_tokens)
```

```json [Response]
{"input_tokens": 24}
```

## Behavior

- The proxy forwards to the upstream's native `count_tokens` when the selected
  adapter supports it (native Anthropic providers). The forwarded body is filtered to
  `model`, `messages`, `system`, `tools`, `tool_choice`, `thinking`, `cache_control`,
  `output_config`.
- Otherwise it estimates locally with the o200k_base tokenizer (system text, messages,
  and tool schemas counted).
- Auth failures and unknown models are returned as-is (401/404); other upstream
  failures fall back to the local estimate with a warning.

## Related

- [Anthropic Messages](messages.md)
