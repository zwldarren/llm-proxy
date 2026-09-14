---
pageClass: api-reference kicker-embeddings
aside: false
---

# Create embeddings

Text embedding vectors, OpenAI-compatible.

::: endpoint POST /v1/embeddings
:::

```bash [Request]
curl http://localhost:8080/v1/embeddings \
  -H "Authorization: Bearer $LLM_PROXY_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "text-embedding-model", "input": ["hello", "world"]}'
```

```python [Request — OpenAI SDK]
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-your-proxy-key")
emb = client.embeddings.create(model="text-embedding-model", input=["hello", "world"])
print(len(emb.data[0].embedding))
```

```json [Response]
{
  "object": "list",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [0.012, -0.031, "…"]},
    {"object": "embedding", "index": 1, "embedding": [0.008, 0.044, "…"]}
  ],
  "model": "text-embedding-model",
  "usage": {"prompt_tokens": 4, "total_tokens": 4}
}
```

## Request fields

`model` (required), `input` (string, list of strings, or token-id forms),
`encoding_format` (`float` default, or `base64`), `dimensions`, `user`.

- Upstream, only `model`, `input`, `dimensions`, and `encoding_format` are forwarded
  (the `user` field is parsed but not forwarded by the default builder).
- Gemini sends multi-item input as a single `batchEmbedContents` call, falling back to
  per-item `embedContent` calls only on HTTP 400/401/403/404/405.

## Response

`{"object":"list","data":[{"object":"embedding","embedding":…,"index":…}],
"model": <alias>}` plus `usage` when the provider reports it. `base64` passes
through as returned. No streaming variant.

## Provider support

`openai`, the `openai-compatible` family (including DeepSeek, Kimi, MiniMax,
Moonshot, Qwen, xAI, vLLM, SGLang, Chutes, Mistral, NanoGPT, OpenRouter), `gemini`,
and `ollama` — see the [capability matrix](endpoints.md#capability-matrix).

## Related

- [Errors & Rate Limits](errors.md)
