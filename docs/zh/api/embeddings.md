---
pageClass: api-reference kicker-embeddings
aside: false
---

# 创建 Embeddings

文本嵌入向量，兼容 OpenAI。

::: endpoint POST /v1/embeddings
:::

```bash [请求]
curl http://localhost:8080/v1/embeddings \
  -H "Authorization: Bearer $LLM_PROXY_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "text-embedding-model", "input": ["hello", "world"]}'
```

```python [请求 — OpenAI SDK]
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-your-proxy-key")
emb = client.embeddings.create(model="text-embedding-model", input=["hello", "world"])
print(len(emb.data[0].embedding))
```

```json [响应]
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

## 请求字段

`model`（必填）、`input`（字符串、字符串列表，或 Token id 形式）、`encoding_format`
（默认 `float`，或 `base64`）、`dimensions`、`user`。

- 发往上游的只有 `model`、`input`、`dimensions` 和 `encoding_format`（`user` 会被
  解析，但默认的请求构造器不转发它）。
- Gemini 把多项输入合成一次 `batchEmbedContents` 调用，只有在 HTTP 400/401/403/404/405
  时才退回到逐项 `embedContent`。

## 响应

`{"object":"list","data":[{"object":"embedding","embedding":…,"index":…}],
"model": <alias>}`，提供商报告时还会带上 `usage`。`base64` 按上游返回的原样透传。没有
流式形态。

## 提供商支持

`openai`、`openai-compatible` 家族（包括 DeepSeek、Kimi、MiniMax、Moonshot、Qwen、
xAI、vLLM、SGLang、Chutes、Mistral、NanoGPT、OpenRouter）、`gemini` 和 `ollama`——见
[能力矩阵](/zh/api/endpoints#capability-matrix)。

## 相关

- [错误与速率限制](/zh/api/errors)
