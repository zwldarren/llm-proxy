---
pageClass: api-reference kicker-rerank
aside: false
---

# 重排文档

按查询语句给文档列表打分，返回最相关的文档，得分高的排前面。兼容 Cohere。

::: endpoint POST /v1/rerank
:::

`POST /v1/rerank` 是正式路由。`/v2/rerank` 作为别名由同一个处理器响应，因此 Cohere 官方
SDK（`ClientV2.rerank`，默认打这个路径）无需改动即可使用。

```bash [请求]
curl http://localhost:8080/v1/rerank \
  -H "Authorization: Bearer $LLM_PROXY_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cohere/rerank-v3.5",
    "query": "What is the capital of France?",
    "documents": [
      "Paris is the capital of France.",
      "Berlin is the capital of Germany."
    ],
    "top_n": 1
  }'
```

```python [请求 — Cohere SDK]
import cohere

# ``ClientV2.rerank`` 请求 ``/v2/rerank``；代理把这个路径作为正式路由
# ``/v1/rerank`` 的别名响应，所以 SDK 无需改动即可使用。
client = cohere.ClientV2(api_key="sk-your-proxy-key", base_url="http://localhost:8080")
res = client.v2.rerank(
    model="rerank-v3.5",
    query="What is the capital of France?",
    documents=[
        "Paris is the capital of France.",
        "Berlin is the capital of Germany.",
    ],
    top_n=1,
)
print(res.results[0].index, res.results[0].relevance_score)
```

```json [响应]
{
  "id": "gen-rerank-1234567890-abc",
  "model": "cohere/rerank-v3.5",
  "provider": "Cohere",
  "results": [
    {
      "index": 0,
      "relevance_score": 0.98,
      "document": {"text": "Paris is the capital of France."}
    }
  ],
  "usage": {"total_tokens": 150, "search_units": 1, "cost": 0.001}
}
```

## 请求字段

`model`（必填）、`query`（必填）、`documents`（必填，且不能为空）。

- `documents` 的每一项可以是纯字符串，也可以是带 `text` 和／或 `image`（远程 URL 或
  base64 data URI）的对象，供多模态重排模型使用。两者都没有的对象会被以 `422` 拒绝。
- `top_n`（`≥ 1`）限制返回多少条文档。取值超过文档总数时，代理侧会把它压到文档总数：
  Cohere 只说明了省略 `top_n` 会返回全部文档，压缩上限让多要的客户端继续可用，而不必
  依赖一条没有写明的上游行为。省略时沿用上游自己的默认值。
- `return_documents`（默认 `true`）：设为 `false` 时，每条结果都不再带 `document`
  字段，只留索引和分数。这一裁剪由代理完成，不会转发给上游。
- `user` 会被接受，并转发给在该端点上对此有文档说明的提供商（OpenRouter）；没有说明的
  提供商（Cohere）收不到它。

OpenRouter 还文档化了 `provider`（它的路由块）、`session_id` 和 `trace`。这里同样接受，
并且只转发给对它们有文档说明的提供商。

Cohere 的 v2 专属选项会被接受，并转发给 `cohere` 类型的提供商：`max_tokens_per_doc`
（截断过长的文档）、`priority`（接纳顺序）和 `relevant_spans`
（`{mode: enabled|disabled, threshold?, max_spans?}`，Rerank 5 及更新版本——打开后，
驱动每条得分的片段会通过 `results[].relevant_spans` 返回）。

结构化文档是开放的：代理不认识但上游有文档说明的键会被透传，而不是被拒绝。

## 响应

`results` 里带 `index`（在请求 `documents` 列表中的位置）、`relevance_score`（越大越
相关），以及上游回显的 `document`（除非被抑制）。请求启用了 `relevant_spans` 且上游
有报告时，`relevant_spans` 会原样出现在结果里——就是上游自己的
`{start, end, score}` 对象。上游报告了 `id` 时响应里才有 `id`；由路由器（OpenRouter）
指明应答提供商时才有 `provider`。

上游返回 `usage` 时才给出 `usage`，各提供商用同一套字段：`total_tokens`、
`search_units`，以及提供商报告时的计费 `cost`。OpenRouter 本身就用这些键；Cohere 的
`meta.tokens` 和 `meta.billed_units` 会被改写成它们，而提供商在 usage 块里加的其他键
留在上游，不会传给客户端。

没有流式形态：重排以单个 JSON 响应体返回。提供商报告的 `cost` 优先于代理自己的价格估算
用于计费。

## 提供商支持

`cohere`（原生 `/v2/rerank`，例如 `rerank-v3.5`、`rerank-v5.0-pro`）、`openrouter`
（它的 `/api/v1/rerank` 路由，例如 `cohere/rerank-v3.5`、`cohere/rerank-4-fast`、
`cohere/rerank-4-pro`），以及 `openai-compatible` 家族——它们的 `/rerank` 路由由提供商的
base URL（或其 `endpoint_base_urls` 里的 `rerank` 项）解析而来，例如 vLLM、SGLang、
Jina 或自建的 Cohere 兼容服务。原生 Cohere 不接受只有图片的文档（v2 只接受文本文档）。
见[能力矩阵](/zh/api/endpoints#capability-matrix)。

请求指定的模型必须带 `supports_rerank` 标志；没有该标志的模型会在调用任何提供商之前
以 `400 invalid_request_error` 拒绝。

## 相关

- [错误与速率限制](/zh/api/errors)
