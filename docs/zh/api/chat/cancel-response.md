---
pageClass: api-reference kicker-chat
aside: false
---

# 取消响应

取消一个仍处于 `queued` 或 `in_progress` 的后台响应（`background: true`）。之后用
[检索响应](/zh/api/chat/retrieve-response) 轮询，会看到 `status: "cancelled"`。

::: endpoint POST /v1/responses/{id}/cancel
对已经取消的响应是幂等的。未知或已过期的 id → **404**；已知响应但状态不是
`queued`/`in_progress` → **409**。需要 API 密钥鉴权；没有 Redis 时返回 **503
`redis_not_available`**。
:::

```bash [请求]
curl -X POST http://localhost:8080/v1/responses/resp_9f2c…/cancel \
  -H "Authorization: Bearer $LLM_PROXY_KEY"
```

```python [请求 — OpenAI SDK]
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-your-proxy-key")
resp = client.responses.cancel("resp_9f2c…")
print(resp.status)
```

```json [响应]
{
  "id": "resp_9f2c…",
  "object": "response",
  "status": "cancelled",
  "model": "auto",
  "output": []
}
```

## 行为

- 未知或已过期的 id → **404**（存储的响应在 24 小时后过期）。
- `queued` / `in_progress` 的后台响应会翻转为 `cancelled`。
- 已经是 `cancelled` → 成功（幂等）。
- 其他任何状态 → **409** 冲突。

## 相关内容

- [创建响应](/zh/api/chat/create-response) — 后台模式
- [检索响应](/zh/api/chat/retrieve-response)
