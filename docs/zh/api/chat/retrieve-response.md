---
pageClass: api-reference kicker-chat
aside: false
---

# 检索响应

读取用 `store: true`（默认值）创建并存储的响应。存储由 Redis 支撑，按 API 密钥分区，
**24 小时**后过期——已过期或未知的 id 返回 **404**。

::: endpoint GET /v1/responses/{response_id}
需要 API 密钥鉴权，只有控制台 JWT 不够。未启用 Redis 时返回 **503
`redis_not_available`**。
:::

```bash [请求]
curl http://localhost:8080/v1/responses/resp_9f2c… \
  -H "Authorization: Bearer $LLM_PROXY_KEY"
```

```python [请求 — OpenAI SDK]
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-your-proxy-key")
resp = client.responses.retrieve("resp_9f2c…")
print(resp.status)
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
    "total_tokens": 34
  }
}
```

## 说明

- 存储的表示中会去掉内部的 `input`。
- 后台响应在完成时会被最终结果覆盖：轮询这个端点，直到 `status` 不再是 `queued` /
  `in_progress`。
- `store: false` 的响应（以及未带 API 密钥的流式响应）不会持久化，在这里返回 404。

## 相关内容

- [创建响应](/zh/api/chat/create-response) — `store`、后台模式
- [删除响应](/zh/api/chat/delete-response) · [取消响应](/zh/api/chat/cancel-response) ·
  [列出输入条目](/zh/api/chat/list-input-items)
