---
pageClass: api-reference kicker-chat
aside: false
---

# 列出输入条目

列出已存储响应的输入条目，默认按从新到旧排序。未知 id 返回 **404**。

::: endpoint GET /v1/responses/{id}/input_items
查询参数：`limit`（1–100，默认 20）、`order`（`asc`/`desc`，默认 `desc`）、`after`
（游标）、`include`。需要 API 密钥鉴权；没有 Redis 时返回 **503
`redis_not_available`**。
:::

```bash [请求]
curl "http://localhost:8080/v1/responses/resp_9f2c…/input_items?limit=2&order=asc" \
  -H "Authorization: Bearer $LLM_PROXY_KEY"
```

```python [请求 — OpenAI SDK]
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-your-proxy-key")
items = client.responses.input_items.list("resp_9f2c…", limit=2, order="asc")
for item in items.data:
    print(item.type)
```

```json [响应]
{
  "object": "list",
  "data": [
    {
      "id": "msg_9f2c…",
      "type": "message",
      "role": "user",
      "content": [{"type": "input_text", "text": "Hello!"}]
    },
    {
      "id": "msg_8b1d…",
      "type": "message",
      "role": "assistant",
      "status": "completed",
      "content": [
        {"type": "output_text", "text": "Hello! How can I help?", "annotations": []}
      ]
    }
  ],
  "first_id": "msg_9f2c…",
  "last_id": "msg_8b1d…",
  "has_more": false
}
```

## `include` 取值

`include` 接受：

- `file_search_call.results`
- `web_search_call.results`
- `web_search_call.action.sources`
- `message.input_image.image_url`
- `computer_call_output.output.image_url`
- `code_interpreter_call.outputs`
- `reasoning.encrypted_content`
- `message.output_text.logprobs`

存储的条目按存储时的样子返回。除非通过 `include` 显式请求，否则
`reasoning.encrypted_content` 会从已存储的推理条目中剔除——规范规定只有指定该取值时
才返回加密推理内容，而代理构造的输入可能带有客户端从未收到过的数据块。未知取值 →
**400** `invalid_include`。`after` 指定的 id 不是该响应的输入条目 → **400**
`invalid_cursor`。

## 分页

在 `has_more` 为 true 时，把上一页最后一条的 id 作为 `after` 传进去，即可取下一页。

## 相关内容

- [创建响应](/zh/api/chat/create-response) — 存储方式
- [检索响应](/zh/api/chat/retrieve-response)
