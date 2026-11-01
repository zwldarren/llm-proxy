---
pageClass: api-reference kicker-chat
aside: false
---

# 压缩响应

在对话超出模型的上下文窗口之前先做压缩：把此前的轮次（前提是提供了
`previous_response_id` 且能在存储中解析到）与新的 `input` 一起打包成单个压缩条目，
再次发回时会透明地还原。

::: endpoint POST /v1/responses/compact
:::

```bash [请求]
curl http://localhost:8080/v1/responses/compact \
  -H "Authorization: Bearer $LLM_PROXY_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "previous_response_id": "resp_9f2c…",
    "input": [
      {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "Summarize what we discussed."}]
      }
    ]
  }'
```

省略 `previous_response_id` 时只打包新的 `input`。只有设置了 `previous_response_id`
且能在存储中解析到时，才会包含此前的轮次。

```json [响应]
{
  "id": "compaction_7a3e…",
  "object": "response.compaction",
  "output": [
    {
      "type": "compaction",
      "id": "comp_7a3e…",
      "encrypted_content": "llm-proxy-compaction:v1:{…}",
      "created_by": "llm-proxy"
    }
  ],
  "created_at": 1718047200,
  "usage": {
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
    "input_tokens_details": {"cached_tokens": 0},
    "output_tokens_details": {"reasoning_tokens": 0}
  }
}
```

如果由原生 Responses 上游处理请求，则原样返回上游自己的压缩响应体。

## 请求参数

| 参数 | 说明 |
| --- | --- |
| `model` | 必填；模型 ID |
| `input` | 必填；字符串或条目列表——要压缩的新指令 |
| `previous_response_id` | 可选；把已存储响应的 `input` + `output` 一起纳入打包 |
| `instructions`、`tools`、`parallel_tool_calls`、`reasoning`、`service_tier`、`text`、`prompt_cache_key`、`prompt_cache_options`、`prompt_cache_retention` | 为兼容 Codex 客户端而接受，但会被忽略——压缩在本地完成 |

错误：

- 400 `previous_response_not_found` — 设置了 `previous_response_id`，但存储里没有。
- 503 `redis_not_available` — 使用了 `previous_response_id` 但没有 Redis。

## 工作原理

- 如果存在原生 Responses 上游，且引用的响应**不在**本地存储中，请求体会转发给上游
  做压缩。
- 否则代理在本地把已存储的轮次（前提是 `previous_response_id` 能解析到）与新的
  `input` 打包成一个 `compaction` 条目
  （`encrypted_content: "llm-proxy-compaction:v1:…"`），它作为输入发回时会还原。
- 外来的压缩数据块（例如 Codex 的）保持不透明：原样透传，不做修改。

把返回的压缩条目放进下一次 [创建响应](/zh/api/chat/create-response) 调用的 `input` 里一起发送，
就能以更小的占用继续对话。

## 相关内容

- [创建响应](/zh/api/chat/create-response)
- [列出输入条目](/zh/api/chat/list-input-items)
