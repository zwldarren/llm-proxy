---
pageClass: api-reference kicker-chat
aside: false
---

# List input items

List the input items of a stored response, newest first by default. Unknown ids
return **404**.

::: endpoint GET /v1/responses/{id}/input_items
Query parameters: `limit` (1–100, default 20), `order` (`asc`/`desc`, default `desc`),
`after` (cursor), `include`. Requires API-key authentication; **503
`redis_not_available`** without Redis.
:::

```bash [Request]
curl "http://localhost:8080/v1/responses/resp_9f2c…/input_items?limit=2&order=asc" \
  -H "Authorization: Bearer $LLM_PROXY_KEY"
```

```python [Request — OpenAI SDK]
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-your-proxy-key")
items = client.responses.input_items.list("resp_9f2c…", limit=2, order="asc")
for item in items.data:
    print(item.type)
```

```json [Response]
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

## `include` values

`include` accepts:

- `file_search_call.results`
- `web_search_call.results`
- `web_search_call.action.sources`
- `message.input_image.image_url`
- `computer_call_output.output.image_url`
- `code_interpreter_call.outputs`
- `reasoning.encrypted_content`
- `message.output_text.logprobs`

Stored items already carry whatever the client sent, so supported values return the
stored data verbatim. Unknown values → **400** `invalid_include`. An `after` id that
is not an input item of the response → **400** `invalid_cursor`.

## Pagination

Pass the last item's id as `after` to fetch the next page while `has_more` is true.

## Related

- [Create response](create-response.md) — storage model
- [Retrieve response](retrieve-response.md)
