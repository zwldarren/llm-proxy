---
pageClass: api-reference kicker-images
aside: false
---

# Edit image

Edit (or extend) an existing image with a text prompt — JSON body referencing
uploaded files, or multipart with raw image files.

::: endpoint POST /v1/images/edits
JSON or multipart. Strict schema: unknown fields return **400** `invalid_request_error`.
:::

```bash [Request — multipart]
curl http://localhost:8080/v1/images/edits \
  -H "Authorization: Bearer $LLM_PROXY_KEY" \
  -F image=@lighthouse.png \
  -F mask=@mask.png \
  -F model=gpt-image-1 \
  -F prompt="Replace the sky with a starry night" \
  -F size=1024x1024
```

```python [Request — OpenAI SDK]
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-your-proxy-key")
img = client.images.edit(
    model="gpt-image-1",
    image=open("lighthouse.png", "rb"),
    mask=open("mask.png", "rb"),
    prompt="Replace the sky with a starry night",
)
print(img.data[0].url)
```

```json [Response]
{
  "created": 1718047200,
  "data": [
    {"url": "https://…/lighthouse-edited.png"}
  ]
}
```

## JSON body

`images` (1–16 entries with `file_id` or `image_url`), optional `mask`
(`file_id`/`image_url`), plus `prompt`, `model`, `n`, `size`, `quality`,
`response_format`, `user`, `background`, `input_fidelity`, `moderation`,
`output_compression`, `output_format`, `partial_images`, `stream`.

## Multipart

File parts `image` and `image[]` (up to 16) plus a `mask` file; other fields as form
strings. Every image **must** be an uploaded file, or the request fails with
`Multipart image edits require every image to be an uploaded file`.

## Response

Same shape as [Create image](create.md): `{"created", "data":[{"url"|"b64_json"}], …}`
with optional `usage`. Streamed edits emit `image_edit.partial_image` /
`image_edit.completed` SSE events.

## Provider support

`openai`, the `openai-compatible` family, `gemini`, and `qwen` providers — see the
[capability matrix](../endpoints.md#capability-matrix).

## Related

- [Create image](create.md)
- [Errors & Rate Limits](../errors.md)
