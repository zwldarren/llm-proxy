---
pageClass: api-reference kicker-images
aside: false
---

# Create image

Generate an image from a text prompt.

::: endpoint POST /v1/images/generations
JSON body. Unlike chat, the image schemas are **strict**: an unrecognized field
returns **400** `invalid_request_error` with `Invalid value for <field>: …`.
:::

```bash [Request]
curl http://localhost:8080/v1/images/generations \
  -H "Authorization: Bearer $LLM_PROXY_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-image-1", "prompt": "A lighthouse at dusk", "size": "1024x1024"}'
```

```python [Request — OpenAI SDK]
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-your-proxy-key")
img = client.images.generate(model="gpt-image-1", prompt="A lighthouse at dusk")
print(img.data[0].url)
```

```json [Response]
{
  "created": 1718047200,
  "data": [
    {
      "url": "https://…/lighthouse.png",
      "revised_prompt": "A lighthouse at dusk, warm light"
    }
  ],
  "usage": {
    "input_tokens": 50,
    "output_tokens": 1240,
    "input_tokens_details": {"text_tokens": 50, "image_tokens": 0}
  }
}
```

## Request fields

`prompt` (required), `model`, `n` (1–10), `quality`, `size`, `style`,
`response_format` (`url` default, or `b64_json`), `user`, `background`, `moderation`,
`output_format`, `output_compression`, `partial_images`, `stream`.

- `size`: `auto`, or `WIDTHxHEIGHT` where both edges ≥ 16 and divisible by 16, the
  larger edge ≤ 3840, and the aspect ratio within 1:3–3:1.
- `gpt-image-*` models additionally receive `background`, `moderation`,
  `output_compression`, `output_format`, `partial_images`; `response_format` and
  `style` are omitted for them.

## Response

`{"created": <int>, "data":[{"url"|"b64_json", "revised_prompt"?}], …}` with optional
`usage` (`input_tokens`, `output_tokens`,
`input_tokens_details.{text_tokens,image_tokens}`).

- **There is no `model` field in image responses**, and the proxy does not convert
  between `url` and `b64_json` — you get whatever the provider produced.
- `stream: true` emits SSE events `image_generation.partial_image` (with `b64_json`,
  `partial_image_index`) and `image_generation.completed`, then `[DONE]`.
- There is no async/background image mode.

## Provider support

`openai`, the `openai-compatible` family, `gemini`, and `qwen` providers can serve
image requests — see the [capability matrix](../endpoints.md#capability-matrix).

## Related

- [Edit image](edit.md)
- [Errors & Rate Limits](../errors.md)
