---
pageClass: api-reference kicker-images
aside: false
---

# Create image

Generate an image from a text prompt.

::: endpoint POST /v1/images/generations
JSON body. Unrecognized fields are forwarded to the provider as extension
parameters rather than rejected — see [Vendor extension fields](#vendor-extension-fields).
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

`prompt` (required), `model`, `n` (1–12), `quality`, `size`, `style`,
`response_format` (`url` default, or `b64_json`), `user`, `background`, `moderation`,
`output_format`, `output_compression`, `partial_images`, `stream`.

- `size`: `auto`, or `WIDTHxHEIGHT` where both edges ≥ 16 and divisible by 16, the
  larger edge ≤ 3840, and the aspect ratio within 1:3–3:1.
- `n`: the schema accepts 1–12 (the widest documented provider range). Providers
  apply their own caps; see [Vendor extension fields](#vendor-extension-fields).
- `gpt-image-*` models additionally receive `background`, `moderation`,
  `output_compression`, `output_format`, `partial_images`; `response_format` and
  `style` are omitted for them.

## Vendor extension fields

Fields outside the OpenAI Images schema are passed through to the provider's own
request body instead of being rejected, so upstream-specific options stay
reachable without `extra_body` (which the OpenAI SDK offers for chat, not images).

`qwen` / `qwen-intl` (DashScope) accept:

- `image` — a public image URL, a base64 data URL, or an array of either. Supplying
  it turns the call into image-to-image; omitting it is text-to-image.
- `negative_prompt`, `seed`, `prompt_extend`, `prompt_extend_mode`, `enable_thinking`,
  `watermark` — forwarded into DashScope's `parameters` block.
- On `/v1/images/edits`: `color_palette`, `bbox_list`, `enable_sequential`,
  `thinking_mode`.

`qwen`/`qwen-intl` per-model caps (violations return **400** `invalid_request_error`):
`n` is 1–6 for `qwen-image-*` and 1–4 for `wan*` (1–12 with `enable_sequential`).

Strictness is configurable: with `server_params.unknown_fields_policy: error` the
proxy rejects unknown fields instead of forwarding them (per-provider
`EXEMPT_EXTRA_KEYS` still keeps documented upstream fields working).

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
image requests — see the [capability matrix](../endpoints.md#capability-matrix) and
[Vendor extension fields](#vendor-extension-fields).

## Related

- [Edit image](edit.md)
- [Errors & Rate Limits](../errors.md)
