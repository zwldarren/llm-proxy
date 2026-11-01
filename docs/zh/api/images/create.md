---
pageClass: api-reference kicker-images
aside: false
---

# 生成图像

根据文字提示生成图像。

::: endpoint POST /v1/images/generations
JSON 请求体。无法识别的字段不会报错，而是作为扩展参数转发给提供商——见
[厂商扩展字段](#vendor-extension-fields)。
:::

```bash [请求]
curl http://localhost:8080/v1/images/generations \
  -H "Authorization: Bearer $LLM_PROXY_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-image-1", "prompt": "A lighthouse at dusk", "size": "1024x1024"}'
```

```python [请求 — OpenAI SDK]
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-your-proxy-key")
img = client.images.generate(model="gpt-image-1", prompt="A lighthouse at dusk")
print(img.data[0].url)
```

```json [响应]
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

## 请求字段

`prompt`（必填）、`model`、`n`（1–12）、`quality`、`size`、`style`、
`response_format`（默认 `url`，或 `b64_json`）、`user`、`background`、`moderation`、
`output_format`、`output_compression`、`partial_images`、`stream`。

- `size`：`auto`，或 `宽x高`：两条边都 ≥ 16 且能被 16 整除，较长边 ≤ 3840，宽高比
  在 1:3–3:1 之间。
- `n`：schema 接受 1–12（各提供商公开的最宽范围）。实际生效的上限由提供商自己决定，
  见[厂商扩展字段](#vendor-extension-fields)。
- `gpt-image-*` 系列的模型还会收到 `background`、`moderation`、`output_compression`、
  `output_format`、`partial_images`；对它们不下发 `response_format` 和 `style`。

## 厂商扩展字段 {#vendor-extension-fields}

OpenAI Images schema 之外的字段不会被拒，而是直接进入提供商自己的请求体，因此不借助
`extra_body` 也能用上特定上游的选项（OpenAI SDK 只在 chat 上提供 `extra_body`，图像
接口没有）。

`qwen` / `qwen-intl`（DashScope）接受：

- `image`：公网图片 URL、base64 data URL，或二者的数组。带上它就是图生图，不带就是
  文生图。
- `negative_prompt`、`seed`、`prompt_extend`、`prompt_extend_mode`、`enable_thinking`、
  `watermark`——放进 DashScope 的 `parameters` 块。
- 在 `/v1/images/edits` 上：`color_palette`、`bbox_list`、`enable_sequential`、
  `thinking_mode`。

`qwen`/`qwen-intl` 按模型规定上限（超出返回 **400** `invalid_request_error`）：
`qwen-image-*` 的 `n` 为 1–6，`wan*` 为 1–4（带 `enable_sequential` 时为 1–12）。

严格程度可以配置：设 `server_params.unknown_fields_policy: error` 后，代理改为拒收未知
字段而不再转发（提供商级别的 `EXEMPT_EXTRA_KEYS` 仍会让已文档化的上游字段继续可用）。

## 响应

`{"created": <int>, "data":[{"url"|"b64_json", "revised_prompt"?}], …}`，可能附带
`usage`（`input_tokens`、`output_tokens`、
`input_tokens_details.{text_tokens,image_tokens}`）。

- 图像响应里没有 `model` 字段，代理也不会在 `url` 和 `b64_json` 之间做转换：提供商
  返回什么就是什么。
- `stream: true` 会发出 SSE 事件 `image_generation.partial_image`（含 `b64_json`、
  `partial_image_index`）和 `image_generation.completed`，最后是 `[DONE]`。
- 没有异步／后台上传模式。

## 提供商支持

`openai`、`openai-compatible` 家族、`gemini` 和 `qwen` 类型的提供商都能承接图像请求——见
[能力矩阵](/zh/api/endpoints#capability-matrix)和[厂商扩展字段](#vendor-extension-fields)。

## 相关

- [编辑图像](/zh/api/images/edit)
- [错误与速率限制](/zh/api/errors)
