---
pageClass: api-reference kicker-images
aside: false
---

# 编辑图像

用文字提示编辑（或向外扩展）已有图像——可以走 JSON 请求体引用已上传的文件，也可以走
multipart 直接上传图片文件。

::: endpoint POST /v1/images/edits
JSON 或 multipart。无法识别的字段不会报错，而是作为扩展参数转发给提供商——见
[厂商扩展字段](/zh/api/images/create#vendor-extension-fields)。
:::

```bash [请求 — multipart]
curl http://localhost:8080/v1/images/edits \
  -H "Authorization: Bearer $LLM_PROXY_KEY" \
  -F image=@lighthouse.png \
  -F mask=@mask.png \
  -F model=gpt-image-1 \
  -F prompt="Replace the sky with a starry night" \
  -F size=1024x1024
```

```python [请求 — OpenAI SDK]
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

```json [响应]
{
  "created": 1718047200,
  "data": [
    {"url": "https://…/lighthouse-edited.png"}
  ]
}
```

## JSON 请求体

`images`（1–16 条，每条带 `file_id` 或 `image_url`），可选的 `mask`
（`file_id`/`image_url`），以及 `prompt`、`model`、`n`、`size`、`quality`、
`response_format`、`user`、`background`、`input_fidelity`、`moderation`、
`output_compression`、`output_format`、`partial_images`、`stream`。未知字段作为扩展
参数进入提供商（见[厂商扩展字段](/zh/api/images/create#vendor-extension-fields)）；对
`qwen`/`qwen-intl` 来说，这些扩展字段包括 `watermark`、`seed`、`bbox_list`、
`color_palette` 和 `enable_sequential`。

## Multipart

文件字段 `image` 和 `image[]`（最多 16 个），外加一个 `mask` 文件；其他字段以表单字符串
提交。每张图片**必须**是上传的文件，否则请求会以
`Multipart image edits require every image to be an uploaded file` 失败。

## 响应

结构与[生成图像](/zh/api/images/create)相同：`{"created", "data":[{"url"|"b64_json"}], …}`，可能带
`usage`。流式编辑会发出 `image_edit.partial_image` / `image_edit.completed` SSE 事件。

## 提供商支持

`openai`、`openai-compatible` 家族、`gemini` 和 `qwen` 类型的提供商——见
[能力矩阵](/zh/api/endpoints#capability-matrix)。

## 相关

- [生成图像](/zh/api/images/create)
- [错误与速率限制](/zh/api/errors)
