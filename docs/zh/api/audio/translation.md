---
pageClass: api-reference kicker-audio
aside: false
---

# 语音翻译

把上传的音频文件翻译成英文。使用 multipart 表单数据。

::: endpoint POST /v1/audio/translations
`multipart/form-data`。**不支持流式**——即使请求里带了 `stream` 字段，处理流程也始终
按非流式执行。
:::

```bash [请求]
curl http://localhost:8080/v1/audio/translations \
  -H "Authorization: Bearer $LLM_PROXY_KEY" \
  -F file=@recording-fr.mp3 \
  -F model=whisper-model \
  -F response_format=json
```

```python [请求 — OpenAI SDK]
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-your-proxy-key")
text = client.audio.translations.create(
    model="whisper-model",
    file=open("recording-fr.mp3", "rb"),
)
print(text.text)
```

```json [响应]
{"text": "Hello from the proxy!"}
```

## 请求字段

`file`（必填）、`model`（必填）、`prompt`、`response_format`（默认 `json`）、
`temperature`。没有 `language`，也没有 `timestamp_granularities`。

## 响应格式

- `json` → `{"text"}`（默认）
- `text` / `srt` / `vtt` → 纯文本
- `verbose_json` → 额外带上 `language`、`duration`、`segments`

## 提供商支持

`openai` 和 `openai-compatible` 家族（`qwen`/`qwen-intl` 除外，它们直接拒收该路由）——见
[能力矩阵](/zh/api/endpoints#capability-matrix)。

## 相关

- [语音转写](/zh/api/audio/transcription) · [生成语音](/zh/api/audio/speech)
- [错误与速率限制](/zh/api/errors)
