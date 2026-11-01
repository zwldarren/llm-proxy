---
pageClass: api-reference kicker-audio
aside: false
---

# 语音转写

对上传的音频文件做语音转文本。使用 multipart 表单数据。

::: endpoint POST /v1/audio/transcriptions
`multipart/form-data`。缺少 `model`/`file`，或 `temperature` 不是数字 → 400，返回
OpenAI 结构的错误。
:::

```bash [请求]
curl http://localhost:8080/v1/audio/transcriptions \
  -H "Authorization: Bearer $LLM_PROXY_KEY" \
  -F file=@recording.mp3 \
  -F model=whisper-model \
  -F response_format=verbose_json \
  -F timestamp_granularities[]=word
```

```python [请求 — OpenAI SDK]
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-your-proxy-key")
text = client.audio.transcriptions.create(
    model="whisper-model",
    file=open("recording.mp3", "rb"),
)
print(text.text)
```

```json [响应]
{
  "task": "transcribe",
  "language": "en",
  "duration": 12.5,
  "text": "Hello from the proxy!",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 2.1,
      "text": " Hello from the proxy!",
      "words": [
        {"word": " Hello", "start": 0.0, "end": 0.4},
        {"word": " proxy!", "start": 1.7, "end": 2.1}
      ]
    }
  ]
}
```

## 请求字段

`file`（必填）、`model`（必填）、`language`、`prompt`、`response_format`（默认
`json`）、`temperature`、`stream`、多值字段 `timestamp_granularities[]` 和
`include[]`。其他表单字段会原样转发。

## 响应格式

- `json` → `{"text": …}`（默认）
- `verbose_json` → 额外带上 `task`、`language`、`duration`、`segments`、`words`
- `text` / `srt` / `vtt` → `text/plain`
- `diarized_json` → `duration`、`segments`、`task`

上游报告了 `usage` 时会一并返回（Token 数或音频时长）。`stream: true` 返回 SSE；代理会
自动在上游请求里补上 `include[]=usage`。

## 提供商支持

`openai`、`openai-compatible` 家族（`qwen`/`qwen-intl` 除外——它们的 OpenAI 兼容 ASR
入口是 `/chat/completions` 加 `input_audio`）、`gemini`（原生 STT）和 `openrouter`
（STT）——见[能力矩阵](/zh/api/endpoints#capability-matrix)。

## 相关

- [生成语音](/zh/api/audio/speech) · [语音翻译](/zh/api/audio/translation)
- [错误与速率限制](/zh/api/errors)
