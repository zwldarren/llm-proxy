---
pageClass: api-reference kicker-audio
aside: false
---

# 生成语音

文本转语音。返回原始音频字节（非流式）或字节流（`stream: true`）。

::: endpoint POST /v1/audio/speech
该端点不参与非流式的保活心跳，因为响应体是二进制。
:::

```bash [请求]
curl http://localhost:8080/v1/audio/speech \
  -H "Authorization: Bearer $LLM_PROXY_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "tts-model", "input": "Hello from the proxy!", "voice": "alloy"}' \
  --output speech.mp3
```

```python [请求 — OpenAI SDK]
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-your-proxy-key")
audio = client.audio.speech.create(
    model="tts-model",
    voice="alloy",
    input="Hello from the proxy!",
)
audio.stream_to_file("speech.mp3")
```

```http [响应]
HTTP/1.1 200 OK
Content-Type: audio/mpeg

<binary audio bytes — save or play directly>
```

## 请求字段

`model`、`input`（≤ 4096 字符）、`voice`、`instructions`、`response_format`
（默认 `mp3`；还可取 `opus`、`aac`、`flac`、`wav`、`pcm`）、`speed`（0.25–4.0）、
`stream_format`（`sse`/`audio`）、`stream`。

## 响应

- 非流式：原始音频字节，content type 沿用上游的（取不到时用 `audio/<format>`；
  `mp3` → `audio/mpeg`）。
- `stream: true`：字节流；Gemini TTS 对外声明的是 PCM/WAV。

## 提供商支持

`openai`、`openai-compatible` 家族（`qwen`/`qwen-intl` 除外，它们没有上游语音端点）、
`gemini`（原生 TTS）以及 `ollama`（v0.40.8+，原生 `/v1/audio/speech`）——见
[能力矩阵](/zh/api/endpoints#capability-matrix)。

Ollama 只提供 WAV 或裸 PCM。请求其他 `response_format` 时会按 WAV 返回（上游的 content
type 也是这么写的）；`instructions`、不为 1 的 `speed`，以及 `audio` 之外的
`stream_format` 都会被 Ollama 以 `400` 拒绝。

## 相关

- [语音转写](/zh/api/audio/transcription) · [语音翻译](/zh/api/audio/translation)
- [错误与速率限制](/zh/api/errors)
