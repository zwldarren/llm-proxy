---
pageClass: api-reference kicker-audio
aside: false
---

# Create speech

Text-to-speech. Returns raw audio bytes (non-streaming) or a byte stream
(`stream: true`).

::: endpoint POST /v1/audio/speech
The endpoint is excluded from the non-streaming keepalive because the body is binary.
:::

```bash [Request]
curl http://localhost:8080/v1/audio/speech \
  -H "Authorization: Bearer $LLM_PROXY_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "tts-model", "input": "Hello from the proxy!", "voice": "alloy"}' \
  --output speech.mp3
```

```python [Request — OpenAI SDK]
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-your-proxy-key")
audio = client.audio.speech.create(
    model="tts-model",
    voice="alloy",
    input="Hello from the proxy!",
)
audio.stream_to_file("speech.mp3")
```

```http [Response]
HTTP/1.1 200 OK
Content-Type: audio/mpeg

<binary audio bytes — save or play directly>
```

## Request fields

`model`, `input` (≤ 4096 chars), `voice`, `instructions`, `response_format`
(`mp3` default; `opus`, `aac`, `flac`, `wav`, `pcm`), `speed` (0.25–4.0),
`stream_format` (`sse`/`audio`), `stream`.

## Response

- Non-streaming: raw audio bytes with the upstream content type (fallback
  `audio/<format>`; `mp3` → `audio/mpeg`).
- `stream: true`: byte stream; Gemini TTS advertises PCM/WAV.

## Provider support

`openai`, the `openai-compatible` family, and `gemini` (native TTS) — see the
[capability matrix](../endpoints.md#capability-matrix).

## Related

- [Create transcription](transcription.md) · [Create translation](translation.md)
- [Errors & Rate Limits](../errors.md)
