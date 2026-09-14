---
pageClass: api-reference kicker-audio
aside: false
---

# Create transcription

Speech-to-text on an uploaded audio file. Multipart form data.

::: endpoint POST /v1/audio/transcriptions
`multipart/form-data`. Missing `model`/`file` or a non-numeric `temperature` → 400
with an OpenAI-shaped error.
:::

```bash [Request]
curl http://localhost:8080/v1/audio/transcriptions \
  -H "Authorization: Bearer $LLM_PROXY_KEY" \
  -F file=@recording.mp3 \
  -F model=whisper-model \
  -F response_format=verbose_json \
  -F timestamp_granularities[]=word
```

```python [Request — OpenAI SDK]
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-your-proxy-key")
text = client.audio.transcriptions.create(
    model="whisper-model",
    file=open("recording.mp3", "rb"),
)
print(text.text)
```

```json [Response]
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

## Request fields

`file` (required), `model` (required), `language`, `prompt`,
`response_format` (default `json`), `temperature`, `stream`, multi-value
`timestamp_granularities[]` and `include[]`. Other form fields are forwarded.

## Response formats

- `json` → `{"text": …}` (default)
- `verbose_json` → adds `task`, `language`, `duration`, `segments`, `words`
- `text` / `srt` / `vtt` → `text/plain`
- `diarized_json` → `duration`, `segments`, `task`

`usage` is included when reported (tokens or audio duration). `stream: true` returns
SSE; the proxy adds `include[]=usage` upstream automatically.

## Provider support

`openai`, the `openai-compatible` family, `gemini` (native STT), and `openrouter`
(STT) — see the [capability matrix](../endpoints.md#capability-matrix).

## Related

- [Create speech](speech.md) · [Create translation](translation.md)
- [Errors & Rate Limits](../errors.md)
