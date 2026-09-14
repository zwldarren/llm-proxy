---
pageClass: api-reference kicker-audio
aside: false
---

# Create translation

Translate an uploaded audio file into English. Multipart form data.

::: endpoint POST /v1/audio/translations
`multipart/form-data`. **Streaming is not supported** — the pipeline always runs
non-streaming even if a `stream` field is sent.
:::

```bash [Request]
curl http://localhost:8080/v1/audio/translations \
  -H "Authorization: Bearer $LLM_PROXY_KEY" \
  -F file=@recording-fr.mp3 \
  -F model=whisper-model \
  -F response_format=json
```

```python [Request — OpenAI SDK]
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-your-proxy-key")
text = client.audio.translations.create(
    model="whisper-model",
    file=open("recording-fr.mp3", "rb"),
)
print(text.text)
```

```json [Response]
{"text": "Hello from the proxy!"}
```

## Request fields

`file` (required), `model` (required), `prompt`, `response_format`
(`json` default), `temperature`. No `language` and no `timestamp_granularities`
fields.

## Response formats

- `json` → `{"text"}` (default)
- `text` / `srt` / `vtt` → plain text
- `verbose_json` → adds `language`, `duration`, `segments`

## Provider support

`openai` and the `openai-compatible` family — see the
[capability matrix](../endpoints.md#capability-matrix).

## Related

- [Create transcription](transcription.md) · [Create speech](speech.md)
- [Errors & Rate Limits](../errors.md)
