# One bounded download pass for client-supplied media

## Status

Accepted (2026-10).

## Context

Five paths turn a URL supplied by a client into bytes the proxy then forwards:

| Path | Where | What it fetches |
| --- | --- | --- |
| chat image URLs | `providers/ollama/adapter.py` | a message's `image_url` |
| chat document URLs | `serialization/openai/url_inline.py` | a `file_url` for providers that take base64 only |
| Gemini content parts | `providers/gemini/adapter.py` (×2) | remote `file_data.file_uri` / `input.items[].uri` |
| Decisions images | `providers/capabilities/decisions.py` | a bridged `input_image` URL no System One upstream fetches (ADR-0045) |

Each was a small `asyncio.gather` over `download_image_as_base64`. The helper
carries the per-URL protections — scheme and credential checks, DNS-resolved
SSRF validation of the URL *and every redirect hop*, a 20 MiB streamed cap
(`HTTP_MAX_DOWNLOAD_BYTES`) — and none of the ones that only exist across URLs:

- **Unbounded fan-out.** A request naming 128 image URLs started 128 fetches at
  once, against a pool sized for the whole process (`HTTP_MAX_CONNECTIONS`).
- **Unbounded retention.** Each download is capped, but the sum was not: 128
  URLs each serving 20 MiB made the proxy hold ≈2.5 GiB of base64, plus the
  conversion copies, for one request. A Decisions request may name that many
  images; a chat turn has no count limit at all.

## Decision

**One object per inlining pass owns the request's downloads, and every path uses
it.**

```python
from llm_proxy.http.media import RequestDownloads

outcomes = await RequestDownloads().gather(
    client, urls
)  # {url: (data_url, mime) | None | Exception}
```

1. **`RequestDownloads` (`http/media.py`) is the chokepoint.** Five call sites
   collapsed into it; no adapter loops over URLs on its own any more. It bounds
   concurrency (`DEFAULT_DOWNLOAD_CONCURRENCY = 4`), charges every inlined
   payload to a request-wide total (`HTTP_MAX_REQUEST_DOWNLOAD_BYTES`, 64 MiB by
   default — the same figure the proxy accepts in a request body), and passes
   `HTTP_MAX_DOWNLOAD_BYTES` down as the per-URL cap instead of letting callers
   re-derive it. Peak memory per pass is therefore the total plus at most
   `concurrency × per-download`.

2. **The total is measured on the inlined payload**, not on bytes off the wire.
   The base64 string is what stays in the process (in the bridge's replacement
   map, in an `ImageSource`, in a `file_data`), so the figure that bounds memory
   is the one after encoding — 4/3 of the raw size. The per-URL cap stays on raw
   bytes, because that is the number the vision APIs publish.

3. **Failures are outcomes, not exceptions.** `gather` returns one entry per
   distinct URL (`None` for an empty body, the exception otherwise), so each
   caller keeps its own rule: a bridged Decisions call fails with a `400` naming
   the URL, because dropping an image answers a question about evidence the
   model never saw; chat media drops that one part and lets the request through.

4. **Duplicate URLs are fetched once.** Deduplication moved inside the shared
   object, so a URL repeated across a request's items or questions costs one
   download and one charge.

5. **The budget is per inlining pass, not per HTTP request.** Threading one
   object through `InternalRequest` would put a collaborator with a lock in the
   model layer for a bound that already holds in practice: the passes belong to
   different protocols (chat documents, `generateContent` parts, Interactions
   input, the Decisions bridge), and no request runs two of them.

## Consequences

- A request can no longer make the proxy fetch an unbounded number of URLs at
  once, nor retain an unbounded amount of media. Both numbers are configuration,
  with defaults chosen from the upstream limits rather than from memory
  arithmetic.
- `download_image_as_base64` remains the per-URL primitive; callers that need one
  file (not a request's media) still call it directly, and its tests pin the SSRF
  and size behaviour without a budget in the way.
- Tests patch `llm_proxy.http.media.download_image_as_base64` — one seam for all
  the call sites, instead of a patch target per adapter.
- A caller that forgets to use the budget silently loses both bounds. The name is
  in every adapter's import list, and `docs/guides/security.md` documents the two
  `HTTP_*` settings as the contract.

## Alternatives considered

- **Per-call `downloads=` parameter on `download_image_as_base64`.** Keeps the
  existing patch targets but hides the collaborator in a helper whose job is one
  file, and leaves the `gather` loops duplicated.
- **A byte-semaphore that reserves a download's whole cap.** Bounds peak more
  tightly (total exactly, instead of total + in-flight), at the cost of a
  condition variable and a reservation protocol in a module that otherwise has
  none. The overshoot is bounded by two configured numbers, so the simpler
  post-charge was kept.
- **Charging raw bytes against the total.** Makes the two settings share a unit,
  but understates what the process actually holds by a third — the direction that
  matters when the point is to bound memory.
- **A count cap ("at most N images").** Easier to explain, but N images of 20 MiB
  and N images of 20 KiB are not the same risk.
