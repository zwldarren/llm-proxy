# One image channel per System One upstream

## Status

Accepted (2026-10).

## Context

ADR-0019 made `/v1/decisions` and `/v1/systemone` two envelopes over the same
evaluation primitives, bridged at the provider. Its loss list had one entry that
turned out to be wrong the moment the upstreams grew image support: "System One
has no image channel except Ollama's raw-base64 `images` field."

Every System One upstream now documents an image channel, and no two are alike:

| Upstream | Channel | Shape |
| --- | --- | --- |
| `ollama` | top-level `images` | array of raw base64 strings, shared by every question |
| `openrouter` | `state` parts | `{"type": "image_url", "image_url": {"url": "data:…"}}` items in the `state` array, also read inside a question's `instructions` |
| `typesafe` | `assets` | a map of placeholder → `{type, format, data_base64}`; the upstream substitutes an image wherever the placeholder string appears in `state` or in instructions |

Two properties matter and pull in opposite directions. Only the `state_parts`
and `assets` channels keep an image where the client put it (order preserved
relative to text); `images` is a separate bag applied to every question. And
`assets` accepts PNG and JPEG only, while a Decisions client may hand over a
WebP or a hosted HTTP(S) URL that only a native Decisions endpoint would fetch.

The bridge is a pure function in `models/decisions_bridge.py` and must stay one:
it is tested without an adapter, and it has no client with which to download a
hosted URL.

## Decision

**Each adapter declares its channel; the bridge renders into it; hosted URLs are
inlined one layer up.**

1. **`SYSTEMONE_IMAGE_CHANNEL`** on the System One capability mixin
   (`SystemOneCapabilityMixin`, and the `SystemOneSelf` host protocol) names the
   channel: `images`, `state_parts`, `assets`, or unset. Adapters declare it next
   to their endpoint and their extra-field exemptions — the same place a
   provider's other wire differences already live. An adapter that documents no
   channel leaves it unset and a bridged request reaches it as text alone,
   instead of being sent a field it might reject.

2. **The bridge takes the channel as an argument.** `_ImageSink` collects the
   images of one request and returns the element to place inline: a placeholder
   for `assets`, an `image_url` part for `state_parts`, nothing for `images`
   (which collects the payload into its array) and nothing for a channel-less
   upstream (which drops it). Text-only content still renders as a string on
   every channel, so the shape only changes when there is an image to place.
   A `score` question's rubric moves into a text block appended to the
   instructions, whether those are a string or an array.

3. **Hosted images are downloaded, not forwarded.** `DecisionsOverSystemOneMixin
   .decisions()` inlines every `http(s)` image URL with the shared
   SSRF-validated downloader before converting. No System One upstream fetches
   one — OpenRouter says so explicitly — so forwarding would hand the model a URL
   it cannot read, and dropping it would answer a question about evidence the
   model never saw. A failed download is a `400` naming the URL. The native
   Decisions path is untouched: OpenAI fetches hosted URLs itself.

   An upstream that documents no image channel at all
   (`SYSTEMONE_IMAGE_CHANNEL is None`) is the one exception: the bridge would
   drop the image a moment later, so the capability neither downloads it nor
   fails the request over one it cannot carry. The text and the other content
   still reach that upstream.

4. **`assets` is also a client field.** The System One protocol schema declares
   it, and the TypeSafe adapter whitelists it for the unknown-fields policy, so a
   client holding a TypeSafe SDK request can send one directly. Any request
   carrying assets or a `box` question sends `TypeSafe-Danger-Zone-Images: true`,
   which TypeSafe requires for both; the header is added by the adapter's
   `_systemone_headers(request)` — the reason that hook takes the request.

## Consequences

- A Decisions request with images keeps them on every evaluation provider
  instead of silently losing them on the three that are not OpenAI.
- TypeSafe rejects a GIF or WebP asset, and its format enum has no way to say
  so; the bridge raises a `400` naming the media type rather than mislabeling the
  bytes as PNG. A client that needs those formats uses a provider whose channel
  accepts them.
- Ollama's channel cannot scope an image to one question: a question-level image
  joins the shared array and is visible to every question in the request. That is
  the upstream's shape, not a choice the bridge makes.
- `box` questions stay System One-only. The Decisions envelope has no
  coordinate-answering question, so `SystemOneOverDecisionsMixin` refuses one with
  a `400` naming the model instead of inventing an answer.
- Adding a fourth upstream means adding one class attribute and, if its shape is
  new, one branch in `_ImageSink.inline` — not another pass over the bridge.
