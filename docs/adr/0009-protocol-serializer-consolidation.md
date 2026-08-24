# Consolidate protocol serializers into the protocol modules

ADR-0001 moved the OpenResponses protocol (endpoint + conversion + streaming) into one deep module; ADR-0003 moved provider-dialect serializers into `serialization/<family>/` and explicitly deferred the protocol-side serializers ("left untouched, candidate for a future consolidation"). This ADR completes that consolidation.

## Problem

The same concept — a `ProtocolSerializer` implementation — lived in three places:

- `serialization/handlers/openai_chat.py` (+ `openai_audio.py`, `openai_embeddings.py`, `openai_images.py`)
- `serialization/anthropic/protocol.py`
- `protocols/openresponses/serializer.py` (the ADR-0001 outlier)

`serialization/` therefore mixed two axes: client-protocol wire conversion (`handlers/`, `streaming/`) and provider-dialect conversion (`providers/`, `<family>/`). Two further issues surfaced during the move:

- `ProtocolSerializer.get_streaming_transformer()` was an abstract method with zero callers — the framework resolves the transformer from `ProtocolEndpoint.get_streaming_transformer()` (`streaming_processor.py`). It was removed.
- `ProtocolEndpoint._detect_caller_module()` walked `inspect.stack()` only to name the re-registration site in a warning; `warnings.warn(stacklevel=2)` already points there. Removed.

## Changes

- Protocol serializers now live next to their protocol module: `protocols/<name>/serializer.py` (openai family: `serializer.py` + `audio_serializer.py` + `embeddings_serializer.py` + `images_serializer.py`), plus the openai parsing/formatting mixins and the protocol-side SSE transformers (`protocols/<name>/streaming.py`).
- `ProtocolSerializer` (ABC) moved to `protocols/serializer_base.py`; the protocol serializer registry merged into `protocols/registry.py` alongside the endpoint registry.
- `serialization/` is now provider-dialect-only (provider serializers, `content_parsers.py`, `responses_toolkit/`, shared contexts). Registration of protocol serializers is triggered by the existing protocol module imports (`protocols/<name>/handler.py` imports its serializer).
- Dead code removed: `get_streaming_transformer` (base + 6 implementations), `_detect_caller_module`/`_module`, the marker-only `AudioCapabilityMixin` was replaced by a real implementation (see below).

## Considered Options

- **Move only the serializers, leave the streaming transformers in `serialization/streaming/`**: rejected — after removing the dead `get_streaming_transformer`, the transformers' only runtime importers were protocol-side files; keeping them in `serialization/` preserved the exact naming collision the consolidation was meant to kill. They moved with the serializers.
- **Merge `ProtocolSerializer` into `protocols/base.py`**: rejected — two small files (`base.py` endpoints, `serializer_base.py` conversion) are clearer than one file mixing two interfaces.
- **Delete the `protocols/openai/serializer.py` composition shell** (23 lines combining `OpenAIParsingMixin` + `OpenAIFormattingMixin` + `ProtocolSerializer`; the shape ADR-0003 deleted for gemini/ollama): rejected — it is intentionally kept as the host of the `@register_protocol_serializer("openai")` decorator, giving the openai family's serializers a uniform one-class-per-file layout (`serializer.py`, `audio_serializer.py`, `embeddings_serializer.py`, `images_serializer.py`) with the registration site next to the composed class.

## Consequences

- `serialization/handlers/` and `serialization/streaming/` are deleted. `llm_proxy.serialization` no longer exports `get_protocol_serializer` — import it from `llm_proxy.protocols.registry`.
- `protocols/openai/parsing.py` / `formatting.py` import shared content parsers from `serialization.content_parsers` and the shared `FormatContext` — the same direction as the `responses_toolkit` precedent (ADR-0001): protocol code may use shared dialect helpers from `serialization/`, never provider *adapters*.
- `serialization/anthropic/mixin.py` (`AnthropicContentMixin`) stays in `serialization/` — it is shared with the Anthropic *provider* serializer.

## Related (adapter capability consolidation)

While shrinking `providers/base.py` (1406 → ~750 lines), the OpenAI-shaped raw builders and endpoint methods were moved into the capability mixins (`capabilities/chat|embedding|image|audio.py`), which previously duplicated them as MRO-shadowed dead code. Two traps had to be cleared:

- The mixins' copies had drifted from the live `BaseHttpProvider` implementations (e.g. `embeddings` used `_post_json_with_retry` instead of `_post_json_response_with_retry` and dropped rate-limit header capture). The live implementations were the source of truth; the mixins were overwritten from them.
- Mixins must precede `BaseHttpProvider` **and** `BaseAdapter` in adapter MROs, otherwise `BaseAdapter`'s `NotImplementedError` interface methods shadow the mixin implementations. All adapter class bases were reordered to `(*CapabilityMixins, BaseHttpProvider)`.
- `OpenAICompatibleBase` gained the embedding/image/audio mixins so the openai-compatible family keeps the OpenAI-shaped defaults it previously inherited from `BaseHttpProvider`.
- Bundled churn (no behavior change): `PassThroughChunkConverter` was renamed to `IdentityChunkConverter` (`serialization/providers/base.py`), and `ProviderSerializer.compatible_protocols` changed from `list[str]` to `frozenset[str]`.
