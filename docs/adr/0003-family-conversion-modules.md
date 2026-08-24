# One conversion module per provider dialect family

Each provider family's conversion logic was split between `serialization/<family>/` (dialect knowledge: mixins, builders, parsers, chunk converters) and `providers/<family>/serializer.py` (the registered serializer class composing that knowledge), with three reverse import edges (`serialization/<family>/__init__` re-importing the provider serializer for re-export), two pure-composition shell files (gemini 49 lines, ollama 41 lines existing only to compose mixins), private cross-package imports (`providers/openai/serializer.py` importing `_message_to_openai`), and one loose file (`serialization/gemini_streaming.py`) outside its family directory. We moved the registered provider serializers of the four dialect families (openai, anthropic, gemini, ollama) into `serialization/<family>/serializer.py`, making the dependency strictly one-way: `providers/` (adapters, HTTP) → `serialization/` (dialect conversion).

## Considered Options

- **`providers/<family>/` absorbs `serialization/<family>/`** (provider vertical slice): rejected — the OpenAI dialect is the proxy's canonical intermediate format, shared by the protocol-side handlers (`handlers/openai_chat`), `core/token_estimator`, and five provider-side parties; housing it under `providers/` would force the protocol layer and core to depend on a provider package.
- **Move every registered provider serializer under `serialization/`** (including chutes/nanogpt): rejected — those serializers are provider-specific tweaks (custom embedding decoding, chat-completions nudges), not dialect-generic knowledge; they stay provider-local. `serialization/providers/chat_completions.py` (openrouter/deepseek) was already in the right place.
- **Unify the protocol-side serializers into the family dirs too** (move `handlers/openai_*`, formalize `anthropic/protocol.py`): rejected — client-protocol parsing is a separate axis from provider-dialect conversion; left untouched (candidate for a future consolidation).
- **Rename all family modules to one convention** (converter/conversation/mixin): rejected — cosmetic churn; only the genuinely misplaced `gemini_streaming.py` moves (to `gemini/streaming_converter.py`, matching anthropic/openai).

## Consequences

- `serialization/<family>/` now owns the complete dialect conversion story per family: knowledge modules + the registered `serializer.py`. Registration is triggered by `serialization/__init__` auto-discovery of `serialization.<family>.serializer` (the `providers/*/serializer.py` discovery remains for chutes/nanogpt).
- The three reverse `serialization/<family>/__init__` re-export edges are dropped (they had zero consumers); private imports like `_message_to_openai` become intra-package, which is acceptable within one trust boundary.
- Adapters are untouched except two helper imports (`_parse_usage_from_response`, anthropic message helpers) now resolving from the new location; adapters obtain serializers through the registry as before.
