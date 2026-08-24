"""Protocol layer: HTTP endpoint schemas, serializers, and registry.

Each protocol (``openai/``, ``anthropic/``, ``openresponses/``) is a deep
module owning everything client-facing about that wire format:

- endpoint configuration (``handler.py`` — ProtocolEndpoint: paths, request
  models, streaming transformer, middleware, lifecycle hooks)
- wire-format conversion (``serializer.py`` — ProtocolSerializer:
  wire ↔ Internal* models)
- protocol-side SSE transformers (``streaming.py``)

The registries for endpoints and protocol serializers live in
``registry.py``. Provider-dialect conversion (provider serializers) is a
separate axis and lives in ``llm_proxy.serialization``; adapters live in
``llm_proxy.providers``. All protocols use InternalRequest/InternalResponse
as the internal format.
"""

__all__: list[str] = []
