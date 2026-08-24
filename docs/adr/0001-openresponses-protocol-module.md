# Consolidate OpenResponses into one protocol module

The OpenResponses protocol was split across 13+ files in three directories (`protocols/openresponses/`, `serialization/handlers/openresponses/`, plus private imports in `core/processing/`), and core pipeline stages called the serializer's private functions (`_dispatch_input_item`, `_flush_pending_turn`) directly. We consolidated everything into a single deep module at `llm_proxy.protocols.openresponses` whose public interface is four verbs — parse, format, replay (`replay_stored_response`), materialize (`conversation_to_input_items`) — plus the endpoint registration and a streaming transformer with lifecycle hooks (`terminal_frames`, `error_frames`, `finalize_persistence`) so `streaming_processor` no longer branches on protocol names. Responses-shaped helpers shared by other families (tool-name namespaces, reasoning extraction, item ids) live in `llm_proxy.serialization.responses_toolkit`, not inside the module.

## Considered Options

- **Conversion-half only** (leave handler/schemas/store in `protocols/`, move nothing else): rejected — the handler's FormatContext contextvar is consumed by the serializer and streaming layer; splitting there would keep the seam cutting through the middle of one conversation.
- **Minimal leak fix** (expose one public materialize function, no relocation): rejected — understanding a single OpenResponses behavior would still require bouncing across three directories.
- **Top-level vertical slice** (`src/llm_proxy/openresponses/`): rejected for now — OpenResponses is a protocol, so it stays on the `protocols/` axis; a vertical-slice layout can be revisited if provider families are later consolidated the same way.
- **Separate ProtocolStreamHooks registry** instead of deepening the existing streaming-transformer seam: rejected — the lifecycle hooks need the transformer's per-stream state anyway, and a third registry concept adds interface without leverage.

## Consequences

- The two native-passthrough branches in `streaming_processor` (usage capture, `[DONE]` append) were deliberately left in place; they are passthrough knowledge, not protocol knowledge, and belong to a future Passthrough consolidation.
- Anthropic's streaming error shaping rides the same new transformer seam (minor, intentional scope extension) so no protocol-name branches remain in `streaming_processor`.
- `serialization/handlers/` keeps the OpenAI-family handlers; only the `openresponses` subpackage moved.
- Tests: compliance/roundtrip/handler tests are black-box against the public interface; white-box unit tests of parser internals live next to the module's tests and are the only ones allowed to import internals.
