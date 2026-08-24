# Canonical usage record: one field per billable fact

Billable usage flowed through five representations — raw provider dict, `Usage`/`StreamingUsage`, `EventContext` flat fields, `to_usage_dict()`, `TokenUsage` — and the "canonical" `Usage` model itself carried provider-flavored aliases: `cached_content_tokens` (Gemini) duplicated `cache_read_input_tokens`, `thoughts_tokens` (Gemini) duplicated `reasoning_tokens`, and cache-read could be expressed simultaneously as flat `cache_read_input_tokens` and nested `prompt_tokens_details.cached_tokens`. Every hop re-mapped the bag, and the cost calculation charged both expressions when a parser set both. We made the record canonical: provider serializers normalize dialect aliases at parse time (the alias fields are deleted from `Usage`), and `extract_tokens_from_usage` treats cache-read as one fact — the nested OpenAI-dialect expression is dropped when the flat field is present.

## Considered Options

- **Collapse the dual flat fields on `EventContext` (`cache_read_input_tokens` vs `cached_prompt_tokens`) into one**: rejected — they are audit-log DB columns; merging them is a persistence-visible migration, not a code deepening. The dialect split survives at the persistence seam deliberately.
- **Have `calculate_cost` consume the `Usage` dataclass end-to-end instead of a dict**: rejected for now — tracing/audit serialization needs the dict shape, and `TokenUsage` already canonicalizes field names for the cost functions; the leak was dual expressions within one record, not the dict itself.
- **Keep the alias fields as tolerated inputs**: rejected — an alias that parses is an alias that will be set again; deleting the fields makes the contract unrepresentable-when-violated.

## Consequences

- `Usage` gains a docstring contract: cache read → `cache_read_input_tokens`, cache write → `cache_creation_input_tokens`, thinking → `reasoning_tokens`; parsers must not introduce new aliases.
- The Gemini streaming converter emits canonical usage keys (`cache_read_input_tokens`, `reasoning_tokens`) in canonical chunks — the previous Gemini-flavored keys were stripped by the protocol-side transformer's usage cleaning, silently dropping cache data from streaming billing.
- `EventContext.update_usage` keeps its dual flat fields (persistence contract) but its alias-fallback branches are gone; the remaining multi-expression tolerance lives in exactly one place (`extract_tokens_from_usage`).
