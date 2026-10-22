"""One adapter per provider wire shape.

The dialect is a wire *shape*, not a provider: one dialect serves many
providers (chat-completions covers OpenAI, OpenRouter, Ollama, vLLM, SGLang,
…), and the native-passthrough tier has no provider serializer in play at all.
An adapter owns two things:

- **its envelope** — where the usage block sits in the payload. Anthropic
  nests it under ``message`` on ``message_start`` and at the top level on
  ``message_delta``; Responses stream events nest it under ``response``;
  Realtime nests it two deep in a ``response.done`` event;
- **its aliases** — which dialect key expresses a canonical fact, folded at
  parse time so the record holds one field per fact (ADR-0006);
- **the turn-level facts it needs** — a fact the wire states *beside* the usage
  block, which a later chunk may not repeat. The caller that saw the whole turn
  passes it in ``turn`` (gemini's ``search_grounded``); a fact the payload states
  itself wins, and a dialect that needs none ignores the argument.

Extraction only: nothing here emits a client-facing shape, and nothing here
estimates. See ADR-0025.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from llm_proxy.models.types import CompletionTokensDetails, PromptTokensDetails, Usage
from llm_proxy.serialization.usage.folds import (
    fold_deepseek_cache_hits,
)

type UsageDialect = Literal[
    "chat",
    "responses",
    "anthropic",
    "realtime",
    "ollama",
    "gemini",
    "audio",
    "cohere",
    "rerank",
]

__all__ = ["UsageDialect", "parse_usage"]


# ---------------------------------------------------------------------------
# Tolerant readers — wire payloads carry whatever the provider felt like
# ---------------------------------------------------------------------------


def _as_dict(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _int(value: Any) -> int | None:
    """An int the wire actually reported; booleans and floats are not counts."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _positive_int(value: Any) -> int | None:
    """A count the wire actually reported, and one that is a count (> 0)."""
    count = _int(value)
    return count if count is not None and count > 0 else None


def _positive_float(value: Any) -> float | None:
    """A positive number the wire actually reported; a bool is not a price."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if value > 0 else None


def _duration(value: Any) -> float | None:
    """A wire-reported duration in seconds; a bool is not a duration.

    The sibling reader of the same fact in the audio dialect (``_parse_audio``)
    checks the type before trusting ``seconds``; this keeps the realtime dialect
    from handing a string or a dict to billing's duration arithmetic.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _web_search_requests(server_tool_use: Any) -> int | None:
    """``server_tool_use.web_search_requests`` — provider-reported search count."""
    block = _as_dict(server_tool_use)
    if block is None:
        return None
    return _positive_int(block.get("web_search_requests"))


def _prompt_details(raw: Any) -> PromptTokensDetails | None:
    details = _as_dict(raw)
    if details is None:
        return None
    return PromptTokensDetails(
        audio_tokens=_int(details.get("audio_tokens")),
        cached_tokens=_int(details.get("cached_tokens")),
        image_tokens=_int(details.get("image_tokens")),
        text_tokens=_int(details.get("text_tokens")),
        cache_write_tokens=_int(details.get("cache_write_tokens")),
        video_tokens=_int(details.get("video_tokens")),
    )


def _completion_details(raw: Any) -> CompletionTokensDetails | None:
    details = _as_dict(raw)
    if details is None:
        return None
    return CompletionTokensDetails(
        accepted_prediction_tokens=_int(details.get("accepted_prediction_tokens")),
        audio_tokens=_int(details.get("audio_tokens")),
        reasoning_tokens=_int(details.get("reasoning_tokens")),
        rejected_prediction_tokens=_int(details.get("rejected_prediction_tokens")),
        image_tokens=_int(details.get("image_tokens")),
    )


def _lift_cache_facts(
    usage: dict[str, Any],
    prompt_details: PromptTokensDetails | None,
) -> tuple[int | None, int | None, PromptTokensDetails | None]:
    """Fold the OpenAI-dialect cache aliases into the canonical flat fields.

    Returns ``(cache_read, cache_write, prompt_details)``. The flat field wins
    and the nested copy is cleared, because the same tokens in both fields are
    priced twice downstream. Client emission already tolerates the flat-only
    form: the OpenAI protocol formatter falls back to
    ``usage.cache_read_input_tokens`` and OpenResponses takes the max of the
    two (``protocols/openai/formatting.py``,
    ``protocols/openresponses/serializer.py``).
    """
    cache_read = _int(usage.get("cache_read_input_tokens"))
    if cache_read is None:
        nested_cached = prompt_details.cached_tokens if prompt_details is not None else None
        cache_read = fold_deepseek_cache_hits(nested_cached, usage.get("prompt_cache_hit_tokens"))
    if cache_read is not None and prompt_details is not None:
        prompt_details.cached_tokens = None

    cache_write = _int(usage.get("cache_creation_input_tokens"))
    if cache_write is None and prompt_details is not None:
        cache_write = prompt_details.cache_write_tokens
    if cache_write is not None and prompt_details is not None:
        prompt_details.cache_write_tokens = None

    # A block whose only facts were the lifted ones says nothing: an all-``None``
    # details object is not ``None`` to a caller testing truthiness, so leaving
    # it behind would hand downstream code an empty nested block to emit.
    if prompt_details is not None and all(value is None for value in vars(prompt_details).values()):
        prompt_details = None

    return cache_read, cache_write, prompt_details


def _lift_reasoning(
    usage: dict[str, Any],
    completion_details: CompletionTokensDetails | None,
) -> tuple[int | None, CompletionTokensDetails | None]:
    """Fold the thinking aliases into the canonical flat ``reasoning_tokens``.

    Thinking tokens are one fact, and the canonical record keeps it flat
    (CONTEXT.md -> Canonical usage record). Dialects express it nested (OpenAI's
    and the Responses API's ``completion_tokens_details.reasoning_tokens``) or
    top-level (SGLang), so both are read and the alias is cleared — the same
    treatment the cache facts get, for the same reason: an alias left in place is
    a second expression of one fact that a reader may pick up twice.

    Returns ``(reasoning, completion_details)``: the details block keeps the
    details that have no canonical field of their own, and is dropped when nothing
    is left.
    """
    nested = completion_details.reasoning_tokens if completion_details is not None else None
    reasoning = _int(nested)
    if reasoning is None:
        # SGLang reports the count top-level, and reports it as zero by default, so
        # only a positive one is a report; an explicit nested value wins.
        reasoning = _positive_int(usage.get("reasoning_tokens"))
    if reasoning is not None and completion_details is not None:
        completion_details.reasoning_tokens = None
        if all(value is None for value in vars(completion_details).values()):
            completion_details = None
    return reasoning, completion_details


def _provider_reported_cost(usage: dict[str, Any]) -> float | None:
    """Cost the provider itself reported (NanoGPT's ``nanogpt_cost``, OpenRouter's ``cost``)."""
    for key in ("nanogpt_cost", "cost"):
        value = usage.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    return None


def _count_items(payload: dict[str, Any], item_type: str) -> int:
    """Count response items of one type in a non-streaming response body."""
    count = 0
    for container in (payload, _as_dict(payload.get("response")) or {}):
        items = container.get("output")
        if not isinstance(items, list):
            continue
        count += sum(
            1 for item in items if _as_dict(item) is not None and item.get("type") == item_type
        )
    return count


# ---------------------------------------------------------------------------
# Chat Completions
# ---------------------------------------------------------------------------


def _parse_chat(payload: Mapping[str, Any], turn: Mapping[str, Any] | None = None) -> Usage | None:
    block = _as_dict(payload.get("usage"))
    if block is None:
        return None
    usage = dict(block)
    prompt_details = _prompt_details(usage.get("prompt_tokens_details"))
    cache_read, cache_write, prompt_details = _lift_cache_facts(usage, prompt_details)
    reasoning, completion_details = _lift_reasoning(
        usage, _completion_details(usage.get("completion_tokens_details"))
    )

    input_tokens = _int(usage.get("prompt_tokens")) or 0
    output_tokens = _int(usage.get("completion_tokens")) or 0
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=_int(usage.get("total_tokens")) or (input_tokens + output_tokens),
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_write,
        reasoning_tokens=reasoning,
        prompt_tokens_details=prompt_details,
        completion_tokens_details=completion_details,
        provider_reported_cost=_provider_reported_cost(usage),
        web_search_requests=_web_search_requests(usage.get("server_tool_use")),
    )


# ---------------------------------------------------------------------------
# Anthropic Messages
# ---------------------------------------------------------------------------


def _anthropic_usage_block(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    """``message_start`` nests usage under ``message``; everything else is flat."""
    message = _as_dict(payload.get("message"))
    if message is not None:
        block = _as_dict(message.get("usage"))
        if block is not None:
            return block
    return _as_dict(payload.get("usage"))


def _parse_anthropic(
    payload: Mapping[str, Any], turn: Mapping[str, Any] | None = None
) -> Usage | None:
    usage = _anthropic_usage_block(payload)
    if usage is None:
        return None
    # Absent means "not reported", so both counters stay ``None``; the inclusive
    # input count needs a number, hence the local zero-default.
    cache_read = _int(usage.get("cache_read_input_tokens"))
    cache_write = _int(usage.get("cache_creation_input_tokens"))
    # The canonical invariant makes input_tokens INCLUDE the cache counters:
    # Anthropic reports them beside it, and emission subtracts them again.
    input_tokens = (_int(usage.get("input_tokens")) or 0) + (cache_read or 0) + (cache_write or 0)
    output_tokens = _int(usage.get("output_tokens")) or 0
    output_details = _as_dict(usage.get("output_tokens_details"))
    reasoning_tokens = _int(output_details.get("thinking_tokens")) if output_details else None

    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_write,
        reasoning_tokens=reasoning_tokens,
        web_search_requests=_web_search_requests(usage.get("server_tool_use")),
    )


# ---------------------------------------------------------------------------
# OpenAI Responses
# ---------------------------------------------------------------------------


def _responses_usage_block(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    """A response body is flat; a stream event nests usage under ``response``."""
    response = _as_dict(payload.get("response"))
    if response is not None:
        block = _as_dict(response.get("usage"))
        if block is not None:
            return block
    return _as_dict(payload.get("usage"))


def _parse_responses(
    payload: Mapping[str, Any], turn: Mapping[str, Any] | None = None
) -> Usage | None:
    # The Responses wire has no usage-level search count: a response body
    # reports its calls as output items, so the items are the count whenever the
    # provider did not report one itself. A stream event announces its calls the
    # same way (:func:`_count_responses_events`).
    counted = _count_items(dict(payload), "web_search_call")
    usage = _responses_usage_block(payload)
    if usage is None:
        # Native web search calls bill per request even with no usage object.
        return Usage(web_search_requests=counted) if counted else None

    input_details = _as_dict(usage.get("input_tokens_details")) or _as_dict(
        usage.get("prompt_tokens_details")
    )
    output_details = _as_dict(usage.get("output_tokens_details")) or _as_dict(
        usage.get("completion_tokens_details")
    )
    prompt_details = _prompt_details(input_details)
    cache_read, cache_write, prompt_details = _lift_cache_facts(usage, prompt_details)
    reasoning, completion_details = _lift_reasoning(usage, _completion_details(output_details))
    input_tokens = _int(usage.get("input_tokens")) or 0
    output_tokens = _int(usage.get("output_tokens")) or 0

    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=_int(usage.get("total_tokens")) or (input_tokens + output_tokens),
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_write,
        reasoning_tokens=reasoning,
        prompt_tokens_details=prompt_details,
        completion_tokens_details=completion_details,
        provider_reported_cost=_provider_reported_cost(usage),
        web_search_requests=_positive_int(usage.get("web_search_requests")) or counted or None,
    )


def _count_responses_events(payload: Mapping[str, Any]) -> int:
    """Billable events a Responses stream announces as they happen.

    Only ``response.output_item.added`` announces a call — the same rule the
    converted tier counts by. The terminal ``response.completed`` event
    restates every item in ``response.output``, so counting that array here
    would double every search; a non-streaming body is counted by
    :func:`_parse_responses` from its own array instead.
    """
    if payload.get("type") != "response.output_item.added":
        return 0
    item = _as_dict(payload.get("item"))
    return 1 if item is not None and item.get("type") == "web_search_call" else 0


# ---------------------------------------------------------------------------
# Realtime
# ---------------------------------------------------------------------------


def _realtime_usage_block(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    response = _as_dict(payload.get("response"))
    if response is not None:
        block = _as_dict(response.get("usage"))
        if block is not None:
            return block
    return _as_dict(payload.get("usage"))


def _parse_realtime(
    payload: Mapping[str, Any], turn: Mapping[str, Any] | None = None
) -> Usage | None:
    usage = _realtime_usage_block(payload)
    if usage is None:
        return None
    # The Realtime dialect nests under singular "token" keys.
    input_details = _as_dict(usage.get("input_token_details")) or _as_dict(
        usage.get("input_tokens_details")
    )
    output_details = _as_dict(usage.get("output_token_details")) or _as_dict(
        usage.get("output_tokens_details")
    )
    prompt_details = _prompt_details(input_details)
    cache_read, cache_write, prompt_details = _lift_cache_facts(usage, prompt_details)
    reasoning, completion_details = _lift_reasoning(usage, _completion_details(output_details))
    input_tokens = _int(usage.get("input_tokens")) or 0
    output_tokens = _int(usage.get("output_tokens")) or 0

    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=_int(usage.get("total_tokens")) or (input_tokens + output_tokens),
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_write,
        reasoning_tokens=reasoning,
        prompt_tokens_details=prompt_details,
        completion_tokens_details=completion_details,
        audio_duration_seconds=_duration(usage.get("audio_duration_seconds")),
        web_search_requests=_positive_int(usage.get("web_search_requests")),
    )


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------


def _ollama_counts(payload: Mapping[str, Any]) -> tuple[int, int] | None:
    """Ollama's ``(input, output)`` counts.

    Ollama reports ``prompt_eval_count`` and ``eval_count`` on the terminal
    response or chunk, and a reported ``0`` is a real value (the prompt was
    served from cache, or no output was generated), so the guard must not treat
    it as "missing". Either counter may be absent while the other is present;
    ``None`` means neither was there, and the caller builds no record at all.
    """
    input_tokens = _int(payload.get("prompt_eval_count"))
    output_tokens = _int(payload.get("eval_count"))
    if input_tokens is None and output_tokens is None:
        return None
    return input_tokens or 0, output_tokens or 0


def _ollama_cached_tokens(payload: Mapping[str, Any], input_tokens: int) -> int | None:
    """``prompt_eval_cached_count`` — how much of the prompt the cache served.

    A *subset* of ``prompt_eval_count``, which Ollama keeps as the logical input
    total — the same contract as the canonical record, whose ``input_tokens``
    includes its cache counters — so the value is clamped into
    ``[0, input_tokens]``: a provider breaking that invariant yields a sane count
    instead of a negative or oversized one. The field is a nullable pointer
    upstream and only exists on newer servers, so absent means ``None``
    ("reported nothing"), which is not the same as a reported zero.
    """
    cached = _int(payload.get("prompt_eval_cached_count"))
    if cached is None:
        return None
    return max(0, min(cached, input_tokens))


def _parse_ollama(
    payload: Mapping[str, Any], turn: Mapping[str, Any] | None = None
) -> Usage | None:
    counts = _ollama_counts(payload)
    if counts is None:
        return None
    input_tokens, output_tokens = counts
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        # Ollama reports no total; the canonical record derives one from the two
        # counters it does report (``Usage.__post_init__``).
        total_tokens=input_tokens + output_tokens,
        cache_read_input_tokens=_ollama_cached_tokens(payload, input_tokens),
    )


# ---------------------------------------------------------------------------
# Audio (transcription / translation)
# ---------------------------------------------------------------------------


def _audio_prompt_details(usage: Mapping[str, Any]) -> PromptTokensDetails | None:
    """The audio dialect nests its input split under either key name.

    ``gpt-4o-transcribe`` reports ``input_token_details``; a chat-shaped audio
    wire reports ``prompt_tokens_details``. A block with no audio/cache/text
    fact is ``None`` rather than an all-``None`` object, matching the rest of the
    seam.
    """
    raw = _as_dict(usage.get("input_token_details")) or _as_dict(usage.get("prompt_tokens_details"))
    details = _prompt_details(raw)
    if details is not None and all(value is None for value in vars(details).values()):
        return None
    return details


def _audio_completion_details(usage: Mapping[str, Any]) -> CompletionTokensDetails | None:
    """The audio wire carries only the audio split of the output tokens."""
    raw = _as_dict(usage.get("completion_tokens_details"))
    audio = _int(raw.get("audio_tokens")) if raw is not None else None
    return CompletionTokensDetails(audio_tokens=audio) if audio is not None else None


def _parse_audio(payload: Mapping[str, Any], turn: Mapping[str, Any] | None = None) -> Usage | None:
    """The audio endpoints' usage block: token-based or duration-based.

    The block sits under ``usage`` on the response body and on every streaming
    ``transcript.text.done`` event. A token-based block (gpt-4o-transcribe)
    reports ``input_tokens``/``output_tokens`` with the audio/text split nested;
    a duration-based one (whisper) reports ``{"type": "duration", "seconds":
    N}``. OpenRouter reports a bare ``{"seconds": N, "cost": ...}`` with no
    type discriminator — reading that as token usage would bill a per-second
    endpoint from token estimation.
    """
    usage = _as_dict(payload.get("usage"))
    if not usage:
        return None

    seconds = _duration(usage.get("seconds"))
    if usage.get("type") == "duration" or seconds is not None:
        # Duration-based responses may still carry the input token split
        # (gpt-4o-transcribe with ``include[]=usage``).
        return Usage(
            audio_duration_seconds=seconds,
            prompt_tokens_details=_audio_prompt_details(usage),
            # OpenRouter's bare ``{"seconds": N, "cost": ...}`` reports what it
            # charged; a duration the proxy can price is not a reason to discard
            # a cost the wire stated (cost.py prefers the reported one).
            provider_reported_cost=_provider_reported_cost(usage),
        )

    return Usage(
        input_tokens=_int(usage.get("input_tokens")) or 0,
        output_tokens=_int(usage.get("output_tokens")) or 0,
        total_tokens=_int(usage.get("total_tokens")),
        prompt_tokens_details=_audio_prompt_details(usage),
        completion_tokens_details=_audio_completion_details(usage),
        provider_reported_cost=_provider_reported_cost(usage),
    )


# ---------------------------------------------------------------------------
# Cohere
# ---------------------------------------------------------------------------


def _cohere_usage_block(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    """Cohere's usage block, wherever this endpoint keeps it.

    The chat body states it at ``usage``, the terminal ``message-end`` stream
    event nests it under ``delta``, and the embed/rerank responses carry the same
    ``ApiMeta`` shape under ``meta`` (docs.cohere.com/reference/chat, /embed,
    /rerank) — one wire shape, three positions.
    """
    delta = payload.get("delta")
    candidates = [payload.get("usage"), payload.get("meta")]
    if isinstance(delta, dict):
        candidates.append(delta.get("usage"))
    for candidate in candidates:
        block = _as_dict(candidate)
        if block is not None:
            return block
    return None


def _parse_cohere(
    payload: Mapping[str, Any], turn: Mapping[str, Any] | None = None
) -> Usage | None:
    """Cohere's ``ApiMeta``: ``tokens`` counters plus a billed-units mirror.

    ``tokens.input_tokens`` / ``output_tokens`` are the counts the model used and
    the ones a per-token price applies to; ``billed_units`` mirrors them after
    cache hits and is only a fallback for a block that states nothing else.
    ``cached_tokens`` is documented as prompt tokens that hit the inference
    cache, i.e. a subset of the prompt count — the canonical invariant — so it
    lands in the flat cache field alone, clamped for a provider that breaks it.
    """
    usage = _cohere_usage_block(payload)
    if usage is None:
        return None

    tokens = _as_dict(usage.get("tokens")) or {}
    input_tokens = _int(tokens.get("input_tokens"))
    output_tokens = _int(tokens.get("output_tokens"))
    if input_tokens is None and output_tokens is None:
        billed = _as_dict(usage.get("billed_units")) or {}
        input_tokens = _int(billed.get("input_tokens"))
        output_tokens = _int(billed.get("output_tokens"))
    if input_tokens is None and output_tokens is None:
        return None

    input_tokens = input_tokens or 0
    output_tokens = output_tokens or 0
    cached_tokens = _int(usage.get("cached_tokens"))
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cache_read_input_tokens=(
            max(0, min(cached_tokens, input_tokens)) if cached_tokens is not None else None
        ),
    )


# ---------------------------------------------------------------------------
# Shared /rerank usage block
# ---------------------------------------------------------------------------


def _parse_rerank(
    payload: Mapping[str, Any], turn: Mapping[str, Any] | None = None
) -> Usage | None:
    """The shared ``/rerank`` usage block: ``{total_tokens, search_units?, cost?}``.

    OpenRouter's ``/api/v1/rerank`` states it at ``usage``; Cohere's v2 answers the
    same facts inside ``ApiMeta`` and is read by its own dialect. A reranker scores
    a query against a fixed document list and generates nothing, so the whole count
    is a prompt-side count: a per-token price applies to ``input_tokens`` and
    ``output_tokens`` stays zero.

    ``search_units`` — the billed query units Cohere also reports — is not a token
    fact and has no canonical field; the client-facing echo renders it from the
    wire. The ``cost`` *is* a record fact (``provider_reported_cost``), so billing
    prefers it over the proxy's own estimate without a provider-info extra.
    """
    block = _as_dict(payload.get("usage"))
    if block is None:
        return None
    total_tokens = _int(block.get("total_tokens"))
    if total_tokens is None:
        return None
    return Usage(
        input_tokens=total_tokens,
        total_tokens=total_tokens,
        provider_reported_cost=_positive_float(block.get("cost")),
    )


# ---------------------------------------------------------------------------
# Gemini generateContent
# ---------------------------------------------------------------------------


def _gemini_search_queries(payload: Mapping[str, Any]) -> int:
    """Native Google Search grounding: how many queries the model issued.

    ``groundingMetadata.webSearchQueries`` lives on the candidate, not in
    ``usageMetadata``, and lists the queries the model actually issued; Google
    bills each as a per-request search fee. The count is of *distinct non-empty*
    queries across every candidate, which is Google's documented rule ("we ignore
    the empty web search queries when counting unique queries"): a query the
    model repeated, or issued in more than one candidate of a multi-candidate
    request, is one billable search, not two.
    """
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        return 0
    queries: set[str] = set()
    for candidate in candidates:
        grounding = _as_dict(candidate.get("groundingMetadata")) if _as_dict(candidate) else None
        if grounding is None:
            continue
        issued = grounding.get("webSearchQueries")
        if isinstance(issued, list):
            queries.update(query for query in issued if isinstance(query, str) and query)
    return len(queries)


def _gemini_billable_counts(
    meta: Mapping[str, Any], *, has_search_grounding: bool
) -> tuple[int, int]:
    """``(input, output)`` from a Gemini ``usageMetadata``, as Google bills it.

    ``usageMetadata`` carries more dimensions than the OpenAI-shaped usage the
    proxy prices:

    - ``toolUsePromptTokenCount`` — tool-use prompt tokens, reported SEPARATELY
      from ``promptTokenCount``. They are billed at the input rate, EXCEPT when
      they came from Google Search grounding (Google bills those via the
      per-request search fee instead), so search-grounded calls exclude them. The
      API only exposes the aggregate across ALL tools, so when a request mixes
      search grounding with function calling, the function-call tool tokens are
      excluded too and input is undercounted — a documented limitation (no
      per-tool breakdown is available).
    - ``thoughtsTokenCount`` — thinking tokens. Depending on model/version,
      ``candidatesTokenCount`` may or may not include them. Detected via the
      total: ``prompt + candidates + toolUse + thoughts == total`` means
      candidates exclude thoughts, so they are added to the output side.
      Requiring the exact match (rather than "sum != total") keeps the heuristic
      from misfiring when ``totalTokenCount`` carries additional dimensions.
      ``cachedContentTokenCount`` is NOT one of them: per the Gemini docs,
      ``promptTokenCount`` already includes the cached content tokens, so they are
      inside the sum above.
    """
    prompt = _int(meta.get("promptTokenCount")) or 0
    candidates = _int(meta.get("candidatesTokenCount")) or 0
    tool_use = _int(meta.get("toolUsePromptTokenCount")) or 0
    thoughts = _int(meta.get("thoughtsTokenCount")) or 0
    total = _int(meta.get("totalTokenCount")) or 0

    input_tokens = prompt if has_search_grounding else prompt + tool_use
    output_tokens = candidates
    if thoughts and total and prompt + candidates + tool_use + thoughts == total:
        output_tokens = candidates + thoughts
    return input_tokens, output_tokens


def _gemini_modality_counts(entries: Any) -> dict[str, int]:
    """Read a ``UsageMetadata`` modality breakdown into ``{modality: tokens}``.

    Gemini reports its prompt / candidate / cached token totals twice: as the
    flat count and, since the multimodal models, as a list of
    ``ModalityTokenCount`` (``modality``: ``TEXT`` | ``IMAGE`` | ``VIDEO`` |
    ``AUDIO`` | ``DOCUMENT``). The breakdown is what the per-modality price
    differentials are read from, so it is folded into the canonical details
    blocks instead of being dropped — except ``DOCUMENT``, which has no slot in
    :class:`PromptTokensDetails` and is therefore dropped here.
    """
    counts: dict[str, int] = {}
    if not isinstance(entries, list):
        return counts
    for entry in entries:
        block = _as_dict(entry)
        if block is None:
            continue
        modality = str(block.get("modality") or "").lower()
        count = _int(block.get("tokenCount"))
        if not modality or modality.startswith("modality_") or count is None:
            continue
        counts[modality] = counts.get(modality, 0) + count
    return counts


def _gemini_prompt_details(meta: Mapping[str, Any]) -> PromptTokensDetails | None:
    """The prompt side of the modality breakdown, in the canonical details block.

    ``cached_tokens`` is deliberately left unset: ``cachedContentTokenCount`` is
    already flat in ``cache_read_input_tokens`` (one fact, one field).
    """
    counts = _gemini_modality_counts(meta.get("promptTokensDetails"))
    details = PromptTokensDetails(
        text_tokens=counts.get("text"),
        image_tokens=counts.get("image"),
        audio_tokens=counts.get("audio"),
        video_tokens=counts.get("video"),
    )
    return details if any(value is not None for value in vars(details).values()) else None


def _gemini_completion_details(meta: Mapping[str, Any]) -> CompletionTokensDetails | None:
    """The candidate side of the modality breakdown (image/audio output tokens)."""
    counts = _gemini_modality_counts(meta.get("candidatesTokensDetails"))
    details = CompletionTokensDetails(
        image_tokens=counts.get("image"),
        audio_tokens=counts.get("audio"),
    )
    return details if any(value is not None for value in vars(details).values()) else None


def _parse_gemini(
    payload: Mapping[str, Any], turn: Mapping[str, Any] | None = None
) -> Usage | None:
    meta = _as_dict(payload.get("usageMetadata"))
    # Native Google Search grounding is a turn-level fact: it lives on the
    # candidate, and a usage-only stream chunk carries none. It is read from the
    # payload when the payload states it and from ``turn`` otherwise (the caller
    # is the one that saw the whole turn) — and it is read before the metadata,
    # because the input count below depends on it.
    search_requests = _gemini_search_queries(payload)
    grounded = search_requests > 0 or bool((turn or {}).get("search_grounded"))
    if meta is None:
        # Search grounding bills per request even with no usage block (the
        # Responses dialect's ``web_search_call`` items do the same).
        return Usage(web_search_requests=search_requests) if search_requests else None
    input_tokens, output_tokens = _gemini_billable_counts(meta, has_search_grounding=grounded)
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=_int(meta.get("totalTokenCount")) or (input_tokens + output_tokens),
        # ``cachedContentTokenCount`` is inside ``promptTokenCount`` (Gemini
        # documents the prompt count as the logical input total), which is the
        # canonical invariant, so it goes in the flat field only.
        cache_read_input_tokens=_int(meta.get("cachedContentTokenCount")),
        # Thoughts are billed as output; ``reasoning_tokens`` documents how many of
        # those output tokens they were, the same subset relationship the OpenAI
        # dialects express.
        reasoning_tokens=_int(meta.get("thoughtsTokenCount")),
        web_search_requests=search_requests or None,
        prompt_tokens_details=_gemini_prompt_details(meta),
        completion_tokens_details=_gemini_completion_details(meta),
    )


# ---------------------------------------------------------------------------
# The seam
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DialectAdapter:
    """What a dialect adapter must answer."""

    parse: Callable[[Mapping[str, Any], Mapping[str, Any] | None], Usage | None]
    #: Billable events the payload announces outside its usage block (a
    #: streaming terminal event may never restate them). Summed per turn.
    count_events: Callable[[Mapping[str, Any]], int] = lambda _payload: 0
    #: Fields whose parsed value is a function of the ``turn`` facts, not of the
    #: payload alone (gemini's grounded input count). An observation parsed before
    #: a fact was known answers a different question, so the accumulator drops
    #: these fields when the facts change and re-reads its last payload under the
    #: new ones.
    turn_dependent_fields: tuple[str, ...] = ()


_ADAPTERS: frozendict[str, DialectAdapter] = frozendict(
    {
        "chat": DialectAdapter(_parse_chat),
        "responses": DialectAdapter(_parse_responses, _count_responses_events),
        "anthropic": DialectAdapter(_parse_anthropic),
        "realtime": DialectAdapter(_parse_realtime),
        "ollama": DialectAdapter(_parse_ollama),
        "gemini": DialectAdapter(_parse_gemini, turn_dependent_fields=("input_tokens",)),
        "audio": DialectAdapter(_parse_audio),
        "cohere": DialectAdapter(_parse_cohere),
        "rerank": DialectAdapter(_parse_rerank),
    }
)


def parse_usage(
    dialect: UsageDialect,
    payload: Mapping[str, Any] | None,
    *,
    turn: Mapping[str, Any] | None = None,
) -> Usage | None:
    """Extract the canonical usage record from one provider wire payload.

    Args:
        dialect: the wire shape — declared once per conversion path, never
            re-decided at the call site (a provider serializer knows its own;
            a native-tier protocol declares its own).
        payload: the protocol envelope — a response body, a stream chunk, or a
            stream event. The adapter knows where its dialect hides the usage
            block inside it.
        turn: facts about the turn that the payload cannot carry, named by the
            dialect that needs them (``search_grounded`` for gemini). A caller
            that has seen the whole turn states them; a caller that has not
            leaves them out and the adapter reads the payload alone.

    Returns:
        The canonical record, or ``None`` when the payload carries no usage
        (a non-terminal chunk, or a response whose provider reported nothing).
    """
    if payload is None:
        return None
    adapter = _ADAPTERS.get(dialect)
    if adapter is None:
        raise ValueError(f"unknown usage dialect: {dialect!r}")
    return adapter.parse(payload, turn)


def count_billable_events(dialect: UsageDialect, payload: Mapping[str, Any] | None) -> int:
    """Count billable events this payload announces outside its usage block."""
    if payload is None:
        return 0
    adapter = _ADAPTERS.get(dialect)
    if adapter is None:
        raise ValueError(f"unknown usage dialect: {dialect!r}")
    return adapter.count_events(payload)


def turn_dependent_fields(dialect: UsageDialect) -> tuple[str, ...]:
    """The record fields this dialect's parse derives from the ``turn`` facts."""
    adapter = _ADAPTERS.get(dialect)
    if adapter is None:
        raise ValueError(f"unknown usage dialect: {dialect!r}")
    return adapter.turn_dependent_fields
