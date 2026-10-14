"""The segment accumulator: one engine assembles canonical content blocks.

A streaming turn arrives as interleaved segments — text, thinking (with its
signature and its redacted/encrypted variants), tool-call arguments, media,
refusal — and this module owns the machine that decides when a segment opens,
when it closes, and which canonical :class:`ContentBlock` it becomes
(ADR-0031). The protocols stay the knowledge (ADR-0003, ADR-0009): each
declares a :class:`SegmentPolicy` and maps the returned :class:`SegmentEvent`
stream onto its wire; the engine owns the buffering, the closing and the
materialization, including the drift rulings ADR-0031 records — where the
per-protocol copies disagreed, the canonical level keeps what every copy
dropped (reasoning after text, signature-only segments, refusal,
``thought_signature``, media on a truncated flush).

The wire-shaping rules that legitimately differ per protocol are policy, not
engine code: tool-call coexistence (parallel lanes vs a sequential single
call), the block cursor (the engine's next-free count vs a protocol-owned
index), and a tool classifier hook (the Anthropic ``server_tool_use`` rule).
Policies are frozen dataclasses declared next to the protocol transformer that
consumes them; the engine imports nothing from ``protocols`` or
``serialization``.

The native/wire rebuilds (the Anthropic ``_native_blocks`` machine, the
OpenResponses terminal snapshot) are separate machines by design — they carry
wire facts the canonical block model cannot hold, and this engine
deliberately does not absorb them (ADR-0031).
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import orjson
from orjson import JSONDecodeError

from llm_proxy.core.utils import create_image_source_from_url
from llm_proxy.models.content_blocks import (
    AudioBlock,
    ContentBlock,
    CustomToolUseBlock,
    ImageBlock,
    RedactedThinkingBlock,
    RefusalBlock,
    ServerToolUseBlock,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
)
from llm_proxy.models.content_blocks.extended import SignatureOrigin
from llm_proxy.models.types import AudioSource

#: The segment kinds the engine distinguishes. Canonical facts only — a
#: wire-only notion (citations, raw provider blocks) is protocol knowledge
#: and stays out until a protocol needs it as an event kind.
DeltaKind = Literal["text", "reasoning", "tool", "media", "refusal"]

#: The order segments close in: canonical content order (reasoning precedes
#: visible content, which precedes tool calls), used both for the eviction
#: a newly-arriving kind performs and for the terminal flush.
_CLOSE_ORDER: tuple[str, ...] = ("reasoning", "text", "refusal", "tool", "media")

ToolVariant = Literal["function", "custom", "server_tool_use"]


class Coexistence(Enum):
    """How tool lanes coexist within one open tool segment.

    ``PARALLEL`` pools tool calls by their provider index (Chat Completions
    streams several calls concurrently). ``SEQUENTIAL`` closes the open call
    when a new one starts (Anthropic's one-block-at-a-time wire).
    """

    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"


class CursorPolicy(Enum):
    """Who owns the block index the web-search continuation extends.

    ``NONE`` — the protocol tracks no cursor (Chat Completions);
    ``NEXT_FREE`` — the engine's closed-block count is the next free index
    (Anthropic's ``_current_block_index``).
    """

    NONE = "none"
    NEXT_FREE = "next_free"


@dataclass(frozen=True)
class SegmentPolicy:
    """What a protocol declares about its segments; everything else is engine.

    Fields exist only where two real adapters disagree (or a committed slice
    needs them): the gemini-style whole-turn buffering adds its fields when
    that migration lands, not before.
    """

    #: How concurrent tool calls coexist (see :class:`Coexistence`).
    tool_coexistence: Coexistence = Coexistence.PARALLEL
    #: Who owns the block cursor (see :class:`CursorPolicy`).
    cursor: CursorPolicy = CursorPolicy.NONE
    #: Block-variant classifier evaluated at tool open; ``None`` keeps the
    #: structural default (``custom`` deltas make custom blocks, everything
    #: else a function block, and the converter's server-tool marker is
    #: ignored — matching the Chat Completions machine it replaces).
    classify_tool: ToolClassifier | None = None
    #: When set, the turn accumulates whole: a delta never evicts the other
    #: kinds' open lanes, and ``finish()`` closes them in this order. The
    #: Gemini-family converters buffer the entire turn and materialize it
    #: once (audio, then thinking, then text, then tools); segmented turns —
    #: the default — close a lane the moment a delta of another kind arrives.
    whole_turn_close_order: tuple[DeltaKind, ...] | None = None


@dataclass(frozen=True)
class CanonicalDelta:
    """One segment-bearing fragment, extracted from a canonical delta.

    The engine's feed unit. ``feed(dict)`` builds these from a canonical
    chat-completions delta; an item-addressed caller states its own deltas
    through :meth:`SegmentAccumulator.feed_delta`.
    """

    kind: DeltaKind
    #: Text carried by the segment: visible text, reasoning text, a refusal,
    #: or a tool-arguments fragment (concatenated per lane).
    text: str = ""
    #: Image markdown (``![image](url)``) appended with the lane's separator
    #: rule rather than concatenated blind, so a text lane keeps the newline
    #: between prose and an embedded image the way the machine always has.
    markdown: str = ""
    #: Reasoning signature fragment (concatenated per lane).
    signature: str | None = None
    #: Which provider produced the signature, stamped on the block.
    signature_origin: SignatureOrigin | None = None
    #: The encrypted reasoning payload (or the opaque redacted one).
    encrypted_content: str | None = None
    is_redacted: bool = False
    #: OpenRouter's structured reasoning array; last non-empty snapshot wins.
    reasoning_details: tuple[dict[str, Any], ...] | None = None
    #: Tool call identity (parallel lanes pool by provider index).
    tool_index: int = 0
    tool_id: str | None = None
    tool_name: str | None = None
    tool_variant: Literal["function", "custom"] | None = None
    #: The converter's server-side marker (``server_tool_use`` on the raw
    #: ``tool_calls`` entry); the structural default ignores it, the
    #: Anthropic classifier turns it into a ``ServerToolUseBlock``.
    server_tool: bool = False
    thought_signature: str | None = None
    #: Media payload: ``{"image": {...}}`` or ``{"audio_data": str,
    #: "audio_transcript": str}``.
    payload: dict[str, Any] | None = None
    #: Close the open lane of ``kind`` before feeding this delta, so two
    #: same-kind deltas materialize as separate blocks. An item-addressed
    #: caller states its item/part boundary this way; the default merges a
    #: run of same-kind deltas into one block.
    close: bool = False


#: Decides a tool call's block variant at open time from what the canonical
#: delta itself states: the tool name, the converter's server-side marker and
#: the custom variant. The default is structural (custom vs function); the
#: Anthropic protocol overrides it with the web-search rule.
ToolClassifier = Callable[[CanonicalDelta], ToolVariant]


@dataclass(frozen=True)
class SegmentOpened:
    """A segment opened; ``index`` is the block index it will occupy."""

    kind: DeltaKind
    index: int | None
    delta: CanonicalDelta


@dataclass(frozen=True)
class SegmentAppended:
    """A fragment appended to the open segment of its kind."""

    kind: DeltaKind
    index: int | None
    delta: CanonicalDelta


@dataclass(frozen=True)
class SegmentClosed:
    """A segment closed into one or more canonical blocks.

    A media lane closes into several blocks (one per image, plus audio), so
    the event carries a tuple; a text, reasoning, refusal or tool lane closes
    into exactly one.
    """

    kind: DeltaKind
    index: int | None
    blocks: tuple[ContentBlock, ...]


SegmentEvent = SegmentOpened | SegmentAppended | SegmentClosed


@dataclass
class _TextLane:
    text: str = ""


@dataclass
class _ReasoningLane:
    text: str = ""
    signature: str = ""
    signature_origin: SignatureOrigin | None = None
    encrypted_content: str | None = None
    is_redacted: bool = False
    details: list[dict[str, Any]] | None = None


@dataclass
class _ToolLane:
    index: int
    id: str = ""
    name: str = ""
    args: str = ""
    variant: ToolVariant = "function"
    thought_signature: str | None = None


@dataclass
class _MediaLane:
    images: list[dict[str, Any]] = field(default_factory=list)
    audio_data: str = ""
    audio_transcript: str = ""
    #: The chunk's inline MIME type; empty falls back to ``audio/wav``.
    audio_media_type: str = ""


def _build_reasoning_block(lane: _ReasoningLane) -> ThinkingBlock | RedactedThinkingBlock:
    """Materialize a reasoning lane; a signature-only segment stays a block.

    The redacted variant carries the opaque payload from ``encrypted_content``
    and only falls back to the buffered text — which on the redacted path is
    the ``[redacted]`` display placeholder, never a real payload to echo.
    """
    details = lane.details or None
    extra = {"reasoning_details": details} if details else {}
    if lane.is_redacted:
        return RedactedThinkingBlock(data=lane.encrypted_content or lane.text, extra=extra)
    return ThinkingBlock(
        thinking=lane.text,
        signature=lane.signature or None,
        signature_origin=lane.signature_origin,
        encrypted_content=lane.encrypted_content,
        extra=extra,
    )


def _build_tool_block(
    lane: _ToolLane,
) -> ToolUseBlock | ServerToolUseBlock | CustomToolUseBlock | None:
    """Materialize one tool call; a call missing its id or name is dropped.

    Function arguments are JSON-parsed at close with the ``(JSONDecodeError,
    TypeError)`` fallback; the ``custom`` variant's freeform input is never
    parsed — it is carried verbatim (ADR-0031's drift rulings). The
    ``server_tool_use`` variant comes from the classifier's open-time
    decision and keeps the same argument handling as a function call.
    """
    if not (lane.id and lane.name):
        return None
    try:
        tool_input = orjson.loads(lane.args) if lane.args and lane.variant != "custom" else {}
    except JSONDecodeError, TypeError:
        tool_input = {}
    extra = {"thought_signature": lane.thought_signature} if lane.thought_signature else {}
    if lane.variant == "server_tool_use":
        return ServerToolUseBlock(id=lane.id, name=lane.name, input=tool_input, extra=extra)
    if lane.variant == "custom":
        return CustomToolUseBlock(id=lane.id, name=lane.name, input=lane.args)
    return ToolUseBlock(id=lane.id, name=lane.name, input=tool_input, extra=extra)


def _build_media_blocks(lane: _MediaLane) -> list[ContentBlock]:
    """Materialize buffered images and audio; an unparseable image URL drops."""
    blocks: list[ContentBlock] = []
    for img in lane.images:
        if isinstance(img, dict) and img.get("type") == "image_url":
            image_url = img.get("image_url") or {}
            source = create_image_source_from_url(image_url.get("url", ""))
            if source:
                blocks.append(ImageBlock(source=source))
    if lane.audio_data:
        blocks.append(
            AudioBlock(
                source=AudioSource(
                    type="base64",
                    data=lane.audio_data,
                    media_type=lane.audio_media_type or "audio/wav",
                    transcript=lane.audio_transcript or None,
                )
            )
        )
    return blocks


class SegmentAccumulator:
    """Assembles canonical content blocks from a streaming turn's segments.

    The single open segment per kind is the machine's core invariant: a delta
    of one kind closes the other kinds' open lanes (in canonical content
    order) and appends to its own. Tool lanes may pool (parallel) or evict
    their predecessor (sequential) per policy. Closed blocks are observable
    at any moment through :meth:`blocks` — including mid-stream, which the
    web-search continuation and the truncated-stream error path rely on —
    with the still-open text/reasoning/refusal segments projected in their
    current state. Tool and media lanes appear only once closed: a partial
    tool-call has no faithful block form, and partial media is a growing
    base64 blob, not a block.

    ``finish()`` is idempotent and non-latching: feeding after it re-opens
    buffering and appends. Nothing a caller feeds after a terminal event is
    lost, and nothing on the assembly path raises (ADR-0031).
    """

    def __init__(self, policy: SegmentPolicy | None = None) -> None:
        self._policy = policy or SegmentPolicy()
        self._closed: list[ContentBlock] = []
        self._open_text: _TextLane | None = None
        self._open_reasoning: _ReasoningLane | None = None
        self._open_refusal: _TextLane | None = None
        self._open_media: _MediaLane | None = None
        # Parallel: pooled by provider index; sequential: the single open call.
        self._open_tools: dict[int, _ToolLane] = {}
        self._open_tool: _ToolLane | None = None
        # An encrypted reasoning blob that arrived before (or without) any
        # reasoning lane: held for the next one, never opening a lane itself.
        self._pending_encrypted: str | None = None

    # ── feeding ───────────────────────────────────────────────────────────

    def feed(self, delta: dict[str, Any]) -> tuple[SegmentEvent, ...]:
        """Lower one canonical chat-completions delta and feed its fragments.

        The lowering is canonical-dialect knowledge (the alias pair
        ``reasoning``/``reasoning_content``, content parts, tool calls,
        OpenRouter's images, audio and refusal fields) — not protocol
        knowledge, so every chat-shaped consumer shares it. Malformed members
        are skipped: the proxy is a faithful wire and never rejects a stream
        over one bad fragment.

        Returns the ordered segment events; callers that only accumulate
        (Chat Completions) ignore them, wire-driving adapters render them.
        """
        if not isinstance(delta, dict):
            return ()
        events: list[SegmentEvent] = []
        for fragment in self._lower(delta):
            events.extend(self.feed_delta(fragment))
        return tuple(events)

    def feed_delta(self, delta: CanonicalDelta) -> tuple[SegmentEvent, ...]:
        """Feed one pre-extracted delta (item-addressed callers use this)."""
        events: list[SegmentEvent] = []
        if delta.close:
            events.extend(self._close_kind(delta.kind))
        # A delta of one kind closes the other kinds' open lanes, in canonical
        # content order — the one eviction rule all the per-protocol machines
        # shared, now stated once. Media is the exception: every current
        # machine buffers it for the whole turn (OpenRouter interleaves image
        # deltas with text), so it never evicts and is never evicted; it
        # closes only at finish().
        if delta.kind != "media" and self._policy.whole_turn_close_order is None:
            for kind in _CLOSE_ORDER:
                if kind not in (delta.kind, "media"):
                    events.extend(self._close_kind(kind))
        match delta.kind:
            case "reasoning":
                events.extend(self._feed_reasoning(delta))
            case "text":
                events.extend(self._feed_text(delta))
            case "tool":
                events.extend(self._feed_tool(delta))
            case "media":
                events.extend(self._feed_media(delta))
            case "refusal":
                events.extend(self._feed_refusal(delta))
        return tuple(events)

    # ── closing and reading ───────────────────────────────────────────────

    def finish(self) -> tuple[SegmentEvent, ...]:
        """Close every non-empty lane, in the turn's close order.

        Idempotent by construction (empty lanes produce nothing) and
        non-latching: feeding after ``finish()`` re-opens buffering and
        appends after the finalized blocks.
        """
        events: list[SegmentEvent] = []
        for kind in self._policy.whole_turn_close_order or _CLOSE_ORDER:
            events.extend(self._close_kind(kind))
        return tuple(events)

    def blocks(self) -> list[ContentBlock]:
        """The canonical blocks: every closed segment, plus open projections.

        Tool and media lanes project nothing until they close (a partial
        tool call has no faithful block form; partial media is a growing
        base64 blob). The returned list is a copy — replacing a closed block
        is :meth:`supersede`'s job, not the caller's.
        """
        result = list(self._closed)
        if self._open_text and self._open_text.text:
            result.append(TextBlock(text=self._open_text.text))
        lane = self._open_reasoning
        if lane and (lane.text or lane.signature or lane.details or lane.encrypted_content):
            result.append(_build_reasoning_block(lane))
        if self._open_refusal and self._open_refusal.text:
            result.append(RefusalBlock(refusal=self._open_refusal.text))
        return result

    def cursor(self) -> int | None:
        """The next free block index under ``NEXT_FREE``, else ``None``.

        Result blocks emitted at explicit indices never move the cursor: it
        counts what this engine closed, nothing else.
        """
        if self._policy.cursor is CursorPolicy.NEXT_FREE:
            return len(self._closed)
        return None

    def supersede(self, index: int, block: ContentBlock) -> None:
        """Replace a closed block — the one mutation a caller may perform.

        Named for the web-search interception's ``ToolUseBlock`` →
        ``ServerToolUseBlock`` substitution (ADR-0007): the act is part of the
        interface, so it is typed, testable, and cannot be mistaken for
        reaching into the engine's internals. An out-of-range index is a
        programming error and raises.
        """
        if not 0 <= index < len(self._closed):
            raise IndexError(f"no closed block at index {index}")
        self._closed[index] = block

    def adopt(self, block: ContentBlock) -> tuple[SegmentEvent, ...]:
        """Close every open lane, then append a pre-built block as closed.

        For protocol-level blocks that live outside the segment vocabulary —
        Anthropic's ``raw_content_block`` replay (server tool results the
        converter ships losslessly): the block lands after whatever the open
        lanes close into, so the canonical order matches the wire order.
        The returned events cover the lanes the adoption closed; the adopted
        block itself is not announced (the caller emits its own wire pair).
        """
        events: list[SegmentEvent] = []
        for kind in self._policy.whole_turn_close_order or _CLOSE_ORDER:
            events.extend(self._close_kind(kind))
        self._closed.append(block)
        return tuple(events)

    # ── the canonical-dialect lowering ────────────────────────────────────

    def _lower(self, delta: dict[str, Any]) -> list[CanonicalDelta]:
        """Extract one canonical delta's segment fragments, in content order."""
        fragments: list[CanonicalDelta] = []

        reasoning = delta.get("reasoning_content")
        if not isinstance(reasoning, str) or not reasoning:
            alternate = delta.get("reasoning")
            reasoning = alternate if isinstance(alternate, str) and alternate else ""
        signature = delta.get("reasoning_signature")
        encrypted = delta.get("encrypted_content")
        details = delta.get("reasoning_details")
        has_signature = isinstance(signature, str) and bool(signature)
        has_encrypted = isinstance(encrypted, str) and bool(encrypted)
        has_details = isinstance(details, list) and bool(details)
        if reasoning or has_signature or has_encrypted or has_details:
            origin = delta.get("signature_origin")
            fragments.append(
                CanonicalDelta(
                    kind="reasoning",
                    text=reasoning,
                    signature=signature if has_signature else None,
                    signature_origin=origin if origin in ("gemini", "openai") else None,
                    encrypted_content=encrypted if has_encrypted else None,
                    is_redacted=bool(delta.get("reasoning_is_redacted")),
                    reasoning_details=tuple(d for d in details if isinstance(d, dict))
                    if has_details
                    else None,
                )
            )

        content = delta.get("content")
        if content is not None:
            text_parts: list[str] = []
            image_parts: list[str] = []
            if isinstance(content, str):
                if content:
                    text_parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "text":
                        text = part.get("text", "")
                        if text:
                            text_parts.append(text)
                    elif part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        if url:
                            image_parts.append(f"![image]({url})")
            if text_parts or image_parts:
                fragments.append(
                    CanonicalDelta(
                        kind="text",
                        text="".join(text_parts),
                        markdown="\n".join(image_parts),
                    )
                )

        tool_calls = delta.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                index = tc.get("index", 0)
                index = index if isinstance(index, int) and not isinstance(index, bool) else 0
                custom_raw = tc.get("custom")
                custom: dict[str, Any] | None = custom_raw if isinstance(custom_raw, dict) else None
                function = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                variant: Literal["function", "custom"]
                if custom is not None or tc.get("type") == "custom":
                    variant = "custom"
                else:
                    variant = "function"
                if (
                    variant == "custom"
                    and custom is not None
                    and isinstance(custom.get("input"), str)
                ):
                    args = custom["input"]
                elif isinstance(function.get("arguments"), str):
                    args = function["arguments"]
                else:
                    args = ""
                name = (custom or {}).get("name") or function.get("name") or None
                fragments.append(
                    CanonicalDelta(
                        kind="tool",
                        tool_index=index,
                        tool_id=tc.get("id") if isinstance(tc.get("id"), str) else None,
                        tool_name=name if isinstance(name, str) else None,
                        tool_variant=variant,
                        server_tool=bool(tc.get("server_tool_use")),
                        text=args,
                        thought_signature=tc.get("thought_signature")
                        if isinstance(tc.get("thought_signature"), str)
                        else None,
                    )
                )

        images = delta.get("images")
        if isinstance(images, list):
            for img in images:
                if isinstance(img, dict):
                    fragments.append(CanonicalDelta(kind="media", payload={"image": img}))

        audio = delta.get("audio")
        if isinstance(audio, dict):
            audio_data = audio.get("data")
            audio_transcript = audio.get("transcript")
            audio_media_type = audio.get("mime_type")
            audio_data = audio_data if isinstance(audio_data, str) else ""
            audio_transcript = audio_transcript if isinstance(audio_transcript, str) else ""
            audio_media_type = audio_media_type if isinstance(audio_media_type, str) else ""
            if audio_data or audio_transcript:
                fragments.append(
                    CanonicalDelta(
                        kind="media",
                        payload={
                            "audio_data": audio_data,
                            "audio_transcript": audio_transcript,
                            "audio_media_type": audio_media_type,
                        },
                    )
                )

        refusal = delta.get("refusal")
        if isinstance(refusal, str) and refusal:
            fragments.append(CanonicalDelta(kind="refusal", text=refusal))

        return fragments

    # ── per-kind feeding ──────────────────────────────────────────────────

    def _feed_reasoning(self, delta: CanonicalDelta) -> list[SegmentEvent]:
        lane = self._open_reasoning
        events: list[SegmentEvent] = []
        if (
            lane is not None
            and not lane.is_redacted
            and delta.is_redacted
            and (delta.text or delta.signature or delta.reasoning_details)
        ):
            # A redacted marker landing in a non-redacted lane is a *new*
            # upstream block (Anthropic's thinking → redacted_thinking), not a
            # continuation: close the open lane rather than merging, so the
            # canonical list keeps two blocks and the wire gets its boundary.
            events.extend(self._close_reasoning())
            lane = None
        if lane is None:
            # An encrypted blob never opens a lane on its own; it waits for
            # the next reasoning fragment to join (ADR-0031).
            if not (delta.text or delta.signature or delta.reasoning_details):
                if delta.encrypted_content:
                    self._pending_encrypted = delta.encrypted_content
                return events
            lane = _ReasoningLane()
            if self._pending_encrypted is not None:
                lane.encrypted_content = self._pending_encrypted
                self._pending_encrypted = None
            self._open_reasoning = lane
            events.append(SegmentOpened(kind="reasoning", index=self._index(), delta=delta))
        if delta.text:
            lane.text += delta.text
        if delta.signature:
            lane.signature += delta.signature
        if delta.signature_origin:
            lane.signature_origin = delta.signature_origin
        if delta.encrypted_content:
            # A delta's own payload wins over a held one (the machine's rule).
            lane.encrypted_content = delta.encrypted_content
        if delta.is_redacted:
            lane.is_redacted = True
        if delta.reasoning_details:
            lane.details = list(delta.reasoning_details)
        events.append(SegmentAppended(kind="reasoning", index=self._index(), delta=delta))
        return events

    def _feed_text(self, delta: CanonicalDelta) -> list[SegmentEvent]:
        lane = self._open_text
        events: list[SegmentEvent] = []
        if lane is None:
            lane = _TextLane()
            self._open_text = lane
            events.append(SegmentOpened(kind="text", index=self._index(), delta=delta))
        if delta.text:
            lane.text += delta.text
        if delta.markdown:
            lane.text += ("\n" + delta.markdown) if lane.text else delta.markdown
        events.append(SegmentAppended(kind="text", index=self._index(), delta=delta))
        return events

    def _feed_tool(self, delta: CanonicalDelta) -> list[SegmentEvent]:
        events: list[SegmentEvent] = []
        if self._policy.tool_coexistence is Coexistence.SEQUENTIAL:
            lane = self._open_tool
            if lane is not None and delta.tool_id and delta.tool_name:
                # A new call starts: the sequential wire closes the open one.
                events.extend(self._close_tool_sequential())
                lane = None
            if lane is None:
                lane = _ToolLane(index=0, variant=self._variant_for(delta))
                self._open_tool = lane
                events.append(SegmentOpened(kind="tool", index=self._index(), delta=delta))
            self._merge_tool_lane(lane, delta)
            events.append(SegmentAppended(kind="tool", index=self._index(), delta=delta))
            return events

        lane = self._open_tools.get(delta.tool_index)
        if lane is None:
            lane = _ToolLane(index=delta.tool_index, variant=self._variant_for(delta))
            self._open_tools[delta.tool_index] = lane
            events.append(SegmentOpened(kind="tool", index=self._index(), delta=delta))
        self._merge_tool_lane(lane, delta)
        events.append(SegmentAppended(kind="tool", index=self._index(), delta=delta))
        return events

    def _variant_for(self, delta: CanonicalDelta) -> ToolVariant:
        """Classify a tool call at open (ADR-0031: decide-at-open).

        The policy's classifier wins when declared; the structural default
        keeps the Chat Completions machine's rule (custom deltas make custom
        blocks, the server marker is ignored).
        """
        if self._policy.classify_tool is not None:
            return self._policy.classify_tool(delta)
        return delta.tool_variant or "function"

    @staticmethod
    def _merge_tool_lane(lane: _ToolLane, delta: CanonicalDelta) -> None:
        if delta.tool_id:
            lane.id = delta.tool_id
        if delta.tool_name:
            lane.name = delta.tool_name
        # The variant is decided once, at open (``_variant_for``); later
        # fragments of the same call carry the structural default and must
        # not clobber the classifier's verdict.
        if delta.text:
            lane.args += delta.text
        if delta.thought_signature:
            lane.thought_signature = delta.thought_signature

    def _feed_media(self, delta: CanonicalDelta) -> list[SegmentEvent]:
        lane = self._open_media
        events: list[SegmentEvent] = []
        if lane is None:
            lane = _MediaLane()
            self._open_media = lane
            events.append(SegmentOpened(kind="media", index=self._index(), delta=delta))
        payload = delta.payload or {}
        image = payload.get("image")
        if isinstance(image, dict) and image not in lane.images:
            lane.images.append(image)
        if isinstance(payload.get("audio_data"), str) and payload["audio_data"]:
            lane.audio_data += payload["audio_data"]
        if isinstance(payload.get("audio_transcript"), str) and payload["audio_transcript"]:
            lane.audio_transcript += payload["audio_transcript"]
        if isinstance(payload.get("audio_media_type"), str) and payload["audio_media_type"]:
            lane.audio_media_type = payload["audio_media_type"]
        events.append(SegmentAppended(kind="media", index=self._index(), delta=delta))
        return events

    def _feed_refusal(self, delta: CanonicalDelta) -> list[SegmentEvent]:
        lane = self._open_refusal
        events: list[SegmentEvent] = []
        if lane is None:
            lane = _TextLane()
            self._open_refusal = lane
            events.append(SegmentOpened(kind="refusal", index=self._index(), delta=delta))
        lane.text += delta.text
        events.append(SegmentAppended(kind="refusal", index=self._index(), delta=delta))
        return events

    # ── closing ───────────────────────────────────────────────────────────

    def _close_kind(self, kind: str) -> list[SegmentEvent]:
        match kind:
            case "reasoning":
                return self._close_reasoning()
            case "text":
                return self._close_text()
            case "refusal":
                return self._close_refusal()
            case "tool":
                return self._close_tools()
            case "media":
                return self._close_media()
        return []

    def _close_reasoning(self) -> list[SegmentEvent]:
        lane = self._open_reasoning
        if lane is None:
            return []
        self._open_reasoning = None
        if not (lane.text or lane.signature or lane.details or lane.encrypted_content):
            return []
        block = _build_reasoning_block(lane)
        index = self._index()
        self._closed.append(block)
        return [SegmentClosed(kind="reasoning", index=index, blocks=(block,))]

    def _close_text(self) -> list[SegmentEvent]:
        lane = self._open_text
        if lane is None or not lane.text:
            self._open_text = None
            return []
        self._open_text = None
        block = TextBlock(text=lane.text)
        index = self._index()
        self._closed.append(block)
        return [SegmentClosed(kind="text", index=index, blocks=(block,))]

    def _close_refusal(self) -> list[SegmentEvent]:
        lane = self._open_refusal
        if lane is None or not lane.text:
            self._open_refusal = None
            return []
        self._open_refusal = None
        block = RefusalBlock(refusal=lane.text)
        index = self._index()
        self._closed.append(block)
        return [SegmentClosed(kind="refusal", index=index, blocks=(block,))]

    def _close_tools(self) -> list[SegmentEvent]:
        if self._policy.tool_coexistence is Coexistence.SEQUENTIAL:
            return self._close_tool_sequential()
        events: list[SegmentEvent] = []
        for tool_index in sorted(self._open_tools):
            lane = self._open_tools.pop(tool_index)
            block = _build_tool_block(lane)
            if block is None:
                continue
            index = self._index()
            self._closed.append(block)
            events.append(SegmentClosed(kind="tool", index=index, blocks=(block,)))
        return events

    def _close_tool_sequential(self) -> list[SegmentEvent]:
        lane = self._open_tool
        if lane is None:
            return []
        self._open_tool = None
        block = _build_tool_block(lane)
        if block is None:
            return []
        index = self._index()
        self._closed.append(block)
        return [SegmentClosed(kind="tool", index=index, blocks=(block,))]

    def _close_media(self) -> list[SegmentEvent]:
        lane = self._open_media
        if lane is None:
            return []
        self._open_media = None
        blocks = tuple(_build_media_blocks(lane))
        if not blocks:
            return []
        index = self._index()
        self._closed.extend(blocks)
        return [SegmentClosed(kind="media", index=index, blocks=blocks)]

    # ── cursor bookkeeping ────────────────────────────────────────────────

    def _index(self) -> int | None:
        """The next free block index under ``NEXT_FREE``, else ``None``.

        Always measured *before* an append: for a segment opening it is the
        index the block will occupy, for a closing lane the index the
        block(s) about to land occupy. One site, one meaning — the open and
        close events can be paired on it.
        """
        if self._policy.cursor is CursorPolicy.NEXT_FREE:
            return len(self._closed)
        return None
