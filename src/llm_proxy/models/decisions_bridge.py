"""Bridge between the Decisions and System One evaluation shapes.

Decisions (OpenAI's ``/v1/decisions``) and System One (TypeSafe's Jev, also
served by OpenRouter and Ollama at ``/v1/systemone``) answer the same three
primitives under different names and different envelopes:

==================  ==========================================  ===============================
                    Decisions                                   System One
==================  ==========================================  ===============================
Evidence            ``input``: string or user messages          ``state``: string, object or array
Questions           ordered array, each with ``name``           map of id to question
Yes/no              ``predicate`` (probability)                 ``noul`` (probability)
Question type       ``predicate`` | ``choice`` | ``score``    ``noul`` | ``choice`` | ``score``
Pick one            ``choices[{value, desc}]``                 ``criteria{value: desc}``
Rate                ``levels[{label, desc}]``                   ``criteria[level, ...]``
Answers             ordered array, echoing ``name``             map keyed by question id
==================  ==========================================  ===============================

A provider that speaks one envelope can therefore serve the other endpoint by
converting here. The conversions are deliberately as lossless as the two
envelopes allow, and every lossy step is marked ``# lossy`` with the reason:

* Decisions ``predicate`` has no rubric field, so a ``noul`` rubric
  (``criteria.true``/``criteria.false``) is folded into the instructions text.
* System One score levels are plain descriptions while Decisions levels carry a
  ``label`` *and* a description, so the descriptions are folded into the
  instructions as a numbered rubric and the labels stay the level strings — the
  answer labels the client sees are still exactly the ones it sent.
* Decisions ``input`` may hold ``input_image`` parts, and how those reach a
  System One upstream depends on the channel that upstream documents: Ollama
  takes a raw-base64 ``images`` array shared by every question, OpenRouter takes
  ``image_url`` parts inside ``state`` (and inside a question's instructions),
  and TypeSafe takes an ``assets`` map whose placeholder strings sit in the
  ``state`` and instruction text. An upstream that documents no channel gets the
  text alone.
* Decisions ``input`` may also be a conversation — developer and system messages,
  assistant output and refusals, function and custom tool calls with their
  outputs — while System One evaluates one ``state``. Conversation items are
  rendered as text in request order; tool items become their own JSON.
* System One ``state`` may be an object or array, while Decisions ``input`` is
  a string or a list of messages: the object is serialized to JSON text.
* The optional fields each envelope has and the other does not
  (``safety_identifier``/``prompt_cache_key``/``prompt_cache_options``;
  ``provider``/``session_id``/``trace``/``user``/``keep_alive``) have no
  counterpart and are dropped.

Both conversions preserve ``model``, the pipeline's shared fields and the
override markers so a bridged request is still field-policy-correct.
"""

import base64
import json
from typing import Any

from llm_proxy.core.exceptions import ValidationError
from llm_proxy.models.decisions import (
    InternalDecisionRequest,
    InternalDecisionResponse,
)
from llm_proxy.models.systemone import (
    InternalSystemOneRequest,
    InternalSystemOneResponse,
    SystemOneImageChannel,
)

#: System One keys named questions in a map, so a Decisions question without a
#: ``name`` needs a synthetic key. The prefix is stripped again on the way back,
#: and such an answer carries ``name: null`` (the official response shape), so
#: the client never sees a name it did not send. Because a client-named question
#: with this prefix would collide with the synthetic key of an unnamed one (one
#: question silently lost), the Decisions schema rejects the prefix; it lives
#: here, next to the synthesis it protects.
UNNAMED_KEY_PREFIX = "__unnamed_question_"


def _unnamed_key(index: int) -> str:
    return f"{UNNAMED_KEY_PREFIX}{index}"


def _json_text(value: Any) -> str:
    """Serialize a structured value to compact JSON text."""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _question_name(question: dict[str, Any] | None) -> str | None:
    name = (question or {}).get("name")
    return name if isinstance(name, str) and name else None


def choice_key(value: Any) -> str:
    """Key a Decisions choice value for a System One ``criteria`` map.

    Decisions types its choice values (``true`` and ``"true"`` are distinct),
    while a criteria map is keyed by strings. Booleans therefore become their
    lowercase words and are restored to booleans on the way back by matching
    the request's own options — so the round trip is typed, not stringly.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    return value if isinstance(value, str) else str(value)


def _base64_from_data_url(value: Any) -> str | None:
    """Extract the base64 payload of an inline ``data:`` image URL."""
    split = _split_data_url(value)
    return split[1] if split else None


# --------------------------------------------------------------------------- #
# Decisions -> System One
# --------------------------------------------------------------------------- #


#: Prefix of the placeholder strings TypeSafe substitutes for an image asset.
#: TypeSafe recommends an opaque per-image id; a fixed prefix keeps the body
#: deterministic (and therefore testable) while staying clear of ordinary text.
ASSET_PLACEHOLDER_PREFIX = "llm-proxy-image-"

#: TypeSafe's asset format enum. Its image channel accepts PNG and JPEG only.
_ASSET_FORMATS: frozendict[str, str] = frozendict({"image/png": "png", "image/jpeg": "jpeg"})

#: One rendered piece of Decisions content: ``("text", text)`` or
#: ``("image", url)``, in request order.
_Segment = tuple[str, str]


class _ImageSink:
    """Collect the images of one bridged request for a System One image channel.

    The channel decides where an image goes, not whether it survives:
    ``state_parts`` and ``assets`` place it inline (so its position relative to
    the text is kept), ``images`` collects it in the shared array Ollama reads,
    and ``None`` drops it because the upstream documents no channel at all.
    """

    def __init__(self, channel: SystemOneImageChannel | None) -> None:
        self.channel = channel
        self.images: list[str] = []
        self.assets: dict[str, Any] = {}

    def takes_inline(self) -> bool:
        """Whether this channel places an image where it appears in the text."""
        return self.channel in ("state_parts", "assets")

    def inline(self, data_url: str) -> Any:
        """Take one image, returning the element to place inline for it.

        Returns ``None`` for the channels that do not carry images inline: the
        shared-array channel has collected the payload already, and a channel-less
        upstream drops it.
        """
        if self.channel == "state_parts":
            return {"type": "image_url", "image_url": {"url": data_url}}
        if self.channel == "assets":
            placeholder, asset = _asset_for(data_url, len(self.assets) + 1)
            self.assets[placeholder] = asset
            return placeholder
        if self.channel == "images":
            payload = _base64_from_data_url(data_url)
            if payload is not None:
                self.images.append(payload)
        return None


def _asset_for(data_url: str, index: int) -> tuple[str, dict[str, Any]]:
    """Build one TypeSafe asset plus the placeholder that stands in for it."""
    split = _split_data_url(data_url)
    if split is None:
        raise ValidationError(
            message=(
                "TypeSafe image assets take inline base64 data URLs; "
                f"got '{data_url[:32]}'. Inline the image or use a provider with "
                "a wider image channel."
            ),
            code="invalid_request_error",
            status_code=400,
        )
    media_type, payload = split
    asset_format = _ASSET_FORMATS.get(media_type)
    if asset_format is None:
        raise ValidationError(
            message=(
                f"TypeSafe image assets accept PNG and JPEG only; got '{media_type}'. "
                "Re-encode the image or use a provider with a wider image channel."
            ),
            code="invalid_request_error",
            status_code=400,
        )
    return f"{ASSET_PLACEHOLDER_PREFIX}{index}", {
        "type": "image",
        "format": asset_format,
        "data_base64": payload,
    }


def _split_data_url(value: Any) -> tuple[str, str] | None:
    """Split an inline base64 data URL into its media type and payload."""
    if not isinstance(value, str) or not value.startswith("data:"):
        return None
    header, separator, payload = value.partition(",")
    if not separator or ";base64" not in header or not payload:
        return None
    media_type = header.removeprefix("data:").split(";", 1)[0]
    return media_type or "image/png", payload


def _content_segments(parts: Any) -> list[_Segment]:
    """Render one content-part array (a message's, or an output's) into segments."""
    if not isinstance(parts, list):
        return []
    segments: list[_Segment] = []
    for part in parts:
        if isinstance(part, str):
            segments.append(("text", part))
            continue
        if not isinstance(part, dict):
            segments.append(("text", _json_text(part)))
            continue
        kind = part.get("type")
        match kind:
            case "input_text" | "output_text" if isinstance(part.get("text"), str):
                segments.append(("text", part["text"]))
            case "refusal" if isinstance(part.get("refusal"), str):
                segments.append(("text", part["refusal"]))
            case "input_json":
                segments.append(("text", _json_text(part.get("data"))))
            case "input_image" if isinstance(part.get("image_url"), str):
                segments.append(("image", part["image_url"]))
            case _:
                segments.append(("text", _json_text(part)))
    return segments


def _decision_input_segments(payload: Any) -> list[_Segment]:
    """Flatten Decisions ``input`` into ordered text and image segments.

    A string input is the evidence verbatim. A conversation is rendered in
    request order: message content as its parts read, and tool calls and their
    outputs as their own JSON — ``# lossy``, since System One evaluates one
    ``state`` and has no field for a call or an output.
    """
    if isinstance(payload, str):
        return [("text", payload)]
    if not isinstance(payload, list):
        # The schema admits a string or a conversation only; anything else is
        # rendered as its JSON rather than dropped.
        return [("text", _json_text(payload))]

    segments: list[_Segment] = []
    for item in payload:
        if not isinstance(item, dict):
            segments.append(("text", _json_text(item)))
            continue
        content = item.get("content")
        if isinstance(content, str | list):
            segments.extend(
                [("text", content)] if isinstance(content, str) else _content_segments(content)
            )
        elif isinstance(item.get("output"), str | list):
            output = item["output"]
            segments.extend(
                [("text", output)] if isinstance(output, str) else _content_segments(output)
            )
        else:
            segments.append(("text", _json_text(item)))
    return segments


def _render_segments(segments: list[_Segment], sink: _ImageSink) -> Any:
    """Render segments for the upstream's channel.

    Text-only content stays a string, which every upstream accepts. When the
    content carries images and the channel places them inline, the result is an
    array with the text as plain strings and the images between them, in request
    order.
    """
    texts = [value for kind, value in segments if kind == "text"]
    if not any(kind == "image" for kind, _ in segments):
        return "\n\n".join(texts)
    if not sink.takes_inline():
        # The shared-array and channel-less upstreams keep the text alone; the
        # former still needs each image offered to the sink for collection.
        for kind, value in segments:
            if kind == "image":
                sink.inline(value)
        return "\n\n".join(texts)
    rendered: list[Any] = []
    for kind, value in segments:
        if kind == "text":
            if value:
                rendered.append(value)
        else:
            rendered.append(sink.inline(value))
    return rendered


def _render_instructions(instructions: Any, sink: _ImageSink) -> Any:
    """Render a question's instructions for the upstream's channel.

    A string passes through. A part array — the form that lets one question carry
    its own image — is rendered like message content: text joins into one string
    on a text-only channel, and stays an array with the images in place on a
    channel that carries them inline.
    """
    if isinstance(instructions, str):
        return instructions
    if not isinstance(instructions, list):
        # The schema admits a string or a part array; anything else is passed
        # through rather than dropped.
        return instructions
    return _render_segments(_content_segments(instructions), sink)


def _score_instructions(instructions: Any, levels: list[dict[str, Any]]) -> Any:
    """Fold Decisions level descriptions into the System One instructions.

    System One score criteria are the level strings themselves (the upstream
    echoes them back as the answer legend), so the descriptions have nowhere to
    live except the question text. ``# lossy``: placement only — no wording is
    discarded, and the labels the client sent come back unchanged.
    """
    if not any(
        isinstance(level.get("description"), str) and level["description"] for level in levels
    ):
        return instructions
    lines = []
    for index, level in enumerate(levels):
        label = level.get("label")
        description = level.get("description")
        if isinstance(description, str) and description:
            lines.append(f"{index}. {label} — {description}")
        else:
            lines.append(f"{index}. {label}")
    rubric = "Levels, lowest to highest:\n" + "\n".join(lines)
    if isinstance(instructions, list):
        # Instructions that carry their own image stay an array; the rubric is
        # one more text block at the end.
        return [*instructions, rubric]
    return f"{instructions}\n\n{rubric}"


def _decision_question_to_systemone(question: dict[str, Any], sink: _ImageSink) -> dict[str, Any]:
    """Convert one Decisions question into its System One form."""
    kind = question.get("type")
    instructions = _render_instructions(question.get("instructions"), sink)
    if kind == "predicate":
        # lossy: Decisions predicates carry no rubric, so the ``noul`` criteria
        # map is omitted rather than invented.
        return {"type": "noul", "instructions": instructions}
    if kind == "choice":
        criteria: dict[str, Any] = {}
        for option in question.get("choices") or []:
            if not isinstance(option, dict):
                continue
            criteria[choice_key(option.get("value"))] = option.get("description")
        return {"type": "choice", "instructions": instructions, "criteria": criteria}
    if kind == "score":
        levels = [level for level in question.get("levels") or [] if isinstance(level, dict)]
        return {
            "type": "score",
            "instructions": _score_instructions(instructions, levels),
            "criteria": [level.get("label") for level in levels],
        }
    # A question type System One cannot express: pass it through unchanged so an
    # upstream that understands the type still receives the question.
    return dict(question)


def decisions_to_systemone_request(
    request: InternalDecisionRequest,
    image_channel: SystemOneImageChannel | None = None,
) -> InternalSystemOneRequest:
    """Convert a Decisions request into the System One request that answers it.

    ``image_channel`` is the target upstream's own declaration (see
    :data:`~llm_proxy.models.systemone.SystemOneImageChannel`): it decides whether
    an image rides ``images``, ``state`` parts or ``assets``, and a channel-less
    upstream receives the text alone.
    """
    sink = _ImageSink(image_channel)
    state = _render_segments(_decision_input_segments(request.input), sink)
    questions: dict[str, Any] = {}
    for index, question in enumerate(request.questions):
        if not isinstance(question, dict):
            continue
        name = _question_name(question)
        questions[name or _unnamed_key(index)] = _decision_question_to_systemone(question, sink)
    extra: dict[str, Any] = {}
    if sink.images:
        extra["images"] = sink.images
    if sink.assets:
        extra["assets"] = sink.assets
    return InternalSystemOneRequest(
        model=request.model,
        state=state,
        questions=questions,
        extra=extra,
        metadata=request.metadata,
        params=request.params,
        _override_injected_keys=set(request._override_injected_keys),
        user_facing_model=request.user_facing_model,
    )


def _question_for_answer(
    questions: list[Any], key: str, index: int
) -> tuple[dict[str, Any] | None, str | None]:
    """Find the Decisions question an answer belongs to.

    Prefers the answer key (which is the question's ``name``, or the synthetic
    key for a nameless question); the positional fallback only serves leftover
    answers — an upstream that renamed its keys but kept the question order, in
    which case the key it used is echoed as the name so the client can still
    correlate. Question order itself is owned by
    ``systemone_to_decisions_response``, which maps answers onto questions by
    key.
    """
    for position, question in enumerate(questions):
        if not isinstance(question, dict):
            continue
        name = _question_name(question)
        if (name or _unnamed_key(position)) == key:
            return question, name
    if 0 <= index < len(questions) and isinstance(questions[index], dict):
        return questions[index], key
    return None, key


def _typed_choice_value(question: dict[str, Any] | None, value: Any) -> Any:
    """Restore a System One choice key to the typed value the client sent."""
    for option in (question or {}).get("choices") or []:
        if isinstance(option, dict) and choice_key(option.get("value")) == value:
            return option.get("value")
    return value


def _choice_probabilities(question: dict[str, Any] | None, raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, dict):
        return []
    return [
        {"value": _typed_choice_value(question, key), "probability": probability}
        for key, probability in raw.items()
    ]


def _score_probabilities(
    question: dict[str, Any] | None, raw: Any, legend: Any
) -> list[dict[str, Any]]:
    """Convert System One's index-keyed score distribution into Decisions' array.

    Labels come from the question the client sent — it defined them, so they are
    authoritative — and only fall back to the upstream's legend or the raw index
    when the upstream answered a level the question did not declare.
    """
    if not isinstance(raw, dict):
        return []
    levels = [level for level in (question or {}).get("levels") or [] if isinstance(level, dict)]
    legend = legend if isinstance(legend, dict) else {}
    probabilities: list[dict[str, Any]] = []
    for key, probability in raw.items():
        try:
            value = int(key)
        except TypeError, ValueError:
            continue
        label: str | None = None
        if 0 <= value < len(levels):
            declared = levels[value].get("label")
            if isinstance(declared, str):
                label = declared
        if label is None:
            reported = legend.get(str(key))
            label = reported if isinstance(reported, str) else str(key)
        probabilities.append({"value": value, "label": label, "probability": probability})
    return probabilities


def _systemone_answer_to_decision(
    raw: Any, question: dict[str, Any] | None, name: str | None
) -> dict[str, Any]:
    """Convert one System One answer into its Decisions form.

    ``name`` is always written, ``None`` for an unnamed question: the official
    response contract requires ``name`` on every completed answer, null when
    unnamed, while the question itself stays nameless.
    """
    if not isinstance(raw, dict):
        # A bare scalar: System One types its answers, so this is a predicate
        # probability (a number) or a choice value (a string) from an upstream
        # that omitted the answer envelope.
        numeric = isinstance(raw, (int, float)) and not isinstance(raw, bool)
        answer = (
            {"type": "predicate", "probability": raw}
            if numeric
            else {"type": "choice", "choice": raw}
        )
    elif raw.get("type") == "noul":
        answer = {"type": "predicate", "probability": raw.get("noul")}
    elif raw.get("type") == "choice":
        answer = {
            "type": "choice",
            "choice": _typed_choice_value(question, raw.get("choice")),
            "probabilities": _choice_probabilities(question, raw.get("probabilities")),
        }
        if raw.get("confidence") is not None:
            answer["confidence"] = raw["confidence"]
    elif raw.get("type") == "score":
        answer = {
            "type": "score",
            "score": raw.get("score"),
            "probabilities": _score_probabilities(
                question, raw.get("probabilities"), raw.get("legend")
            ),
        }
        if raw.get("confidence") is not None:
            answer["confidence"] = raw["confidence"]
    else:
        # Unknown or already-Decisions answer shape: pass it through untouched
        # rather than dropping a question's answer.
        answer = dict(raw)
    answer["name"] = name
    return answer


def systemone_to_decisions_response(
    response: InternalSystemOneResponse, request: InternalDecisionRequest
) -> InternalDecisionResponse:
    """Convert the System One answers to a Decisions response.

    The Decisions wire promises ordered answers — one per ``questions`` entry,
    in the same order — so the conversion walks the client's questions and
    looks up each answer by its key. Upstream answers that no question claims
    (renamed or extra keys) are appended afterwards in upstream order so no
    answer is silently dropped; the positional fallback in
    ``_question_for_answer`` correlates them for an upstream that kept order
    but renamed its keys.

    ``systemone_usage`` is dropped: it is the shape of the System One wire (a
    flat input/output pair, plus OpenRouter's ``cost``), not of the Decisions
    usage object the protocol formatter has to emit.
    """
    answers: list[dict[str, Any]] = []
    claimed_keys: set[str] = set()
    for position, question in enumerate(request.questions):
        if not isinstance(question, dict):
            continue
        key = _question_name(question) or _unnamed_key(position)
        raw = response.answers.get(key)
        if raw is None:
            continue
        claimed_keys.add(key)
        answers.append(_systemone_answer_to_decision(raw, question, _question_name(question)))
    for index, (key, raw) in enumerate(response.answers.items()):
        if key in claimed_keys:
            continue
        question, name = _question_for_answer(request.questions, key, index)
        answers.append(_systemone_answer_to_decision(raw, question, name))
    return InternalDecisionResponse(
        model=response.model,
        answers=answers,
        usage=response.usage,
        id=response.id,
        provider=response.provider,
        request_id=response.request_id,
        provider_info={
            key: value for key, value in response.provider_info.items() if key != "systemone_usage"
        },
    )


# --------------------------------------------------------------------------- #
# System One -> Decisions
# --------------------------------------------------------------------------- #


def _noul_instructions(instructions: str, criteria: Any) -> str:
    """Fold a ``noul`` rubric into the instructions Decisions can carry.

    ``# lossy``: placement only — the ``predicate`` envelope has no rubric field,
    so the two rubric descriptions are appended to the question text.
    """
    if not isinstance(criteria, dict):
        return instructions
    parts = []
    for key in ("true", "false"):
        description = criteria.get(key)
        if isinstance(description, str) and description:
            parts.append(f"Answer {key} when: {description}")
    return f"{instructions}\n\n" + "\n".join(parts) if parts else instructions


def _level_from_systemone(entry: Any) -> dict[str, Any]:
    """Turn one System One score level into a Decisions level."""
    if isinstance(entry, str):
        return {"label": entry}
    if isinstance(entry, dict):
        label = entry.get("label")
        if isinstance(label, str):
            level: dict[str, Any] = {"label": label}
            description = entry.get("description")
            if isinstance(description, str):
                level["description"] = description
            return level
    # A structured level with no label: keep the whole thing readable as one.
    return {"label": _json_text(entry)}


def _systemone_question_to_decision(
    question: dict[str, Any], name: str, model: str
) -> dict[str, Any]:
    """Convert one System One question into its Decisions form."""
    instructions = question.get("instructions")
    if not isinstance(instructions, str):
        # Decisions requires a string; System One accepts a structured payload
        # holding the question plus the data it references.
        instructions = _json_text(instructions)
    kind = question.get("type")
    criteria = question.get("criteria")
    if kind == "noul":
        return {
            "type": "predicate",
            "name": name,
            "instructions": _noul_instructions(instructions, criteria),
        }
    if kind == "choice":
        choices: list[dict[str, Any]] = []
        if isinstance(criteria, dict):
            for value, description in criteria.items():
                option: dict[str, Any] = {"value": value}
                if isinstance(description, str):
                    option["description"] = description
                choices.append(option)
        return {
            "type": "choice",
            "name": name,
            "instructions": instructions,
            "choices": choices,
        }
    if kind == "score":
        levels = (
            [_level_from_systemone(entry) for entry in criteria]
            if isinstance(criteria, list)
            else []
        )
        return {
            "type": "score",
            "name": name,
            "instructions": instructions,
            "levels": levels,
        }
    raise ValidationError(
        message=(
            f"Question '{name}' uses the System One '{kind}' type, which the Decisions "
            f"endpoint has no counterpart for, so model '{model}' cannot answer it "
            "through /v1/decisions. Send it to /v1/systemone on a provider that serves "
            "that question type."
        ),
        code="invalid_request_error",
        status_code=400,
    )


def _image_part(base64_data: str) -> dict[str, Any]:
    """Build an ``input_image`` part from raw base64.

    The source is raw base64, which Decisions cannot take as-is, so the media
    type is sniffed from the payload's magic bytes and a data URL is built.
    ``# lossy``: an unrecognized format is declared PNG, which is what System
    One's own image channel assumes.
    """
    media_type = _sniff_image_media_type(base64_data)
    return {
        "type": "input_image",
        "image_url": f"data:{media_type};base64,{base64_data}",
    }


def _sniff_image_media_type(base64_data: str) -> str:
    """Identify an image's media type from the leading bytes of its base64.

    Only the first bytes are decoded, so a truncated or padded payload still
    yields its magic number. A payload that cannot be decoded at all is an
    upstream problem (it will reject the image), not a proxy one, so it falls
    back to PNG.
    """
    try:
        header = base64.b64decode(base64_data[:24], validate=False)
    except ValueError:
        # binascii.Error is a ValueError: bad padding or alphabet.
        return "image/png"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header.startswith(b"GIF8"):
        return "image/gif"
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def systemone_to_decisions_request(request: InternalSystemOneRequest) -> InternalDecisionRequest:
    """Convert a System One request into the Decisions request that answers it."""
    state = request.state if isinstance(request.state, str) else _json_text(request.state)
    # Ollama's raw-base64 ``images`` is the only System One image channel; the
    # other upstreams strip it under the unknown-fields policy, so a request
    # that reaches here with images was addressed to an image-capable upstream.
    images = request.extra.get("images")
    image_parts = (
        [_image_part(entry) for entry in images if isinstance(entry, str)]
        if isinstance(images, list)
        else []
    )
    payload: Any = (
        [{"role": "user", "content": [{"type": "input_text", "text": state}, *image_parts]}]
        if image_parts
        else state
    )
    questions = [
        _systemone_question_to_decision(question, key, request.model)
        for key, question in request.questions.items()
        if isinstance(question, dict)
    ]
    return InternalDecisionRequest(
        model=request.model,
        input=payload,
        questions=questions,
        metadata=request.metadata,
        params=request.params,
        _override_injected_keys=set(request._override_injected_keys),
        user_facing_model=request.user_facing_model,
    )


def _decision_answer_to_systemone(
    answer: dict[str, Any], question: dict[str, Any] | None
) -> dict[str, Any]:
    """Convert one Decisions answer into its System One form."""
    kind = answer.get("type")
    if kind == "predicate":
        return {"type": "noul", "noul": answer.get("probability")}
    if kind == "choice":
        probabilities = {
            choice_key(entry.get("value")): entry.get("probability")
            for entry in answer.get("probabilities") or []
            if isinstance(entry, dict)
        }
        result: dict[str, Any] = {
            "type": "choice",
            "choice": choice_key(answer.get("choice")),
            "probabilities": probabilities,
        }
        if answer.get("confidence") is not None:
            result["confidence"] = answer["confidence"]
        return result
    if kind == "score":
        probabilities: dict[str, Any] = {}
        legend: dict[str, Any] = {}
        for entry in answer.get("probabilities") or []:
            if not isinstance(entry, dict):
                continue
            index = entry.get("value")
            probabilities[str(index)] = entry.get("probability")
            legend[str(index)] = entry.get("label")
        result = {
            "type": "score",
            "score": answer.get("score"),
            "probabilities": probabilities,
            "legend": legend,
        }
        if answer.get("confidence") is not None:
            result["confidence"] = answer["confidence"]
        return result
    # Refusal or an answer type System One does not model: passed through so the
    # client sees the answer the upstream actually gave. ``name`` is dropped
    # because the answer map's key already carries it.
    return {key: value for key, value in answer.items() if key != "name"}


def decisions_to_systemone_response(
    response: InternalDecisionResponse, request: InternalSystemOneRequest
) -> InternalSystemOneResponse:
    """Convert the Decisions answers to a System One response.

    Answers are keyed by the question ids the client sent, in the client's
    order; ``decisions_usage`` is dropped because the System One wire reports a
    flat input/output pair rather than Decisions' detailed usage object.
    """
    keys = list(request.questions)
    answers: dict[str, Any] = {}
    for index, answer in enumerate(response.answers):
        if not isinstance(answer, dict):
            continue
        key = keys[index] if index < len(keys) else f"answer_{index}"
        question = request.questions.get(key)
        answers[key] = _decision_answer_to_systemone(
            answer, question if isinstance(question, dict) else None
        )
    return InternalSystemOneResponse(
        model=response.model,
        answers=answers,
        usage=response.usage,
        id=response.id,
        provider=response.provider,
        request_id=response.request_id,
        provider_info={
            key: value for key, value in response.provider_info.items() if key != "decisions_usage"
        },
    )


__all__ = [
    "ASSET_PLACEHOLDER_PREFIX",
    "UNNAMED_KEY_PREFIX",
    "choice_key",
    "decisions_to_systemone_request",
    "decisions_to_systemone_response",
    "systemone_to_decisions_request",
    "systemone_to_decisions_response",
]
