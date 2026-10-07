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
* Decisions ``input`` may hold ``input_image`` parts, but System One has no
  image channel except Ollama's raw-base64 ``images`` field. Images are moved
  there (data URL prefix stripped, since Ollama rejects data URLs); the
  upstreams that do not document ``images`` drop them under the unknown-fields
  policy, exactly as a client-sent ``images`` field would be dropped.
* System One ``state`` may be an object or array, while Decisions ``input`` is
  a string or a list of user messages: the object is serialized to JSON text.
* The optional fields each envelope has and the other does not
  (``safety_identifier``; ``provider``/``session_id``/``trace``/``user``/
  ``keep_alive``) have no counterpart and are dropped.

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
    if not isinstance(value, str) or not value.startswith("data:"):
        return None
    marker = ";base64,"
    position = value.find(marker)
    if position == -1:
        return None
    payload = value[position + len(marker) :]
    return payload or None


# --------------------------------------------------------------------------- #
# Decisions -> System One
# --------------------------------------------------------------------------- #


def _render_decision_input(payload: Any) -> tuple[Any, list[str]]:
    """Render Decisions ``input`` into a System One ``state`` plus base64 images.

    A string input is the state verbatim. A message array is flattened to one
    text block — every message is a user message, so joining them preserves the
    evidence in order — with its inline images returned separately for the
    upstream's own image channel.
    """
    if isinstance(payload, str):
        return payload, []
    if not isinstance(payload, list):
        # The schema admits a string or a message array only; anything else is
        # passed through as the state rather than silently dropped.
        return payload, []

    texts: list[str] = []
    images: list[str] = []
    for message in payload:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            texts.append(content)
            continue
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "input_text" and isinstance(part.get("text"), str):
                texts.append(part["text"])
            elif part.get("type") == "input_image":
                image = _base64_from_data_url(part.get("image_url"))
                if image is not None:
                    images.append(image)
    return "\n\n".join(texts), images


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
    return f"{instructions}\n\nLevels, lowest to highest:\n" + "\n".join(lines)


def _decision_question_to_systemone(question: dict[str, Any]) -> dict[str, Any]:
    """Convert one Decisions question into its System One form."""
    kind = question.get("type")
    instructions = question.get("instructions")
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


def decisions_to_systemone_request(request: InternalDecisionRequest) -> InternalSystemOneRequest:
    """Convert a Decisions request into the System One request that answers it."""
    state, images = _render_decision_input(request.input)
    questions: dict[str, Any] = {}
    for index, question in enumerate(request.questions):
        if not isinstance(question, dict):
            continue
        name = _question_name(question)
        questions[name or _unnamed_key(index)] = _decision_question_to_systemone(question)
    return InternalSystemOneRequest(
        model=request.model,
        state=state,
        questions=questions,
        extra={"images": images} if images else {},
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


def _systemone_question_to_decision(question: dict[str, Any], name: str) -> dict[str, Any]:
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
            f"Question '{name}' uses System One type '{kind}', which the Decisions "
            "endpoint cannot express. Send the request to /v1/systemone instead."
        ),
        code="invalid_request_error",
        status_code=400,
    )


def _image_part(base64_data: str) -> dict[str, Any]:
    """Build an ``input_image`` part from raw base64.

    Decisions requires a data URL and rejects hosted URLs, so the media type is
    sniffed from the payload's magic bytes. ``# lossy``: an unrecognized format
    is declared PNG, which is what System One's own image channel assumes.
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
        _systemone_question_to_decision(question, key)
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
    "UNNAMED_KEY_PREFIX",
    "choice_key",
    "decisions_to_systemone_request",
    "decisions_to_systemone_response",
    "systemone_to_decisions_request",
    "systemone_to_decisions_response",
]
