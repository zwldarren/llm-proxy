"""Gemini's Decisions API as the evaluation wire (Gemini Decision 1).

Gemini serves its evaluation models on ``POST {base}/decisions`` — a *third*
spelling of the same three primitives, closest to the System One envelope:

- the questions are a **map** keyed by an id, answered under the same keys, with
  the System One kind names (``noul`` / ``choice`` / ``score``) and the same
  ``criteria`` shapes (a map for ``choice``, an ordered array for ``score``);
- the evidence field is ``input``, and it is made of Interactions-style content
  blocks (``{"type": "text", "text": …}``, ``{"type": "image", "data" | "uri",
  "mime_type": …}``, plus audio, video and document blocks) instead of a
  ``state``;
- the usage block speaks the Interactions vocabulary (``total_input_tokens``,
  ``total_output_tokens``, ``input_tokens_by_modality``), and decisions generate
  zero output tokens.

So the adapter declares the System One capability against this route (see the
Gemini adapter), and this module owns the two wire translations that differ:
``build_provider_systemone_request`` / ``parse_provider_systemone_response``.
The capabilities, the Decisions bridge and the protocol serializers stay
untouched — a client on ``/v1/systemone`` and one on ``/v1/decisions`` both reach
Gemini Decision 1 through the shapes they already write.

Everything a client puts in a System One ``state`` or question map that Gemini's
Decisions API has no field for is either translated here or refused with a named
400 — never forwarded as a field Gemini would reject, and never dropped without
saying so in a docstring:

* ``state`` is a string or a content-block array on this wire; an object or a
  JSON array of related context is serialized to text, which is what Gemini's own
  docs recommend for structured evidence.
* A question's ``instructions`` are text only (``"Media inside questions isn't
  supported"``), so a structured instructions value becomes JSON text.
* A ``noul`` rubric moves from System One's ``{"true", "false"}`` keys to
  Gemini's ``"true_description"`` / ``"false_description"``.
* A ``score`` level that carries both a label and a description (System One
  allows a structured level) becomes one description string — its criteria are
  the level descriptions Gemini echoes back as the answer legend.
* ``box`` — the image-locating question only Jev has — has no counterpart.
* A question id Gemini would refuse is named here too: its keys are 1–64 ASCII
  characters matching ``[A-Za-z0-9_-]+``, and a System One client (or a Decisions
  client's question ``name``) may hold anything.
"""

import json
import re
from typing import Any

from llm_proxy.core.exceptions import ValidationError
from llm_proxy.models.systemone import (
    InternalSystemOneRequest,
    InternalSystemOneResponse,
)
from llm_proxy.models.types import PromptTokensDetails, Usage
from llm_proxy.serialization.gemini_interactions.usage import (
    interactions_billable_token_counts,
    interactions_input_tokens_by_modality,
)

#: The question kinds Gemini's Decisions API documents.
_QUESTION_KINDS: frozenset[str] = frozenset({"noul", "choice", "score"})

#: The `noul` rubric keys System One/TypeSafe use, and Gemini's spelling.
_NOUL_CRITERIA_KEYS: frozendict[str, str] = frozendict(
    {"true": "true_description", "false": "false_description"}
)
_GEMINI_NOUL_CRITERIA_KEYS: frozenset[str] = frozenset(_NOUL_CRITERIA_KEYS.values())

#: Content block kinds Gemini's Decisions ``input`` accepts.
_CONTENT_BLOCK_KINDS: frozenset[str] = frozenset({"text", "image", "audio", "video", "document"})

#: Gemini's question keys: 1–64 ASCII characters, ``[A-Za-z0-9_-]+``. A key it
#: rejects is worth naming at the proxy: the id is the client's own handle for the
#: answer, so it cannot be rewritten on the way out.
_QUESTION_KEY_RE = re.compile(r"[A-Za-z0-9_-]+")
_MAX_QUESTION_KEY_LENGTH = 64


class GeminiEvaluationMixin:
    """System One body building and parsing for Gemini's Decisions route.

    Mixed into the ``gemini`` provider serializer. The route is Google's own
    evaluation resource and is independent of the adapter's chat dialect
    (``api_variant``), so the translation lives in one place rather than per
    chat variant.
    """

    def build_provider_systemone_request(self, request: InternalSystemOneRequest) -> dict[str, Any]:
        """Build Gemini's Decisions body from the System One envelope."""
        questions = {
            _validated_key(key, model=request.model): _evaluation_question(
                question, key=key, model=request.model
            )
            for key, question in request.questions.items()
        }
        return {
            "model": request.model,
            "input": _evaluation_input(request.state),
            "questions": questions,
        }

    def parse_provider_systemone_response(
        self, response: dict[str, Any], model: str = ""
    ) -> InternalSystemOneResponse:
        """Parse Gemini's answers into the System One response record.

        The `answers` map is the System One one (typed ``noul`` / ``choice`` /
        ``score`` answers keyed like the questions), so it passes through
        verbatim. The usage block is a different vocabulary and is mapped onto the
        canonical record; it is deliberately *not* echoed as ``systemone_usage``,
        because the System One wire reports a flat ``{input_tokens,
        output_tokens}`` pair and a client asking for that wire must not receive
        Gemini's.
        """
        answers = response.get("answers")
        return InternalSystemOneResponse(
            model=response.get("model") or model,
            answers=answers if isinstance(answers, dict) else {},
            usage=_evaluation_usage(response.get("usage")),
            id=response.get("id"),
            provider=response.get("provider"),
        )


def _evaluation_input(state: Any) -> Any:
    """Render a System One ``state`` as Gemini's ``input``.

    A string passes through (the wire accepts one). A content-block array — the
    shape the image channel renders, and one a client may send directly — passes
    through too: Gemini concatenates adjacent text blocks with no separator, so
    the blocks already say what the evidence is. Anything else is structured
    context, which Gemini's docs say to serialize to text so its keys stay data
    rather than being read as content blocks.
    """
    if isinstance(state, str):
        return state
    if isinstance(state, dict) and state.get("type") in _CONTENT_BLOCK_KINDS:
        return state
    if isinstance(state, list) and all(
        isinstance(item, dict) and item.get("type") in _CONTENT_BLOCK_KINDS for item in state
    ):
        return state
    return _json_text(state)


def _validated_key(key: str, *, model: str) -> str:
    """Return the question id, or name the rule Gemini would break on it."""
    if (
        isinstance(key, str)
        and _QUESTION_KEY_RE.fullmatch(key)
        and len(key) <= _MAX_QUESTION_KEY_LENGTH
    ):
        return key
    raise ValidationError(
        message=(
            f"Question id '{key}' is not one model '{model}' can answer under: Gemini's "
            "Decisions API takes question keys of 1 to 64 characters matching "
            "[A-Za-z0-9_-]+. Rename the question in this request."
        ),
        code="invalid_request_error",
        status_code=400,
    )


def _evaluation_question(question: Any, *, key: str, model: str) -> Any:
    """Convert one System One question into Gemini's question object."""
    if not isinstance(question, dict):
        # A question that is not an object has no kind to convert, and the
        # upstream's own rejection would name nothing useful.
        raise ValidationError(
            message=(
                f"Question '{key}' is not a question object, so model '{model}' cannot "
                "answer it: Gemini's Decisions API takes a typed question per key."
            ),
            code="invalid_request_error",
            status_code=400,
        )
    kind = question.get("type")
    if kind not in _QUESTION_KINDS:
        raise _unsupported_question(key, kind, model)
    body: dict[str, Any] = {
        "type": kind,
        "instructions": _evaluation_instructions(question.get("instructions")),
    }
    criteria = question.get("criteria")
    if kind == "noul":
        rubric = _noul_criteria(criteria)
        if rubric:
            body["criteria"] = rubric
    elif kind == "choice":
        body["criteria"] = _choice_criteria(criteria, key=key, model=model)
    else:
        body["criteria"] = _score_criteria(criteria, key=key, model=model)
    return body


def _unsupported_question(key: str, kind: Any, model: str) -> ValidationError:
    """Name a question type Gemini's Decisions API has no counterpart for."""
    return ValidationError(
        message=(
            f"Question '{key}' uses the System One '{kind}' question type, which Gemini's "
            f"Decisions API does not answer, so model '{model}' cannot serve it. Send it to "
            "a provider that speaks that question type."
        ),
        code="invalid_request_error",
        status_code=400,
    )


def _evaluation_instructions(instructions: Any) -> Any:
    """Render a question's instructions as text, the only form Gemini accepts.

    A string and a text content block (or an array of them) pass through; a
    structured payload — System One allows one that holds the question beside the
    data it references — becomes JSON text. ``# lossy``: spelling only, the
    payload's keys stay addressable in the instructions.
    """
    if isinstance(instructions, str):
        return instructions
    if isinstance(instructions, dict) and instructions.get("type") == "text":
        return instructions
    if isinstance(instructions, list) and all(
        isinstance(part, dict) and part.get("type") == "text" for part in instructions
    ):
        return instructions
    return _json_text(instructions)


def _noul_criteria(criteria: Any) -> dict[str, Any] | None:
    """Translate System One's ``noul`` rubric to Gemini's key spelling.

    System One describes the two outcomes under ``true`` / ``false``; Gemini names
    the same two descriptions ``true_description`` / ``false_description``. Either
    spelling is accepted, a value that is not a non-empty string is skipped (there
    is nothing to say with it), and an empty rubric is omitted rather than sent as
    an object Gemini may reject.
    """
    if not isinstance(criteria, dict):
        return None
    rubric: dict[str, Any] = {}
    for name, description in criteria.items():
        spelling = _NOUL_CRITERIA_KEYS.get(name, name)
        if spelling in _GEMINI_NOUL_CRITERIA_KEYS and isinstance(description, str) and description:
            rubric[spelling] = description
    return rubric or None


def _choice_criteria(criteria: Any, *, key: str, model: str) -> dict[str, Any]:
    """Render System One's ``choice`` criteria map as Gemini's.

    Both envelopes key the options by id, so the map passes through. Gemini
    documents every value as a description string and an empty one for a
    self-explanatory option, where System One allows ``null``.
    """
    if not isinstance(criteria, dict) or not criteria:
        raise ValidationError(
            message=(
                f"Question '{key}' is a choice without a criteria map, which model '{model}' "
                "cannot answer: Gemini's Decisions API takes one description per option."
            ),
            code="invalid_request_error",
            status_code=400,
        )
    return {
        option: "" if description is None else _evaluation_text(description)
        for option, description in criteria.items()
    }


def _score_criteria(criteria: Any, *, key: str, model: str) -> list[str]:
    """Render System One's ``score`` rubric as Gemini's ordered level descriptions.

    Gemini's criteria *are* the level descriptions it echoes back as the answer
    legend, so a level that carries a label and a description becomes one string
    holding both. ``# lossy``: they arrive as one description instead of two
    fields.
    """
    if not isinstance(criteria, list) or not criteria:
        raise ValidationError(
            message=(
                f"Question '{key}' is a score without an ordered criteria array, which model "
                f"'{model}' cannot answer: Gemini's Decisions API takes one description per "
                "level."
            ),
            code="invalid_request_error",
            status_code=400,
        )
    return [_evaluation_text(level) for level in criteria]


def _evaluation_text(value: Any) -> str:
    """Render one rubric value as the description string Gemini expects."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        label = value.get("label")
        description = value.get("description")
        if isinstance(label, str) and isinstance(description, str) and description:
            return f"{label} — {description}"
        for candidate in (description, label):
            if isinstance(candidate, str):
                return candidate
    return _json_text(value)


def _evaluation_usage(raw_usage: Any) -> Usage | None:
    """Map Gemini's Decisions usage block onto the canonical record.

    The block is the Interactions vocabulary, so its counts come from the same
    shared mapping the Interactions parser uses (``total_input_tokens`` /
    ``total_output_tokens``, plus the per-modality breakdown). Decisions generate
    no output tokens, so output stays 0; the modalities are kept in the canonical
    prompt details, where per-modality pricing reads them.
    """
    if not isinstance(raw_usage, dict):
        return None
    input_tokens, output_tokens = interactions_billable_token_counts(
        raw_usage, has_search_grounding=False
    )
    counts = interactions_input_tokens_by_modality(raw_usage)
    details = PromptTokensDetails(
        text_tokens=counts.get("text"),
        image_tokens=counts.get("image"),
        audio_tokens=counts.get("audio"),
        video_tokens=counts.get("video"),
    )
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=_count(raw_usage.get("total_tokens")),
        cache_read_input_tokens=_count(raw_usage.get("total_cached_tokens")) or None,
        reasoning_tokens=_count(raw_usage.get("total_thought_tokens")) or None,
        prompt_tokens_details=(
            details if any(v is not None for v in vars(details).values()) else None
        ),
    )


def _count(value: Any) -> int | None:
    """An int the wire actually reported; booleans and floats are not counts."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _json_text(value: Any) -> str:
    """Serialize a structured value to compact JSON text."""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


__all__ = ["GeminiEvaluationMixin"]
