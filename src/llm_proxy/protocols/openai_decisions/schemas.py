"""Pydantic request schemas for the OpenAI Decisions protocol.

The wire format is OpenAI's Decisions API (``POST /v1/decisions``, model
``gpt-6-luna``): shared evidence as a string or user messages holding text and
inline images, plus a list of typed questions. It is the second wire format for
the evaluation primitives System One serves first (see
:mod:`llm_proxy.protocols.systemone.schemas`), and the limits mirror that
schema's so the two endpoints reject the same requests.

Only ``predicate``, ``choice`` and ``score`` questions are modelled; unknown
question keys are allowed through so a future upstream field is not rejected
client-side.
"""

from collections import Counter
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from llm_proxy.models.decisions_bridge import UNNAMED_KEY_PREFIX, choice_key

#: Decisions choice values are typed: a string and a boolean with the same text
#: are distinct options. The bridge keys System One's ``criteria`` map by
#: string, so two options that collapse to the same key would lose one of them.
ChoiceValue = str | bool

#: Upstream-documented limits, mirrored here so an invalid request fails at the
#: proxy with a 422 naming the offending question instead of at the upstream
#: with a 400. This schema runs before a provider is selected, so it enforces
#: the union across every upstream that can answer — a native Decisions
#: endpoint, or a System One upstream reached through the bridge (Ollama
#: accepts up to 26 score levels where TypeSafe and OpenRouter stop at 10). A
#: request above a given upstream's own ceiling is still rejected there, with
#: that upstream's error.
MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 26
#: OpenAI documents at most 128 image parts across all messages in one request.
MAX_IMAGE_PARTS = 128


class DecisionChoiceSchema(BaseModel):
    """One selectable option of a ``choice`` question."""

    model_config = ConfigDict(extra="allow")

    value: ChoiceValue
    description: str | None = Field(
        None, description="When this option applies; omit for a self-explanatory value."
    )

    @field_validator("value", mode="before")
    @classmethod
    def _validate_value(cls, value: Any) -> Any:
        # Checked before coercion: pydantic's lax mode would turn ``1`` into
        # ``True``, silently changing which option the client asked for.
        if isinstance(value, str | bool):
            return value
        raise ValueError("choice values must be a string or a boolean")


class DecisionLevelSchema(BaseModel):
    """One ordered level of a ``score`` question's rubric."""

    model_config = ConfigDict(extra="allow")

    label: str
    description: str | None = Field(
        None, description="What the level means; omit when the label says it."
    )


class DecisionQuestionSchema(BaseModel):
    """A typed Decisions question.

    ``predicate`` asks whether a condition holds, ``choice`` picks one of the
    supplied options, and ``score`` rates the evidence across an ordered
    ``levels`` rubric.
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["predicate", "choice", "score"]
    instructions: str
    name: str | None = Field(
        None,
        description=(
            "Identifier the answer echoes back. Optional on the wire, but names "
            "must be unique when given: answers are correlated by name."
        ),
    )
    choices: list[DecisionChoiceSchema] | None = Field(
        None, description="Options of a 'choice' question, in preference order."
    )
    levels: list[DecisionLevelSchema] | None = Field(
        None, description="Ordered levels of a 'score' question, lowest to highest."
    )

    @field_validator("name", mode="before")
    @classmethod
    def _validate_name(cls, value: Any) -> Any:
        # The upstream takes an optional string: absent, or a string — never
        # null. Explicit null would be rejected there as a type error, so name
        # the mistake here instead (an omitted key keeps the field's None
        # default and does not reach this validator).
        if value is None:
            raise ValueError("name must be a string or omitted, not null")
        return value

    @model_validator(mode="after")
    def _validate_shape(self) -> DecisionQuestionSchema:
        if self.type == "predicate":
            if self.choices is not None or self.levels is not None:
                raise ValueError("predicate requires neither 'choices' nor 'levels'")
        elif self.type == "choice":
            if not self.choices:
                raise ValueError("choice requires a non-empty 'choices' array")
            if self.levels is not None:
                raise ValueError("choice takes 'choices', not 'levels'")
            if len(self.choices) > MAX_CHOICE_OPTIONS:
                raise ValueError(f"choice supports at most {MAX_CHOICE_OPTIONS} options")
            keys = [choice_key(option.value) for option in self.choices]
            if len(set(keys)) != len(keys):
                raise ValueError(
                    "choice option values must be distinct (a boolean and its string form "
                    "are the same option)"
                )
        elif self.type == "score":
            if not self.levels:
                raise ValueError("score requires a non-empty 'levels' array")
            if self.choices is not None:
                raise ValueError("score takes 'levels', not 'choices'")
            if not MIN_SCORE_LEVELS <= len(self.levels) <= MAX_SCORE_LEVELS:
                raise ValueError(f"score requires {MIN_SCORE_LEVELS} to {MAX_SCORE_LEVELS} levels")
        return self


class DecisionInputTextSchema(BaseModel):
    """A text part of a user message."""

    model_config = ConfigDict(extra="allow")

    type: Literal["input_text"]
    text: str


class DecisionInputImageSchema(BaseModel):
    """An inline image part of a user message."""

    model_config = ConfigDict(extra="allow")

    type: Literal["input_image"]
    image_url: str
    detail: Literal["low", "high", "auto", "original"] | None = Field(
        None, description="Image detail level; defaults to 'auto'."
    )

    @field_validator("image_url")
    @classmethod
    def _validate_inline(cls, value: str) -> str:
        # Decisions takes inline base64 data URLs only — hosted HTTP(S) URLs and
        # file ids are rejected upstream, so name the mistake here instead.
        if not value.startswith("data:") or ";base64," not in value:
            raise ValueError(
                "image_url must be an inline base64 data URL; hosted URLs and file ids "
                "are not supported"
            )
        return value


DecisionInputPart = Annotated[
    DecisionInputTextSchema | DecisionInputImageSchema,
    Field(discriminator="type"),
]


class DecisionInputMessageSchema(BaseModel):
    """A user message carrying the evidence to evaluate."""

    model_config = ConfigDict(extra="allow")

    role: Literal["user"] = Field(
        ..., description="Only user messages are supported by this endpoint."
    )
    content: str | list[DecisionInputPart]
    type: Literal["message"] | None = None


class DecisionsRequestSchema(BaseModel):
    """Top-level Decisions request body."""

    model_config = ConfigDict(extra="forbid")

    input: str | list[DecisionInputMessageSchema] = Field(
        ...,
        description=(
            "Shared evidence for every question: a plain string, or user messages "
            "holding text and inline base64 images."
        ),
    )
    model: str = Field(..., description="The Decisions model, e.g. 'gpt-6-luna'.")
    questions: list[DecisionQuestionSchema] = Field(
        ...,
        min_length=1,
        description="Typed questions, in the order their answers come back.",
    )
    safety_identifier: str | None = Field(
        None,
        description=(
            "Opaque caller-provided end-user identifier, scoped by the verified org. "
            "Forwarded only to upstreams that document it."
        ),
    )

    @model_validator(mode="after")
    def _validate_questions(self) -> DecisionsRequestSchema:
        names = [question.name for question in self.questions if question.name]
        reserved = sorted(name for name in names if name.startswith(UNNAMED_KEY_PREFIX))
        if reserved:
            # A name under this prefix would collide with the synthetic key the
            # bridge gives an unnamed question at that index, silently merging
            # two questions into one answer — rejected at the edge instead.
            raise ValueError(
                f"question names starting with '{UNNAMED_KEY_PREFIX}' are reserved "
                "for unnamed questions"
            )
        duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
        if duplicates:
            # A duplicate name makes an answer ambiguous: the answer echoes the
            # name, so two questions sharing one cannot be told apart.
            raise ValueError(f"question names must be unique; repeated: {', '.join(duplicates)}")
        return self

    @model_validator(mode="after")
    def _validate_image_count(self) -> DecisionsRequestSchema:
        if not isinstance(self.input, list):
            return self
        parts = sum(
            1
            for message in self.input
            if isinstance(message.content, list)
            for part in message.content
            if part.type == "input_image"
        )
        if parts > MAX_IMAGE_PARTS:
            raise ValueError(f"at most {MAX_IMAGE_PARTS} inline image parts are allowed")
        return self


__all__ = [
    "MAX_CHOICE_OPTIONS",
    "MAX_IMAGE_PARTS",
    "MAX_SCORE_LEVELS",
    "MIN_SCORE_LEVELS",
    "DecisionChoiceSchema",
    "DecisionInputImageSchema",
    "DecisionInputMessageSchema",
    "DecisionInputTextSchema",
    "DecisionLevelSchema",
    "DecisionQuestionSchema",
    "DecisionsRequestSchema",
]
