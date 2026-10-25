"""Pydantic request schemas for the OpenAI Decisions protocol.

The wire format is OpenAI's Decisions API (``POST /v1/decisions``, model
``gpt-6-luna``): shared evidence — a string, or an ordered list of conversation
items holding text, images, JSON, tool calls and their outputs — plus a list of
typed questions. It is the second wire format for the evaluation primitives
System One serves first (see :mod:`llm_proxy.protocols.systemone.schemas`).

Only ``predicate``, ``choice`` and ``score`` questions are modelled; unknown
question keys are allowed through so a future upstream field is not rejected
client-side. Conversation items are validated by role and type, because the
proxy forwards ``input`` verbatim to a native Decisions upstream and flattens it
for a System One one — a shape neither can use is worth naming at the edge.
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
MAX_QUESTIONS = 400
MIN_CHOICE_OPTIONS = 2
MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 26
#: OpenAI documents at most 128 image parts across one request. Question
#: instructions draw on the same budget.
MAX_IMAGE_PARTS = 128

#: Conversation item types that carry no ``content``: tool calls and outputs.
TOOL_ITEM_TYPES: frozenset[str] = frozenset(
    {"function_call", "function_call_output", "custom_tool_call", "custom_tool_call_output"}
)
#: Required fields per tool item type. The output items carry ``output``, the
#: call items the arguments or input the model produced.
_TOOL_ITEM_FIELDS: frozendict[str, tuple[str, ...]] = frozendict(
    {
        "function_call": ("call_id", "name", "arguments"),
        "function_call_output": ("call_id", "output"),
        "custom_tool_call": ("call_id", "name", "input"),
        "custom_tool_call_output": ("call_id", "output"),
    }
)
#: Content part types each message role may carry.
_ALLOWED_PARTS: frozendict[str, frozenset[str]] = frozendict(
    {
        "user": frozenset({"input_text", "input_image", "input_json"}),
        "developer": frozenset({"input_text"}),
        "system": frozenset({"input_text"}),
        "assistant": frozenset({"output_text", "refusal"}),
    }
)


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


class DecisionPromptCacheBreakpoint(BaseModel):
    """Marks the exact end of a reusable prompt prefix at one content part."""

    model_config = ConfigDict(extra="allow")

    mode: Literal["explicit"] = Field("explicit", description="The breakpoint mode.")


class DecisionInputTextSchema(BaseModel):
    """A text part of a message or of a question's instructions."""

    model_config = ConfigDict(extra="allow")

    type: Literal["input_text"]
    text: str
    prompt_cache_breakpoint: DecisionPromptCacheBreakpoint | None = Field(
        None, description="Marks the end of a reusable prompt prefix at this part."
    )


class DecisionInputImageSchema(BaseModel):
    """An image part of a message or of a question's instructions."""

    model_config = ConfigDict(extra="allow")

    type: Literal["input_image"]
    image_url: str = Field(
        ...,
        description="An inline base64 data URL, or a publicly accessible HTTP(S) URL.",
    )
    detail: Literal["low", "high", "auto", "original"] | None = Field(
        None, description="Image detail level; defaults to 'auto'."
    )
    prompt_cache_breakpoint: DecisionPromptCacheBreakpoint | None = Field(
        None, description="Marks the end of a reusable prompt prefix at this part."
    )

    @field_validator("image_url")
    @classmethod
    def _validate_image_url(cls, value: str) -> str:
        # The endpoint fetches a public HTTP(S) URL and accepts an inline
        # base64 data URL; a file id is not an input it supports at all.
        if value.startswith(("http://", "https://")):
            return value
        if value.startswith("data:") and ";base64," in value:
            return value
        raise ValueError(
            "image_url must be an inline base64 data URL or a public HTTP(S) URL; "
            "file ids are not supported"
        )


class DecisionInputJSONSchema(BaseModel):
    """A JSON object, or an array of objects, rendered as text evidence."""

    model_config = ConfigDict(extra="allow")

    type: Literal["input_json"]
    data: dict[str, Any] | list[Any] = Field(
        ..., description="Structured evidence; nested values may be any JSON value."
    )


class DecisionOutputTextSchema(BaseModel):
    """An assistant message's output text."""

    model_config = ConfigDict(extra="allow")

    type: Literal["output_text"]
    text: str


class DecisionRefusalSchema(BaseModel):
    """An assistant message's refusal, carried as conversation evidence."""

    model_config = ConfigDict(extra="allow")

    type: Literal["refusal"]
    refusal: str


DecisionInputPart = Annotated[
    DecisionInputTextSchema | DecisionInputImageSchema | DecisionInputJSONSchema,
    Field(discriminator="type"),
]
DecisionOutputPart = Annotated[
    DecisionOutputTextSchema | DecisionRefusalSchema,
    Field(discriminator="type"),
]


class DecisionQuestionSchema(BaseModel):
    """A typed Decisions question.

    ``predicate`` asks whether a condition holds, ``choice`` picks one of the
    supplied options, and ``score`` rates the evidence across an ordered
    ``levels`` rubric. ``instructions`` may also be a list of content parts, so a
    question can carry its own image next to text.
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["predicate", "choice", "score"]
    instructions: str | list[DecisionInputPart]
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
            if len(self.choices) < MIN_CHOICE_OPTIONS:
                raise ValueError(f"choice requires at least {MIN_CHOICE_OPTIONS} options")
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

    def image_parts(self) -> int:
        """Count the inline images carried by this question's instructions."""
        return _count_images(self.instructions)


class DecisionInputItemSchema(BaseModel):
    """One conversation item: a message, or a tool call/output.

    A message carries a ``role`` and ``content``; a tool item carries a ``type``,
    a ``call_id`` and the fields that type requires. The two shapes are mutually
    exclusive, and a message's content parts have to match its role — an
    assistant message holds ``output_text``/``refusal`` parts, while a user
    message holds text, image and JSON parts.
    """

    model_config = ConfigDict(extra="allow")

    type: (
        Literal[
            "message",
            "function_call",
            "function_call_output",
            "custom_tool_call",
            "custom_tool_call_output",
        ]
        | None
    ) = None
    role: Literal["user", "developer", "system", "assistant"] | None = None
    content: str | list[DecisionInputPart] | list[DecisionOutputPart] | None = Field(
        None,
        description="A message's text or ordered content parts.",
    )
    phase: Literal["commentary", "final_answer"] | None = Field(
        None, description="The phase of an assistant message."
    )
    # Tool items. ``call_id`` pairs a call with its output.
    call_id: str | None = None
    name: str | None = None
    arguments: str | None = None
    input: str | None = None
    output: str | list[DecisionInputPart] | None = None

    @model_validator(mode="after")
    def _validate_shape(self) -> DecisionInputItemSchema:
        required = _TOOL_ITEM_FIELDS.get(self.type or "")
        if self.role is None:
            if required is None:
                raise ValueError(
                    "input items must be a message (with a 'role') or a tool "
                    f"call/output (one of {', '.join(sorted(TOOL_ITEM_TYPES))})"
                )
            if self.content is not None:
                raise ValueError(f"a {self.type} item carries no 'content'")
            missing = [field for field in required if getattr(self, field) is None]
            if missing:
                raise ValueError(f"a {self.type} item requires {', '.join(missing)}")
            return self

        if self.type not in (None, "message"):
            raise ValueError(f"a message takes type 'message', not {self.type!r}")
        if self.content is None:
            raise ValueError("a message requires 'content'")
        if self.role != "assistant" and self.phase is not None:
            raise ValueError("only an assistant message carries a 'phase'")
        if isinstance(self.content, list):
            allowed = _ALLOWED_PARTS[self.role]
            rejected = sorted({part.type for part in self.content} - allowed)
            if rejected:
                raise ValueError(
                    f"a {self.role} message takes {', '.join(sorted(allowed))} parts; "
                    f"got {', '.join(rejected)}"
                )
        return self

    def image_parts(self) -> int:
        """Count the inline images this item carries."""
        return _count_images(self.content) + _count_images(self.output)


def _count_images(content: Any) -> int:
    """Count ``input_image`` parts in one message or tool-output payload."""
    if not isinstance(content, list):
        return 0
    return sum(1 for part in content if part.type == "input_image")


class DecisionPromptCacheOptionsSchema(BaseModel):
    """Options for the Decisions request's prompt cache."""

    model_config = ConfigDict(extra="allow")

    mode: Literal["implicit", "explicit"] | None = Field(
        None, description="Whether the upstream creates an implicit cache breakpoint."
    )
    ttl: str | None = Field(None, description="Lifetime of every breakpoint the request writes.")


class DecisionsRequestSchema(BaseModel):
    """Top-level Decisions request body."""

    model_config = ConfigDict(extra="forbid")

    input: str | list[DecisionInputItemSchema] = Field(
        ...,
        description=(
            "Shared evidence for every question: a plain string, or an ordered list of "
            "conversation items holding text, images, JSON, tool calls and tool outputs."
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
    prompt_cache_key: str | None = Field(
        None,
        max_length=64,
        description=(
            "Partitions prompt cache reuse by a caller-provided key. Forwarded only to "
            "upstreams that document it."
        ),
    )
    prompt_cache_options: DecisionPromptCacheOptionsSchema | None = Field(
        None,
        description=(
            "Prompt cache controls (mode and TTL). Forwarded only to upstreams that document them."
        ),
    )

    @model_validator(mode="after")
    def _validate_questions(self) -> DecisionsRequestSchema:
        if len(self.questions) > MAX_QUESTIONS:
            raise ValueError(f"at most {MAX_QUESTIONS} questions are allowed")
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
        parts = 0
        if isinstance(self.input, list):
            parts += sum(item.image_parts() for item in self.input)
        parts += sum(question.image_parts() for question in self.questions)
        if parts > MAX_IMAGE_PARTS:
            raise ValueError(f"at most {MAX_IMAGE_PARTS} inline image parts are allowed")
        return self


__all__ = [
    "MAX_CHOICE_OPTIONS",
    "MAX_IMAGE_PARTS",
    "MAX_QUESTIONS",
    "MAX_SCORE_LEVELS",
    "MIN_CHOICE_OPTIONS",
    "MIN_SCORE_LEVELS",
    "DecisionChoiceSchema",
    "DecisionInputImageSchema",
    "DecisionInputItemSchema",
    "DecisionInputJSONSchema",
    "DecisionInputTextSchema",
    "DecisionLevelSchema",
    "DecisionOutputTextSchema",
    "DecisionPromptCacheBreakpoint",
    "DecisionPromptCacheOptionsSchema",
    "DecisionQuestionSchema",
    "DecisionRefusalSchema",
    "DecisionsRequestSchema",
]
