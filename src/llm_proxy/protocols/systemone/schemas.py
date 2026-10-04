"""Pydantic request schemas for the System One protocol.

The wire format is shared by TypeSafe (``api.typesafe.ai/v1/systemone``),
OpenRouter (``openrouter.ai/api/v1/systemone``) and Ollama
(``localhost:11434/v1/systemone``, v0.35+). OpenRouter accepts a superset: the
optional ``provider`` routing block, ``session_id``, ``trace`` and ``user``;
Ollama adds its own optional ``images`` and ``keep_alive``. Those fields are
declared here so the endpoint validates them, then they ride
``InternalSystemOneRequest.extra`` so a provider that documents them receives
them and one that does not can strip them under its field policy.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

#: Instruction payloads may be a string, or a structured object/array holding
#: the question plus the data it references.
Instructions = str | dict[str, Any] | list[Any]


#: Upstream-documented limits, mirrored here so an invalid request fails at the
#: proxy with a 422 naming the offending question instead of at the provider
#: with a 400. Ollama's System One accepts up to 26 score levels where
#: TypeSafe and OpenRouter stop at 10, so the cap is the union: this schema runs
#: before a provider is selected (the same reasoning that governs ``noul``
#: criteria below). A request above a given upstream's own ceiling is still
#: rejected there, with that provider's error.
_MAX_CHOICE_OPTIONS = 255
_MIN_SCORE_LEVELS = 2
_MAX_SCORE_LEVELS = 26


class SystemOneQuestionSchema(BaseModel):
    """A typed System One question.

    ``noul`` is a yes/no question, ``choice`` picks one option from a map, and
    ``score`` rates the state across an ordered array of levels. Unknown keys
    are allowed so a future upstream field is not rejected client-side.
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["noul", "choice", "score"]
    instructions: Instructions
    criteria: dict[str, Any] | list[Any] | None = None

    @model_validator(mode="after")
    def _validate_criteria(self) -> SystemOneQuestionSchema:
        if self.type == "noul" and self.criteria is not None:
            if not isinstance(self.criteria, dict):
                raise ValueError("noul criteria must be an object with 'true'/'false' descriptions")
            # TypeSafe accepts either rubric key on its own; only OpenRouter
            # insists on both when criteria is given. This schema runs before
            # a provider is selected, so it accepts the union: enforce the
            # object shape and non-null descriptions, not both keys.
            for key in ("true", "false"):
                if key in self.criteria and self.criteria[key] is None:
                    raise ValueError(f"noul criteria '{key}' must not be null")
        elif self.type == "choice":
            if not isinstance(self.criteria, dict):
                raise ValueError(
                    "choice requires criteria as a map of option to rubric description"
                )
            if len(self.criteria) > _MAX_CHOICE_OPTIONS:
                raise ValueError(f"choice criteria supports at most {_MAX_CHOICE_OPTIONS} options")
        elif self.type == "score":
            if not isinstance(self.criteria, list):
                raise ValueError("score requires criteria as an ordered array of levels")
            if not _MIN_SCORE_LEVELS <= len(self.criteria) <= _MAX_SCORE_LEVELS:
                raise ValueError(
                    f"score criteria requires {_MIN_SCORE_LEVELS} to {_MAX_SCORE_LEVELS} levels"
                )
        return self


class SystemOneRequestSchema(BaseModel):
    """Top-level System One request body."""

    model_config = ConfigDict(extra="forbid")

    state: str | dict[str, Any] | list[Any] = Field(
        ...,
        description=(
            "The content to evaluate: a plain string, or a JSON object/array of "
            "related context (chat logs, records, application state)."
        ),
    )
    model: str = Field(..., description="The System One model, e.g. 'jev-latest'.")
    questions: dict[str, SystemOneQuestionSchema] = Field(
        ...,
        description="Map of question id to a typed question; answers come back under the same ids.",
    )

    # Ollama-only optional fields. Declared so they are validated and documented;
    # forwarded upstream only where documented (see EXEMPT_EXTRA_KEYS on the
    # Ollama adapter).
    images: list[str] | None = Field(
        None,
        description=(
            "Base64-encoded images shared by all questions, in request order "
            "(Ollama only; needs a vision-capable System One model such as Clef). "
            "URLs and data URLs are not accepted."
        ),
    )
    keep_alive: str | int | float | None = Field(
        None,
        description=(
            "How long to keep the model loaded after the request, as a duration "
            "string such as '5m' or seconds (Ollama only)."
        ),
    )

    @field_validator("images")
    @classmethod
    def _validate_images(cls, value: list[str] | None) -> list[str] | None:
        # Ollama takes raw base64 only. Sending a URL is a client mistake that
        # upstream reports as an opaque 400, so name it here instead. Base64
        # cannot contain ':', so the check cannot reject a valid payload.
        for entry in value or []:
            if entry.startswith(("http://", "https://", "data:")):
                raise ValueError(
                    "images entries must be base64 data; URLs and data URLs are not supported"
                )
        return value

    @field_validator("keep_alive", mode="before")
    @classmethod
    def _validate_keep_alive(cls, value: str | int | float | None) -> str | int | float | None:
        # ``bool`` is an ``int`` subclass and pydantic coerces it in lax mode, so
        # ``keep_alive: true`` would otherwise be read as the duration 1
        # (second) and unload the model immediately. Checked before coercion.
        if isinstance(value, bool):
            raise ValueError("keep_alive must be a duration string or a number of seconds")
        return value

    # OpenRouter-only optional fields. Declared so they are validated and
    # documented; forwarded upstream only where documented (see
    # EXEMPT_EXTRA_KEYS on the OpenRouter adapter).
    provider: dict[str, Any] | None = Field(
        None, description="Provider routing preferences (OpenRouter only)."
    )
    session_id: str | None = Field(
        None,
        max_length=256,
        description="Observability grouping id, never sent to the provider (OpenRouter only).",
    )
    trace: dict[str, Any] | None = Field(
        None, description="Observability/tracing metadata (OpenRouter only)."
    )
    user: str | None = Field(
        None, max_length=256, description="End-user identifier (OpenRouter only)."
    )


__all__ = ["SystemOneQuestionSchema", "SystemOneRequestSchema"]
