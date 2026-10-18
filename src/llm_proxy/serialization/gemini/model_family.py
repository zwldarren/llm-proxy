"""Capability gates keyed by the Gemini model family's version.

The generateContent API changed shape between families, and each rule below is
stated by Google's own documentation for the family it applies to:

- **2.5** — ``thinkingConfig.thinkingBudget`` (a ``thinkingLevel`` there is an
  error); thought signatures exist but only for function calling.
- **3.x** — ``thinkingConfig.thinkingLevel``; thought signatures are required
  for function calling on every turn; ``generationConfig.responseFormat`` and
  ``FunctionDeclaration.parametersJsonSchema`` accept full JSON Schema, which
  supersedes the OpenAPI-subset ``responseSchema`` / ``parameters`` pair.
- **3.6+ / 4.x** — ``temperature``, ``top_p`` and ``top_k`` are ignored (and the
  API returns deprecation-warning headers; a future release turns them into
  400s), so the proxy stops sending them.

The family is read from the model id the client asked for, because that is the
only model identity the proxy has: ``gemini-3.8-flash`` → ``(3, 8)``. A model
alias the proxy cannot parse (an aggregator's nickname, a fine-tune) yields
``None`` and every gate falls back to the legacy-safe answer — the shape that
works on the oldest family still in service — instead of guessing a new one;
the fallback is logged once per id so it is never silent.

Every gate takes an optional ``model_family_overrides`` mapping (the provider's
``metadata.model_family_overrides``, exact model id → family version like
``"4.0"``): the operator's pin wins over the regex, which is how an
unparseable id gets a family without a code change.
"""

import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

#: The version segment of a Gemini model id: ``gemini-3.8-flash`` → 3.8,
#: ``gemini-4-argon`` → 4.0, ``gemini-2.5-flash`` → 2.5.
_FAMILY_RE = re.compile(r"gemini-(\d+)(?:\.(\d+))?")

logger = logging.getLogger(__name__)

#: Model ids whose missing version has already been logged. Process-lifetime
#: and bounded: the fallback is per-model deterministic, so one line per id is
#: all an operator needs, and the set stops growing rather than leaking.
_UNPARSABLE_WARNED: set[str] = set()
_UNPARSABLE_WARNED_LIMIT = 128


def _warn_unparsable_model_once(model: str) -> None:
    """Log the legacy-safe fallback for *model* once per process.

    An id without a ``gemini-X.Y`` version gets every family gate's legacy-safe
    answer silently; without this line the only symptom an operator sees is an
    upstream 400 (or a silently dropped thinking config). The log names the
    escape hatch that pins the id to a family.
    """
    if model in _UNPARSABLE_WARNED or len(_UNPARSABLE_WARNED) >= _UNPARSABLE_WARNED_LIMIT:
        return
    _UNPARSABLE_WARNED.add(model)
    logger.warning(
        "Model id %r carries no gemini-X.Y version; every Gemini family gate "
        "falls back to the legacy-safe shape (thinkingConfig omitted, "
        "sanitized schemas, sampling parameters sent, thought signatures "
        "stripped from foreign history). If this model is a newer family, pin "
        "it via the provider metadata key model_family_overrides.",
        model,
    )


@dataclass(frozen=True, order=True)
class GeminiFamilyVersion:
    """A Gemini model family version, comparable with ``(major, minor)`` order."""

    major: int
    minor: int = 0


def parse_family_version(value: str | None) -> GeminiFamilyVersion | None:
    """Parse an override value like ``"4.0"`` / ``"3"`` into a family version."""
    if not value:
        return None
    match = re.fullmatch(r"(\d+)(?:\.(\d+))?", str(value).strip())
    if match is None:
        return None
    return GeminiFamilyVersion(int(match.group(1)), int(match.group(2) or 0))


def normalize_family_overrides(raw: Any) -> dict[str, str]:
    """Validate a provider's ``model_family_overrides`` metadata value.

    The mapping pins a model id (exact match) to a family version, so a model
    whose id the proxy cannot parse — an alias like ``gemini-flash-latest``, an
    aggregator nickname — still gets the family shape its capabilities need.
    Entries whose value is not a version are warn-and-dropped: guessing a
    family from a malformed pin would defeat the point of pinning.
    """
    if not raw:
        return {}
    if not isinstance(raw, Mapping):
        logger.warning(
            "Ignoring model_family_overrides: expected a mapping of model id to "
            'version (e.g. {"gemini-flash-latest": "4.0"}), got %s',
            type(raw).__name__,
        )
        return {}
    overrides: dict[str, str] = {}
    for model_id, version in raw.items():
        if parse_family_version(str(version)) is None:
            logger.warning(
                "Dropping model_family_overrides entry %r: %r is not a family "
                'version like "4.0" or "3.8"',
                model_id,
                version,
            )
            continue
        overrides[str(model_id)] = str(version)
    return overrides


def gemini_model_version(
    model: str | None, model_family_overrides: Mapping[str, str] | None = None
) -> GeminiFamilyVersion | None:
    """The family version of *model*, or ``None`` when it cannot be read.

    *model_family_overrides* (the provider's ``metadata.model_family_overrides``)
    is consulted first, by exact model id: an operator pin wins over the regex,
    which is how a model the proxy cannot parse gets a family anyway.
    """
    if not model:
        return None
    if model_family_overrides:
        pinned = parse_family_version(model_family_overrides.get(model))
        if pinned is not None:
            return pinned
    match = _FAMILY_RE.search(model.lower())
    if match is None:
        _warn_unparsable_model_once(model)
        return None
    return GeminiFamilyVersion(int(match.group(1)), int(match.group(2) or 0))


def supports_thinking_config(
    model: str | None, model_family_overrides: Mapping[str, str] | None = None
) -> bool:
    """Whether the model accepts ``generationConfig.thinkingConfig`` at all.

    ``thinkingConfig`` arrived with the 2.5 series; older models reject the
    field.
    """
    version = gemini_model_version(model, model_family_overrides)
    return version is not None and version >= GeminiFamilyVersion(2, 5)


def uses_thinking_level(
    model: str | None, model_family_overrides: Mapping[str, str] | None = None
) -> bool:
    """Whether the model takes ``thinkingLevel`` (3.x and later) not a budget."""
    version = gemini_model_version(model, model_family_overrides)
    return version is not None and version >= GeminiFamilyVersion(3, 0)


def uses_thinking_budget(
    model: str | None, model_family_overrides: Mapping[str, str] | None = None
) -> bool:
    """Whether the model takes ``thinkingBudget`` (the 2.5 series only)."""
    version = gemini_model_version(model, model_family_overrides)
    return version is not None and GeminiFamilyVersion(2, 5) <= version < GeminiFamilyVersion(3, 0)


def supports_minimal_thinking_level(
    model: str | None, model_family_overrides: Mapping[str, str] | None = None
) -> bool:
    """Whether ``thinkingLevel: MINIMAL`` is accepted by this model.

    Google's thinking table marks ``minimal`` as an error on Gemini 4 Argon and
    Gemini 3.8 Flash only. Every other 3.x Flash / Flash-Lite / robotics model
    the table lists takes it — including Gemini 3.1 Flash-Lite **Image**, where
    ``minimal`` is the documented *default* and ``low``/``medium`` are the
    levels marked "Not Supported". Pro models are left out (no documented
    support), so they keep the default level with a warning instead of failing
    the request.
    """
    version = gemini_model_version(model, model_family_overrides)
    if version is None or not GeminiFamilyVersion(3, 0) <= version < GeminiFamilyVersion(3, 8):
        return False
    name = (model or "").lower()
    return "flash" in name or "robotics" in name


def supports_thought_signatures(
    model: str | None, model_family_overrides: Mapping[str, str] | None = None
) -> bool:
    """Whether the model issues (and validates) ``thoughtSignature``.

    Signatures arrived with the 2.5 series. From Gemini 3 they are *required*
    for function calling, so foreign history must carry the documented dummy
    signature; 2.5 validates them too, while 2.0 and earlier reject the field
    outright.
    """
    version = gemini_model_version(model, model_family_overrides)
    return version is not None and version >= GeminiFamilyVersion(2, 5)


def supports_json_schema_fields(
    model: str | None, model_family_overrides: Mapping[str, str] | None = None
) -> bool:
    """Whether to use ``responseFormat`` / ``parametersJsonSchema``.

    Both fields take plain JSON Schema (``anyOf``, ``$ref``, recursive
    definitions, ``additionalProperties``) and supersede the deprecated
    ``responseSchema`` / ``parameters`` OpenAPI subset. They ship with the
    Gemini 3 series.
    """
    version = gemini_model_version(model, model_family_overrides)
    return version is not None and version >= GeminiFamilyVersion(3, 0)


def ignores_sampling_params(
    model: str | None, model_family_overrides: Mapping[str, str] | None = None
) -> bool:
    """Whether the model ignores ``temperature`` / ``top_p`` / ``top_k``.

    Gemini 4 and Gemini 3.6+ ignore custom values and answer with deprecation
    warnings; the API has announced 400s for them in a future release. The
    deprecation starts with Gemini 3.6 Flash *and Gemini 3.5 Flash-Lite* — the
    release notes name both — so the Flash-Lite line joins at 3.5, while 3.5
    Flash proper and 3.1 Flash-Lite still accept the parameters.
    """
    version = gemini_model_version(model, model_family_overrides)
    if version is None:
        return False
    if "flash-lite" in (model or "").lower():
        return version >= GeminiFamilyVersion(3, 5)
    return version >= GeminiFamilyVersion(3, 6)


__all__ = [
    "GeminiFamilyVersion",
    "gemini_model_version",
    "ignores_sampling_params",
    "normalize_family_overrides",
    "parse_family_version",
    "supports_json_schema_fields",
    "supports_minimal_thinking_level",
    "supports_thought_signatures",
    "supports_thinking_config",
    "uses_thinking_budget",
    "uses_thinking_level",
]
