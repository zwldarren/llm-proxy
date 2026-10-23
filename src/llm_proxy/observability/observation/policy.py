"""The one content and sampling policy every observability sink shares.

Two invariants used to live in whoever happened to write a payload:

1. **Privacy.** ``logging.log_input_output`` and ``logging.mask_sensitive_data``
   governed the database log only. The tracing handler rendered prompts straight
   from the request, so a deployment that had turned body logging off still
   shipped every prompt to Langfuse — the exact content the operator had opted
   out of storing. The policy is resolved once per request (on
   :class:`~llm_proxy.observability.event_context.EventContext`) and every sink
   narrows it, never widens it, unless an operator explicitly opts in.

2. **Size.** The byte cap on a stored body and the masking of sensitive keys were
   separate implementations in the log store and in the tracing payload builders.
   They are one implementation here: :func:`mask_payload` and
   :func:`cap_payload`, called by the log store, by the Langfuse handler, and by
   the sink bus.

Three content modes, ordered by how much they expose:

- ``none`` — the sink receives no content at all (bodies, prompts, tool payloads).
- ``masked`` — content is delivered with every sensitive key's value masked.
- ``full`` — content is delivered verbatim.

A sink cannot escalate: :func:`effective_mode` returns the *narrower* of the
request's mode and the sink's requested mode. The only way to send full content
when the deployment masks it is the sink's own explicit opt-in setting (for
Langfuse, ``send_content="full"``), which is a deliberate, documented operator
decision rather than a silent default.

See ADR-0044.
"""

import random
from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Any, Literal

import orjson

from llm_proxy.security.passwords import SENSITIVE_KEYS, mask_sensitive

if TYPE_CHECKING:
    from llm_proxy.config.types.logging_config import LoggingConfig

#: How much content a sink may see. Ordered: ``none`` < ``masked`` < ``full``.
ContentMode = Literal["none", "masked", "full"]

#: What an operator may *ask a backend* to export. Distinct from
#: :data:`ContentMode`: ``inherit`` is a setting value ("follow the request's
#: policy"), not a mode a payload can be projected through, and an exporter
#: resolves it before any content is touched.
SendContent = Literal["inherit", "none", "full"]

_MODE_RANK: frozendict[ContentMode, int] = frozendict({"none": 0, "masked": 1, "full": 2})


def content_mode(*, bodies_enabled: bool, mask_sensitive: bool) -> ContentMode:
    """Resolve the deployment's content mode from the logging switches.

    ``bodies_enabled`` is ``logging.log_input_output`` — the master switch, so it
    wins outright; ``mask_sensitive`` is ``logging.mask_sensitive_data``.
    """
    if not bodies_enabled:
        return "none"
    return "masked" if mask_sensitive else "full"


def narrow_mode(request_mode: ContentMode, sink_mode: ContentMode) -> ContentMode:
    """The narrower of the request's content mode and the sink's requested mode.

    A sink may always expose less than the request allows; it can never expose
    more.
    """
    return request_mode if _MODE_RANK[request_mode] <= _MODE_RANK[sink_mode] else sink_mode


#: Memoised sensitive-key sets, keyed by the config object's identity. Resolving
#: this twice per request (request and response body) was a measurable cost on
#: the masking hot path; ``resolve_logging_config`` returns the cached config
#: object, so identity is stable until settings change.
_sensitive_keys_cache: tuple[Any, frozenset[str]] | None = None


def sensitive_keys_for(config: LoggingConfig | None) -> frozenset[str]:
    """The lowercased key names to mask: the built-ins plus the configured extras.

    ``None`` means "no configuration available" and yields the built-in set only.
    """
    global _sensitive_keys_cache
    if config is None:
        return frozenset(SENSITIVE_KEYS)
    cached = _sensitive_keys_cache
    if cached is not None and cached[0] is config:
        return cached[1]
    keys = frozenset(SENSITIVE_KEYS | {key.lower() for key in config.sensitive_keys})
    _sensitive_keys_cache = (config, keys)
    return keys


def mask_payload(value: Any, sensitive_keys: frozenset[str]) -> Any:
    """Mask sensitive keys in a nested payload.

    Dict/list payloads are walked; scalars (a rendered prompt string, a bare
    float) have no keys to mask and pass through unchanged.
    """
    if not sensitive_keys or not isinstance(value, (dict, list)) or not value:
        return value
    return mask_sensitive(value, sensitive_keys)


def cap_payload(value: Any, *, max_bytes: int = 0, max_chars: int = 0) -> Any:
    """Replace a payload that exceeds its size budget with a truncation marker.

    ``max_bytes`` is measured on the compact ``orjson`` form (and, for a raw SSE
    string, on its UTF-8 byte length) and bounds a stored body; ``max_chars``
    truncates a rendered string in place. ``0`` disables the corresponding cap.
    A payload that is neither a dict, list nor string (or that cannot be dumped)
    is passed through untouched.
    """
    if max_chars > 0 and isinstance(value, str) and len(value) > max_chars:
        return value[:max_chars] + "…"
    if max_bytes <= 0:
        return value
    if isinstance(value, str):
        size = len(value.encode("utf-8"))
    elif isinstance(value, (dict, list)):
        try:
            size = len(orjson.dumps(value, option=orjson.OPT_NON_STR_KEYS))
        except TypeError:
            return value
    else:
        return value
    if size <= max_bytes:
        return value
    return {"_truncated": True, "size": size}


def prepare_content(
    value: Any,
    *,
    mode: ContentMode,
    sensitive_keys: frozenset[str] = frozenset(),
    max_bytes: int = 0,
    max_chars: int = 0,
) -> Any | None:
    """Project one payload through a content mode.

    Returns ``None`` when the mode is ``none`` (the caller omits the field
    entirely), the masked/verbatim payload otherwise.
    """
    if mode == "none":
        return None
    prepared = mask_payload(value, sensitive_keys) if mode == "masked" else value
    return cap_payload(prepared, max_bytes=max_bytes, max_chars=max_chars)


@dataclass(frozen=True)
class ContentPolicy:
    """The request's content decision, captured once when the request starts.

    ``mode`` is what the deployment allows (see :func:`content_mode`);
    ``sensitive_keys`` is the set to mask when it is ``masked``.
    """

    mode: ContentMode = "none"
    sensitive_keys: frozenset[str] = frozenset()

    @classmethod
    def resolve(cls, config: LoggingConfig | None) -> ContentPolicy:
        """Resolve the policy from a logging config (``None`` = safe defaults)."""
        if config is None:
            return cls(mode="none", sensitive_keys=frozenset(SENSITIVE_KEYS))
        return cls(
            mode=content_mode(
                bodies_enabled=config.log_input_output,
                mask_sensitive=config.mask_sensitive_data,
            ),
            sensitive_keys=sensitive_keys_for(config),
        )


@dataclass(frozen=True)
class SinkPolicy:
    """What one sink wants, expressed once instead of in each sink's code.

    ``content`` is the sink's *requested* mode — the bus narrows it against the
    request's :class:`ContentPolicy`. ``sample_record`` drops whole records
    (usage records always set ``1.0``: accounting is not sampled), and
    ``sample_content`` independently subsamples content.
    """

    record: bool = True
    content: ContentMode = "none"
    sample_record: float = 1.0
    sample_content: float = 1.0
    max_bytes: int = 0
    max_chars: int = 0


#: The policy of every sink that stores structured facts but no content.
FACTS_ONLY: SinkPolicy = SinkPolicy(record=True, content="none")


def sampled(policy: SinkPolicy) -> bool:
    """Whether a record passes the sink's ``sample_record`` draw.

    A rate of ``1.0`` (the default) is not a coin flip, ``0.0`` is a deliberate
    "keep none", and an *invalid* rate (negative, NaN, above one) keeps the record
    rather than silently dropping it: only a rate an operator can actually mean is
    allowed to discard data.
    """
    if not policy.record:
        return False
    rate = policy.sample_record
    if not isfinite(rate) or rate < 0.0:
        # NaN, +/-inf and a negative rate are configuration mistakes, not
        # sampling decisions; keeping the record is the safe reading.
        return True
    if rate >= 1.0:
        return True
    if rate == 0.0:
        return False
    return random.random() < rate


__all__ = [
    "FACTS_ONLY",
    "ContentMode",
    "ContentPolicy",
    "SendContent",
    "SinkPolicy",
    "cap_payload",
    "content_mode",
    "mask_payload",
    "narrow_mode",
    "prepare_content",
    "sampled",
    "sensitive_keys_for",
]
