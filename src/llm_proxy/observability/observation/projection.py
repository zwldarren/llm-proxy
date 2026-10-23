"""Project one observation through one sink's policy.

The bus calls this before ``sink.emit``: it applies the sink's record sampling,
narrows the request's content mode against the sink's requested mode, masks and
caps the content, and scrubs content-bearing metadata when the sink may not see
it. Sinks therefore receive a ready-to-write observation — none of them repeats
the privacy or size rules.

See ADR-0044.
"""

import random
from dataclasses import replace
from typing import Any

from llm_proxy.observability.observation.model import ContentFacts, RequestObservation
from llm_proxy.observability.observation.policy import (
    ContentMode,
    SinkPolicy,
    cap_payload,
    mask_payload,
    narrow_mode,
    sampled,
)
from llm_proxy.observability.redaction import scrub_log_metadata
from llm_proxy.security.passwords import mask_headers


def _project_content(
    content: ContentFacts,
    *,
    mode: ContentMode,
    sensitive_keys: frozenset[str],
    policy: SinkPolicy,
) -> ContentFacts:
    """Mask/cap one payload set according to the effective mode.

    Every content-bearing field is projected, not just the two bodies:
    ``stream_output_blocks`` carries the streamed tool calls (and therefore their
    arguments), so a sink allowed only masked content must not receive it
    verbatim. ``raw_stream`` is the exception by construction — it is the verbatim
    wire text, and rewriting it would destroy the one property it exists for
    (byte-exact capture) — so it is only size-capped, and a deployment that turns
    raw capture on is choosing to store the frames unmasked.
    """
    request_headers: Any = dict(content.request_headers)
    response_headers: Any = dict(content.response_headers)
    if mode == "masked":
        request_headers = mask_headers(request_headers)
        response_headers = mask_headers(response_headers)

    def projected(value: Any) -> Any:
        masked = mask_payload(value, sensitive_keys) if mode == "masked" else value
        return cap_payload(masked, max_bytes=policy.max_bytes, max_chars=policy.max_chars)

    raw_stream = content.raw_stream
    raw_stream_truncated = content.raw_stream_truncated
    if raw_stream is not None and policy.max_bytes > 0 and len(raw_stream) > policy.max_bytes:
        # Truncate rather than replace with a marker: the buffer stays decodable
        # and ``raw_stream_truncated`` already tells the reader what happened.
        raw_stream = raw_stream[: policy.max_bytes]
        raw_stream_truncated = True

    return replace(
        content,
        request_headers=request_headers,
        request_body=projected(content.request_body),
        response_headers=response_headers,
        response_body=projected(content.response_body),
        assembled_response_body=projected(content.assembled_response_body),
        raw_stream=raw_stream,
        raw_stream_truncated=raw_stream_truncated,
        stream_output_blocks=projected(content.stream_output_blocks),
        withheld=False,
    )


def _withheld_content(content: ContentFacts, *, mask_headers_too: bool) -> ContentFacts:
    """The same observation with its payloads removed.

    Headers survive: they are diagnostic metadata (masked, never free-form
    content) and the log store deliberately keeps them on body-less rows. Only
    the bodies, the reassembled stream and the raw SSE buffer are dropped.
    """
    return replace(
        content,
        request_headers=(
            mask_headers(dict(content.request_headers))
            if mask_headers_too
            else dict(content.request_headers)
        ),
        response_headers=(
            mask_headers(dict(content.response_headers))
            if mask_headers_too
            else dict(content.response_headers)
        ),
        request_body=None,
        response_body=None,
        assembled_response_body=None,
        raw_stream=None,
        stream_output_blocks=None,
        withheld=True,
    )


def apply_sink_policy(
    observation: RequestObservation,
    policy: SinkPolicy,
) -> RequestObservation | None:
    """Return the observation as *this* sink may see it, or ``None`` to drop it.

    ``None`` means the sink asked not to record, the record lost
    ``sample_record``, or (for content-bearing sinks) the content lost
    ``sample_content`` while the sink exists only for content.
    """
    if not sampled(policy):
        return None

    content = observation.content
    if content is None:
        return observation

    # The sink's own requested mode can only narrow what the deployment allows.
    mode = narrow_mode(observation.policy.mode, policy.content)

    if mode == "none" or (
        policy.sample_content < 1.0
        and (policy.sample_content <= 0.0 or random.random() >= policy.sample_content)
    ):
        return replace(
            observation,
            content=_withheld_content(content, mask_headers_too=True),
            attributes=scrub_log_metadata(dict(observation.attributes)),
            policy=replace(observation.policy, mode=mode),
        )

    return replace(
        observation,
        content=_project_content(
            content,
            mode=mode,
            sensitive_keys=observation.policy.sensitive_keys,
            policy=policy,
        ),
        attributes=dict(observation.attributes),
        policy=replace(observation.policy, mode=mode),
    )


__all__ = ["apply_sink_policy"]
