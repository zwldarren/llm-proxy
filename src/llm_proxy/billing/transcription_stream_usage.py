"""Parse provider SSE chunks for transcription billing data.

Extracted from ``core/processing/streaming_processor.py`` (issue LLMP-3):
billing is not a streaming-processor concern — this module is public so
tests and callers import the tracker by name instead of reaching for a
processor-private class.
"""

from typing import Any

import orjson
from orjson import JSONDecodeError

from llm_proxy.observability.event_context import EventContext
from llm_proxy.serialization.usage import parse_usage as parse_wire_usage


class TranscriptionStreamUsageTracker:
    """Parse provider SSE chunks for transcription billing data.

    OpenAI streaming transcription: ``data: {...,"usage":{...}}`` in
    ``transcript.text.done`` events (when ``include[]=usage`` is used).
    The usage dict may be token-based (gpt-4o-transcribe) or duration-based
    (whisper: ``{"type":"duration","seconds":N}``).
    """

    def __init__(self) -> None:
        # The whole event, not just its usage block: the intake seam reads the
        # envelope (ADR-0025), and the audio dialect adapter owns where the
        # block sits inside it.
        self._payload: dict[str, Any] | None = None

    @property
    def captured_usage(self) -> dict[str, Any] | None:
        if self._payload is None:
            return None
        usage = self._payload.get("usage")
        return usage if isinstance(usage, dict) else None

    def observe(self, chunk: Any) -> None:
        if not isinstance(chunk, str):
            return
        for line in chunk.split("\n"):
            line = line.strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if not payload or payload == "[DONE]":
                continue
            try:
                data: dict[str, Any] = orjson.loads(payload)
            except JSONDecodeError:
                continue
            if isinstance(data.get("usage"), dict):
                self._payload = data

    def apply_to(self, ctx: EventContext) -> None:
        """Write the captured usage into an ``EventContext`` through the seam.

        The audio dialect adapter is the single parser of the shape (ADR-0025),
        so a streamed transcription and a non-streaming one bill the same way.
        """
        if self._payload is None:
            return
        parsed = parse_wire_usage("audio", self._payload)
        if parsed is not None:
            ctx.update_usage(parsed)
