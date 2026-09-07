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


class TranscriptionStreamUsageTracker:
    """Parse provider SSE chunks for transcription billing data.

    OpenAI streaming transcription: ``data: {...,"usage":{...}}`` in
    ``transcript.text.done`` events (when ``include[]=usage`` is used).
    The usage dict may be token-based (gpt-4o-transcribe) or duration-based
    (whisper: ``{"type":"duration","seconds":N}``).
    """

    def __init__(self) -> None:
        self._usage: dict[str, Any] | None = None

    @property
    def captured_usage(self) -> dict[str, Any] | None:
        return self._usage

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
            usage = data.get("usage")
            if isinstance(usage, dict):
                self._usage = usage

    def apply_to(self, ctx: EventContext, adapter: Any) -> None:
        """Write captured billing data into an EventContext using the adapter's
        ``_parse_usage`` for proper Usage object construction."""
        if self._usage is None:
            return
        parsed = adapter._parse_usage(self._usage)
        if parsed is not None:
            ctx.update_usage(parsed)
