"""Stream chunk parsing strategies.

Provides pluggable chunk parsing for different streaming protocols.
"""

from typing import Any

import orjson

from llm_proxy.observability.logger import get_logger

logger = get_logger(__name__)


class OpenAIStreamChunkParser:
    """OpenAI-compatible stream chunk parser.

    Parses SSE chunks, checks for meaningful content, and detects
    non-role output in streaming choices.
    """

    def parse_chunk(self, chunk: str | dict[str, Any]) -> dict[str, Any] | None:
        """Parse a raw stream chunk into a dict, or return None if not parseable."""
        if isinstance(chunk, dict):
            return chunk
        if not chunk.startswith("data: "):
            return None

        data = chunk[6:].strip()
        if not data or data == "[DONE]":
            return None

        try:
            parsed = orjson.loads(data)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            logger.debug("Failed to parse SSE chunk data", exc_info=True)
            return None

    def chunk_has_meaningful_content(self, parsed: dict[str, Any]) -> bool:
        """Check if a parsed chunk contains meaningful content beyond metadata."""

        def _has_value(value: Any) -> bool:
            if value is None:
                return False
            if isinstance(value, str):
                return value != ""
            if isinstance(value, list | dict | tuple | set):
                return bool(value)
            return True

        choices = parsed.get("choices", [])
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            if _has_value(choice.get("finish_reason")):
                return True

            delta = choice.get("delta", {})
            if not isinstance(delta, dict):
                delta = {}

            for key, value in delta.items():
                if key == "role":
                    continue
                if _has_value(value):
                    return True

            if _has_value(choice.get("logprobs")):
                return True

        metadata_keys = {
            "id",
            "object",
            "created",
            "model",
            "system_fingerprint",
            "choices",
        }
        for key, value in parsed.items():
            if key in metadata_keys:
                continue
            if _has_value(value):
                return True

        return False

    def choice_has_non_role_output(self, choice: dict[str, Any]) -> bool:
        """Check if a streaming choice has output beyond just a role field."""
        delta = choice.get("delta", {})
        if not isinstance(delta, dict):
            return False

        for key, value in delta.items():
            if key == "role":
                continue
            if value is None:
                continue
            if isinstance(value, str) and value == "":
                continue
            if isinstance(value, list | dict | tuple | set) and not value:
                continue
            return True

        return False
