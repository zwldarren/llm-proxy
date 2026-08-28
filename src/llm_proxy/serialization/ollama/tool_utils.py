"""Ollama tool call normalization utilities.

Standalone functions shared by OllamaStreamingMixin and OllamaResponseParserMixin.
Extracted to eliminate fragile MRO dependency between the two mixins.
"""

import time
from typing import Any, cast

import orjson


def normalize_tool_calls(
    tool_calls: Any,
    *,
    include_index: bool,
    created_at: str | None = None,
) -> list[dict[str, Any]] | None:
    """Normalize Ollama tool calls to OpenAI-compatible format.

    Args:
        tool_calls: Raw tool calls from Ollama response
        include_index: Whether to include the index field in output
        created_at: Optional timestamp for generating call IDs

    Returns:
        List of normalized tool call dicts, or None if input is empty/invalid
    """
    if not isinstance(tool_calls, list) or not tool_calls:
        return None

    normalized: list[dict[str, Any]] = []
    for position, raw_call in enumerate(tool_calls):
        if not isinstance(raw_call, dict):
            continue

        call_dict = cast(dict[str, Any], raw_call)
        fn_raw = call_dict.get("function")
        fn: dict[str, Any] = fn_raw if isinstance(fn_raw, dict) else {}

        name = fn.get("name")
        if not isinstance(name, str) or not name:
            continue

        raw_index = call_dict.get("index")
        if raw_index is None:
            raw_index = fn.get("index")
        index: int
        if isinstance(raw_index, int):
            index = raw_index
        elif isinstance(raw_index, str) and raw_index.isdigit():
            index = int(raw_index)
        else:
            index = position

        raw_args = fn.get("arguments")
        if isinstance(raw_args, (dict, list)):
            args_str = orjson.dumps(raw_args).decode()
        elif raw_args is None:
            args_str = "{}"
        elif isinstance(raw_args, str):
            args_str = raw_args
        else:
            args_str = orjson.dumps(raw_args).decode() if raw_args is not None else "{}"

        raw_id = call_dict.get("id")
        call_id = str(raw_id).strip() if isinstance(raw_id, str) else ""
        if not call_id:
            base = created_at or str(int(time.time() * 1000))
            call_id = f"call_{base}_{index}"

        call_type = call_dict.get("type")
        if not isinstance(call_type, str) or not call_type:
            call_type = "function"

        out_call: dict[str, Any] = {
            "id": call_id,
            "type": call_type,
            "function": {"name": name, "arguments": args_str},
        }
        if include_index:
            out_call["index"] = index

        normalized.append(out_call)

    return normalized or None


def normalize_logprob_entries(ollama_logprobs: list[Any] | None) -> list[dict[str, Any]]:
    """Normalize raw Ollama logprob entries to a shared dict shape.

    Single source of truth for entry normalization (token/logprob/bytes/
    top_logprobs defaults): ``convert_logprobs`` wraps the result in the
    OpenAI wire dict, and the non-streaming response parser maps it onto
    typed ``TokenLogprob`` models.

    Args:
        ollama_logprobs: Raw logprobs from Ollama response

    Returns:
        List of normalized entries; empty for empty/invalid input
    """
    if not ollama_logprobs or not isinstance(ollama_logprobs, list):
        return []

    entries: list[dict[str, Any]] = []
    for item in ollama_logprobs:
        if not isinstance(item, dict):
            continue

        entry: dict[str, Any] = {
            "token": item.get("token", ""),
            "logprob": item.get("logprob", 0.0),
        }

        if item.get("bytes") is not None:
            entry["bytes"] = item["bytes"]

        if item.get("top_logprobs"):
            top_logprobs: list[dict[str, Any]] = []
            for t in item["top_logprobs"]:
                if not isinstance(t, dict):
                    continue
                top_entry: dict[str, Any] = {
                    "token": t.get("token", ""),
                    "logprob": t.get("logprob", 0.0),
                }
                if t.get("bytes") is not None:
                    top_entry["bytes"] = t["bytes"]
                top_logprobs.append(top_entry)
            if top_logprobs:
                entry["top_logprobs"] = top_logprobs

        entries.append(entry)

    return entries


def convert_logprobs(ollama_logprobs: list[Any] | None) -> dict[str, Any] | None:
    """Convert Ollama logprobs to OpenAI-compatible format.

    Thin wrapper over ``normalize_logprob_entries`` producing the wire dict.

    Args:
        ollama_logprobs: Raw logprobs from Ollama response

    Returns:
        Dict with "content" key containing normalized logprob entries, or None
    """
    content = normalize_logprob_entries(ollama_logprobs)
    return {"content": content} if content else None
