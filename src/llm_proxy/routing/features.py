"""Routing feature extraction from a conversation message list.

These helpers are decoupled from protocol/transport concerns and work on the
plain message dictionaries used across adapters.
"""

import re
from typing import Any

import orjson

from llm_proxy.routing.signal_tuning import (
    contextual_followup_floor_from_text,
    strip_client_wrapper_blocks,
)
from llm_proxy.routing.types import RoutingFeatures, Tier

_TOOL_FAILURE_MARKERS = (
    "traceback",
    "exception",
    "assertionerror",
    "syntaxerror",
    "importerror",
    "modulenotfounderror",
    "failed",
    "error:",
    "command not found",
    "no such file",
)
_TOOL_FAILURE_PATTERNS = (re.compile(r"\b\d+\s+failed\b", re.IGNORECASE),)
_EXPLICIT_FAIL_STATUS_RE = re.compile(
    r"(?:^|\n)\s*(?:[\w./:= -]+\s*[:=]\s*)?FAIL(?:\s|$)",
    re.IGNORECASE,
)
_SUCCESSFUL_FAILURE_SUMMARY_RE = re.compile(
    r"\b(?:0\s+(?:failed|failures?|errors?)|(?:failed|failures?|errors?)\s*[:=]\s*0|no\s+(?:failed|failures?|errors?))\b",
    re.IGNORECASE,
)
_NONZERO_FAILURE_SUMMARY_RE = re.compile(
    r"\b(?:[1-9]\d*\s+(?:failed|failures?|errors?)|(?:failed|failures?|errors?)\s*[:=]\s*[1-9]\d*)\b",
    re.IGNORECASE,
)
_VERIFICATION_FAILURE_MARKERS = (
    "assertionerror",
    "short test summary info",
    "=== failures ===",
    "=== errors ===",
    "error collecting",
)
_VERIFICATION_CONTEXT_MARKERS = (
    "final verification",
    "verification",
    "verify",
    "test from",
    "tests:",
    "pytest",
    "unittest",
    "runtests",
    "failures",
    "assertionerror",
    "expected",
    "actual",
    "lint",
    "typecheck",
    "type-check",
    "mypy",
    "tsc",
    "eslint",
    "ruff",
)
_INVOCATION_FAILURE_MARKERS = (
    "unittest.loader._failedtest",
    "failedtest",
    "failed to import test module",
    "error importing test module",
    "no tests ran",
    "not found:",
    "module has no attribute",
)
_ENVIRONMENT_FAILURE_MARKERS = (
    "modulenotfounderror",
    "importerror",
    "no module named",
    "module not found",
    "command not found",
    "could not find a version",
    "no matching distribution",
    "successfully installed",
    "successfully uninstalled",
    "pip install",
    "site-packages/numpy",
    "module 'numpy' has no attribute",
)
_READ_ONLY_COMMAND_RE = re.compile(
    r"^\s*(?:"
    r"cat|sed\b|grep\b|rg\b|find\b|ls\b|head\b|tail\b|"
    r"git\s+(?:diff|status|show|log|rev-parse)\b"
    r")",
    re.IGNORECASE,
)
_XML_RETURN_CODE_RE = re.compile(r"<returncode>\s*(-?\d+)\s*</returncode>", re.IGNORECASE)


def _message_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
                elif item.get("type") == "tool_result":
                    parts.append(_message_text(item.get("content")))
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        return "\n".join(
            _message_text(value.get(key))
            for key in ("content", "text", "output", "error")
            if value.get(key) is not None
        )
    return str(value)


def _user_context_text(value: Any) -> str:
    return strip_client_wrapper_blocks(_message_text(value))


def _message_has_tool_result(value: Any) -> bool:
    if isinstance(value, list):
        return any(isinstance(item, dict) and item.get("type") == "tool_result" for item in value)
    return False


def _tool_call_command(tool_call: dict[str, Any]) -> str:
    fn = tool_call.get("function") or {}
    if not isinstance(fn, dict):
        return ""
    raw_args = fn.get("arguments")
    if isinstance(raw_args, str):
        try:
            parsed = orjson.loads(raw_args)
        except orjson.JSONDecodeError:
            return raw_args
        if isinstance(parsed, dict) and isinstance(parsed.get("command"), str):
            return str(parsed["command"])
    return ""


def _command_for_tool_result(
    messages: list[dict[str, Any]],
    *,
    before_index: int,
    tool_call_id: str,
) -> str:
    if not tool_call_id:
        return ""
    for prior in reversed(messages[:before_index]):
        if not isinstance(prior, dict):
            continue
        for tool_call in prior.get("tool_calls") or ():
            if isinstance(tool_call, dict) and tool_call.get("id") == tool_call_id:
                command = _tool_call_command(tool_call)
                if command:
                    return command
    return ""


def _latest_tool_result_message(
    messages: list[dict[str, Any]] | None,
) -> tuple[dict[str, Any] | None, str, str]:
    if not messages:
        return None, "", ""
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, dict):
            continue
        content = message.get("content")

        if message.get("role") == "tool":
            command = _command_for_tool_result(
                messages,
                before_index=index,
                tool_call_id=str(message.get("tool_call_id") or ""),
            )
            return message, _message_text(content), command
        if _message_has_tool_result(content):
            command = ""
            if isinstance(content, list):
                for block in reversed(content):
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        command = _command_for_tool_result(
                            messages,
                            before_index=index,
                            tool_call_id=str(block.get("tool_use_id") or ""),
                        )
                        break
            return message, _message_text(content), command
    return None, "", ""


def _has_zero_xml_returncode(text: str) -> bool:
    match = _XML_RETURN_CODE_RE.search(text or "")
    if match is None:
        return False
    try:
        return int(match.group(1)) == 0
    except ValueError:
        return False


def _command_is_read_only_observation(command: str) -> bool:
    return bool(_READ_ONLY_COMMAND_RE.search(command or ""))


def _tool_result_is_error(
    message: dict[str, Any] | None,
    text: str,
    command: str = "",
) -> bool:
    if not message:
        return False
    if bool(message.get("is_error")):
        return True
    if (
        _has_zero_xml_returncode(text)
        and _command_is_read_only_observation(command)
        and not _NONZERO_FAILURE_SUMMARY_RE.search(text)
    ):
        summary_stripped = _SUCCESSFUL_FAILURE_SUMMARY_RE.sub(" ", text)
        has_verification_context = any(
            marker in f"{command}\n{text}".lower() for marker in _VERIFICATION_CONTEXT_MARKERS
        )
        if not (has_verification_context and _EXPLICIT_FAIL_STATUS_RE.search(summary_stripped)):
            return False
    lowered = text.lower()
    if "<returncode>" in lowered and "<returncode>0</returncode>" not in lowered:
        return True
    if _tool_result_is_verification_failure(text):
        return True
    if any(pattern.search(text) for pattern in _TOOL_FAILURE_PATTERNS):
        return True
    return any(marker in lowered for marker in _TOOL_FAILURE_MARKERS)


def _tool_result_is_verification_failure(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    if _NONZERO_FAILURE_SUMMARY_RE.search(text):
        return True
    if any(marker in lowered for marker in _VERIFICATION_FAILURE_MARKERS):
        return True
    summary_stripped = _SUCCESSFUL_FAILURE_SUMMARY_RE.sub(" ", text)
    has_verification_context = any(marker in lowered for marker in _VERIFICATION_CONTEXT_MARKERS)
    return bool(has_verification_context and _EXPLICIT_FAIL_STATUS_RE.search(summary_stripped))


def _tool_result_failure_kind(text: str, command: str = "") -> str:
    """Classify the latest tool failure by routing significance.

    Semantic failures mean the attempted solution is probably wrong and can
    justify premium review. Environment and invocation failures are still
    high-risk, but usually need cheaper recovery steps rather than stronger
    reasoning.
    """
    if not text:
        return ""
    haystack = f"{command}\n{text}".lower()
    if any(marker in haystack for marker in _ENVIRONMENT_FAILURE_MARKERS):
        return "environment"
    if any(marker in haystack for marker in _INVOCATION_FAILURE_MARKERS):
        return "invocation"
    if _tool_result_is_verification_failure(text):
        return "semantic"
    return "unknown" if _tool_result_is_error({"role": "tool"}, text, command) else ""


def _agent_state_pressure(
    messages: list[dict[str, Any]] | None,
    step_risk: str,
) -> tuple[int, float]:
    """Estimate how much the current agent trajectory needs stronger review.

    This is deliberately continuous and model-agnostic. It does not say "use
    Opus after N steps"; it says long tool trajectories and accumulated tool
    failures should reduce the force of cheap/routine caps.
    """
    if not messages:
        return 0, 0.0

    tool_call_steps = 0
    tool_result_steps = 0
    failure_steps = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        has_tool_call = bool(message.get("tool_calls"))
        is_tool_result = message.get("role") == "tool" or _message_has_tool_result(
            message.get("content")
        )
        if has_tool_call:
            tool_call_steps += 1
        if is_tool_result:
            tool_result_steps += 1
            text = _message_text(message.get("content"))
            if _tool_result_is_error(message, text):
                failure_steps += 1

    tool_steps = max(tool_call_steps, tool_result_steps)
    step_component = min(0.45, max(0, tool_steps) / 24.0)
    failure_component = min(0.35, failure_steps / 6.0)
    risk_component = 0.20 if str(step_risk or "").lower() == "high" else 0.0
    pressure = min(1.0, step_component + failure_component + risk_component)
    return tool_steps, pressure


def _infer_routing_features_from_messages(
    messages: list[dict[str, Any]] | None,
    *,
    max_output_tokens: int,
) -> RoutingFeatures:
    tool_message, tool_text, tool_command = _latest_tool_result_message(messages)
    has_tool_results = tool_message is not None
    last_message = messages[-1] if messages else None
    last_is_tool_result = bool(
        isinstance(last_message, dict)
        and (
            last_message.get("role") == "tool"
            or _message_has_tool_result(last_message.get("content"))
        )
    )
    step_type = (
        "tool-result-followup"
        if last_is_tool_result
        else ("general_agent" if has_tool_results else "general")
    )

    step_risk = "normal"
    verification_failed = False
    failure_kind = ""
    if step_type == "tool-result-followup":
        failure_kind = _tool_result_failure_kind(tool_text, tool_command)
        verification_failed = failure_kind == "semantic"
        if _tool_result_is_error(tool_message, tool_text, tool_command):
            step_risk = "high"
        elif tool_text and len(tool_text) <= 800:
            step_risk = "low"

    agent_step_count, agent_pressure = _agent_state_pressure(messages, step_risk)

    return RoutingFeatures(
        step_type=step_type,
        has_tool_results=has_tool_results,
        step_risk=step_risk,
        requested_max_output_tokens=max(1, int(max_output_tokens)),
        tier_floor=Tier.MEDIUM if step_risk == "high" else None,
        tier_cap=(
            Tier.MEDIUM
            if step_risk == "low" or failure_kind in {"environment", "invocation"}
            else None
        ),
        tier_cap_reason=(
            "environment-recovery"
            if failure_kind == "environment"
            else (
                "invocation-recovery"
                if failure_kind == "invocation"
                else ("low-risk" if step_risk == "low" else "")
            )
        ),
        agent_step_count=agent_step_count,
        agent_pressure=agent_pressure,
        verification_failed=verification_failed,
        failure_kind=failure_kind,
    )


def _messages_contextual_followup_floor(
    messages: list[dict[str, Any]] | None,
) -> Tier | None:
    if not messages:
        return None
    user_indexes = [
        index
        for index, message in enumerate(messages)
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    if len(user_indexes) < 2:
        return None

    latest_index = user_indexes[-1]
    latest = _user_context_text(messages[latest_index].get("content"))
    prior_context = "\n".join(
        (
            _user_context_text(message.get("content"))
            if isinstance(message, dict) and message.get("role") == "user"
            else _message_text(message.get("content"))
        )
        for message in messages[:latest_index]
        if isinstance(message, dict) and message.get("role") != "system"
    )
    return contextual_followup_floor_from_text(
        prior_text=prior_context,
        latest_text=latest,
    )


# Public API -----------------------------------------------------------------


def extract_routing_features(
    messages: list[dict[str, Any]] | None,
    *,
    max_output_tokens: int = 0,
) -> RoutingFeatures:
    """Build a RoutingFeatures snapshot from a message list."""
    return _infer_routing_features_from_messages(
        messages,
        max_output_tokens=max_output_tokens,
    )


def messages_contextual_followup_floor(
    messages: list[dict[str, Any]] | None,
) -> Tier | None:
    """Infer a contextual follow-up tier floor from prior and latest user turns."""
    return _messages_contextual_followup_floor(messages)
